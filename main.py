import argparse
import importlib
import os
import random

# ─────────────────────────────────────────────────────────────────────────────
# Headless Rendering Setup
# ─────────────────────────────────────────────────────────────────────────────
if "DISPLAY" not in os.environ and "WAYLAND_DISPLAY" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl" 

import gymnasium as gym
import h5py
import numpy as np
import torch
import wandb

import src.envs 


def set_global_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─────────────────────────────────────────────────────────────────────────────
# 1. Environment Configuration
# ─────────────────────────────────────────────────────────────────────────────

ENV_REGISTRY = {
    'leaper-v1': 'Leaper-v1',
    'bastion-v1': 'Bastion-v1',
    'queen-v1': 'Queen-v1',
    'ant': 'Ant-v5',
}

DROPOUT_RATES = {
    'leaper': 0.01,
    'bastion': 0.01,
    'queen': 0.01,
}

def make_env(env_name: str, render_mode: str = None):
    key = env_name.lower()
    if key not in ENV_REGISTRY and f"{key}-v1" in ENV_REGISTRY:
        key = f"{key}-v1"
    return gym.make(ENV_REGISTRY[key], render_mode=render_mode)

def get_env_info(env):
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = float(env.action_space.high[0])
    max_ep_len = getattr(env, 'spec', None).max_episode_steps if getattr(env, 'spec', None) else 1000
    return obs_dim, act_dim, act_limit, max_ep_len


# ─────────────────────────────────────────────────────────────────────────────
# 2. Episode Collection & Evaluation
# ─────────────────────────────────────────────────────────────────────────────

class EpisodeCollector:
    def __init__(self, save_path: str):
        self.save_path = save_path
        self._current = {'observations': [], 'actions': [], 'rewards': [], 'terminations': [], 'truncations': []}
        self._total_episodes = 0

    def step(self, obs, action, reward, next_obs, terminated, truncated):
        if not self._current['observations']: self._current['observations'].append(obs.copy())
        self._current['actions'].append(action.copy())
        self._current['rewards'].append(float(reward))
        self._current['terminations'].append(bool(terminated))
        self._current['truncations'].append(bool(truncated))
        self._current['observations'].append(next_obs.copy())

        if terminated or truncated:
            self.flush()

    def flush(self):
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        with h5py.File(self.save_path, 'a') as f:
            grp = f.create_group(f"episode_{self._total_episodes}")
            for k, v in self._current.items(): grp.create_dataset(k, data=np.array(v))
            self._total_episodes += 1
        self._current = {'observations': [], 'actions': [], 'rewards': [], 'terminations': [], 'truncations': []}

def evaluate(agent, env, max_ep_len):
    rewards = []
    for _ in range(5):
        obs, _ = env.reset(); ep_ret = 0
        for _ in range(max_ep_len):
            obs, reward, term, trunc, _ = env.step(agent.get_test_action(obs))
            ep_ret += reward
            if term or trunc: break
        rewards.append(ep_ret)
    return np.mean(rewards)

def record_eval_video(agent, env, max_ep_len):
    frames = []
    obs, _ = env.reset()
    for _ in range(max_ep_len):
        frames.append(env.render())
        obs, _, term, trunc, _ = env.step(agent.get_test_action(obs))
        if term or trunc: break
    return np.stack(frames).transpose(0, 3, 1, 2)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Training Logic
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(agent, save_dir, step):
    os.makedirs(save_dir, exist_ok=True)
    ckpt = {'policy': agent.policy_net.state_dict(), 'qs': [q.state_dict() for q in agent.q_net_list]}
    path = os.path.join(save_dir, "checkpoint_latest.pt")
    torch.save(ckpt, path)
    return path

def load_weights(agent, path, device):
    print(f"--- Loading weights from {path} ---")
    ckpt = torch.load(path, map_location=device)
    agent.policy_net.load_state_dict(ckpt['policy'])
    for q, qs in zip(agent.q_net_list, ckpt['qs']): q.load_state_dict(qs)

def train(args):
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    env = make_env(args.env)
    test_env = make_env(args.env)
    
    video_env = make_env(args.env, render_mode='rgb_array') if args.log_wandb else None
    obs_dim, act_dim, act_limit, max_ep_len = get_env_info(env)

    path_map = {
        'sac':       'src.algos.agent_sac.SACAgent',
        'droq':      'src.algos.agent_sac.SACAgent',
        'sacfd':     'src.algos.agent_sacfd.SACfDAgent',
        'speq':      'src.algos.agent_speq.SPEQAgent',
        'speq_o2o':  'src.algos.agent_speq.SPEQAgent',
        'sope':      'src.algos.agent_sope.SOPEAgent',
        'sope_eo':   'src.algos.agent_sope.SOPEEOAgent',
    }
    
    module_path, class_name = path_map[args.algo.lower()].rsplit('.', 1)
    AgentClass = getattr(importlib.import_module(module_path), class_name)

    extra_kwargs = {}
    
    # ── DROPOUT SAFETY BLOCK ──
    if args.algo.lower() in ['speq', 'speq_o2o', 'sope', 'sope_eo']:
        dropout = args.target_drop_rate if args.target_drop_rate >= 0 else DROPOUT_RATES.get(args.env.lower(), 0.005)
    else:
        dropout = 0.0
    
    if args.algo.lower() in ['sacfd', 'speq_o2o', 'sope']:
        extra_kwargs['o2o'] = True
    elif args.algo.lower() in ['speq', 'sope_eo', 'sac']:
        extra_kwargs['o2o'] = False

    if args.algo.lower() in ['sacfd', 'speq_o2o', 'sope']:
        if args.offline_dataset is None:
            args.offline_dataset = f"data/{args.env}_cpg.hdf5"
            
        if not os.path.exists(args.offline_dataset):
            print(f"\n[Setup] Local dataset '{args.offline_dataset}' not found.")
            
            if not args.hf_repo:
                raise ValueError("Offline dataset is missing and no --hf-repo was provided to download it.")
                
            try:
                from huggingface_hub import hf_hub_download
                print(f"[Setup] Downloading from Hugging Face Hub ({args.hf_repo})...")
                
                os.makedirs(os.path.dirname(args.offline_dataset), exist_ok=True)
                downloaded_path = hf_hub_download(
                    repo_id=args.hf_repo,
                    filename=f"{args.env}_cpg.hdf5",
                    repo_type="dataset",
                    local_dir=os.path.dirname(args.offline_dataset),
                    local_dir_use_symlinks=False
                )
                print(f"[Setup] Successfully downloaded to '{downloaded_path}'\n")
            except ImportError:
                raise ImportError("huggingface_hub is required. Run `pip install huggingface_hub`.")
            except Exception as e:
                raise RuntimeError(f"Failed to download dataset from HF: {e}")
        else:
            print(f"\n[Setup] Found existing cached dataset at '{args.offline_dataset}'.\n")
            
        extra_kwargs['offline_dataset_path'] = args.offline_dataset

    agent = AgentClass(
        env_name=args.env, obs_dim=obs_dim, act_dim=act_dim, act_limit=act_limit,
        device=device,
        start_steps=args.start_steps if not args.load_weights else 0,
        hidden_sizes=(args.network_width, args.network_width),
        target_drop_rate=dropout,
        **extra_kwargs,
    )

    if args.load_weights: load_weights(agent, args.load_weights, device)

    exp_name = f"{args.algo}_{args.env}"
    run_dir = os.path.join(args.checkpoint_dir, exp_name, f"seed_{args.seed}")
    
    collector = EpisodeCollector(os.path.join(run_dir, "dataset.hdf5")) if args.save_dataset else None

    obs, _ = env.reset(seed=args.seed); ep_len = 0

    for t in range(args.steps_per_epoch * args.epochs):
        global_t = t + 1
        action = agent.get_exploration_action(obs, env)
        obs_next, reward, term, trunc, _ = env.step(action)
        ep_len += 1
        
        agent.store_data(obs, action, reward, obs_next, term and ep_len < max_ep_len)
        if collector: collector.step(obs, action, reward, obs_next, term, trunc or ep_len >= max_ep_len)
        
        agent.train(current_env_step=global_t)
        obs = obs_next
        
        if term or trunc or ep_len >= max_ep_len:
            obs, _ = env.reset(); ep_len = 0

        if (t + 1) % args.steps_per_epoch == 0:
            epoch = (t + 1) // args.steps_per_epoch
            reward_val = evaluate(agent, test_env, max_ep_len)
            
            log = {"epoch": epoch, "EvalReward": reward_val}
            
            if args.log_wandb and epoch % 10 == 0:
                video_frames = record_eval_video(agent, video_env, max_ep_len)
                log["video"] = wandb.Video(video_frames, fps=20, format="mp4")
                
            if args.log_wandb:
                wandb.log(log, step=global_t)
                
            save_checkpoint(agent, os.path.join(run_dir, "checkpoints"), global_t)

    env.close(); test_env.close()
    if video_env: video_env.close()


# ─────────────────────────────────────────────────────────────────────────────
# 4. CLI & Execution
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default='sac',
                        choices=['sac', 'droq', 'speq', 'speq_o2o', 'sope', 'sope_eo', 'sacfd'])
    parser.add_argument("--env", type=str, default='queen', 
                        choices=['leaper', 'bastion', 'queen', 'ant'])
    parser.add_argument("--save-dataset", action="store_true")
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--steps-per-epoch", type=int, default=1000)
    parser.add_argument("--start-steps", type=int, default=5000)
    parser.add_argument("--network-width", type=int, default=256)
    parser.add_argument("--target-drop-rate", type=float, default=-1.0)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--log-wandb", action="store_true")
    parser.add_argument("--checkpoint-dir", type=str, default='outputs')
    parser.add_argument("--load-weights", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    
    # Dataset Handling Arguments
    parser.add_argument("--offline-dataset", type=str, default=None,
                        help="Local path to HDF5 demo dataset.")
    parser.add_argument('--hf-repo', type=str, default='CarloRomeoHugging/ARC_RL')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    set_global_seeds(args.seed)

    name = f"{args.algo}_{args.env}"
    if args.log_wandb:
        wandb.init(name=name, project="ARC_RL", config=vars(args), mode='online')
        
    train(args)
    
    if args.log_wandb:
        wandb.finish()