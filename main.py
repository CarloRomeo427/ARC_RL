import argparse
import importlib
import os
import time

# ─────────────────────────────────────────────────────────────────────────────
# Headless Rendering Setup
# Must be set before importing gymnasium/mujoco
# ─────────────────────────────────────────────────────────────────────────────
if "DISPLAY" not in os.environ and "WAYLAND_DISPLAY" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl" 

import gymnasium as gym
import h5py
import numpy as np
import torch
import wandb

import src.envs  # registers Leaper-v1, Bastion-v1, Queen-v1
import random

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
    'queen': 0.015,
}

def make_env(env_name: str, render_mode: str = None):
    """Creates Gym env with default (light) configuration."""
    key = env_name.lower()
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
    video_env = make_env(args.env, render_mode='rgb_array')
    obs_dim, act_dim, act_limit, max_ep_len = get_env_info(env)

    # Initialize Agent
    path = {'sac': 'src.algos.agent_sac.SACAgent', 'droq': 'src.algos.agent_sac.SACAgent', 
            'speq': 'src.algos.agent_speq.SPEQAgent', 'sope': 'src.algos.agent_speq.SOPEAgent'}[args.algo.lower()]
    module_path, class_name = path.rsplit('.', 1)
    AgentClass = getattr(importlib.import_module(module_path), class_name)
    
    dropout = args.target_drop_rate if args.target_drop_rate >= 0 else DROPOUT_RATES.get(args.env.lower(), 0.005)
    agent = AgentClass(env_name=args.env, obs_dim=obs_dim, act_dim=act_dim, act_limit=act_limit, device=device, 
                       start_steps=args.start_steps if not args.load_weights else 0,
                       hidden_sizes=(args.network_width, args.network_width), target_drop_rate=dropout)

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
            if epoch % 10 == 0: log["video"] = wandb.Video(record_eval_video(agent, video_env, max_ep_len), fps=20, format="mp4")
            wandb.log(log, step=global_t)
            save_checkpoint(agent, os.path.join(run_dir, "checkpoints"), global_t)

    env.close(); test_env.close(); video_env.close()

# ─────────────────────────────────────────────────────────────────────────────
# 4. CLI & Execution
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default='sac', choices=['sac', 'droq', 'speq', 'sope'])
    parser.add_argument("--env", type=str, default='queen')
    parser.add_argument("--save-dataset", action="store_true")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--steps-per-epoch", type=int, default=1000)
    parser.add_argument("--start-steps", type=int, default=5000)
    parser.add_argument("--network-width", type=int, default=256)
    parser.add_argument("--target-drop-rate", type=float, default=-1.0)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--log-wandb", action="store_true")
    parser.add_argument("--checkpoint-dir", type=str, default='outputs')
    parser.add_argument("--load-weights", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    set_global_seeds(args.seed)

    name = f"{args.algo}_{args.env}"
    wandb.init(name=name, project="ARC_RL", config=vars(args), mode='online' if args.log_wandb else 'disabled')
    train(args)
    wandb.finish()