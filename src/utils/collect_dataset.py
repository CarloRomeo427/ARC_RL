import argparse
import importlib
import os

import h5py
import gymnasium as gym
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--env', type=str, required=True, choices=['leaper', 'bastion', 'queen', 'tick'])
    p.add_argument('--preset', type=str, default='medium', choices=['slow', 'medium', 'fast'])
    p.add_argument('--n-episodes', type=int, default=1000)
    p.add_argument('--max-ep-len', type=int, default=1000)
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--out-dir', type=str, default='data/')
    
    p.add_argument('--hf-repo', type=str, default=None)
    p.add_argument('--hf-token', type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    file_name = f"{args.env}_cpg.hdf5"
    out_path = os.path.join(args.out_dir, file_name)

    mod = importlib.import_module(f"src.controllers.cpg_controller_{args.env}")
    
    env_id_func = getattr(mod, '_env_id')
    ctrl_class = getattr(mod, f"ScriptedCPGController{args.env.capitalize()}")
    collect_episode = getattr(mod, 'collect_episode')

    env = gym.make(env_id_func(args.env))
    ctrl = ctrl_class(env, robot=args.env, preset=args.preset)

    base_seed = args.seed if args.seed is not None else int(np.random.SeedSequence().entropy % (2**31 - 1))
    rng = np.random.default_rng(base_seed)
    returns = []

    with h5py.File(out_path, 'w') as f:
        for i in range(args.n_episodes):
            current_seed = int(rng.integers(0, 2**31 - 1))

            ep = collect_episode(env, ctrl, args.max_ep_len, seed=current_seed)
            
            g = f.create_group(f"episode_{i}")
            for k, v in ep.items():
                g.create_dataset(k, data=np.array(v))
            
            returns.append(float(np.sum(ep['rewards'])))

        f.attrs['env'] = args.env
        f.attrs['n_episodes'] = args.n_episodes
        f.attrs['preset'] = args.preset

    env.close()

    print(f"[collect_dataset] {args.env} ({args.preset}): {args.n_episodes} eps -> {out_path}")
    print(f"[collect_dataset] Return bounds: min={np.min(returns):.1f}, max={np.max(returns):.1f}, mean={np.mean(returns):.1f}")
    
    if args.hf_repo:
        try:
            from huggingface_hub import HfApi
            print(f"[collect_dataset] Uploading {file_name} to Hugging Face Hub ({args.hf_repo})...")
            
            api = HfApi(token=args.hf_token)
            api.upload_file(
                path_or_fileobj=out_path,
                path_in_repo=file_name,
                repo_id=args.hf_repo,
                repo_type="dataset"
            )
            print(f"[collect_dataset] Successfully uploaded to https://huggingface.co/datasets/{args.hf_repo}")
            
        except ImportError:
            print("[collect_dataset] ERROR: huggingface_hub is not installed. Run `pip install huggingface_hub`.")
        except Exception as e:
            print(f"[collect_dataset] ERROR during Hugging Face upload: {e}")


if __name__ == '__main__':
    main()