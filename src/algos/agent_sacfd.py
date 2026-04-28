from __future__ import annotations

import os

import h5py
import numpy as np
import torch

from src.algos.agent_sac import SACAgent


class SACfDAgent(SACAgent):


    DEFAULT_ONLINE_CAPACITY = int(1e6)

    def __init__(
        self,
        env_name: str,
        obs_dim: int,
        act_dim: int,
        act_limit: float,
        device: torch.device,
        offline_dataset_path: str = None,
        online_capacity: int = None,
        **kwargs,
    ):
        kwargs['o2o'] = False
        kwargs['target_drop_rate'] = 0.0

        super().__init__(env_name, obs_dim, act_dim, act_limit, device, **kwargs)

        if offline_dataset_path is None:
            raise ValueError(
                "SACfD requires --offline-dataset pointing to an HDF5 file "
                "produced by src/scripted/collect_dataset.py"
            )

        n_demos = self._count_demo_transitions(offline_dataset_path)
        if n_demos == 0:
            raise RuntimeError(f"No demonstration transitions found in {offline_dataset_path}")

        online_capacity = int(online_capacity or self.DEFAULT_ONLINE_CAPACITY)
        required_size = n_demos + online_capacity

        if required_size > self.replay_buffer.max_size:
            self._resize_buffer_in_place(self.replay_buffer, required_size)
            print(
                f"[sacfd] Buffer resized to {required_size} "
                f"({n_demos} demo slots + {online_capacity} online slots)"
            )
        else:
            online_capacity = self.replay_buffer.max_size - n_demos
            print(
                f"[sacfd] Existing buffer (max_size={self.replay_buffer.max_size}) "
                f"already accommodates {n_demos} demos + {online_capacity} online"
            )

        self._load_offline_dataset_into_main(offline_dataset_path)
        assert self.replay_buffer.size == n_demos, (
            f"Demo load count mismatch: counted {n_demos}, buffer size is "
            f"{self.replay_buffer.size}. Aborting to avoid corrupt fence."
        )

        self._demo_fence = n_demos               
        self._online_ptr = self._demo_fence     
        self._online_capacity = online_capacity

        if hasattr(self, 'replay_buffer_offline'):
            del self.replay_buffer_offline
            self.replay_buffer_offline = None

    def store_data(self, o, a, r, o2, d):
        buf = self.replay_buffer
        idx = self._online_ptr

        buf.obs1_buf[idx] = o
        buf.obs2_buf[idx] = o2
        buf.acts_buf[idx] = a
        buf.rews_buf[idx] = r
        buf.done_buf[idx] = d
        if hasattr(buf, 'episode_ids') and hasattr(buf, 'current_episode_id'):
            buf.episode_ids[idx] = buf.current_episode_id

        self._online_ptr += 1
        if self._online_ptr >= buf.max_size:
            self._online_ptr = self._demo_fence

        buf.size = min(buf.size + 1, buf.max_size)

    @staticmethod
    def _count_demo_transitions(path: str) -> int:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Offline dataset not found: {path}")
        n = 0
        with h5py.File(path, 'r') as f:
            for k in f.keys():
                if k.startswith('episode_'):
                    n += int(np.asarray(f[k]['actions']).shape[0])
        return n

    def _load_offline_dataset_into_main(self, path: str) -> None:
        n_loaded = 0
        n_episodes = 0
        total_return = 0.0

        with h5py.File(path, 'r') as f:
            ep_keys = sorted(
                [k for k in f.keys() if k.startswith('episode_')],
                key=lambda k: int(k.split('_')[1]),
            )
            for ep_key in ep_keys:
                ep = f[ep_key]
                obs    = np.asarray(ep['observations'])
                acts   = np.asarray(ep['actions'])
                rews   = np.asarray(ep['rewards'])
                terms  = np.asarray(ep['terminations']).astype(bool)
                truncs = np.asarray(ep['truncations']).astype(bool)

                T = len(acts)
                assert obs.shape[0] == T + 1, (
                    f"{ep_key}: expected obs len {T+1}, got {obs.shape[0]}"
                )

                for t in range(T):
                    if self.replay_buffer.size >= self.replay_buffer.max_size:
                        raise RuntimeError(
                            f"Main buffer full ({self.replay_buffer.max_size}) during demo load. "
                            f"This should not happen after _resize_buffer_in_place."
                        )
                    done = float(bool(terms[t]) and not bool(truncs[t]))
                    self.replay_buffer.store(
                        obs[t], acts[t], rews[t], obs[t + 1], done
                    )
                    n_loaded += 1

                n_episodes += 1
                total_return += float(np.sum(rews))

        mean_ret = total_return / max(n_episodes, 1)
        print(
            f"[sacfd] loaded {n_loaded} demo transitions from {path} "
            f"({n_episodes} eps, mean return {mean_ret:.1f})"
        )


    @staticmethod
    def _resize_buffer_in_place(buf, new_size: int) -> None:
        if new_size <= buf.max_size:
            return

        old_size = buf.size
        obs_dim = buf.obs1_buf.shape[1]
        act_dim = buf.acts_buf.shape[1]

        new_obs1 = np.zeros((new_size, obs_dim), dtype=buf.obs1_buf.dtype)
        new_obs2 = np.zeros((new_size, obs_dim), dtype=buf.obs2_buf.dtype)
        new_acts = np.zeros((new_size, act_dim), dtype=buf.acts_buf.dtype)
        new_rews = np.zeros(new_size,            dtype=buf.rews_buf.dtype)
        new_done = np.zeros(new_size,            dtype=buf.done_buf.dtype)

        if old_size > 0:
            new_obs1[:old_size] = buf.obs1_buf[:old_size]
            new_obs2[:old_size] = buf.obs2_buf[:old_size]
            new_acts[:old_size] = buf.acts_buf[:old_size]
            new_rews[:old_size] = buf.rews_buf[:old_size]
            new_done[:old_size] = buf.done_buf[:old_size]

        buf.obs1_buf = new_obs1
        buf.obs2_buf = new_obs2
        buf.acts_buf = new_acts
        buf.rews_buf = new_rews
        buf.done_buf = new_done

        if hasattr(buf, 'episode_ids'):
            new_eids = np.zeros(new_size, dtype=buf.episode_ids.dtype)
            if old_size > 0:
                new_eids[:old_size] = buf.episode_ids[:old_size]
            buf.episode_ids = new_eids

        buf.max_size = new_size
