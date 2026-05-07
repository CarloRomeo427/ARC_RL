import os

import h5py
import numpy as np
import torch

from src.algos.agent_sac import SACAgent


class RLPDAgent(SACAgent):
    """RLPD (Ball et al., 2023): SAC + large critic ensemble + high UTD +
    50/50 online-offline batches + min over a random subset of target critics."""

    def __init__(
        self,
        env_name: str,
        obs_dim: int,
        act_dim: int,
        act_limit: float,
        device: torch.device,
        offline_dataset_path: str = None,
        subset_size: int = 2,
        **kwargs,
    ):
        kwargs.setdefault('num_Q', 10)
        kwargs.setdefault('utd_ratio', 20)
        kwargs['target_drop_rate'] = 0.0
        kwargs.setdefault('o2o', True)

        super().__init__(env_name, obs_dim, act_dim, act_limit, device, **kwargs)

        self.subset_size = min(subset_size, self.num_Q)

        if offline_dataset_path is None:
            raise ValueError(
                "RLPD requires --offline-dataset pointing to an HDF5 file "
                "produced by src/scripted/collect_dataset.py"
            )
        self._load_offline_dataset(offline_dataset_path)

    def _load_offline_dataset(self, path: str) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Offline dataset not found: {path}")

        n_loaded = 0
        with h5py.File(path, 'r') as f:
            for ep_key in f.keys():
                ep = f[ep_key]
                obs = np.asarray(ep['observations'])
                acts = np.asarray(ep['actions'])
                rews = np.asarray(ep['rewards'])
                terms = np.asarray(ep['terminations']).astype(bool)
                truncs = np.asarray(ep['truncations']).astype(bool)

                T = len(acts)
                for t in range(T):
                    done = float(bool(terms[t]) and not bool(truncs[t]))
                    self.replay_buffer_offline.store(
                        obs[t], acts[t], rews[t], obs[t + 1], done
                    )
                    n_loaded += 1
        print(f"[RLPD] Loaded {n_loaded} offline transitions from {path}")

    def get_random_subset_q_target(self, obs_next, rews, done):
        with torch.no_grad():
            next_action, _, _, next_log_prob, _, _ = self.policy_net.forward(obs_next)

            idxs = np.random.choice(self.num_Q, size=self.subset_size, replace=False)
            q_next_list = [
                self.q_target_net_list[i](torch.cat([obs_next, next_action], 1))
                for i in idxs
            ]
            q_next_min = torch.min(torch.cat(q_next_list, dim=1), dim=1, keepdim=True)[0]

            return rews + self.gamma * (1 - done) * (q_next_min - self.alpha * next_log_prob)

    def train(self, current_env_step: int = None):
        for i_update in range(self.utd_ratio):
            if getattr(self, 'o2o', False) and self.replay_buffer_offline.size > 0:
                obs, next_obs, acts, rews, done = self.sample_data_mix(self.batch_size)
            else:
                obs, next_obs, acts, rews, done = self.sample_data(self.batch_size)

            y_q = self.get_random_subset_q_target(next_obs, rews, done)
            q_values = [q_net(torch.cat([obs, acts], 1)) for q_net in self.q_net_list]
            q_cat = torch.cat(q_values, dim=1)
            y_q_expanded = y_q.expand((-1, self.num_Q)) if y_q.shape[1] == 1 else y_q

            q_loss = self.mse_criterion(q_cat, y_q_expanded) * self.num_Q

            for q_opt in self.q_optimizer_list:
                q_opt.zero_grad()
            q_loss.backward()

            if ((i_update + 1) % self.policy_update_delay == 0) or i_update == self.utd_ratio - 1:
                action, _, _, log_prob, _, _ = self.policy_net.forward(obs)

                for q_net in self.q_net_list:
                    q_net.requires_grad_(False)

                q_values = [q_net(torch.cat([obs, action], 1)) for q_net in self.q_net_list]
                q_mean = torch.mean(torch.cat(q_values, dim=1), dim=1, keepdim=True)
                policy_loss = (self.alpha * log_prob - q_mean).mean()

                self.policy_optimizer.zero_grad()
                policy_loss.backward()

                for q_net in self.q_net_list:
                    q_net.requires_grad_(True)

                if self.auto_alpha:
                    self.update_alpha(log_prob)

            for q_opt in self.q_optimizer_list:
                q_opt.step()
            if ((i_update + 1) % self.policy_update_delay == 0) or i_update == self.utd_ratio - 1:
                self.policy_optimizer.step()

            self.update_target_networks()
