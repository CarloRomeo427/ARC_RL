import os

import h5py
import numpy as np
import torch
import wandb
from tqdm import tqdm

from src.algos.agent_sac import SACAgent


class SPEQAgent(SACAgent):

    def __init__(
        self,
        env_name: str,
        obs_dim: int,
        act_dim: int,
        act_limit: float,
        device: torch.device,
        offline_dataset_path: str = None,
        offline_epochs: int = 75000,
        trigger_interval: int = 10000,
        **kwargs,
    ):
        kwargs.setdefault('policy_update_delay', 20)
        super().__init__(env_name, obs_dim, act_dim, act_limit, device, **kwargs)

        self.offline_epochs = offline_epochs
        self.trigger_interval = trigger_interval
        self.next_trigger_step = trigger_interval
        self.env_steps = 0

        if offline_dataset_path is not None:
            if getattr(self, 'o2o', False):
                self._load_offline_dataset(offline_dataset_path)
            else:
                print(f"[{self.__class__.__name__}] Info: running strictly online "
                      f"(o2o=False). Offline dataset ignored.")

    def store_data(self, o, a, r, o2, d):
        super().store_data(o, a, r, o2, d)
        self.env_steps += 1

    def check_should_trigger_offline_stabilization(self) -> bool:
        if self.env_steps >= self.next_trigger_step:
            self.next_trigger_step += self.trigger_interval
            print(f"[{self.__class__.__name__}] Stabilization @ env_step {self.env_steps}")
            return True
        return False

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
        print(f"[{self.__class__.__name__}] Loaded {n_loaded} offline transitions from {path}")

    def _eval_test_env(self, test_env, max_ep_len: int = 1000, n_episodes: int = 1) -> float:
        returns = []
        for _ in range(n_episodes):
            obs, _ = test_env.reset()
            ep_ret = 0.0
            for _ in range(max_ep_len):
                obs, r, term, trunc, _ = test_env.step(self.get_test_action(obs))
                ep_ret += float(r)
                if term or trunc:
                    break
            returns.append(ep_ret)
        return float(np.mean(returns))


    def train(self, current_env_step: int = None, test_env=None):

        super().train(current_env_step)
        if self.check_should_trigger_offline_stabilization():
            self.train_offline(
                self.offline_epochs,
                test_env=test_env,
                current_env_step=current_env_step,
            )

    def train_offline(
        self,
        epochs: int,
        test_env=None,
        current_env_step: int = None,
    ) -> int:
 
        epochs_performed = 0
        pbar = tqdm(range(epochs), desc=f"[{self.__class__.__name__}] Offline Stabilization")
        for i_update in pbar:
            epochs_performed = i_update + 1

            if getattr(self, 'o2o', False):
                obs, next_obs, acts, rews, done = self.sample_data_mix(self.batch_size)
            else:
                obs, next_obs, acts, rews, done = self.sample_data(self.batch_size)

            y_q = self.get_sac_q_target(next_obs, rews, done)
            q_preds = [q_net(torch.cat([obs, acts], 1)) for q_net in self.q_net_list]
            q_cat = torch.cat(q_preds, dim=1)
            y_q_expanded = y_q.expand((-1, self.num_Q)) if y_q.shape[1] == 1 else y_q
            q_loss = self.mse_criterion(q_cat, y_q_expanded) * self.num_Q

            for q_opt in self.q_optimizer_list:
                q_opt.zero_grad()
            q_loss.backward()
            for q_opt in self.q_optimizer_list:
                q_opt.step()

            self.update_target_networks()

            if (i_update + 1) % 5000 == 0:
                pbar.write(f"  [{self.__class__.__name__}] Offline epoch {i_update+1}/{epochs}")
                if (
                    test_env is not None
                    and current_env_step is not None
                    and wandb.run is not None
                ):
                    test_rw = self._eval_test_env(test_env, max_ep_len=1000)
                    wandb.log({"OfflineEvalReward": test_rw}, step=current_env_step)

        if current_env_step is not None and wandb.run is not None:
            wandb.log({"OfflineEpochs": epochs_performed}, step=current_env_step)

        return epochs_performed