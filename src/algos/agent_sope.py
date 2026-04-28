import numpy as np
import torch
import wandb
from tqdm import tqdm

from src.algos.agent_speq import SPEQAgent


class SOPEAgent(SPEQAgent):

    def __init__(
        self,
        *args,
        val_pct: float = 0.1,
        val_check_interval: int = 1000,
        val_patience: int = 10000,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.val_pct = val_pct
        self.val_check_interval = val_check_interval
        self.val_patience = val_patience

        self.val_batches_online = None
        self.val_batches_offline = None
        self._removed_online_data = None
        self._removed_offline_data = None


        self._stopping_epochs_history = []

    def _build_batches_from_indices(self, buf, indices):
        batches = []
        n_complete = len(indices) // self.batch_size
        for i in range(n_complete):
            s = i * self.batch_size
            idx = indices[s:s + self.batch_size]
            batches.append({
                'obs':      torch.FloatTensor(buf.obs1_buf[idx].copy()).to(self.device),
                'obs_next': torch.FloatTensor(buf.obs2_buf[idx].copy()).to(self.device),
                'acts':     torch.FloatTensor(buf.acts_buf[idx].copy()).to(self.device),
                'rews':     torch.FloatTensor(buf.rews_buf[idx].copy()).unsqueeze(1).to(self.device),
                'done':     torch.FloatTensor(buf.done_buf[idx].copy()).unsqueeze(1).to(self.device),
            })
        return batches

    def _sample_fixed_validation_batches(self):

        self.val_batches_online = []
        self.val_batches_offline = []
        self._removed_online_data = None
        self._removed_offline_data = None

        if self.replay_buffer.size < 2 * self.batch_size:
            print(f"  [{self.__class__.__name__}] Online buffer too small for val split, skipping.")
            return

        n_val_transitions = max(self.batch_size, int(self.replay_buffer.size * self.val_pct))

        n_online = min(n_val_transitions, self.replay_buffer.size - self.batch_size)
        if n_online >= self.batch_size:
            online_idx = np.random.choice(self.replay_buffer.size, size=n_online, replace=False)
            self._removed_online_data = {
                'obs1': self.replay_buffer.obs1_buf[online_idx].copy(),
                'obs2': self.replay_buffer.obs2_buf[online_idx].copy(),
                'acts': self.replay_buffer.acts_buf[online_idx].copy(),
                'rews': self.replay_buffer.rews_buf[online_idx].copy(),
                'done': self.replay_buffer.done_buf[online_idx].copy(),
            }
            self.val_batches_online = self._build_batches_from_indices(self.replay_buffer, online_idx)
            self.replay_buffer.remove_indices(online_idx)

        if (
            getattr(self, 'o2o', False)
            and hasattr(self, 'replay_buffer_offline')
            and self.replay_buffer_offline.size >= self.batch_size
        ):
            n_offline = min(n_val_transitions, self.replay_buffer_offline.size - self.batch_size)
            if n_offline >= self.batch_size:
                offline_idx = np.random.choice(
                    self.replay_buffer_offline.size, size=n_offline, replace=False
                )
                self._removed_offline_data = {
                    'obs1': self.replay_buffer_offline.obs1_buf[offline_idx].copy(),
                    'obs2': self.replay_buffer_offline.obs2_buf[offline_idx].copy(),
                    'acts': self.replay_buffer_offline.acts_buf[offline_idx].copy(),
                    'rews': self.replay_buffer_offline.rews_buf[offline_idx].copy(),
                    'done': self.replay_buffer_offline.done_buf[offline_idx].copy(),
                }
                self.val_batches_offline = self._build_batches_from_indices(
                    self.replay_buffer_offline, offline_idx
                )
                self.replay_buffer_offline.remove_indices(offline_idx)

        n_on = len(self.val_batches_online) * self.batch_size
        n_off = len(self.val_batches_offline) * self.batch_size
        print(
            f"  [{self.__class__.__name__}] Validation set "
            f"({self.val_pct*100:.0f}%): {len(self.val_batches_online)} online + "
            f"{len(self.val_batches_offline)} offline batches"
        )
        print(f"  Removed {n_on} online + {n_off} offline transitions from training")
        remaining = f"online={self.replay_buffer.size}"
        if getattr(self, 'o2o', False) and hasattr(self, 'replay_buffer_offline'):
            remaining += f", offline={self.replay_buffer_offline.size}"
        print(f"  Remaining: {remaining}")

    def _restore_validation_data(self):
        """Restore removed transitions to their buffers after stabilization."""
        if self._removed_online_data is not None:
            n = len(self._removed_online_data['obs1'])
            for i in range(n):
                self.replay_buffer.store(
                    self._removed_online_data['obs1'][i],
                    self._removed_online_data['acts'][i],
                    self._removed_online_data['rews'][i],
                    self._removed_online_data['obs2'][i],
                    self._removed_online_data['done'][i],
                )
            print(f"  Restored {n} online transitions")

        if self._removed_offline_data is not None:
            n = len(self._removed_offline_data['obs1'])
            for i in range(n):
                self.replay_buffer_offline.store(
                    self._removed_offline_data['obs1'][i],
                    self._removed_offline_data['acts'][i],
                    self._removed_offline_data['rews'][i],
                    self._removed_offline_data['obs2'][i],
                    self._removed_offline_data['done'][i],
                )
            print(f"  Restored {n} offline transitions")

        self.val_batches_online = None
        self.val_batches_offline = None
        self._removed_online_data = None
        self._removed_offline_data = None

    def _evaluate_policy_loss_on_fixed_batches(self) -> float:
        """Mean policy loss over all fixed held-out batches (online + offline)."""
        batches = (self.val_batches_online or []) + (self.val_batches_offline or [])
        if not batches:
            return 0.0

        total = 0.0
        with torch.no_grad():
            for b in batches:
                action, _, _, log_prob, _, _ = self.policy_net.forward(b['obs'])
                q_values = [q(torch.cat([b['obs'], action], 1)) for q in self.q_net_list]
                q_mean = torch.mean(torch.cat(q_values, dim=1), dim=1, keepdim=True)
                total += (self.alpha * log_prob - q_mean).mean().item()
        return total / len(batches)

    def train_offline(
        self,
        epochs: int,
        test_env=None,
        current_env_step: int = None,
    ) -> int:
        self._sample_fixed_validation_batches()

        if not self.val_batches_online and not self.val_batches_offline:
            print(f"  [{self.__class__.__name__}] No validation data, skipping stabilization.")
            self._restore_validation_data()
            return 0

        initial_loss = self._evaluate_policy_loss_on_fixed_batches()
        print(f"  [{self.__class__.__name__}] Initial val policy_loss: {initial_loss:.6f}")

        best_loss = initial_loss
        steps_without_improvement = 0
        epochs_performed = 0

        pbar = tqdm(range(epochs), desc=f"[{self.__class__.__name__}] Offline Stabilization")
        for e in pbar:
            epochs_performed = e + 1

            if e > 0 and e % self.val_check_interval == 0:
                val_loss = self._evaluate_policy_loss_on_fixed_batches()
                improved = val_loss < best_loss * 0.999
                status = "improved" if improved else "no improvement"
                pbar.write(
                    f"  Epoch {e}: val policy_loss={val_loss:.6f} "
                    f"(best={best_loss:.6f}, {status}, "
                    f"patience={steps_without_improvement}/{self.val_patience})"
                )
                if improved:
                    best_loss = val_loss
                    steps_without_improvement = 0
                else:
                    steps_without_improvement += self.val_check_interval

                if steps_without_improvement >= self.val_patience:
                    pbar.write(f"  Early stop @ epoch {e} "
                               f"(no improvement for {self.val_patience} steps)")
                    break

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

            if (e + 1) % 5000 == 0:
                current_val = self._evaluate_policy_loss_on_fixed_batches()
                pbar.write(f"  Offline epoch {e+1}/{epochs}, val policy_loss={current_val:.6f}")
                if (
                    test_env is not None
                    and current_env_step is not None
                    and wandb.run is not None
                ):
                    test_rw = self._eval_test_env(test_env, max_ep_len=1000)
                    wandb.log({"OfflineEvalReward": test_rw}, step=current_env_step)

        final_loss = self._evaluate_policy_loss_on_fixed_batches()
        print(
            f"  [{self.__class__.__name__}] Final val policy_loss: {final_loss:.6f} "
            f"(started at {initial_loss:.6f}, ran {epochs_performed} epochs)"
        )

        self._stopping_epochs_history.append(epochs_performed)
        if current_env_step is not None and wandb.run is not None:
            wandb.log({"OfflineEpochs": epochs_performed}, step=current_env_step)

        self._restore_validation_data()
        return epochs_performed


class SOPEEOAgent(SOPEAgent):

    def __init__(self, *args, **kwargs):
        kwargs['o2o'] = False
        super().__init__(*args, **kwargs)
        self.o2o = False 

    def _sample_fixed_validation_batches(self):
        self.val_batches_online = []
        self.val_batches_offline = []
        self._removed_online_data = None
        self._removed_offline_data = None

        if self.replay_buffer.size < 2 * self.batch_size:
            print(f"  [{self.__class__.__name__}] Online buffer too small for val split, skipping.")
            return

        n_val_transitions = max(self.batch_size, int(self.replay_buffer.size * self.val_pct))
        n_val_transitions = min(n_val_transitions, self.replay_buffer.size - self.batch_size)

        val_idx = np.random.choice(self.replay_buffer.size, size=n_val_transitions, replace=False)
        self._removed_online_data = {
            'obs1': self.replay_buffer.obs1_buf[val_idx].copy(),
            'obs2': self.replay_buffer.obs2_buf[val_idx].copy(),
            'acts': self.replay_buffer.acts_buf[val_idx].copy(),
            'rews': self.replay_buffer.rews_buf[val_idx].copy(),
            'done': self.replay_buffer.done_buf[val_idx].copy(),
        }
        self.val_batches_online = self._build_batches_from_indices(self.replay_buffer, val_idx)
        self.replay_buffer.remove_indices(val_idx)

        n_on = len(self.val_batches_online) * self.batch_size
        print(
            f"  [{self.__class__.__name__}] Validation set "
            f"({self.val_pct*100:.0f}%): {len(self.val_batches_online)} online batches "
            f"({n_on} transitions)"
        )
        print(f"  Remaining training buffer: {self.replay_buffer.size}")