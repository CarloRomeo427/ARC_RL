import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from abc import ABC, abstractmethod

from src.algos.core import (
    TanhGaussianPolicy, Mlp, soft_update_model1_with_model2,
    ReplayBuffer,
)


class BaseAgent(ABC):
    
    def __init__(
        self,
        env_name: str,
        obs_dim: int,
        act_dim: int,
        act_limit: float,
        device: torch.device,
        hidden_sizes: tuple = (256, 256),
        replay_size: int = int(1e6),
        batch_size: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        polyak: float = 0.995,
        alpha: float = 0.2,
        auto_alpha: bool = True,
        target_entropy: str = 'auto',
        start_steps: int = 5000,
        utd_ratio: int = 1,
        num_Q: int = 2,
        policy_update_delay: int = 1,
        target_drop_rate: float = 0.0,
        layer_norm: bool = True,
        o2o: bool = False,
    ):
        self.env_name = env_name
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.act_limit = act_limit
        self.device = device
        self.hidden_sizes = hidden_sizes
        self.replay_size = replay_size
        self.batch_size = batch_size
        self.lr = lr
        self.gamma = gamma
        self.polyak = polyak
        self.start_steps = start_steps
        self.delay_update_steps = start_steps
        self.utd_ratio = utd_ratio
        self.num_Q = num_Q
        self.policy_update_delay = policy_update_delay
        self.target_drop_rate = target_drop_rate
        self.layer_norm = layer_norm
        self.o2o = o2o
        
        # Initialize policy network
        self.policy_net = TanhGaussianPolicy(
            obs_dim, act_dim, hidden_sizes,
            action_limit=act_limit,
            layer_norm=layer_norm
        ).to(device)
        
        # Initialize Q-networks
        self.q_net_list = []
        self.q_target_net_list = []
        for _ in range(num_Q):
            q_net = Mlp(
                obs_dim + act_dim, 1, hidden_sizes,
                target_drop_rate=target_drop_rate,
                layer_norm=layer_norm
            ).to(device)
            q_target_net = Mlp(
                obs_dim + act_dim, 1, hidden_sizes,
                target_drop_rate=target_drop_rate,
                layer_norm=layer_norm
            ).to(device)
            q_target_net.load_state_dict(q_net.state_dict())
            self.q_net_list.append(q_net)
            self.q_target_net_list.append(q_target_net)
        
        # Initialize optimizers
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.q_optimizer_list = [optim.Adam(q.parameters(), lr=lr) for q in self.q_net_list]
        
        # Entropy coefficient (alpha)
        self.auto_alpha = auto_alpha
        if auto_alpha:
            self.target_entropy = self._get_target_entropy(env_name, act_dim, target_entropy)
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optim = optim.Adam([self.log_alpha], lr=lr)
            self.alpha = self.log_alpha.cpu().exp().item()
        else:
            self.alpha = alpha
            self.target_entropy = None
            self.log_alpha = None
            self.alpha_optim = None
        
        # Replay buffers
        self.replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)
        self.replay_buffer_offline = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)
        
        # Loss criterion
        self.mse_criterion = nn.MSELoss()
    
    def _get_target_entropy(self, env_name: str, act_dim: int, target_entropy: str) -> float:
        if target_entropy == 'auto':
            return -act_dim
        return float(target_entropy)
    
    @property
    def buffer_size(self) -> int:
        return self.replay_buffer.size
    
    # ─────────────────────────────────────────────────────────────────────────────
    # Action Selection
    # ─────────────────────────────────────────────────────────────────────────────
    
    def get_exploration_action(self, obs: np.ndarray, env) -> np.ndarray:
        """Get action for exploration (random initially, then from policy)."""
        with torch.no_grad():
            if self.buffer_size > self.start_steps:
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                action = self.policy_net.forward(
                    obs_tensor, deterministic=False, return_log_prob=False
                )[0]
                return action.cpu().numpy().reshape(-1)
            return env.action_space.sample()
    
    def get_test_action(self, obs: np.ndarray) -> np.ndarray:
        """Get deterministic action for evaluation."""
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            action = self.policy_net.forward(
                obs_tensor, deterministic=True, return_log_prob=False
            )[0]
            return action.cpu().numpy().reshape(-1)
    
    
    
    # ─────────────────────────────────────────────────────────────────────────────
    # Data Storage
    # ─────────────────────────────────────────────────────────────────────────────
    
    def store_data(self, o, a, r, o2, d):
        """Store transition in online replay buffer."""
        self.replay_buffer.store(o, a, r, o2, d)
    
    def store_data_offline(self, o, a, r, o2, d):
        """Store transition in offline replay buffer."""
        self.replay_buffer_offline.store(o, a, r, o2, d)
    
    # ─────────────────────────────────────────────────────────────────────────────
    # Data Sampling
    # ─────────────────────────────────────────────────────────────────────────────
    
    def sample_data(self, batch_size: int):
        """Sample batch from online buffer."""
        batch = self.replay_buffer.sample_batch(batch_size)
        return self._batch_to_tensors(batch)
    
    def sample_data_offline_only(self, batch_size: int):
        """Sample batch from offline buffer."""
        batch = self.replay_buffer_offline.sample_batch(batch_size)
        return self._batch_to_tensors(batch)
    
    def sample_data_mix(self, batch_size: int):
        """Sample mixed batch from online and offline buffers (50/50)."""
        batch_online = self.replay_buffer.sample_batch(batch_size)
        batch_offline = self.replay_buffer_offline.sample_batch(batch_size)
        
        batch = {
            'obs1': np.concatenate([batch_online['obs1'], batch_offline['obs1']]),
            'obs2': np.concatenate([batch_online['obs2'], batch_offline['obs2']]),
            'acts': np.concatenate([batch_online['acts'], batch_offline['acts']]),
            'rews': np.concatenate([batch_online['rews'], batch_offline['rews']]),
            'done': np.concatenate([batch_online['done'], batch_offline['done']]),
        }
        return self._batch_to_tensors(batch)
    
    def _batch_to_tensors(self, batch: dict):
        """Convert numpy batch to device tensors."""
        obs = torch.FloatTensor(batch['obs1']).to(self.device)
        obs_next = torch.FloatTensor(batch['obs2']).to(self.device)
        acts = torch.FloatTensor(batch['acts']).to(self.device)
        rews = torch.FloatTensor(batch['rews']).unsqueeze(1).to(self.device)
        done = torch.FloatTensor(batch['done']).unsqueeze(1).to(self.device)
        return obs, obs_next, acts, rews, done
    
    # ─────────────────────────────────────────────────────────────────────────────
    # Common Training Utilities
    # ─────────────────────────────────────────────────────────────────────────────
    
    def update_target_networks(self):
        """Soft update all target Q-networks."""
        for q_net, q_target in zip(self.q_net_list, self.q_target_net_list):
            soft_update_model1_with_model2(q_target, q_net, self.polyak)
    
    def update_alpha(self, log_prob: torch.Tensor):
        """Update entropy coefficient alpha."""
        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.cpu().exp().item()
            return alpha_loss.item()
        return 0.0
    
    def get_sac_q_target(self, obs_next: torch.Tensor, rews: torch.Tensor, done: torch.Tensor):
        """Compute SAC-style Q-target: r + γ(1-d)(min Q' - α log π)."""
        with torch.no_grad():
            next_action, _, _, next_log_prob, _, _ = self.policy_net.forward(obs_next)
            
            q_next_list = [
                q_target(torch.cat([obs_next, next_action], 1))
                for q_target in self.q_target_net_list
            ]
            q_next_cat = torch.cat(q_next_list, dim=1)
            q_next_min = torch.min(q_next_cat, dim=1, keepdim=True)[0]
            
            return rews + self.gamma * (1 - done) * (q_next_min - self.alpha * next_log_prob)
    
    # ─────────────────────────────────────────────────────────────────────────────
    # Abstract Methods (to be implemented by subclasses)
    # ─────────────────────────────────────────────────────────────────────────────
    
    @abstractmethod
    def train(self, logger, current_env_step: int = None):
        """Perform one training step. Must be implemented by subclasses."""
        pass
    
    def check_should_trigger_offline_stabilization(self) -> bool:
        """Check if offline stabilization should be triggered. Override in SPEQ variants."""
        return False
    
    def train_offline(self, epochs: int, test_env=None, current_env_step: int = None):
        """Offline pretraining. Override in algorithms that support it (IQL, CalQL)."""
        pass
