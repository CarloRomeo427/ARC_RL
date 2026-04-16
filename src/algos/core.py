from random import random, sample

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Normal

# following SAC authors' and OpenAI implementation
LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
ACTION_BOUND_EPSILON = 1E-6


def weights_init_(m):
    # weight init helper function
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer with support for index-based removal.
    """

    def __init__(self, obs_dim, act_dim, size):
        """
        :param obs_dim: size of observation
        :param act_dim: size of the action
        :param size: size of the buffer
        """
        ## init buffers as numpy arrays
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.episode_ids = np.zeros(size, dtype=np.int32)
        self.ptr, self.size, self.max_size = 0, 0, size
        self.current_episode_id = 0
        self.obs_dim = obs_dim
        self.act_dim = act_dim

    def store(self, obs, act, rew, next_obs, done):
        """
        data will get stored in the pointer's location
        """
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.episode_ids[self.ptr] = self.current_episode_id

        ## move the pointer to store in next location in buffer
        self.ptr = (self.ptr + 1) % self.max_size
        ## keep track of the current buffer size
        self.size = min(self.size + 1, self.max_size)

        if done:
            self.current_episode_id += 1

    def sample_batch(self, batch_size=32, idxs=None):
        """
        :param batch_size: size of minibatch
        :param idxs: specify indexes if you want specific data points
        :return: mini-batch data as a dictionary
        """
        if idxs is None:
            idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs1=self.obs1_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs],
                    idxs=idxs)

    def sample_batch_with_indices(self, batch_size=32):
        """
        Sample a batch and return with the sampled indices for potential removal.
        :param batch_size: size of minibatch
        :return: mini-batch data as a dictionary including indices
        """
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs1=self.obs1_buf[idxs].copy(),
                    obs2=self.obs2_buf[idxs].copy(),
                    acts=self.acts_buf[idxs].copy(),
                    rews=self.rews_buf[idxs].copy(),
                    done=self.done_buf[idxs].copy(),
                    idxs=idxs.copy())

    def remove_indices(self, indices_to_remove):
        """
        Remove specified indices from the buffer by compacting remaining data.
        This is an O(n) operation but maintains buffer integrity.
        
        :param indices_to_remove: array of indices to remove from buffer
        """
        if len(indices_to_remove) == 0:
            return
        
        # Create mask for indices to keep
        indices_to_remove = np.unique(indices_to_remove)
        mask = np.ones(self.size, dtype=bool)
        valid_indices = indices_to_remove[indices_to_remove < self.size]
        mask[valid_indices] = False
        
        # Get indices of data to keep
        keep_indices = np.where(mask)[0]
        new_size = len(keep_indices)
        
        if new_size == 0:
            # Buffer becomes empty
            self.ptr = 0
            self.size = 0
            return
        
        # Compact the data
        self.obs1_buf[:new_size] = self.obs1_buf[keep_indices]
        self.obs2_buf[:new_size] = self.obs2_buf[keep_indices]
        self.acts_buf[:new_size] = self.acts_buf[keep_indices]
        self.rews_buf[:new_size] = self.rews_buf[keep_indices]
        self.done_buf[:new_size] = self.done_buf[keep_indices]
        self.episode_ids[:new_size] = self.episode_ids[keep_indices]
        
        # Update size and pointer
        self.size = new_size
        self.ptr = new_size % self.max_size

    def sample_batch_with_episode_ids(self, batch_size=32):
        """
        Sample a batch of transitions along with their episode IDs.
        :param batch_size: size of minibatch
        :return: mini-batch data as a dictionary including episode IDs
        """
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs1=self.obs1_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs],
                    episode_ids=self.episode_ids[idxs],
                    idxs=idxs)

    def sample_all(self):
        """
        Sample all transitions in the buffer.
        :return: all data as a dictionary
        """
        return dict(obs1=self.obs1_buf[:self.size],
                    obs2=self.obs2_buf[:self.size],
                    acts=self.acts_buf[:self.size],
                    rews=self.rews_buf[:self.size],
                    done=self.done_buf[:self.size],
                    episode_ids=self.episode_ids[:self.size])

    def filter_top_x_percent(self, x=0.5):

        all_batches = self.sample_all()
        episode_rewards = {}

        for i in range(len(all_batches['obs1'])):
            episode_id = all_batches['episode_ids'][i]
            reward = all_batches['rews'][i]
            if episode_id not in episode_rewards:
                episode_rewards[episode_id] = 0
            episode_rewards[episode_id] += reward

        sorted_episodes = sorted(episode_rewards.items(), key=lambda item: item[1], reverse=True)
        top_x_percent_episodes = [item[0] for item in sorted_episodes[:int(len(sorted_episodes) * x)]]

        filtered_batches = {
            'obs1': [],
            'obs2': [],
            'acts': [],
            'rews': [],
            'done': [],
            'episode_ids': []
        }

        for i in range(len(all_batches['obs1'])):
            episode_id = all_batches['episode_ids'][i]
            if episode_id in top_x_percent_episodes:
                filtered_batches['obs1'].append(all_batches['obs1'][i])
                filtered_batches['obs2'].append(all_batches['obs2'][i])
                filtered_batches['acts'].append(all_batches['acts'][i])
                filtered_batches['rews'].append(all_batches['rews'][i])
                filtered_batches['done'].append(all_batches['done'][i])
                filtered_batches['episode_ids'].append(all_batches['episode_ids'][i])

        return dict(obs1=filtered_batches['obs1'],
                    obs2=filtered_batches['obs2'],
                    acts=filtered_batches['acts'],
                    rews=filtered_batches['rews'],
                    done=filtered_batches['done'],
                    episode_ids=filtered_batches['episode_ids'])

    def filter_top_x_transitions(self, x=5000):
        """Filter the top x transitions based on episode rewards"""

        all_batches = self.sample_all()
        episode_rewards = {}

        # Aggregate rewards for each episode
        for i in range(len(all_batches['obs1'])):
            episode_id = all_batches['episode_ids'][i]
            reward = all_batches['rews'][i]
            if episode_id not in episode_rewards:
                episode_rewards[episode_id] = 0
            episode_rewards[episode_id] += reward

        # Sort episodes by total reward in descending order
        sorted_episodes = sorted(episode_rewards.items(), key=lambda item: item[1], reverse=True)
        top_x_percent_episodes = [item[0] for item in sorted_episodes]

        filtered_batches = {
            'obs1': [],
            'obs2': [],
            'acts': [],
            'rews': [],
            'done': [],
            'episode_ids': []
        }

        # Accumulate transitions from top episodes until we reach the desired number of transitions
        accumulated_transitions = 0
        for episode_id in top_x_percent_episodes:
            for i in range(len(all_batches['obs1'])):
                if all_batches['episode_ids'][i] == episode_id:
                    filtered_batches['obs1'].append(all_batches['obs1'][i])
                    filtered_batches['obs2'].append(all_batches['obs2'][i])
                    filtered_batches['acts'].append(all_batches['acts'][i])
                    filtered_batches['rews'].append(all_batches['rews'][i])
                    filtered_batches['done'].append(all_batches['done'][i])
                    filtered_batches['episode_ids'].append(all_batches['episode_ids'][i])
                    accumulated_transitions += 1
                    if accumulated_transitions >= x:
                        break
            if accumulated_transitions >= x:
                break

        return dict(obs1=filtered_batches['obs1'],
                    obs2=filtered_batches['obs2'],
                    acts=filtered_batches['acts'],
                    rews=filtered_batches['rews'],
                    done=filtered_batches['done'],
                    episode_ids=filtered_batches['episode_ids'])

    def random_x_transitions(self, x=5000):
        """Filter the top x transitions based on episode rewards"""

        all_batches = self.sample_all()
        sample_indices = sample(range(len(all_batches['obs1'])), x)

        filtered_batches = {
            'obs1': [],
            'obs2': [],
            'acts': [],
            'rews': [],
            'done': [],
            'episode_ids': []
        }

        for key in filtered_batches.keys():
            filtered_batches[key] = [all_batches[key][i] for i in sample_indices]

        return dict(obs1=filtered_batches['obs1'],
                    obs2=filtered_batches['obs2'],
                    acts=filtered_batches['acts'],
                    rews=filtered_batches['rews'],
                    done=filtered_batches['done'],
                    episode_ids=filtered_batches['episode_ids'])


class Mlp(nn.Module):
    def __init__(
            self,
            input_size,
            output_size,
            hidden_sizes,
            hidden_activation=F.relu,
            #
            target_drop_rate=0.0, layer_norm=False

    ):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_activation = hidden_activation
        ## here we use ModuleList so that the layers in it can be
        ## detected by .parameters() call
        self.hidden_layers = nn.ModuleList()
        in_size = input_size

        ## initialize each hidden layer
        for i, next_size in enumerate(hidden_sizes):
            fc_layer = nn.Linear(in_size, next_size)
            in_size = next_size
            self.hidden_layers.append(fc_layer)

            # added 20211206
            if target_drop_rate > 0.0:
                self.hidden_layers.append(nn.Dropout(p=target_drop_rate))  # dropout
            if layer_norm:
                self.hidden_layers.append(nn.LayerNorm(fc_layer.out_features))  # layer norm

        # added to fix bug 20211207
        self.apply_activation_per = 1
        if target_drop_rate > 0.0:
            self.apply_activation_per += 1
        if layer_norm:
            self.apply_activation_per += 1

        ## init last fully connected layer with small weight and bias
        self.last_fc_layer = nn.Linear(in_size, output_size)
        self.apply(weights_init_)

    def forward(self, input):
        h = input
        for i, fc_layer in enumerate(self.hidden_layers):
            h = fc_layer(h)
            # h = self.hidden_activation(h)
            if ((i + 1) % self.apply_activation_per) == 0:
                h = self.hidden_activation(h)
        output = self.last_fc_layer(h)
        return output


class TanhGaussianPolicy(nn.Module):
    """
    A Gaussian policy network with Tanh to enforce action limits.
    Now supports LayerNorm for numerical stability with high-dimensional observations.
    """

    def __init__(
            self,
            obs_dim,
            action_dim,
            hidden_sizes,
            hidden_activation=F.relu,
            action_limit=1.0,
            layer_norm=False  # NEW: Added layer_norm parameter
    ):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_activation = hidden_activation
        self.action_limit = action_limit
        self.layer_norm = layer_norm
        
        # Build hidden layers with optional LayerNorm
        self.hidden_layers = nn.ModuleList()
        in_size = obs_dim
        
        for hidden_size in hidden_sizes:
            self.hidden_layers.append(nn.Linear(in_size, hidden_size))
            if layer_norm:
                self.hidden_layers.append(nn.LayerNorm(hidden_size))
            in_size = hidden_size
        
        # Track how many modules per "layer" for activation application
        self.modules_per_layer = 2 if layer_norm else 1
        
        # Output layers for mean and log_std
        self.last_fc_layer = nn.Linear(in_size, action_dim)
        self.last_fc_log_std = nn.Linear(in_size, action_dim)
        
        # Initialize weights
        self.apply(weights_init_)

    def forward(
            self,
            obs,
            deterministic=False,
            return_log_prob=True,
    ):
        """
        :param obs: Observation
        :param deterministic: If True, take deterministic (test) action
        :param return_log_prob: If True, return a sample and its log probability
        """
        h = obs
        for i, layer in enumerate(self.hidden_layers):
            h = layer(h)
            # Apply activation after each complete "block" (Linear + optional LayerNorm)
            if (i + 1) % self.modules_per_layer == 0:
                h = self.hidden_activation(h)
        
        mean = self.last_fc_layer(h)
        log_std = self.last_fc_log_std(h)
        log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        std = torch.exp(log_std)

        normal = Normal(mean, std)

        if deterministic:
            pre_tanh_value = mean
            action = torch.tanh(mean)
        else:
            pre_tanh_value = normal.rsample()
            action = torch.tanh(pre_tanh_value)

        if return_log_prob:
            log_prob = normal.log_prob(pre_tanh_value)
            log_prob -= torch.log(1 - action.pow(2) + ACTION_BOUND_EPSILON)
            log_prob = log_prob.sum(1, keepdim=True)
        else:
            log_prob = None

        return (
            action * self.action_limit, mean, log_std, log_prob, std, pre_tanh_value,
        )


def soft_update_model1_with_model2(model1, model2, rou):
    """
    used to polyak update a target network
    :param model1: a pytorch model
    :param model2: a pytorch model of the same class
    :param rou: the update is model1 <- rou*model1 + (1-rou)model2
    """
    for model1_param, model2_param in zip(model1.parameters(), model2.parameters()):
        model1_param.data.copy_(rou * model1_param.data + (1 - rou) * model2_param.data)



def test_agent(agent, test_env, max_ep_len, logger, n_eval=10):
    """
    This will test the agent's performance by running <n_eval> episodes
    During the runs, the agent should only take deterministic action
    This function assumes the agent has a <get_test_action()> function
    :param agent: agent instance
    :param test_env: the environment used for testing
    :param max_ep_len: max length of an episode
    :param logger: logger to store info in
    :param n_eval: number of episodes to run the agent
    :return: test return for each episode as a numpy array
    """
    ep_return_list = np.zeros(n_eval)
    ep_len_list = np.zeros(n_eval)
    for j in range(n_eval):
        # gymnasium reset() returns (observation, info) tuple
        o, info = test_env.reset()
        r, ep_ret, ep_len = 0, 0, 0
        terminated, truncated = False, False
        d = False
        
        while not (d or (ep_len == max_ep_len)):
            # Take deterministic actions at test time
            a = agent.get_test_action(o)
            # gymnasium step() returns (obs, reward, terminated, truncated, info)
            o, r, terminated, truncated, _ = test_env.step(a)
            d = terminated or truncated
            ep_ret += r
            ep_len += 1
        ep_return_list[j] = ep_ret
        ep_len_list[j] = ep_len
    if logger is not None:
        logger.store(TestEpRet=ep_return_list, TestEpLen=ep_len_list)
    return ep_return_list