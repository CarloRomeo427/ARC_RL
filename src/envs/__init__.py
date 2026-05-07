import gymnasium as gym

gym.register(
    id='Leaper-v1',
    entry_point='src.envs.leaper_env_v1:LeaperEnv',
    max_episode_steps=1000,
    reward_threshold=6000.0,
)

gym.register(
    id='Bastion-v1',
    entry_point='src.envs.bastion_env_v1:BastionEnv',
    max_episode_steps=1000,
    reward_threshold=6000.0,
)

gym.register(
    id='Queen-v1',
    entry_point='src.envs.queen_env_v1:QueenEnv',
    max_episode_steps=1000,
    reward_threshold=6000.0,
)

gym.register(
    id='Tick-v1',
    entry_point='src.envs.tick_env_v1:TickEnv',
    max_episode_steps=1000,
    reward_threshold=6000.0,
)