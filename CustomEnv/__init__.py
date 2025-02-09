from gym.envs.registration import register
# from gymnasium.envs.registration import register


register(
     id="Swimmer-modified-Angle-v0",
     entry_point="CustomEnv.envs:SwimmerAngle",
     max_episode_steps=1000,
)

register(
     id="HalfCheetah-modified-Angle-v0",
     entry_point="CustomEnv.envs:CrippleCheetah",
     max_episode_steps=1000,
)

register(
     id="Hopper-modified-Z-v0",
     entry_point="CustomEnv.envs:HopperZ",
     max_episode_steps=1000,
)

register(
     id="Ant-modified-Y-v0",
     entry_point="CustomEnv.envs:AntY",
     max_episode_steps=1000,
)
