from gym.envs.registration import register

register(
    id='CartPoleSwingUpContinuous-v0',
    entry_point='cartpole_swingup_envs.cartpole_swingup:CartPoleSwingUpEnv'
    )

register(
    id='CartPoleSwingUpContinuous-v1',
    entry_point='cartpole_swingup_envs.cartpole_swingup_grav:CartPoleSwingUpEnv'
    )

register(
    id='CartPoleSwingUpDiscrete-v0',
    entry_point='cartpole_swingup_envs.discrete_cartpole_swingup:CartPoleSwingUpEnv'
    )

register(
    id='CartPoleGrav-v1',
    entry_point='cartpole_swingup_envs.cartpole_v1_grav:CartPoleEnv'
    )