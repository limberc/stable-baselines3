import gym
from procgen import ProcgenEnv

from stable_baselines3.common.vec_env import VecMonitor


def procgen_wrapper(num_envs: int = 2,
                    env_name: str = 'starpilot'):
    assert len([key for key in gym.envs.registry.env_specs.keys() if env_name in key]), 'Not found in procgen.'
    venv = ProcgenEnv(num_envs=num_envs, env_name=env_name)
    return VecMonitor(venv=venv)
