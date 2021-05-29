import gym
from procgen import ProcgenEnv

from stable_baselines3.common.vec_env import VecMonitor, VecNormalize


def procgen_wrapper(num_envs: int = 2,
                    env_name: str = 'starpilot'):
    assert len([key for key in gym.envs.registry.env_specs.keys() if env_name in key]), 'Not found in procgen.'
    venv = ProcgenEnv(num_envs=num_envs, env_name=env_name)
    venv = VecMonitor(venv=venv)
    venv = VecNormalize(venv=venv, ob=False)
    return venv