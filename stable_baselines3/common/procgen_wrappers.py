import gym
from procgen import ProcgenEnv
from stable_baselines3.common.vec_env import VecExtractDictObs, VecMonitor, VecNormalize


def make_procgen_env(n_envs: int = 2,
                     env_name: str = 'starpilot'):
    assert len([key for key in gym.envs.registry.env_specs.keys() if env_name in key]), 'Not found in procgen.'
    venv = ProcgenEnv(num_envs=n_envs, env_name=env_name)
    venv = VecExtractDictObs(venv, 'rgb')
    venv = VecMonitor(venv=venv)
    venv = VecNormalize(venv=venv, norm_obs=False)
    return venv
