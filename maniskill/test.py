import gymnasium as gym
import gymnasium.spaces as spaces
from tqdm.notebook import tqdm
import numpy as np
import mani_skill2.envs
import matplotlib.pyplot as plt
import torch.nn as nn
import torch as th

from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from mani_skill2.utils.wrappers import RecordEpisode
from stable_baselines3.common.utils import set_random_seed
import mani_skill2.envs


num_envs = 1 # you can increases this and decrease the n_steps parameter if you have more cores to speed up training
env_id = "LiftCube-v0"
obs_mode = "state"
control_mode = "pd_ee_delta_pose"
reward_mode = "normalized_dense" # this the default reward mode which is a dense reward scaled to [0, 1]
max_episode_steps = 50
if __name__ == '__main__':
    env = gym.make(env_id, obs_mode=obs_mode, reward_mode=reward_mode, control_mode=control_mode, max_episode_steps=max_episode_steps, render_mode="cameras")
    o = env.reset() 
    for i in range(100):
        o, r, d, info = env.step(env.action_space.sample())
        env.render()
        if d:
            env.reset()
        