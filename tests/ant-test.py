import time

import gym
import pybulletgym  # register PyBullet enviroments with open ai gym

import numpy as np
import pickle


def discretize_action(a: int):
    if a == 0:
        return -1
    if a == 1:
        return -0.5
    if a == 2:
        return 0.5
    if a == 3:
        return 1
    raise ValueError


action_heads = pickle.load(open("../pickles/ant_action_heads_episode_500.p", "rb"))

env = gym.make("AntPyBulletEnv-v0")
env.render()
env.reset()
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]

while True:

    observation = env.reset()
    episode_length = 0

    rewards = []
    action = np.zeros(shape=(8,))
    for timestep in range(100000000000000):
        env.render()
        time.sleep(0.08)
        prev_obs = observation

        for i, head in enumerate(action_heads):
            head_action, _ = head.act(prev_obs)
            action[i] = discretize_action(head_action)

        observation, reward, done, _ = env.step(action)
        rewards.append(reward)
        if done:
            print(f"Done at timestep: {timestep}")
            break
        episode_length = timestep

env.close()
