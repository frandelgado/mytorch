import time
import numpy as np

import gym  # open ai gym
import pybulletgym  # register PyBullet enviroments with open ai gym

env = gym.make('AntPyBulletEnv-v0')
env.render()
env.reset()
while True:
    action = np.zeros(shape=(8, 1))
    action[1, 0] = 1
    observation, reward, done, _ = env.step(action)
    env.render()
    time.sleep(0.5)
