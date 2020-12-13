import time
from configed_logging import log

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
        return 0
    if a == 3:
        return 0.5
    if a == 4:
        return 1
    raise ValueError


def ant_test(enable_rendering=True, episode_length=1024, agent_save=30000, agent_type="ant"):
    action_heads = pickle.load(open(f"../pickles/{agent_type}/ant_action_heads_episode_{agent_save}.p", "rb"))

    env = gym.make("AntPyBulletEnv-v0")
    if enable_rendering:
        env.render()

    env.reset()
    observation = env.reset()

    action = np.zeros(shape=(8,))
    dist_to_target = 1000
    for timestep in range(episode_length):
        if enable_rendering:
            env.render()
            time.sleep(0.08)

        prev_obs = observation
        for i, head in enumerate(action_heads):
            head_action, _ = head.act(prev_obs)
            action[i] = discretize_action(head_action)

        observation, reward, done, info = env.step(action)
        dist_to_target = info["dist_to_target"]
        if done:
            log.info(f"Done at timestep: {timestep}")
            break

    env.close()
    distance_covered = 1000 - dist_to_target
    log.info(f"Distance covered: {distance_covered}")
    return distance_covered


if __name__ == '__main__':
    ant_test()
