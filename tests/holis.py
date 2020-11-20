import pickle
import gym
import pybulletgym  # register PyBullet enviroments with open ai gym

action_heads = pickle.load(open("../pickles/ant_action_heads_episode_5500.p", "rb"))
env = gym.make("AntPyBulletEnv-v0")

print("cmon man")
