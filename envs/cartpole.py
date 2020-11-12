import pickle

import numpy as np

import gym

from agents.ppo_agent import PPOAgent
from agents.pytoch_nn_agent import PytorchNNAgent
from agents.vpg_agent import VPGAgent

env = gym.make("CartPole-v0")

results = {
    "loss": [],
    "episode_length": [],
    "entropy": [],
    "learning_rate": [],
}
agent = VPGAgent(4, 2)
i_episode = 0

mean_losses = []
mean_entropies = []
mean_episode_lengths = []
learning_rates = []

while True:

    observation = env.reset()
    episode_length = 0

    for timestep in range(200):
        prev_obs = observation
        action, action_prob = agent.act(prev_obs)
        observation, reward, done, _ = env.step(action)
        if done:
            break
        agent.store_transition(prev_obs, observation, action, action_prob, reward)
        episode_length = timestep

    loss_mean, entropy_mean, learning_rate = agent.train()
    mean_losses.append(loss_mean)
    mean_entropies.append(entropy_mean)
    mean_episode_lengths.append(episode_length)
    learning_rates.append(learning_rate)

    if i_episode % 100 == 0:
        print("Saved results")
        results["loss"] = mean_losses
        results["entropy"] = mean_entropies
        results["episode_length"] = mean_episode_lengths
        results["learning_rate"] = learning_rates
        if i_episode % 1000 == 0:
            agent.save(i_episode)
        with open("../pickles/results.p", "wb") as file:
            pickle.dump(results, file)

    i_episode += 1

env.close()

