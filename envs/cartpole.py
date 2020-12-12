import pickle

import numpy as np

import gym

from configed_logging import log
from agents.ppo_agent import PPOAgent
from agents.pytoch_nn_agent import PytorchNNAgent
from agents.vpg_agent import VPGAgent


def cartpole(to_file=True, episodes=None):

    loop_forever = False
    if episodes is None:
        loop_forever = True

    env = gym.make("CartPole-v0")

    results = {
        "loss": [],
        "episode_length": [],
        "entropy": [],
        "learning_rate": [],
    }
    agent = PPOAgent(4, 2)
    i_episode = 0

    mean_losses = []
    mean_entropies = []
    mean_episode_lengths = []
    learning_rates = []

    while loop_forever or i_episode < episodes:

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

        results["loss"] = mean_losses
        results["entropy"] = mean_entropies
        results["episode_length"] = mean_episode_lengths
        results["learning_rate"] = learning_rates

        if i_episode % 100 == 0:
            log.info(f"Finished episode {i_episode}")
        if to_file:
            if i_episode % 100 == 0:
                with open("../pickles/results.p", "wb") as file:
                    pickle.dump(results, file)
            if i_episode % 1000 == 0:
                agent.save(i_episode)

        i_episode += 1

    env.close()
    return results


if __name__ == '__main__':
    cartpole()
