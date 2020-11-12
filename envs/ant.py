import pickle
import numpy as np
import gym
import pybulletgym  # register PyBullet enviroments with open ai gym


from agents.ant.ppo_agent import PPOAgent
from agents.pytoch_nn_agent import PytorchNNAgent
from agents.vpg_agent import VPGAgent

env = gym.make("AntPyBulletEnv-v0")

env.reset()

obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]

results = {
    "loss": [],
    "episode_length": [],
    "entropy": [],
    "learning_rate": [],
    "reward_sums": [],
    "returns": []
}

# From https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ppo/core.py : 84
log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
agent = PPOAgent(obs_dim, act_dim, categorical=False, log_std=log_std)
i_episode = 0

mean_losses = []
mean_entropies = []
mean_episode_lengths = []
learning_rates = []
reward_sums = []
returns = []

while True:

    observation = env.reset()
    episode_length = 0

    rewards = []
    for timestep in range(2048):
        prev_obs = observation
        action, action_prob, _, _ = agent.act(prev_obs)  # TODO consider using entropy returned here
        observation, reward, done, _ = env.step(action)
        rewards.append(reward)
        if done:
            print(f"Done at timestep: {timestep}")
            break
        agent.store_transition(prev_obs, observation, action, action_prob, reward)
        episode_length = timestep

    loss_mean, entropy_mean, learning_rate = agent.train()
    mean_losses.append(loss_mean)
    mean_entropies.append(entropy_mean)
    mean_episode_lengths.append(episode_length)
    learning_rates.append(learning_rate)
    reward_sums.append(np.sum(rewards))

    ret = 0
    discount = 0.9
    base_discount = 0.9
    for reward in rewards:
        ret += reward*discount
        discount *= base_discount
    returns.append(ret)

    if i_episode % 10 == 0:
        print(f"Saved results for episode: {i_episode}")
        results["loss"] = mean_losses
        results["entropy"] = mean_entropies
        results["episode_length"] = mean_episode_lengths
        results["learning_rate"] = learning_rates
        results["reward_sums"] = reward_sums
        results["returns"] = returns
        if i_episode % 10 == 0:
            agent.save(i_episode)
        with open("../pickles/results.p", "wb") as file:
            pickle.dump(results, file)

    i_episode += 1

env.close()

