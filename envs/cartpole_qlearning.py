import pickle

import numpy as np

import gym

from agents.qlearning_agent import QAgent

env = gym.make("CartPole-v0")

results = {
    "loss": [],
    "episode_length": [],
}
agent = QAgent(env.observation_space, env.action_space)

for i_episode in range(401):
    observation = env.reset()
    max_time = 0
    ep_accum = []
    agent.set_epsilon(i_episode)
    agent.set_learning_rate(i_episode)

    for t in range(200):
        prev_obs = observation
        action = agent.act(observation)
        observation, reward, done, info = env.step(action)
        if done:
            break
        agent.store_transition(prev_obs, observation, action, 0, reward)
        max_time = t

    results["episode_length"].append(max_time)
    if i_episode % 100 == 0:
        print("Saved results")
        agent.save(i_episode)
        with open("../pickles/q_agent_results/results4.p", "wb") as file:
            pickle.dump(results, file)

env.close()
