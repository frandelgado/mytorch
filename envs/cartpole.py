import pickle

import numpy as np

import gym

from agents.vpg_agent import VPGAgent

env = gym.make("CartPole-v0")

results = {
    "loss": [],
    "episode_length": [],
    "entropy": [],
    "layers": []
}
i_episode = 0
agent = VPGAgent(4, 2)
while True:

    observation = env.reset()
    max_time = 0
    loss_accum = []
    entropy_accum = []
    ep_accum = []

    for t in range(100):
        prev_obs = observation
        action, action_prob = agent.act(observation)
        observation, reward, done, info = env.step(action)
        if done:
            break
        agent.store_transition(prev_obs, observation, action, action_prob, reward)
        max_time = t

    loss_mean, entropy_mean = agent.train(batch_size=1)
    loss_accum.append(loss_mean)
    entropy_accum.append(entropy_mean)
    ep_accum.append(max_time)

    if i_episode % 1000 == 0:
        print("Saved results")
        results["loss"].append(np.mean(loss_accum))
        results["entropy"].append((np.mean(entropy_accum)))
        results["episode_length"].append(np.mean(ep_accum))
        results["layers"].append(agent.net.expose_layers())
        agent.save(i_episode)
        with open("../pickles/results.p", "wb") as file:
            pickle.dump(results, file)

    i_episode += 1
env.close()

