import pickle

import numpy as np

import gym
import pybulletgym  # register PyBullet enviroments with open ai gym

from agents.ppo_agent import PPOAgent
from agents.pytoch_nn_agent import PytorchNNAgent
from agents.vpg_agent import VPGAgent

env = gym.make("AntPyBulletEnv-v0")


def trim_results(res, max_len):
    for idx in range(len(res["loss"])):
        res["loss"][idx]            = res["loss"][idx][:max_len]

    for idx in range(len(res["entropy"])):
        res["entropy"][idx]         = res["entropy"][idx][:max_len]

    for idx in range(len(res["learning_rate"])):
        res["learning_rate"][idx]   = res["learning_rate"][idx][:max_len]

    res["episode_length"][0]    = res["episode_length"][0][:max_len]
    res["returns"][0]           = res["returns"][0][:max_len]


resume_from_episode = None
if resume_from_episode is not None:
    with open(f"../pickles/bac2/ant_action_heads_episode_{resume_from_episode}.p", "rb") as f:
        action_heads = pickle.load(f)
    with open("../pickles/bac2/results.p", "rb") as f:
        results = pickle.load(f)
        trim_results(results, resume_from_episode)
    i_episode = resume_from_episode
else:
    results = {
        "loss":             np.zeros(shape=(8,), dtype=object),
        "entropy":          np.zeros(shape=(8,), dtype=object),
        "learning_rate":    np.zeros(shape=(8,), dtype=object),
        "episode_length":   np.zeros(shape=(1,), dtype=object),
        "returns":          np.zeros(shape=(1,), dtype=object),
    }
    results["episode_length"][0] = []
    results["returns"][0] = []
    for i in range(8):
        results["loss"][i] = []
        results["entropy"][i] = []
        results["learning_rate"][i] = []

    action_heads = [
        PPOAgent(28, 5),
        PPOAgent(28, 5),
        PPOAgent(28, 5),
        PPOAgent(28, 5),
        PPOAgent(28, 5),
        PPOAgent(28, 5),
        PPOAgent(28, 5),
        PPOAgent(28, 5),
    ]
    i_episode = 0

print("loaded agent")

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


while True:
    observation = env.reset()
    episode_length = 0

    # Allocate space for action
    action      = np.zeros(shape=(8,))
    raw_action  = np.zeros(shape=(8,))
    action_prob = np.zeros(shape=(8,))

    # Return statistic
    ret                      = 0
    base_discount_factor    = 0.99
    discount_factor         = 0.99
    # Episode loop
    print(f"Starting episode {i_episode}")
    for timestep in range(1024):
        prev_obs = observation
        # Collect action vector from action heads
        for i, head in enumerate(action_heads):
            head_action, head_action_prob   = head.act(prev_obs)
            raw_action[i]                   = head_action
            action[i]                       = discretize_action(head_action)
            action_prob[i]                  = head_action_prob
        # Act
        observation, reward, done, _ = env.step(action)
        # Accumulate return statistic
        ret += reward * discount_factor
        discount_factor *= base_discount_factor
        if done:
            break
        # Store transitions for each of the action heads
        for i, head in enumerate(action_heads):
            head.store_transition(prev_obs, observation, raw_action[i], action_prob[i], reward)
        episode_length = timestep

    for i, action_head in enumerate(action_heads):
        loss_mean, entropy_mean, learning_rate = action_head.train(batch_size=64)
        results["loss"][i]          .append(loss_mean)
        results["entropy"][i]       .append(entropy_mean)
        results["learning_rate"][i] .append(learning_rate)

    results["episode_length"][0]    .append(episode_length)
    results["returns"][0]           .append(ret)

    if i_episode % 500 == 0:
        with open("../pickles/results.p", "wb") as file:
            pickle.dump(results, file)
            print(f"Saved results for episode {i_episode}")

    # Save agent
    if i_episode % 500 == 0:
        with open(f"../pickles/ant_action_heads_episode_{i_episode}.p", "wb") as f:
            pickle.dump(action_heads, f)

    print(f"Finished episode {i_episode}, total return: {ret}")
    i_episode += 1

env.close()
