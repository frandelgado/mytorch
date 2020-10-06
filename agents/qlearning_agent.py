import pickle

import numpy as np
import math
from agents import Agent


class QAgent(Agent):

    def __init__(
            self, observation_space, action_space, buckets=(1, 1, 6, 12,),
            n_episodes=1000, min_lr=0.1, min_epsilon=0.1, gamma=1.0, action_space_cardinality=2
    ):
        self.buckets = buckets
        self.n_episodes = n_episodes
        self.gamma = gamma
        self.action_space_cardinality = action_space_cardinality
        self.Q = np.zeros(self.buckets + (action_space_cardinality,))
        self.state_upper_bounds = [observation_space.high[0], 0.5, observation_space.high[2], np.math.radians(50) / 1.]
        self.state_lower_bounds = [observation_space.low[0], -0.5, observation_space.low[2], -np.math.radians(50) / 1.]
        self.action_space = action_space
        self.min_lr = min_lr
        self.min_epsilon = min_epsilon

    def discretize(self, state):
        discrete_state = []
        for i in range(len(state)):
            scaling = (state[i] + abs(self.state_lower_bounds[i])) / (self.state_upper_bounds[i] - self.state_lower_bounds[i])
            new_state = int(round((self.buckets[i] - 1) * scaling))
            discrete_state.append(min(self.buckets[i] - 1, max(0, new_state)))
        return tuple(discrete_state)

    def act(self, state):
        discrete_state = self.discretize(state)
        if np.random.random() < self.epsilon:
            return self.action_space.sample()
        return np.argmax(self.Q[discrete_state])

    def store_transition(self, state, new_state, action, a_prob, reward):
        discrete_state = self.discretize(state)
        discrete_new_state = self.discretize(new_state)
        self.Q[discrete_state][action] += self.lr * (
            reward + self.gamma * np.max(self.Q[discrete_new_state]) - self.Q[discrete_state][action])

    def train(self, *args, **kwargs):
        return

    def set_epsilon(self, t):
        self.epsilon = max(self.min_epsilon, min(1., 1. - math.log10((t + 1) / 25)))

    def set_learning_rate(self, t):
        self.lr = max(self.min_lr, min(1., 1. - math.log10((t + 1) / 25)))

    def save(self, episode):
        with open(f"../pickles/q_agent_episode_{episode}.p", "wb") as file:
            pickle.dump(self, file)

