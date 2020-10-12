import pickle

import numpy as np
import torch
from torch.utils.data import BatchSampler, SubsetRandomSampler

from agents import Agent
from nets.nets import Net


class VPGAgent(Agent):

    def _normalize_state(self, state):
        state[0] /= 4.8
        state[1] /= 0.5
        state[2] /= 0.41887903
        state[3] /= 0.8726646259971648
        return state

    def __init__(self, state_space: int, action_space: int, hidden=50, lr=1e-4, gamma=0.9):

        self.state_space = state_space
        self.action_space = action_space
        self.lr = lr
        self.gamma = gamma

        self.net = Net(
            [
                {"input_dim": 4, "output_dim": 20, "activation": "sigmoid"},
                {"input_dim": 20, "output_dim": 10, "activation": "sigmoid"},
                {"input_dim": 10, "output_dim": 1, "activation": "sigmoid"},
            ],
            observer=None
        )

        self.states = []
        self.new_states = []
        self.actions = []
        self.a_probs = []
        self.rewards = []

    def act(self, state):
        state = self._normalize_state(state)
        state = np.reshape(state, newshape=(self.state_space, -1))
        action_probs, _ = self.net.full_forward_propagation(state)
        action_probs = action_probs.squeeze()
        action_probs = [action_probs, 1 - action_probs]
        action = np.random.choice(2, p=action_probs)
        return action, action_probs[action]

    def store_transition(self, state, new_state, action, a_prob, reward):
        self.states.append(self._normalize_state(state))
        self.new_states.append(self._normalize_state(new_state))
        self.actions.append(action)
        self.a_probs.append(a_prob)
        self.rewards.append(reward)

    def train(self, batch_size=1):

        # Unroll rewards
        rewards = np.array(self.rewards)
        reward = 0
        for i in reversed(range(len(self.rewards))):
            rewards[i] += self.gamma * reward
            reward = rewards[i]

        # Normalize rewards
        # rewards -= rewards.mean()
        # std = rewards.std()
        # if std != 0:
        #     rewards /= std

        states = torch.tensor(self.states, dtype=torch.float)
        actions = torch.tensor(self.actions, dtype=torch.long).view(-1, 1)
        rewards = rewards.reshape(-1, 1)

        losses = []
        entropies = []

        for timestep in range(len(states)):
            states_batch = states[timestep].numpy().reshape((self.state_space, -1))
            probs, cache = self._get_probs(states_batch)
            entropy = -np.sum(np.log(probs) * probs)
            actions_batch = actions[timestep].numpy()
            selected_probabilities = np.take_along_axis(probs, actions_batch, axis=0)

            loss = -(np.log(selected_probabilities) * rewards[timestep])
            # dLoss = -1/selected_probabilities * rewards[timestep]

            grads_values = self.net.full_backward_propagation(np.array([[1]]), cache, actions)
            self.net.update(
                grads_values,
                reward=rewards[timestep],
                timestep=timestep,
                gamma=self.gamma,
                probability=selected_probabilities,
                learning_rate=self.lr
            )

            losses.append(loss)
            entropies.append(entropy)
        self._clear_buffers()

        return np.mean(losses), np.mean(entropies)

    def _get_probs(self, states):
        probs, cache = self.net.full_forward_propagation(states)
        probs = probs.squeeze()
        probs = np.array([probs, 1 - probs])
        probs += 1e-8
        return probs, cache

    def _clear_buffers(self):
        self.states = []
        self.new_states = []
        self.actions = []
        self.a_probs = []
        self.rewards = []

    def save(self, episode):
        with open(f"../pickles/vpg_agent_episode_{episode}.p", "wb") as file:
            pickle.dump(self, file)
