import pickle

import numpy as np
import torch
from torch.utils.data import BatchSampler, SubsetRandomSampler

from agents import Agent
from nets.nets import Net


class VPGAgent(Agent):

    def __init__(self, state_space: int, action_space: int, hidden=50, lr=1e-4, gamma=0.9):

        self.state_space = state_space
        self.action_space = action_space
        self.lr = lr
        self.gamma = gamma
        self.episode_lengths = []

        self.net = Net(
            [
                {"input_dim": 4, "output_dim": 50, "activation": "sigmoid"},
                {"input_dim": 50, "output_dim": 2, "activation": "softmax"},
            ]
        )

        self.states = []
        self.new_states = []
        self.actions = []
        self.a_probs = []
        self.rewards = []

    def act(self, state):
        state = np.reshape(state, newshape=(self.state_space, -1))
        action_probs, _ = self.net.full_forward_propagation(state)
        action = np.random.choice(2, p=action_probs.squeeze())
        return action, action_probs[action]

    def store_transition(self, state, new_state, action, a_prob, reward):
        self.states.append(state)
        self.new_states.append(new_state)
        self.actions.append(action)
        self.a_probs.append(a_prob)
        self.rewards.append(reward)

    def train(self, batch_size=6):

        # Unroll rewards
        rewards = np.array(self.rewards)
        reward = 0
        for i in reversed(range(len(self.rewards))):
            rewards[i] += self.gamma * reward
            reward = rewards[i]

        states = torch.tensor(self.states, dtype=torch.float)
        actions = torch.tensor(self.actions, dtype=torch.long).view(-1, 1)
        rewards = rewards.reshape(-1, 1)

        losses = []
        entropies = []

        self.episode_lengths.append(len(self.rewards))
        # calcular LR
        avg_length_window = np.mean(self.episode_lengths[-100:])
        exp = -0.02 * avg_length_window - 2
        learning_rate = 10 ** exp

        for batch in BatchSampler(SubsetRandomSampler(range(len(self.states))), batch_size, drop_last=False):
            states_batch = states[batch].numpy()
            actions_batch = actions[batch].numpy()
            rewards_batch = rewards[batch]
            grads_values_batch = []
            loss = []
            entropy = []
            for state, action, reward in zip(states_batch, actions_batch, rewards_batch):
                state = state.reshape((-1, 1))
                probs, cache = self.net.full_forward_propagation(state)
                probs = probs.squeeze() + 1e-8
                entropy.append(-np.sum(np.log(probs) * probs))
                action_prob = probs[action]
                loss.append(-(np.log(action_prob) * reward))
                dLoss = 1/action_prob * reward
                grads_values = self.net.full_backward_propagation(dLoss, cache, action)
                grads_values_batch.append(grads_values)

            losses.append(np.mean(loss))
            entropies.append(np.mean(entropy))
            grads_values_mean = self.net.mean_grads(grads_values_batch, batch_size)

            self.net.update(grads_values_mean, learning_rate)
        self._clear_buffers()

        return np.mean(losses), np.mean(entropies), learning_rate

    def _clear_buffers(self):
        self.states = []
        self.new_states = []
        self.actions = []
        self.a_probs = []
        self.rewards = []

    def save(self, episode):
        with open(f"../pickles/vpg_agent_episode_{episode}.p", "wb") as file:
            pickle.dump(self, file)
