import pickle

import numpy as np
import torch
from torch.utils.data import BatchSampler, SubsetRandomSampler

from agents import Agent
from nets.nets import Net


class VPGAgent(Agent):

    def __init__(self, state_space: int, action_space: int, hidden=50, lr=1e-2, gamma=0.9):

        self.state_space = state_space
        self.action_space = action_space
        self.lr = lr
        self.gamma = gamma

        self.net = Net(
            [
                {"input_dim": 4, "output_dim": 20, "activation": "sigmoid"},
                {"input_dim": 20, "output_dim": 2, "activation": "sigmoid"},
            ],
            observer=None
        )

        self.states = []
        self.new_states = []
        self.actions = []
        self.a_probs = []
        self.rewards = []

    def act(self, state):
        state = np.reshape(state, newshape=(self.state_space, -1))
        action_probs, _ = self.net.full_forward_propagation(state)
        action_probs = action_probs.squeeze()
        action = np.argmax(action_probs)
        # action = np.random.choice(2, p=action_probs)
        return action, action_probs[action]

    def store_transition(self, state, new_state, action, a_prob, reward):
        self.states.append(state)
        self.new_states.append(new_state)
        self.actions.append(action)
        self.a_probs.append(a_prob)
        self.rewards.append(reward)

    def train(self, batch_size=8):

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

        for batch in BatchSampler(SubsetRandomSampler(range(len(self.states))), batch_size, drop_last=False):
            states_batch = states[batch].numpy().reshape((self.state_space, -1))
            probs, _ = self.net.full_forward_propagation(states_batch)
            probs += 1e-8
            entropy = -np.sum(np.log(probs) * probs)
            actions_batch = actions[batch].numpy()
            probs = np.take_along_axis(probs, actions_batch, axis=0)
            loss = -(np.log(probs) * rewards[batch])
            dLoss = -1/probs * rewards[batch]
            self.net.train(states_batch, loss, dLoss, epochs=1, learning_rate=self.lr, actions=actions_batch)
            losses.append(loss)
            entropies.append(entropy)
        self._clear_buffers()

        return np.mean(losses), np.mean(entropies)

    def _clear_buffers(self):
        self.states = []
        self.new_states = []
        self.actions = []
        self.a_probs = []
        self.rewards = []

    def save(self, episode):
        with open(f"../pickles/vpg_agent_episode_{episode}.p", "wb") as file:
            pickle.dump(self, file)
