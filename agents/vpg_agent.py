import pickle

import numpy as np
import torch
from torch.utils.data import BatchSampler, SubsetRandomSampler

from agents import Agent
from nets.layers import Sigmoid, Softmax
from nets.nets import Net
from nets.optim import Adam


class VPGAgent(Agent):

    def __init__(self, state_space: int, action_space: int, hidden=10, lr=1e-3, gamma=0.9):
        # Config
        self._adapt_lr_on_ep_len = True

        self.state_space = state_space
        self.action_space = action_space
        self.lr = lr
        self.gamma = gamma
        self.episode_lengths = []
        
        self.net = Net(
            layers=[
                Sigmoid(state_space, hidden),
                Softmax(hidden, action_space)
            ], 
            optimizer=Adam(),
            lr=lr,
        )

        self.states = []
        self.new_states = []
        self.actions = []
        self.a_probs = []
        self.rewards = []

    def act(self, state):
        state = np.reshape(state, newshape=(self.state_space, -1))
        action_probs = self.net.forward(state)
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

        # Deprecated (LR is now fixed within the net)
        if self._adapt_lr_on_ep_len:
            self.episode_lengths.append(len(self.rewards))
            # calcular LR
            avg_length_window = np.mean(self.episode_lengths[-100:])
            exp = -0.02 * avg_length_window - 2
            learning_rate = 10 ** exp
        else:
            learning_rate = self.lr

        for batch in BatchSampler(SubsetRandomSampler(range(len(self.states))), batch_size, drop_last=False):
            states_batch = states[batch].numpy()
            actions_batch = actions[batch].numpy()
            rewards_batch = rewards[batch]
            loss = []
            entropy = []
            for state, action, reward in zip(states_batch, actions_batch, rewards_batch):
                state = state.reshape((-1, 1))
                probs = self.net.forward(state)
                probs = probs.squeeze() + 1e-8
                entropy.append(-np.sum(np.log(probs) * probs))
                action_prob = probs[action]
                loss.append(-(np.log(action_prob) * reward))
                dLoss = 1/action_prob * reward
                self.net.backward(dLoss, action)

            losses.append(np.mean(loss))
            entropies.append(np.mean(entropy))
            self.net.mean_grads(batch_size)
            self.net.update()
        self._clear_buffers()

        return np.mean(losses), np.mean(entropies), self.lr

    def _clear_buffers(self):
        self.states = []
        self.new_states = []
        self.actions = []
        self.a_probs = []
        self.rewards = []

    def save(self, episode):
        with open(f"../pickles/vpg_agent_episode_{episode}.p", "wb") as file:
            pickle.dump(self, file)
