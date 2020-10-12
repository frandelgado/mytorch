import pickle

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.distributions import Categorical
from torch.utils.data import SubsetRandomSampler, BatchSampler

from agents import Agent


class PytorchNNAgent(Agent):

    def __init__(self, state_space: int, action_space: int, hidden=50, lr=1e-3, gamma=0.9):
        self.state_space = state_space
        self.action_space = action_space
        self.lr = lr
        self.gamma = gamma

        self.net = Net(state_space, hidden, action_space)
        self.optimizer = optim.SGD(self.net.parameters(), lr=lr)

        self.states = []
        self.new_states = []
        self.actions = []
        self.a_probs = []
        self.rewards = []

    def act(self, state):
        state = torch.tensor(state, dtype=torch.float).view(-1, self.state_space)
        with torch.no_grad():
            probs = self.net(state)
        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), probs[:, action.item()].item()

    def train(self, *args):

        losses = []
        entropies = []

        rewards = np.array(self.rewards)
        reward = 0
        for i in reversed(range(len(self.rewards))):
            rewards[i] += self.gamma * reward
            reward = rewards[i]

        states = torch.tensor(self.states, dtype=torch.float)
        actions = torch.tensor(self.actions, dtype=torch.long).view(-1, 1)
        rewards = torch.tensor(rewards, dtype=torch.float).view(-1, 1)

        for batch in BatchSampler(SubsetRandomSampler(range(len(self.states))), 1, drop_last=False):
            # Calculate loss
            probs = self.net(states[batch])
            probs_copy = probs.clone().detach().numpy()
            entropy = -np.sum(np.log(probs_copy)*probs_copy)
            probs = probs.gather(1, actions[batch])
            loss = -(torch.log(probs) * rewards[batch]).mean()  # Mean is here for when batch_size > 1
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses.append(loss)
            entropies.append(entropy)

        self._clear_buffers()
        return torch.mean(torch.tensor(losses)).item(), torch.mean(torch.tensor(entropies)).item()

    def store_transition(self, state, new_state, action, a_prob, reward):
        self.states.append(state)
        self.new_states.append(new_state)
        self.actions.append(action)
        self.a_probs.append(a_prob)
        self.rewards.append(reward)

    def _clear_buffers(self):
        self.states = []
        self.new_states = []
        self.actions = []
        self.a_probs = []
        self.rewards = []

    def save(self, episode):
        with open(f"../pickles/pytorch_agent_episode_{episode}.p", "wb") as file:
            pickle.dump(self, file)


class Net(nn.Module):
    def __init__(self, input, hidden, output):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input, hidden)
        self.fc2 = nn.Linear(hidden, int(hidden/2))
        self.fc3 = nn.Linear(int(hidden/2), output)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x
