import pickle

import numpy as np
import torch
from torch.utils.data import BatchSampler, SubsetRandomSampler

from agents import Agent
from nets.nets import Net


class PPOAgent(Agent):

    def __init__(self, state_space: int, action_space: int, a_hidden=50, c_hidden=50,
                 a_lr=3e-4, c_lr=3e-3, gamma=0.9, clip_e=0.2, categorical=True, log_std=None):
        self.a_lr = a_lr
        self.c_lr = c_lr
        self.state_space = state_space
        self.action_space = action_space
        self.gamma = gamma
        self.clip_e = clip_e
        self.categorical = categorical
        if not categorical and log_std is None:
            raise ValueError("Must supply log standard deviations for a non categorical (diagonal gaussian) policy")
        self.log_std = log_std

        self.actor = Net(
            [
                {"input_dim": state_space,  "output_dim": 50,           "activation": "sigmoid"},
                {"input_dim": 50,           "output_dim": 20,           "activation": "sigmoid"},
                {"input_dim": 20,           "output_dim": action_space, "activation": "linear"},
            ],
            optimizer="adam"
        )
        self.critic = Net(
            [
                {"input_dim": state_space,  "output_dim": 50,   "activation": "sigmoid"},
                {"input_dim": 50,           "output_dim": 50,   "activation": "sigmoid"},
                {"input_dim": 50,           "output_dim": 1,    "activation": "linear"},
            ],
            optimizer="adam"
        )

        self.states = []
        self.new_states = []
        self.actions = []
        self.a_probs = []
        self.rewards = []

    def act(self, state):
        state = np.reshape(state, newshape=(self.state_space, -1))
        if self.categorical:
            # Select according to a categorical distribution
            return self._act_categorical(state)
        else:
            # Select according to a diagonal gaussian
            return self._act_gaussian(state)

    def _act_gaussian(self, state):
        """
        Acts according to a diagonal gaussian policy
        :return: action, log_likelihood
        """
        means, cache = self.actor.full_forward_propagation(state)
        loc = torch.from_numpy(means)
        scale = torch.from_numpy(np.exp(self.log_std).reshape(-1, 1))
        normal = torch.distributions.Normal(loc=loc, scale=scale)
        action = normal.sample()
        prob = np.exp(normal.log_prob(action).numpy())
        entropy = normal.entropy()
        return action.numpy(), prob, entropy,  cache

    def _act_categorical(self, state):
        action_probs, cache = self.actor.full_forward_propagation(state)
        action = np.random.choice(self.action_space, p=action_probs.squeeze())
        entropy = -np.sum(np.log(action_probs) * action_probs)
        return action, action_probs[action], entropy, cache

    def store_transition(self, state, new_state, action, a_prob, reward):
        self.states.append(state)
        self.new_states.append(new_state)
        self.actions.append(action)
        self.a_probs.append(a_prob)
        self.rewards.append(reward)

    def train(self, batch_size=128):
        # PPO algorithm
        # Unroll rewards
        rewards = np.array(self.rewards)
        reward = 0
        for i in reversed(range(len(self.rewards))):
            rewards[i] += self.gamma * reward
            reward = rewards[i]

        states = torch.tensor(self.states, dtype=torch.float)
        actions = torch.tensor(self.actions, dtype=torch.float).view(-1, self.action_space)
        all_old_probs = torch.tensor(self.a_probs, dtype=torch.float).view(-1, self.action_space)
        rewards = rewards.reshape(-1, 1)

        actor_entropies = []
        actor_losses = []

        for batch in BatchSampler(SubsetRandomSampler(range(len(self.states))), batch_size, drop_last=False):
            states_batch = states[batch].numpy()
            actions_batch = actions[batch].numpy()
            old_action_probs = all_old_probs[batch].numpy()
            rewards_batch = rewards[batch]
            actor_grads_batch = []
            critic_grads_batch = []

            actor_loss = []
            actor_entropy = []
            for state, action, reward, old_action_probs in zip(states_batch, actions_batch, rewards_batch, old_action_probs):
                state = state.reshape((-1, 1))  # TODO maybe check shape? Does not coincide with act
                V, critic_cache = self.critic.full_forward_propagation(state)
                advantage = reward - V
                _, action_prob, entropy, actor_cache = self.act(state)
                actor_entropy.append(np.sum(entropy.numpy()))

                old_action_probs = old_action_probs.reshape(-1, 1)
                action_prob_ratio = action_prob/old_action_probs
                surr = action_prob_ratio * advantage

                clamp_prob = action_prob_ratio.copy()
                clamp_prob[action_prob_ratio < (1 - self.clip_e)] = 1 - self.clip_e
                clamp_prob[action_prob_ratio > (1 + self.clip_e)] = 1 + self.clip_e

                clipped_surr = clamp_prob * advantage

                actor_loss.append(-np.minimum(surr, clipped_surr))

                actor_dLoss = np.zeros(shape=(self.action_space, 1))
                for i in range(self.action_space):
                    if surr[i] < clipped_surr[i]:
                        actor_dLoss[i, 0] = 1 / old_action_probs[i] * advantage
                    else:
                        if action_prob_ratio[i] < 1 - self.clip_e or action_prob_ratio[i] > 1 + self.clip_e:
                            actor_dLoss[i, 0] = 0
                        else:
                            actor_dLoss[i, 0] = 1 / old_action_probs[i] * advantage

                actor_grads = self.actor.full_backward_propagation(actor_dLoss, actor_cache, action)
                actor_grads_batch.append(actor_grads)

                critic_loss = np.square(advantage)
                critic_dloss = 2 * advantage
                critic_grads = self.critic.full_backward_propagation(critic_dloss, critic_cache, action)
                critic_grads_batch.append(critic_grads)

            actor_grads_values_mean = self.actor.mean_grads(actor_grads_batch, batch_size)
            self.actor.update(actor_grads_values_mean, self.a_lr)
            actor_losses.append(np.mean(actor_loss))
            actor_entropies.append(np.mean(actor_entropy))

            critic_grads_values_mean = self.critic.mean_grads(critic_grads_batch, batch_size)
            self.critic.update(critic_grads_values_mean, self.c_lr)

        self._clear_buffers()
        return np.mean(actor_losses), np.mean(actor_entropies), self.a_lr

    def _clear_buffers(self):
        self.states = []
        self.new_states = []
        self.actions = []
        self.a_probs = []
        self.rewards = []


    def save(self, episode):
        with open(f"../pickles/ppo_agent_episode_{episode}.p", "wb") as file:
            pickle.dump(self, file)
