import numpy as np
import random
import copy
import os
import yaml
from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from policies import ActorCriticPolicy

# Hyperparameters
LR = 0.0001
DISCOUNT = 0.99
EPSILON = 0, 1
BETA = 0.01
UPDATE_EVERY = 1
TAU = 0.95
BATCH_SIZE = 64
GRADIENT_CLIP = 5

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Batcher:
    '''Learn in batches. Inspired from: https://bit.ly/2yrYoHy'''
    def __init__(self, batch_size, data):
        self.batch_size = batch_size
        self.data = data
        self.num_entries = len(data[0])
        self.reset()

    def reset(self):
        self.batch_start = 0
        self.batch_end = self.batch_start + self.batch_size

    def end(self):
        return self.batch_start >= self.num_entries

    def next_batch(self):
        batch = []
        for d in self.data:
            batch.append(d[self.batch_start: self.batch_end])
        self.batch_start = self.batch_end
        self.batch_end = min(self.batch_start + self.batch_size, self.num_entries)
        return batch

    def shuffle(self):
        indices = np.arange(self.num_entries)
        np.random.shuffle(indices)
        self.data = [d[indices] for d in self.data]


class PPO:
    def __init__(self, nb_agents, state_size, action_size, hidden_size, lr, betas, gamma, K_epochs, eps_clip):
        self.nb_agents = nb_agents
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.K_epochs = K_epochs
        self.eps_clip = eps_clip
        
        self.policy = ActorCriticPolicy(self.state_size, self.action_size, self.hidden_size).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)

        self.t_step = 0

    def act(self, states):
        states = torch.from_numpy(states).float().to(device)   
        self.policy.eval()  # disable graph
        with torch.no_grad():
            actions, log_probs, _, values = self.policy(states)
            actions = actions.cpu().data.numpy()
            log_probs = log_probs.cpu().data.numpy()
            values = values.cpu().data.numpy()
        self.policy.train() # enable graph

        return actions, log_probs, values

    def step(self, trajectories, eps, beta):
        self.t_step += 1
        if self.t_step % UPDATE_EVERY == 0:
            for _ in range(self.K_epochs):
                self.learn(trajectories, eps, beta)

    def learn(self, trajectories, eps, beta):
        states = torch.Tensor(trajectories['state'])
        rewards = torch.Tensor(trajectories['reward'])
        old_probs = torch.Tensor(trajectories['prob'])
        actions = torch.Tensor(trajectories['action'])
        old_values = torch.Tensor(trajectories['value'])
        dones = torch.Tensor(trajectories['done'])
        nb = self.nb_agents

        # calculate the advantages
        processed_rollout = [None] * (len(dones))
        advantages = torch.Tensor(np.zeros((self.nb_agents, 1)))
        i_max = len(states)
        for i in reversed(range(i_max)):
            terminals_ = 1. - torch.Tensor(dones[i]).unsqueeze(1)
            rwrds_ = torch.Tensor(rewards[i]).unsqueeze(1)
            values_ = torch.Tensor(old_values[i])
            next_value_ = old_values[min(i_max-1, i + 1)]

            td_error = rwrds_ + DISCOUNT * terminals_ * next_value_.detach()
            td_error -= values_.detach()
            advantages = advantages * TAU * DISCOUNT * terminals_ + td_error
            processed_rollout[i] = advantages

        advantages = torch.stack(processed_rollout).squeeze(2)
        advantages = (advantages - advantages.mean()) / advantages.std()

        # learn in batches
        batcher = Batcher(BATCH_SIZE, [np.arange(states.size(0))])
        batcher.shuffle()
        while not batcher.end():
            batch_indices = batcher.next_batch()[0]
            batch_indices = torch.Tensor(batch_indices).long()

            loss = self._surrogate(self.policy,
                                          old_probs[batch_indices],
                                          states[batch_indices],
                                          actions[batch_indices],
                                          rewards[batch_indices],
                                          old_values[batch_indices],
                                          advantages[batch_indices],
                                          eps,
                                          beta)
            
            # Take gradient step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def _surrogate(self, policy, old_probs, states, actions, rewards, old_values, advantages, eps=0.1, beta=0.01):
        # discount rewards
        discount = DISCOUNT**np.arange(len(rewards))
        rewards = np.asarray(rewards)*discount[:, np.newaxis]
        # convert to future rewards
        rewards = rewards[::-1].cumsum(axis=0)[::-1]
        # normalize rewards
        mean = np.mean(rewards, axis=1)
        std = np.std(rewards, axis=1) + 1.0e-10
        rewards = (rewards-mean[:,np.newaxis]) / std[:,np.newaxis]
        
        # convert data to tensors and move to device
        actions = torch.tensor(actions, dtype=torch.float, device=device)
        old_probs = torch.tensor(old_probs, dtype=torch.float, device=device)
        old_values = torch.tensor(old_values, dtype=torch.float, device=device)
        rewards = torch.tensor(rewards, dtype=torch.float, device=device)

        _, new_probs, entropy_loss, values = policy(states, actions)

        # Find new/old policy ratio
        ratio = torch.exp(new_probs - old_probs)

        # Find surrogate loss
        surr = ratio * advantages[:, :, np.newaxis]
        surr_clip = torch.clamp(ratio, 1-eps, 1+eps) * advantages[:, :, np.newaxis]

        entropy_loss = entropy_loss[:, :, np.newaxis]
        
        loss = torch.mean(-torch.min(surr, surr_clip) + beta*entropy_loss) # add mse loss
        return -loss
