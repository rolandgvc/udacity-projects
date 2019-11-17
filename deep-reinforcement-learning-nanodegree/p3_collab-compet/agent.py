import torch
from torch import nn, optim
from torch.nn import functional as F
import numpy as np
from collections import deque
import random
import copy
from model import Actor, Critic


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ReplayBuffer():
    def __init__(self, buffer_size=100000, seed=0):
        self.buffer = deque(maxlen=buffer_size)
        random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        experiences = random.sample(self.buffer, k=batch_size)
        states      = torch.tensor([e[0] for e in experiences]).float().to(device)
        actions     = torch.tensor([e[1] for e in experiences]).float().to(device)
        rewards     = torch.tensor([e[2] for e in experiences]).float().unsqueeze(-1).to(device)
        next_states = torch.tensor([e[3] for e in experiences]).float().to(device)
        dones       = torch.tensor([e[4] for e in experiences]).float().to(device)
        return (states, actions, rewards, next_states, dones)


class Agent():
    def __init__(self, state_size, action_size, num_agents=1, hidden_size=512, batch_size=256, update_epochs=3, eps=1.0,
                 grad_clip=1.0, gamma=.99, lr=0.001, seed=0):
        random.seed(seed)
        self.batch_size = batch_size
        self.update_epochs = update_epochs
        self.eps = eps
        self.num_agents = num_agents
        self.grad_clip = grad_clip
        self.gamma = gamma
        self.lr = lr

        self.actor_local = Actor(state_size, action_size, hidden_size, seed).to(device)
        self.actor_target = Actor(state_size, action_size, hidden_size, seed).to(device)
        self.actor_target.load_state_dict(self.actor_local.state_dict())

        self.critic_local = Critic(state_size, hidden_size, seed).to(device)
        self.critic_target = Critic(state_size, hidden_size, seed).to(device)
        self.critic_target.load_state_dict(self.critic_local.state_dict())

        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.lr)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.lr, weight_decay=0)

        self.replay_buffer = ReplayBuffer()

        self.noise = OUNoise((self.num_agents, 2), seed)

    def act(self, state, add_noise=True):
        state = torch.tensor(state).to(device).float()
        actions = np.zeros((self.num_agents, 2))
        with torch.no_grad():
            self.actor_local.eval()
            for agent in range(self.num_agents):
                actions[agent, :] = self.actor_local(state[agent:agent+1, :]).cpu().data.numpy()
            self.actor_local.train()

        if add_noise:
            actions = actions + self.eps * self.noise.sample()

        return np.clip(actions, -1, 1)

    def reset(self):
        self.noise.reset()

    def step(self, state, action, reward, next_state, done):
        for agent in range(self.num_agents):
            self.replay_buffer.add(state[agent, :], action[agent, :], reward[agent], next_state[agent, :], done)

        if len(self.replay_buffer.buffer) > self.batch_size:
            for _ in range(self.update_epochs):
                experiences = self.replay_buffer.sample(self.batch_size)
                self.learn(experiences)

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0-tau) * target_param.data)

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        # Train critic to predict the current state's Q_value
        next_actions = self.actor_target(next_states)
        next_Q_targets = self.critic_target(next_states, next_actions)
        Q_targets = rewards + (self.gamma * next_Q_targets * (1-dones))

        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)

        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), self.grad_clip)
        self.critic_optimizer.step()

        # Train actor to predict the best action according to critic
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()

        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # soft-update the target networks
        self.soft_update(self.critic_local, self.critic_target, 0.002)
        self.soft_update(self.actor_local, self.actor_target, 0.002)

        if self.eps > 0:
            self.eps -= 0.000001
            self.noise.reset()
            if self.eps <= 0:
                self.eps = 0


class OUNoise:
    """Ornstein-Uhlenbeck process."""
    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.01):
        self.size = size
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        self.state = copy.copy(self.mu)

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        self.state = x + dx
        return self.state