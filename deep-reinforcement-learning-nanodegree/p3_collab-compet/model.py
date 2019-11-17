import torch
from torch import nn
import numpy as np
from collections import deque
import random

# Xavier init for linear layers
def linear_init(layer):
    classname = layer.__class__.__name__
    if classname.find('Linear') != -1:
        fan_in = layer.weight.data.size()[0]
        lim = 1./np.sqrt(fan_in)

        nn.init.uniform_(layer.weight.data, -lim, lim)


class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidden_size, seed=0):
        super(Actor, self).__init__()
        torch.manual_seed(seed)

        self.actor = nn.Sequential(
            nn.BatchNorm1d(state_size),
            nn.Linear(state_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size//2),
            nn.BatchNorm1d(hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, action_size),
            nn.Tanh()
        )
        self.apply(linear_init)

    def forward(self, state):
        return self.actor(state)


class Critic(nn.Module):
    def __init__(self, state_size, hidden_size, seed=0):
        super(Critic, self).__init__()
        torch.manual_seed(seed)

        self.first_layer = nn.Sequential(
            nn.BatchNorm1d(state_size),
            nn.Linear(state_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
        )

        self.critic = nn.Sequential(
            nn.Linear(hidden_size+2, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        self.apply(linear_init)

    def forward(self, state, action):
        p_state = self.first_layer(state)
        a_state = torch.cat((p_state, action), dim=1)
        return self.critic(a_state)

