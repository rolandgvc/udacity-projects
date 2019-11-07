import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

# Xavier weight initialization for better learning
def w_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class ActorCriticPolicy(nn.Module):
    def __init__(self, state_size, action_size, fc1_size, fc2_size, seed):
        super().__init__()

        self.actor = Actor(state_size, action_size, fc1_size, fc2_size, seed)
        self.critic = Critic(state_size, fc1_size, fc2_size, seed)
        self.std = torch.Tensor(nn.Parameter(torch.ones(1, action_size)))

    def forward(self, states, actions=None):
        estimated_actions = self.actor(states)
        estimated_values = self.critic(states)

        # Define a gaussian distribution over the actions given by actor net
        gaussian = Normal(estimated_actions, self.std)
        i_dim = 2 # when evaluating: shape (batch_size, 20)

        if isinstance(actions, type(None)): 
            i_dim = 1 # when collecting trajectory: shape (20)
            actions = gaussian.sample()

        log_prob = torch.sum(gaussian.log_prob(actions), dim=i_dim, keepdim=True)
        entropy_loss = torch.sum(gaussian.entropy(), dim=i_dim) / 4

        return actions, log_prob, entropy_loss, estimated_values


class Actor(nn.Module):
    def __init__(self, state_size, action_size, fc1_size, fc2_size, seed):
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.fc3 = nn.Linear(fc2_size, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*w_init(self.fc1))
        self.fc2.weight.data.uniform_(*w_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))


class Critic(nn.Module):
    def __init__(self, state_size, fc1_size, fc2_size, seed):
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.fc3 = nn.Linear(fc2_size, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*w_init(self.fc1))
        self.fc2.weight.data.uniform_(*w_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
