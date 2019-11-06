import torch
import torch.nn as nn
from torch.distributions import Categorical
import gym
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size, hidden_size):
        super(ActorCritic, self).__init__()

        self.actor = nn.Sequential(
                nn.Linear(state_size, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, action_size),
                nn.Softmax(dim=-1))

        self.critic = nn.Sequential(
                nn.Linear(state_size, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, 1))
        
    def forward(self):
        raise NotImplementedError
        
    def act(self, states, memory):
        state = torch.from_numpy(states).float().to(device)
        print('\n\nstates received', states.shape)
        m = Categorical(self.actor(state))
        action = m.sample()
        print('actions given', action.shape, '\n\n')

        # save experience to memory
        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(m.log_prob(action))
        
        return action.item()
    
    def evaluate(self, state, action):
        m = Categorical(self.actor(state))
        state_value = self.critic(state)
        
        return m.log_prob(action), torch.squeeze(state_value), m.entropy()
        

class PPO:
    def __init__(self, state_size, action_size, hidden_size, lr, gamma, K_epochs, eps_clip):
        self.lr = lr
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.policy = ActorCritic(state_size, action_size, hidden_size).to(device)
        self.policy_old = self.policy

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.MseLoss = nn.MSELoss()
    
    def update(self, memory):
        # Discount rewards
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # Normalize rewards
        rewards = torch.tensor(rewards).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        
        # Convert states, actions, logprobs to tensors
        old_states = torch.stack(memory.states).to(device).detach()
        old_actions = torch.stack(memory.actions).to(device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(device).detach()
        
        # Perform K updates according to state values
        for _ in range(self.K_epochs):
            # Evaluate old policy
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            
            # Find new/old policy ratio:
            ratios = torch.exp(logprobs - old_logprobs.detach()) # same as probs / old_probs
                
            # Find Surrogate Loss:
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # Final loss: surrogate - baseline_loss + entropy
            loss = torch.mean(-torch.min(surr1, surr2) - 0.5*self.MseLoss(state_values, rewards) + 0.01*dist_entropy)
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())

