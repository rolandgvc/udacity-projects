import numpy as np

import torch
import torch.optim as optim

from .policies import ActorCriticPolicy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Batcher:
    '''Helper class for enabling agent to learn in batches'''
    def __init__(self, batch_size, data):
        self.batch_size = batch_size
        self.data = data
        self.data_length = len(data[0])
        self.reset()

    def reset(self):
        self.batch_start = 0
        self.batch_end = self.batch_start + self.batch_size

    def end(self):
        return self.batch_start >= self.data_length

    def next_batch(self):
        batch = []
        for d in self.data:
            batch.append(d[self.batch_start: self.batch_end])
        self.batch_start = self.batch_end
        self.batch_end = min(self.batch_start + self.batch_size, self.data_length)
        return batch

    def shuffle(self):
        indices = np.arange(self.data_length)
        np.random.shuffle(indices)
        self.data = [d[indices] for d in self.data]


class PPO:
    '''PPO Learning Agent'''
    def __init__(self, state_size, action_size, fc1_size=64, fc2_size=64, num_agents=1, seed=0, batch_size=128,
                 lr=0.00025, tau=0.95, gamma=0.99, nminibatches=4, cliprange=0.2, vf_coef=0.5, ent_coef=0.01, learn_every=1):
        self.state_size = state_size
        self.action_size = action_size
        self.fc1_size = fc1_size
        self.fc2_size = fc2_size
        self.num_agents = num_agents
        self.seed = seed
        self.batch_size = batch_size
        self.lr = lr
        self.tau = tau
        self.gamma = gamma
        self.nminibatches = nminibatches
        self.cliprange = cliprange
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.learn_every = learn_every
        
        self.policy = ActorCriticPolicy(self.state_size, self.action_size, self.fc1_size, self.fc2_size, self.seed).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)

        self.t_step = 0

    def act(self, states):
        states = torch.from_numpy(states).float().to(device)   
        self.policy.eval()
        with torch.no_grad():
            actions, log_probs, _, values = self.policy(states)
            actions = actions.cpu().data.numpy()
            log_probs = log_probs.cpu().data.numpy()
            values = values.cpu().data.numpy()
        self.policy.train()

        return actions, log_probs, values

    def step(self, trajectories):
        self.t_step += 1
        if self.t_step % self.learn_every == 0:
            # Update on collected trajectories
            for _ in range(self.nminibatches):
                self._learn(trajectories)

    def _learn(self, trajectories):
        states = torch.Tensor(trajectories['state'])
        rewards = torch.Tensor(trajectories['reward'])
        old_probs = torch.Tensor(trajectories['prob'])
        actions = torch.Tensor(trajectories['action'])
        old_values = torch.Tensor(trajectories['value'])
        dones = torch.Tensor(trajectories['done'])

        # Calculate the advantages (TD online style)
        processed_rollout = [None] * (len(dones))
        advantages = torch.Tensor(np.zeros((self.num_agents, 1)))
        i_max = len(states)
        for i in reversed(range(i_max)):
            terminals_ = 1. - torch.Tensor(dones[i]).unsqueeze(1)
            rwrds_ = torch.Tensor(rewards[i]).unsqueeze(1)
            values_ = torch.Tensor(old_values[i])
            next_value_ = old_values[min(i_max-1, i + 1)]

            td_error = rwrds_ + self.gamma * terminals_ * next_value_.detach()
            td_error -= values_.detach()
            advantages = advantages * self.tau * self.gamma * terminals_ + td_error
            processed_rollout[i] = advantages

        advantages = torch.stack(processed_rollout).squeeze(2)
        advantages = (advantages - advantages.mean()) / advantages.std()

        # Learn in batches
        batcher = Batcher(self.batch_size, [np.arange(states.size(0))])
        batcher.shuffle()
        while not batcher.end():
            batch_indices = batcher.next_batch()[0]
            batch_indices = torch.Tensor(batch_indices).long()

            loss = self._surrogate(self.policy,
                                   old_probs[batch_indices],
                                   states[batch_indices],
                                   actions[batch_indices],
                                   rewards[batch_indices],
                                   advantages[batch_indices])
            
            # Take gradient step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def _surrogate(self, policy, old_probs, states, actions, rewards, advantages):
        # Discount rewards
        discount = self.gamma**np.arange(len(rewards))
        dis_rewards = np.asarray(rewards)*discount[:, np.newaxis]
        # Convert to future rewards
        fut_rewards = dis_rewards[::-1].cumsum(axis=0)[::-1]
        # Normalize rewards
        mean = np.mean(fut_rewards, axis=1)
        std = np.std(fut_rewards, axis=1) + 1.0e-10
        norm_rewards = (fut_rewards-mean[:,np.newaxis]) / std[:,np.newaxis]
        
        # Convert data to tensors and move to device
        actions = torch.tensor(actions, dtype=torch.float, device=device)
        old_probs = torch.tensor(old_probs, dtype=torch.float, device=device)
        norm_rewards = torch.tensor(norm_rewards, dtype=torch.float, device=device)

        _, new_probs, entropy_loss, values = policy(states, actions)

        # Find new/old policy ratio
        ratio = torch.exp(new_probs - old_probs)

        # Define surrogate loss
        surr = ratio * advantages[:, :, np.newaxis]
        surr_clip = torch.clamp(ratio, 1-self.cliprange, 1+self.cliprange) * advantages[:, :, np.newaxis]
        vf_loss = torch.nn.MSELoss()
        norm_rewards = norm_rewards[:, :, np.newaxis]
        entropy_loss = entropy_loss[:, :, np.newaxis]
        
        loss = torch.min(surr, surr_clip) - self.vf_coef*vf_loss(values, norm_rewards) + self.ent_coef*entropy_loss
      
        return -loss.mean()
