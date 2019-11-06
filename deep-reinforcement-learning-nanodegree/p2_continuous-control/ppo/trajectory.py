import numpy as np

class Trajectory(object):
    def __init__(self):
        self.data = {}
        self.data['state'] = []
        self.data['reward'] = []
        self.data['prob'] = []
        self.data['action'] = []
        self.data['value'] = []
        self.data['done'] = []
        self.score = 0.

    def add(self, states, rewards, probs, actions, values, dones):
        # multi-agent
        self.data['state'].append(states)
        self.data['reward'].append(rewards)
        self.data['prob'].append(probs)
        self.data['action'].append(actions)
        self.data['value'].append(values)
        self.data['done'].append(dones)
        self.score += np.mean(rewards)
        
        # single-agent
        # traj = zip(states, rewards, probs, actions, values)
        # for state, reward, prob, action, value in traj:
        #     self.data['state'].append(state)
        #     self.data['reward'].append(reward)
        #     self.data['prob'].append(prob)
        #     self.data['action'].append(action)
        #     self.data['value'].append(value)
        #     self.score += np.mean(reward)

    def __len__(self):
        return len(self.data['state'])

    def __getitem__(self, key):
        return self.data[key]
