import numpy as np

class Trajectory(object):
    def __init__(self):
        self.traj = {}
        self.traj['state'] = []
        self.traj['reward'] = []
        self.traj['prob'] = []
        self.traj['action'] = []
        self.traj['value'] = []
        self.traj['done'] = []
        self.score = 0.

    def add(self, states, rewards, log_probs, actions, values, dones):
        self.traj['state'].append(states)
        self.traj['reward'].append(rewards)
        self.traj['prob'].append(log_probs)
        self.traj['action'].append(actions)
        self.traj['value'].append(values)
        self.traj['done'].append(dones)
        self.score += np.mean(rewards)


    def __len__(self):
        return len(self.traj['state'])

    def __getitem__(self, key):
        return self.traj[key]
