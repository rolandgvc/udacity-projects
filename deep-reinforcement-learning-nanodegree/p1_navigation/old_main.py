import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

#%% ENVIRONMENT AND AGENT

env = gym.make('LunarLander-v2')
env.seed(0)

from dqn import DQN
model = DQN(state_size=8, action_size=4, seed=0)

#%% UNTRAINED TEST

# state = env.reset()
# for j in range(200):
#     action = model.predict(state)
#     env.render()
#     state, reward, done, _ = env.step(action)
#     if done:
#         break
#
# env.close()

#%% TRAINING THE AGENT

def train_dqn(n_episodes=2000, max_t=1000, eps_decay=0.995):
    episode_scores = []
    recent_scores = deque(maxlen=100)
    eps = 1.0

    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        score = 0

        for t in range(max_t):
            action = model.predict(state, eps)
            next_state, reward, done, _ = env.step(action)
            model.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break

        recent_scores.append(score)
        episode_scores.append(score)
        eps = max(0.01, eps_decay*eps)

        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(recent_scores)), end="")

        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(recent_scores)))

        if np.mean(recent_scores) >= 200.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(recent_scores)))
            torch.save(model.local_network.state_dict(), 'checkpoint.pth')
            break

    return episode_scores


scores = train_dqn()

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()