import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from unityagents import UnityEnvironment

# Environment Setup
env = UnityEnvironment(file_name="Banana.app")
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
env_info = env.reset(train_mode=True)[brain_name]
state = env_info.vector_observations[0]
state_size = len(state)
action_size = brain.vector_action_space_size


def train_dqn(model, n_episodes=2000, max_t=1000, eps_decay=0.995):
    episode_scores = []
    recent_scores = deque(maxlen=100)
    eps = 1.0
    
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]
        score = 0
        
        for t in range(max_t):
            action = model.predict(state, eps)
            env_info = env.step(action)[brain_name]        
            next_state = env_info.vector_observations[0]   
            reward = env_info.rewards[0]                   
            done = env_info.local_done[0]

            model.step(state, action, reward, next_state, done)                  
            state = next_state                             

            score += reward
            if done:
                break
        
        recent_scores.append(score)
        episode_scores.append(score)
        eps = max(0.1, eps_decay*eps)
        
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(recent_scores)), end="")
        
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(recent_scores)))

        if np.mean(recent_scores) >= 13.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(recent_scores)))
            torch.save(model.local_network.state_dict(), 'dqn_banana_checkpoint.pth')
            break
    
    return episode_scores


from dqn import DQN
model = DQN(state_size=state_size, action_size=action_size, seed=0)        
scores = train_dqn(model)
print("scores", scores)

# # plot the scores
# fig = plt.figure()
# ax = fig.add_subplot(111)
# plt.plot(np.arange(len(scores)), scores)
# plt.ylabel('Score')
# plt.xlabel('Episode #')
# plt.show()

env.close()