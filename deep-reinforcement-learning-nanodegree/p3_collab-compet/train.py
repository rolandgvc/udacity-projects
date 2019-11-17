from unityagents import UnityEnvironment
import numpy as np
import torch
from matplotlib import pyplot as plt
from agent import Agent
from collections import deque
import time

def train_ddpg(agent, episodes=300, max_t=500, print_every=10):
    max_scores = []
    scores_window = deque(maxlen=100)
    solved = False
    max_score = 0

    for i_episode in range(1, episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        scores = np.zeros(num_agents)
        agent.reset()

        for _ in range(max_t):
            actions = agent.act(states, add_noise=True)
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done

            agent.step(states, actions, rewards, next_states, dones)
            states = next_states
            scores += rewards
            if np.any(dones):
                break

        avg_score = np.max(scores)
        max_scores.append(avg_score)
        scores_window.append(avg_score)
        avg_list.append(np.average(scores_window))  
        
        if i_episode % print_every == 0:
            print('\rEpisode {}\tAverage Score: {:.4f}'.format(i_episode, np.mean(scores_window)))

        if np.mean(scores_window) >= 0.5:
            if not solved:
                print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)),'\n')
                solved = True
            if np.mean(scores_window) > max_score:
                max_score = np.mean(scores_window)
                torch.save(agent.actor_local.state_dict(), 'best_actor.pth')
                torch.save(agent.critic_local.state_dict(), 'best_critic.pth')
    
    return max_scores

# Load the environment
env = UnityEnvironment(file_name="Tennis.app", no_graphics=True)
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
env_info = env.reset(train_mode=True)[brain_name]
num_agents = len(env_info.agents)
action_size = brain.vector_action_space_size
states = env_info.vector_observations
state_size = states.shape[1]

# Train agent
avg_list = []
agent = Agent(state_size, action_size, num_agents)
scores = train_ddpg(agent)

# Plot
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores)+1), scores)
plt.plot(np.arange(1, len(avg_list)+1), avg_list)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()
