import numpy as np
import torch
from collections import deque

from ppo.trajectory import Trajectory

def train(agent, max_episodes=300, max_timesteps=1000, solved_reward=35, log_interval=10):
    episode_scores = []
    recent_scores = deque(maxlen=100)

    for i_episode in range(1, max_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        traj = Trajectory()

        for _ in range(max_timesteps):
            actions, log_probs, values = agent.act(states)

            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done

            traj.add(states, rewards, log_probs, actions, values, dones)
            states = next_states

            if np.any(dones):
                break

        agent.step(traj)

        # Metrics
        recent_scores.append(traj.score)
        episode_scores.append(traj.score)
        
        if i_episode % log_interval == 1:
            print('Episode {}\tAverage Score: {:.2f}\t'.format(i_episode, np.mean(recent_scores)))

        if np.mean(recent_scores) >= solved_reward:
            # print('Environemnt solved!. Episode {}\tAverage Score: {:.2f}\t'.format(i_episode, np.mean(recent_scores)))
            torch.save(agent.policy.state_dict(), 'solved_ppo.pth')
            # break

    return episode_scores
            
            
if __name__ == "__main__": 
    # Load the environment
    from unityagents import UnityEnvironment
    env = UnityEnvironment(file_name='Reacher.app')
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]
    num_agents = len(env_info.agents) # 20
    states = env_info.vector_observations
    state_size = states.shape[1] # 33
    action_size = brain.vector_action_space_size # 4

    # Load the agent
    from ppo.agent import PPO
    agent = PPO(state_size=state_size,
                action_size=action_size,
                fc1_size=400,
                fc2_size=300,
                nb_agents=num_agents)

    # Train the agent
    scores = train(agent)
    env.close()

    # Plot the scores
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode')
    plt.show()
