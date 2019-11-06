import numpy as np
import torch

def train(model, memory, max_episodes=50000, max_timesteps=300, update_timestep=20, solved_reward=35, log_interval=1):

    timestep = 0
    for i_episode in range(1, max_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        scores = np.zeros(num_agents)        

        for _ in range(max_timesteps):
            timestep += 1
            # Running policy_old:
            actions = [model.policy_old.act(state, memory) for state in states]
            actions = np.clip(actions, -1, 1)                  # all actions between -1 and 1
            env_info = env.step(actions)[brain_name]           # send all actions to tne environment
            next_states = env_info.vector_observations         # get next state (for each agent)
            rewards = env_info.rewards                         # get reward (for each agent)
            dones = env_info.local_done                        # see if episode finished
            scores += env_info.rewards                         # update the score (for each agent)
            states = next_states                               # roll over states to next time step

            # Saving rewards and is_terminal:
            memory.rewards.append(rewards)
            memory.is_terminals.append(dones)
            
            # update policy every 'x' timesteps
            if timestep % update_timestep == 0:
                model.update(memory)
                memory.clear_memory()
                timestep = 0 
            
            if np.any(dones):                                  # exit loop if episode finished
                break
        

        # stop training if avg_reward > solved_reward
        if np.mean(scores) > solved_reward:
            print("########## Solved! ##########")
            torch.save(model.policy.state_dict(), './PPO_reacher.pth')
            break
            
        # logging
        if i_episode % log_interval == 0:            
            print('Episode {} \t reward: {}'.format(i_episode, np.mean(scores)))








if __name__ == "__main__":
    from unityagents import UnityEnvironment
    env = UnityEnvironment(file_name='Reacher.app')
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]
    num_agents = len(env_info.agents)             # 20
    states = env_info.vector_observations
    
    state_size = states.shape[1]          
    action_size = brain.vector_action_space_size  # 4
    hidden_size = 64
    lr = 0.002
    gamma = 0.99
    K_epochs = 5
    eps_clip = 0.2

    from ppo import PPO, Memory
    model = PPO(state_size, action_size, hidden_size, lr, gamma, K_epochs, eps_clip)
    memory = Memory()

    train(model, memory)

    env.close()
