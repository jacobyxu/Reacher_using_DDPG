from unityagents import UnityEnvironment
import numpy as np
import random
import torch
from collections import deque
from ddpg_agent import Agent

def main():
    env = UnityEnvironment(file_name='data/Reacher_Linux/Reacher.x86_64')
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]
    # number of agents
    num_agents = len(env_info.agents)
    # size of each action
    action_size = brain.vector_action_space_size
    # examine the state space 
    states = env_info.vector_observations
    state_size = states.shape[1]

    n_agent = 20
    agent = Agent(state_size=state_size, action_size=action_size, random_seed=2, n_agent=n_agent)
    # load trained model
    agent.actor_local.load_state_dict(torch.load('model/checkpoint_actor.pth'))
    agent.critic_local.load_state_dict(torch.load('model/checkpoint_critic.pth'))

    state = env.reset()
    env_info = env.reset(train_mode=False)[brain_name]
    state = env_info.vector_observations
    for t in range(1000):
        action = [agent.act(state[agent_x], agent_x, add_noise=False) for agent_x in range(n_agent)]
        env_info = env.step(action)[brain_name]
        next_state = env_info.vector_observations
        reward = env_info.rewards
        done = env_info.local_done
        state = next_state
        if all(done):
            break

    env.close()
    
if __name__ == "__main__":
    main()
