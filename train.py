from unityagents import UnityEnvironment
import numpy as np
import random
import torch
from collections import deque
from datetime import datetime
def MSG(txt):
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), str(txt))
from ddpg_agent import Agent

def ddpg(env, agent, n_agent, n_episodes=150, max_t=1000, print_every=100):
    """
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        print_every (int): number of episodes to print result
    """
    MSG('start!')
    brain_name = env.brain_names[0]
    scores_deque = deque(maxlen=print_every)
    scores = []
    best_score = 0.
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations
        agent.reset()
        agent_scores = [0]*n_agent
        for t in range(max_t):
            action = [agent.act(state[agent_x], agent_x) for agent_x in range(n_agent)]
            # get needed information from environment
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations
            reward = env_info.rewards
            done = env_info.local_done
            for agent_x in range(n_agent):
                agent_scores[agent_x] += reward[agent_x]
                agent.step(state[agent_x], action[agent_x], reward[agent_x], 
                           next_state[agent_x], done[agent_x], agent_x)
            state = next_state
            if any(done):
                break
        score = np.mean(agent_scores)
        scores_deque.append(score)
        scores.append(score)
        print('\rEpisode {}\tCurrent Episode Average Score: {:.2f}\tAverage Score on 100 Episode: {:.2f}'.format(i_episode, score, np.mean(scores_deque)), end="")
        if score > best_score:
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
            best_score = score
        if i_episode % (print_every/10) == 0:
            print('\rEpisode {}\tCurrent Episode Average Score: {:.2f}\tAverage Score on 100 Episode: {:.2f}'.format(i_episode, score, np.mean(scores_deque)))
        if score > 30 and np.mean(scores_deque) > 30:
            break
    MSG('\nend!') 
    return scores

def main():
    # select this option to load version 2 (with 20 agents) of the environment
    env = UnityEnvironment(file_name='data/Reacher_Linux/Reacher.x86_64')
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]
    # number of agents
    n_agent = len(env_info.agents)
    # size of each action
    action_size = brain.vector_action_space_size
    # size of state space 
    state_size = env_info.vector_observations.shape[1]
    # train
    agent = Agent(state_size=state_size, action_size=action_size, random_seed=2, n_agent=n_agent)
    scores = ddpg(env, agent, n_agent)

if __name__ == "__main__":
    main()


