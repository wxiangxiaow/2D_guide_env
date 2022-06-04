import torch
import numpy as np
from normalization import Normalization, RewardScaling
from replaybuffer import ReplayBuffer
from ppo_continuous import PPO_continuous
import env_2022
import os
from matplotlib import pyplot as plt

def draw_trajectory(states, times):
    '''
    Draw trajectory of agents
    input:whole logs of the episode(before normalization)
    output:none 
    '''
    save_dir = './test_img/'
    if not os.path.exists(save_dir):
        os,os.mkdir(save_dir)
    save_path = save_dir + 'Evaluate_No_' + str(times) + '.png'
    target_position = [states[0][3] * 10, states[0][4] * 10]
    #print(target_position)
    plt.scatter(target_position[0], target_position[1], s=4)
    agent_position = []
    for i in range(len(states)):
        agent_position.append([states[i][0]*10, states[i][1]*10])
    #print(states)
    x = np.array(agent_position)[:, 0]
    y = np.array(agent_position)[:, 1]
    plt.plot(x, y, 'r')
    plt.savefig(save_path)
    plt.cla()
    print('Test '+ str(times) + ' . Image had saved.')




def main():
    times = 10
    evaluate_reward = 0
    seed = 10
    env = env_2022.UAVEnv(seed=seed)
    #state_norm = Normalization(shape=len(env.get_state()))
    checkpoint = './checkpoint/PPO_continuous_Beta_env_UAV_number_7_seed_10.pth.tar'
    state_dict = torch.load(checkpoint)
    #print(state_dict)
    agent = state_dict['agent']
    state_norm = state_dict['Normalization']

    
    for t in range(times):
        log_ep = []
        s = env.reset()
        log_ep.append(s)
        s = state_norm(s, update=False)  # During the evaluating,update=False
        done = False
        episode_reward = 0
        step = 0
       
        pre_a = 0
        while not done:
            step += 1
            s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)
            if step % 5 == 1:
                a = agent.actor.mean(s).detach().numpy().flatten()  # We use the deterministic policy during the evaluating
                #dist = actor.get_dist(s)
                #a = dist.sample().numpy().flatten()
            else:
                a = pre_a
            action = a
            s_, r, done, _ = env.step(action, step)
            log_ep.append(s_)
            #env.render()
            s_ = state_norm(s_, update=False)
            episode_reward += r
            s = s_    
            pre_a = a
        print('episode '+str(t)+' , rewards = '+str(episode_reward))
        draw_trajectory(log_ep, t)
        
        evaluate_reward += episode_reward

    return evaluate_reward / times
    

if __name__ == '__main__':
    reward = main()
    print("evaluate_reward:{} \t".format(reward))
