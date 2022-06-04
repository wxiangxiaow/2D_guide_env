from random import random
import gym
import numpy as np
import matplotlib.pyplot as plt
import math
import random
import torch

class UAVEnv(object):
    viewer_switch = None

    def __init__(self, seed):
        self.done = False
        
        ###ENVS ARGS####
        self.seed = seed
        np.random.seed(self.seed)
        self.Max_epochs = 500
        self.xmin = -10.
        self.xmax = 10.
        self.ymin = -10.
        self.ymax = 10.
        self.xita_max = 2 * math.pi
        self.attacker_space = gym.spaces.Box( np.array([self.xmin, self.ymin]), np.array([self.xmax, self.ymax]) )
        
    
        ### UAV ARGS###
        #attacker init  
        self.xita = 0
        self.a_x, self.a_y = self.init_position()
        '''
        action space:{velocity, angle velocity}
        '''
        self.angle_w = 0
        self.velocity = 0.2      #inital speed = 200m/s
        self.velocity_max = 0.25
        self.velocity_min = 0.1
        self.w_max = math.pi/3
        self.w_min = 0.
        self.action_space = gym.spaces.Box( np.array([self.velocity_min, self.w_min]), np.array([self.velocity_max, self.w_max]) )
        
        #Stationary target init
        self.t_x, self.t_y = self.init_position()
        self.t_xita = random.uniform(0, math.pi)

        ###REWARD ARGS####
        self.d_max = 0.5
        self.rewards = 0
        self.reward_step = 0

        self.pre_d = math.sqrt((self.a_x-self.t_x)**2 + (self.a_y-self.t_y)**2)
        self.info = False #Successful
    
    def init_position(self):
        x = random.randint(self.xmin, self.xmax)
        y = random.randint(self.ymin, self.ymax)
        return x, y
    
    def get_reward(self, ax, ay, xita, t):
        '''
        Calculate Reward:
        1.Victory/Out of Space/Max Steps/None of them:T
        2.Compared with the previous step:F
        '''
        d = math.sqrt((self.t_x - ax)**2 + (self.t_y - ay)**2)
        F = -0.01 * ( d - self.pre_d)

        if d < self.d_max:
            T = 100
            done = True
            self.info = True
        elif [ax, ay] not in self.attacker_space:
            T = -2 
            done = True
        elif t == self.Max_epochs:
            T = -1
            done = True

        reward_step = T + 0.99 * F
        return done, reward_step, d

        
    def step(self, action, t):
        #decode action
        ts = 1
        v = action[0] * 0.15 + 0.10
        w = (action[1]-0.5) * 2 * self.w_max

        xita = self.xita + w * ts
        if xita > math.pi*2 :
            xita = xita - 2*math.pi
        elif xita < 0:
            xita = xita + 2*math.pi
        
        ax = self.a_x + math.cos(xita) * v * ts
        ay = self.a_y + math.sin(xita) * v * ts

        done, rstep, d = self.get_reward(ax, ay, xita, t)       
        #refresh
        self.velocity = v
        self.xita = xita
        self.a_x = ax
        self.a_y = ay    
        self.rewards = self.reward_step + self.rewards
        self.done = done
        self.reward_step = rstep
        self.pre_d = d

        return self.get_state(), rstep, done, self.info

    def get_state(self):
        '''
        St = [agent position | target postion | relative position]
        '''
        a_x = self.a_x / self.xmax
        a_y = self.a_y / self.ymax
        a_xita = self.xita / (math.pi*2)
        t_x = self.t_x / self.xmax
        t_y = self.t_y / self.ymax
        t_xita = self.t_xita / (math.pi*2)
        d_x = t_x - a_x
        d_y = t_y - a_y
        d_xita = t_xita - a_xita

        state = [a_x, a_y,a_xita, t_x, t_y, t_xita, d_x, d_y, d_xita]
        return state

    def get_epoch_reward(self):
        return self.rewards

    def reset(self):
        '''
        Reset the state of attacker and target
        return a list of state
        '''
        self.done = False
        #attacker init   
        self.xita = 0
        self.a_x, self.a_y = self.init_position()
        '''
        action space:{velocity, angle velocity}
        '''
        self.angle_w = 0
        self.velocity = 200.      #inital speed = 200m/s
        #Stationary target init
        self.t_x, self.t_y = self.init_position()
        self.t_xita = random.uniform(0, math.pi)
        
        self.rewards = 0
        self.reward_step = 0
        
        if self.viewer_switch:
            self.viewer.reset([self.t_x, self.t_y])
        self.info = False #Successful
        self.pre_d = math.sqrt((self.a_x-self.t_x)**2 + (self.a_y-self.t_y)**2)

        return self.get_state()
    
    def render(self):
        '''
        for 3D render, need
        1.Target position(Now static)
        2.Attacker postion(Move)
        '''
        if self.viewer_switch is None:
            self.viewer = Viewer([self.t_x, self.t_y])
            self.viewer_switch = True
        self.viewer.render([self.a_x, self.a_y])
        
#visualization
class Viewer(object):
    def __init__(self, target_position):
        self.fig = plt.figure()
        plt.xlabel('x')
        plt.ylabel('y')

        ##plot target as a ball##
        plt.scatter(target_position[0], target_position[1], s=4)
        #store attacker position
        self.logs = []
        
    def render(self, attacker_position):
        plt.ion()
        self.logs.append(attacker_position)
        x = np.array(self.logs)[:,0]
        y = np.array(self.logs)[:,1]

        plt.plot(x, y, 'r')
        plt.pause(0.01)
    
    def reset(self, target_position):
        plt.cla()
        ##plot target as a ball##
        plt.scatter(target_position[0], target_position[1], s=4)
        #for store attacker position
        self.logs = []



if __name__ == '__main__':
    '''
    random test for render part
    '''
    center = [3,3]
    viewer = Viewer(center)
    x, y= 10, 10
    for i in range(50):
        x = x-0.08
        y = y-0.08
        viewer.render([x, y])
    
        
