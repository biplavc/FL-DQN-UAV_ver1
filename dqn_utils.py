import numpy as np

from collections import deque

import time


import math
import random
from collections import namedtuple
from itertools import count
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as T

import os
import datetime
import json

from gym_UAV import *

class ReplayMemory():
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        
    def add(self, state, action, reward, done, next_state):
        experience = (state, action, reward, done, next_state)
        self.buffer.append(experience)
        
    def sample(self, batch_size):
        if self.count() < batch_size:
            batch = random.sample(self.buffer, self.count())
        else:
            batch = random.sample(self.buffer, batch_size)
            
        state_batch = np.array([np.array(experience[0]) for experience in batch])
        action_batch = np.array([experience[1] for experience in batch])
        reward_batch = np.array([experience[2] for experience in batch])
        done_batch = np.array([experience[3] for experience in batch])
        next_state_batch = np.array([np.array(experience[4]) for experience in batch])
        
        return state_batch, action_batch, reward_batch, done_batch, next_state_batch
    
    def count(self):
        return len(self.buffer)



class DQN(nn.Module):
    def __init__(self, num_actions):
        super(DQN, self).__init__()
        # self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=0)
        # # self.bn1 = nn.BatchNorm2d(16)
        # self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0)
        # # self.bn2 = nn.BatchNorm2d(32)
        # self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        # # self.bn3 = nn.BatchNorm2d(64)
        
        self.h1 = nn.Linear(state_size, 64)
        self.h2 = nn.Linear(64, 64)
        self.h3 = nn.Linear(64, 64)
        self.h4 = nn.Linear(64, 64)
        self.h5 = nn.Linear(64, 64)
        self.h6 = nn.Linear(64, 64)
        self.h7 = nn.Linear(64, num_actions)
        
    def forward(self, inputs):
        out = F.relu(self.h1(inputs))
        out = F.relu(self.h2(out))
        out = F.relu(self.h3(out))
        out = F.relu(self.h4(out))
        out = F.relu(self.h5(out))
        out = F.relu(self.h6(out))
        out = F.relu(self.h7(out))

        
        # out = out.view(out.size(0), -1)
        # out = F.relu(self.fc1(out))
        # out = self.fc2(out)
        
        return out



class PongAgent:
    def __init__(self, args, name = ""):
        # self.env = make_env(args.env_name)
        self.env = UAV_network(3, {0:[1,2,3]}, "UAV_network", "None", {1:0,2:0,3:0}, {1:0,2:0,3:0}, {1:2,2:1,3:1}) ## biplav
        self.num_actions = self.env.action_space.n
        
        self.args = args
        self.name = name
        
        self.dqn = DQN(self.num_actions)
        self.target_dqn = DQN(self.num_actions)
        
        if args.use_gpu:
            self.dqn.cuda()
            self.target_dqn.cuda()        
        
        self.buffer = ReplayMemory(1000000 // 4)
        
        self.gamma = 0.99
        
        self.mse_loss = nn.MSELoss()
        self.optim = optim.RMSprop(self.dqn.parameters(), lr=0.0001)
        
        self.out_dir = './model'
        
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

        
    def to_var(self, x):
        x_var = Variable(x)
        if self.args.use_gpu:
            x_var = x_var.cuda()
        return x_var

        
    def predict_q_values(self, states):
        states = self.to_var(torch.from_numpy(states).float())
        actions = self.dqn(states)
        return actions

    
    def predict_q_target_values(self, states):
        states = self.to_var(torch.from_numpy(states).float())
        actions = self.target_dqn(states)
        return actions

    
    def select_action(self, state, epsilon):
        choice = np.random.choice([0, 1], p=(epsilon, (1 - epsilon)))
        
        if choice == 0:
            return np.random.choice(range(self.num_actions))
        else:
            state = np.expand_dims(state, 0)
            actions = self.predict_q_values(state)
            return np.argmax(actions.data.cpu().numpy())

        
    def update(self, states, targets, actions):
        targets = self.to_var(torch.unsqueeze(torch.from_numpy(targets).float(), -1))
        actions = self.to_var(torch.unsqueeze(torch.from_numpy(actions).long(), -1))
        
        predicted_values = self.predict_q_values(states)
        affected_values = torch.gather(predicted_values, 1, actions)
        loss = self.mse_loss(affected_values, targets)
        
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        
    def get_epsilon(self, total_steps, max_epsilon_steps, epsilon_start, epsilon_final):
        return max(epsilon_final, epsilon_start - total_steps / max_epsilon_steps)

    
    def sync_target_network(self):
        primary_params = list(self.dqn.parameters())
        target_params = list(self.target_dqn.parameters())
        for i in range(0, len(primary_params)):
            target_params[i].data[:] = primary_params[i].data[:]
            
            
    def calculate_q_targets(self, next_states, rewards, dones):
        dones_mask = (dones == 1)
        predicted_q_target_values = self.predict_q_target_values(next_states)
        next_max_q_values = np.max(predicted_q_target_values.data.cpu().numpy(), axis=1)
        next_max_q_values[dones_mask] = 0 # no next max Q values if the game is over
        q_targets = rewards + self.gamma * next_max_q_values
        
        return q_targets
    
        
    def load_model(self, filename):
        self.dqn.load_state_dict(torch.load(filename))
        self.sync_target_network()
        
        
    def play(self, episodes):
        rewards = []
        # for i in range(1, episodes + 1):
        for i in range(0, episodes): # biplav
            done = False
            state = self.env.reset()
            total_reward = 0
            while not done:
                action = self.select_action(state, 0) # force to choose an action from the network
                state, reward, done, _ = self.env.step(action)
                total_reward += reward
                # self.env.render()
            rewards.append(total_reward)
            
        return rewards
            
                
    def close_env(self):
        self.env.close()
        
        
    def train(self, replay_buffer_fill_len, batch_size, episodes,
              max_epsilon_steps, epsilon_start, epsilon_final, sync_target_net_freq):


        rewards = []
        running_rewards = []
        total_steps = 0
        running_episode_reward = 0
        
        state = self.env.reset()
        for i in range(replay_buffer_fill_len):
            action = self.select_action(state, 1) # force to choose a random action
            next_state, reward, done, _ = self.env.step(action)
            
            self.buffer.add(state, action, reward, done, next_state)
            
            state = next_state
            if done:
                self.env.reset()

                
        # main loop - iterate over episodes
        for i in range(1, episodes + 1):
            # reset the environment
            done = False
            state = self.env.reset()
            
            # reset episode reward and length
            episode_reward = 0
            episode_length = 0
            
            # play until it is possible
            while not done:
                # synchronize target network with estimation network in required frequence
                if (total_steps % sync_target_net_freq) == 0:
                    self.sync_target_network()

                # calculate epsilon and select greedy action
                epsilon = self.get_epsilon(total_steps, max_epsilon_steps, epsilon_start, epsilon_final)
                action = self.select_action(state, epsilon)
                
                # execute action in the environment
                next_state, reward, done, _ = self.env.step(action)
                self.buffer.add(state, action, reward, done, next_state)
                
                # sample random minibatch of transactions
                s_batch, a_batch, r_batch, d_batch, next_s_batch = self.buffer.sample(batch_size)
                
                # estimate Q value using the target network
                q_targets = self.calculate_q_targets(next_s_batch, r_batch, d_batch)
                
                # update weights in the estimation network
                self.update(s_batch, q_targets, a_batch)
                
                # set the state for the next action selction and update counters and reward
                state = next_state
                total_steps += 1
                episode_length += 1
                episode_reward += reward
                
            running_episode_reward = running_episode_reward * 0.9 + 0.1 * episode_reward
            
            running_rewards.append(running_episode_reward)
            rewards.append(episode_reward)

            
        return rewards, running_rewards