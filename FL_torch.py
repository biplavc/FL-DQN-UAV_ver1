# import gym
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
from atari_utils import *
from dqn_utils import *


os.environ['CUDA_VISIBLE_DEVICES'] = "1"

class ARGS():
    def __init__(self):
        # self.env_name = 'PongDeterministic-v4'
        self.env_name = UAV_network(3, {0:[1,2,3]}, "UAV_network", "None", {1:0,2:0,3:0}, {1:0,2:0,3:0}, {1:2,2:1,3:1})

        self.render = False
        self.episodes = 1500
        self.batch_size = 32
        self.epsilon_start = 1.0
        self.epsilon_final=0.02
        self.seed = 1773
        
        self.use_gpu = torch.cuda.is_available()
        
        self.mode = ["rl", "fl_normal"][1] ## biplav
        
        self.number_of_samples = 5 if self.mode != "rl" else 1
        self.fraction = 0.4 if self.mode != "rl" else 1
        self.local_steps = 50 if self.mode != "rl" else 100
        self.rounds = 25 if self.mode != "rl" else 25
        
        
        self.max_epsilon_steps = self.local_steps*200
        self.sync_target_net_freq = self.max_epsilon_steps // 10
        
        self.folder_name = f"runs/{self.mode}/" + time.asctime(time.gmtime()).replace(" ", "_").replace(":", "_")
        
        self.replay_buffer_fill_len = 1_000
        
        
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



class FederatedLearning:
    
    def __init__(self, args):
        self.args = args
        self.main_agent = PongAgent(args, "main")
        
        self.create_clients()
        
        self.logs = {}
        
    def create_clients(self):
        self.clients = {}
        self.client_names = []
        self.updated_clients = {}
        for i in range(self.args.number_of_samples):
            self.client_names.append(f"client_{i}")
            self.clients[f"client_{i}"] = PongAgent(args, i)
            self.updated_clients[f"client_{i}"] = 0
            
            
    def update_clients(self):
        with torch.no_grad():
            def update(client_layer, main_layer):
                client_layer.weight.data = main_layer.weight.data.clone()
                client_layer.bias.data = main_layer.bias.data.clone()
                
            for i in range(self.args.number_of_samples):
                update(self.clients[self.client_names[i]].dqn.conv1, self.main_agent.dqn.conv1)
                update(self.clients[self.client_names[i]].dqn.conv2, self.main_agent.dqn.conv2)
                update(self.clients[self.client_names[i]].dqn.conv3, self.main_agent.dqn.conv3)
                
                
                update(self.clients[self.client_names[i]].dqn.fc1, self.main_agent.dqn.fc1)
                update(self.clients[self.client_names[i]].dqn.fc2, self.main_agent.dqn.fc2)
                

                del self.clients[self.client_names[i]].buffer
                self.clients[self.client_names[i]].buffer = ReplayMemory(1000000 // 4)

                del self.clients[self.client_names[i]].target_dqn
                self.clients[self.client_names[i]].target_dqn = DQN(self.main_agent.num_actions)
                self.clients[self.client_names[i]].target_dqn.load_state_dict(self.clients[self.client_names[i]].dqn.state_dict()) 
                
                if self.args.use_gpu:
                    self.clients[self.client_names[i]].target_dqn.cuda()     



    def update_main_agent(self, round_no):
        # meaning
        conv1_mean_weight = torch.zeros(size=self.main_agent.dqn.conv1.weight.shape).to(device)
        conv1_mean_bias = torch.zeros(size=self.main_agent.dqn.conv1.bias.shape).to(device)

        conv2_mean_weight = torch.zeros(size=self.main_agent.dqn.conv2.weight.shape).to(device)
        conv2_mean_bias = torch.zeros(size=self.main_agent.dqn.conv2.bias.shape).to(device)

        conv3_mean_weight = torch.zeros(size=self.main_agent.dqn.conv3.weight.shape).to(device)
        conv3_mean_bias = torch.zeros(size=self.main_agent.dqn.conv3.bias.shape).to(device)

        linear1_mean_weight = torch.zeros(size=self.main_agent.dqn.fc1.weight.shape).to(device)
        linear1_mean_bias = torch.zeros(size=self.main_agent.dqn.fc1.bias.shape).to(device)

        linear2_mean_weight = torch.zeros(size=self.main_agent.dqn.fc2.weight.shape).to(device)
        linear2_mean_bias = torch.zeros(size=self.main_agent.dqn.fc2.bias.shape).to(device)
        
        number_of_samples = self.args.number_of_samples
        with torch.no_grad():

            for i in range(number_of_samples):
                conv1_mean_weight += self.clients[self.client_names[i]].dqn.conv1.weight.clone()
                conv1_mean_bias += self.clients[self.client_names[i]].dqn.conv1.bias.clone()

                conv2_mean_weight += self.clients[self.client_names[i]].dqn.conv2.weight.clone()
                conv2_mean_bias += self.clients[self.client_names[i]].dqn.conv2.bias.clone()

                conv3_mean_weight += self.clients[self.client_names[i]].dqn.conv3.weight.clone()
                conv3_mean_bias += self.clients[self.client_names[i]].dqn.conv3.bias.clone()

                linear1_mean_weight += self.clients[self.client_names[i]].dqn.fc1.weight.clone()
                linear1_mean_bias += self.clients[self.client_names[i]].dqn.fc1.bias.clone()

                linear2_mean_weight += self.clients[self.client_names[i]].dqn.fc2.weight.clone()
                linear2_mean_bias += self.clients[self.client_names[i]].dqn.fc2.bias.clone()

                
            conv1_mean_weight = conv1_mean_weight / number_of_samples
            conv1_mean_bias = conv1_mean_bias / number_of_samples

            conv2_mean_weight = conv2_mean_weight / number_of_samples
            conv2_mean_bias = conv2_mean_bias / number_of_samples

            conv3_mean_weight = conv3_mean_weight / number_of_samples
            conv3_mean_bias = conv3_mean_bias / number_of_samples

            linear1_mean_weight = linear1_mean_weight / number_of_samples
            linear1_mean_bias = linear1_mean_bias / number_of_samples

            linear2_mean_weight = linear2_mean_weight / number_of_samples
            linear2_mean_bias = linear2_mean_bias / number_of_samples
            
            
            with torch.no_grad():
                def update(main_layer, averaged_layer_weight, averaged_layer_bias):
                    main_layer.weight.data = averaged_layer_weight.data.clone()
                    main_layer.bias.data = averaged_layer_bias.data.clone()
                
                update(self.main_agent.dqn.conv1, conv1_mean_weight, conv1_mean_bias)
                update(self.main_agent.dqn.conv2, conv2_mean_weight, conv2_mean_bias)
                update(self.main_agent.dqn.conv3, conv3_mean_weight, conv3_mean_bias)
                
                
                update(self.main_agent.dqn.fc1, linear1_mean_weight, linear1_mean_bias)
                update(self.main_agent.dqn.fc2, linear2_mean_weight, linear2_mean_bias)
            

        
        
    def step(self, idx_users, round_no):
        
        self.update_clients()
        
        for user in idx_users:
            print(f"Client {user}")
            
            rewards, running_rewards = self.clients[self.client_names[user]].train(
                replay_buffer_fill_len = self.args.replay_buffer_fill_len, 
                batch_size = self.args.batch_size, 
                episodes = self.args.local_steps,
                max_epsilon_steps = self.args.max_epsilon_steps,
                epsilon_start = self.args.epsilon_start - 0.03*(round_no - 1),
                epsilon_final = self.args.epsilon_final,
                sync_target_net_freq = self.args.sync_target_net_freq)
            
            print(f'LOCAL TRAIN: Avg Reward: {np.array(rewards).mean():.5f},  Avg Running Reward: {np.array(running_rewards).mean():.5f}')
            

            self.logs[f"{round_no}"]["train"]["rewards"].append(rewards)
            self.logs[f"{round_no}"]["train"]["running_rewards"].append(running_rewards)
            
        self.update_main_agent(round_no)
            
        self.logs[f"{round_no}"]["eval"]["rewards"] = self.main_agent.play(10)
        
        
        
    def run(self):
        
        m = max(int(self.args.fraction * self.args.number_of_samples), 1) 
        for round_no in range(self.args.rounds):
            
            self.logs[f"{round_no + 1}"] = {"train": {
                                            "rewards": [],
                                            "running_rewards": []
                                        },
                                        "eval": {
                                            "rewards": None
                                        }
                                       }
            idxs_users = np.random.choice(range(self.args.number_of_samples), m, replace=False)

            for user in idxs_users:
                self.updated_clients[f"client_{user}"] = round_no + 1
                
            self.step(idxs_users, round_no+1)
            print(f'{round_no + 1}/{self.args.rounds}')
            print(f'TRAIN: Avg Reward: {np.array(self.logs[f"{round_no + 1}"]["train"]["rewards"]).mean():.5f},  Avg Running Reward: {np.array(self.logs[f"{round_no + 1}"]["train"]["running_rewards"]).mean():.5f}')
            print(f'EVAL: Avg Reward: {np.array(self.logs[f"{round_no + 1}"]["eval"]["rewards"]).mean():.5f}')

        
        with open(args.folder_name + "/train.txt", 'w') as convert_file:
             convert_file.write(json.dumps(self.logs))
                
        torch.save(self.main_agent.dqn.state_dict(), f'{args.folder_name}/model.pt')
        

if __name__ == '__main__':
    
    args = ARGS()
    set_seed(args.seed)

    device = torch.device("cuda:0")
    dtype = torch.float

    os.makedirs('runs/', exist_ok=True)
    os.makedirs(f'runs/{args.mode}/', exist_ok=True)
    os.makedirs(args.folder_name, exist_ok=True)

    # save the hyperparameters in a file
    f = open(f'{args.folder_name}/args.txt', 'w')
    for i in args.__dict__:
        f.write(f'{i}: {args.__dict__[i]}\n')
    f.close()
    fl = FederatedLearning(args)
    fl.run()