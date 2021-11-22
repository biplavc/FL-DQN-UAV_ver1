# import gym
import sys

import numpy as np
import pickle
from collections import deque

# from joblib import Parallel, delayed
import multiprocessing as mp

import time
import datetime # from datetime 

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

class NpEncoder(json.JSONEncoder): ## https://www.javaprogramto.com/2019/11/python-typeerror-integer-json-not-serializable.html
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


        
## self, n_users, coverage, name, folder_name, packet_update_loss, packet_sample_loss, periodicity
class UAV_ARGS():
    def __init__(self, n_users, coverage, name, folder_name, packet_update_loss, packet_sample_loss, periodicity):
        self.n_users = n_users
        self.coverage = coverage
        self.name = name
        self.folder_name = folder_name
        self.packet_update_loss = packet_update_loss
        self.packet_sample_loss = packet_sample_loss
        self.periodicity = periodicity
        
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# each agent has a name, the main federator has name main while the other agents has the number as the name

class FederatedLearning:
    
    def __init__(self, args, UAV_args):
        self.args = args
        self.UAV_args = UAV_args
        self.main_agent = UavAgent(args, name  ="main", UAV_args = UAV_args)
        # self, args, name = "", UAV_args = UAV_args
        
        self.create_clients()
        
        print(f"args.folder_name = {args.folder_name}")
        
        self.logs = {}
        
    def create_clients(self): 
        self.clients = {}
        self.client_names = []
        self.updated_clients = {}
        for i in range(self.args.number_of_samples):
            self.client_names.append(f"client_{i}")
            self.clients[f"client_{i}"] = UavAgent(self.args, i, self.UAV_args)
            self.updated_clients[f"client_{i}"] = 0
            
            
    def update_clients(self):
        with torch.no_grad():
            def update(client_layer, main_layer):
                client_layer.weight.data = main_layer.weight.data.clone()
                client_layer.bias.data = main_layer.bias.data.clone()
                
            for i in range(self.args.number_of_samples):
                update(self.clients[self.client_names[i]].dqn.h1, self.main_agent.dqn.h1)
                update(self.clients[self.client_names[i]].dqn.h2, self.main_agent.dqn.h2)
                update(self.clients[self.client_names[i]].dqn.h3, self.main_agent.dqn.h3)
                update(self.clients[self.client_names[i]].dqn.h4, self.main_agent.dqn.h4)
                update(self.clients[self.client_names[i]].dqn.h5, self.main_agent.dqn.h5)
                update(self.clients[self.client_names[i]].dqn.h6, self.main_agent.dqn.h6)


                del self.clients[self.client_names[i]].buffer ## clear buffer
                self.clients[self.client_names[i]].buffer = ReplayMemory(1000000 // 4)

                del self.clients[self.client_names[i]].target_dqn
                self.clients[self.client_names[i]].target_dqn = DQN(self.main_agent.num_actions, self.main_agent.state_size)
                self.clients[self.client_names[i]].target_dqn.load_state_dict(self.clients[self.client_names[i]].dqn.state_dict())

                if self.args.use_gpu:
                    self.clients[self.client_names[i]].target_dqn.cuda()



    def update_main_agent(self): #, round_no):
        # meaning
        h1_mean_weight = torch.zeros(size=self.main_agent.dqn.h1.weight.shape).to(device)
        h1_mean_bias = torch.zeros(size=self.main_agent.dqn.h1.bias.shape).to(device)

        h2_mean_weight = torch.zeros(size=self.main_agent.dqn.h2.weight.shape).to(device)
        h2_mean_bias = torch.zeros(size=self.main_agent.dqn.h2.bias.shape).to(device)

        h3_mean_weight = torch.zeros(size=self.main_agent.dqn.h3.weight.shape).to(device)
        h3_mean_bias = torch.zeros(size=self.main_agent.dqn.h3.bias.shape).to(device)

        h4_mean_weight = torch.zeros(size=self.main_agent.dqn.h4.weight.shape).to(device)
        h4_mean_bias = torch.zeros(size=self.main_agent.dqn.h4.bias.shape).to(device)

        h5_mean_weight = torch.zeros(size=self.main_agent.dqn.h5.weight.shape).to(device)
        h5_mean_bias = torch.zeros(size=self.main_agent.dqn.h5.bias.shape).to(device)
        
        h6_mean_weight = torch.zeros(size=self.main_agent.dqn.h6.weight.shape).to(device)
        h6_mean_bias = torch.zeros(size=self.main_agent.dqn.h6.bias.shape).to(device)
        
        number_of_samples = self.args.number_of_samples
        with torch.no_grad():

            for i in range(number_of_samples):
                h1_mean_weight += self.clients[self.client_names[i]].dqn.h1.weight.clone()
                h1_mean_bias += self.clients[self.client_names[i]].dqn.h1.bias.clone()

                h2_mean_weight += self.clients[self.client_names[i]].dqn.h2.weight.clone()
                h2_mean_bias += self.clients[self.client_names[i]].dqn.h2.bias.clone()

                h3_mean_weight += self.clients[self.client_names[i]].dqn.h3.weight.clone()
                h3_mean_bias += self.clients[self.client_names[i]].dqn.h3.bias.clone()

                h4_mean_weight += self.clients[self.client_names[i]].dqn.h4.weight.clone()
                h4_mean_bias += self.clients[self.client_names[i]].dqn.h4.bias.clone()

                h5_mean_weight += self.clients[self.client_names[i]].dqn.h5.weight.clone()
                h5_mean_bias += self.clients[self.client_names[i]].dqn.h5.bias.clone()

                h6_mean_weight += self.clients[self.client_names[i]].dqn.h6.weight.clone()
                h6_mean_bias += self.clients[self.client_names[i]].dqn.h6.bias.clone()

                
            h1_mean_weight = h1_mean_weight / number_of_samples
            h1_mean_bias = h1_mean_bias / number_of_samples

            h2_mean_weight = h2_mean_weight / number_of_samples
            h2_mean_bias = h2_mean_bias / number_of_samples

            h3_mean_weight = h3_mean_weight / number_of_samples
            h3_mean_bias = h3_mean_bias / number_of_samples

            h4_mean_weight = h4_mean_weight / number_of_samples
            h4_mean_bias = h4_mean_bias / number_of_samples

            h5_mean_weight = h5_mean_weight / number_of_samples
            h5_mean_bias = h5_mean_bias / number_of_samples

            h6_mean_weight = h6_mean_weight / number_of_samples
            h6_mean_bias = h6_mean_bias / number_of_samples
  

            with torch.no_grad():
                def update(main_layer, averaged_layer_weight, averaged_layer_bias):
                    main_layer.weight.data = averaged_layer_weight.data.clone()
                    main_layer.bias.data = averaged_layer_bias.data.clone()
                
                update(self.main_agent.dqn.h1, h1_mean_weight, h1_mean_bias)
                update(self.main_agent.dqn.h2, h2_mean_weight, h2_mean_bias)
                update(self.main_agent.dqn.h3, h3_mean_weight, h3_mean_bias)
                update(self.main_agent.dqn.h4, h4_mean_weight, h4_mean_bias)
                update(self.main_agent.dqn.h5, h5_mean_weight, h5_mean_bias)
                update(self.main_agent.dqn.h6, h6_mean_weight, h6_mean_bias)
                

        
    def step(self, idx_users, round_no):
        
        self.update_clients()
        
        for user in idx_users:
            print(f"Client {user}", flush = True)

            rewards, running_rewards = self.clients[self.client_names[user]].train(
                replay_buffer_fill_len = self.args.replay_buffer_fill_len, 
                batch_size = self.args.batch_size, 
                local_episodes = self.args.local_episodes,
                max_epsilon_steps = self.args.max_epsilon_steps,
                epsilon_start = self.args.epsilon_start - 0.03*(round_no - 1),
                epsilon_final = self.args.epsilon_final,
                sync_target_net_freq = self.args.sync_target_net_freq)

            print(f'LOCAL TRAIN: Avg Reward: {np.array(rewards).mean():.2f},  Avg Running Reward: {np.array(running_rewards).mean():.2f}', flush = True)
            

            self.logs[f"{round_no}"]["train"]["rewards"].append(rewards)
            self.logs[f"{round_no}"]["train"]["running_rewards"].append(running_rewards)

        # self.update_main_agent(round_no)
        self.update_main_agent() ## biplav

        self.logs[f"{round_no}"]["eval"]["rewards"] = self.main_agent.play(self.args.eval_episodes) # biplav


    def run(self):
        print(f"self.args.mode = {self.args.mode}")

        # m = max(int(self.args.fraction * self.args.number_of_samples), 1) 
        for round_no in range(self.args.rounds):
            
            self.logs[f"{round_no + 1}"] = {"train": {
                                            "rewards": [],
                                            "running_rewards": []
                                        },
                                        "eval": {
                                            "rewards": None
                                        }
                                       }
            # idxs_users = np.random.choice(range(self.args.number_of_samples), m, replace=False) # false ## biplav

            idxs_users = range(self.args.number_of_samples) ## biplav

            for user in idxs_users:
                self.updated_clients[f"client_{user}"] = round_no + 1

            self.step(idxs_users, round_no + 1)
            print(f'{round_no + 1}/{self.args.rounds}', flush = True)
            # print(f'TRAIN: Avg Reward: {np.array(self.logs[f"{round_no + 1}"]["train"]["rewards"]).mean():.2f},  Avg Running Reward: {np.array(self.logs[f"{round_no + 1}"]["train"]["running_rewards"]).mean():.2f}', flush = True)
            print(f'EVAL: Avg Reward: {np.array(self.logs[f"{round_no + 1}"]["eval"]["rewards"]).mean():.2f}', flush = True)


        with open(self.args.folder_name + "/train.txt", 'w') as convert_file:
            convert_file.write(json.dumps(self.logs, cls=NpEncoder)) ## biplav
        
        final_round = str(round_no)
                
        if self.args.mode == "rl": ## rl
            print(f"pickling the RL results")
            pickle.dump(self.logs[final_round]["eval"]["rewards"], open(self.args.folder_name + "rl_returns.pickle", "wb"))
        if self.args.mode == "fl": ## rl
            print(f"pickling the FL results")
            pickle.dump(self.logs[final_round]["eval"]["rewards"], open(self.args.folder_name + "fl_returns.pickle", "wb"))

        torch.save(self.main_agent.dqn.state_dict(), f'{self.args.folder_name}/model.pt')
        

def generate_images(logs): ## logs is a dict
    pass
    
    
class ARGS():
    def __init__(self, mode, current_time):
        self.env_name = UAV_network(3, {0:[1,2,3]}, "UAV_network", "None", {1:0,2:0,3:0}, {1:0,2:0,3:0}, {1:2,2:1,3:1})

        self.render = False
        self.episodes = 50
        self.batch_size = 32
        self.epsilon_start = 1.0
        self.epsilon_final=0.02
        self.seed = 1773
        self.eval_episodes = 100
        
        self.use_gpu = torch.cuda.is_available()
        
        self.mode = ["rl", "fl"][mode] ## biplav 0 for RL and 1 for FL
        
        print(f"starting in {self.mode} mode", flush = True)
        
        self.number_of_samples = 5 if self.mode != "rl" else 1
        self.fraction = 1 if self.mode != "rl" else 1 ## biplav
        self.local_episodes = 50 if self.mode != "rl" else 10
        self.rounds = 2 if self.mode != "rl" else 2


        self.max_epsilon_steps = self.local_episodes*200 ## 10_000
        self.sync_target_net_freq = self.max_epsilon_steps // 10 ## 1_000
        # now = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        now = current_time
        os.makedirs('runs/', exist_ok=True)
        self.folder_name = f"runs/" + now + "/" + self.mode + "/"
        os.makedirs(f'{self.folder_name}/', exist_ok=True)
        self.replay_buffer_fill_len = 1_000
        
def generate_images(logs, rounds):
    pass



def start_execution(mode, now):
    args = ARGS(mode = mode, current_time = now)
    set_seed(args.seed)

    ## args.folder_name = f"runs/" + now + "/" + self.mode + "/"
    os.makedirs(args.folder_name, exist_ok=True)

    # save the hyperparameters in a file
    f = open(f'{args.folder_name}/args.txt', 'w')
    for i in args.__dict__:
        f.write(f'{i}: {args.__dict__[i]}\n')
    f.close()
    n_users = 3
    coverage = {0:[1,2,3]}
    name = "UAV_network"
    folder_name = 'models/' +  now ## biplav not used
    packet_update_loss = {1:0,2:0,3:0}
    packet_sample_loss = {1:0,2:0,3:0}
    periodicity = {1:2,2:1,3:1}
    UAV_args = UAV_ARGS(n_users, coverage, name, folder_name, packet_update_loss, packet_sample_loss, periodicity)
    fl = FederatedLearning(args, UAV_args)
  
    fl.run()
    generate_images(fl.logs, fl.args.rounds)


if __name__ == '__main__':
    # print(f"sys.argv = {sys.argv}", flush = True)   
    # mode = int(sys.argv[1])
    # device = torch.device("cuda:0") ## biplav
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  ## https://stackoverflow.com/questions/53266350/how-to-tell-pytorch-to-not-use-the-gpu
    dtype = torch.float
    now = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    modes = [0, 1] ## 0 for RL and 1 for FL
    now = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    pool = mp.Pool(mp.cpu_count())
    pool.starmap(start_execution, [(mode, now) for mode in modes])
    pool.close()
