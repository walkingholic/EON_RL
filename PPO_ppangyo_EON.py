import math
import os
import matplotlib.pyplot as plt
import datetime
import random
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import torch.multiprocessing as mp
from simulation_v2 import Simulation

import copy

################################## set device ##################################

print("============================================================================================")

# set device to cpu or cuda
device = torch.device('cpu')

# if (torch.cuda.is_available()):
#     device = torch.device('cuda:0')
#     torch.cuda.empty_cache()
#     print("Device set to : " + str(torch.cuda.get_device_name(device)))
# else:
#     print("Device set to : cpu")
#
# print("============================================================================================")
#




# Hyperparameters
# learning_rate = 0.0001
# gamma = 0.99
# lmbda = 0.95
# eps_clip = 0.1
# K_epoch = 32



class ActorCritic(nn.Module):    #  for  v2 && v2_1
    def __init__(self, statesize, hiddensize, action_dim):
        super(ActorCritic, self).__init__()

        # self.fc1 = nn.Linear(346, 512)
        # self.fc1 = nn.Linear(96, 512)
        # self.fc1 = nn.Linear(68, 512)
        # self.fc1 = nn.Linear(262, 512)
        # self.fc1 = nn.Linear(237, 128)
        # self.fc1 = nn.Linear(147, 128)
        # self.fc1 = nn.Linear(177, hiddensize)
        self.fc1 = nn.Linear(statesize, hiddensize)
        # self.fc1 = nn.Linear(162, 512)
        # self.fc1 = nn.Linear(562, 512)
        self.fc2 = nn.Linear(hiddensize, hiddensize)
        self.fc3 = nn.Linear(hiddensize, hiddensize)
        self.fc4 = nn.Linear(hiddensize, hiddensize)
        self.fc5 = nn.Linear(hiddensize, hiddensize)
        self.fc6 = nn.Linear(hiddensize, hiddensize)

        self.fc7 = nn.Linear(hiddensize, hiddensize)
        self.fc8 = nn.Linear(hiddensize, hiddensize)
        self.fc_pi = nn.Linear(hiddensize, action_dim)
        self.fc_v = nn.Linear(hiddensize, 1)

    def pi(self, state, req_info, softmax_dim=0):

        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))

        # x = F.elu(self.fc1(state),1)
        # x = F.elu(self.fc2(x), 1)
        # x = F.elu(self.fc3(x), 1)
        # x = F.elu(self.fc4(x), 1)
        # x = F.elu(self.fc5(x), 1)
        # x = F.elu(self.fc6(x), 1)

        x = self.fc_pi(x)
        # print(x, x.size())
        prob = F.softmax(x, dim=1)
        # print(prob, prob.size())
        return prob

    def v(self, state, req_info):

        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))

        # x = F.elu(self.fc1(state), 1)
        # x = F.elu(self.fc2(x), 1)
        # x = F.elu(self.fc3(x), 1)
        # x = F.elu(self.fc4(x), 1)
        # x = F.elu(self.fc5(x), 1)
        # x = F.elu(self.fc6(x), 1)

        v = self.fc_v(x)
        return v

# class ActorCritic(nn.Module):
#     def __init__(self, state_dim, action_dim):
#         super(ActorCritic, self).__init__()
#
#         self.layer1 = torch.nn.Sequential(
#             torch.nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
#             torch.nn.ReLU(),
#             torch.nn.MaxPool2d(kernel_size=2, stride=2))
#
#         self.layer2 = torch.nn.Sequential(
#             torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
#             torch.nn.ReLU(),
#             torch.nn.MaxPool2d(kernel_size=2, stride=2))
#
#         self.layer3 = torch.nn.Sequential(
#             torch.nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
#             torch.nn.ReLU(),
#             torch.nn.MaxPool2d(kernel_size=2, stride=2))
#
#         self.layer4 = torch.nn.Sequential(
#             torch.nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
#             torch.nn.ReLU(),
#             torch.nn.MaxPool2d(kernel_size=2, stride=2))
#
#         self.fc1 = nn.Linear(1239, 512)
#         self.fc2 = nn.Linear(512, 256)
#         self.fc3 = nn.Linear(256, 256)
#         self.fc4 = nn.Linear(256, 256)
#         self.fc_pi = nn.Linear(256, action_dim)
#         self.fc_v = nn.Linear(256, 1)
#
#     def pi(self, state, req_info, softmax_dim=0):
#         out = self.layer1(state)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = self.layer4(out)
#         out = out.view(out.size(0), -1)
#         in_data = torch.cat([out, req_info], dim=1)
#         x = F.relu(self.fc1(in_data))
#         x = F.relu(self.fc2(x))
#         x = F.relu(self.fc3(x))
#         x = F.relu(self.fc4(x))
#         x = self.fc_pi(x)
#         # print(x, x.size())
#         prob = F.softmax(x, dim=1)
#         # print(prob, prob.size())
#         return prob
#
#     def v(self, state, req_info):
#         out = self.layer1(state)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = self.layer4(out)
#         out = out.view(out.size(0), -1)
#         in_data = torch.cat([out, req_info], dim=1)
#         x = F.relu(self.fc1(in_data))
#         x = F.relu(self.fc2(x))
#         x = F.relu(self.fc3(x))
#         x = F.relu(self.fc4(x))
#         v = self.fc_v(x)
#         return v


# class ActorCritic(nn.Module):
#     def __init__(self, state_dim, action_dim):
#         super(ActorCritic, self).__init__()
#
#         self.layer1 = torch.nn.Sequential(
#             torch.nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
#             torch.nn.ReLU(),
#             torch.nn.MaxPool2d(kernel_size=2, stride=2))
#         # 25*25*16
#         self.layer2 = torch.nn.Sequential(
#             torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
#             torch.nn.ReLU(),
#             torch.nn.MaxPool2d(kernel_size=2, stride=2))
#         # 12*12*32
#         self.layer3 = torch.nn.Sequential(
#             torch.nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
#             torch.nn.ReLU(),
#             torch.nn.MaxPool2d(kernel_size=2, stride=2))
#         # 6*6*32
#
#         self.layer4 = torch.nn.Sequential(
#             torch.nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
#             torch.nn.ReLU(),
#             torch.nn.MaxPool2d(kernel_size=2, stride=2))
#
#         # self.fc1 = nn.Linear(350, 512)
#         self.fc1 = nn.Linear(1214, 512)
#         self.fc2 = nn.Linear(512, 256)
#         self.fc3 = nn.Linear(256, 256)
#         self.fc4 = nn.Linear(256, 256)
#         self.fc_pi = nn.Linear(256, action_dim)
#         self.fc_v = nn.Linear(256, 1)
#
#     def pi(self, state, req_info, softmax_dim=0):
#         out = self.layer1(state)
#         # print(out.size())
#         out = self.layer2(out)
#         # print(out.size())
#         out = self.layer3(out)
#         # print(out.size())
#         # out = self.layer4(out)
#         out = out.view(out.size(0), -1)
#         in_data = torch.cat([out, req_info], dim=1)
#         # print(out.size())
#         # print(in_data.size())
#
#         x = F.relu(self.fc1(in_data))
#         x = F.relu(self.fc2(x))
#         x = F.relu(self.fc3(x))
#         x = F.relu(self.fc4(x))
#         x = self.fc_pi(x)
#         prob = F.softmax(x, dim=1)
#         # print(prob, prob.size())
#         return prob
#
#     def v(self, state, req_info):
#         out = self.layer1(state)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         # out = self.layer4(out)
#         out = out.view(out.size(0), -1)
#         in_data = torch.cat([out, req_info], dim=1)
#         x = F.relu(self.fc1(in_data))
#         x = F.relu(self.fc2(x))
#         x = F.relu(self.fc3(x))
#         x = F.relu(self.fc4(x))
#         v = self.fc_v(x)
#         return v

# class ActorCritic(nn.Module):
#     def __init__(self, state_dim, action_dim):
#         super(ActorCritic, self).__init__()
#
#         # self.fc1 = nn.Linear(562, 512)
#         # self.fc1 = nn.Linear(96, 512)
#         # self.fc1 = nn.Linear(86, 512)
#         # self.fc1 = nn.Linear(68, 512)
#         self.fc1 = nn.Linear(262, 512)
#         self.fc2 = nn.Linear(512, 256)
#         self.fc3 = nn.Linear(256, 256)
#         self.fc4 = nn.Linear(256, 256)
#         self.fc_pi = nn.Linear(256, action_dim)
#         self.fc_v = nn.Linear(256, 1)
#
#     def pi(self, state, req_info, softmax_dim=0):
#
#         x = F.relu(self.fc1(state))
#         x = F.relu(self.fc2(x))
#         x = F.relu(self.fc3(x))
#         x = F.relu(self.fc4(x))
#         x = self.fc_pi(x)
#         # print(x, x.size())
#         prob = F.softmax(x, dim=1)
#         # print(prob, prob.size())
#         return prob
#
#     def v(self, state, req_info):
#
#         x = F.relu(self.fc1(state))
#         x = F.relu(self.fc2(x))
#         x = F.relu(self.fc3(x))
#         x = F.relu(self.fc4(x))
#         v = self.fc_v(x)
#         return v









# class ActorCritic(nn.Module):
#     def __init__(self, state_dim, action_dim):
#         super(ActorCritic, self).__init__()
#
#         def conv_bn(inp, oup, stride):
#             return nn.Sequential(
#                 nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
#                 nn.BatchNorm2d(oup),
#                 nn.ReLU(inplace=True)
#             )
#
#         def conv_dw(inp, oup, stride):
#             return nn.Sequential(
#                 # dw
#                 nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
#                 nn.BatchNorm2d(inp),
#                 nn.ReLU(inplace=True),
#
#                 # pw
#                 nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
#                 nn.BatchNorm2d(oup),
#                 nn.ReLU(inplace=True),
#             )
#
#         self.model = nn.Sequential(
#             conv_bn(1, 32, 2),
#             conv_dw(32, 64, 1),
#             conv_dw(64, 128, 2),
#             conv_dw(128, 128, 1),
#             conv_dw(128, 256, 2),
#             conv_dw(256, 256, 1),
#             conv_dw(256, 512, 2),
#             conv_dw(512, 512, 1),
#             # conv_dw(512, 512, 1),
#             # conv_dw(512, 512, 1),
#             # conv_dw(512, 512, 1),
#             conv_dw(512, 512, 1),
#             conv_dw(512, 1024, 2),
#             conv_dw(1024, 1024, 1),
#             nn.AdaptiveAvgPool2d(1)
#         )
#         # self.fc1 = nn.Linear(350, 512)
#         # self.fc1 = nn.Linear(1214, 1024)
#         # self.fc2 = nn.Linear(1024, 512)
#         # self.fc3 = nn.Linear(512, 256)
#         # self.fc4 = nn.Linear(256, 128)
#         self.fc_pi = nn.Linear(1086, action_dim)
#         self.fc_v = nn.Linear(1086, 1)
#
#     def pi(self, state, req_info, softmax_dim=0):
#         out = self.model(state)
#         # print(out.size())
#
#         out = out.view(out.size(0), -1)
#         in_data = torch.cat([out, req_info], dim=1)
#         # print(out.size())
#         # print(in_data.size())
#
#         # x = F.relu(self.fc1(in_data))
#         # x = F.relu(self.fc2(x))
#         # x = F.relu(self.fc3(x))
#         # x = F.relu(self.fc4(x))
#         x = self.fc_pi(in_data)
#         prob = F.softmax(x, dim=1)
#         # print(prob, prob.size())
#         return prob
#
#     def v(self, state, req_info):
#         out = self.model(state)
#
#         out = out.view(out.size(0), -1)
#         in_data = torch.cat([out, req_info], dim=1)
#
#         v = self.fc_v(in_data)
#         return v
#
#









class PPO():
    def __init__(self, statesize, hiddensize, state_dim, action_dim, K_epoch, gamma, alpha, lmbda, eps_clip, learning_rate):
        ################################## set device ##################################

        print("============================================================================================")

        # set device to cpu or cuda
        self.device = torch.device('cpu')

        if (torch.cuda.is_available()):
            self.device = torch.device('cuda:0')
            torch.cuda.empty_cache()
            print("Device set to : " + str(torch.cuda.get_device_name(self.device)))
        else:
            print("Device set to : cpu")

        print("============================================================================================")



        self.data = []

        self.K_epoch = K_epoch
        self.gamma = gamma
        self.lmbda = lmbda
        self.eps_clip = eps_clip
        self.alpha = alpha


        self.lnet = ActorCritic(statesize, hiddensize,action_dim).to(self.device)

        self.optimizer = optim.Adam(self.lnet.parameters(), lr=learning_rate)


    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):
        s_lst, req_ist, a_lst, r_lst, s_prime_lst, req_prime_ist, prob_a_lst, done_lst = [], [], [], [], [], [], [], []
        for transition in self.data:
            s, req, a, r, s_prime, req_prime, prob_a, done = transition

            s_lst.append(s)
            req_ist.append(req)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            req_prime_ist.append(req_prime)
            prob_a_lst.append([prob_a])
            done_mask = 0 if done else 1
            done_lst.append([done_mask])

        s, req, a, r, s_prime, req_prime, done_mask, prob_a = torch.tensor(s_lst, dtype=torch.float).to(self.device).squeeze(1), \
                                               torch.tensor(req_ist, dtype=torch.float).to(self.device).squeeze(1), \
                                               torch.tensor(a_lst).to(self.device), \
                                              torch.tensor(r_lst, dtype=torch.float).to(self.device), \
                                              torch.tensor(s_prime_lst, dtype=torch.float).to(self.device).squeeze(1), \
                                              torch.tensor(req_prime_ist, dtype=torch.float).to(self.device).squeeze(1), \
                                              torch.tensor(done_lst, dtype=torch.float).to(self.device), \
                                              torch.tensor(prob_a_lst).to(self.device)
        self.data = []

        return s, req, a, r, s_prime, req_prime, done_mask, prob_a

    def train_net(self):
        s, req, a, r, s_prime, req_prime, done_mask, prob_a = self.make_batch()
        # print('\n update')

        for i in range(self.K_epoch):
            # print(i, end='')
            td_target = r + self.gamma * self.lnet.v(s_prime, req_prime) * done_mask
            delta = td_target - self.lnet.v(s, req)
            delta = delta.cpu().detach().numpy()

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = self.gamma * self.lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float).to(self.device)

            pi = self.lnet.pi(s, req, softmax_dim=1)
            # print('pi size: ', pi.size())
            pi_a = pi.gather(1, a) #dim — dimension along to collect values, index — tensor with indices of values to collect
            #pi_a: action 확률, 과거 샘플로 현재 폴리시의 액션확률,
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))

            dist = Categorical(pi)
            dist_entropy = dist.entropy()


            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage
            # loss = -torch.min(surr1, surr2) + 0.5*F.smooth_l1_loss(self.lnet.v(s, req), td_target.detach()) - 0.01 * dist_entropy
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.lnet.v(s, req), td_target.detach()) - self.alpha * dist_entropy
            # loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.lnet.v(s, req), td_target.detach())

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

            del advantage
            del advantage_lst
            torch.cuda.empty_cache()
        # print('')

class PPO_MULTI(mp.Process):
    def __init__(self, id, state_dim, action_dim,  learning_rate, gamma, lmbda, eps_clip, K_epoch, res_queue, gnet=None):
        super(PPO_MULTI, self).__init__()
        self.data = []
        self.queue = res_queue

        self.K_epoch = K_epoch
        self.gamma = gamma
        self.lmbda = lmbda
        self.eps_clip = eps_clip


        self.lnet = ActorCritic(state_dim, action_dim).to(device)
        self.lnet.load_state_dict(gnet.state_dict())
        self.gnet = gnet

        self.optimizer = optim.Adam(self.gnet.parameters(), lr=learning_rate)
        # self.optimizer = optim



    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):
        s_lst, req_ist, a_lst, r_lst, s_prime_lst, req_prime_ist, prob_a_lst, done_lst = [], [], [], [], [], [], [], []
        for transition in self.data:
            s, req, a, r, s_prime, req_prime, prob_a, done = transition

            s_lst.append(s)
            req_ist.append(req)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            req_prime_ist.append(req_prime)
            prob_a_lst.append([prob_a])
            done_mask = 0 if done else 1
            done_lst.append([done_mask])

        s, req, a, r, s_prime, req_prime, done_mask, prob_a = torch.tensor(s_lst, dtype=torch.float).to(device).squeeze(1), \
                                               torch.tensor(req_ist, dtype=torch.float).to(device).squeeze(1), \
                                               torch.tensor(a_lst).to(device), \
                                              torch.tensor(r_lst, dtype=torch.float).to(device), \
                                              torch.tensor(s_prime_lst, dtype=torch.float).to(device).squeeze(1), \
                                              torch.tensor(req_prime_ist, dtype=torch.float).to(device).squeeze(1), \
                                              torch.tensor(done_lst, dtype=torch.float).to(device), \
                                              torch.tensor(prob_a_lst).to(device)
        self.data = []

        return s, req, a, r, s_prime, req_prime, done_mask, prob_a

    def train_net(self):
        s, req, a, r, s_prime, req_prime, done_mask, prob_a = self.make_batch()
        # print('\n update')

        for i in range(self.K_epoch):
            # print(i, end='')
            td_target = r + self.gamma * self.lnet.v(s_prime, req_prime) * done_mask
            delta = td_target - self.lnet.v(s, req)
            delta = delta.cpu().detach().numpy()

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = self.gamma * self.lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float).to(device)

            pi = self.lnet.pi(s, req, softmax_dim=1)
            # print('pi size: ', pi.size())
            pi_a = pi.gather(1, a) #dim — dimension along to collect values, index — tensor with indices of values to collect
            #pi_a: action 확률, 과거 샘플로 현재 폴리시의 액션확률,
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))

            dist = Categorical(pi)
            dist_entropy = dist.entropy()


            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage
            # loss = -torch.min(surr1, surr2) + 0.5*F.smooth_l1_loss(self.lnet.v(s, req), td_target.detach()) - 0.01 * dist_entropy
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.lnet.v(s, req), td_target.detach())

            self.optimizer.zero_grad()
            loss.mean().backward()
            for lp, gp in zip(self.lnet.parameters(), self.gnet.parameters()):
                gp._grad = lp.grad
            self.optimizer.step()
            self.lnet.load_state_dict(self.gnet.state_dict())

        # print('')

    def run(self):
        num_EP = 1000000
        timestep_max = 1000
        update_interval=100
        # update_timestep = timestep_max * 10
        num_of_req = 60000
        num_of_warmup = 10000
        ar = 4
        dif = 1
        avg_holding = 50
        range_band_s = 1
        range_band_e = 10
        TotalNumofSlots = 100
        num_kpath = 5
        rt_name = 'KSP'
        sa_name = 'DRL'
        i = 0


        env = Simulation(num_of_req, num_kpath, num_of_warmup, (ar + (i) * dif), avg_holding, range_band_s,
                         range_band_e, TotalNumofSlots,
                         rt_name, sa_name)

        env.env_init()
        timestep = 0
        log_entropy = []
        log_rwd = []
        log_succ_req = []
        log_blk_req = []

        for e in range(num_EP):
            done = False
            state, req_info, req = env.env_reset()

            ept_score = 0
            succ_req, blk_req = 0, 0
            entropy = 0
            actlist = []
            while done != 1:
                for t in range(update_interval):
                    prob = self.lnet.pi(torch.from_numpy(state).float().to(device),
                                    torch.from_numpy(req_info).float().to(device))
                    m = Categorical(prob)
                    action = m.sample().item()
                    actlist.append(action)
                    entropy += m.entropy().item()
                    next_state, next_req_info, next_req, reward, done = env.env_step(req, action)
                    ept_score += reward

                    if reward >= 1:
                        succ_req += 1
                    else:
                        blk_req += 1

                    self.put_data(
                        (state, req_info, action, reward, next_state, next_req_info, prob[0, action].item(), done))

                    state = next_state
                    req_info = next_req_info
                    req = next_req

                    timestep += 1
                    if timestep % timestep_max == 0:
                        done = 1
                        log_rwd.append(ept_score)
                        log_succ_req.append(succ_req)
                        log_blk_req.append(blk_req)
                        log_entropy.append(entropy / timestep_max)

                        self.queue.put((ept_score, succ_req, blk_req, entropy / timestep_max))

                        print('Epi: ', e, '   Score: ', ept_score, '   BBP: ', blk_req / timestep_max,
                              '  Epi avg Entropy: ',
                              entropy / timestep_max)
                        print(actlist)

                self.train_net()
        self.queue.put(None)
#
# class SharedAdam(torch.optim.Adam):
#     def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8,
#                  weight_decay=0):
#         super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
#         # State initialization
#         for group in self.param_groups:
#             for p in group['params']:
#                 state = self.state[p]
#                 state['step'] = 0
#                 state['exp_avg'] = torch.zeros_like(p.data)
#                 state['exp_avg_sq'] = torch.zeros_like(p.data)
#
#                 # share in memory
#                 state['exp_avg'].share_memory_()
#                 state['exp_avg_sq'].share_memory_()