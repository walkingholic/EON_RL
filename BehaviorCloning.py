import numpy as np
import random
import matplotlib.pyplot as plt
from simulation_v2 import Simulation
import copy
import csv
import datetime
import request
import pandas as pd
import os, time
import torch.optim as optim
import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset # 텐서데이터셋



n_pro = 1
env_version=2
N_S = 100
num_kpath = 5
num_Subblock = 1
actionsize = num_kpath * num_Subblock
statesize = 94
hiddensize = 128
alpha = 0.01

learning_rate = 0.00001
gamma = 0.90
lmbda = 0.95

eps_clip = 0.1
K_epoch = 16
num_EP = 5000
timestep_max = 10000 #ep 샘플수
update_interval = 10000 # 샘플 수집
TotalNumofSlots = 100
num_of_req = 60000
num_of_warmup = 10000
ar = 10
dif = 1
avg_holding = 20
range_band_s = 1
range_band_e = 10

rt_name = 'KSP'
sa_name = 'DRL'
i = 0

# ################################## set device ##################################

# print("============================================================================================")
#
# # set device to cpu or cuda
# device = torch.device('cpu')
#
# if (torch.cuda.is_available() and n_pro==1):
#     device = torch.device('cuda:0')
#     torch.cuda.empty_cache()
#     print("Device set to : " + str(torch.cuda.get_device_name(device)))
# else:
#     device = torch.device('cpu')
#     print("Device set to : cpu")
#
# print("============================================================================================")


basepath = 'result'

def createFolder(directory):
    try:
        if not os.path.isdir(directory):
            os.makedirs(directory)
    except OSError:
        print('error')




class ActorCritic(nn.Module):    #  for  v2 && v2_1
    def __init__(self, state_dim, action_dim):
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

    def pi(self, state, softmax_dim=0):

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

    def v(self, state):

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


def BehaviorClone():
    print("============================================================================================")
    # set device to cpu or cuda
    # device = torch.device('cpu')
    if (torch.cuda.is_available()):
        device = torch.device('cuda:0')
        torch.cuda.empty_cache()
        print("Device set to : " + str(torch.cuda.get_device_name(device)))
    else:
        print("Device set to : cpu")
        device = torch.device('cpu')
    print("============================================================================================")

    now_start = datetime.datetime.now()
    resultdir = '{0:02}-{1:02} {2:02}-{3:02}_PPOsg_Env={14}_p={5}_EP={15}k_K={6}_SB={7}_eplen={8}k_uint={9}k_lr={10}_gm={11}_alp={12}_hid={13}'.format(
        now_start.month, now_start.day, now_start.hour, now_start.minute, now_start.second, n_pro, num_kpath,
        num_Subblock, timestep_max / 1000, update_interval / 1000,
        learning_rate, gamma, alpha, hiddensize, env_version, num_EP / 1000)

    dirpath = os.path.join(basepath, resultdir)
    createFolder(dirpath)
    print(dirpath)

    model = ActorCritic(N_S, actionsize).to(device)
    # criterion = nn.CrossEntropyLoss().to(device)
    # criterion = nn.MSELoss().to(device)
    criterion = nn.NLLLoss().to(device)

    # optimizer = optim.Adam(model.parameters(), lr=0.0001)
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    env = Simulation(num_of_req, num_kpath, actionsize, num_of_warmup, ar, avg_holding, range_band_s,
                     range_band_e, TotalNumofSlots,
                     rt_name, sa_name, dirpath, env_version)
    # env.env_init()

    num_epoch = 6000
    num_samples = 5000
    batsize = 1000
    testsamplesize = 500
    statelist, reqinfolist, actOneHot, actlist = env.env_BC(num_samples, statesize, actionsize)
    slist = torch.tensor(statelist[:-testsamplesize], dtype=torch.float).squeeze(1).to(device)
    # aOnehotlist = torch.tensor(actOneHot[:-2000], dtype=torch.float).squeeze(1).to(device)
    alist = torch.tensor(actlist[:-testsamplesize]).squeeze(1).to(device)

    print(alist.shape)

    dataset = TensorDataset(slist, alist)
    dataloader = DataLoader(dataset, batch_size=batsize, shuffle=True)

    for epoch in range(num_epoch):

        for batch_idx, samples in enumerate(dataloader):
            x_train, y_train = samples

            pi = model.pi(x_train)
            # print(pi.sum(dim=1))
            # print(torch.log(pi))
            # print(pi)
            loss = criterion(torch.log(pi+1e-9) , y_train)

            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()

        if epoch % 100 == 0:
            print('Epoch {:4d}/{} Cost: {:.4f}'.format(epoch, num_epoch, loss.item()))

    with torch.no_grad():  # torch.no_grad()를 하면 gradient 계산을 수행하지 않는다.
        slist = torch.tensor(statelist[-testsamplesize:], dtype=torch.float).squeeze(1).to(device)
        aOnehotlist = torch.tensor(actOneHot[-testsamplesize:], dtype=torch.float).squeeze(1).to(device)
        alist = torch.tensor(actlist[-testsamplesize:]).squeeze(1).to(device)

        # slist = torch.tensor(statelist[:-testsamplesize], dtype=torch.float).squeeze(1).to(device)
        # aOnehotlist = torch.tensor(actOneHot[:-testsamplesize], dtype=torch.float).squeeze(1).to(device)
        # alist = torch.tensor(actlist[:-testsamplesize]).squeeze(1).to(device)

        pi = model.pi(slist)
        # print(torch.argmax(pi, 1))
        # print(torch.argmax(aOnehotlist, 1))
        # print(torch.argmax(pi, 1) == torch.argmax(aOnehotlist, 1))
        print(torch.argmax(pi, 1) == alist)
        # print(alist)
        # correct_prediction = torch.argmax(pi, 1) == torch.argmax(aOnehotlist, 1)
        correct_prediction = torch.argmax(pi, 1) == alist
        accuracy = correct_prediction.float().mean()
        print('Accuracy:', accuracy.item())

        # # MNIST 테스트 데이터에서 무작위로 하나를 뽑아서 예측을 해본다
        # r = random.randint(0, len(mnist_test) - 1)
        # X_single_data = mnist_test.test_data[r:r + 1].view(-1, 28 * 28).float().to(device)
        # Y_single_data = mnist_test.test_labels[r:r + 1].to(device)
        #
        # print('Label: ', Y_single_data.item())
        # single_prediction = linear(X_single_data)
        # print('Prediction: ', torch.argmax(single_prediction, 1).item())


if __name__ == '__main__':
    BehaviorClone()

