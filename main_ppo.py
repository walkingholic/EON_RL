import numpy as np
import random
import matplotlib.pyplot as plt
# from simulation_v2 import Simulation
import copy
import csv
import datetime
import request
import pandas as pd
import os, time
from ConvPPO import PPO, ActorCritic
# from PPO import RolloutBuffer
import torch.multiprocessing as mp
# from multiprocessing import Manager
import torch


#
# ################################## set device ##################################
#
# print("============================================================================================")
#
# # set device to cpu or cuda
# device = torch.device('cpu')
#
# if (torch.cuda.is_available()):
#     device = torch.device('cuda:0')
#     torch.cuda.empty_cache()
#     print("Device set to : " + str(torch.cuda.get_device_name(device)))
# else:
#     print("Device set to : cpu")
#
# print("============================================================================================")
#

#
# num_of_senario = 1
# num_of_req = 60000
# num_of_warmup = 10000
# ar=4
# dif = 1
# avg_holding = 50
# range_band_s = 1
# range_band_e = 10
# TotalNumofSlots = 100
# num_kpath=5
#
# #    NOB = np.zeros((num_of_senario,1))
# NOB = [0 for x in range(num_of_senario)]
# Total_Gen_Band = [0 for x in range(num_of_senario)]
# Total_Block_Band = [0 for x in range(num_of_senario)]
# BBP = [0 for x in range(num_of_senario)]
# Utilization = [0 for x in range(num_of_senario)]
# AvgHop = [0 for x in range(num_of_senario)]
# GEN_BAND = []
# BLOCK_BAND = []
# SUCC_BAND = []
# SUCC_NOS = []
# FI = []
#
# BP_BAND = []
# Elrang = []



basepath = 'result\\'

def createFolder(directory):
    try:
        if not os.path.isdir(directory):
            os.makedirs(directory)
    except OSError:
        print('error')


def doTrain():
    n_pro=8
    gamma = 0.99  # discount factor
    K_epochs = 4  # update policy for K epochs
    update_timestep = 1000
    eps_clip = 0.1  # clip parameter for PPO
    lr_actor = 0.01  # learning rate for actor network
    lr_critic = 0.0001  # learning rate for critic network

    N_A = 5
    N_S = 100

    gmodel = ActorCritic(N_S, N_A)
    gmodel.share_memory()

    now_start = datetime.datetime.now()
    resultdir = '{0:02}-{1:02} {2:02}-{3:02} {4:02}'.format(
        now_start.month, now_start.day, now_start.hour, now_start.minute, now_start.second)
    dirpath = os.path.join(basepath, resultdir)
    createFolder(dirpath)
    print(dirpath)

    res_queue = mp.Queue()
    agents = [PPO(i, update_timestep, gmodel, N_S, N_A, lr_actor, gamma, K_epochs, eps_clip, res_queue) for i
              in range(n_pro)]


    [ag.start() for ag in agents]

    res = []  # record episode reward to plot
    log_avg_ep = []
    log_avg_rwd = []
    log_tmp_add_ep_rwd = []

    flag = 0
    cnt = 0
    avg_rwd = 0
    while True:
        r = res_queue.get()
        cnt += 1
        if r is not None:
            res.append(r)

            log_tmp_add_ep_rwd.append(r)
            if cnt % 20 == 0:
                log_avg_ep.append(cnt)
                avg_rwd = sum(log_tmp_add_ep_rwd) / len(log_tmp_add_ep_rwd)
                log_avg_rwd.append(avg_rwd)
                log_tmp_add_ep_rwd.clear()
                print(cnt, 'Avg. Reward: ', avg_rwd)

                training_time = datetime.datetime.now() - now_start
                #
                plt.title('Training avg eptscores: {}'.format(training_time))
                plt.xlabel('Epoch')
                plt.ylabel('score')
                plt.plot(log_avg_ep, log_avg_rwd, 'b')
                plt.grid(True, axis='y')
                fig = plt.gcf()
                fig.savefig('{}/train avg eptscores.png'.format(dirpath), facecolor='white', dpi=600)
                plt.clf()
            # if cnt % 1000 == 0:
            #     file_name = "PPO_{:05d}_{}.pth".format(cnt, round(avg_rwd))
            #     file_path = os.path.join(pthpath, file_name)
            #     gmodel.save(file_path)

        else:
            flag += 1
            if flag == n_pro:
                break

    [ag.join() for ag in agents]



if __name__ == '__main__':


    doTrain()

