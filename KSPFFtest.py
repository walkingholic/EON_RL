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
from ConvPPO import PPO, ActorCritic
# from PPO import RolloutBuffer
import torch.multiprocessing as mp
# from multiprocessing import Manager


env_version=2
n_pro = 16
num_kpath = 3
num_Subblock = 1
N_A = num_kpath*num_Subblock
num_of_senario = 1

statesize = 86
hiddensize = 128

N_S = 100
timestep_max = 1000
update_interval = 100
learning_rate = 0.00001
gamma = 0.95
global alpha
alpha = 0.01

TotalNumofSlots = 100
num_EP = 10000

num_of_req = int((num_EP*timestep_max)/n_pro)
num_of_warmup = 1000
ar = 10
dif = 1
avg_holding = 20
range_band_s = 1
range_band_e = 10





basepath = 'result\\'

def createFolder(directory):
    try:
        if not os.path.isdir(directory):
            os.makedirs(directory)
    except OSError:
        print('error')

def doWork():

    now_start = datetime.datetime.now()
    resultdir = '{0:02}-{1:02} {2:02}-{3:02} {4:02} result_KSP-FF_EP={5}k_K={6}_SB={7}_eplen={8}k'.format(
        now_start.month, now_start.day, now_start.hour, now_start.minute, now_start.second, num_EP/1000,  num_kpath, num_Subblock, timestep_max/1000)
    dirpath = os.path.join(basepath, resultdir)
    createFolder(dirpath)
    print(dirpath)

    setting_data = {}
    setting_file = 'setting.csv'
    setting_data["Date"] = now_start
    setting_data["num_of_senario"]= num_of_senario
    setting_data["num_of_req"] = num_of_req
    setting_data["num_of_warmup"] = num_of_warmup
    setting_data["num_kpath"] = num_kpath
    setting_data["ar"] = ar
    setting_data["dif"] = dif
    # setting_data["erlang"] = erlang
    setting_data["avg_holding"] = avg_holding
    setting_data["range_band_s"] = range_band_s
    setting_data["range_band_e"] = range_band_e
    setting_data["num_of_slots"] = TotalNumofSlots


    setting_file_path = os.path.join(dirpath, setting_file)
    print('Print setting data ',setting_file_path)
    # print(setting_data)
    # , columns=['Date', 'num_of_senario', 'num_of_req', 'num_of_warmup', 'num_kpath', 'ar', 'dif', 'avg_holding', 'range_band_s', 'range_band_e', 'num_of_slots']
    setting_pdata = pd.DataFrame([setting_data])
    setting_pdata.to_csv(setting_file_path, index=False)




    result = []
    start_time = time.time()


    rt_name = 'KSP'
    # sa_name = 'FF'
    # sa_name = '2D'
    sa_name = 'FF'
    # sa_name = 'FF'
    res_queue = mp.Queue()

    # result_dic = mp.Manager().dict()
    agents = []
    for i in range(n_pro):
        print(i,' senerio start : ', rt_name ,'-',sa_name,'  ', ar * avg_holding, ' erlang')
        sim = Simulation(num_of_req, num_kpath, 0, num_of_warmup, ar, avg_holding, range_band_s, range_band_e, TotalNumofSlots,
                         rt_name, sa_name, dirpath, None, None, res_queue)
        #RT_name, SA_name, dirpath=None, env_version=None, dic=None, queue=None, req_in_gen=None
        agents.append(sim)

    [ag.start() for ag in agents]



    log_avg_ept_score = []
    log_avg_succ_req = []
    log_avg_blk_req = []
    log_avg_BP_req = []
    log_avg_entropy = []

    cnt = 0

    while True:
        result = res_queue.get()
        cnt += 1
        if result is not None:
            ept_score, succ_req, blk_req, entropy = result
            log_avg_ept_score.append(ept_score)
            log_avg_succ_req.append(succ_req)
            log_avg_blk_req.append(blk_req)
            log_avg_BP_req.append(blk_req / timestep_max)
            log_avg_entropy.append(entropy)

            if cnt%100 == 0:
                result_data = {}
                result_data['log_avg_ept_score'] = log_avg_ept_score
                result_data['log_avg_succ_req'] = log_avg_succ_req
                result_data['log_avg_blk_req'] = log_avg_blk_req
                result_data['log_avg_BP_req'] = log_avg_BP_req
                result_data['log_avg_entropy'] = log_avg_entropy

                result_file_name = 'result_{16}-{17}_EP={15}k_K={6}_SB={7}_eplen={8}k.csv'.format(now_start.month, now_start.day,
                                                                                      now_start.hour, now_start.minute,
                                                                                      now_start.second, n_pro,  num_kpath, num_Subblock, timestep_max/1000, update_interval/1000, learning_rate,
                                                                                               gamma,alpha, hiddensize, env_version, num_EP/1000, rt_name, sa_name)




                result_file_path = os.path.join(dirpath, result_file_name)
                result_data = pd.DataFrame(result_data)
                print((result_file_path))
                print(len(result_file_path))
                result_data.to_csv(result_file_path, index=False)


        if cnt == num_EP:
            for p in agents:
                p.terminate()

            break

    [ag.join() for ag in agents]



if __name__ == '__main__':

    doWork()

