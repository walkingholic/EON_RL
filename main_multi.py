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


num_of_senario = 1
num_of_req = 50000
num_of_warmup = 10000
ar=10
dif = 1
avg_holding = 20
range_band_s = 1
range_band_e = 10
TotalNumofSlots = 100
num_kpath=5

#    NOB = np.zeros((num_of_senario,1))
NOB = [0 for x in range(num_of_senario)]
Total_Gen_Band = [0 for x in range(num_of_senario)]
Total_Block_Band = [0 for x in range(num_of_senario)]
BBP = [0 for x in range(num_of_senario)]
Utilization = [0 for x in range(num_of_senario)]
AvgHop = [0 for x in range(num_of_senario)]
GEN_BAND = []
BLOCK_BAND = []
SUCC_BAND = []
SUCC_NOS = []
FI = []

BP_BAND = []
Elrang = []



basepath = 'result\\'

def createFolder(directory):
    try:
        if not os.path.isdir(directory):
            os.makedirs(directory)
    except OSError:
        print('error')

def doWork(erlang, req_in_gen, *algo):

    now_start = datetime.datetime.now()
    resultdir = '{0:02}-{1:02} {2:02}-{3:02} {4:02}'.format(
        now_start.month, now_start.day, now_start.hour, now_start.minute, now_start.second)
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

    for al in algo:
        rt_name, sa_name, _, _, _ = al

        req_list = copy.deepcopy(req_in_gen)
        data = []
        res_queue = mp.Queue()

        result_dic = mp.Manager().dict()
        flag = 0
        agents = []
        for i in range(num_of_senario):
            print(i,' senerio start : ', rt_name ,'-',sa_name,'  ', (ar + (i) * dif) * avg_holding, ' erlang')
            sim = Simulation(num_of_req, num_kpath,0, num_of_warmup, (ar + (i) * dif), avg_holding, range_band_s, range_band_e, TotalNumofSlots,
                             rt_name, sa_name, dirpath, _, result_dic,_, req_list[i]) # dirpath=None, env_version=None, dic=None, queue=None, req_in_gen=None
            agents.append(sim)
            # nob, cnt_gen_band, cnt_blk_band, cnt_suc_band, total_Gen_bandwidth, total_Block_bandwidth, cnt_suc_nos, utilization, avghop, routing_name, assingment_name, networktopo = sim.start()
            # sim.start()
        #     data.append(
        #         [nob, cnt_gen_band, cnt_blk_band, cnt_suc_band, total_Gen_bandwidth, total_Block_bandwidth, cnt_suc_nos,
        #          utilization, avghop, routing_name, assingment_name, networktopo])
        #
        # data = np.array(data, dtype=object)
        # result.append(data)
        [ag.start() for ag in agents]

        [ag.join() for ag in agents]
        result_sorted_result = sorted(result_dic.items())

        for k, l in result_sorted_result:
            data.append(l)
            # print(type(l))
            # print(l)
        data = np.array(data, dtype=object)
        result.append(data)

    elapsed_time = time.time() - start_time
    print(elapsed_time)
    # print(result)

    for i, rdata in enumerate(result):
        out_filename = 'result_{}_{}_{}.csv'.format(algo[i][0], algo[i][1], elapsed_time)
        out_filename_path = os.path.join(dirpath, out_filename)
        print(out_filename_path)

        selectdata = pd.DataFrame(rdata,
                                  columns=['nob', 'cnt_gen_band', 'cnt_blk_band', 'cnt_suc_band', 'total_Gen_bandwidth',
                                           'total_Block_bandwidth', 'cnt_suc_nos', 'utilization', 'avghop',
                                           'routing_name', 'assingment_name', 'networktopo'])
        selectdata.to_csv(out_filename_path, index=False)  # header=None, index=None






    plt.figure(figsize=(12, 10))
    plt.subplot(221)

    plt.title('Total Blocked Req')
    for i, rdata in enumerate(result):
        plt.plot(erlang, rdata[:, 0] / num_of_req, color=algo[i][2], marker=algo[i][3], linestyle=algo[i][4],
                 label='%s_%s' % (algo[i][0], algo[i][1]))
    plt.yscale("log")
    plt.grid(True, axis='y', color='gray', alpha=0.5, linestyle='--')
    plt.legend()



    plt.subplot(222)
    plt.title('Utilization')
    for i, rdata in enumerate(result):
        plt.plot(erlang, rdata[:, 7], color=algo[i][2], marker=algo[i][3],linestyle=algo[i][4],  label='%s_%s' % (algo[i][0], algo[i][1]))
    plt.grid(True, axis='y', color='gray', alpha=0.5, linestyle='--')
    plt.legend()


    plt.subplot(223)
    plt.title('Avg. hop')
    for i, rdata in enumerate(result):
        plt.plot(erlang, rdata[:, 8], color=algo[i][2], marker=algo[i][3], linestyle=algo[i][4],label='%s_%s' % (algo[i][0], algo[i][1]))
    plt.grid(True, axis='y', color='gray', alpha=0.5, linestyle='--')
    plt.legend()


    plt.subplot(224)
    plt.title('Total Blocked Band')
    for i, rdata in enumerate(result):
        plt.plot(erlang, rdata[:, 5] / rdata[:, 4], color=algo[i][2], marker=algo[i][3], linestyle=algo[i][4],
                 label='%s_%s' % (algo[i][0], algo[i][1]))
    plt.yscale("log")
    plt.grid(True, axis='y', color='gray', alpha=0.5, linestyle='--')
    plt.legend()


    # plt.show()
    out_img_path = os.path.join(dirpath, 'fig.png')
    # imgname = dirpath
    plt.savefig(out_img_path)

def doTrain():
    n_pro=1
    gamma = 0.99  # discount factor
    K_epochs = 32  # update policy for K epochs
    n_pro = 8
    update_timestep = 60
    eps_clip = 0.1  # clip parameter for PPO
    lr_actor = 0.0001  # learning rate for actor network
    lr_critic = 0.0001  # learning rate for critic network

    graph = Graph_34()
    N_A = 5
    N_S = 100

    gmodel = ActorCritic(N_S, N_A)
    gmodel.share_memory()

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
            if cnt % 2000 == 0:
                log_avg_ep.append(cnt)
                avg_rwd = sum(log_tmp_add_ep_rwd) / len(log_tmp_add_ep_rwd)
                log_avg_rwd.append(avg_rwd)
                log_tmp_add_ep_rwd.clear()
                print(cnt, 'Avg. Reward: ', avg_rwd)

                # training_time = datetime.datetime.now() - now_start
                #
                # plt.title('Training avg eptscores: {}'.format(training_time))
                # plt.xlabel('Epoch')
                # plt.ylabel('score')
                # plt.plot(log_avg_ep, log_avg_rwd, 'b')
                # plt.grid(True, axis='y')
                # fig = plt.gcf()
                # fig.savefig('{}/train avg eptscores.png'.format(resultdir), facecolor='white', dpi=600)
                # plt.clf()
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



    # doTrain()






    req_in_gen=[]
    erlang=[]

    for i in range(num_of_senario):
        print('Making REQ -- senerio start : ', (ar + (i) * dif) * avg_holding, ' erlang')
        sim = Simulation(num_of_req, num_kpath,0,  num_of_warmup, (ar + (i) * dif), avg_holding, range_band_s, range_band_e, TotalNumofSlots, '__', '==')
        req_in_gen.append(copy.deepcopy(sim.req_in_gen))
        erlang.append((ar + i*dif) * avg_holding)

    # doWork(erlang, req_in_gen, ('KSP', 'FF', 'b', 'o', '--'), ('KSP', 'BF', 'b', 'x', ':'), ('DJP', 'FLEF', 'r', 'o', '-'))





    doWork(erlang, req_in_gen, ('KSP', 'FF', 'b', 'o', '--'))

    # doWork(erlang, req_in_gen, ('KSP', 'FF', 'b', 'o', '--'), ('KSP','BF','b','x',':'), ('KSP','2D', 'b','^','-'), ('KSP','FZ2D','r','o',':'), ('DJP','FLEF','r','o','-'), ('FZ','FZ2D', 'g','^','-'), ('FZ','2D', 'g','^','--'))


    # doWork(erlang, req_in_gen, ('KSP','FF', 'black','^','-'), ('FZ','2D','b','x','-'), ('DJP','FLEF','r','o','--'), ('LUP','2D', 'g','^',':'))
    # doWork(erlang, req_in_gen, ('KSP','FF', 'black','^','-'), ('FZ','2D','b','x','-'), ('DJP','FLEF','r','o','--'), ('KSP','BF','b','x','-') )
    # doWork(erlang, req_in_gen, ('KSP','FF','b','x','-'), ('DJP','FLEF','r','o','--'))
    # doWork(erlang, req_in_gen, ('KSP','FF','b','x','-'),  ('KSP','BF','b','p','-'), ('KSP','2D', 'g','^','-'),  ('DJP','FLEF','r','o','--'), ('LUP','2D', 'g','^','--'))
    # doWork(erlang, req_in_gen, ('KSP','FF','b','x','-'),  ('KSP','BF','b','p','-'), ('KSP','2D', 'g','^','-'),  ('DJP','FLEF','r','o','--'))
    # doWork(erlang, req_in_gen, ('KSP','2D', 'b','x','-'), ('LU','2D', 'g','^','-'),  ('LUP','2D','r','o','--'))
    # doWork(erlang, req_in_gen, ('KSP','2D', 'b','x','-'), ('KSP','SL', 'g','^','-'),  ('KSP','HT','r','o','--'))
    # doWork(erlang, req_in_gen, ('KSP','2D', 'b','x','-'), ('KSP','SL', 'g','^','-'),  ('KSP','HT','r','o','-'), ('LUP','2D', 'b','x','--'), ('LUP','SL', 'g','^','--'),  ('LUP','HT','r','o','--'))
    # doWork(erlang, req_in_gen, ('KSP','2D', 'g','^','-'), ('FZ','HT', 'b','x','--'),('LUP','HT','r','o','-'))
    # doWork(erlang, req_in_gen, ('LUP','FZ2D','r','o','-'), ('LUP','2D', 'g','^','-'), ('LUP','HT', 'b','x','--'))
    # doWork(erlang, req_in_gen, ('LUP','2D', 'g','^','-'), ('LUP','HT', 'b','x','--'))
    # doWork(erlang, req_in_gen, ('KSP','FZ2D','r','o','-'), ('KSP','2D', 'g','^','-'), ('KSP','HT', 'b','x','--'))
