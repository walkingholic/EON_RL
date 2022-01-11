import numpy as np
import matplotlib.pyplot as plt
from simulation_v2 import Simulation
import copy
import csv
import datetime
import pandas as pd
import os


def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


rollingvalue = 200

now_start=0
num_of_senario=0
num_of_req=0
num_of_warmup=0
num_kpath=0
ar=0
dif=0
# setdata["erlang"] = erlang
avg_holding=0
range_band_s=0
range_band_e =0
TotalNumofSlots=0

read_path = 'analysis'
fList = os.listdir(read_path)


# labels = ['Proposed 2 Worker', 'Proposed 4 Worker', 'KSP-FF']
labels = []

result = []
for fin in fList:
    if fin.startswith('result'):
        fname = os.path.join(read_path, fin)
        pdata = pd.read_csv(fname)
        labels.append(fname)
        print(pdata.columns)
        # nblock = pdata['cnt_gen_band']
        # print(nblock)
        log_avg_ept_score = pdata['log_avg_ept_score']
        log_avg_succ_req = pdata['log_avg_succ_req']#+log_avg_succ_req
        log_avg_blk_req = pdata['log_avg_blk_req']#+log_avg_succ_req
        log_avg_BP_req = pdata['log_avg_BP_req']#+log_avg_succ_req
        log_avg_entropy = pdata['log_avg_entropy']#+log_avg_succ_req



        data_log_avg_ept_score = np.array(log_avg_ept_score, dtype=object)
        data_log_avg_succ_req = np.array(log_avg_succ_req, dtype=object)
        data_log_avg_blk_req = np.array(log_avg_blk_req, dtype=object)
        data_log_avg_BP_req = np.array(log_avg_BP_req, dtype=object)
        data_log_avg_entropy = np.array(log_avg_entropy, dtype=object)



        data = np.array(pdata, dtype=object)
        result.append(data)
        print(len(pdata.columns))



    # elif fin.startswith('setting'):
    #     fname = os.path.join(read_path, fin)
    #     setdata = pd.read_csv(fname)
    #     print(setdata.columns)
    #     print(setdata)
    #     # setdata = setdata.transform
    #     print(setdata.transform)
    #
    #     now_start = setdata["Date"][0]
    #     num_of_senario = setdata["num_of_senario"][0]
    #     num_of_req = setdata["num_of_req"][0]
    #     num_of_warmup = setdata["num_of_warmup"][0]
    #     num_kpath = setdata["num_kpath"][0]
    #     ar = setdata["ar"][0]
    #     dif = setdata["dif"][0]
    #     # setdata["erlang"] = erlang
    #     avg_holding = setdata["avg_holding"][0]
    #     range_band_s = setdata["range_band_s"][0]
    #     range_band_e = setdata["range_band_e"][0]
    #     TotalNumofSlots = setdata["num_of_slots"][0]
    #
    #     print(num_of_senario)

# datalen = len(pdata)
# print('datalen' , datalen)

# for i in range(num_of_senario):
#     erlang.append((ar + i*dif) * avg_holding)

# algo = [('b', 'o', '--'), ('b','x',':'), ('b','^','-'), ('b','o','-'), ('r', 'o', '--'), ('r','x',':'), ('r','^','-') ]#('r','o','-'), ('g','x','-')
cmap = ['blue','red','green','darkorange','indigo',  'magenta' ,  'pink', 'aqua', 'black', 'yellow', 'gray', 'teal' ]#('r','o','-'), ('g','x','-')
# plt.figure(figsize=(6, 6))
# plt.subplot(221)

print(np.shape(result))
# print((result))


plt.figure(figsize=(12, 10))


plt.title('Blocking Probability')
for i, rdata in enumerate(result):
    # print(i, rdata[i][-3], rdata[i][-2])
    # print(rdata)
    df = pd.DataFrame(rdata[:, 3])
    # print('df ',len(df))
    dataidx = [idx for idx in range(len(df))]
    # plt.plot(dataidx, rdata[:, 3], alpha=0.5, label='%s' % (labels[i]), color=cmap[i])
    plt.plot(dataidx, df.rolling(rollingvalue).mean(), label='%s' % (labels[i]), color=cmap[i])



# kspff=[0.1215 for _ in range(datalen)]
# plt.plot(erlang, kspff ,label='%s' % (labels[-1]))
# plt.yscale("log")
plt.xlabel('Episode')
plt.ylabel('Probability')
plt.grid(True, axis='y', color='gray', alpha=0.5, linestyle='--')
plt.legend()
# plt.show()
fig = plt.gcf()
fig.savefig('{}/Blocking Probability.png'.format(read_path), facecolor='white', dpi=600)
plt.clf()


plt.title('Cumulative Reward')
for i, rdata in enumerate(result):
    # print(i, rdata[i][-3], rdata[i][-2])
    # print(rdata)
    df = pd.DataFrame(rdata[:, 0])
    dataidx = [idx for idx in range(len(df))]
    # plt.plot(dataidx, rdata[:, 0], alpha=0.5, label='%s' % (labels[i]), color=cmap[i])
    plt.plot(dataidx, df.rolling(rollingvalue).mean(), label='%s' % (labels[i]), color=cmap[i])



# plt.yscale("log")
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.grid(True, axis='y', color='gray', alpha=0.5, linestyle='--')
plt.legend()
# plt.show()
fig = plt.gcf()
fig.savefig('{}/Cumulative Reward.png'.format(read_path), facecolor='white', dpi=600)
plt.clf()

plt.title('Entropy')
for i, rdata in enumerate(result):
    # print(i, rdata[i][-3], rdata[i][-2])
    # print(rdata)

    df = pd.DataFrame(rdata[:, 4])
    dataidx = [idx for idx in range(len(df))]
    # plt.plot(dataidx, rdata[:, 4], alpha=0.5, label='%s' % (labels[i]), color=cmap[i])
    plt.plot(dataidx, df.rolling(rollingvalue).mean(), label='%s' % (labels[i]), color=cmap[i])



# plt.yscale("log")
plt.xlabel('Episode')
plt.ylabel('Entropy')
plt.grid(True, axis='y', color='gray', alpha=0.5, linestyle='--')
plt.legend()
# plt.show()

fig = plt.gcf()
fig.savefig('{}/Entropy.png'.format(read_path), facecolor='white', dpi=600)
plt.clf()









#
#
# plt.subplot(222)
# plt.title('Utilization')
# for i, rdata in enumerate(result):
#     plt.plot(erlang, rdata[:, 7], color=algo[i][0], marker=algo[i][1], linestyle=algo[i][2],  label='%s_%s' % (rdata[0][-3], rdata[0][-2]))
# plt.grid(True, axis='y', color='gray', alpha=0.5, linestyle='--')
# plt.legend()
#
#
# plt.subplot(223)
# plt.title('Avg. hop')
# for i, rdata in enumerate(result):
#     plt.plot(erlang, rdata[:, 8], color=algo[i][0], marker=algo[i][1], linestyle=algo[i][2], label='%s_%s' %(rdata[0][-3], rdata[0][-2]))
# plt.grid(True, axis='y', color='gray', alpha=0.5, linestyle='--')
# plt.legend()
#
#
# plt.subplot(224)
# plt.title('Total Blocked Band')
# for i, rdata in enumerate(result):
#     plt.plot(erlang, rdata[:, 5] / rdata[:, 4], color=algo[i][0], marker=algo[i][1], linestyle=algo[i][2],
#              label='%s_%s' % (rdata[0][-3], rdata[0][-2]))
# plt.yscale("log")
# plt.grid(True, axis='y', color='gray', alpha=0.5, linestyle='--')
# plt.legend()



# out_img_path = os.path.join(dirpath, 'fig.png')
# imgname = dirpath
# plt.savefig(out_img_path)