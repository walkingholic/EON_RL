import numpy as np
import matplotlib.pyplot as plt
from simulation_v2 import Simulation
import copy
import csv
import datetime
import pandas as pd
import os


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

result = []
for fin in fList:
    if fin.startswith('result'):
        fname = os.path.join(read_path, fin)
        pdata = pd.read_csv(fname)
        print(pdata.columns)
        nblock = pdata['cnt_gen_band']
        print(nblock)
        data = np.array(pdata, dtype=object)
        result.append(data)
    elif fin.startswith('setting'):
        fname = os.path.join(read_path, fin)
        setdata = pd.read_csv(fname)
        print(setdata.columns)
        print(setdata)
        # setdata = setdata.transform
        print(setdata.transform)

        now_start = setdata["Date"][0]
        num_of_senario = setdata["num_of_senario"][0]
        num_of_req = setdata["num_of_req"][0]
        num_of_warmup = setdata["num_of_warmup"][0]
        num_kpath = setdata["num_kpath"][0]
        ar = setdata["ar"][0]
        dif = setdata["dif"][0]
        # setdata["erlang"] = erlang
        avg_holding = setdata["avg_holding"][0]
        range_band_s = setdata["range_band_s"][0]
        range_band_e = setdata["range_band_e"][0]
        TotalNumofSlots = setdata["num_of_slots"][0]

        print(num_of_senario)


erlang = []
for i in range(num_of_senario):
    erlang.append((ar + i*dif) * avg_holding)

algo = [('b', 'o', '--'), ('b','x',':'), ('b','^','-'), ('b','o','-'), ('r', 'o', '--'), ('r','x',':'), ('r','^','-') ]#('r','o','-'), ('g','x','-')
plt.figure(figsize=(12, 10))
plt.subplot(221)

print(np.shape(result))
# print((result))
plt.title('Total Blocked Req')
for i, rdata in enumerate(result):
    # print(i, rdata[i][-3], rdata[i][-2])
    # print(rdata)
    plt.plot(erlang, rdata[:, 0] / num_of_req, color=algo[i][0], marker=algo[i][1], linestyle=algo[i][2],
             label='%s_%s' % (rdata[0][-3], rdata[0][-2]))

plt.yscale("log")
plt.grid(True, axis='y', color='gray', alpha=0.5, linestyle='--')
plt.legend()



plt.subplot(222)
plt.title('Utilization')
for i, rdata in enumerate(result):
    plt.plot(erlang, rdata[:, 7], color=algo[i][0], marker=algo[i][1], linestyle=algo[i][2],  label='%s_%s' % (rdata[0][-3], rdata[0][-2]))
plt.grid(True, axis='y', color='gray', alpha=0.5, linestyle='--')
plt.legend()


plt.subplot(223)
plt.title('Avg. hop')
for i, rdata in enumerate(result):
    plt.plot(erlang, rdata[:, 8], color=algo[i][0], marker=algo[i][1], linestyle=algo[i][2], label='%s_%s' %(rdata[0][-3], rdata[0][-2]))
plt.grid(True, axis='y', color='gray', alpha=0.5, linestyle='--')
plt.legend()


plt.subplot(224)
plt.title('Total Blocked Band')
for i, rdata in enumerate(result):
    plt.plot(erlang, rdata[:, 5] / rdata[:, 4], color=algo[i][0], marker=algo[i][1], linestyle=algo[i][2],
             label='%s_%s' % (rdata[0][-3], rdata[0][-2]))
plt.yscale("log")
plt.grid(True, axis='y', color='gray', alpha=0.5, linestyle='--')
plt.legend()


plt.show()
# out_img_path = os.path.join(dirpath, 'fig.png')
# imgname = dirpath
# plt.savefig(out_img_path)