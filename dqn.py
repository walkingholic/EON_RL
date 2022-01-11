import random
import numpy as np
import matplotlib.pyplot as plt
import request

random.seed(1)
np.random.seed(1)
arrival_rate = 20
holding_time = 50
# cur_time = 0
NumOfNode = 14
range_band_s = 10
range_band_e = 100
SlotWidth = 12.5

def gene_request(idx_req, ct, warmup):
    s = random.randint(1, NumOfNode)
    d = random.randint(1, NumOfNode)
    while s == d:
        d = random.randint(1, NumOfNode)
    b = random.randint(range_band_s, range_band_e)
    inter_arrival_time = round(np.random.exponential(1/arrival_rate), 4)

    h = round(np.random.exponential(holding_time), 4)
    while h == 0:
        h = round(np.random.exponential(holding_time), 4)


    ct = ct + inter_arrival_time
    req = request.Request(idx_req, s, d, b, ct, h,  warmup)
    return req, ct


ct=0
gened_req = []
for i in range(10000):

    req, ct = gene_request(i, ct, 0)
    gened_req.append(req)

traffic_matrix = dict()

for req in gened_req:
    src = req.source
    dst = req.destination
    band = req.bandwidth
    if src not in traffic_matrix.keys():
        traffic_matrix[src] = {}
    if dst not in traffic_matrix[src].keys():
        traffic_matrix[src][dst] = band
    else:
        traffic_matrix[src][dst] += band




#
# for s in traffic_matrix:
#     print(s, traffic_matrix[s])

for u in range(1,15):
    for v in range(1,15):
        # print(u, v)
        if v not in traffic_matrix[u].keys():
            print('    ', end='  ')
        else:
            print(traffic_matrix[u][v], end='  ')
    print('\n')




plt.subplot(321)
a = np.random.exponential(1/arrival_rate, 100000)
# a = np.random.poisson(arrival_rate, 100000)
print(a)
plt.hist(a, bins=100, color='red', histtype='step', density=True)
plt.subplot(322)
plt.plot(a, color='red')


plt.subplot(323)
p = np.random.poisson(arrival_rate, 100000)
print(p)
plt.hist(p, bins=100, color='red', histtype='stepfilled', density=True)


plt.subplot(324)
plt.plot(a, color='red')


plt.subplot(325)
h = np.random.exponential(holding_time, 100000)
plt.hist(h, bins=100, color='blue', histtype='step', density=True)
plt.subplot(326)
plt.plot(h, color='blue')

plt.show()


for i in h:
    if i==0:
        print('Here', i)






aa=np.arange(100).reshape(10,10)
print(aa)
print(aa[1,2])









        # sd_pair = []
# for i in range(100000):
#     s = np.random.randint(1, 14)
#     d = np.random.randint(1, 14)
#     while s==d:
#         d = np.random.randint(1, 14)
#     sd_pair.append((s,d))
#
# print(sd_pair)



# a = [random.randint(1,10) for _ in range(10)]
# print(a)
#
# b = [random.randint(1,10) for _ in range(10)]
# print(b)
#
#
# n = np.random.random((6,6))
# print(n)
# print(sum(n))
#
# n = np.random.randint(1, 11, (6,6))
# print(n)
# print(sum(n))
#
#
# n = np.random.normal(0, 1, (6,6))
# print(n)
# print(sum(n))