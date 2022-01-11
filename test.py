from graph import DiGraph
import algorithms
import copy
import networkx as nx
import random
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import request
import csv
import pandas as pd
from PIL import Image
from PIL import ImageDraw
import torch





# # actionlist = [random.randint(0,49) for i in range(10)]
# # actionlist = [ np.random.randint(0, 1) for i in range(100)]
# actionlist = [ random.randint(1, 4) for i in range(100)]
# # actionlist = np.random.randint(0, 1, 50)
# print(actionlist)
# x = np.arange(len(actionlist))
# # plt.hist(actionlist, bins=49, cumulative=True,density=True,  label='CDF', histtype='step')
# # plt.hist(actionlist, bins=49, density=True,  label='CDF', histtype='step')
# # plt.show()
# # actionlist = np.array(actionlist)
#
# print(np.bincount(actionlist))
# bincnt = np.bincount(actionlist)
#
# for i, b in enumerate(bincnt):
#     if i%10==9:
#         print()
#     else:
#         print('{0:03}'.format(b), end=' ')
#
# print('test, this is a test!')




baseslot = [[random.randint(0,2), 0] for i in range(50)]
print(len(baseslot))

baseslot = np.array(baseslot)
nofrslot = 2

print(baseslot[:,0])



contig_block = []
slot_count = 0
s = 0

for i in range(len(baseslot)):
    if baseslot[i][0] == 0:
        if slot_count == 0:
            s = i
        slot_count += 1
        if i == 50 - 1:
            # if slot_count >= nofrslot:
            contig_block.append((s, s + slot_count - 1, slot_count, {99}))  # start, end, count,
    elif baseslot[i][0] > 0:
        if slot_count != 0:
            # if slot_count >= nofrslot:
            contig_block.append((s, s + slot_count - 1, slot_count, {99}))
            slot_count = 0


# contig_block = []

# contig_block = [(1,1,0,1), (1,1,1,1), (1,1,0,1), (1,1,0,1), (1,1,0,1), (1,1,0,1)]
contig_block = np.array(contig_block)
print(contig_block)
# print(len(contig_block))
# print(contig_block.sum(axis=0))
#
# c = contig_block.sum(axis=0)[2]


# maxlen = contig_block.max(axis=0)[2]
# minlen = contig_block.min(axis=0)[2]
# avgc = contig_block.mean(axis=0)[2]
#
# print(maxlen)
# print(minlen)
# print(avgc)



candi_SB = []
available_SB=[]
s = 0

# for i in range(len(baseslot)):
#     c=0
#
#     while c < nofrslot and i+c<50:
#         if baseslot[i+c][0] == 0:
#             c += 1
#             # print('i+c : ',i+c)
#         else:
#             break
#
#     if c >= nofrslot:
#         # print(i, c)
#         candi_SB.append((i, i+c-1, c))
#
#     if len(candi_SB) >= 2:
#         break
#
for sb in contig_block:
    sidx, eidx, slen, pp = sb
    if slen < nofrslot:
        continue
    if slen == nofrslot:
        available_SB.append((sidx, eidx, slen))
    else:
        available_SB.append((sidx, sidx+nofrslot-1, nofrslot))
        available_SB.append((eidx-nofrslot+1, eidx, nofrslot))

# print(candi_SB)

print('nofrslot', nofrslot)

testlist = []
print(available_SB)
testlist.append(copy.copy(available_SB[:2]))

for sb in available_SB[:2]:
    print(sb)


print(testlist)
lsb = 1
j = 4

for k in range(j-lsb):
    print('kk', k)



# source = np.eye(30)[1]
# dest = np.eye(30)[2]
# # state = np.concatenate([source, dest, baseslot[:,1]])
# state = np.concatenate([source, dest, [-1 for _ in range(50)]])
# print(state)













#
#
# for i in range(100):
#     print(random.randint(1,10))
#
# state = [[1 for i in range(100)] for r in range(20)]
# print(state)
#
#
# arr=[[1,2],[3,4],[5,6],[7,8],[9,10]]
# slot=[]
# arr = np.array(arr)
# print(arr[0][0])
#
# for i in range(5):
#     slot = np.concatenate([slot, arr[:,0]])
#
# # arr_np = np.array(arr)
# print(type(slot),slot)
# print(arr_np.shape)

# h = torch.tensor([151], dtype=torch.float32)
# print(torch.reshape(h,(120,)))
# req_info = torch.cat([s, d], dim=1)
# req_info = torch.stack([s, d], dim=0)
# print(req_info.size())
# print(req_info)
# print(s.size(), s)
# print(d.size(), d)
# print(b.size(), b)
# print(h.size(), h)


# s = np.eye(30)[3]
# d = np.eye(30)[12]
#
# bh = [holding_time, bandwidth]
#
# print(s)
# print(d)
# print(bh)
#
# req = np.concatenate([s, d, bh])
# req = np.expand_dims(req, axis=0)
# print(req)
# print(type(req))
# print(np.shape(req))
#
#
# arr = [1,4,2,3,1,2,3,45,5,3,14]
#
# print(arr/100)



# simg = torch.rand(100,100)
# print(simg)
# # simg = torch.tensor(slot, dtype=torch.float32)
# print(simg.shape)
# slotimg= simg.unsqueeze(0)
# print(slotimg.shape)
# slotimg=slotimg.unsqueeze(-1)
# print(slotimg.shape)
#
# s = torch.eye(30)[1]
# d = torch.eye(30)[12]
# bh = torch.FloatTensor([100, 50.2])
# req_info = torch.cat([s, d, bh]).unsqueeze(0)
# state = (slotimg, req_info)
#
# slotimg = slotimg.view(slotimg.size(0), -1)
#
# print(slotimg.size())
# print(req_info.size())
#
# data = torch.cat([slotimg, req_info], dim=1)
# print(data.size())
# #
# img = img.unsqueeze(0)
# img = img.unsqueeze(-1)
# print(img.shape)
#
# print(img.size(0))
# img = img.view(img.size(0), -1)
# print(img.size())
#
# encoding = np.eye(10)[4]
# print(encoding)
#
# s = torch.eye(10)[3]
# print(s)
# d = torch.eye(10)[4]
# print(d)
# b = torch.eye(10)[4]
# print(b)
# h = 40.2
# h = torch.FloatTensor([h])
# print(h, h.shape)
#
# data = torch.cat([s, d, b, h]).unsqueeze(0)
# print(data, data.shape)
#
# print(img.size(), data.size())
#
# data = torch.cat([img, data], dim=1)
# print(data, data.shape)



# z = np.zeros((100,100))
# print(z)

# slot_state = [[0 for i in range(10)] for r in range(5)]
# print(slot_state)

#
# plt.figure(figsize=(8,8),dpi=100)
#
# ax1 = plt.subplot(411)
#
# ax2 = plt.subplot(412)
#
# ax3 = plt.subplot(413)
#
# ax4 = plt.subplot(414)
#
# data = np.random.randint(1,80,100)
# x = np.arange(len(data))
# ax1.bar(x, data, color='blue', alpha=0.6)
# # ax1.grid(True, axis='both', color='gray', alpha=1, linestyle='--')
# ax1.axis('off')
#
# data = np.random.randint(1,80,100)
# x = np.arange(len(data))
# ax2.bar(x, data, color='blue', alpha=0.6)
#
# plt.show()

#
# img = np.random.randn(10, 10)
# print(img)
# plt.imshow(img)
# plt.show()


# image = Image.new("RGBA",(1200,100),(255,255,255,255))
# # image = Image.new("L",(1200,100), 255)
#
# (width, height) = image.size
# draw = ImageDraw.Draw(image)
# x1=0
# x2=0
# b=2
# d=10
# for i in range(100):
#       x2=x1+d
#       draw.rectangle([(x1, 1), (x2, 80)], fill=(0,0,255,255))
#       x1=x2+b
# x1=0
# x2=0
# b=2
# d=10
# for i in range(100):
#       x2=x1+d
#       draw.rectangle([(x1, 1), (x2, 40)], fill=(0,0,255,100))
#       x1=x2+b
#
# # draw.rectangle([(30, 30), (10, 10)], fill=(100))
#
# # (width, height) = image.size
# #
# draw.rectangle(((0, 0),(width-1, height-1)), outline=(0), width=1)
# # draw.rectangle(((30, 30),(230, 130)), outline=(0,0,255, 100), width=2)
#
# # im = image.load()
# #
# # (width, height) = image.size
# #
# # for i in range(0,width):
# #     if(im[i,i] != (255,255,255)):
# #         color = im[i,i]
# #
# # for i in range(0,width):
# #     for j in range(0,height):
# #         if(im[i,j] != (255,255,255) and im[i,j] != color):
# #             im[i,j] = (255-color[0],255-color[1],255-color[2])
#
# image.show()



# dic = {'A' : 1, 'D' : 4, 'C' : 3, 'B' : 2}
#
# sdic = sorted(dic.items())
# print(sdic)
# for k, v in sdic:
#     print(v)



# in_filename = 'result/result.csv'
#
#
# pdata = pd.read_csv(in_filename)
# print(pdata)
# print(type(pdata))
# data = np.array(pdata, dtype=object)
#
# print(data[:,1])
#





# aa = [[1,[2,2,2,2,2,2,2],3],
#       [4,[2,2,2,2,2,2,2],6],
#       [7,[2,2,2,2,2,2,2],9]]
# aa = [[1,2,3],
#       [4,5,6],
#       [7,8,9]]
# # print(aa[:, 1])
# aa = np.array(aa)
# print(aa[:, 1])


#
#
#
#
#
#
#
#
#
#
# K=5
# random.seed(1)
# np.random.seed(1)
# arrival_rate = 20
# holding_time = 50
# # cur_time = 0
# NumOfNode = 14
# range_band_s = 10
# range_band_e = 100
# SlotWidth = 12.5
#
# def precal_path(base_Graph):
#     nNode = len(base_Graph._data)
#     pre_cal_path = {}
#     cnt = 0
#     dist = 0
#     for i in range(1, nNode + 1):
#         pre_cal_path[i] = {}
#         for j in range(1, nNode + 1):
#             s = str(i)
#             d = str(j)
#             if s != d:
#                 items = algorithms.ksp_yen(base_Graph, s, d, K)
#                 for it in items:
#                     it['dist'] = it['cost']
#                     it['hop'] = len(it['path']) - 1
#                     cnt += 1
#                     dist += it['dist']
#                 #                        print(it)
#                 pre_cal_path[i][j] = copy.deepcopy(items)
#     #                    print('items  ',items)
#
#     return pre_cal_path
#
#
# def gene_request(idx_req, ct, warmup):
#     s = random.randint(1, NumOfNode)
#     d = random.randint(1, NumOfNode)
#     while s == d:
#         d = random.randint(1, NumOfNode)
#     b = random.randint(range_band_s, range_band_e)
#     inter_arrival_time = round(np.random.exponential(1/arrival_rate), 4)
#
#     h = round(np.random.exponential(holding_time), 4)
#     while h == 0:
#         h = round(np.random.exponential(holding_time), 4)
#
#
#     ct = ct + inter_arrival_time
#     req = request.Request(idx_req, s, d, b, ct, h,  warmup)
#     return req, ct
#
#
# # networktopo="test_nsf"
# networktopo="sixnode"
# base_Graph = DiGraph(networktopo)
# all_path = precal_path(base_Graph)
#
#
# vgraph = nx.Graph()
# # tmp = nx.DiGraph()
# ct=0
# gened_req = []
#
#
# for i in range(10000):
#
#     req, ct = gene_request(i, ct, 0)
#     gened_req.append(req)
#
# traffic_matrix = dict()
#
# for req in gened_req:
#     src = req.source
#     dst = req.destination
#     band = req.bandwidth
#     if src not in traffic_matrix.keys():
#         traffic_matrix[src] = {}
#     if dst not in traffic_matrix[src].keys():
#         traffic_matrix[src][dst] = band
#     else:
#         traffic_matrix[src][dst] += band
#
#
# for u in range(1,15):
#     for v in range(1,15):
#         # print(u, v)
#         if v not in traffic_matrix[u].keys():
#             print('    ', end='  ')
#         else:
#             print(traffic_matrix[u][v], end='  ')
#     print('\n')
#
#
#
# for s in all_path:
#     for d in all_path[s]:
#         for k in range(len(all_path[s][d])):
#             w = traffic_matrix[s][d]
#             path = all_path[s][d][k]['path']
#             # print(s, d, all_path[s][d][k]['path'])
#             edgelist = []
#
#             for e in range(len(path)-1):
#                 fn = int(path[e])
#                 tn = int(path[e+1])
#                 edgelist.append('%02d%02d'%(fn,tn))
#
#             vgraph.add_node('%02d%02d%02d' % (s, d, k), weight=w, path=path, edgelist=edgelist, src=s, dst=d)
#
#
# vert = sorted(vgraph.nodes(data=True), key=lambda element:element[1]['weight'], reverse=True)
# print(len(vert))
# # print((vert))
#
#
# for u in vert:
#     upath = u[1]['edgelist']
#     upID = u[0]
#     usrc = u[1]['src']
#     udst = u[1]['dst']
#     # print(upID, u[1]['weight'], usrc, udst, upath)
#
#     for v in vert:
#         vpath = v[1]['edgelist']
#         vpID = v[0]
#         if vpID==upID:
#             break
#         vsrc = v[1]['src']
#         vdst = v[1]['dst']
#         # print(upID, usrc, udst, upath)
#
#         for e in upath:
#             if e in vpath or (usrc==vsrc and udst == vdst):
#                 # print('-> ', vpID, '속함', e, vpath)
#                 vgraph.add_edge(upID, vpID)
#                 break
#
# dsj = []
# ndsj = []
#
# for e in vert:
#     tmp_adj = list(vgraph.adj[e[0]])
#     # print(e[0], e[1]['weight'], type(tmp_adj), tmp_adj)
#     isDsj = True
#     for v in tmp_adj:
#         if v in dsj:
#             isDsj = False
#             break
#
#     if isDsj:
#         dsj.append(e[0])
#         # print(type(e[0]))
#     else:
#         ndsj.append(e[0])
#
# print(len(dsj), dsj)
# print(len(ndsj), ndsj)
#
#
#
#
#
# # src = 14
# # dst = 11
# # # dict(vert)
# # src_dst = '%02d%02d'%(src,dst)
# # print(src_dst)
# # for p in dsj:
# #     result = p.startswith(src_dst)
# #     if result:
# #         print('it is', result, p)
# #         k = int(p[-2:])
# #         print(k)
# #         path = all_path[src][dst][k]['path']
# #         print(path)
# #         break
# # color_map=[]
# # for node in vgraph.nodes:
# #     if node in dsj:
# #         color_map.append('red')
# #     else:
# #         color_map.append('blue')
# # # font_weight='bold'
# # nx.draw(vgraph, with_labels=True, node_color=color_map)
# # plt.show()
# #
# #
# #
# # nG = nx.Graph(base_Graph._data)
# # nx.draw(nG, with_labels=True)
# # plt.show()
# #
#
#
# def sortpath(all_path):
#     for s in all_path:
#         for d in all_path[s]:
#             # print(all_path[s][d])
#             # paths = all_path[s][d]
#             paths = sorted(all_path[s][d], key=lambda element: element['cost'], reverse=True)
#
#             # print(s,d,paths)
#             # print(all_path[s][d])
#
#             # paths.sort(key=lambda element: element['cost'], reverse=True)
#             # print(s, d, paths)
#             # print(all_path[s][d])
#             # for k in range(K):
#             #     w = traffic_matrix[s][d]
#             #     path = all_path[s][d][k]['path']
#
# print(all_path)
# sortpath(all_path)
# print(all_path)
#


