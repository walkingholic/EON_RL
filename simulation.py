# # -*- coding: utf-8 -*-
# """
# Created on Wed Apr 24 20:18:54 2019
#
# @author: User
# """
# import random
# import request
# import numpy as np
# from graph import DiGraph
# import algorithms
# import copy
# import math
# import matplotlib.pyplot as plt
# import time
# import networkx as nx
#
#
# class Simulation:
#
#     def __init__(self, n_of_req, n_of_warm, inter_a_rate, holding_time, range_band_s, range_band_e):
#         self.N_Req = n_of_req
#         self.req_in_service = []
#         self.req_in_gen = []
#         self.req_in_success = []
#         self.req_in_blocked = []
#         self.modeShow = 10000
#         self.INF = 1000000
#         self.inter_arr_rate = inter_a_rate
#         self.holding_time = holding_time
#         self.cur_time = 0.0
# #        self.networktopo="sixnode"
#         self.networktopo="test_nsf"
# #        self.networktopo="test_usnet"
# #        self.G = DiGraph("sixnode")
# #        self.G = DiGraph("test_nsf")
#         self.G = DiGraph(self.networktopo)
#
#         # self.G.save_graph_as_svg(self.G._data, 'test_nsf')
#         # print(type(self.G._data))
#
#         self.NumOfNode = len(self.G._data)
#         self.Distance = copy.deepcopy(self.G._data)
# #        self.slot_info ={}
#         self.slot_info_2D ={}
#         self.TotalNumofSlots=320
#         self.K = 4
#
#         self.NOB = 0
#         self.NO_Success = 0
#
#         self.SlotWidth = 12.5
#         self.SimWarmup = n_of_warm
#         self.NumOfEdge = 0
#         self.range_band_s=range_band_s
#         self.range_band_e=range_band_e
#         self.Dthreshold = 0
#         self.linkfreq = {}
#         self.totalusedslot=0
#         self.utilization=0
#         self.avghop=0
#         self.routing_name=''
#         self.assingment_name=''
#
#         self.cnt_gen_band=np.zeros((self.range_band_e))
#         self.cnt_suc_band=np.zeros((self.range_band_e))
#         self.cnt_blk_band=np.zeros((self.range_band_e))
#         self.cnt_suc_nos=np.zeros((self.range_band_e))
#
#         self.total_Gen_bandwidth=0
#         self.total_Block_bandwidth=0
#
#         for i in self.G._data:
# #            self.slot_info[i]={}
#             self.slot_info_2D[i]={}
#             self.linkfreq[i]={}
#             for j in self.G._data[i]:
# #                self.slot_info[i][j]=np.zeros((self.TotalNumofSlots))
#                 self.slot_info_2D[i][j]=[[0, 0] for i in range(self.TotalNumofSlots) ]
#                 self.linkfreq[i][j]=0
# #                print(self.slot_info_2D[i][j])
#                 self.NumOfEdge += 1
#
#
#
#
#     def link_freq(self, allpath):
#
#         for s in allpath:
#             print('s', s)
#             for d in allpath[s]:
#                 print('d ', d)
#                 k=0
#                 for path in allpath[s][d]:
#                     print(k,'th ',path['cost'], '  ',  path['path'])
#                     k+=1
#                     for i in range(path['hop']):
#                         fromnode = path['path'][i]
#                         tonode = path['path'][i+1]
#                         f = fromnode
#                         t = tonode
#                         self.linkfreq[f][t] += 1
# #        print(self.linkfreq)
# #         nofl=0
# #         for f in self.linkfreq:
# #             for t in self.linkfreq[f]:
# #                 nofl+=1
# #                 print(nofl,'  ',f,' ',t,' ', self.linkfreq[f][t])
# #
#
#     def get_dsj_paths(self, all_path):
#
#         tmp = nx.Graph()
#         # tmp = nx.DiGraph()
#
#         for s in all_path:
#             for d in all_path[s]:
#                 for k in range(self.K):
#                     weight = random.randint(1, 300)
#                     path = all_path[s][d][k]['path']
#                     # print(s, d, all_path[s][d][k]['path'])
#                     edgelist = []
#
#                     for e in range(len(path) - 1):
#                         fn = int(path[e])
#                         tn = int(path[e + 1])
#                         edgelist.append('%02d%02d' % (fn, tn))
#
#                     tmp.add_node('%02d%02d%02d' % (s, d, k), weight=weight, path=path, edgelist=edgelist, src=s, dst=d)
#
#         vert = sorted(tmp.nodes(data=True), key=lambda element: element[1]['weight'], reverse=True)
#
#         for u in vert:
#             upath = u[1]['edgelist']
#             upID = u[0]
#             usrc = u[1]['src']
#             udst = u[1]['dst']
#             # print(upID, u[1]['weight'], usrc, udst, upath)
#
#             for v in vert:
#                 vpath = v[1]['edgelist']
#                 vpID = v[0]
#                 if vpID == upID:
#                     break
#                 vsrc = v[1]['src']
#                 vdst = v[1]['dst']
#                 # print(upID, usrc, udst, upath)
#
#                 for e in upath:
#                     if e in vpath or (usrc == vsrc and udst == vdst):
#                         # print('-> ', vpID, '속함', e, vpath)
#                         tmp.add_edge(upID, vpID)
#                         break
#
#         dsj = []
#         ndsj = []
#
#         for e in vert:
#             tmp_adj = list(tmp.adj[e[0]])
#             # print(e[0], e[1]['weight'], type(tmp_adj), tmp_adj)
#             isDsj = True
#             for v in tmp_adj:
#                 if v in dsj:
#                     isDsj = False
#                     break
#
#             if isDsj:
#                 dsj.append(e[0])
#             else:
#                 ndsj.append(e[0])
#
#         print(len(dsj), dsj)
#         print(len(ndsj), ndsj)
#
#         return dsj, ndsj
#
#
#
#     def sim_main(self):
#
#         all_k_path = self.precal_path()
#         # self.link_freq(all_k_path)
#         # print(all_k_path)
#
#         for idx_req in range(self.SimWarmup):
#             self.sim_main_operation(idx_req,all_k_path, 1)
# #            time.sleep(10000)
#
#         for idx_req in range(self.N_Req):
#             self.sim_main_operation(idx_req, all_k_path, 0)
# #            time.sleep(1)
#
#         print(len(self.req_in_blocked))
#         print('finish sim')
#
#         self.utilization = self.totalusedslot/(self.NumOfEdge*self.TotalNumofSlots*self.N_Req)
#         self.avghop =  self.avghop/(self.N_Req-self.NOB)
# #        print(cnt_gen_band)
# #        print(cnt_blk_band)
# #        print(cnt_blk_band/cnt_gen_band)
#
#         return self.NOB, self.cnt_gen_band, self.cnt_blk_band, self.cnt_suc_band, self.total_Gen_bandwidth, self.total_Block_bandwidth, self.cnt_suc_nos, self.utilization, self.avghop, self.routing_name, self.assingment_name, self.networktopo
#
#
#
#     def sim_main_operation(self, idx_req, all_k_path, warmup):
#
#         req = self.gene_request(idx_req, warmup)
#         self.release(self.cur_time)
#         # k_path = all_k_path[req.source][req.destination]
#
#         k_path = self.routing(req, all_k_path)
#
#         # for k in k_path:
#         #    print(k_path)
#         # print('ori ', id(all_k_path[req.source][req.destination][0]))
#         # print('copy ', id(k_path[0]))
#
#
# #        req.req_print()
#         if warmup != 1:
#             self.req_in_gen.append(req)
#             self.total_Gen_bandwidth += req.bandwidth
#             self.cnt_gen_band[int(req.bandwidth/self.SlotWidth)-1]+= 1
#
#             for i in self.G._data:
#                 for j in self.G._data[i]:
#                     self.totalusedslot += sum(int(u) for u, v in self.slot_info_2D[i][j])
#         flag=0
# #        print("=========================================================================================")
#         if idx_req % self.modeShow == 0 and warmup != 1 :
#             print(idx_req, 'th - req gen - ', self.cur_time, 'len(Q)', len(self.req_in_service),' NOB:', self.NOB)
#
#
#         for path in k_path:
#
#
# #            print('B', id(path))
# #            path = self.routing(req, path)
# #            print('A', id(path))
#
#
# #            flag=self.spec_assin_FF(path, req)
#
# #            flag=self.spec_assin_fair_TrunkReserv(path, req) #first fit
# #            flag=self.spec_assin_fair_TrunkReserv_v02(path, req) #best fit
# #            flag=self.spec_assin_fair_TrunkReserv_BPaware(path, req)
# #            flag=self.spec_assin_fair_TrunkReserv_BPaware_2D(path, req)
# #            flag=self.spec_assin_fair_TrunkReserv_BPaware_slots(path, req)
#
#             flag=self.spec_assin_fair_TrunkReserv_BPaware_FF(path, req)
#
# #            flag=self.spec_assin_fair_BPaware(path, req)
#
# #            flag=self.spec_assin_fair_proportion(path, req)
# #            flag=self.spec_assin_fair_fixed(path, req)
#
# #            flag=self.spec_assin_fair_TrunkReserv_v03(path, req) #best fit and half threshold
# #            flag=self.spec_assin_fair_EOSA(path, req) #best fit and half threshold
# #            flag=self.spec_assin_fair_NFSA(path, req)
# #            flag=self.spec_assin_fair_Proposed_SP(path, req) #best fit and half threshold
# #            flag=self.spec_assin_fair_Proposed_2D(path, req)
# #            flag=self.spec_assin_fair_Proposed_Holding(path, req)
#
#
# #            flag=self.spec_assin_fair_TrunkReserv_origin(path, req) #first fit
# #            flag=self.spec_assin_fair_TrunkReserv_origin_best(path, req) #best fit
#
# #            flag=self.spec_assin_fair_TrunkReserv_DThreshold(path, req)
#
# #            print(req.id, '  ',path['path'],'  ', req.nos)
#
#             if flag:
# #                req.req_print()
# #                for i in range(path['hop']):
# #                    fromnode = path['path'][i]
# #                    tonode = path['path'][i+1]
# #                    print(fromnode,' ',tonode)
# #                    data=''
# #                    for i in range(self.TotalNumofSlots):
# #                        data=data + str(self.slot_info_2D[fromnode][tonode][i][0])+' '
# #                        if i%20==19:
# #                            data+='\n'
# #                    print(data)
#
#
#
#                 self.updata_link_state(path, req)
#                 self.req_in_service.append(req)
#                 if warmup != 1:
#                     self.cnt_suc_band[int(req.bandwidth/self.SlotWidth)-1]+= 1
#                     self.cnt_suc_nos[req.nos-1]+= 1
#                     self.req_in_success.append(req)
#                     self.NO_Success += 1
#                     self.avghop += req.hop
#
# #                req.req_print()
# #                for i in range(path['hop']):
# #                    fromnode = path['path'][i]
# #                    tonode = path['path'][i+1]
# #                    print(fromnode,' ',tonode)
# #                    data=''
# #                    for i in range(self.TotalNumofSlots):
# #                        data=data + str(self.slot_info_2D[fromnode][tonode][i][0])+' '
# #                        if i%20==19:
# #                            data+='\n'
# #                    print(data)
#
#                 break
#
#         if flag != 1 and warmup != 1:
# #            req.req_print()
#             self.total_Block_bandwidth += req.bandwidth
#             self.cnt_blk_band[int(req.bandwidth/self.SlotWidth)-1]+= 1
#             self.NOB += 1
#             self.req_in_blocked.append(req)
#
#
#
#
#     def routing(self, req, all_k_path):
#         self.routing_name = 'Shortest path'
#         temp_path = copy.deepcopy(all_k_path[req.source][req.destination])
# #        print("%d-%d" % (req.source, req.destination))
#
# #        temp_path=self.BandFragRatio(temp_path)
# #        temp_path = self.min_hop(temp_path)
# #        temp_path = self.least_used_path(temp_path)
#
# #        temp_path = self.least_used(temp_path)
# #        temp_path=self.BandFragRatio_path(temp_path)
#
# #        req.req_print()
# #         print('b     ', temp_path)
#         temp_path.sort(key=lambda element:element['cost'])
# #        for k in temp_path:
# #            print(k['cost'])
# #         print('a     ',temp_path)
#         return temp_path
#
#     def BandFragRatio(self, kpaths):# each link sum
#         self.routing_name = 'BandFragRatio'
#         for k in kpaths:
#             cost=0
#             for i in range(k['hop']):
#                 unused_slot=0
#                 maxblock=0
#                 fromnode = k['path'][i]
#                 tonode = k['path'][i+1]
#                 unused_slot = 0
#                 blocksize=0
#                 for n in range(self.TotalNumofSlots):
#                     if self.slot_info_2D[fromnode][tonode][n][0] == 0:
#                         blocksize+=1
#                         unused_slot+=1
#
#                     elif self.slot_info_2D[fromnode][tonode][n][0] != 0:
#                         blocksize=0
#                     if maxblock<blocksize:
#                             maxblock = blocksize
#
#                 if unused_slot!=0:
#                     cost += (1-(maxblock/unused_slot))
#                 else:
#                     cost = self.INF
#
#             k['cost'] = cost
#
#         return kpaths
#
#     def BandFragRatio_path(self, kpaths):# each link sum
#         self.routing_name = 'BandFragRatio_path'
#         for k in kpaths:
#             cost=0
#             baseslot = self.base_slot(k)
#             maxblock=0
#             blocksize=0
#             unused_slot=0
#
#             for n in range(self.TotalNumofSlots):
#                 if baseslot[n][0] == 0:
#                     blocksize+=1
#                     unused_slot+=1
#
#                 elif baseslot[n][0]  >= 1:
#                     blocksize=0
#
#                 if maxblock<blocksize:
#                     maxblock = blocksize
#
#             if unused_slot!=0:
#                 cost = (1-(maxblock/unused_slot))
#             else:
#                 cost = self.INF
#
#             k['cost'] = cost
#
#
#         return kpaths
#
#     def min_hop(self, kpaths):
#         self.routing_name = 'min_hop'
# #        print('b  ', kpaths)
#         for k in kpaths:
#             k['cost'] = k['hop']
# #        print('a  ', kpaths)
#         return kpaths
#
#     def least_used(self, kpaths):
#         self.routing_name = 'least_used'
#         for k in kpaths:
# #            print( k)
#             cost=0
#             hap=0
#             for i in range(k['hop']):
#                 fromnode = k['path'][i]
#                 tonode = k['path'][i+1]
#                 hap += sum(int(u) for u, v in self.slot_info_2D[fromnode][tonode])
# #                print(fromnode, '  ', tonode,' ', hap)
#             cost = hap/(self.TotalNumofSlots*k['hop'])
#             k['cost'] = cost
#         return kpaths
#
#     def least_used_path(self, kpaths):
#         self.routing_name = 'least_used_path'
#         for k in kpaths:
# #            print( k)
#             cost=0
#             hap=0
#
#             baseslot = self.base_slot(k)
#
#             for i in range(self.TotalNumofSlots):
#                 if baseslot[i][0] > 0:
#                     hap+=1
#             cost = hap/(self.TotalNumofSlots)
#             k['cost'] = cost
# #            print(k)
#         return kpaths
#
#     def updata_link_state(self, path, req):
#         req.path = copy.deepcopy(path)
#         for i in range(len(path['path'])-1):
#             fromnode = path['path'][i]
#             tonode = path['path'][i+1]
# #            slot = self.slot_info[fromnode][tonode]
#             slot_2D = self.slot_info_2D[fromnode][tonode]
# #            print(fromnode, '-', tonode,' >> ', req.bandwidth)
# #            print(slot_2D)
# #            print('s: ',req.slot_start,' e:', req.slot_end)
#             for z in range(req.slot_start, req.slot_end+1):
# #                slot[z] = 1
# #                print('z: ',z, '  ', slot_2D[z])
#                 slot_2D[z][0] = 1
#                 slot_2D[z][1] = req.end_time
# #            print(fromnode, '-', tonode,' >> ', req.bandwidth)
# #            print(slot)
# #            req.req_print()
# #            print(slot_2D)
#
#     def gene_request(self, idx_req, warmup):
#         s = random.randint(1,self.NumOfNode)
#         d = random.randint(1,self.NumOfNode)
#         while s==d:
#             d = random.randint(1,self.NumOfNode)
#         b = random.randint(self.range_band_s, self.range_band_e)*self.SlotWidth
#         inter_arrival_time = round(random.expovariate(self.inter_arr_rate),4)
#         holding_time = 0
#         while holding_time == 0:
#             holding_time = round(np.random.exponential(self.holding_time),4)
#
#         self.cur_time = self.cur_time+inter_arrival_time
#
#         req = request.Request(idx_req, s, d, b, self.cur_time, holding_time, warmup)
# #        req = request.Request(idx_req, 1, 2, b, self.cur_time, holding_time, warmup)
#
#         return req
#
#
#     def precal_path(self):
#         pre_cal_path={}
#         cnt=0
#         dist=0
#         for i in range(1,self.NumOfNode+1):
#             pre_cal_path[i]={}
#             for j in range(1,self.NumOfNode+1):
#                 s = str(i)
#                 d = str(j)
#                 if s!=d:
#                     items = algorithms.ksp_yen(self.G, s, d, self.K)
#                     for it in items:
#                         it['dist']=it['cost']
#                         it['hop']=len(it['path'])-1
#                         cnt+=1
#                         dist+=it['dist']
# #                        print(it)
#                     pre_cal_path[i][j]=copy.deepcopy(items)
# #                    print('items  ',items)
#         return pre_cal_path
#
#
#     def spec_assin_FF(self, path, req):
#         self.assingment_name = 'first_fit'
#         flag = 0
#         numofslots = self.modul_format(path, req)
#         baseslot = self.base_slot(path)
#         flag=self.first_fit(flag, path, req, baseslot, numofslots)
#
#         return flag
#
#     def spec_assin_fair_proportion(self, path, req):
#         self.assingment_name = 'spec_assin_fair_proportion'
#         flag = 0
#         numofslots = self.modul_format(path, req)
#         baseslot = self.base_slot(path)
#         startslot=0
#         endslot= self.TotalNumofSlots-1
#         dedica_index={1:[0,5],2:[6,17],3:[18,34],4:[35,57],5:[58,86],6:[87,121],7:[122,162],8:[163,209],9:[210,261],10:[262,319]}
#         startslot=dedica_index[numofslots][0]
#         endslot= dedica_index[numofslots][1]
#         flag=self.first_fit_v_partition(flag, path, req, baseslot, numofslots, startslot, endslot) # 실제 인덱스를 범위로 0~9"""
#         return flag
#
#     def spec_assin_fair_fixed(self, path, req):
#         self.assingment_name = 'spec_assin_fair_fixed'
#         flag = 0
#         numofslots = self.modul_format(path, req)
#         baseslot = self.base_slot(path)
#         startslot=0
#         endslot= self.TotalNumofSlots-1
#         dedica_index={1:[0,31],2:[31,63],3:[64,95],4:[96,127],5:[128,159],6:[160,191],7:[192,223],8:[224,255],9:[256,287],10:[288,319]}
#         startslot=dedica_index[numofslots][0]
#         endslot= dedica_index[numofslots][1]
#         flag=self.first_fit_v_partition(flag, path, req, baseslot, numofslots, startslot, endslot) # 실제 인덱스를 범위로 0~9"""
#         return flag
#
#     def spec_assin_fair_TrunkReserv_origin(self, path, req): #fit fit
#         self.assingment_name = 'spec_assin_fair_TrunkReserv_origin- first fit'
#         flag = 0
#         numofslots = self.modul_format(path, req)
#         baseslot = self.base_slot(path)
#         threshold = self.range_band_e
#
#         startslot=0
#         endslot= self.TotalNumofSlots-1
#         baseslot = np.array(baseslot)
#
# #        print(type(baseslot ), baseslot.shape)
#         candi_SB = []
#
#         s=0
#         c=0
#         remain=0
#         for i in range(len(baseslot)):
#             if baseslot[i][0] == 0:
#                 if c == 0:
#                     s = i
#                 c += 1
#                 if i == self.TotalNumofSlots-1:
#                     candi_SB.append((s,s+c-1,c))
#                     remain += c
#             elif baseslot[i][0] > 0:
#                 if c!=0:
#                     candi_SB.append((s,s+c-1,c))
#                     remain += c
#                     c=0
# #        print('remain ', remain)
#         candi_SB.sort(key=lambda element:element[2])
# #        print(candi_SB)
# #        print(candi_SB[0][2])
#         if len(candi_SB) > 0 and remain>=threshold:
#             startslot=0
#             endslot= self.TotalNumofSlots-1
#             flag=self.first_fit_v_partition(flag, path, req, baseslot, numofslots, startslot, endslot) # 실제 인덱스를 범위로 0~9"""
#         else :
#             flag = 0
#
#         return flag
#
#     def spec_assin_fair_TrunkReserv_origin_best(self, path, req): #fit fit
#         self.assingment_name = 'spec_assin_fair_TrunkReserv_origin_best fit'
#         flag = 0
#         numofslots = self.modul_format(path, req)
#         baseslot = self.base_slot(path)
#         threshold = self.range_band_e
#
#         startslot=0
#         endslot= self.TotalNumofSlots-1
#         baseslot = np.array(baseslot)
#
# #        print(type(baseslot ), baseslot.shape)
#         candi_SB = []
#
#         s=0
#         c=0
#         remain=0
#         for i in range(len(baseslot)):
#             if baseslot[i][0] == 0:
#                 if c == 0:
#                     s = i
#                 c += 1
#                 if i == self.TotalNumofSlots-1:
#                     candi_SB.append((s,s+c-1,c))
#                     remain += c
#             elif baseslot[i][0] > 0:
#                 if c!=0:
#                     candi_SB.append((s,s+c-1,c))
#                     remain += c
#                     c=0
# #        print('remain ', remain)
#         candi_SB.sort(key=lambda element:element[2])
# #        print(candi_SB)
# #        print(candi_SB[0][2])
#         if len(candi_SB) > 0 and remain>=threshold:
#            for sb in candi_SB:
#                 if sb[-1] >= numofslots:
#                     startslot=sb[0]
#                     endslot= sb[1]
#                     flag=self.first_fit_v_partition(flag, path, req, baseslot, numofslots, startslot, endslot) # 실제 인덱스를 범위로 0~9"""
# #                    print(req.id,' : ',flag,' : ',sb, '  -  ',  numofslots)
#                     if flag == 1:
#                         break
#         else :
#             flag = 0
#
#         return flag
#
#
#     def spec_assin_fair_TrunkReserv(self, path, req): #fit fit
#         self.assingment_name = 'spec_assin_fair_TrunkReserv fit fit'
#         flag = 0
#         numofslots = self.modul_format(path, req)
#         baseslot = self.base_slot(path)
#         threshold = self.range_band_e
#
#         startslot=0
#         endslot= self.TotalNumofSlots-1
#
# #        print(baseslot )
#         candi_SB = []
#
#         s=0
#         c=0
#         for i in range(len(baseslot)):
#             if baseslot[i][0] == 0:
#                 if c == 0:
#                     s = i
#                 c += 1
#                 if i == self.TotalNumofSlots-1:
#                     candi_SB.append((s,s+c-1,c))
#             elif baseslot[i][0] > 0:
#                 if c!=0:
#                     candi_SB.append((s,s+c-1,c))
#                     c=0
#
#         candi_SB.sort(key=lambda element:element[2], reverse=True)
# #        print(candi_SB)
# #        print(candi_SB[0][2])
#         if len(candi_SB) > 0 and candi_SB[0][2]>=threshold:
#             startslot=0
#             endslot= self.TotalNumofSlots-1
#             flag=self.first_fit_v_partition(flag, path, req, baseslot, numofslots, startslot, endslot) # 실제 인덱스를 범위로 0~9"""
#         else :
#             flag = 0
#
#         return flag
#
#     def spec_assin_fair_TrunkReserv_v02(self, path, req): #best fit
#         self.assingment_name = 'spec_assin_fair_TrunkReserv best fit'
#         flag = 0
#         numofslots = self.modul_format(path, req)
#         baseslot = self.base_slot(path)
#         threshold = self.range_band_e
#
#         startslot=0
#         endslot= self.TotalNumofSlots-1
#
# #        print(baseslot )
#         candi_SB = []
#
#         s=0
#         c=0
#         for i in range(len(baseslot)):
#             if baseslot[i][0] == 0:
#                 if c == 0:
#                     s = i
#                 c += 1
#                 if i == self.TotalNumofSlots-1:
#                     candi_SB.append((s,s+c-1,c))
#             elif baseslot[i][0] > 0:
#                 if c!=0:
#                     candi_SB.append((s,s+c-1,c))
#                     c=0
#
#         candi_SB.sort(key=lambda element:element[2])
#
#         if len(candi_SB) > 0 and candi_SB[-1][2]>=threshold:
#             for sb in candi_SB:
#                 if sb[-1] >= numofslots:
#                     startslot=sb[0]
#                     endslot= sb[1]
#                     flag=self.first_fit_v_partition(flag, path, req, baseslot, numofslots, startslot, endslot) # 실제 인덱스를 범위로 0~9"""
# #                    print(req.id,' : ',flag,' : ',sb, '  -  ',  numofslots)
#                     if flag == 1:
#                         break
#         else :
#             flag = 0
#
#         return flag
#
#     def spec_assin_fair_TrunkReserv_BPaware(self, path, req): #best fit
#         self.assingment_name = 'spec_assin_fair_TrunkReserv_BPaware best fit'
#         flag = 0
#         numofslots = self.modul_format(path, req)
#         baseslot = self.base_slot(path)
#         threshold = self.range_band_e
#
#         if self.NOB+self.NO_Success == 0:
#             BP_Total = 0
#         else:
#             BP_Total = self.NOB/(self.NOB+self.NO_Success)
#
#         gen_class = self.cnt_gen_band[int(req.bandwidth/self.SlotWidth)-1]
#         block_class = self.cnt_blk_band[int(req.bandwidth/self.SlotWidth)-1]
#
#         if gen_class == 0:
#             BP_Class = 0
#         else:
#             BP_Class = block_class/gen_class
#
# #        print(BP_Class, '  ',BP_Total)
#
#         startslot=0
#         endslot= self.TotalNumofSlots-1
#
# #        print(baseslot )
#         candi_SB = []
#         s=0
#         c=0
#         for i in range(len(baseslot)):
#             if baseslot[i][0] == 0:
#                 if c == 0:
#                     s = i
#                 c += 1
#                 if i == self.TotalNumofSlots-1:
#                     candi_SB.append((s,s+c-1,c))
#             elif baseslot[i][0] > 0:
#                 if c!=0:
#                     candi_SB.append((s,s+c-1,c))
#                     c=0
#
#         candi_SB.sort(key=lambda element:element[2])
#
#         if len(candi_SB) > 0 and candi_SB[-1][2]>=threshold:
#             for sb in candi_SB:
#                 if sb[-1] >= numofslots:
#                     startslot=sb[0]
#                     endslot= sb[1]
#                     flag=self.first_fit_v_partition(flag, path, req, baseslot, numofslots, startslot, endslot) # 실제 인덱스를 범위로 0~9"""
# #                    print(req.id,' : ',flag,' : ',sb, '  -  ',  numofslots)
#                     if flag == 1:
#                         break
#         elif len(candi_SB) > 0 and BP_Total < BP_Class :
#             for sb in candi_SB:
#                 if sb[-1] >= numofslots:
#                     startslot=sb[0]
#                     endslot= sb[1]
#                     flag=self.first_fit_v_partition(flag, path, req, baseslot, numofslots, startslot, endslot) # 실제 인덱스를 범위로 0~9"""
# #                    print(req.id,' : ',flag,' : ',sb, '  -  ',  numofslots)
#                     if flag == 1:
#                         break
#         else :
#             flag = 0
#
#         return flag
#
#     def spec_assin_fair_TrunkReserv_BPaware_FF(self, path, req): #best fit
#         self.assingment_name = 'spec_assin_fair_TrunkReserv_BPaware first fit'
#         flag = 0
#         numofslots = self.modul_format(path, req)
#         baseslot = self.base_slot(path)
#         threshold = self.range_band_e
#
#         if self.NOB+self.NO_Success == 0:
#             BP_Total = 0
#         else:
#             BP_Total = self.NOB/(self.NOB+self.NO_Success)
#
#         gen_class = self.cnt_gen_band[int(req.bandwidth/self.SlotWidth)-1]
#         block_class = self.cnt_blk_band[int(req.bandwidth/self.SlotWidth)-1]
#
#         if gen_class == 0:
#             BP_Class = 0
#         else:
#             BP_Class = block_class/gen_class
#
# #        print(BP_Class, '  ',BP_Total)
#
#         startslot=0
#         endslot= self.TotalNumofSlots-1
#
# #        print(baseslot )
#         candi_SB = []
#         s=0
#         c=0
#         maxsize=0
#         for i in range(len(baseslot)):
#             if baseslot[i][0] == 0:
#                 if c == 0:
#                     s = i
#                 c += 1
#                 if i == self.TotalNumofSlots-1:
#                     candi_SB.append((s,s+c-1,c))
#                     if maxsize < c:
#                         maxsize = c
#             elif baseslot[i][0] > 0:
#                 if c!=0:
#                     candi_SB.append((s,s+c-1,c))
#                     if maxsize < c:
#                         maxsize = c
#                     c=0
#
# #        candi_SB.sort(key=lambda element:element[2])
#
#         if len(candi_SB) > 0 and maxsize>=threshold:
#             for sb in candi_SB:
#                 if sb[-1] >= numofslots:
#                     startslot=sb[0]
#                     endslot= sb[1]
#                     flag=self.first_fit_v_partition(flag, path, req, baseslot, numofslots, startslot, endslot) # 실제 인덱스를 범위로 0~9"""
# #                    print(req.id,' : ',flag,' : ',sb, '  -  ',  numofslots)
#                     if flag == 1:
#                         break
#         elif len(candi_SB) > 0 and BP_Total < BP_Class :
#             for sb in candi_SB:
#                 if sb[-1] >= numofslots:
#                     startslot=sb[0]
#                     endslot= sb[1]
#                     flag=self.first_fit_v_partition(flag, path, req, baseslot, numofslots, startslot, endslot) # 실제 인덱스를 범위로 0~9"""
# #                    print(req.id,' : ',flag,' : ',sb, '  -  ',  numofslots)
#                     if flag == 1:
#                         break
#         else :
#             flag = 0
#
#         return flag
#
#     def spec_assin_fair_BPaware(self, path, req): #best fit
#         self.assingment_name = 'spec_assin_fair_BPaware'
#         flag = 0
#         numofslots = self.modul_format(path, req)
#         baseslot = self.base_slot(path)
#         threshold = self.range_band_e
#
#         if self.NOB+self.NO_Success == 0:
#             BP_Total = 0
#         else:
#             BP_Total = self.NOB/(self.NOB+self.NO_Success)
#
#         gen_class = self.cnt_gen_band[int(req.bandwidth/self.SlotWidth)-1]
#         block_class = self.cnt_blk_band[int(req.bandwidth/self.SlotWidth)-1]
#
#         if gen_class == 0:
#             BP_Class = 0
#         else:
#             BP_Class = block_class/gen_class
#
#
# #        print(BP_Class, '  ',BP_Total)
#
#         startslot=0
#         endslot= self.TotalNumofSlots-1
#
# #        print(baseslot )
#         candi_SB = []
#         s=0
#         c=0
#         for i in range(len(baseslot)):
#             if baseslot[i][0] == 0:
#                 if c == 0:
#                     s = i
#                 c += 1
#                 if i == self.TotalNumofSlots-1:
#                     candi_SB.append((s,s+c-1,c))
#             elif baseslot[i][0] > 0:
#                 if c!=0:
#                     candi_SB.append((s,s+c-1,c))
#                     c=0
#
#         candi_SB.sort(key=lambda element:element[2])
#
#         if len(candi_SB) > 0 and BP_Total <= BP_Class :
#             for sb in candi_SB:
#                 if sb[-1] >= numofslots:
#                     startslot=sb[0]
#                     endslot= sb[1]
#                     flag=self.first_fit_v_partition(flag, path, req, baseslot, numofslots, startslot, endslot) # 실제 인덱스를 범위로 0~9"""
# #                    print(req.id,' : ',flag,' : ',sb, '  -  ',  numofslots)
#                     if flag == 1:
#                         break
#         else :
# #            print('else')
#             flag = 0
#
#         return flag
#
#     def spec_assin_fair_TrunkReserv_BPaware_2D(self, path, req): #best fit
#         self.assingment_name = 'spec_assin_fair_TrunkReserv_BPaware_2D'
#         flag = 0
#         numofslots = self.modul_format(path, req)
#         baseslot = self.base_slot(path)
#         threshold = self.range_band_e
#         startslot=0
#         endslot= self.TotalNumofSlots-1
#         hop = len(path['path'])-1
#
#         if self.NOB+self.NO_Success == 0:
#             BP_Total = 0
#         else:
#             BP_Total = self.NOB/(self.NOB+self.NO_Success)
#
#         gen_class = self.cnt_gen_band[int(req.bandwidth/self.SlotWidth)-1]
#         block_class = self.cnt_blk_band[int(req.bandwidth/self.SlotWidth)-1]
#
#         if gen_class == 0:
#             BP_Class = 0
#         else:
#             BP_Class = block_class/gen_class
#
#         endtime = req.end_time
#         holdtime = req.holding_time
#         if holdtime == 0:
#             print('holdtime', holdtime)
#
#         candi_SB = []
#
#         s=0
#         c=0
#         for i in range(len(baseslot)):
#             if baseslot[i][0] == 0:
#                 if c == 0:
#                     s = i
#                 c += 1
#                 if i == self.TotalNumofSlots-1:
#                     candi_SB.append((s,s+c-1,c))# start, end, count,
#             elif baseslot[i][0] > 0:
#                 if c!=0:
#                     candi_SB.append((s,s+c-1,c))
#                     c=0
#
#         candi_SB.sort(key=lambda element:element[2])
# #        print('CAND ', candi_SB)
#         if len(candi_SB) > 0 and candi_SB[-1][2]>=threshold:
#             SpBlock = []
#             for sb in candi_SB:
#                 if sb[-1] >= numofslots:
#                     cost_S = 0
#                     cost_T = 0
#                     sb_start = sb[0]
#                     sb_end = sb[1]
#                     if sb_start == 0:
#                         cost_S = (hop-baseslot[sb_start+numofslots][0])/hop
#                         cost_T = abs(endtime-baseslot[sb_start+numofslots][1])/holdtime
#                         if cost_T >1:
#                             cost_T = 1
#                     elif sb_start>0 and sb_start+numofslots < self.TotalNumofSlots:
#                         cost_S = ((hop-baseslot[sb_start-1][0])+(hop-baseslot[sb_start+numofslots][0]))/hop
#                         cost_T_l = abs(endtime-baseslot[sb_start-1][1])/holdtime
#                         cost_T_r = abs(endtime-baseslot[sb_start+numofslots][1])/holdtime
#                         if cost_T_l >1:
#                             cost_T_l = 1
#                         if cost_T_r >1:
#                             cost_T_r = 1
#                         cost_T = cost_T_l+cost_T_r
#                     elif sb_start+numofslots == self.TotalNumofSlots:
#                         cost_S = (hop-baseslot[sb_start-1][0])/hop
#                         cost_T = abs(endtime-baseslot[sb_start-1][1])/holdtime
#                         if cost_T >1:
#                             cost_T = 1
#
#                     SpBlock.append((sb_start, sb_start+numofslots-1, cost_S, cost_T, cost_S+cost_T ))
#
#                     cost_S = 0
#                     cost_T = 0
#                     sb_start = sb[1]-numofslots+1
#                     sb_end = sb[1]
#                     if sb_start == 0:
#                         cost_S = (hop-baseslot[sb_start+numofslots][0])/hop
#                         cost_T = abs(endtime-baseslot[sb_start+numofslots][1])/holdtime
#                         if cost_T >1:
#                             cost_T = 1
#                     elif sb_start>0 and sb_start+numofslots < self.TotalNumofSlots:
#                         cost_S = ((hop-baseslot[sb_start-1][0])+(hop-baseslot[sb_start+numofslots][0]))/hop
#                         cost_T_l = abs(endtime-baseslot[sb_start-1][1])/holdtime
#                         cost_T_r = abs(endtime-baseslot[sb_start+numofslots][1])/holdtime
#                         if cost_T_l >1:
#                             cost_T_l = 1
#                         if cost_T_r >1:
#                             cost_T_r = 1
#                         cost_T = cost_T_l+cost_T_r
#                     elif sb_start+numofslots == self.TotalNumofSlots:
#                         cost_S = (hop-baseslot[sb_start-1][0])/hop
#                         cost_T = abs(endtime-baseslot[sb_start-1][1])/holdtime
#                         if cost_T >1:
#                             cost_T = 1
#
#                     SpBlock.append((sb_start, sb_end, cost_S, cost_T,cost_S+cost_T ))
#
#             SpBlock.sort(key=lambda element:element[-1])
# #            print('B ', SpBlock)
#             for sb in SpBlock:
#
#                 startslot=sb[0]
#                 endslot= sb[1]
#                 flag=self.first_fit_v_partition(flag, path, req, baseslot, numofslots, startslot, endslot) # 실제 인덱스를 범위로 0~9"""
# #                print(req.id,' : ',flag,' : ',sb, '  -  ',  numofslots)
#                 if flag == 1:
#                     break
#         elif len(candi_SB) > 0 and BP_Total < BP_Class :
#             SpBlock = []
#             for sb in candi_SB:
#                 if sb[-1] >= numofslots:
#                     cost_S = 0
#                     cost_T = 0
#                     sb_start = sb[0]
#                     sb_end = sb[1]
#                     if sb_start == 0:
#                         cost_S = (hop-baseslot[sb_start+numofslots][0])/hop
#                         cost_T = abs(endtime-baseslot[sb_start+numofslots][1])/holdtime
#                         if cost_T >1:
#                             cost_T = 1
#                     elif sb_start>0 and sb_start+numofslots < self.TotalNumofSlots:
#                         cost_S = ((hop-baseslot[sb_start-1][0])+(hop-baseslot[sb_start+numofslots][0]))/hop
#                         cost_T_l = abs(endtime-baseslot[sb_start-1][1])/holdtime
#                         cost_T_r = abs(endtime-baseslot[sb_start+numofslots][1])/holdtime
#                         if cost_T_l >1:
#                             cost_T_l = 1
#                         if cost_T_r >1:
#                             cost_T_r = 1
#                         cost_T = cost_T_l+cost_T_r
#                     elif sb_start+numofslots == self.TotalNumofSlots:
#                         cost_S = (hop-baseslot[sb_start-1][0])/hop
#                         cost_T = abs(endtime-baseslot[sb_start-1][1])/holdtime
#                         if cost_T >1:
#                             cost_T = 1
#
#                     SpBlock.append((sb_start, sb_start+numofslots-1, cost_S, cost_T, cost_S+cost_T ))
#
#                     cost_S = 0
#                     cost_T = 0
#                     sb_start = sb[1]-numofslots+1
#                     sb_end = sb[1]
#                     if sb_start == 0:
#                         cost_S = (hop-baseslot[sb_start+numofslots][0])/hop
#                         cost_T = abs(endtime-baseslot[sb_start+numofslots][1])/holdtime
#                         if cost_T >1:
#                             cost_T = 1
#                     elif sb_start>0 and sb_start+numofslots < self.TotalNumofSlots:
#                         cost_S = ((hop-baseslot[sb_start-1][0])+(hop-baseslot[sb_start+numofslots][0]))/hop
#                         cost_T_l = abs(endtime-baseslot[sb_start-1][1])/holdtime
#                         cost_T_r = abs(endtime-baseslot[sb_start+numofslots][1])/holdtime
#                         if cost_T_l >1:
#                             cost_T_l = 1
#                         if cost_T_r >1:
#                             cost_T_r = 1
#                         cost_T = cost_T_l+cost_T_r
#                     elif sb_start+numofslots == self.TotalNumofSlots:
#                         cost_S = (hop-baseslot[sb_start-1][0])/hop
#                         cost_T = abs(endtime-baseslot[sb_start-1][1])/holdtime
#                         if cost_T >1:
#                             cost_T = 1
#
#                     SpBlock.append((sb_start, sb_end, cost_S, cost_T,cost_S+cost_T ))
#
#             SpBlock.sort(key=lambda element:element[-1])
# #            print('B ', SpBlock)
#             for sb in SpBlock:
#
#                 startslot=sb[0]
#                 endslot= sb[1]
#                 flag=self.first_fit_v_partition(flag, path, req, baseslot, numofslots, startslot, endslot) # 실제 인덱스를 범위로 0~9"""
# #                print(req.id,' : ',flag,' : ',sb, '  -  ',  numofslots)
#                 if flag == 1:
#                     break
#
#
#
#         else :
#             flag = 0
#
#         return flag
#
#     def spec_assin_fair_TrunkReserv_BPaware_slots(self, path, req): #best fit
#         self.assingment_name = 'spec_assin_fair_TrunkReserv_BPaware_slots'
#         flag = 0
#         numofslots = self.modul_format(path, req)
#         baseslot = self.base_slot(path)
#         threshold = self.range_band_e
#         startslot=0
#         endslot= self.TotalNumofSlots-1
#         hop = len(path['path'])-1
#
#         if self.NOB+self.NO_Success == 0:
#             BP_Total = 0
#         else:
#             BP_Total = self.NOB/(self.NOB+self.NO_Success)
#
#         gen_class = self.cnt_gen_band[int(req.bandwidth/self.SlotWidth)-1]
#         block_class = self.cnt_blk_band[int(req.bandwidth/self.SlotWidth)-1]
#
#         if gen_class == 0:
#             BP_Class = 0
#         else:
#             BP_Class = block_class/gen_class
#
#         endtime = req.end_time
#         holdtime = req.holding_time
#         if holdtime == 0:
#             print('holdtime', holdtime)
#
#         candi_SB = []
#
#         s=0
#         c=0
#         for i in range(len(baseslot)):
#             if baseslot[i][0] == 0:
#                 if c == 0:
#                     s = i
#                 c += 1
#                 if i == self.TotalNumofSlots-1:
#                     candi_SB.append((s,s+c-1,c))# start, end, count,
#             elif baseslot[i][0] > 0:
#                 if c!=0:
#                     candi_SB.append((s,s+c-1,c))
#                     c=0
#
#         candi_SB.sort(key=lambda element:element[2])
# #        print('CAND ', candi_SB)
#         if len(candi_SB) > 0 and candi_SB[-1][2]>=threshold:
#             SpBlock = []
#             for sb in candi_SB:
#                 if sb[-1] >= numofslots:
#                     cost_S = 0
#                     cost_T = 0
#                     sb_start = sb[0]
#                     sb_end = sb[1]
#                     if sb_start == 0:
#                         cost_S = (hop-baseslot[sb_start+numofslots][0])/hop
#                         cost_T = abs(endtime-baseslot[sb_start+numofslots][1])/holdtime
#                         if cost_T >1:
#                             cost_T = 1
#                     elif sb_start>0 and sb_start+numofslots < self.TotalNumofSlots:
#                         cost_S = ((hop-baseslot[sb_start-1][0])+(hop-baseslot[sb_start+numofslots][0]))/hop
#                         cost_T_l = abs(endtime-baseslot[sb_start-1][1])/holdtime
#                         cost_T_r = abs(endtime-baseslot[sb_start+numofslots][1])/holdtime
#                         if cost_T_l >1:
#                             cost_T_l = 1
#                         if cost_T_r >1:
#                             cost_T_r = 1
#                         cost_T = cost_T_l+cost_T_r
#                     elif sb_start+numofslots == self.TotalNumofSlots:
#                         cost_S = (hop-baseslot[sb_start-1][0])/hop
#                         cost_T = abs(endtime-baseslot[sb_start-1][1])/holdtime
#                         if cost_T >1:
#                             cost_T = 1
#
#                     SpBlock.append((sb_start, sb_start+numofslots-1, cost_S, cost_T, cost_S ))
#
#                     cost_S = 0
#                     cost_T = 0
#                     sb_start = sb[1]-numofslots+1
#                     sb_end = sb[1]
#                     if sb_start == 0:
#                         cost_S = (hop-baseslot[sb_start+numofslots][0])/hop
#                         cost_T = abs(endtime-baseslot[sb_start+numofslots][1])/holdtime
#                         if cost_T >1:
#                             cost_T = 1
#                     elif sb_start>0 and sb_start+numofslots < self.TotalNumofSlots:
#                         cost_S = ((hop-baseslot[sb_start-1][0])+(hop-baseslot[sb_start+numofslots][0]))/hop
#                         cost_T_l = abs(endtime-baseslot[sb_start-1][1])/holdtime
#                         cost_T_r = abs(endtime-baseslot[sb_start+numofslots][1])/holdtime
#                         if cost_T_l >1:
#                             cost_T_l = 1
#                         if cost_T_r >1:
#                             cost_T_r = 1
#                         cost_T = cost_T_l+cost_T_r
#                     elif sb_start+numofslots == self.TotalNumofSlots:
#                         cost_S = (hop-baseslot[sb_start-1][0])/hop
#                         cost_T = abs(endtime-baseslot[sb_start-1][1])/holdtime
#                         if cost_T >1:
#                             cost_T = 1
#
#                     SpBlock.append((sb_start, sb_end, cost_S, cost_T,cost_S ))
#
#             SpBlock.sort(key=lambda element:element[-1])
# #            print('B ', SpBlock)
#             for sb in SpBlock:
#
#                 startslot=sb[0]
#                 endslot= sb[1]
#                 flag=self.first_fit_v_partition(flag, path, req, baseslot, numofslots, startslot, endslot) # 실제 인덱스를 범위로 0~9"""
# #                print(req.id,' : ',flag,' : ',sb, '  -  ',  numofslots)
#                 if flag == 1:
#                     break
#         elif len(candi_SB) > 0 and BP_Total < BP_Class :
#             SpBlock = []
#             for sb in candi_SB:
#                 if sb[-1] >= numofslots:
#                     cost_S = 0
#                     cost_T = 0
#                     sb_start = sb[0]
#                     sb_end = sb[1]
#                     if sb_start == 0:
#                         cost_S = (hop-baseslot[sb_start+numofslots][0])/hop
#                         cost_T = abs(endtime-baseslot[sb_start+numofslots][1])/holdtime
#                         if cost_T >1:
#                             cost_T = 1
#                     elif sb_start>0 and sb_start+numofslots < self.TotalNumofSlots:
#                         cost_S = ((hop-baseslot[sb_start-1][0])+(hop-baseslot[sb_start+numofslots][0]))/hop
#                         cost_T_l = abs(endtime-baseslot[sb_start-1][1])/holdtime
#                         cost_T_r = abs(endtime-baseslot[sb_start+numofslots][1])/holdtime
#                         if cost_T_l >1:
#                             cost_T_l = 1
#                         if cost_T_r >1:
#                             cost_T_r = 1
#                         cost_T = cost_T_l+cost_T_r
#                     elif sb_start+numofslots == self.TotalNumofSlots:
#                         cost_S = (hop-baseslot[sb_start-1][0])/hop
#                         cost_T = abs(endtime-baseslot[sb_start-1][1])/holdtime
#                         if cost_T >1:
#                             cost_T = 1
#
#                     SpBlock.append((sb_start, sb_start+numofslots-1, cost_S, cost_T, cost_S ))
#
#                     cost_S = 0
#                     cost_T = 0
#                     sb_start = sb[1]-numofslots+1
#                     sb_end = sb[1]
#                     if sb_start == 0:
#                         cost_S = (hop-baseslot[sb_start+numofslots][0])/hop
#                         cost_T = abs(endtime-baseslot[sb_start+numofslots][1])/holdtime
#                         if cost_T >1:
#                             cost_T = 1
#                     elif sb_start>0 and sb_start+numofslots < self.TotalNumofSlots:
#                         cost_S = ((hop-baseslot[sb_start-1][0])+(hop-baseslot[sb_start+numofslots][0]))/hop
#                         cost_T_l = abs(endtime-baseslot[sb_start-1][1])/holdtime
#                         cost_T_r = abs(endtime-baseslot[sb_start+numofslots][1])/holdtime
#                         if cost_T_l >1:
#                             cost_T_l = 1
#                         if cost_T_r >1:
#                             cost_T_r = 1
#                         cost_T = cost_T_l+cost_T_r
#                     elif sb_start+numofslots == self.TotalNumofSlots:
#                         cost_S = (hop-baseslot[sb_start-1][0])/hop
#                         cost_T = abs(endtime-baseslot[sb_start-1][1])/holdtime
#                         if cost_T >1:
#                             cost_T = 1
#
#                     SpBlock.append((sb_start, sb_end, cost_S, cost_T,cost_S ))
#
#             SpBlock.sort(key=lambda element:element[-1])
# #            print('B ', SpBlock)
#             for sb in SpBlock:
#
#                 startslot=sb[0]
#                 endslot= sb[1]
#                 flag=self.first_fit_v_partition(flag, path, req, baseslot, numofslots, startslot, endslot) # 실제 인덱스를 범위로 0~9"""
# #                print(req.id,' : ',flag,' : ',sb, '  -  ',  numofslots)
#                 if flag == 1:
#                     break
#
#
#
#         else :
#             flag = 0
#
#         return flag
#
#     def spec_assin_fair_TrunkReserv_v03(self, path, req): #best fit and half threshold
#         self.assingment_name = 'spec_assin_fair_TrunkReserv best fit and half threshold'
#         flag = 0
#         numofslots = self.modul_format(path, req)
#         baseslot = self.base_slot(path)
#         threshold = self.range_band_e/2
#         startslot=0
#         endslot= self.TotalNumofSlots-1
# #        print(baseslot )
#         candi_SB = []
#
#         s=0
#         c=0
#         for i in range(len(baseslot)):
#             if baseslot[i][0] == 0:
#                 if c == 0:
#                     s = i
#                 c += 1
#                 if i == self.TotalNumofSlots-1:
#                     candi_SB.append((s,s+c-1,c))
#             elif baseslot[i][0] > 0:
#                 if c!=0:
#                     candi_SB.append((s,s+c-1,c))
#                     c=0
#
#         candi_SB.sort(key=lambda element:element[2])
#         if len(candi_SB) > 0 and candi_SB[-1][2]>=threshold:
#             for sb in candi_SB:
#                 if sb[-1] >= numofslots:
#                     startslot=sb[0]
#                     endslot= sb[1]
#                     flag=self.first_fit_v_partition(flag, path, req, baseslot, numofslots, startslot, endslot) # 실제 인덱스를 범위로 0~9"""
# #                    print(req.id,' : ',flag,' : ',sb, '  -  ',  numofslots)
#                     if flag == 1:
#                         break
#         else :
#             flag = 0
#
#         return flag
#
#     def spec_assin_fair_TrunkReserv_DThreshold(self, path, req): #best fit and half threshold
#         self.assingment_name = 'spec_assin_fair_TrunkReserv_DThreshold best fit and half threshold'
#         flag = 0
#         numofslots = self.modul_format(path, req)
#         baseslot = self.base_slot(path)
#         threshold = self.range_band_e/2
#         startslot=0
#         endslot= self.TotalNumofSlots-1
# #        print(baseslot )
#         candi_SB = []
#
#         s=0
#         c=0
#         for i in range(len(baseslot)):
#             if baseslot[i][0] == 0:
#                 if c == 0:
#                     s = i
#                 c += 1
#                 if i == self.TotalNumofSlots-1:
#                     candi_SB.append((s,s+c-1,c))
#             elif baseslot[i][0] > 0:
#                 if c!=0:
#                     candi_SB.append((s,s+c-1,c))
#                     c=0
#
#         candi_SB.sort(key=lambda element:element[2])
#         if len(candi_SB) > 0 and candi_SB[-1][2]>=threshold:
#             for sb in candi_SB:
#                 if sb[-1] >= numofslots:
#                     startslot=sb[0]
#                     endslot= sb[1]
#                     flag=self.first_fit_v_partition(flag, path, req, baseslot, numofslots, startslot, endslot) # 실제 인덱스를 범위로 0~9"""
# #                    print(req.id,' : ',flag,' : ',sb, '  -  ',  numofslots)
#                     if flag == 1:
#                         break
#         else :
#             flag = 0
#
#         return flag
#
#     def spec_assin_fair_EOSA(self, path, req): #
#         self.assingment_name = 'spec_assin_fair_EOSA'
#         flag = 0
#         numofslots = self.modul_format(path, req)
#         baseslot = self.base_slot(path)
#         startslot=0
#         endslot= self.TotalNumofSlots-1
#         flag=self.EOSA(flag, path, req, baseslot, numofslots, startslot, endslot)
#         return flag
#
#     def spec_assin_fair_NFSA(self, path, req): #
#         self.assingment_name = 'spec_assin_fair_NFSA'
#         flag = 0
#         numofslots = self.modul_format(path, req)
#         baseslot = self.base_slot(path)
#         startslot=0
#         endslot= self.TotalNumofSlots-1
#         flag=self.NFSA(flag, path, req, baseslot, numofslots, startslot, endslot)
#         return flag
#
#
#     def spec_assin_fair_Proposed_SP(self, path, req): #best fit
#         self.assingment_name = 'spec_assin_fair_Proposed_SP'
#         flag = 0
#         numofslots = self.modul_format(path, req)
#         baseslot = self.base_slot(path)
#         threshold = self.range_band_e
#         startslot=0
#         endslot= self.TotalNumofSlots-1
#         hop = len(path['path'])-1
#         candi_SB = []
#
#         s=0
#         c=0
#         for i in range(len(baseslot)):
#             if baseslot[i][0] == 0:
#                 if c == 0:
#                     s = i
#                 c += 1
#                 if i == self.TotalNumofSlots-1:
#                     candi_SB.append((s,s+c-1,c))# start, end, count,
#             elif baseslot[i][0] > 0:
#                 if c!=0:
#                     candi_SB.append((s,s+c-1,c))
#                     c=0
#
#         candi_SB.sort(key=lambda element:element[2])
# #        print('CAND ', candi_SB)
#         if len(candi_SB) > 0 and candi_SB[-1][2]>=threshold:
#             SpBlock = []
#             for sb in candi_SB:
#                 if sb[-1] >= numofslots:
#                     cost_S = 0
#                     cost_T = 0
#                     sb_start = sb[0]
#                     sb_end = sb[1]
#                     if sb_start == 0:
#                         cost_S = (hop-baseslot[sb_start+numofslots][0])/hop
#                     elif sb_start>0 and sb_start+numofslots < self.TotalNumofSlots:
#                         cost_S = ((hop-baseslot[sb_start-1][0])+(hop-baseslot[sb_start+numofslots][0]))/hop
#                     elif sb_start+numofslots == self.TotalNumofSlots:
#                         cost_S = (hop-baseslot[sb_start-1][0])/hop
#
#                     SpBlock.append((sb_start, sb_start+numofslots-1, cost_S))
#
#                     cost_S = 0
#                     cost_T = 0
#                     sb_start = sb[1]-numofslots+1
#                     sb_end = sb[1]
#                     if sb_start == 0:
#                         cost_S = (hop-baseslot[sb_start+numofslots][0])/hop
#                     elif sb_start>0 and sb_start+numofslots < self.TotalNumofSlots:
#                         cost_S = ((hop-baseslot[sb_start-1][0])+(hop-baseslot[sb_start+numofslots][0]))/hop
#                     elif sb_start+numofslots == self.TotalNumofSlots:
#                         cost_S = (hop-baseslot[sb_start-1][0])/hop
#
#                     SpBlock.append((sb_start, sb_end, cost_S))
#
#             SpBlock.sort(key=lambda element:element[2])
# #            print('B ', SpBlock)
#             for sb in SpBlock:
#
#                 startslot=sb[0]
#                 endslot= sb[1]
#                 flag=self.first_fit_v_partition(flag, path, req, baseslot, numofslots, startslot, endslot) # 실제 인덱스를 범위로 0~9"""
# #                print(req.id,' : ',flag,' : ',sb, '  -  ',  numofslots)
#                 if flag == 1:
#                     break
#         else :
#             flag = 0
#
#         return flag
#
#     def spec_assin_fair_Proposed_2D(self, path, req): #best fit
#         self.assingment_name = 'spec_assin_fair_Proposed_2D'
#         flag = 0
#         numofslots = self.modul_format(path, req)
#         baseslot = self.base_slot(path)
#         threshold = self.range_band_e
#         startslot=0
#         endslot= self.TotalNumofSlots-1
#         hop = len(path['path'])-1
#         endtime = req.end_time
#         holdtime = req.holding_time
#         if holdtime == 0:
#             print('holdtime', holdtime)
#
#         candi_SB = []
#
#         s=0
#         c=0
#         for i in range(len(baseslot)):
#             if baseslot[i][0] == 0:
#                 if c == 0:
#                     s = i
#                 c += 1
#                 if i == self.TotalNumofSlots-1:
#                     candi_SB.append((s,s+c-1,c))# start, end, count,
#             elif baseslot[i][0] > 0:
#                 if c!=0:
#                     candi_SB.append((s,s+c-1,c))
#                     c=0
#
#         candi_SB.sort(key=lambda element:element[2])
# #        print('CAND ', candi_SB)
#         if len(candi_SB) > 0 and candi_SB[-1][2]>=threshold:
#             SpBlock = []
#             for sb in candi_SB:
#                 if sb[-1] >= numofslots:
#                     cost_S = 0
#                     cost_T = 0
#                     sb_start = sb[0]
#                     sb_end = sb[1]
#                     if sb_start == 0:
#                         cost_S = (hop-baseslot[sb_start+numofslots][0])/hop
#                         cost_T = abs(endtime-baseslot[sb_start+numofslots][1])/holdtime
#                         if cost_T >1:
#                             cost_T = 1
#                     elif sb_start>0 and sb_start+numofslots < self.TotalNumofSlots:
#                         cost_S = ((hop-baseslot[sb_start-1][0])+(hop-baseslot[sb_start+numofslots][0]))/hop
#                         cost_T_l = abs(endtime-baseslot[sb_start-1][1])/holdtime
#                         cost_T_r = abs(endtime-baseslot[sb_start+numofslots][1])/holdtime
#                         if cost_T_l >1:
#                             cost_T_l = 1
#                         if cost_T_r >1:
#                             cost_T_r = 1
#                         cost_T = cost_T_l+cost_T_r
#                     elif sb_start+numofslots == self.TotalNumofSlots:
#                         cost_S = (hop-baseslot[sb_start-1][0])/hop
#                         cost_T = abs(endtime-baseslot[sb_start-1][1])/holdtime
#                         if cost_T >1:
#                             cost_T = 1
#
#                     SpBlock.append((sb_start, sb_start+numofslots-1, cost_S, cost_T, cost_S+cost_T ))
#
#                     cost_S = 0
#                     cost_T = 0
#                     sb_start = sb[1]-numofslots+1
#                     sb_end = sb[1]
#                     if sb_start == 0:
#                         cost_S = (hop-baseslot[sb_start+numofslots][0])/hop
#                         cost_T = abs(endtime-baseslot[sb_start+numofslots][1])/holdtime
#                         if cost_T >1:
#                             cost_T = 1
#                     elif sb_start>0 and sb_start+numofslots < self.TotalNumofSlots:
#                         cost_S = ((hop-baseslot[sb_start-1][0])+(hop-baseslot[sb_start+numofslots][0]))/hop
#                         cost_T_l = abs(endtime-baseslot[sb_start-1][1])/holdtime
#                         cost_T_r = abs(endtime-baseslot[sb_start+numofslots][1])/holdtime
#                         if cost_T_l >1:
#                             cost_T_l = 1
#                         if cost_T_r >1:
#                             cost_T_r = 1
#                         cost_T = cost_T_l+cost_T_r
#                     elif sb_start+numofslots == self.TotalNumofSlots:
#                         cost_S = (hop-baseslot[sb_start-1][0])/hop
#                         cost_T = abs(endtime-baseslot[sb_start-1][1])/holdtime
#                         if cost_T >1:
#                             cost_T = 1
#
#                     SpBlock.append((sb_start, sb_end, cost_S, cost_T,cost_S+cost_T ))
#
#             SpBlock.sort(key=lambda element:element[-1])
# #            print('B ', SpBlock)
#             for sb in SpBlock:
#
#                 startslot=sb[0]
#                 endslot= sb[1]
#                 flag=self.first_fit_v_partition(flag, path, req, baseslot, numofslots, startslot, endslot) # 실제 인덱스를 범위로 0~9"""
# #                print(req.id,' : ',flag,' : ',sb, '  -  ',  numofslots)
#                 if flag == 1:
#                     break
#         else :
#             flag = 0
#
#         return flag
#
#     def spec_assin_fair_Proposed_Holding(self, path, req): #best fit
#         self.assingment_name = 'spec_assin_fair_Proposed_Holding'
#         flag = 0
#         numofslots = self.modul_format(path, req)
#         baseslot = self.base_slot(path)
#         threshold = self.range_band_e
#         startslot=0
#         endslot= self.TotalNumofSlots-1
#         hop = len(path['path'])-1
#         endtime = req.end_time
#         holdtime = req.holding_time
#         if holdtime == 0:
#             print('holdtime', holdtime)
#
#         candi_SB = []
#
#         s=0
#         c=0
#         for i in range(len(baseslot)):
#             if baseslot[i][0] == 0:
#                 if c == 0:
#                     s = i
#                 c += 1
#                 if i == self.TotalNumofSlots-1:
#                     candi_SB.append((s,s+c-1,c))# start, end, count,
#             elif baseslot[i][0] > 0:
#                 if c!=0:
#                     candi_SB.append((s,s+c-1,c))
#                     c=0
#
#         candi_SB.sort(key=lambda element:element[2])
# #        print('CAND ', candi_SB)
#         if len(candi_SB) > 0 and candi_SB[-1][2]>=threshold:
#             SpBlock = []
#             for sb in candi_SB:
#                 if sb[-1] >= numofslots:
#                     cost_S = 0
#                     cost_T = 0
#                     sb_start = sb[0]
#                     sb_end = sb[1]
#                     if sb_start == 0:
#                         cost_S = (hop-baseslot[sb_start+numofslots][0])/hop
#                         cost_T = abs(endtime-baseslot[sb_start+numofslots][1])/holdtime
#                         if cost_T >1:
#                             cost_T = 1
#                     elif sb_start>0 and sb_start+numofslots < self.TotalNumofSlots:
#                         cost_S = ((hop-baseslot[sb_start-1][0])+(hop-baseslot[sb_start+numofslots][0]))/hop
#                         cost_T_l = abs(endtime-baseslot[sb_start-1][1])/holdtime
#                         cost_T_r = abs(endtime-baseslot[sb_start+numofslots][1])/holdtime
#                         if cost_T_l >1:
#                             cost_T_l = 1
#                         if cost_T_r >1:
#                             cost_T_r = 1
#                         cost_T = cost_T_l+cost_T_r
#                     elif sb_start+numofslots == self.TotalNumofSlots:
#                         cost_S = (hop-baseslot[sb_start-1][0])/hop
#                         cost_T = abs(endtime-baseslot[sb_start-1][1])/holdtime
#                         if cost_T >1:
#                             cost_T = 1
#
#                     SpBlock.append((sb_start, sb_start+numofslots-1, cost_S, cost_T, cost_S+cost_T ))
#
#                     cost_S = 0
#                     cost_T = 0
#                     sb_start = sb[1]-numofslots+1
#                     sb_end = sb[1]
#                     if sb_start == 0:
#                         cost_S = (hop-baseslot[sb_start+numofslots][0])/hop
#                         cost_T = abs(endtime-baseslot[sb_start+numofslots][1])/holdtime
#                         if cost_T >1:
#                             cost_T = 1
#                     elif sb_start>0 and sb_start+numofslots < self.TotalNumofSlots:
#                         cost_S = ((hop-baseslot[sb_start-1][0])+(hop-baseslot[sb_start+numofslots][0]))/hop
#                         cost_T_l = abs(endtime-baseslot[sb_start-1][1])/holdtime
#                         cost_T_r = abs(endtime-baseslot[sb_start+numofslots][1])/holdtime
#                         if cost_T_l >1:
#                             cost_T_l = 1
#                         if cost_T_r >1:
#                             cost_T_r = 1
#                         cost_T = cost_T_l+cost_T_r
#                     elif sb_start+numofslots == self.TotalNumofSlots:
#                         cost_S = (hop-baseslot[sb_start-1][0])/hop
#                         cost_T = abs(endtime-baseslot[sb_start-1][1])/holdtime
#                         if cost_T >1:
#                             cost_T = 1
#
#                     SpBlock.append((sb_start, sb_end, cost_S, cost_T, cost_T ))
#
#             SpBlock.sort(key=lambda element:element[-1])
# #            print('B ', SpBlock)
#             for sb in SpBlock:
#
#                 startslot=sb[0]
#                 endslot= sb[1]
#                 flag=self.first_fit_v_partition(flag, path, req, baseslot, numofslots, startslot, endslot) # 실제 인덱스를 범위로 0~9"""
# #                print(req.id,' : ',flag,' : ',sb, '  -  ',  numofslots)
#                 if flag == 1:
#                     break
#         else :
#             flag = 0
#
#         return flag
#
#     def modul_format(self, path, req):
#         dist = path['dist']
#         bandw = req.bandwidth
# #        numofslots = bandw
#
#         if dist <= 500:
#             numofslots = math.ceil( bandw/(self.SlotWidth*4))
#         elif dist <= 1000 :
#             numofslots = math.ceil( bandw/(self.SlotWidth*3))
#         elif dist <= 2000 :
#             numofslots = math.ceil( bandw/(self.SlotWidth*2))
#         elif dist > 2000 :
#             numofslots = math.ceil( bandw/(self.SlotWidth*1))
#
# #        numofslots = math.ceil( bandw/(self.SlotWidth*1)) #MF를 무시
#
#         req.nos = numofslots
# #        print(bandw, ' - ' ,numofslots)
#         return numofslots
#
#     def base_slot(self, path):
#         baseslot_2D=[[0,0] for i in range(self.TotalNumofSlots) ]
# #        print(baseslot_2D)
#         for i in range(len(path['path'])-1):
#             fromnode = path['path'][i]
#             tonode = path['path'][i+1]
# #            slot = self.slot_info[fromnode][tonode]
#             slot_2D = self.slot_info_2D[fromnode][tonode]
# #            print(fromnode,'  ', tonode)
# #            print(slot)
# #            if i ==0 :
# #                c = slot
# #            else :
# #                c = np.c_[c,slot]
# #            print(fromnode,'-',tonode, '\n',slot)
# #            baseslot = baseslot+slot
#             for n in range(self.TotalNumofSlots):
#                 baseslot_2D[n][0] = baseslot_2D[n][0]+slot_2D[n][0]
#                 if baseslot_2D[n][1] < slot_2D[n][1]:
#                     baseslot_2D[n][1] = slot_2D[n][1]
#
#
# #        print(baseslot_2D)
#         return baseslot_2D
#
#
#     def first_fit(self, flag, path, req, baseslot, numofslots):
#         cnt=0
#         idx=0
#         while cnt != numofslots :
#             if idx+cnt >= self.TotalNumofSlots:
#                 break
#             if baseslot[idx+cnt][0] == 0:
#                 cnt += 1
#             else:
#                 idx = idx+1
#                 cnt = 0
#
#
#         if cnt == numofslots:
#             req.slot_start = idx
#             req.slot_end = idx+cnt-1
#             req.state = 1
#             req.hop = path['hop']
#             flag = 1
#         else:
#             req.state = 0
#             flag = 0
#         return flag
#
#     def first_fit_v_partition(self, flag, path, req, baseslot, numofslots, range_start, range_end):
#         cnt=0
#         idx=range_start
#         while cnt != numofslots :
#             if idx+cnt > range_end:
#                 break
#             if baseslot[idx+cnt][0] == 0:
#                 cnt += 1
#             else:
#                 idx = idx+1
#                 cnt = 0
#
#         if cnt == numofslots:
#             req.slot_start = idx
#             req.slot_end = idx+cnt-1
#             req.state = 1
#             req.hop = path['hop']
#             flag = 1
#         else:
#             req.state = 0
#             flag = 0
#
#         return flag
#
#     def best_fit_v_partition(self, flag, path, req, baseslot, numofslots, range_start, range_end):
#
#         cnt=0
#         idx=range_start
#
#         while cnt != numofslots :
#             if idx+cnt > range_end:
#                 break
#             if baseslot[idx+cnt][0] == 0:
#                 cnt += 1
#             else:
#                 idx = idx+1
#                 cnt = 0
#
#         if cnt == numofslots:
#             req.slot_start = idx
#             req.slot_end = idx+cnt-1
#             req.state = 1
#             req.hop = path['hop']
#             flag = 1
#         else:
#             req.state = 0
#             flag = 0
#
#         return flag
#
#     def EOSA(self, flag, path, req, baseslot, width, range_start, range_end):
#
#         idx=range_start
#         maxwidth = self.range_band_e
# #        req.req_print()
# #        print('width', width)
#         for p in range(self.TotalNumofSlots):
#             cnt=0
# #            print('p  ',p)
#             if width <(maxwidth/2):
#                 if (p%maxwidth==0) or (p%maxwidth == maxwidth/2-width):
#                     idx=p
#
#                     for n in range(width):
#                         if baseslot[idx+cnt][0] == 0:
#                             cnt += 1
#                         else:
#                             idx = idx+1
#                             cnt = 0
#                             break
#                     if cnt == width:
#                         break
# #                else:
#
#             elif (width== maxwidth/2):
#                 if (p%(maxwidth/2)==0):
#                     idx=p
#                     for n in range(width):
#                         if baseslot[idx+cnt][0] == 0:
#                             cnt += 1
#                         else:
#                             idx = idx+1
#                             cnt = 0
#                             break
#                     if cnt == width:
#                         break
# #                else:
#
#             else:
#                 if (p%maxwidth==0) or (p%maxwidth==maxwidth-width):
#                     idx=p
#                     for n in range(width):
#                         if baseslot[idx+cnt][0] == 0:
#                             cnt += 1
#                         else:
#                             idx = idx+1
#                             cnt = 0
#                             break
#                     if cnt == width:
#                         break
#
#         if cnt == width:
#
#             req.slot_start = idx
#             req.slot_end = idx+cnt-1
#             req.state = 1
#             req.hop = path['hop']
#             flag = 1
#         else:
#             req.state = 0
#             flag = 0
#
#         return flag
#
#     def NFSA(self, flag, path, req, baseslot, width, range_start, range_end):
#
#         idx=range_start
#         maxwidth = self.range_band_e
# #        req.req_print()
# #        print('ID', req.id, 'width', width)
# #        print(baseslot)
#         for p in range(self.TotalNumofSlots):
#             cnt=0
# #            print('p  ',p)
#             if width > (maxwidth/2): #width가 5보다 큰 경우
#                 if p%maxwidth == maxwidth-width:
#                     idx=p
#
#                     for n in range(width):
#                         if baseslot[idx+cnt][0] == 0:
#                             cnt += 1
#                         else:
#                             idx = idx+1
#                             cnt = 0
#                             break
#                     if cnt == width:
#                         break
# #                else:
#
#             elif (width == maxwidth/2): #width가 5이고
#                 if (p%(maxwidth/2)==0): #p도 5의 배수인 경우
#                     idx=p
#                     for n in range(width):
#                         if baseslot[idx+cnt][0] == 0:
#                             cnt += 1
#                         else:
#                             idx = idx+1
#                             cnt = 0
#                             break
#                     if cnt == width:
#                         break
# #                else:
#
#             else:
#                 idx=p
# #                print('else idx  ', idx)
#                 while cnt != width :
#                     if idx+cnt >= self.TotalNumofSlots:
#                         break
#                     if baseslot[idx+cnt][0] == 0:
#                         cnt += 1
#                     else:
#                         idx = idx+1
#                         cnt = 0
#                 if cnt == width:
#                         break
#
#
#         if cnt == width:
# #            print(idx, '  ' ,idx+cnt-1 )
#             req.slot_start = idx
#             req.slot_end = idx+cnt-1
#             req.state = 1
#             req.hop = path['hop']
#             flag = 1
#         else:
#             req.state = 0
#             flag = 0
#
#         return flag
#
#
#
#     def release(self, cur_time):
# #        print('rel cur_time ', cur_time)
#         self.req_in_service.sort(key=lambda object:object.end_time)
#
# #        print('B', len(self.req_in_service))
#
#         n=0
#         while n<len(self.req_in_service):
#             temp = self.req_in_service[n]
# #            temp.req_print()
#             if temp.end_time < cur_time:
# #                temp.req_print()
#                 ios = temp.slot_start
#                 ioe = temp.slot_end
#                 path = temp.path
#                 for i in range(len(path['path'])-1):
#                     fromnode = path['path'][i]
#                     tonode = path['path'][i+1]
# #                    print(fromnode,' ', tonode,' ', ios,' ', ioe )
# #                    slot = self.slot_info[fromnode][tonode]
#                     slot_2D = self.slot_info_2D[fromnode][tonode]
# #                    print(slot_2D)
#                     for z in range(ios, ioe+1):
#                         if slot_2D[z][0] == 1:
#                             slot_2D[z] = [0,0]
#                         else:
#                             print('error')
# #                    print(slot)
#                 self.req_in_service.pop(n)
# #                print('rm ')
#             else:
#                 n += 1
# #        print('A', len(self.req_in_service))
#
#
#
#
#
#
# if __name__ == "__main__":
#     NOB = 0
#     sim = Simulation( 100000, 0, 15.0, 50, 1, 10)
#     sim.sim_main()
#
#