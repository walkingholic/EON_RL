# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 20:18:54 2019

@author: User
"""
import random
import request
import numpy as np
from graph import DiGraph
import algorithms
import copy
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time
import networkx as nx
import fuzzylite as fl
import torch.multiprocessing as mp
# from ConvPPO import PPO
from CNN import CNN
import torch
import torch.nn as nn


# random.seed(1)
# np.random.seed(1)


print("============================================================================================")

# set device to cpu or cuda
device = torch.device('cpu')

# if (torch.cuda.is_available()):
#     device = torch.device('cuda:0')
#     torch.cuda.empty_cache()
#     print("Device set to : " + str(torch.cuda.get_device_name(device)))
# else:
#     print("Device set to : cpu")

print("============================================================================================")

class Simulation(mp.Process):

    def __init__(self, n_of_req, num_kpath, num_of_action, n_of_warm, inter_a_rate, holding_time, range_band_s, range_band_e, TotalNumofSlots, RT_name, SA_name, dirpath=None, env_version=None, dic=None, queue=None, req_in_gen=None):
        super(Simulation, self).__init__()
        self.N_Req = n_of_req
        self.rid = 0
        self.req_in_service = []
        self.req_in_gen = []
        self.req_in_success = []
        self.req_in_blocked = []
        self.modeShow = 1000
        self.INF = 1000000
        self.num_of_action = num_of_action
        self.num_of_subblock = int(num_of_action/num_kpath)
        self.inter_arr_rate = inter_a_rate
        self.holding_time = holding_time
        self.erlang = inter_a_rate*holding_time
        print('Erlang: ', self.erlang)
        self.cur_time = 0.0
        # self.networktopo="sixnode"
        self.networktopo = "test_nsf"

        #        self.networktopo="test_usnet"
        #        self.G = DiGraph("sixnode")
        #        self.G = DiGraph("test_nsf")
        self.G = DiGraph(self.networktopo)
        self.all_k_path=None
        print(self.G[1])

        # self.G.save_graph_as_svg(self.G._data, 'test_nsf')
        # print(type(self.G._data))

        self.NumOfNode = len(self.G._data)
        self.Distance = copy.deepcopy(self.G._data)
        #        self.slot_info ={}
        self.slot_info_2D = {}
        self.TotalNumofSlots = TotalNumofSlots
        self.K = num_kpath
        self.env_version='default'
        self.lastNOB = 0
        self.NOB = 0
        self.NO_Success = 0

        self.SlotWidth = 12.5
        self.SimWarmup = n_of_warm
        self.NumOfEdge = 0
        self.range_band_s = range_band_s
        self.range_band_e = range_band_e
        self.Dthreshold = 0
        self.linkfreq = {}
        self.totalusedslot = 0
        self.utilization = 0
        self.avghop = 0
        # self.routing_name = ''
        # self.assingment_name = ''

        self.cnt_gen_band = np.zeros((self.range_band_e))
        self.cnt_suc_band = np.zeros((self.range_band_e))
        self.cnt_blk_band = np.zeros((self.range_band_e))
        self.cnt_suc_nos = np.zeros((self.range_band_e))

        self.total_Gen_bandwidth = 0
        self.total_Block_bandwidth = 0

        self.dsj = []
        self.ndsj = []

        self.engine_RT = None
        self.engine_SA = None
        self.init_fuzzy_RT()
        self.init_fuzzy_SA()

        self.dic = dic
        self.que = queue

        self.vg = nx.Graph()
        self.dirpath = dirpath
        self.SA_name = SA_name
        self.RT_name = RT_name
        self.env_version = env_version

        if req_in_gen:
            self.req_in_gen = req_in_gen
            print('copy req in gen')
        else:
            print('make req in gen')
            for i in range(self.N_Req):
                req = self.gene_request(i, 0)
                self.req_in_gen.append(req)

        for i in self.G._data:
            #            self.slot_info[i]={}
            self.slot_info_2D[i] = {}
            self.linkfreq[i] = {}
            for j in self.G._data[i]:
                #                self.slot_info[i][j]=np.zeros((self.TotalNumofSlots))
                self.slot_info_2D[i][j] = [[0, 0] for _ in range(self.TotalNumofSlots)]
                self.linkfreq[i][j] = 0
                #                print(self.slot_info_2D[i][j])
                self.NumOfEdge += 1

    def link_freq(self, allpath):

        for s in allpath:
            print('s', s)
            for d in allpath[s]:
                print('d ', d)
                k = 0
                for path in allpath[s][d]:
                    print(k, 'th ', path['cost'], '  ', path['path'])
                    k += 1
                    for i in range(path['hop']):
                        fromnode = path['path'][i]
                        tonode = path['path'][i + 1]
                        f = fromnode
                        t = tonode
                        self.linkfreq[f][t] += 1



    def init_fuzzy_SA(self):
        self.engine_SA = fl.Engine(
            name="sa",
            description=""
        )
        # ramp start, end 경사 오르기.
        self.engine_SA.input_variables = [
            fl.InputVariable(
                name="Slot",
                description="",
                enabled=True,
                minimum=0.000,
                maximum=2.000,
                lock_range=False,
                terms=[
                    fl.Ramp("L", 0.800, 0.400),
                    fl.Trapezoid("M", 0.400, 0.800, 1.200, 1.600),
                    fl.Ramp("H", 1.200, 1.600)
                ]
            )
            ,
            fl.InputVariable(
                name="Time",
                description="",
                enabled=True,
                minimum=0.000,
                maximum=2.000,
                lock_range=False,
                terms=[
                    fl.Ramp("L", 0.800, 0.400),
                    fl.Trapezoid("M", 0.400, 0.800, 1.200, 1.600),
                    fl.Ramp("H", 1.200, 1.600)
                ]
            )

        ]
        self.engine_SA.output_variables = [
            fl.OutputVariable(
                name="Priority",
                description="",
                enabled=True,
                minimum=0.000,
                maximum=10.000,
                lock_range=False,
                aggregation=fl.Maximum(),
                defuzzifier=fl.Centroid(100),
                lock_previous=False,
                terms=[
                    fl.Ramp("VH", 3.000, 1.000),
                    fl.Triangle("H", 1.000, 3.000, 5.000),
                    fl.Triangle("M", 3.000, 5.000, 7.000),
                    fl.Triangle("L", 5.000, 7.000, 9.000),
                    fl.Ramp("VL", 7.000, 9.000)
                ]
            )
        ]

        self.engine_SA.rule_blocks = [
            fl.RuleBlock(
                name="mamdani",
                description="",
                enabled=True,
                conjunction=fl.Minimum(),
                disjunction=fl.Maximum(),
                implication=fl.Minimum(),
                activation=fl.General(),
                rules=[
                    fl.Rule.create("if Slot is L and Time is L then Priority is VH", self.engine_SA),
                    fl.Rule.create("if Slot is L and Time is M then Priority is H", self.engine_SA),
                    fl.Rule.create("if Slot is L and Time is H then Priority is M", self.engine_SA),

                    fl.Rule.create("if Slot is M and Time is L then Priority is H", self.engine_SA),
                    fl.Rule.create("if Slot is M and Time is M then Priority is M", self.engine_SA),
                    fl.Rule.create("if Slot is M and Time is H then Priority is L", self.engine_SA),

                    fl.Rule.create("if Slot is H and Time is L then Priority is M", self.engine_SA),
                    fl.Rule.create("if Slot is H and Time is M then Priority is L", self.engine_SA),
                    fl.Rule.create("if Slot is H and Time is H then Priority is VL", self.engine_SA)
                ]
            )
        ]


    def init_fuzzy_RT(self):
        self.engine_RT = fl.Engine(
            name="routing",
            description=""
        )
        # ramp start, end 경사 오르기.
        self.engine_RT.input_variables = [
            fl.InputVariable(
                name="Distance",
                description="",
                enabled=True,
                minimum=0.000,
                maximum=2500.000,
                lock_range=False,
                terms=[
                    fl.Ramp("S", 1000.000, 500.000),
                    fl.Trapezoid("M", 500.000, 1000.000, 1500.000, 2000.000),
                    fl.Ramp("L", 1500.000, 2000.000)
                ]
            )
            ,
            fl.InputVariable(
                name="Hop",
                description="",
                enabled=True,
                minimum=1.000,
                maximum=8.000,
                lock_range=False,
                terms=[
                    fl.Ramp("S", 4.000, 3.000),
                    fl.Trapezoid("M", 3.000, 4.000, 5.000, 6.000),
                    fl.Ramp("L", 5.000, 6.000)
                ]
            ),
            fl.InputVariable(
                name="Utilization",
                description="",
                enabled=True,
                minimum=0.000,
                maximum=1.000,
                lock_range=False,
                terms=[
                    fl.Ramp("L", 0.200, 0.100),
                    fl.Trapezoid("M", 0.100, 0.200, 0.400, 0.500),
                    fl.Ramp("H", 0.400, 0.500)
                ]
            )

        ]
        self.engine_RT.output_variables = [
            fl.OutputVariable(
                name="Priority",
                description="",
                enabled=True,
                minimum=0.000,
                maximum=10.000,
                lock_range=False,
                aggregation=fl.Maximum(),
                defuzzifier=fl.Centroid(100),
                lock_previous=False,
                terms=[
                    fl.Ramp("VH", 2.000, 0.000),
                    fl.Triangle("H", 1.000, 3.000, 5.000),
                    fl.Triangle("M", 3.000, 5.000, 7.000),
                    fl.Triangle("L", 5.000, 7.000, 9.000),
                    fl.Ramp("VL", 8.000, 10.000)
                ]
            )
        ]

        self.engine_RT.rule_blocks = [
            fl.RuleBlock(
                name="mamdani",
                description="",
                enabled=True,
                conjunction=fl.Minimum(),
                disjunction=fl.Maximum(),
                implication=fl.Minimum(),
                activation=fl.General(),
                rules=[

                    fl.Rule.create("if Utilization is H then Priority is VL", self.engine_RT),
                    fl.Rule.create("if Utilization is L then Priority is VH", self.engine_RT),


                    fl.Rule.create("if Distance is S and Hop is S and Utilization is M then Priority is VH", self.engine_RT),

                    fl.Rule.create("if Distance is S and Hop is M and Utilization is M then Priority is H", self.engine_RT),

                    fl.Rule.create("if Distance is S and Hop is L and Utilization is M then Priority is L", self.engine_RT),

                    fl.Rule.create("if Distance is M and Hop is S and Utilization is M then Priority is H", self.engine_RT),

                    fl.Rule.create("if Distance is M and Hop is M and Utilization is M then Priority is M", self.engine_RT),

                    fl.Rule.create("if Distance is M and Hop is L and Utilization is M then Priority is L", self.engine_RT),

                    fl.Rule.create("if Distance is L and Hop is S and Utilization is M then Priority is H", self.engine_RT),

                    fl.Rule.create("if Distance is L and Hop is M and Utilization is M then Priority is L", self.engine_RT),

                    fl.Rule.create("if Distance is L and Hop is L and Utilization is M then Priority is VH", self.engine_RT)
                ]
            )
        ]

    '''
    fl.Rule.create("if Distance is S and Hop is S and Utilization is L then Priority is VH", self.engine),
                        fl.Rule.create("if Distance is S and Hop is S and Utilization is M then Priority is M", self.engine),
                        fl.Rule.create("if Distance is S and Hop is S and Utilization is H then Priority is VL", self.engine),

                        fl.Rule.create("if Distance is S and Hop is M and Utilization is L then Priority is VH", self.engine),
                        fl.Rule.create("if Distance is S and Hop is M and Utilization is M then Priority is M", self.engine),
                        fl.Rule.create("if Distance is S and Hop is M and Utilization is H then Priority is VL", self.engine),

                        fl.Rule.create("if Distance is S and Hop is L and Utilization is L then Priority is VH", self.engine),
                        fl.Rule.create("if Distance is S and Hop is L and Utilization is M then Priority is M", self.engine),
                        fl.Rule.create("if Distance is S and Hop is L and Utilization is H then Priority is VL", self.engine),

                        fl.Rule.create("if Distance is M and Hop is S and Utilization is L then Priority is VH", self.engine),
                        fl.Rule.create("if Distance is M and Hop is S and Utilization is M then Priority is M", self.engine),
                        fl.Rule.create("if Distance is M and Hop is S and Utilization is H then Priority is VL", self.engine),

                        fl.Rule.create("if Distance is M and Hop is M and Utilization is L then Priority is VH", self.engine),
                        fl.Rule.create("if Distance is M and Hop is M and Utilization is M then Priority is M", self.engine),
                        fl.Rule.create("if Distance is M and Hop is M and Utilization is H then Priority is VL", self.engine),

                        fl.Rule.create("if Distance is M and Hop is L and Utilization is L then Priority is VH", self.engine),
                        fl.Rule.create("if Distance is M and Hop is L and Utilization is M then Priority is M", self.engine),
                        fl.Rule.create("if Distance is M and Hop is L and Utilization is H then Priority is VL", self.engine),

                        fl.Rule.create("if Distance is L and Hop is S and Utilization is L then Priority is VH", self.engine),
                        fl.Rule.create("if Distance is L and Hop is S and Utilization is M then Priority is M", self.engine),
                        fl.Rule.create("if Distance is L and Hop is S and Utilization is H then Priority is VL", self.engine),

                        fl.Rule.create("if Distance is L and Hop is M and Utilization is L then Priority is VH", self.engine),
                        fl.Rule.create("if Distance is L and Hop is M and Utilization is M then Priority is M", self.engine),
                        fl.Rule.create("if Distance is L and Hop is M and Utilization is H then Priority is VL", self.engine),

                        fl.Rule.create("if Distance is L and Hop is L and Utilization is L then Priority is VH", self.engine),
                        fl.Rule.create("if Distance is L and Hop is L and Utilization is M then Priority is M", self.engine),
                        fl.Rule.create("if Distance is L and Hop is L and Utilization is H then Priority is VL", self.engine)
    '''

    def get_dsj_paths(self, all_k_path):

        tmp = nx.Graph()
        # tmp = nx.DiGraph()

        for s in all_k_path:
            for d in all_k_path[s]:
                for k in range(len(all_k_path[s][d])):
                    weight = random.randint(1, 300)
                    path = all_k_path[s][d][k]['path']
                    # print(s, d, all_path[s][d][k]['path'])
                    edgelist = []

                    for e in range(len(path) - 1):
                        fn = int(path[e])
                        tn = int(path[e + 1])
                        edgelist.append('%02d%02d' % (fn, tn))

                    tmp.add_node('%02d%02d%02d' % (s, d, k), weight=weight, path=path, edgelist=edgelist, src=s, dst=d)

        vert = sorted(tmp.nodes(data=True), key=lambda element: element[1]['weight'], reverse=True)

        for u in vert:
            upath = u[1]['edgelist']
            upID = u[0]
            usrc = u[1]['src']
            udst = u[1]['dst']
            # print(upID, u[1]['weight'], usrc, udst, upath)

            for v in vert:
                vpath = v[1]['edgelist']
                vpID = v[0]
                if vpID == upID:
                    break
                vsrc = v[1]['src']
                vdst = v[1]['dst']
                # print(upID, usrc, udst, upath)

                for e in upath:
                    if e in vpath or (usrc == vsrc and udst == vdst):
                        # print('-> ', vpID, '속함', e, vpath)
                        tmp.add_edge(upID, vpID)
                        break

        dsj = []
        ndsj = []

        for e in vert:
            tmp_adj = list(tmp.adj[e[0]])
            # print(e[0], e[1]['weight'], type(tmp_adj), tmp_adj)
            isDsj = True
            for v in tmp_adj:
                if v in dsj:
                    isDsj = False
                    break

            if isDsj:
                dsj.append(e[0])
            else:
                ndsj.append(e[0])

        print(len(dsj))
        print(len(ndsj))

        return dsj, ndsj

    def make_traffic_mtx(self, all_k_path):
        traffic_matrix = dict()

        for req in self.req_in_gen:
            src = req.source
            dst = req.destination
            band = req.bandwidth
            if src not in traffic_matrix.keys():
                traffic_matrix[src] = {}
            if dst not in traffic_matrix[src].keys():
                traffic_matrix[src][dst] = band
            else:
                traffic_matrix[src][dst] += band

        for s in all_k_path:
            for d in all_k_path[s]:
                for k in range(len(all_k_path[s][d])):
                    w = traffic_matrix[s][d]
                    path = all_k_path[s][d][k]['path']
                    # print(s, d, all_path[s][d][k]['path'])
                    edgelist = []

                    for e in range(len(path) - 1):
                        fn = int(path[e])
                        tn = int(path[e + 1])
                        edgelist.append('%02d%02d' % (fn, tn))

                    self.vg.add_node('%02d%02d%02d' % (s, d, k), weight=w, path=path, edgelist=edgelist, src=s, dst=d)

        vert = sorted(self.vg.nodes(data=True), key=lambda element: element[1]['weight'], reverse=True)
        # print(len(vert))
        # print((vert))


        for u in vert:
            upath = u[1]['edgelist']
            upID = u[0]
            usrc = u[1]['src']
            udst = u[1]['dst']
            # print(upID, u[1]['weight'], usrc, udst, upath)

            for v in vert:
                vpath = v[1]['edgelist']
                vpID = v[0]
                if vpID == upID:
                    break
                vsrc = v[1]['src']
                vdst = v[1]['dst']
                # print(upID, usrc, udst, upath)

                for e in upath:
                    if e in vpath or (usrc == vsrc and udst == vdst):
                        # print('-> ', vpID, '속함', e, vpath)
                        self.vg.add_edge(upID, vpID)
                        break




        for e in vert:
            tmp_adj = list(self.vg.adj[e[0]])
            # print(e[0], e[1]['weight'], type(tmp_adj), tmp_adj)
            isDsj = True
            for v in tmp_adj:
                if v in self.dsj:
                    isDsj = False
                    break

            if isDsj:
                self.dsj.append(e[0])
                # print(type(e[0]))
            else:
                self.ndsj.append(e[0])

        # print(len(dsj), dsj)
        # print(len(ndsj), ndsj)


        # color_map = []
        # for node in self.vg.nodes:
        #     if node in dsj:
        #         color_map.append('red')
        #     else:
        #         color_map.append('blue')
        # # font_weight='bold'
        # nx.draw(self.vg, with_labels=True, node_color=color_map)
        # plt.show()

        # nG = nx.Graph(self.G._data)
        # nx.draw(nG, with_labels=True)
        # plt.show()





    def specslot_assign(self, path, req, action):

        flag = 0
        numofslots = self.modul_format(path, req)
        baseslot = self.base_slot(path)

        idx = action
        cnt = 0

        while (idx+cnt) < self.TotalNumofSlots :


            if baseslot[idx + cnt][0] == 0:
                cnt += 1
                if cnt == numofslots:
                    break
            else:
                break

            # print('Cnt: ', cnt)


        if cnt == numofslots:
            req.slot_start = idx
            req.slot_end = idx + cnt - 1
            req.state = 1
            req.hop = path['hop']
            flag = 1

            # ssbaseslot = np.array(baseslot)
            # print(path)
            # req.req_print()
            # print(flag, ssbaseslot[:, 0])

        else:
            req.state = 0
            flag = 0

        # print('Flag :=> ',flag)
        # if flag and idx == 99:
        #     ssbaseslot = np.array(baseslot)
        #     print(path)
        #     req.req_print()
        #     print(flag, ssbaseslot[:, 0])
        #     print(idx, cnt)
        #     print('Flag :=> ', flag)


        return flag




    def specslot_assign_specific(self, path, req, subblock):

        flag = 0
        baseslot = self.base_slot(path)
        idx_start, idx_end, count, numofslots, sb_path = subblock
        # cnt=0
        # while cnt<numofslots and idx_start+cnt < self.TotalNumofSlots:
        #     if baseslot[idx_start + cnt][0] == 0:
        #         cnt += 1
        #     else:
        #         break

        if numofslots != 0:
            req.slot_start = idx_start
            req.slot_end = idx_start + numofslots -1
            req.state = 1
            req.hop = path['hop']
            flag = 1

        else:
            # print('Error: specslot_assign_specific')
            # print('Current path: ', path)
            # print('CNT==COUNT is not equal !', numofslots)
            # print('Sub block info: ',subblock)
            #
            # for i in range(numofslots):
            #     print(baseslot[idx_start + i][0], end=' ')
            # print()

            req.state = 0
            flag = 0

        return flag

    def env_BC(self, nsamples, statesize, actionsize):

        statelist, reqinfolist, actionOnehotlist = np.empty((0, statesize)), np.empty((0, 30)), np.empty((0, actionsize))
        actlist = []
        all_k_path = self.precal_path()


        print('Behavior Storing..')

        succ_cnt_req=0
        blk_cnt_req=0
        idx_req = 0
        while succ_cnt_req != nsamples:
            # print(succ_cnt_req)
            idx_req+=1
            if idx_req%1000==0:
                print(idx_req)
            req = self.gene_request(idx_req, 0)
            self.req_in_gen.append(req)
            self.cur_time = req.arrival_time
            self.release(self.cur_time)
            k_path = self.KSP_routing(req, all_k_path)
            state, req_info, kth_candi_SB = self.base_slot_presentation_v2_5_reference_modified(k_path, req)

            flag = 0
            flag, act, path = self.spec_assin_2D_forBC(kth_candi_SB, req)
            # print(flag)
            # print(path)
            # req.req_print()
            if flag == 1:
                succ_cnt_req += 1
                self.updata_link_state(path, req)
                self.req_in_service.append(req)
                statelist = np.vstack((statelist, state))
                reqinfolist = np.vstack((reqinfolist, req_info))
                actlist.append([act])
                action = np.eye(5)[act]
                actionOnehotlist = np.vstack((actionOnehotlist, action))
                # print(actionlist)
            else:
                blk_cnt_req += 1

        print('BBP: ', blk_cnt_req/(succ_cnt_req+blk_cnt_req), blk_cnt_req)

        # print(succ_cnt_req)

        return statelist, reqinfolist, actionOnehotlist, actlist



    def env_init(self):
        # all_k_path = self.precal_path()
        self.all_k_path = self.precal_path()
        # self.make_traffic_mtx(self.all_k_path)
        self.req_in_service = []
        # self.req_in_gen = []
        self.req_in_success = []
        self.req_in_blocked = []
        state, req_info, req, candi_sb = self.env_reset()
        # print(state.shape)
        # action_contain=[]
        # for i in range(2):
        #     act = np.random.randint(0, self.num_of_action-1)
        #     action_contain.append(act)
        for i in range(self.SimWarmup):
            # state, req_info, req, reward, done, candi_sb = self.env_step(req, np.random.randint(0, self.num_of_action-1), candi_sb)#0~4
            state, req_info, req, reward, done, candi_sb = self.env_step(req, np.random.randint(0, self.num_of_action), candi_sb)#0~4
            # print(state.shape)

    def env_step(self, req, action, kth_cand_sb):

        # k_path = self.KSP_routing(req, self.all_k_path)
        # print('env_step:', action_contain)
        # req.req_print()
        npath = int(action/self.num_of_subblock)
        sb_number = action%self.num_of_subblock
        # print(len(kth_cand_sb))
        # for sb in kth_cand_sb:
        #     print(sb)
        candi_sb = kth_cand_sb[npath]
        # print(npath, sb_number)
        # print(candi_sb)
        _, _, count_slots, numofslots, path= candi_sb[sb_number] #count_slots: 서브블록의 크기, numofslots: 루트에서 모듈레이션 포멧에 따른 슬롯수,
        # flag = self.spec_assin_FF(path, req)
        # path = k_path[npath]
        # print('path', path)
        # print('sbpath', sbpath)

        if count_slots>0:
            flag = self.specslot_assign_specific(path, req, candi_sb[sb_number])
        else:
            flag = 0
        if flag:
            # req.req_print()
            # baseslot = np.array(self.base_slot(path))
            # print(baseslot[:, 0])
            self.updata_link_state(path, req)
            # default_reward = 10
            # baseslot = self.base_slot(path)
            # ssi = req.slot_start
            # esi = req.slot_end
            # maxindex = self.TotalNumofSlots-1
            # hop = req.hop
            # cost=0
            # if ssi==0:
            #     cost = (1 + (hop - baseslot[esi + 1][0])) / hop
            # elif ssi>0 and esi<maxindex:
            #     cost = ((hop - baseslot[ssi - 1][0]) + (hop - baseslot[esi+1][0])) / hop
            # else:
            #     cost = ((hop - baseslot[ssi - 1][0])) / hop

            # for i in range(len(path['path']) - 1):
            #     fromnode = path['path'][i]
            #     tonode = path['path'][i + 1]
            #     #            slot = self.slot_info[fromnode][tonode]
            #     slot_2D = self.slot_info_2D[fromnode][tonode]
            #     for z in range(req.slot_start, req.slot_end + 1):
            #         #                slot[z] = 1
            #         #                print('z: ',z, '  ', slot_2D[z])
            #         print('B: ', slot_2D[z][0], slot_2D[z][1])

            # for i in range(len(path['path']) - 1):
            #     fromnode = path['path'][i]
            #     tonode = path['path'][i + 1]
            #     #            slot = self.slot_info[fromnode][tonode]
            #     slot_2D = self.slot_info_2D[fromnode][tonode]
            #     for z in range(req.slot_start, req.slot_end + 1):
            #         #                slot[z] = 1
            #         #                print('z: ',z, '  ', slot_2D[z])
            #         print('A: ', slot_2D[z][0], slot_2D[z][1])
            # baseslot = np.array(self.base_slot(path))
            # print(baseslot[:, 0])
            self.req_in_service.append(req)
            # reward = cost*default_reward
            reward = 1

            self.cnt_suc_band[int(req.bandwidth / self.SlotWidth) - 1] += 1
            self.cnt_suc_nos[req.nos - 1] += 1
            self.req_in_success.append(req)
            self.NO_Success += 1
            self.avghop += req.hop
            # print('Succ')
            # req.req_print()
            # print('in service req: ', len(self.req_in_service))
        else:
            # state, req_info = self.base_slot_presentation(k_path, req)
            # state = state.squeeze()
            # print('Block', action)
            # req.req_print()
            # for sbb in kth_cand_sb:
            #     print((sbb))
            #
            # baseslot = self.base_slot(path)
            self.total_Block_bandwidth += req.bandwidth
            self.cnt_blk_band[int(req.bandwidth / self.SlotWidth) - 1] += 1
            self.NOB += 1
            self.req_in_blocked.append(req)
            reward = -1



        self.rid += 1
        req = self.gene_request(self.rid, 0)

        self.cur_time = req.arrival_time
        self.release(self.cur_time)
        k_path = self.KSP_routing(req, self.all_k_path)






        if self.env_version == 2:
            state, req_info, kth_cand_sb = self.base_slot_presentation_v2_reference(k_path, req)
        elif self.env_version == 2.1:
            state, req_info, kth_cand_sb = self.base_slot_presentation_v2_1_reference_modified(k_path, req)
        elif self.env_version == 2.2:
            state, req_info, kth_cand_sb = self.base_slot_presentation_v2_2_reference_modified_add_imgMap(k_path, req)
        elif self.env_version == 2.3:
            state, req_info, kth_cand_sb = self.base_slot_presentation_v2_3_reference_modified(k_path, req)
        elif self.env_version == 2.4:
            state, req_info, kth_cand_sb = self.base_slot_presentation_v2_4_reference_modified(k_path, req)
        elif self.env_version == 2.5:
            state, req_info, kth_cand_sb = self.base_slot_presentation_v2_5_reference_modified(k_path, req)


        done = 0
        # print()
        # req.req_print()
        # print(len(kth_cand_sb))
        # print((kth_cand_sb))

        return state, req_info, req, reward, done, kth_cand_sb

    def env_reset(self):

        # self.req_in_service = []
        # self.req_in_gen = []
        # self.req_in_success = []
        # self.req_in_blocked = []
        # self.NOB = 0

        req = self.gene_request(self.rid, 0)

        self.cur_time = req.arrival_time
        self.release(self.cur_time)
        k_path = self.KSP_routing(req, self.all_k_path)
        # state, req_info = self.base_slot_presentation(k_path, req)
        # state, req_info = self.base_slot_presentation_v1(k_path, req)
        # state, req_info = self.base_slot_presentation_v3(k_path, req)
        # state, req_info = self.base_slot_presentation_v4(k_path, req)
        # state, req_info = self.base_slot_presentation_v5(k_path, req)

        # state, req_info, kth_cand_sb = self.base_slot_presentation_v2_2_reference_modified_add_imgMap(k_path, req)# 100*100,97
        # state, req_info, kth_cand_sb = self.base_slot_presentation_v2_reference(k_path, req)
        # state, req_info, kth_cand_sb = self.base_slot_presentation_v2_1_reference_modified(k_path, req)
        # state, req_info, kth_cand_sb = self.base_slot_presentation_v2_3_reference_modified(k_path, req)


        if self.env_version == 2:
            state, req_info, kth_cand_sb = self.base_slot_presentation_v2_reference(k_path, req)
        elif self.env_version == 2.1:
            state, req_info, kth_cand_sb = self.base_slot_presentation_v2_1_reference_modified(k_path, req)
        elif self.env_version == 2.2:
            state, req_info, kth_cand_sb = self.base_slot_presentation_v2_2_reference_modified_add_imgMap(k_path, req)
        elif self.env_version == 2.3:
            state, req_info, kth_cand_sb = self.base_slot_presentation_v2_3_reference_modified(k_path, req)
        elif self.env_version == 2.4:
            state, req_info, kth_cand_sb = self.base_slot_presentation_v2_4_reference_modified(k_path, req)
        elif self.env_version == 2.5:
            state, req_info, kth_cand_sb = self.base_slot_presentation_v2_5_reference_modified(k_path, req)
        # else:
        #     return None,None,None,None


        return state, req_info, req, kth_cand_sb

    def run(self):
        all_k_path = self.precal_path()
        modeShow = 1000
        # for i in range(self.N_Req):
        #     req = self.gene_request(i, 0)
        #     self.req_in_gen.append(req)

        # self.make_traffic_mtx(all_k_path)

        idx_req = 0
        for req in self.req_in_gen:
            self.sim_main_operation(req, idx_req, all_k_path, 0)
            idx_req += 1

            if idx_req % self.modeShow ==0:

                blk_req = self.NOB - self.lastNOB
                succ_req = self.modeShow-blk_req

                self.que.put((2*succ_req-self.modeShow, succ_req, blk_req, 0))

        print(len(self.req_in_blocked))
        print('BBP: ', self.NOB/self.N_Req)
        print(self.erlang, 'finish sim')

        self.utilization = self.totalusedslot / (self.NumOfEdge * self.TotalNumofSlots * self.N_Req)
        self.avghop = self.avghop / (self.N_Req - self.NOB)

        if self.dic != None:
            self.dic[self.erlang] = [self.NOB, self.cnt_gen_band, self.cnt_blk_band, self.cnt_suc_band, self.total_Gen_bandwidth, self.total_Block_bandwidth, self.cnt_suc_nos, self.utilization, self.avghop, self.RT_name, self.SA_name, self.networktopo]
        # return self.NOB, self.cnt_gen_band, self.cnt_blk_band, self.cnt_suc_band, self.total_Gen_bandwidth, self.total_Block_bandwidth, self.cnt_suc_nos, self.utilization, self.avghop, self.RT_name, self.SA_name, self.networktopo

    def sim_main(self):

        all_k_path = self.precal_path()
        # for i in range(self.N_Req):
        #     req = self.gene_request(i, 0)
        #     self.req_in_gen.append(req)
        self.make_traffic_mtx(all_k_path)

        idx_req = 0
        for req in self.req_in_gen:
            self.sim_main_operation(req, idx_req, all_k_path, 0)
            idx_req += 1

        print(len(self.req_in_blocked))
        print('finish sim')

        self.utilization = self.totalusedslot / (self.NumOfEdge * self.TotalNumofSlots * self.N_Req)
        self.avghop = self.avghop / (self.N_Req - self.NOB)

        return self.NOB, self.cnt_gen_band, self.cnt_blk_band, self.cnt_suc_band, self.total_Gen_bandwidth, self.total_Block_bandwidth, self.cnt_suc_nos, self.utilization, self.avghop, self.RT_name, self.SA_name, self.networktopo

    def sim_main_operation(self, req, idx_req, all_k_path, warmup):

        # req = self.gene_request(idx_req, warmup)
        self.cur_time = req.arrival_time
        self.release(self.cur_time)

        if self.RT_name == 'KSP':
            k_path = self.KSP_routing(req, all_k_path)
        elif self.RT_name == 'BFR':
            k_path = self.BandFragRatio_path(req, all_k_path)
        elif self.RT_name == 'LUP':
            k_path = self.least_used_path(req, all_k_path)
        elif self.RT_name == 'DJP':
            k_path = self.disjoint_path(req, all_k_path)
        elif self.RT_name == 'LU':
            k_path = self.least_used(req, all_k_path)
        elif self.RT_name == 'FZ':
            k_path = self.fuzzy_path(req, all_k_path)
        else:
            k_path = None

        if warmup != 1:
            self.total_Gen_bandwidth += req.bandwidth
            self.cnt_gen_band[int(req.bandwidth / self.SlotWidth) - 1] += 1

            for i in self.G._data:
                for j in self.G._data[i]:
                    self.totalusedslot += sum(int(u) for u, v in self.slot_info_2D[i][j])
        flag = 0
        if idx_req % self.modeShow == 0 and warmup != 1:
            print(self.RT_name,'-', self.SA_name,': ', self.erlang ,': ' ,idx_req, 'th - req gen - ', self.cur_time, 'len(Q)', len(self.req_in_service), ' NOB:', self.NOB, ' BPP:', self.NOB/(idx_req+1), 'part BPP:', (self.NOB-self.lastNOB)/(self.modeShow))
            self.lastNOB = self.NOB

        for path in k_path:
            if self.SA_name=='FF':
                flag=self.spec_assin_FF(path, req)
            elif self.SA_name == 'BF':
                flag = self.spec_assin_BF(path, req)  # best fit
            elif self.SA_name == '2D':
                flag = self.spec_assin_2D(path, req)
            elif self.SA_name == 'FLEF':
                flag = self.spec_assin_FLEF(path, req)
            elif self.SA_name == 'HT':
                flag = self.spec_assin_HTime(path, req)
            elif self.SA_name == 'SL':
                flag = self.spec_assin_Slot(path, req)
            elif self.SA_name == 'FZ2D':
                flag = self.spec_assin_FZ2D(path, req)

            if flag:
                self.updata_link_state(path, req)
                self.req_in_service.append(req)
                if warmup != 1:
                    self.cnt_suc_band[int(req.bandwidth / self.SlotWidth) - 1] += 1
                    self.cnt_suc_nos[req.nos - 1] += 1
                    self.req_in_success.append(req)
                    self.NO_Success += 1
                    self.avghop += req.hop

                break

        if flag != 1 and warmup != 1:
            self.total_Block_bandwidth += req.bandwidth
            self.cnt_blk_band[int(req.bandwidth / self.SlotWidth) - 1] += 1
            self.NOB += 1
            self.req_in_blocked.append(req)

    def KSP_routing(self, req, all_k_path):
        # temp_path = copy.deepcopy(all_k_path[req.source][req.destination])
        temp_path = sorted(all_k_path[req.source][req.destination], key=lambda element: element['cost'])

        # for path in temp_path:
        #     print("%d-%d" % (req.source, req.destination))
        #     print(path)

        #        print("%d-%d" % (req.source, req.destination))

        #        temp_path=self.BandFragRatio(temp_path)
        #        temp_path = self.min_hop(temp_path)
        #        temp_path = self.least_used_path(temp_path)

        #        temp_path = self.least_used(temp_path)
        #        temp_path=self.BandFragRatio_path(temp_path)

        #        req.req_print()
        #         print('b     ', temp_path)
        # temp_path.sort(key=lambda element: element['cost'])
        #        for k in temp_path:
        #            print(k['cost'])
        #         print('a     ',temp_path)

        # print(temp_path)

        return temp_path

    def BandFragRatio(self, req, all_k_path):  # each link sum
        kpaths = all_k_path[req.source][req.destination]

        for k in kpaths:
            cost = 0
            for i in range(k['hop']):
                unused_slot = 0
                maxblock = 0
                fromnode = k['path'][i]
                tonode = k['path'][i + 1]
                unused_slot = 0
                blocksize = 0
                for n in range(self.TotalNumofSlots):
                    if self.slot_info_2D[fromnode][tonode][n][0] == 0:
                        blocksize += 1
                        unused_slot += 1

                    elif self.slot_info_2D[fromnode][tonode][n][0] != 0:
                        blocksize = 0
                    if maxblock < blocksize:
                        maxblock = blocksize

                if unused_slot != 0:
                    cost += (1 - (maxblock / unused_slot))
                else:
                    cost = self.INF

            k['cost'] = cost
        #            print()
        #            print(k,'  ', cost)
        temp_path = sorted(all_k_path[req.source][req.destination], key=lambda element: element['cost'])

        return temp_path

    def BandFragRatio_path(self, req, all_k_path):  # each link sum
        kpaths = all_k_path[req.source][req.destination]

        for k in kpaths:
            cost = 0
            baseslot = self.base_slot(k)
            maxblock = 0
            blocksize = 0
            unused_slot = 0

            for n in range(self.TotalNumofSlots):
                if baseslot[n][0] == 0:
                    blocksize += 1
                    unused_slot += 1

                elif baseslot[n][0] >= 1:
                    blocksize = 0

                if maxblock < blocksize:
                    maxblock = blocksize

            if unused_slot != 0:
                cost = (1 - (maxblock / unused_slot))
            else:
                #                print('unused = zero ', unused_slot)
                cost = self.INF

            k['cost'] = cost

        temp_path = sorted(all_k_path[req.source][req.destination], key=lambda element: element['cost'])

        # print(all_k_path[req.source][req.destination])
        # print(temp_path)

        return temp_path

    def min_hop(self, req, all_k_path):
        kpaths = all_k_path[req.source][req.destination]
        #        print('b  ', kpaths)
        for k in kpaths:
            k['cost'] = k['hop']
        #        print('a  ', kpaths)
        temp_path = sorted(all_k_path[req.source][req.destination], key=lambda element: element['cost'])

        return temp_path

    def least_used(self, req, all_k_path):
        kpaths = all_k_path[req.source][req.destination]

        for k in kpaths:
            #            print( k)
            cost = 0
            hap = 0
            for i in range(k['hop']):
                fromnode = k['path'][i]
                tonode = k['path'][i + 1]
                hap += sum(int(u) for u, v in self.slot_info_2D[fromnode][tonode])
            #                print(fromnode, '  ', tonode,' ', hap)
            cost = hap / (self.TotalNumofSlots * k['hop'])
            k['cost'] = cost
        temp_path = sorted(all_k_path[req.source][req.destination], key=lambda element: element['cost'])
        return temp_path

    def least_used_path(self, req, all_k_path):
        kpaths = all_k_path[req.source][req.destination]

        for k in kpaths:
            #            print( k)
            cost = 0
            hap = 0
            baseslot = self.base_slot(k)

            for i in range(self.TotalNumofSlots):
                if baseslot[i][0] > 0:
                    hap += 1
            cost = hap / (self.TotalNumofSlots)
            k['cost'] = cost
        #            print(k)
        temp_path = sorted(all_k_path[req.source][req.destination], key=lambda element: element['cost'])
        return temp_path

    def fuzzy_path(self, req, all_k_path):
        kpaths = all_k_path[req.source][req.destination]

        for k in kpaths:
            #            print( k)
            cost = 0
            hap = 0
            baseslot = self.base_slot(k)

            for i in range(self.TotalNumofSlots):
                if baseslot[i][0] > 0:
                    hap += 1
            util = hap / (self.TotalNumofSlots)

            self.engine_RT.input_variable('Distance').value = k['dist']
            self.engine_RT.input_variable('Hop').value = k['hop']
            self.engine_RT.input_variable('Utilization').value = util
            self.engine_RT.process()

            k['cost'] = self.engine_RT.output_variable('Priority').value
        #            print(k)
        temp_path = sorted(all_k_path[req.source][req.destination], key=lambda element: element['cost'])
        return temp_path

    def disjoint_path(self, req, all_k_path):

        # self.vg.add_node('%02d%02d%02d' % (s, d, k), weight=w, path=path, edgelist=edgelist, src=s, dst=d)
        # for u in vert:
        #     upath = u[1]['edgelist']
        #     upID = u[0]
        #     usrc = u[1]['src']
        #     udst = u[1]['dst']

        temp_djpath=[]
        temp_ndjpath=[]
        src = req.source
        dst = req.destination
        # dict(vert)
        src_dst = '%02d%02d'%(src,dst)
        # print(src_dst)
        for p in self.dsj:
            result = p.startswith(src_dst)
            if result:
                k = int(p[-2:])
                path = all_k_path[src][dst][k]
                # path = all_k_path[src][dst][k]['path']
                path['DJP']=True
                temp_djpath.append(path)
                break

        for p in self.ndsj:
            result = p.startswith(src_dst)
            if result:
                k = int(p[-2:])
                path = all_k_path[src][dst][k]
                path['DJP'] = False
                temp_ndjpath.append(path)

        # print(temp_djpath)
        temp_ndjpath = sorted(temp_ndjpath, key=lambda element: element['cost'])
        # print(temp_ndjpath)

        temp_path = temp_djpath+temp_ndjpath
        # print(temp_path)

        return temp_path




    def updata_link_state(self, path, req):
        req.path = copy.deepcopy(path)
        for i in range(len(path['path']) - 1):
            fromnode = path['path'][i]
            tonode = path['path'][i + 1]
            #            slot = self.slot_info[fromnode][tonode]
            slot_2D = self.slot_info_2D[fromnode][tonode]
            #            print(fromnode, '-', tonode,' >> ', req.bandwidth)
            #            print(slot_2D)
            #            print('s: ',req.slot_start,' e:', req.slot_end)
            for z in range(req.slot_start, req.slot_end + 1):
                #                slot[z] = 1
                #                print('z: ',z, '  ', slot_2D[z])
                slot_2D[z][0] = 1
                slot_2D[z][1] = req.end_time

    #            print(fromnode, '-', tonode,' >> ', req.bandwidth)
    #            print(slot)
    #            req.req_print()
    #            print(slot_2D)

    def gene_request(self, idx_req, warmup):
        s = random.randint(1, self.NumOfNode)
        d = random.randint(1, self.NumOfNode)

        while s == d:
            d = random.randint(1, self.NumOfNode)
        b = random.randint(self.range_band_s, self.range_band_e) * self.SlotWidth
        inter_arrival_time = round(random.expovariate(self.inter_arr_rate), 4)
        holding_time = 0
        while holding_time == 0:
            holding_time = round(np.random.exponential(self.holding_time), 4)

        self.cur_time = self.cur_time + inter_arrival_time

        req = request.Request(idx_req, s, d, b, self.cur_time, holding_time, warmup)

        return req

    def precal_path(self):
        pre_cal_path = {}
        cnt = 0
        dist = 0
        dist_list = []
        hop_list = []
        for i in range(1, self.NumOfNode + 1):
            pre_cal_path[i] = {}
            for j in range(1, self.NumOfNode + 1):
                s = str(i)
                d = str(j)
                if s != d:
                    items = algorithms.ksp_yen(self.G, s, d, self.K)
                    for it in items:
                        it['dist'] = it['cost']
                        it['hop'] = len(it['path']) - 1
                        cnt += 1
                        dist += it['dist']

                        dist_list.append(it['dist'])
                        hop_list.append(it['hop'])
                    #                        print(it)
                    pre_cal_path[i][j] = copy.deepcopy(items)
        #                    print('items  ',items)


        # print('sum hop:', sum(hop_list))
        # print('avg hop:', sum(hop_list)/len(hop_list))
        # print('max hop:', max(hop_list))
        # print('min hop:', min(hop_list))
        #
        # print('sum dist:', sum(dist_list))
        # print('avg dist:', sum(dist_list)/len(dist_list))
        # print('max dist:', max(dist_list))
        # print('min dist:', min(dist_list))

        return pre_cal_path

    def spec_assin_FF(self, path, req):
        flag = 0
        numofslots = self.modul_format(path, req)
        baseslot = self.base_slot(path)

        flag = self.first_fit(flag, path, req, baseslot, numofslots)

        return flag

    def spec_assin_LF(self, path, req):
        flag = 0
        numofslots = self.modul_format(path, req)
        baseslot = self.base_slot(path)

        flag = self.last_fit(flag, path, req, baseslot, numofslots)

        return flag


    def spec_assin_BF(self, path, req):
        flag = 0
        numofslots = self.modul_format(path, req)
        baseslot = self.base_slot(path)
        baseslot = np.array(baseslot)
        candi_SB = []

        start_idx = 0
        cnt = 0
        remain = 0
        for i in range(len(baseslot)):
            if baseslot[i][0] == 0:
                if cnt == 0:
                    start_idx = i
                cnt += 1
                if i == self.TotalNumofSlots - 1:
                    candi_SB.append((start_idx, start_idx + cnt - 1, cnt))
                    remain += cnt
            elif baseslot[i][0] > 0:
                if cnt != 0:
                    candi_SB.append((start_idx, start_idx + cnt - 1, cnt))
                    remain += cnt
                    cnt = 0

        candi_SB.sort(key=lambda element: element[2])
        if len(candi_SB) > 0 :
            for sb in candi_SB:
                if sb[-1] >= numofslots:
                    startslot = sb[0]
                    endslot = sb[1]
                    flag = self.first_fit_v_partition(flag, path, req, baseslot, numofslots, startslot,
                                                      endslot)  # 실제 인덱스를 범위로 0~9"""
                    if flag == 1:
                        break
        else:
            flag = 0

        return flag

    def spec_assin_FZ2D(self, path, req):  # best fit
        flag = 0
        numofslots = self.modul_format(path, req)
        baseslot = self.base_slot(path)
        startslot = 0
        endslot = self.TotalNumofSlots - 1
        hop = len(path['path']) - 1
        endtime = req.end_time
        holdtime = req.holding_time
        if holdtime == 0:
            print('holdtime', holdtime)

        candi_SB = []

        s = 0
        c = 0
        for i in range(len(baseslot)):
            if baseslot[i][0] == 0:
                if c == 0:
                    s = i
                c += 1
                if i == self.TotalNumofSlots - 1:
                    candi_SB.append((s, s + c - 1, c))  # start, end, count,
            elif baseslot[i][0] > 0:
                if c != 0:
                    candi_SB.append((s, s + c - 1, c))
                    c = 0

        candi_SB.sort(key=lambda element: element[2])
        #        print('CAND ', candi_SB)
        if len(candi_SB) > 0 :
            SpBlock = []
            for sb in candi_SB:
                if sb[-1] >= numofslots:
                    cost_S = 0
                    cost_T = 0
                    sb_start = sb[0]
                    sb_end = sb[1]
                    if sb_start == 0:
                        cost_S = (hop - baseslot[sb_start + numofslots][0]) / hop
                        cost_T = abs(endtime - baseslot[sb_start + numofslots][1]) / holdtime
                        if cost_T > 1:
                            cost_T = 1
                    elif sb_start > 0 and sb_start + numofslots < self.TotalNumofSlots:
                        cost_S = ((hop - baseslot[sb_start - 1][0]) + (hop - baseslot[sb_start + numofslots][0])) / hop
                        cost_T_l = abs(endtime - baseslot[sb_start - 1][1]) / holdtime
                        cost_T_r = abs(endtime - baseslot[sb_start + numofslots][1]) / holdtime
                        if cost_T_l > 1:
                            cost_T_l = 1
                        if cost_T_r > 1:
                            cost_T_r = 1
                        cost_T = cost_T_l + cost_T_r
                    elif sb_start + numofslots == self.TotalNumofSlots:
                        cost_S = (hop - baseslot[sb_start - 1][0]) / hop
                        cost_T = abs(endtime - baseslot[sb_start - 1][1]) / holdtime
                        if cost_T > 1:
                            cost_T = 1

                    self.engine_SA.input_variable('Slot').value = cost_S
                    self.engine_SA.input_variable('Time').value = cost_T
                    self.engine_SA.process()

                    SpBlock.append((sb_start, sb_start + numofslots - 1, cost_S, cost_T, self.engine_SA.output_variable('Priority').value))

                    cost_S = 0
                    cost_T = 0
                    sb_start = sb[1] - numofslots + 1
                    sb_end = sb[1]
                    if sb_start == 0:
                        cost_S = (hop - baseslot[sb_start + numofslots][0]) / hop
                        cost_T = abs(endtime - baseslot[sb_start + numofslots][1]) / holdtime
                        if cost_T > 1:
                            cost_T = 1
                    elif sb_start > 0 and sb_start + numofslots < self.TotalNumofSlots:
                        cost_S = ((hop - baseslot[sb_start - 1][0]) + (hop - baseslot[sb_start + numofslots][0])) / hop
                        cost_T_l = abs(endtime - baseslot[sb_start - 1][1]) / holdtime
                        cost_T_r = abs(endtime - baseslot[sb_start + numofslots][1]) / holdtime
                        if cost_T_l > 1:
                            cost_T_l = 1
                        if cost_T_r > 1:
                            cost_T_r = 1
                        cost_T = cost_T_l + cost_T_r
                    elif sb_start + numofslots == self.TotalNumofSlots:
                        cost_S = (hop - baseslot[sb_start - 1][0]) / hop
                        cost_T = abs(endtime - baseslot[sb_start - 1][1]) / holdtime
                        if cost_T > 1:
                            cost_T = 1

                    self.engine_SA.input_variable('Slot').value = cost_S
                    self.engine_SA.input_variable('Time').value = cost_T
                    self.engine_SA.process()
                    SpBlock.append((sb_start, sb_end, cost_S, cost_T, self.engine_SA.output_variable('Priority').value))

            SpBlock.sort(key=lambda element: element[-1])
            #            print('B ', SpBlock)
            for sb in SpBlock:

                startslot = sb[0]
                endslot = sb[1]
                flag = self.first_fit_v_partition(flag, path, req, baseslot, numofslots, startslot,
                                                  endslot)  # 실제 인덱스를 범위로 0~9"""
                #                print(req.id,' : ',flag,' : ',sb, '  -  ',  numofslots)
                if flag == 1:
                    break
        else:
            flag = 0

        return flag


    def spec_assin_2D(self, path, req):  # best fit
        flag = 0
        numofslots = self.modul_format(path, req)
        baseslot = self.base_slot(path)
        startslot = 0
        endslot = self.TotalNumofSlots - 1
        hop = len(path['path']) - 1
        endtime = req.end_time
        holdtime = req.holding_time
        if holdtime == 0:
            print('holdtime', holdtime)

        candi_SB = []

        s = 0
        c = 0
        for i in range(len(baseslot)):
            if baseslot[i][0] == 0:
                if c == 0:
                    s = i
                c += 1
                if i == self.TotalNumofSlots - 1:
                    candi_SB.append((s, s + c - 1, c))  # start, end, count,
            elif baseslot[i][0] > 0:
                if c != 0:
                    candi_SB.append((s, s + c - 1, c))
                    c = 0

        candi_SB.sort(key=lambda element: element[2])
        #        print('CAND ', candi_SB)
        if len(candi_SB) > 0 :
            SpBlock = []
            for sb in candi_SB:
                if sb[-1] >= numofslots:
                    cost_S = 0
                    cost_T = 0
                    sb_start = sb[0]
                    sb_end = sb[1]
                    if sb_start == 0:
                        cost_S = (hop - baseslot[sb_start + numofslots][0]) / hop
                        cost_T = abs(endtime - baseslot[sb_start + numofslots][1]) / holdtime
                        if cost_T > 1:
                            cost_T = 1
                    elif sb_start > 0 and sb_start + numofslots < self.TotalNumofSlots:
                        cost_S = ((hop - baseslot[sb_start - 1][0]) + (hop - baseslot[sb_start + numofslots][0])) / hop
                        cost_T_l = abs(endtime - baseslot[sb_start - 1][1]) / holdtime
                        cost_T_r = abs(endtime - baseslot[sb_start + numofslots][1]) / holdtime
                        if cost_T_l > 1:
                            cost_T_l = 1
                        if cost_T_r > 1:
                            cost_T_r = 1
                        cost_T = cost_T_l + cost_T_r
                    elif sb_start + numofslots == self.TotalNumofSlots:
                        cost_S = (hop - baseslot[sb_start - 1][0]) / hop
                        cost_T = abs(endtime - baseslot[sb_start - 1][1]) / holdtime
                        if cost_T > 1:
                            cost_T = 1

                    SpBlock.append((sb_start, sb_start + numofslots - 1, cost_S, cost_T, cost_S + cost_T))

                    cost_S = 0
                    cost_T = 0
                    sb_start = sb[1] - numofslots + 1
                    sb_end = sb[1]
                    if sb_start == 0:
                        cost_S = (hop - baseslot[sb_start + numofslots][0]) / hop
                        cost_T = abs(endtime - baseslot[sb_start + numofslots][1]) / holdtime
                        if cost_T > 1:
                            cost_T = 1
                    elif sb_start > 0 and sb_start + numofslots < self.TotalNumofSlots:
                        cost_S = ((hop - baseslot[sb_start - 1][0]) + (hop - baseslot[sb_start + numofslots][0])) / hop
                        cost_T_l = abs(endtime - baseslot[sb_start - 1][1]) / holdtime
                        cost_T_r = abs(endtime - baseslot[sb_start + numofslots][1]) / holdtime
                        if cost_T_l > 1:
                            cost_T_l = 1
                        if cost_T_r > 1:
                            cost_T_r = 1
                        cost_T = cost_T_l + cost_T_r
                    elif sb_start + numofslots == self.TotalNumofSlots:
                        cost_S = (hop - baseslot[sb_start - 1][0]) / hop
                        cost_T = abs(endtime - baseslot[sb_start - 1][1]) / holdtime
                        if cost_T > 1:
                            cost_T = 1

                    SpBlock.append((sb_start, sb_end, cost_S, cost_T, cost_S + cost_T))



            SpBlock.sort(key=lambda element: element[-1])
            #            print('B ', SpBlock)
            for sb in SpBlock:

                startslot = sb[0]
                endslot = sb[1]
                flag = self.first_fit_v_partition(flag, path, req, baseslot, numofslots, startslot,
                                                  endslot)  # 실제 인덱스를 범위로 0~9"""
                #                print(req.id,' : ',flag,' : ',sb, '  -  ',  numofslots)
                if flag == 1:
                    break
        else:
            flag = 0

        return flag



    def spec_assin_HTime(self, path, req):  # best fit
        flag = 0
        numofslots = self.modul_format(path, req)
        baseslot = self.base_slot(path)
        startslot = 0
        endslot = self.TotalNumofSlots - 1
        hop = len(path['path']) - 1
        endtime = req.end_time
        holdtime = req.holding_time
        if holdtime == 0:
            print('holdtime', holdtime)

        candi_SB = []

        s = 0
        c = 0
        for i in range(len(baseslot)):
            if baseslot[i][0] == 0:
                if c == 0:
                    s = i
                c += 1
                if i == self.TotalNumofSlots - 1:
                    candi_SB.append((s, s + c - 1, c))  # start, end, count,
            elif baseslot[i][0] > 0:
                if c != 0:
                    candi_SB.append((s, s + c - 1, c))
                    c = 0

        candi_SB.sort(key=lambda element: element[2])
        #        print('CAND ', candi_SB)
        if len(candi_SB) > 0 :
            SpBlock = []
            for sb in candi_SB:
                if sb[-1] >= numofslots:
                    cost_S = 0
                    cost_T = 0
                    sb_start = sb[0]
                    sb_end = sb[1]
                    if sb_start == 0:
                        cost_S = (hop - baseslot[sb_start + numofslots][0]) / hop
                        cost_T = abs(endtime - baseslot[sb_start + numofslots][1]) / holdtime
                        if cost_T > 1:
                            cost_T = 1
                    elif sb_start > 0 and sb_start + numofslots < self.TotalNumofSlots:
                        cost_S = ((hop - baseslot[sb_start - 1][0]) + (hop - baseslot[sb_start + numofslots][0])) / hop
                        cost_T_l = abs(endtime - baseslot[sb_start - 1][1]) / holdtime
                        cost_T_r = abs(endtime - baseslot[sb_start + numofslots][1]) / holdtime
                        if cost_T_l > 1:
                            cost_T_l = 1
                        if cost_T_r > 1:
                            cost_T_r = 1
                        cost_T = cost_T_l + cost_T_r
                    elif sb_start + numofslots == self.TotalNumofSlots:
                        cost_S = (hop - baseslot[sb_start - 1][0]) / hop
                        cost_T = abs(endtime - baseslot[sb_start - 1][1]) / holdtime
                        if cost_T > 1:
                            cost_T = 1

                    SpBlock.append((sb_start, sb_start + numofslots - 1, cost_S, cost_T, cost_S + cost_T))

                    cost_S = 0
                    cost_T = 0
                    sb_start = sb[1] - numofslots + 1
                    sb_end = sb[1]
                    if sb_start == 0:
                        cost_S = (hop - baseslot[sb_start + numofslots][0]) / hop
                        cost_T = abs(endtime - baseslot[sb_start + numofslots][1]) / holdtime
                        if cost_T > 1:
                            cost_T = 1
                    elif sb_start > 0 and sb_start + numofslots < self.TotalNumofSlots:
                        cost_S = ((hop - baseslot[sb_start - 1][0]) + (hop - baseslot[sb_start + numofslots][0])) / hop
                        cost_T_l = abs(endtime - baseslot[sb_start - 1][1]) / holdtime
                        cost_T_r = abs(endtime - baseslot[sb_start + numofslots][1]) / holdtime
                        if cost_T_l > 1:
                            cost_T_l = 1
                        if cost_T_r > 1:
                            cost_T_r = 1
                        cost_T = cost_T_l + cost_T_r
                    elif sb_start + numofslots == self.TotalNumofSlots:
                        cost_S = (hop - baseslot[sb_start - 1][0]) / hop
                        cost_T = abs(endtime - baseslot[sb_start - 1][1]) / holdtime
                        if cost_T > 1:
                            cost_T = 1

                    SpBlock.append((sb_start, sb_end, cost_S, cost_T, cost_S + cost_T))

            SpBlock.sort(key=lambda element: element[-2])
            #            print('B ', SpBlock)
            for sb in SpBlock:

                startslot = sb[0]
                endslot = sb[1]
                flag = self.first_fit_v_partition(flag, path, req, baseslot, numofslots, startslot,
                                                  endslot)  # 실제 인덱스를 범위로 0~9"""
                #                print(req.id,' : ',flag,' : ',sb, '  -  ',  numofslots)
                if flag == 1:
                    break
        else:
            flag = 0

        return flag

    def spec_assin_Slot(self, path, req):  # best fit
        flag = 0
        numofslots = self.modul_format(path, req)
        baseslot = self.base_slot(path)
        startslot = 0
        endslot = self.TotalNumofSlots - 1
        hop = len(path['path']) - 1
        endtime = req.end_time
        holdtime = req.holding_time
        if holdtime == 0:
            print('holdtime', holdtime)

        candi_SB = []

        s = 0
        c = 0
        for i in range(len(baseslot)):
            if baseslot[i][0] == 0:
                if c == 0:
                    s = i
                c += 1
                if i == self.TotalNumofSlots - 1:
                    candi_SB.append((s, s + c - 1, c))  # start, end, count,
            elif baseslot[i][0] > 0:
                if c != 0:
                    candi_SB.append((s, s + c - 1, c))
                    c = 0

        candi_SB.sort(key=lambda element: element[2])
        #        print('CAND ', candi_SB)
        if len(candi_SB) > 0 :
            SpBlock = []
            for sb in candi_SB:
                if sb[-1] >= numofslots:
                    cost_S = 0
                    cost_T = 0
                    sb_start = sb[0]
                    sb_end = sb[1]
                    if sb_start == 0:
                        cost_S = (hop - baseslot[sb_start + numofslots][0]) / hop
                        cost_T = abs(endtime - baseslot[sb_start + numofslots][1]) / holdtime
                        if cost_T > 1:
                            cost_T = 1
                    elif sb_start > 0 and sb_start + numofslots < self.TotalNumofSlots:
                        cost_S = ((hop - baseslot[sb_start - 1][0]) + (hop - baseslot[sb_start + numofslots][0])) / hop
                        cost_T_l = abs(endtime - baseslot[sb_start - 1][1]) / holdtime
                        cost_T_r = abs(endtime - baseslot[sb_start + numofslots][1]) / holdtime
                        if cost_T_l > 1:
                            cost_T_l = 1
                        if cost_T_r > 1:
                            cost_T_r = 1
                        cost_T = cost_T_l + cost_T_r
                    elif sb_start + numofslots == self.TotalNumofSlots:
                        cost_S = (hop - baseslot[sb_start - 1][0]) / hop
                        cost_T = abs(endtime - baseslot[sb_start - 1][1]) / holdtime
                        if cost_T > 1:
                            cost_T = 1

                    SpBlock.append((sb_start, sb_start + numofslots - 1, cost_S, cost_T, cost_S + cost_T))

                    cost_S = 0
                    cost_T = 0
                    sb_start = sb[1] - numofslots + 1
                    sb_end = sb[1]
                    if sb_start == 0:
                        cost_S = (hop - baseslot[sb_start + numofslots][0]) / hop
                        cost_T = abs(endtime - baseslot[sb_start + numofslots][1]) / holdtime
                        if cost_T > 1:
                            cost_T = 1
                    elif sb_start > 0 and sb_start + numofslots < self.TotalNumofSlots:
                        cost_S = ((hop - baseslot[sb_start - 1][0]) + (hop - baseslot[sb_start + numofslots][0])) / hop
                        cost_T_l = abs(endtime - baseslot[sb_start - 1][1]) / holdtime
                        cost_T_r = abs(endtime - baseslot[sb_start + numofslots][1]) / holdtime
                        if cost_T_l > 1:
                            cost_T_l = 1
                        if cost_T_r > 1:
                            cost_T_r = 1
                        cost_T = cost_T_l + cost_T_r
                    elif sb_start + numofslots == self.TotalNumofSlots:
                        cost_S = (hop - baseslot[sb_start - 1][0]) / hop
                        cost_T = abs(endtime - baseslot[sb_start - 1][1]) / holdtime
                        if cost_T > 1:
                            cost_T = 1

                    SpBlock.append((sb_start, sb_end, cost_S, cost_T, cost_S + cost_T))

            SpBlock.sort(key=lambda element: element[-3])
            #            print('B ', SpBlock)
            for sb in SpBlock:

                startslot = sb[0]
                endslot = sb[1]
                flag = self.first_fit_v_partition(flag, path, req, baseslot, numofslots, startslot,
                                                  endslot)  # 실제 인덱스를 범위로 0~9"""
                #                print(req.id,' : ',flag,' : ',sb, '  -  ',  numofslots)
                if flag == 1:
                    break
        else:
            flag = 0

        return flag


    def spec_assin_FLEF(self, path, req):
        flag = 0
        numofslots = self.modul_format(path, req)
        baseslot = self.base_slot(path)
        baseslot = np.array(baseslot)
        candi_SB = []
        exact_SB = []

        start_idx = 0
        cnt = 0
        remain = 0
        for i in range(len(baseslot)):
            if baseslot[i][0] == 0:
                if cnt == 0:
                    start_idx = i
                cnt += 1
                if i == self.TotalNumofSlots - 1:
                    candi_SB.append((start_idx, start_idx + cnt - 1, cnt))
                    if cnt==numofslots:
                        exact_SB.append((start_idx, start_idx + cnt - 1, cnt))

                    remain += cnt
            elif baseslot[i][0] > 0:
                if cnt != 0:
                    candi_SB.append((start_idx, start_idx + cnt - 1, cnt))
                    if cnt==numofslots:
                        exact_SB.append((start_idx, start_idx + cnt - 1, cnt))

                    remain += cnt
                    cnt = 0



        if path['DJP']:
            # print(req.id, 'djp first', path)
            exact_SB.sort(key=lambda element: element[0])
            # print(req.id, 'djp first', exact_SB)
            if len(exact_SB)>0:
                for sb in exact_SB:
                    if sb[-1] >= numofslots:
                        startslot = sb[0]
                        endslot = sb[1]
                        flag = self.first_fit_v_partition(flag, path, req, baseslot, numofslots, startslot, endslot)
                        if flag == 1:
                            break
            else:
                flag = self.first_fit(flag, path, req, baseslot, numofslots)

        else:
            # print(req.id, 'ndjp Last', path)
            exact_SB.sort(key=lambda element: element[0], reverse=True)
            # print(req.id, 'ndjp Last', exact_SB)
            if len(exact_SB) > 0:
                for sb in exact_SB:
                    if sb[-1] >= numofslots:
                        startslot = sb[0]
                        endslot = sb[1]
                        flag = self.first_fit_v_partition(flag, path, req, baseslot, numofslots, startslot, endslot)
                        if flag == 1:
                            break
            else:
                flag = self.last_fit(flag, path, req, baseslot, numofslots)

        # print('ID:',req.id, ' flag:', flag, ' NumSlots:', numofslots)
        # req.req_print()

        return flag





    def modul_format(self, path, req):
        dist = path['dist']
        bandw = req.bandwidth
        #        numofslots = bandw

        if dist <= 500:
            numofslots = math.ceil(bandw / (self.SlotWidth * 4))
        elif dist <= 1000:
            numofslots = math.ceil(bandw / (self.SlotWidth * 3))
        elif dist <= 2000:
            numofslots = math.ceil(bandw / (self.SlotWidth * 2))
        elif dist > 2000:
            numofslots = math.ceil(bandw / (self.SlotWidth * 1))

        #        numofslots = math.ceil( bandw/(self.SlotWidth*1)) #MF를 무시

        req.nos = numofslots
        #        print(bandw, ' - ' ,numofslots)
        return numofslots

    def base_slot(self, path):
        baseslot_2D = [[0, 0] for i in range(self.TotalNumofSlots)]
        #        print(baseslot_2D)
        for i in range(len(path['path']) - 1):
            fromnode = path['path'][i]
            tonode = path['path'][i + 1]
            #            slot = self.slot_info[fromnode][tonode]
            slot_2D = self.slot_info_2D[fromnode][tonode]
            #            print(fromnode,'  ', tonode)
            #            print(slot)
            #            if i ==0 :
            #                c = slot
            #            else :
            #                c = np.c_[c,slot]
            #            print(fromnode,'-',tonode, '\n',slot)
            #            baseslot = baseslot+slot
            for n in range(self.TotalNumofSlots):
                baseslot_2D[n][0] = baseslot_2D[n][0] + slot_2D[n][0]
                if baseslot_2D[n][1] < slot_2D[n][1]:
                    baseslot_2D[n][1] = slot_2D[n][1]

        #        print(baseslot_2D)
        # print(baseslot_2D)
        # plt.imshow(baseslot_2D)
        # plt.show()
        return baseslot_2D



    # state size 596
    def base_slot_presentation_v5(self, kpath, req):  # 각 루트별로 일자로 state 구성
        # plt.figure(figsize=(5, 5), dpi=200, constrained_layout=True)

        source = np.eye(30)[req.source]
        dest = np.eye(30)[req.destination]
        h = [req.holding_time]
        bh = [req.holding_time, req.bandwidth]

        state = np.concatenate([source, dest, h])


        timevalue = req.holding_time
        # 1번째 가능 블록 사이즈, 1번째 가능블록 시작 인덱스, 1번째 가능 블록 사이즈, 1번째 가능블록 시작 인덱스, 루트에서 요구 슬롯사이즈, 평균 가능 블록 사이즈, 총 가능한 fs 수.

        for k in range(self.K):
            if len(kpath) > k:
                path = kpath[k]
                numofslots = self.modul_format(path, req)  # 루트에서 요구 슬롯사이즈
                baseslot = np.array(self.base_slot(path))
                avail_num_FS = 0  # 가능한 블록들의 총 FS 수
                candi_SB = []
                slot_state=[]
                s, c = 0, 0
                for i in range(len(baseslot)):
                    if baseslot[i][0] == 0:
                        if c == 0:
                            s = i
                        c += 1
                        if i == self.TotalNumofSlots - 1:
                            if c >= numofslots:
                                candi_SB.append((s, s + c - 1, c))  # start, end, count,
                                avail_num_FS += c
                    elif baseslot[i][0] > 0:
                        if c != 0:
                            if c >= numofslots:
                                candi_SB.append((s, s + c - 1, c))
                                avail_num_FS += c
                            c = 0

                    if baseslot[i][1] - self.cur_time > timevalue:
                        val = (baseslot[i][1] - self.cur_time) / timevalue
                        if val>2:
                            slot_state.append(2)
                        else:
                            slot_state.append(val)
                    else:
                        slot_state.append((baseslot[i][1] - self.cur_time)/timevalue)


                lenSB = len(candi_SB)
                if lenSB > 0:
                    sb_ff = candi_SB[0]
                    sb_lf = candi_SB[-1]

                    sff, eff, cff = sb_ff
                    slf, elf, clf = sb_lf
                else:
                    sff, eff, cff = -1, -1, 0
                    slf, elf, clf = -1, -1, 0
                state = np.concatenate([state, slot_state, [cff], [sff], [clf], [elf]])

                if lenSB == 0:
                    avg_avail_num_FS = 0
                else:
                    avg_avail_num_FS = avail_num_FS / lenSB  # 평균 가능 블록 사이즈

                state = np.concatenate([state, [numofslots], [avail_num_FS], [avg_avail_num_FS]])

            else:
                emt_slot_state = [ -1 for _ in range(len(baseslot))]
                state = np.concatenate([state, emt_slot_state, [-1], [-1], [-1], [-1]])
                state = np.concatenate([state, [-1], [-1], [-1]])


        req_info = np.concatenate([source, dest, bh])
        req_info = np.expand_dims(req_info, axis=0)

        state = np.expand_dims(state, axis=0)
        # print(state.shape)
        # print(state)
        return state, req_info


    # state size 96
    def base_slot_presentation_v4(self, kpath, req): #각 루트별로 일자로 state 구성
        # plt.figure(figsize=(5, 5), dpi=200, constrained_layout=True)

        source = np.eye(30)[req.source]
        dest = np.eye(30)[req.destination]
        h = [req.holding_time]
        bh = [req.holding_time, req.bandwidth]

        state = np.concatenate([source, dest, h])

        j=1
        # 1번째 가능 블록 사이즈, 1번째 가능블록 시작 인덱스, 1번째 가능 블록 사이즈, 1번째 가능블록 시작 인덱스, 루트에서 요구 슬롯사이즈, 평균 가능 블록 사이즈, 총 가능한 fs 수.
        for k, path in enumerate(kpath):
            numofslots = self.modul_format(path, req) # 루트에서 요구 슬롯사이즈
            baseslot = np.array(self.base_slot(path))
            avail_num_FS=0 #가능한 블록들의 총 FS 수
            candi_SB = []
            s, c = 0, 0
            for i in range(len(baseslot)):
                if baseslot[i][0] == 0:
                    if c == 0:
                        s = i
                    c += 1
                    if i == self.TotalNumofSlots - 1:
                        if c >= numofslots:
                            candi_SB.append( (s, s+c-1, c) )  # start, end, count,
                            avail_num_FS += c
                elif baseslot[i][0] > 0:
                    if c != 0:
                        if c >= numofslots:
                            candi_SB.append( (s, s+c-1, c) )
                            avail_num_FS+=c
                        c = 0
            lenSB = len(candi_SB)
            if lenSB>0:
                sb_ff = candi_SB[0]
                sb_lf = candi_SB[-1]

                sff, eff, cff = sb_ff
                slf, elf, clf = sb_lf
            else:
                sff, eff, cff = -1, -1, 0
                slf, elf, clf = -1, -1, 0
            state = np.concatenate([state, [cff], [sff], [clf], [elf]])

            if lenSB == 0:
                avg_avail_num_FS=0
            else:
                avg_avail_num_FS = avail_num_FS/lenSB #평균 가능 블록 사이즈

            state = np.concatenate([state, [numofslots], [avail_num_FS], [avg_avail_num_FS]])




        req_info = np.concatenate([source, dest, bh])
        req_info = np.expand_dims(req_info, axis=0)

        state = np.expand_dims(state, axis=0)
        # print(state.shape)
        # print(state)
        return state, req_info



    #state size 1239
    def base_slot_presentation_v3(self, kpath, req):
        # plt.figure(figsize=(5, 5), dpi=200, constrained_layout=True)
        # maxvalue = 50
        maxvalue = req.holding_time
        j=1
        state = []
        for k, path in enumerate(kpath):
            numofslots = self.modul_format(path, req) # 루트에서 요구 슬롯사이즈
            baseslot = np.array(self.base_slot(path))
            candi_SB = []
            s, c = 0, 0
            for i in range(len(baseslot)):
                if baseslot[i][0] == 0:
                    if c == 0:
                        s = i
                    c += 1
                    if i == self.TotalNumofSlots - 1:
                        if c >= numofslots:
                            candi_SB.append( (s, s+c-1, c) )  # start, end, count,
                elif baseslot[i][0] > 0:
                    if c != 0:
                        if c >= numofslots:
                            candi_SB.append( (s, s+c-1, c) )
                        c = 0
                if len(candi_SB)>=j:
                    break
            lenSB = len(candi_SB)
            avail_num_FS=0 #가능한 블록들의 총 FS 수
            for i in range(j):
                if len(candi_SB) == 0:
                    s, e, c = -1, -1, 0
                else:
                    sb = candi_SB.pop(0)
                    s, e, c = sb
                state = np.concatenate([state, [c], [s]])
                avail_num_FS += c
            if lenSB == 0:
                avg_avail_num_FS=0
            else:
                avg_avail_num_FS = avail_num_FS/lenSB #평균 가능 블록 사이즈
            state = np.concatenate([state, [numofslots], [avail_num_FS], [avg_avail_num_FS]])






        k_slot_state = []
        for k, path in enumerate(kpath):
            slot_state = [[maxvalue for i in range(self.TotalNumofSlots)] for r in range(20)]
            for i in range(len(path['path']) - 1):
                fromnode = path['path'][i]
                tonode = path['path'][i + 1]
                slot_2D = self.slot_info_2D[fromnode][tonode]
                for n in range(self.TotalNumofSlots):
                    if slot_2D[n][1]>0:
                        if slot_2D[n][1]-self.cur_time > maxvalue:
                            slot_state[i][n] = maxvalue
                        else:
                            slot_state[i][n]=slot_2D[n][1]-self.cur_time
                    else:
                        slot_state[i][n]=0
            k_slot_state += slot_state
        simg = np.array(k_slot_state)

        slotimg= np.expand_dims(simg, axis=0)
        slotimg= np.expand_dims(slotimg, axis=0)

        s = np.eye(30)[req.source]
        d = np.eye(30)[req.destination]

        bh = [req.holding_time, req.bandwidth]



        req_info = np.concatenate([s, d, bh, state])
        req_info = np.expand_dims(req_info, axis=0)

        return slotimg, req_info

    # state size 97
    def base_slot_presentation_v2_2_reference_modified_add_imgMap(self, kpath, req):  # 각 루트별로 일자로 state 구성
        # plt.figure(figsize=(5, 5), dpi=200, constrained_layout=True)
        # maxvalue = req.holding_time
        source = np.eye(self.NumOfNode)[req.source - 1]
        dest = np.eye(self.NumOfNode)[req.destination - 1]
        holdingtime = req.holding_time
        bh = [req.holding_time, req.bandwidth]

        state = np.concatenate([source, dest, [holdingtime/50]])  # 62

        send_kth_candi_SB = []

        j = self.num_of_subblock
        # 1번째 가능 블록 사이즈, 1번째 가능블록 시작 인덱스, 1번째 가능 블록 사이즈, 1번째 가능블록 시작 인덱스, 루트에서 요구 슬롯사이즈, 평균 가능 블록 사이즈, 총 가능한 fs 수.
        # for k in range(self.K):
        #     if len(kpath) > k:
        #         path = kpath[k]
        #         numofslots = self.modul_format(path, req)  # 루트에서 요구 슬롯사이즈
        #         baseslot = np.array(self.base_slot(path))
        #         candi_SB = []
        #
        #         for i in range(len(baseslot)):
        #             cnt = 0
        #             while cnt < numofslots and i + cnt < self.TotalNumofSlots:
        #                 if baseslot[i + cnt][0] == 0:
        #                     cnt += 1
        #                 else:
        #                     break
        #
        #             if cnt >= numofslots:
        #                 # print(i, cnt)
        #                 candi_SB.append((i, i + cnt - 1, cnt, numofslots, path))
        #
        #             if len(candi_SB) >= j:
        #                 break
        #
        #         lenSB = len(candi_SB)
        #         # print(k, candi_SB)
        #
        #         for addEmptSB_i in range(j - lenSB):
        #             candi_SB.append((0, 0, 0, 0, path))
        #
        #         send_kth_candi_SB.append(copy.copy(candi_SB))
        #
        #         avail_num_FS = 0  # 가능한 블록들의 총 FS 수
        #
        #         for i in range(j):
        #             s, _, c, _, _ = candi_SB[i]
        #             state = np.concatenate([state, [c], [s]])  # 2*j = 4  , 62+4 = 66
        #             avail_num_FS += c
        #
        #         if lenSB == 0:
        #             avg_avail_num_FS = 0
        #         else:
        #             avg_avail_num_FS = avail_num_FS / lenSB  # 평균 가능 블록 사이즈
        #
        #         state = np.concatenate([state, [numofslots], [avail_num_FS], [avg_avail_num_FS]])  # 66+3 = 69
        #
        #     else:
        #         for i in range(j):
        #             state = np.concatenate([state, [-1], [-1]])
        #         state = np.concatenate([state, [-1], [-1], [-1]])



        for k in range(self.K):
            if len(kpath)>k:
                path = kpath[k]
                pathhop = path['hop']
                numofslots = self.modul_format(path, req) # 루트에서 요구 슬롯사이즈
                baseslot = np.array(self.base_slot(path))

                contig_block = []
                slot_count = 0
                s = 0

                for i in range(len(baseslot)):
                    if baseslot[i][0] == 0:
                        if slot_count == 0:
                            s = i
                        slot_count += 1
                        if i == self.TotalNumofSlots - 1:
                            # if slot_count >= numofslots:
                            contig_block.append([s, s + slot_count - 1, slot_count])  # start, end, count,
                    elif baseslot[i][0] > 0:
                        if slot_count != 0:
                            # if slot_count >= numofslots:
                            contig_block.append([s, s + slot_count - 1, slot_count])
                            slot_count = 0


                candi_SB = []
                for sb in contig_block:
                    sidx, eidx, slen = sb
                    if slen < numofslots:
                        continue
                    if slen == numofslots:
                        candi_SB.append((sidx, eidx, slen, numofslots, path))
                    else:
                        candi_SB.append((sidx, sidx + numofslots - 1, slen, numofslots, path))
                        candi_SB.append((eidx - numofslots + 1, eidx, slen, numofslots, path))


                lenSB = len(candi_SB)
                for addEmptSB_i in range(j-lenSB):
                    candi_SB.append((-1, -1, 0, 0, path))

                send_kth_candi_SB.append(copy.copy(candi_SB))

                avail_num_FS = 0  #가능한 블록들의 총 FS 수

                for i in range(j):
                    idxs, idxe, c, _, _ = candi_SB[i]
                    # avail_num_FS += c
                    lt = 0 # 블럭 왼쪽 타임 벨류
                    rt = 0 # 블럭 오른쪽 타임 벨류
                    ls = 0
                    rs = 0

                    if idxs==0:
                        ls=1
                        rs = baseslot[idxe][0]/pathhop

                        lt = 1
                        if baseslot[idxe][1] == 0:
                            rt = 0
                        else:
                            rt = (baseslot[idxe][1] - self.cur_time) / holdingtime

                    elif idxe==self.TotalNumofSlots-1:
                        ls = baseslot[idxs][0]/pathhop
                        rs = 1

                        rt = 1
                        if baseslot[idxs][1]==0:
                            lt=0
                        else:
                            lt = (baseslot[idxs][1]-self.cur_time) / holdingtime

                    else:
                        ls = baseslot[idxs][0]/pathhop
                        rs = baseslot[idxe][0] / pathhop

                        if baseslot[idxs][1]==0:
                            lt=0
                        else:
                            lt = (baseslot[idxs][1]-self.cur_time) / holdingtime
                        if baseslot[idxe][1]==0:
                            rt=0
                        else:
                            rt = (baseslot[idxe][1]-self.cur_time) / holdingtime


                    # print(ls, rs, lt, rt)

                    state = np.concatenate([state, [c/10], [idxs/self.TotalNumofSlots], [ls], [rs], [lt], [rt]])  # 2*j = 4  , 62+4 = 66



                contig_block = np.array(contig_block)
                len_cont_block = len(contig_block)
                if len_cont_block:
                    num_empty_slots = contig_block.sum(axis=0)[2]
                    maxlen = contig_block.max(axis=0)[2]
                    minlen = contig_block.min(axis=0)[2]

                    avg_avail_num_FS = avail_num_FS/len_cont_block #평균 가능 블록 사이즈
                else:
                    num_empty_slots=0
                    avg_avail_num_FS=0
                    maxlen=0
                    minlen=0


                state = np.concatenate([state, [pathhop/10], [numofslots/10], [len_cont_block/self.TotalNumofSlots], [num_empty_slots/self.TotalNumofSlots], [avg_avail_num_FS/self.TotalNumofSlots], [maxlen/self.TotalNumofSlots], [minlen/self.TotalNumofSlots]])   # 66+3 = 69
                # state = np.concatenate([state, [pathhop], [numofslots], [len_cont_block], [num_empty_slots], [avg_avail_num_FS], [maxlen], [minlen]])   # 66+3 = 69

            else:
                # state = np.concatenate([state, [-1 for _ in range(self.TotalNumofSlots)]])  # 62
                for i in range(j):
                    state = np.concatenate([state, [0], [0], [0], [0], [0], [0]])
                state = np.concatenate([state, [0], [0], [0], [0], [0], [0], [0]])



        # req_info = np.concatenate([source, dest, bh])
        # req_info = np.expand_dims(req_info, axis=0)

        state = np.expand_dims(state, axis=0)
        # print(state.shape)
        # print(state)

        maxvalue = 50
        # maxvalue = req.holding_time
        k_slot_state = []
        for k in range(self.K):
            slot_state = [[maxvalue for i in range(self.TotalNumofSlots)] for r in range(10)]
            if len(kpath) > k:
                path = kpath[k]
                for i in range(len(path['path']) - 1):
                    fromnode = path['path'][i]
                    tonode = path['path'][i + 1]

                    slot_2D = self.slot_info_2D[fromnode][tonode]
                    for n in range(self.TotalNumofSlots):
                        if slot_2D[n][1] > 0:
                            if slot_2D[n][1] - self.cur_time > maxvalue:
                                slot_state[i][n] = maxvalue
                            else:
                                slot_state[i][n] = slot_2D[n][1] - self.cur_time
                        else:
                            slot_state[i][n] = 0
                    # slot_state.append(stime)

                #        print(baseslot_2D)
                # print(slot_state)
            k_slot_state += slot_state
        simg = np.array(k_slot_state)

        slotimg = np.expand_dims(simg, axis=0)
        slotimg = np.expand_dims(slotimg, axis=0)

        return slotimg, state, send_kth_candi_SB


    #state size 97
    def base_slot_presentation_v2_1_reference_modified(self, kpath, req): #각 루트별로 일자로 state 구성
        # plt.figure(figsize=(5, 5), dpi=200, constrained_layout=True)

        source = np.eye(30)[req.source]
        dest = np.eye(30)[req.destination]
        h = [req.holding_time]
        bh = [req.holding_time, req.bandwidth]

        state = np.concatenate([source, dest, bh]) # 62

        send_kth_candi_SB=[]

        j=self.num_of_subblock
        # 1번째 가능 블록 사이즈, 1번째 가능블록 시작 인덱스, 1번째 가능 블록 사이즈, 1번째 가능블록 시작 인덱스, 루트에서 요구 슬롯사이즈, 평균 가능 블록 사이즈, 총 가능한 fs 수.
        for k in range(self.K):
            if len(kpath)>k:
                path = kpath[k]
                numofslots = self.modul_format(path, req) # 루트에서 요구 슬롯사이즈
                baseslot = np.array(self.base_slot(path))
                candi_SB = []

                for i in range(len(baseslot)):
                    cnt = 0
                    while cnt < numofslots and i+cnt< self.TotalNumofSlots:
                        if baseslot[i + cnt][0] == 0:
                            cnt += 1
                        else:
                            break

                    if cnt >= numofslots:
                        # print(i, cnt)
                        candi_SB.append((i, i + cnt - 1, cnt, numofslots, path))

                    if len(candi_SB) >= j:
                        break

                lenSB = len(candi_SB)

                for addEmptSB_i in range(j-lenSB):
                    candi_SB.append((-1, -1, 0, 0, path))

                send_kth_candi_SB.append(copy.copy(candi_SB))
                avail_num_FS = 0  #가능한 블록들의 총 FS 수

                for i in range(j):
                    s, _, c, _, _ = candi_SB[i]
                    state = np.concatenate([state, [c], [s]])  # 2*j = 4  , 62+4 = 66
                    avail_num_FS += c

                if lenSB == 0:
                    avg_avail_num_FS=0
                else:
                    avg_avail_num_FS = avail_num_FS/lenSB #평균 가능 블록 사이즈

                state = np.concatenate([state, [numofslots], [avail_num_FS], [avg_avail_num_FS]])   # 66+3 = 69

            else:
                for i in range(j):
                    state = np.concatenate([state, [-1], [-1]])
                state = np.concatenate([state, [-1], [-1], [-1]])

        req_info = np.concatenate([source, dest, bh])
        req_info = np.expand_dims(req_info, axis=0)

        state = np.expand_dims(state, axis=0)

        return state, req_info, send_kth_candi_SB

    def base_slot_presentation_v2_3_reference_modified(self, kpath, req): # 각 서브블록의 holding time을 state로 준다.
        # plt.figure(figsize=(5, 5), dpi=200, constrained_layout=True)

        source = np.eye(self.NumOfNode)[req.source - 1]
        dest = np.eye(self.NumOfNode)[req.destination - 1]
        holdingtime = req.holding_time
        bh = [req.holding_time/50, req.bandwidth/125]

        state = np.concatenate([source, dest, bh]) # 62

        send_kth_candi_SB=[]

        j=self.num_of_subblock
        # 1번째 가능 블록 사이즈, 1번째 가능블록 시작 인덱스, 1번째 가능 블록 사이즈, 1번째 가능블록 시작 인덱스, 루트에서 요구 슬롯사이즈, 평균 가능 블록 사이즈, 총 가능한 fs 수.
        for k in range(self.K):
            if len(kpath)>k:
                path = kpath[k]
                pathhop = path['hop']
                numofslots = self.modul_format(path, req) # 루트에서 요구 슬롯사이즈
                baseslot = np.array(self.base_slot(path))
                contig_block = []
                slot_count = 0
                s = 0
                for i in range(len(baseslot)):
                    if baseslot[i][0] == 0:
                        if slot_count == 0:
                            s = i
                        slot_count += 1
                        if i == self.TotalNumofSlots - 1:
                            # if slot_count >= numofslots:
                            contig_block.append([s, s + slot_count - 1, slot_count])  # start, end, count,
                    elif baseslot[i][0] > 0:
                        if slot_count != 0:
                            # if slot_count >= numofslots:
                            contig_block.append([s, s + slot_count - 1, slot_count])
                            slot_count = 0




                candi_SB = []
                for i in range(len(baseslot)):
                    cnt = 0
                    while cnt < numofslots and i+cnt< self.TotalNumofSlots:
                        if baseslot[i + cnt][0] == 0:
                            cnt += 1
                        else:
                            break
                    if cnt >= numofslots:
                        candi_SB.append((i, i + cnt - 1, cnt, numofslots, path))
                    if len(candi_SB) >= j:
                        break

                lenSB = len(candi_SB)
                for addEmptSB_i in range(j-lenSB):
                    candi_SB.append((-1, -1, 0, 0, path))

                send_kth_candi_SB.append(copy.copy(candi_SB))

                avail_num_FS = 0  #가능한 블록들의 총 FS 수

                for i in range(j):
                    idxs, idxe, c, _, _ = candi_SB[i]
                    # avail_num_FS += c
                    lt = 0 # 블럭 왼쪽 타임 벨류
                    rt = 0 # 블럭 오른쪽 타임 벨류
                    ls = 0
                    rs = 0

                    if idxs==0:
                        ls=1
                        rs = baseslot[idxe][0]/pathhop

                        lt = 1
                        if baseslot[idxe][1] == 0:
                            rt = 0
                        else:
                            rt = (baseslot[idxe][1] - self.cur_time) / holdingtime

                    elif idxe==self.TotalNumofSlots-1:
                        ls = baseslot[idxs][0]/pathhop
                        rs = 1

                        rt = 1
                        if baseslot[idxs][1]==0:
                            lt=0
                        else:
                            lt = (baseslot[idxs][1]-self.cur_time) / holdingtime

                    else:
                        ls = baseslot[idxs][0]/pathhop
                        rs = baseslot[idxe][0] / pathhop

                        if baseslot[idxs][1]==0:
                            lt=0
                        else:
                            lt = (baseslot[idxs][1]-self.cur_time) / holdingtime
                        if baseslot[idxe][1]==0:
                            rt=0
                        else:
                            rt = (baseslot[idxe][1]-self.cur_time) / holdingtime


                    # print(ls, rs, lt, rt)

                    state = np.concatenate([state, [c], [idxs/self.TotalNumofSlots], [ls], [rs], [lt], [rt]])  # 2*j = 4  , 62+4 = 66



                contig_block = np.array(contig_block)
                len_cont_block = len(contig_block)
                if len_cont_block:
                    num_empty_slots = contig_block.sum(axis=0)[2]
                    maxlen = contig_block.max(axis=0)[2]
                    minlen = contig_block.min(axis=0)[2]

                    avg_avail_num_FS = avail_num_FS/len_cont_block #평균 가능 블록 사이즈
                else:
                    num_empty_slots=0
                    avg_avail_num_FS=0
                    maxlen=0
                    minlen=0


                state = np.concatenate([state, [pathhop/10], [numofslots/10], [len_cont_block/self.TotalNumofSlots], [num_empty_slots/self.TotalNumofSlots], [avg_avail_num_FS/self.TotalNumofSlots], [maxlen/self.TotalNumofSlots], [minlen/self.TotalNumofSlots]])   # 66+3 = 69

            else:
                for i in range(j):
                    state = np.concatenate([state, [-1], [-1], [-1], [-1], [-1], [-1]])
                state = np.concatenate([state, [-1], [-1], [-1], [-1], [-1], [-1], [-1]])





        req_info = np.concatenate([source, dest, bh])
        req_info = np.expand_dims(req_info, axis=0)

        state = np.expand_dims(state, axis=0)
        # print(state.shape)
        # print(state)
        return state, req_info, send_kth_candi_SB

    def base_slot_presentation_v2_4_reference_modified(self, kpath, req): # 각 패스에 베이스슬롯의 슬롯 점유상태를 표시
        # plt.figure(figsize=(5, 5), dpi=200, constrained_layout=True)

        source = np.eye(self.NumOfNode)[req.source - 1]
        dest = np.eye(self.NumOfNode)[req.destination - 1]
        holdingtime = req.holding_time
        bh = [req.holding_time/50, req.bandwidth/125]

        htime = [req.holding_time/50]
        state = np.concatenate([source, dest, htime]) # 62
        send_kth_candi_SB=[]

        j=self.num_of_subblock
        # 1번째 가능 블록 사이즈, 1번째 가능블록 시작 인덱스, 1번째 가능 블록 사이즈, 1번째 가능블록 시작 인덱스, 루트에서 요구 슬롯사이즈, 평균 가능 블록 사이즈, 총 가능한 fs 수.
        for k in range(self.K):
            if len(kpath)>k:
                path = kpath[k]
                pathhop = path['hop']
                numofslots = self.modul_format(path, req) # 루트에서 요구 슬롯사이즈
                baseslot = np.array(self.base_slot(path))

                state = np.concatenate([state, baseslot[:,0]])  # 62



                contig_block = []
                slot_count = 0
                s = 0

                for i in range(len(baseslot)):
                    if baseslot[i][0] == 0:
                        if slot_count == 0:
                            s = i
                        slot_count += 1
                        if i == self.TotalNumofSlots - 1:
                            # if slot_count >= numofslots:
                            contig_block.append([s, s + slot_count - 1, slot_count])  # start, end, count,
                    elif baseslot[i][0] > 0:
                        if slot_count != 0:
                            # if slot_count >= numofslots:
                            contig_block.append([s, s + slot_count - 1, slot_count])
                            slot_count = 0




                candi_SB = []
                for i in range(len(baseslot)):
                    cnt = 0
                    while cnt < numofslots and i+cnt< self.TotalNumofSlots:
                        if baseslot[i + cnt][0] == 0:
                            cnt += 1
                        else:
                            break
                    if cnt >= numofslots:
                        candi_SB.append((i, i + cnt - 1, cnt, numofslots, path))
                    if len(candi_SB) >= j:
                        break

                lenSB = len(candi_SB)
                for addEmptSB_i in range(j-lenSB):
                    candi_SB.append((-1, -1, 0, 0, path))

                send_kth_candi_SB.append(copy.copy(candi_SB))

                avail_num_FS = 0  #가능한 블록들의 총 FS 수

                for i in range(j):
                    idxs, idxe, c, _, _ = candi_SB[i]
                    # avail_num_FS += c
                    lt = 0 # 블럭 왼쪽 타임 벨류
                    rt = 0 # 블럭 오른쪽 타임 벨류
                    ls = 0
                    rs = 0

                    if idxs==0:
                        ls=1
                        rs = baseslot[idxe][0]/pathhop

                        lt = 1
                        if baseslot[idxe][1] == 0:
                            rt = 0
                        else:
                            rt = (baseslot[idxe][1] - self.cur_time) / holdingtime

                    elif idxe==self.TotalNumofSlots-1:
                        ls = baseslot[idxs][0]/pathhop
                        rs = 1

                        rt = 1
                        if baseslot[idxs][1]==0:
                            lt=0
                        else:
                            lt = (baseslot[idxs][1]-self.cur_time) / holdingtime

                    else:
                        ls = baseslot[idxs][0]/pathhop
                        rs = baseslot[idxe][0] / pathhop

                        if baseslot[idxs][1]==0:
                            lt=0
                        else:
                            lt = (baseslot[idxs][1]-self.cur_time) / holdingtime
                        if baseslot[idxe][1]==0:
                            rt=0
                        else:
                            rt = (baseslot[idxe][1]-self.cur_time) / holdingtime


                    # print(ls, rs, lt, rt)

                    state = np.concatenate([state, [c], [idxs/self.TotalNumofSlots], [ls], [rs], [lt], [rt]])  # 2*j = 4  , 62+4 = 66



                contig_block = np.array(contig_block)
                len_cont_block = len(contig_block)
                if len_cont_block:
                    num_empty_slots = contig_block.sum(axis=0)[2]
                    maxlen = contig_block.max(axis=0)[2]
                    minlen = contig_block.min(axis=0)[2]

                    avg_avail_num_FS = avail_num_FS/len_cont_block #평균 가능 블록 사이즈
                else:
                    num_empty_slots=0
                    avg_avail_num_FS=0
                    maxlen=0
                    minlen=0


                state = np.concatenate([state, [pathhop/10], [numofslots/10], [len_cont_block/self.TotalNumofSlots], [num_empty_slots/self.TotalNumofSlots], [avg_avail_num_FS/self.TotalNumofSlots], [maxlen/self.TotalNumofSlots], [minlen/self.TotalNumofSlots]])   # 66+3 = 69

            else:
                state = np.concatenate([state, [0 for _ in range(self.TotalNumofSlots)]])  # 62
                for i in range(j):
                    state = np.concatenate([state, [0], [0], [0], [0], [0], [0]])
                state = np.concatenate([state, [0], [0], [0], [0], [0], [0], [0]])





        req_info = np.concatenate([source, dest, bh])
        req_info = np.expand_dims(req_info, axis=0)

        state = np.expand_dims(state, axis=0)
        # print(state.shape)
        # print(state)
        return state, req_info, send_kth_candi_SB


    def base_slot_presentation_v2_5_reference_modified(self, kpath, req): # 각 패스에 베이스슬롯의 슬롯 점유상태를 표시 & 후보 서브블록 선택시, 가능한 후보블록에서 양쪽 사이트에 가능한 블록을 찾는다.
        # plt.figure(figsize=(5, 5), dpi=200, constrained_layout=True)
        source = np.eye(self.NumOfNode)[req.source-1]
        dest = np.eye(self.NumOfNode)[req.destination-1]
        holdingtime = req.holding_time
        bh = [req.holding_time/50, req.bandwidth/125]

        htime = [req.holding_time/self.holding_time]
        state = np.concatenate([source, dest, htime]) # 62
        send_kth_candi_SB=[]

        j=self.num_of_subblock
        # 1번째 가능 블록 사이즈, 1번째 가능블록 시작 인덱스, 1번째 가능 블록 사이즈, 1번째 가능블록 시작 인덱스, 루트에서 요구 슬롯사이즈, 평균 가능 블록 사이즈, 총 가능한 fs 수.
        for k in range(self.K):
            if len(kpath)>k:
                path = kpath[k]
                pathhop = path['hop']
                numofslots = self.modul_format(path, req) # 루트에서 요구 슬롯사이즈
                baseslot = np.array(self.base_slot(path))
                contig_block = []
                slot_count = 0
                s = 0

                for i in range(len(baseslot)):
                    if baseslot[i][0] == 0:
                        if slot_count == 0:
                            s = i
                        slot_count += 1
                        if i == self.TotalNumofSlots - 1:
                            # if slot_count >= numofslots:
                            contig_block.append([s, s + slot_count - 1, slot_count])  # start, end, count,
                    elif baseslot[i][0] > 0:
                        if slot_count != 0:
                            # if slot_count >= numofslots:
                            contig_block.append([s, s + slot_count - 1, slot_count])
                            slot_count = 0

                candi_SB = []
                for sb in contig_block:
                    sidx, eidx, slen = sb
                    if slen < numofslots:
                        continue
                    if slen == numofslots:
                        candi_SB.append((sidx, eidx, slen, numofslots, path))
                    else:
                        candi_SB.append((sidx, sidx + numofslots - 1, slen, numofslots, path))
                        candi_SB.append((eidx - numofslots + 1, eidx, slen, numofslots, path))

                lenSB = len(candi_SB)
                for addEmptSB_i in range(j-lenSB):
                    candi_SB.append((-1, -1, 0, 0, path))

                send_kth_candi_SB.append(copy.copy(candi_SB[:j]))
                avail_num_FS = 0  #가능한 블록들의 총 FS 수

                for i in range(j):
                    idxs, idxe, c, _, _ = candi_SB[i]

                    lt = 0 # 블럭 왼쪽 타임 벨류
                    rt = 0 # 블럭 오른쪽 타임 벨류
                    ls = 0
                    rs = 0

                    if c > 0:
                        if idxs==0:
                            ls=1
                            rs = baseslot[idxe][0]/pathhop

                            lt = 1
                            if baseslot[idxe][1] == 0:
                                rt = 0
                            else:
                                rt = (baseslot[idxe][1] - self.cur_time) / holdingtime

                        elif idxe==self.TotalNumofSlots-1:
                            ls = baseslot[idxs][0]/pathhop
                            rs = 1

                            rt = 1
                            if baseslot[idxs][1]==0:
                                lt=0
                            else:
                                lt = (baseslot[idxs][1]-self.cur_time) / holdingtime

                        else:
                            if idxs == -1:
                                print('error~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~', baseslot[idxs][:])
                            ls = baseslot[idxs][0]/pathhop
                            rs = baseslot[idxe][0] / pathhop

                            if baseslot[idxs][1]==0:
                                lt=0
                            else:
                                lt = (baseslot[idxs][1]-self.cur_time) / holdingtime
                            if baseslot[idxe][1]==0:
                                rt=0
                            else:
                                rt = (baseslot[idxe][1]-self.cur_time) / holdingtime

                        state = np.concatenate([state, [c/10], [idxs/self.TotalNumofSlots], [ls], [rs], [lt], [rt]])  # 2*j = 4  , 62+4 = 66
                    else:
                        state = np.concatenate([state, [-1], [-1], [-1], [-1], [-1], [-1]])

                contig_block = np.array(contig_block)
                len_cont_block = len(contig_block)
                if len_cont_block:
                    num_empty_slots = contig_block.sum(axis=0)[2]
                    maxlen = contig_block.max(axis=0)[2]
                    minlen = contig_block.min(axis=0)[2]

                    avg_avail_num_FS = avail_num_FS/len_cont_block #평균 가능 블록 사이즈
                else:
                    num_empty_slots=0
                    avg_avail_num_FS=0
                    maxlen=0
                    minlen=0
                state = np.concatenate([state, [pathhop/10], [numofslots/10], [len_cont_block/10], [num_empty_slots/10], [avg_avail_num_FS/10], [maxlen/10], [minlen/10]])   # 66+3 = 69
                # state = np.concatenate([state, [pathhop], [numofslots], [len_cont_block], [num_empty_slots], [avg_avail_num_FS], [maxlen], [minlen]])   # 66+3 = 69

            else:
                # state = np.concatenate([state, [-1 for _ in range(self.TotalNumofSlots)]])  # 62
                for i in range(j):
                    state = np.concatenate([state, [-1], [-1], [-1], [-1], [-1], [-1]])
                state = np.concatenate([state, [-1], [-1], [-1], [-1], [-1], [-1], [-1]])





        req_info = np.concatenate([source, dest, bh])
        req_info = np.expand_dims(req_info, axis=0)

        state = np.expand_dims(state, axis=0)
        # print(state.shape)
        # print(state)
        return state, req_info, send_kth_candi_SB

    def base_slot_presentation_BehaviorClone(self, kpath, req): # 각 패스에 베이스슬롯의 슬롯 점유상태를 표시 & 후보 서브블록 선택시, 가능한 후보블록에서 양쪽 사이트에 가능한 블록을 찾는다.
        # plt.figure(figsize=(5, 5), dpi=200, constrained_layout=True)
        # maxvalue = req.holding_time
        source = np.eye(self.NumOfNode)[req.source - 1]
        dest = np.eye(self.NumOfNode)[req.destination - 1]
        holdingtime = req.holding_time
        bh = [req.holding_time, req.bandwidth]

        state = np.concatenate([source, dest, [holdingtime / 50]])  # 62
        # print(state)
        send_kth_candi_SB = []

        # maxvalue = self.holding_time
        maxvalue = req.holding_time
        k_slot_state = []
        for k in range(self.K):
            # slot_state = [[maxvalue for i in range(self.TotalNumofSlots)] for r in range(10)]
            slot_state = [maxvalue for i in range(self.TotalNumofSlots)]
            if len(kpath) > k:
                path = kpath[k]
                for i in range(len(path['path']) - 1):
                    fromnode = path['path'][i]
                    tonode = path['path'][i + 1]
                    slot_2D = self.slot_info_2D[fromnode][tonode]
                    for n in range(self.TotalNumofSlots):
                        if slot_2D[n][1] > 0:
                            if slot_2D[n][1] - self.cur_time > maxvalue:
                                slot_state[n] = maxvalue
                            else:
                                slot_state[n] = slot_2D[n][1] - self.cur_time
                        else:
                            slot_state[n] = 0


            state = np.concatenate([state, slot_state])


        # req.req_print()
        # print(state)
        # print(state.shape)
        state = np.expand_dims(state, axis=0)
        # print(state.shape)

        req_info = np.concatenate([source, dest, bh])
        req_info = np.expand_dims(req_info, axis=0)

        return state, req_info, send_kth_candi_SB







    def spec_assin_2D_forBC(self, candi_SB, req):  # best fit
        # candi_SB consist of (sidx, eidx, empty block length, req num of slots, path)

        flag = 0
        idxmax = -1
        candi_SB = np.array(candi_SB).reshape((-1,5))

        # print('Candi Len', len(candi_SB))
        # print(candi_SB)

        SpBlock = []
        for idxaction, sb in enumerate(candi_SB):
            # print(sb)
            sb_start, sb_end, blk_length, nofs, path = sb

            numofslots = self.modul_format(path, req)
            baseslot = self.base_slot(path)
            hop = len(path['path']) - 1
            endtime = req.end_time
            holdtime = req.holding_time

            if nofs >= numofslots:
                cost_S = 0
                cost_T = 0

                if sb_start == 0:
                    cost_S = (hop - baseslot[sb_start + numofslots][0]) / hop
                    cost_T = abs(endtime - baseslot[sb_start + numofslots][1]) / holdtime
                    if cost_T > 1:
                        cost_T = 1
                elif sb_start > 0 and sb_end < self.TotalNumofSlots-1:
                    cost_S = ((hop - baseslot[sb_start - 1][0]) + (hop - baseslot[sb_start + numofslots][0])) / hop
                    cost_T_l = abs(endtime - baseslot[sb_start - 1][1]) / holdtime
                    cost_T_r = abs(endtime - baseslot[sb_start + numofslots][1]) / holdtime
                    if cost_T_l > 1:
                        cost_T_l = 1
                    if cost_T_r > 1:
                        cost_T_r = 1
                    cost_T = cost_T_l + cost_T_r
                elif sb_end == self.TotalNumofSlots-1:
                    cost_S = (hop - baseslot[sb_start - 1][0]) / hop
                    cost_T = abs(endtime - baseslot[sb_start - 1][1]) / holdtime
                    if cost_T > 1:
                        cost_T = 1
                SpBlock.append((sb_start, sb_end, cost_S, cost_T, cost_S + cost_T))
            else:
                SpBlock.append((-1, -1, 100, 100, 100))

        SpBlock = np.array(SpBlock)
        # print(SpBlock)
        # print(SpBlock.argmin(axis=0)[4])
        idxmax = SpBlock.argmin(axis=0)[4]
        # print(candi_SB[idxmax])
        # s

        # req.req_print()
        sss, eee, elength, reqnumofslots, fpath = candi_SB[idxmax]
        # print(SpBlock[idxmax])
        # print('selected action sb',candi_SB[idxmax])


        flag = self.specslot_assign_specific(fpath, req, candi_SB[idxmax])
        # print(flag)

        return flag, idxmax, fpath









    # state size 86,87
    def base_slot_presentation_v2_reference(self, kpath, req):  # 각 루트별로 일자로 state 구성
        # plt.figure(figsize=(5, 5), dpi=200, constrained_layout=True)

        source = np.eye(self.NumOfNode)[req.source-1]
        dest = np.eye(self.NumOfNode)[req.destination-1]
        h = [req.holding_time/self.holding_time]
        bh = [req.holding_time, req.bandwidth]

        state = np.concatenate([source, dest, h])
        send_kth_candi_SB = []

        j=self.num_of_subblock
        # 1번째 가능 블록 사이즈, 1번째 가능블록 시작 인덱스, 1번째 가능 블록 사이즈, 1번째 가능블록 시작 인덱스, 루트에서 요구 슬롯사이즈, 평균 가능 블록 사이즈, 총 가능한 fs 수.
        for k in range(self.K):
            if len(kpath) > k:
                path = kpath[k]
                numofslots = self.modul_format(path, req)  # 루트에서 요구 슬롯사이즈
                baseslot = np.array(self.base_slot(path))
                candi_SB = []
                s, c = 0, 0
                avail_num_FS = 0  # 가능한 블록들의 총 FS 수
                for i in range(len(baseslot)):
                    if baseslot[i][0] == 0:
                        if c == 0:
                            s = i
                        c += 1
                        avail_num_FS += 1
                        if i == self.TotalNumofSlots - 1:
                            if c >= numofslots:
                                candi_SB.append((s, s + c - 1, c, numofslots, path))  # start, end, count,

                    elif baseslot[i][0] > 0:
                        if c != 0:
                            if c >= numofslots:
                                candi_SB.append((s, s + c - 1, c, numofslots, path))
                            c = 0

                lenSB = len(candi_SB)

                for addEmptSB_i in range(self.num_of_subblock-lenSB):
                    candi_SB.append((0, 0, 0, numofslots, path))



                send_kth_candi_SB.append(copy.copy(candi_SB[:self.num_of_subblock]))



                for sb in candi_SB[:self.num_of_subblock]:
                    s, _, c, _, _ = sb
                    state = np.concatenate([state, [c/10], [s/10]])  # 2*j = 4  , 62+4 = 66



                if lenSB>0:
                    avgsize = avail_num_FS/lenSB
                else:
                    avgsize = 0

                # print(candi_SB)
                # print(avail_num_FS)
                # print(avgsize)

                # req.req_print()
                # print(baseslot[:,0])
                # print('numofslots ', numofslots)
                # print('lenSB ', lenSB)
                # print('avgsize ', avgsize)
                # print('avail_num_FS ', avail_num_FS)
                # print()

                state = np.concatenate([state, [numofslots/10], [avgsize/10], [avail_num_FS/10]])  # 66+3 = 69


            else:
                print('No candi path.')
                for i in range(self.num_of_subblock):
                    state = np.concatenate([state, [0], [0]])
                state = np.concatenate([state, [0], [0], [0]])

        req_info = np.concatenate([source, dest, bh])
        req_info = np.expand_dims(req_info, axis=0)

        state = np.expand_dims(state, axis=0)
        # print(state.shape)
        # print(state)
        return state, req_info, send_kth_candi_SB












    def base_slot_presentation_v1(self, kpath, req): #각 루트별로 일자로 state 구성
        # plt.figure(figsize=(5, 5), dpi=200, constrained_layout=True)

        # maxvalue = 50
        maxvalue = req.holding_time

        k_slot_state = []
        for k, path in enumerate(kpath):
            # plt.subplot(pltloc)
            basslot = np.array(self.base_slot(path))
            k_slot_state = np.concatenate([k_slot_state, basslot[:,0]])
            k_slot_state = np.concatenate([k_slot_state, basslot[:,1]])


        s = np.eye(30)[req.source]
        d = np.eye(30)[req.destination]
        bh = [req.holding_time, req.bandwidth]

        req_info = np.concatenate([s, d, bh])
        req_info = np.expand_dims(req_info, axis=0)

        state = np.concatenate([s, d, bh, k_slot_state])
        state = np.expand_dims(state, axis=0)
        # print(state.shape)
        return state, req_info



    def base_slot_presentation(self, kpath, req):
        # plt.figure(figsize=(5, 5), dpi=200, constrained_layout=True)
        maxvalue = 50
        # maxvalue = req.holding_time
        k_slot_state = []

        for k in range(self.K):

            slot_state = [[maxvalue for i in range(self.TotalNumofSlots)] for r in range(10)]
            if len(kpath)>k:
                path = kpath[k]
                for i in range(len(path['path']) - 1):
                    fromnode = path['path'][i]
                    tonode = path['path'][i + 1]

                    slot_2D = self.slot_info_2D[fromnode][tonode]
                    for n in range(self.TotalNumofSlots):
                        if slot_2D[n][1]>0:
                            if slot_2D[n][1]-self.cur_time > maxvalue:
                                slot_state[i][n] = maxvalue
                            else:
                                slot_state[i][n]=slot_2D[n][1]-self.cur_time
                        else:
                            slot_state[i][n]=0
                    # slot_state.append(stime)

                #        print(baseslot_2D)
                # print(slot_state)
            k_slot_state += slot_state
        simg = np.array(k_slot_state)
        # slot = np.expand_dims(slot, axis=0)
        # slot = np.expand_dims(slot, axis=0)

        # print(slot)
        # simg = torch.tensor(slot, dtype=torch.float32)
        # print(simg.shape)
        slotimg= np.expand_dims(simg, axis=0)
        slotimg= np.expand_dims(slotimg, axis=0)

        s = np.eye(30)[req.source]
        d = np.eye(30)[req.destination]
        # print(d)
        # b = torch.eye(10)[int(req.destination/self.SlotWidth)-1]
        # h = torch.tensor([req.holding_time], dtype=torch.float32)

        # bh = torch.tensor([req.holding_time, req.bandwidth], dtype=torch.float32)
        bh = [req.holding_time, req.bandwidth]

        # req_info = torch.cat([s, d, bh])


        # req_info = torch.cat([s, d, bh]).unsqueeze(0)
        req_info = np.concatenate([s, d, bh])
        req_info = np.expand_dims(req_info, axis=0)



        # req_info = req_info

        # print(np.shape(slotimg))
        # print(np.shape(req_info))

        # state = (slotimg, req_info)

        # gamma = 0.99  # discount factor
        # K_epochs = 32  # update policy for K epochs
        # n_pro = 8
        # update_timestep = 200
        # eps_clip = 0.1  # clip parameter for PPO
        # lr_actor = 0.0001  # learning rate for actor network
        # lr_critic = 0.0001  # learning rate for critic network
        # print(state)
        # model = PPO(1, 15, lr_actor,  gamma, K_epochs, eps_clip)
        # re = model.policy.pi(state)
        #
        # # model = CNN()
        # # re = model(state)
        #
        # print(re)
        # print(re.size())
        # plt.axis('off')
        # img = plt.imshow(slot, cmap='gray')
        # # result = re.detach().numpy()
        # # img = plt.imshow(result, cmap='gray')
        # plt.savefig(fname='image.jpg', bbox_inches='tight', pad_inches=0)

        # print(slotimg)

        # if req.id >5000:
        #     plt.imshow(simg, cmap='gray')
        #     plt.show()
        return slotimg, req_info




    def first_fit(self, flag, path, req, baseslot, numofslots):
        cnt = 0
        idx = 0
        while cnt != numofslots:
            if idx + cnt >= self.TotalNumofSlots:
                break
            if baseslot[idx + cnt][0] == 0:
                cnt += 1
            else:
                idx = idx + 1
                cnt = 0

        if cnt == numofslots:
            req.slot_start = idx
            req.slot_end = idx + cnt - 1
            req.state = 1
            req.hop = path['hop']
            flag = 1
        else:
            req.state = 0
            flag = 0
        return flag

    def first_fit_v_partition(self, flag, path, req, baseslot, numofslots, range_start, range_end):
        cnt = 0
        idx = range_start
        while cnt != numofslots:
            if idx + cnt > range_end:
                break
            if baseslot[idx + cnt][0] == 0:
                cnt += 1
            else:
                idx = idx + 1
                cnt = 0

        if cnt == numofslots:
            req.slot_start = idx
            req.slot_end = idx + cnt - 1
            req.state = 1
            req.hop = path['hop']
            flag = 1
        else:
            req.state = 0
            flag = 0

        return flag



    def last_fit(self, flag, path, req, baseslot, numofslots):

        cnt = 0
        idx = self.TotalNumofSlots-1

        while cnt != numofslots:
            if idx - cnt < 0:
                break
            if baseslot[idx - cnt][0] == 0:
                cnt += 1
            else:
                idx = idx - 1
                cnt = 0

        if cnt == numofslots:
            req.slot_start = idx-cnt+1
            req.slot_end = idx
            req.state = 1
            req.hop = path['hop']
            flag = 1
        else:
            req.state = 0
            flag = 0

        return flag



    def last_fit_v_partition(self, flag, path, req, baseslot, numofslots, range_start, range_end):

        cnt = 0
        idx = range_end

        while cnt != numofslots:
            if idx - cnt < range_start:
                break
            if baseslot[idx - cnt][0] == 0:
                cnt += 1
            else:
                idx = idx - 1
                cnt = 0

        if cnt == numofslots:
            req.slot_start = idx-cnt+1
            req.slot_end = idx
            req.state = 1
            req.hop = path['hop']
            flag = 1
        else:
            req.state = 0
            flag = 0

        return flag






    def best_fit_v_partition(self, flag, path, req, baseslot, numofslots, range_start, range_end):

        cnt = 0
        idx = range_start

        while cnt != numofslots:
            if idx + cnt > range_end:
                break
            if baseslot[idx + cnt][0] == 0:
                cnt += 1
            else:
                idx = idx + 1
                cnt = 0

        if cnt == numofslots:
            req.slot_start = idx
            req.slot_end = idx + cnt - 1
            req.state = 1
            req.hop = path['hop']
            flag = 1
        else:
            req.state = 0
            flag = 0

        return flag

    def EOSA(self, flag, path, req, baseslot, width, range_start, range_end):

        idx = range_start
        maxwidth = self.range_band_e
        #        req.req_print()
        #        print('width', width)
        for p in range(self.TotalNumofSlots):
            cnt = 0
            #            print('p  ',p)
            if width < (maxwidth / 2):
                if (p % maxwidth == 0) or (p % maxwidth == maxwidth / 2 - width):
                    idx = p

                    for n in range(width):
                        if baseslot[idx + cnt][0] == 0:
                            cnt += 1
                        else:
                            idx = idx + 1
                            cnt = 0
                            break
                    if cnt == width:
                        break
                    #                else:

            elif (width == maxwidth / 2):
                if (p % (maxwidth / 2) == 0):
                    idx = p
                    for n in range(width):
                        if baseslot[idx + cnt][0] == 0:
                            cnt += 1
                        else:
                            idx = idx + 1
                            cnt = 0
                            break
                    if cnt == width:
                        break
                    #                else:

            else:
                if (p % maxwidth == 0) or (p % maxwidth == maxwidth - width):
                    idx = p
                    for n in range(width):
                        if baseslot[idx + cnt][0] == 0:
                            cnt += 1
                        else:
                            idx = idx + 1
                            cnt = 0
                            break
                    if cnt == width:
                        break

        if cnt == width:

            req.slot_start = idx
            req.slot_end = idx + cnt - 1
            req.state = 1
            req.hop = path['hop']
            flag = 1
        else:
            req.state = 0
            flag = 0

        return flag

    def NFSA(self, flag, path, req, baseslot, width, range_start, range_end):

        idx = range_start
        maxwidth = self.range_band_e
        #        req.req_print()
        #        print('ID', req.id, 'width', width)
        #        print(baseslot)
        for p in range(self.TotalNumofSlots):
            cnt = 0
            #            print('p  ',p)
            if width > (maxwidth / 2):  # width가 5보다 큰 경우
                if p % maxwidth == maxwidth - width:
                    idx = p

                    for n in range(width):
                        if baseslot[idx + cnt][0] == 0:
                            cnt += 1
                        else:
                            idx = idx + 1
                            cnt = 0
                            break
                    if cnt == width:
                        break
                    #                else:

            elif (width == maxwidth / 2):  # width가 5이고
                if (p % (maxwidth / 2) == 0):  # p도 5의 배수인 경우
                    idx = p
                    for n in range(width):
                        if baseslot[idx + cnt][0] == 0:
                            cnt += 1
                        else:
                            idx = idx + 1
                            cnt = 0
                            break
                    if cnt == width:
                        break
                    #                else:

            else:
                idx = p
                #                print('else idx  ', idx)
                while cnt != width:
                    if idx + cnt >= self.TotalNumofSlots:
                        break
                    if baseslot[idx + cnt][0] == 0:
                        cnt += 1
                    else:
                        idx = idx + 1
                        cnt = 0
                if cnt == width:
                    break

        if cnt == width:
            #            print(idx, '  ' ,idx+cnt-1 )
            req.slot_start = idx
            req.slot_end = idx + cnt - 1
            req.state = 1
            req.hop = path['hop']
            flag = 1
        else:
            req.state = 0
            flag = 0

        return flag

    def release(self, cur_time):
        #        print('rel cur_time ', cur_time)
        self.req_in_service.sort(key=lambda object: object.end_time)
        #        print('B', len(self.req_in_service))

        n = 0
        while n < len(self.req_in_service):
            temp = self.req_in_service[n]
            #            temp.req_print()
            if temp.end_time < cur_time:
                #                temp.req_print()
                ios = temp.slot_start
                ioe = temp.slot_end
                path = temp.path
                for i in range(len(path['path']) - 1):
                    fromnode = path['path'][i]
                    tonode = path['path'][i + 1]
                    #                    print(fromnode,' ', tonode,' ', ios,' ', ioe )
                    #                    slot = self.slot_info[fromnode][tonode]
                    slot_2D = self.slot_info_2D[fromnode][tonode]
                    #                    print(slot_2D)
                    for z in range(ios, ioe + 1):
                        if slot_2D[z][0] == 1:
                            slot_2D[z] = [0, 0]
                        else:
                            print('here? error')
                #                    print(slot)
                self.req_in_service.pop(n)
            #                print('rm ')
            else:
                n += 1


if __name__ == "__main__":
    NOB = 0

    sim = Simulation(100000, 3, 0 , 15, 50, 1, 10, 320, 'DJP', 'FLEF')
    sim.sim_main()

