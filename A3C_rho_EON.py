# import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import torch.multiprocessing as mp
from simulation_v2 import Simulation
import matplotlib.pyplot as plt
import os
import datetime
import numpy as np

import pandas as pd

################################## set device ##################################

env_version=2
n_pro = 16

num_kpath = 5
num_Subblock = 1
N_A = num_kpath*num_Subblock
statesize = 54

# num_kpath = 5
# num_Subblock = 2
# N_A = num_kpath*num_Subblock
# statesize = 64
#
# num_kpath = 5
# num_Subblock = 1
# N_A = num_kpath*num_Subblock
# statesize = 54

hiddensize = 128
num_EP = 10000
timestep_max = 1000
update_interval = 100

learning_rate = 0.00001
gamma = 0.9
global alpha
alpha = 0.01

TotalNumofSlots = 100

N_S = 100

num_of_req = 10000
num_of_warmup = 0
ar = 10
dif = 1
avg_holding = 20
range_band_s = 1
range_band_e = 10

# eipsilon = 0.05

rt_name = 'KSP'
sa_name = 'DRL'

print("============================================================================================")

# set device to cpu or cuda
device = torch.device('cpu')

# if (torch.cuda.is_available() and n_pro==1):
#     device = torch.device('cuda:0')
#     torch.cuda.empty_cache()
#     print("Device set to : " + str(torch.cuda.get_device_name(device)))
# else:
#     device = torch.device('cpu')
#     print("Device set to : cpu")

print("============================================================================================")



basepath = 'result\\'

def createFolder(directory):
    try:
        if not os.path.isdir(directory):
            os.makedirs(directory)
    except OSError:
        print('error')

# class ActorCritic(nn.Module):  #  for v2_2
#     def __init__(self, state_dim, action_dim):
#         super(ActorCritic, self).__init__()
#
#         self.layer1 = torch.nn.Sequential(
#             torch.nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
#             torch.nn.ReLU(),
#             torch.nn.MaxPool2d(kernel_size=2, stride=2))
#
#         self.layer2 = torch.nn.Sequential(
#             torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
#             torch.nn.ReLU(),
#             torch.nn.MaxPool2d(kernel_size=2, stride=2))
#
#         self.layer3 = torch.nn.Sequential(
#             torch.nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
#             torch.nn.ReLU(),
#             torch.nn.MaxPool2d(kernel_size=2, stride=2))
#
#         self.layer4 = torch.nn.Sequential(
#             torch.nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
#             torch.nn.ReLU(),
#             torch.nn.MaxPool2d(kernel_size=2, stride=2))
#
#         self.fc1 = nn.Linear(385, 256)
#         self.fc2 = nn.Linear(256, 256)
#         self.fc3 = nn.Linear(256, 256)
#         self.fc4 = nn.Linear(256, 256)
#         self.fc5 = nn.Linear(256, 256)
#         self.fc6 = nn.Linear(256, 256)
#         self.fc_pi = nn.Linear(256, action_dim)
#         self.fc_v = nn.Linear(256, 1)
#
#     def pi(self, state, req_info, softmax_dim=0):
#         out = self.layer1(state)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = self.layer4(out)
#         out = out.view(out.size(0), -1)
#         # print(out.shape)
#         in_data = torch.cat([out, req_info], dim=1)
#         x = F.relu(self.fc1(in_data))
#         x = F.relu(self.fc2(x))
#         x = F.relu(self.fc3(x))
#         x = F.relu(self.fc4(x))
#         x = F.relu(self.fc5(x))
#         x = F.relu(self.fc6(x))
#         x = self.fc_pi(x)
#         # print(x, x.size())
#         prob = F.softmax(x, dim=1)
#         # print(prob, prob.size())
#         return prob
#
#     def v(self, state, req_info):
#         out = self.layer1(state)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = self.layer4(out)
#         out = out.view(out.size(0), -1)
#         in_data = torch.cat([out, req_info], dim=1)
#         x = F.relu(self.fc1(in_data))
#         x = F.relu(self.fc2(x))
#         x = F.relu(self.fc3(x))
#         x = F.relu(self.fc4(x))
#         x = F.relu(self.fc5(x))
#         x = F.relu(self.fc6(x))
#         v = self.fc_v(x)
#         return v


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

    def pi(self, state, req_info, softmax_dim=0):

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

    def v(self, state, req_info):

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

# class ActorCritic(nn.Module):
#     def __init__(self, state_dim, action_dim):
#         super(ActorCritic, self).__init__()
#
#         self.layer1 = torch.nn.Sequential(
#             torch.nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
#             torch.nn.ReLU(),
#             torch.nn.MaxPool2d(kernel_size=2, stride=2))
#         # 25*25*16
#         self.layer2 = torch.nn.Sequential(
#             torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
#             torch.nn.ReLU(),
#             torch.nn.MaxPool2d(kernel_size=2, stride=2))
#         # 12*12*32
#         self.layer3 = torch.nn.Sequential(
#             torch.nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
#             torch.nn.ReLU(),
#             torch.nn.MaxPool2d(kernel_size=2, stride=2))
#         # 6*6*32
#
#         self.layer4 = torch.nn.Sequential(
#             torch.nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
#             torch.nn.ReLU(),
#             torch.nn.MaxPool2d(kernel_size=2, stride=2))
#
#         # self.fc1 = nn.Linear(350, 512)
#         self.fc1 = nn.Linear(2428, 1024)
#         self.fc2 = nn.Linear(1024, 512)
#         self.fc3 = nn.Linear(512, 256)
#         self.fc4 = nn.Linear(256, 128)
#         self.fc_pi = nn.Linear(128, action_dim)
#         self.fc_v = nn.Linear(128, 1)
#
#     def pi(self, state, req_info, softmax_dim=0):
#         out = self.layer1(state)
#         # print(out.size())
#         out = self.layer2(out)
#         # print(out.size())
#         out = self.layer3(out)
#         # print(out.size())
#         # out = self.layer4(out)
#         out = out.view(out.size(0), -1)
#         in_data = torch.cat([out, req_info], dim=1)
#         # print(out.size())
#         # print(in_data.size())
#
#         x = F.relu(self.fc1(in_data))
#         x = F.relu(self.fc2(x))
#         x = F.relu(self.fc3(x))
#         x = F.relu(self.fc4(x))
#         x = self.fc_pi(x)
#         prob = F.softmax(x, dim=1)
#         # print(prob, prob.size())
#         return prob
#
#     def v(self, state, req_info):
#         out = self.layer1(state)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         # out = self.layer4(out)
#         out = out.view(out.size(0), -1)
#         in_data = torch.cat([out, req_info], dim=1)
#         x = F.relu(self.fc1(in_data))
#         x = F.relu(self.fc2(x))
#         x = F.relu(self.fc3(x))
#         x = F.relu(self.fc4(x))
#         v = self.fc_v(x)
#         return v





# class ActorCritic(nn.Module):
#     def __init__(self, state_dim, action_dim):
#         super(ActorCritic, self).__init__()
#
#         def conv_bn(inp, oup, stride):
#             return nn.Sequential(
#                 nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
#                 nn.BatchNorm2d(oup),
#                 nn.ReLU(inplace=True)
#             )
#
#         def conv_dw(inp, oup, stride):
#             return nn.Sequential(
#                 # dw
#                 nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
#                 nn.BatchNorm2d(inp),
#                 nn.ReLU(inplace=True),
#
#                 # pw
#                 nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
#                 nn.BatchNorm2d(oup),
#                 nn.ReLU(inplace=True),
#             )
#
#         self.model = nn.Sequential(
#             conv_bn(1, 16, 2),
#             conv_dw(16, 32, 1),
#             conv_dw(32, 64, 2),
#             conv_dw(64, 64, 1),
#             conv_dw(64, 128, 2),
#             conv_dw(128, 256, 1),
#             conv_dw(256, 512, 2),
#             # conv_dw(512, 512, 1),
#             # conv_dw(512, 512, 1),
#             # conv_dw(512, 512, 1),
#             # conv_dw(512, 512, 1),
#             conv_dw(512, 512, 1),
#             # conv_dw(512, 1024, 2),
#             # conv_dw(1024, 1024, 1),
#             nn.AdaptiveAvgPool2d(1)
#         )
#         # self.fc1 = nn.Linear(350, 512)
#         # self.fc1 = nn.Linear(1214, 1024)
#         # self.fc2 = nn.Linear(1024, 512)
#         # self.fc3 = nn.Linear(512, 256)
#         # self.fc4 = nn.Linear(256, 128)
#         self.fc_pi = nn.Linear(574, action_dim)
#         self.fc_v = nn.Linear(574, 1)
#
#     def pi(self, state, req_info, softmax_dim=0):
#         out = self.model(state)
#         # print(out.size())
#
#         out = out.view(out.size(0), -1)
#         in_data = torch.cat([out, req_info], dim=1)
#         # print(out.size())
#         # print(in_data.size())
#
#         # x = F.relu(self.fc1(in_data))
#         # x = F.relu(self.fc2(x))
#         # x = F.relu(self.fc3(x))
#         # x = F.relu(self.fc4(x))
#         x = self.fc_pi(in_data)
#         prob = F.softmax(x, dim=1)
#         # print(prob, prob.size())
#         return prob
#
#     def v(self, state, req_info):
#         out = self.model(state)
#
#         out = out.view(out.size(0), -1)
#         in_data = torch.cat([out, req_info], dim=1)
#
#         v = self.fc_v(in_data)
#         return v



def train(global_model, rank, res_queue, timestep_max, dirpath):

    local_model = ActorCritic(N_S, N_A)
    local_model.load_state_dict(global_model.state_dict())
    optimizer = optim.Adam(global_model.parameters(), lr=learning_rate)


    env = Simulation(num_of_req, num_kpath, N_A, num_of_warmup, ar, avg_holding, range_band_s,
                     range_band_e, TotalNumofSlots,
                     rt_name, sa_name, dirpath, env_version)


    env.env_init()
    timestep = 0
    for e in range(num_EP):
        done = False
        state, req_info, req, kth_cand_sb = env.env_reset()
        # print(state)
        ept_score = 0
        succ_req, blk_req = 0, 0
        entropy = 0
        actlist = []
        # global alpha
        # if (e+1)%50==0:
        #     if alpha>0.1:
        #         alpha=alpha-0.01
        #     else:
        #         alpha=0.0


        while not done:
            s_lst, req_ist, a_lst, r_lst = [], [], [], []
            for t in range(update_interval):
                prob = local_model.pi(torch.from_numpy(state).float().to(device), torch.from_numpy(req_info).float().to(device))
                m = Categorical(prob)

                action = m.sample().item()
                entropy += m.entropy().item()

                # k_path = env.KSP_routing(req, all_k_path)
                # env.spec_assin_Slot(path, req)
                #
                # for path in k_path:
                #     flag = env.spec_assin_2D(path, req)
                #     if flag:
                #         break


                next_state, next_req_info, next_req, r, done, next_kth_cand_sb = env.env_step(req, action, kth_cand_sb)




                if r<0:
                    for sbb in kth_cand_sb:
                        for sb in sbb:
                            sss,eee,ccc,nnn,ppp = sb
                            if ccc > 0:
                                r -= 1

                # if r<0 and e>10:
                #     print('===================================================')
                #     req.req_print()
                #     print(state)
                #     print(action)
                #     for sbb in kth_cand_sb:
                #         for sb in sbb:
                #             print(sb)
                #     print(r)
                #     print('===================================================')

                ept_score += r

                actlist.append(action)

                s_lst.append(state)
                req_ist.append(req_info)
                a_lst.append([action])
                r_lst.append(r)

                state = next_state
                req_info = next_req_info
                req = next_req
                kth_cand_sb = next_kth_cand_sb

                if r > 0:
                    succ_req += 1
                else:
                    blk_req += 1

                timestep += 1
                # print('timestep: ',timestep)
                if timestep % timestep_max == 0:
                    done = 1
                    res_queue.put((ept_score, succ_req, blk_req, entropy / timestep_max))
                    # log_rwd.append(ept_score)
                    # log_succ_req.append(succ_req)
                    # log_blk_req.append(blk_req)
                    # log_entropy.append(entropy/timestep_max)
                    print('Epi: ', e, '   Score: ' ,ept_score,' Succ. Req: ' ,succ_req, ' Blk. Req: ' ,blk_req, '   BBP: ' ,blk_req/timestep_max, '  Epi avg Entropy: {0:0.4}'.format(entropy/timestep_max), 'alpha: ', alpha)


                    if (e)%50==49:
                        # plt.hist(actlist, bins=N_A-1, density=True, label='CDF', histtype='step')
                        # plt.show()
                        # print(prob*timestep_max)
                        bincnt = np.bincount(actlist)
                        print(bincnt)

                    break

            s_final = torch.tensor(next_state, dtype=torch.float)
            req_final = torch.tensor(next_req_info, dtype=torch.float)
            R = r if done else local_model.v(s_final, req_final).item()
            td_target_lst = []
            for reward in r_lst[::-1]:
                R = gamma * R + reward
                td_target_lst.append([R])
            td_target_lst.reverse()
            # print(td_target_lst)
            s_batch, req_batch, a_batch, td_target = torch.tensor(s_lst, dtype=torch.float).squeeze(1), \
                                                     torch.tensor(req_ist, dtype=torch.float).squeeze(1), \
                                                     torch.tensor(a_lst), \
                                                     torch.tensor(td_target_lst)
            # print('td_target', td_target.shape)

            # print(s_batch.size())
            advantage = td_target - local_model.v(s_batch, req_batch)

            pi = local_model.pi(s_batch, req_batch, softmax_dim=1)
            pi_a = pi.gather(1, a_batch)
            # print(a_batch)
            dist = Categorical(pi)
            dist_entropy = dist.entropy()

            # print(pi_a.shape)
            # print(s_batch.shape)
            # print(req_batch.shape)

            # loss = -torch.log(pi_a) * advantage.detach() + F.smooth_l1_loss(local_model.v(s_batch, req_batch), td_target.detach())
            loss = -torch.log(pi_a) * advantage.detach() + F.smooth_l1_loss(local_model.v(s_batch, req_batch), td_target.detach())- alpha * dist_entropy

            # print('loss', loss.shape)

            optimizer.zero_grad()
            loss.mean().backward()
            for global_param, local_param in zip(global_model.parameters(), local_model.parameters()):
                global_param._grad = local_param.grad
            optimizer.step()
            local_model.load_state_dict(global_model.state_dict())

    print("Training process {} reached maximum episode.".format(rank))
    res_queue.put(None)


if __name__ == '__main__':
    global_model = ActorCritic(100, N_A)
    global_model.share_memory()

    now_start = datetime.datetime.now()
    resultdir = '{0:02}-{1:02} {2:02}-{3:02}_A3C_Env={14}_p={5}_EP={15}k_K={6}_SB={7}_eplen={8}k_uint={9}k_lr={10}_gm={11}_alp={12}_hid={13}'.format(
        now_start.month, now_start.day, now_start.hour, now_start.minute, now_start.second, n_pro,num_kpath, num_Subblock, timestep_max/1000, update_interval/1000,
        learning_rate, gamma, alpha, hiddensize, env_version, num_EP/1000)

    dirpath = os.path.join(basepath, resultdir)
    createFolder(dirpath)
    print(dirpath)

    pth_model_path = os.path.join(dirpath, 'model')
    createFolder(pth_model_path)

    setting_data = {}
    setting_file = 'setting.csv'
    setting_data["Date"] = now_start
    setting_data["n_pro"] = n_pro
    setting_data["num_EP"] = num_EP
    setting_data["num_kpath"] = num_kpath
    setting_data["num_Subblock"] = num_Subblock
    setting_data["hiddensize"] = hiddensize
    setting_data["timestep_max"] = timestep_max
    setting_data["update_interval"] = update_interval
    setting_data["learning_rate"] = learning_rate
    setting_data["gamma"] = gamma
    setting_data["alpha"] = alpha
    setting_data["num_of_warmup"] = num_of_warmup
    setting_data["ar"] = ar
    setting_data["dif"] = dif
    setting_data["avg_holding"] = avg_holding
    setting_data["range_band_s"] = range_band_s
    setting_data["range_band_e"] = range_band_e
    setting_data["num_of_slots"] = TotalNumofSlots

    setting_file_path = os.path.join(dirpath, setting_file)
    print('Print setting data ', setting_file_path)
    # print(setting_data)
    # , columns=['Date', 'num_of_senario', 'num_of_req', 'num_of_warmup', 'num_kpath', 'ar', 'dif', 'avg_holding', 'range_band_s', 'range_band_e', 'num_of_slots']
    setting_pdata = pd.DataFrame([setting_data])
    setting_pdata.to_csv(setting_file_path, index=False)



    res_queue = mp.Queue()

    processes = []
    for rank in range(n_pro):  # + 1 for test process
        # if rank == 0:
        #     p = mp.Process(target=test, args=(global_model,))
        # else:
        p = mp.Process(target=train, args=(global_model, rank, res_queue, timestep_max, dirpath))
        p.start()
        processes.append(p)

    log_avg_ept_score = []
    log_avg_succ_req = []
    log_avg_blk_req = []
    log_avg_BP_req = []
    log_avg_entropy = []

    log_tmp_avg_ept_score = []
    log_tmp_avg_succ_req = []
    log_tmp_avg_blk_req = []
    log_tmp_vg_entropy = []

    flag = 0
    cnt = 0
    while True:
        result = res_queue.get()
        cnt += 1
        if result is not None:
            ept_score, succ_req, blk_req, entropy = result

            log_tmp_avg_ept_score.append(ept_score)
            log_tmp_avg_succ_req.append(succ_req)
            log_tmp_avg_blk_req.append(blk_req)
            log_tmp_vg_entropy.append(entropy)

            log_avg_ept_score.append(ept_score)
            log_avg_succ_req.append(succ_req)
            log_avg_blk_req.append(blk_req)
            log_avg_BP_req.append(blk_req / timestep_max)
            log_avg_entropy.append(entropy)

            avg_blk = 0
            if cnt % 50 == 0:
                # avg_ept_score = sum(log_tmp_avg_ept_score) / len(log_tmp_avg_ept_score)
                # avg_succ_req = sum(log_tmp_avg_succ_req) / len(log_tmp_avg_succ_req)
                # avg_blk_req = sum(log_tmp_avg_blk_req) / len(log_tmp_avg_blk_req)
                # avg_avg_entropy = sum(log_tmp_vg_entropy) / len(log_tmp_vg_entropy)
                #
                # log_tmp_avg_ept_score.clear()
                # log_tmp_avg_succ_req.clear()
                # log_tmp_avg_blk_req.clear()
                # log_tmp_vg_entropy.clear()
                #
                # log_avg_ept_score.append(avg_ept_score)
                # log_avg_succ_req.append(avg_succ_req)
                # log_avg_blk_req.append(avg_blk_req)
                # log_avg_BP_req.append(avg_blk_req / timestep_max)
                # log_avg_entropy.append(avg_avg_entropy)

                # avg_blk = avg_blk_req

                # print(cnt, 'Avg. Reward: ', avg_ept_score)

                training_time = datetime.datetime.now() - now_start

                plt.title('Training avg Reward plot: {}'.format(training_time))
                plt.xlabel('Epi')
                plt.ylabel('Reward')
                plt.plot(log_avg_ept_score, 'b')
                plt.grid(True, axis='y')
                fig = plt.gcf()
                fig.savefig('{}/train avg eptscores.png'.format(dirpath), facecolor='white', dpi=600)
                plt.clf()

                plt.title('Succ req')
                plt.xlabel('Epi')
                plt.ylabel('# of req')
                plt.plot(log_avg_succ_req, 'b')
                plt.grid(True, axis='y')
                fig = plt.gcf()
                fig.savefig('{}/train succ req.png'.format(dirpath), facecolor='white', dpi=600)
                plt.clf()

                plt.title('Blocked req')
                plt.xlabel('Epi')
                plt.ylabel('# of req')
                plt.plot(log_avg_blk_req, 'b')
                plt.grid(True, axis='y')
                fig = plt.gcf()
                fig.savefig('{}/train blk req.png'.format(dirpath), facecolor='white', dpi=600)
                plt.clf()

                plt.title('blocking prob')
                plt.xlabel('Epi')
                plt.ylabel('prob')
                plt.plot(log_avg_BP_req, 'b')
                plt.grid(True, axis='y')
                fig = plt.gcf()
                fig.savefig('{}/train blocking prob.png'.format(dirpath), facecolor='white', dpi=600)
                plt.clf()

                plt.title('Entropy')
                plt.xlabel('Epi')
                plt.ylabel('entropy')
                plt.plot(log_avg_entropy, 'b')
                plt.grid(True, axis='y')
                fig = plt.gcf()
                fig.savefig('{}/train entropy.png'.format(dirpath), facecolor='white', dpi=600)
                plt.clf()

            if cnt%100 == 0:
                result_data = {}
                result_data['log_avg_ept_score'] = log_avg_ept_score
                result_data['log_avg_succ_req'] = log_avg_succ_req
                result_data['log_avg_blk_req'] = log_avg_blk_req
                result_data['log_avg_BP_req'] = log_avg_BP_req
                result_data['log_avg_entropy'] = log_avg_entropy

                result_file_name = 'result_A3C_Env={14}_Pro={5}_EP={15}k_K={6}_SB={7}_eplen={8}k_uint={9}k_lr={10}_gm={11}_alp={12}_hid={13}.csv'.format(now_start.month, now_start.day,
                                                                                      now_start.hour, now_start.minute,
                                                                                      now_start.second, n_pro,  num_kpath, num_Subblock, timestep_max/1000, update_interval/1000, learning_rate,gamma,alpha, hiddensize, env_version, num_EP/1000)




                result_file_path = os.path.join(dirpath, result_file_name)
                result_data = pd.DataFrame(result_data)
                print((result_file_path))
                print(len(result_file_path))
                result_data.to_csv(result_file_path, index=False)

                if cnt == num_EP:

                    for p in processes:
                        p.terminate()

                    break



            # if cnt % 1000 == 0:
            #     file_name = "A3C_{:05d}_{}.pth".format(cnt, avg_blk)
            #     file_path = os.path.join(pthpath, file_name)
            #     # global_model.save(file_path)
            #     torch.save(global_model.state_dict(), file_path)

        else:
            flag += 1
            if flag == n_pro:

                break

    file_name = "A3C_Env={14}_Pro={5}_EP={15}k_K={6}_SB={7}_eplen={8}k_uint={9}k_lr={10}_gm={11}_alp={12}_hid={13}.pth".format(
        now_start.month, now_start.day,
        now_start.hour, now_start.minute,
        now_start.second, n_pro, num_kpath, num_Subblock, timestep_max / 1000, update_interval / 1000, learning_rate,
        gamma, alpha, hiddensize, env_version, num_EP / 1000)
    file_path = os.path.join(pth_model_path, file_name)
    # global_model.save(file_path)
    torch.save(global_model.state_dict(), file_path)

    for p in processes:
        p.join()