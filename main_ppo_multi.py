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
# from ConvPPO_single import PPO as PPOcon
from PPO_ppangyo_EON import ActorCritic
from PPO_ppangyo_EON import PPO_MULTI as PPO_MULTI_EON
from PPO_ppangyo_EON import PPO as PPO_EON
import torch.multiprocessing as mp
# from multiprocessing import Manager
import torch
from torch.distributions import Categorical
import torch.optim as optim
import torch.nn.functional as F


n_pro = 1
env_version=2
N_S = 100
num_kpath = 5
num_Subblock = 1
N_A = num_kpath*num_Subblock
statesize = 54
hiddensize = 128
alpha = 0.01

learning_rate = 0.00001
gamma = 0.90
lmbda = 0.95

eps_clip = 0.1
K_epoch = 16
num_EP = 5000
timestep_max = 10000 #ep 샘플수
update_interval = 10000 # 샘플 수집
TotalNumofSlots = 100
num_of_req = 60000
num_of_warmup = 10000
ar = 10
dif = 1
avg_holding = 20
range_band_s = 1
range_band_e = 10

rt_name = 'KSP'
sa_name = 'DRL'
i = 0

# ################################## set device ##################################

print("============================================================================================")

# set device to cpu or cuda
device = torch.device('cpu')

if (torch.cuda.is_available() and n_pro==1):
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    device = torch.device('cpu')
    print("Device set to : cpu")

print("============================================================================================")





basepath = 'result'

def createFolder(directory):
    try:
        if not os.path.isdir(directory):
            os.makedirs(directory)
    except OSError:
        print('error')



# def put_data(transition, data):
#     data.append(transition)
#
# def make_batch(data):
#     s_lst, req_ist, a_lst, r_lst, s_prime_lst, req_prime_ist, prob_a_lst, done_lst = [], [], [], [], [], [], [], []
#     for transition in data:
#         s, req, a, r, s_prime, req_prime, prob_a, done = transition
#
#         s_lst.append(s)
#         req_ist.append(req)
#         a_lst.append([a])
#         r_lst.append([r])
#         s_prime_lst.append(s_prime)
#         req_prime_ist.append(req_prime)
#         prob_a_lst.append([prob_a])
#         done_mask = 0 if done else 1
#         done_lst.append([done_mask])
#
#     s, req, a, r, s_prime, req_prime, done_mask, prob_a = torch.tensor(s_lst, dtype=torch.float).to(device).squeeze(1), \
#                                            torch.tensor(req_ist, dtype=torch.float).to(device).squeeze(1), \
#                                            torch.tensor(a_lst).to(device), \
#                                           torch.tensor(r_lst, dtype=torch.float).to(device), \
#                                           torch.tensor(s_prime_lst, dtype=torch.float).to(device).squeeze(1), \
#                                           torch.tensor(req_prime_ist, dtype=torch.float).to(device).squeeze(1), \
#                                           torch.tensor(done_lst, dtype=torch.float).to(device), \
#                                           torch.tensor(prob_a_lst).to(device)
#     data = []
#
#     return s, req, a, r, s_prime, req_prime, done_mask, prob_a

#
# def train(global_model, rank, res_queue):
#
#
#
#     local_model = ActorCritic(N_S, N_A)
#     # optimizer = optim.Adam(local_model.parameters(), lr=learning_rate)
#
#     local_model.load_state_dict(global_model.state_dict())
#     optimizer = optim.Adam(global_model.parameters(), lr=learning_rate)
#
#
#     env = Simulation(num_of_req, num_kpath, num_of_warmup, (ar + (i) * dif), avg_holding, range_band_s,
#                      range_band_e, TotalNumofSlots,
#                      rt_name, sa_name)
#
#     env.env_init()
#     timestep = 0
#
#     for e in range(num_EP):
#         # print('Epi-----', e)
#         done = False
#         state, req_info, req_cur = env.env_reset()
#
#         ept_score = 0
#         succ_req, blk_req = 0, 0
#         entropy = 0
#         actlist = []
#
#         cnt_interval=0
#         while done != 1:
#             cnt_interval+=1
#             s_lst, req_cur_lst, a_lst, r_lst, s_prime_lst, req_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], [], [], []
#             # print('interval', cnt_interval)
#             for t in range(update_interval):
#
#                 prob = local_model.pi(torch.from_numpy(state).float().to(device), torch.from_numpy(req_info).float().to(device))
#                 m = Categorical(prob)
#                 action = m.sample().item()
#                 actlist.append(action)
#                 entropy += m.entropy().item()
#                 next_state, req_next_info, req_next, reward, done = env.env_step(req_cur, action)
#                 ept_score += reward
#
#                 s_lst.append(state)
#                 req_cur_lst.append(req_info)
#                 a_lst.append([action])
#                 r_lst.append([reward])
#                 s_prime_lst.append(next_state)
#                 req_prime_lst.append(req_next_info)
#                 done_lst.append([done])
#                 prob_a_lst.append([prob[0, action].item()])
#
#
#                 if reward >= 1:
#                     succ_req += 1
#                 else:
#                     blk_req += 1
#
#                 state = next_state
#                 req_info = req_next_info
#                 req_cur = req_next
#
#                 timestep += 1
#                 # print('tst max')
#                 if timestep % timestep_max == 0:
#                     done = 1
#                     res_queue.put((ept_score, succ_req, blk_req, entropy / timestep_max))
#
#                     print('Epi: ', e, '   Score: ', ept_score, '   BBP: ', blk_req / timestep_max,
#                           '  Epi avg Entropy: ', entropy / timestep_max)
#
#
#             # print(np.shape(s_lst))
#             # print(np.shape(req_cur_lst))
#             # print(np.shape(a_lst))
#             # print(np.shape(r_lst))
#             # print(np.shape(s_prime_lst))
#             # print(np.shape(req_prime_lst))
#             # print(np.shape(done_lst))
#             # print(np.shape(prob_a_lst))
#
#             state_bat, req_curinfo_bat, action_bat, reward_bat, state_prime_bat, req_nextinfo_bat, done_mask_bat, prob_action_bat = torch.tensor(s_lst, dtype=torch.float).to(device).squeeze(1), \
#                                                                   torch.tensor(req_cur_lst, dtype=torch.float).to(
#                                                                       device).squeeze(1), \
#                                                                   torch.tensor(a_lst).to(device), \
#                                                                   torch.tensor(r_lst, dtype=torch.float).to(device), \
#                                                                   torch.tensor(s_prime_lst, dtype=torch.float).to(
#                                                                       device).squeeze(1), \
#                                                                   torch.tensor(req_prime_lst, dtype=torch.float).to(
#                                                                       device).squeeze(1), \
#                                                                   torch.tensor(done_lst, dtype=torch.float).to(device), \
#                                                                   torch.tensor(prob_a_lst).to(device)
#             for _ in range(K_epoch):
#                 td_target = reward_bat + gamma * local_model.v(state_prime_bat, req_nextinfo_bat) * done_mask_bat
#                 delta = td_target - local_model.v(state_bat, req_curinfo_bat)
#                 delta = delta.cpu().detach().numpy()
#
#                 advantage_lst = []
#                 advantage = 0.0
#                 for delta_t in delta[::-1]:
#                     advantage = gamma * lmbda * advantage + delta_t[0]
#                     advantage_lst.append([advantage])
#                 advantage_lst.reverse()
#                 advantage = torch.tensor(advantage_lst, dtype=torch.float).to(device)
#
#                 pi = local_model.pi(state_bat, req_curinfo_bat, softmax_dim=1)
#                 pi_a = pi.gather(1, action_bat)  # dim — dimension along to collect values, index — tensor with indices of values to collect
#                 # pi_a: action 확률, 과거 샘플로 현재 폴리시의 액션확률,
#                 ratio = torch.exp(torch.log(pi_a) - torch.log(prob_action_bat))  # action_bat/b == exp(log(action_bat)-log(b))
#
#                 dist = Categorical(pi)
#                 dist_entropy = dist.entropy()
#
#                 surr1 = ratio * advantage
#                 surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantage
#                 # loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(local_model.v(state_bat, req_curinfo_bat), td_target.detach())
#                 loss = -torch.min(surr1, surr2) + 0.5*F.smooth_l1_loss(local_model.v(state_bat, req_curinfo_bat), td_target.detach())- alpha * dist_entropy
#
#                 optimizer.zero_grad()
#                 loss.mean().backward()
#                 for gp, lp  in zip(global_model.parameters(), local_model.parameters()):
#                     gp._grad = lp.grad
#                 optimizer.step()
#                 local_model.load_state_dict(global_model.state_dict())
#
#                 del advantage
#                 del advantage_lst
#                 torch.cuda.empty_cache()
#
#             del state_bat, req_curinfo_bat, action_bat, reward_bat, state_prime_bat, req_nextinfo_bat, done_mask_bat, prob_action_bat
#
#             # print('')
#     res_queue.put(None)


def ppoTrainSingle_ppangyo():
    # print('run ')
    ################################## set device ##################################
    print("============================================================================================")
    # set device to cpu or cuda
    device = torch.device('cpu')
    if (torch.cuda.is_available()):
        device = torch.device('cuda:0')
        torch.cuda.empty_cache()
        print("Device set to : " + str(torch.cuda.get_device_name(device)))
    else:
        print("Device set to : cpu")
    print("============================================================================================")

    now_start = datetime.datetime.now()
    # resultdir = '{0:02}-{1:02} {2:02}-{3:02} {4:02}_PPOSingle'.format(
    #     now_start.month, now_start.day, now_start.hour, now_start.minute, now_start.second)

    resultdir = '{0:02}-{1:02} {2:02}-{3:02}_PPOsg_Env={14}_p={5}_EP={15}k_K={6}_SB={7}_eplen={8}k_uint={9}k_lr={10}_gm={11}_alp={12}_hid={13}'.format(
        now_start.month, now_start.day, now_start.hour, now_start.minute, now_start.second, n_pro, num_kpath,
        num_Subblock, timestep_max / 1000, update_interval / 1000,
        learning_rate, gamma, alpha, hiddensize, env_version, num_EP / 1000)

    dirpath = os.path.join(basepath, resultdir)
    createFolder(dirpath)
    print(dirpath)

    pthpath = os.path.join(dirpath, 'model')
    createFolder(pthpath)
    model = PPO_EON(statesize, hiddensize, N_S, N_A, K_epoch, gamma, alpha, lmbda, eps_clip, learning_rate)
    env = Simulation(num_of_req, num_kpath, N_A, num_of_warmup, ar, avg_holding, range_band_s,
                     range_band_e, TotalNumofSlots,
                     rt_name, sa_name, dirpath, env_version)

    env.env_init()
    timestep = 0

    log_entropy = []
    log_rwd = []
    log_succ_req = []
    log_blk_req = []
    log_bp = []

    for e in range(num_EP):

        done = False
        state, req_info, req, kth_cand_sb = env.env_reset()
        ept_score = 0
        succ_req, blk_req = 0, 0
        entropy=0
        actlist=[]
        while done != 1:
            for t in range(update_interval):
                prob = model.lnet.pi(torch.from_numpy(state).float().to(device), torch.from_numpy(req_info).float().to(device))
                m = Categorical(prob)
                action = m.sample().item()
                # print('item', action)
                # action = m.sample()
                # action = action.numpy()
                # print('noitem',action)
                actlist.append(action)
                entropy += m.entropy().item()

                next_state, next_req_info, next_req, reward, done, next_kth_cand_sb = env.env_step(req, action, kth_cand_sb)

                if reward<0:
                    for sbb in kth_cand_sb:
                        for sb in sbb:
                            sss,eee,ccc,nnn,ppp = sb
                            if ccc > 0:
                                reward -= 1

                ept_score += reward

                if reward>=0:
                    succ_req += 1
                else:
                    blk_req += 1

                model.put_data((state, req_info, action, reward, next_state, next_req_info, prob[0, action].item(), done))

                state = next_state
                req_info = next_req_info
                req = next_req
                kth_cand_sb = next_kth_cand_sb

                timestep += 1
                if timestep % timestep_max == 0:
                    done = 1


                    log_rwd.append(ept_score)
                    log_succ_req.append(succ_req)
                    log_blk_req.append(blk_req)
                    log_bp.append(blk_req/timestep_max)
                    log_entropy.append(entropy/timestep_max)
                    print('Epi: ', e, '   Score: ' ,ept_score,'   BBP: ' ,blk_req/timestep_max, '  Epi avg Entropy: ' , entropy/timestep_max)
                    # print(actlist)

                    if (e)%50==49:
                        bincnt = np.bincount(actlist)
                        print(bincnt)

            model.train_net()

        if e%10==1:
            training_time = datetime.datetime.now() - now_start

            plt.title('Training avg Reward plot: {}'.format(training_time))
            plt.xlabel('Epi')
            plt.ylabel('Reward')
            plt.plot(log_rwd, 'b')
            plt.grid(True, axis='y')
            fig = plt.gcf()
            fig.savefig('{}/train avg eptscores.png'.format(dirpath), facecolor='white', dpi=600)
            plt.clf()

            plt.title('Succ req')
            plt.xlabel('Epi')
            plt.ylabel('# of req')
            plt.plot(log_succ_req, 'b')
            plt.grid(True, axis='y')
            fig = plt.gcf()
            fig.savefig('{}/train succ req.png'.format(dirpath), facecolor='white', dpi=600)
            plt.clf()

            plt.title('Blocked req')
            plt.xlabel('Epi')
            plt.ylabel('# of req')
            plt.plot(log_blk_req, 'b')
            plt.grid(True, axis='y')
            fig = plt.gcf()
            fig.savefig('{}/train blk req.png'.format(dirpath), facecolor='white', dpi=600)
            plt.clf()

            plt.title('blocking prob')
            plt.xlabel('Epi')
            plt.ylabel('prob')
            plt.plot(np.array(log_blk_req)/timestep_max, 'b')
            plt.grid(True, axis='y')
            fig = plt.gcf()
            fig.savefig('{}/train blocking prob.png'.format(dirpath), facecolor='white', dpi=600)
            plt.clf()

            plt.title('Entropy')
            plt.xlabel('Epi')
            plt.ylabel('entropy')
            plt.plot(log_entropy, 'b')
            plt.grid(True, axis='y')
            fig = plt.gcf()
            fig.savefig('{}/train entropy.png'.format(dirpath), facecolor='white', dpi=600)
            plt.clf()
        if e % 1000 == 0:
            file_name = "PPO_single_{:05d}_{}.pth".format(e, blk_req)
            file_path = os.path.join(pthpath, file_name)
            # model.save(file_path)
            torch.save(model.lnet.state_dict(), file_path)

        if e%10 == 0:


            result_data = {}
            result_data['log_avg_ept_score'] = log_rwd
            result_data['log_avg_succ_req'] = log_succ_req
            result_data['log_avg_blk_req'] = log_blk_req
            result_data['log_avg_BP_req'] = log_bp
            result_data['log_avg_entropy'] = log_entropy
            result_file_name = 'result_PPO_Env={14}_Pro={5}_EP={15}k_K={6}_SB={7}_eplen={8}k_uint={9}k_lr={10}_gm={11}_alp={12}_hid={13}.csv'.format(now_start.month, now_start.day,
                                                                                  now_start.hour, now_start.minute,
                                                                                  now_start.second, n_pro,  num_kpath, num_Subblock, timestep_max/1000, update_interval/1000, learning_rate,gamma,alpha, hiddensize, env_version, num_EP/1000)

            result_file_path = os.path.join(dirpath, result_file_name)
            result_data = pd.DataFrame(result_data)
            print((result_file_path))
            print(len(result_file_path))
            result_data.to_csv(result_file_path, index=False)

    # writer.close()

# def ppoTrainSingle_ppangyo_multi():
#
#
#     gnet = ActorCritic(N_S, N_A)
#     gnet.share_memory()
#     # optim = SharedAdam(gnet.parameters(), lr=1e-4, betas=(0.92, 0.999))
#
#     now_start = datetime.datetime.now()
#     resultdir = '{0:02}-{1:02} {2:02}-{3:02} {4:02}_PPOMulti'.format(
#         now_start.month, now_start.day, now_start.hour, now_start.minute, now_start.second)
#     dirpath = os.path.join(basepath, resultdir)
#     createFolder(dirpath)
#     print(dirpath)
#
#     pthpath = os.path.join(dirpath, 'model')
#     createFolder(pthpath)
#
#     start_time = time.time()
#     num = mp.Value('i', 0)
#     res_queue = mp.Queue()
#
#     agents = [PPO_MULTI_EON(id, N_S, N_A,  learning_rate, gamma, lmbda, eps_clip, K_epoch, res_queue, gnet) for id
#               in range(n_pro)]
#     [ag.start() for ag in agents]
#
#     log_avg_ept_score = []
#     log_avg_succ_req = []
#     log_avg_blk_req = []
#     log_avg_BP_req = []
#     log_avg_entropy = []
#
#     log_tmp_avg_ept_score = []
#     log_tmp_avg_succ_req = []
#     log_tmp_avg_blk_req = []
#     log_tmp_vg_entropy = []
#
#     flag = 0
#     cnt = 0
#     while True:
#         result = res_queue.get()
#         cnt += 1
#         if result is not None:
#             ept_score, succ_req, blk_req, entropy = result
#
#             log_tmp_avg_ept_score.append(ept_score)
#             log_tmp_avg_succ_req.append(succ_req)
#             log_tmp_avg_blk_req.append(blk_req)
#             log_tmp_vg_entropy.append(entropy)
#
#
#             if cnt % n_pro == 0:
#                 avg_ept_score = sum(log_tmp_avg_ept_score) / len(log_tmp_avg_ept_score)
#                 avg_succ_req = sum(log_tmp_avg_succ_req) / len(log_tmp_avg_succ_req)
#                 avg_blk_req = sum(log_tmp_avg_blk_req) / len(log_tmp_avg_blk_req)
#                 avg_avg_entropy = sum(log_tmp_vg_entropy) / len(log_tmp_vg_entropy)
#
#                 log_tmp_avg_ept_score.clear()
#                 log_tmp_avg_succ_req.clear()
#                 log_tmp_avg_blk_req.clear()
#                 log_tmp_vg_entropy.clear()
#
#                 log_avg_ept_score.append(avg_ept_score)
#                 log_avg_succ_req.append(avg_succ_req)
#                 log_avg_blk_req.append(avg_blk_req)
#                 log_avg_BP_req.append(avg_blk_req/1000)
#                 log_avg_entropy.append(avg_avg_entropy)
#
#                 print(cnt, 'Avg. Reward: ', avg_ept_score)
#
#                 training_time = datetime.datetime.now() - now_start
#
#                 plt.title('Training avg Reward plot: {}'.format(training_time))
#                 plt.xlabel('Epi')
#                 plt.ylabel('Reward')
#                 plt.plot(log_avg_ept_score, 'b')
#                 plt.grid(True, axis='y')
#                 fig = plt.gcf()
#                 fig.savefig('{}/train avg eptscores.png'.format(dirpath), facecolor='white', dpi=600)
#                 plt.clf()
#
#                 plt.title('Succ req')
#                 plt.xlabel('Epi')
#                 plt.ylabel('# of req')
#                 plt.plot(log_avg_succ_req, 'b')
#                 plt.grid(True, axis='y')
#                 fig = plt.gcf()
#                 fig.savefig('{}/train succ req.png'.format(dirpath), facecolor='white', dpi=600)
#                 plt.clf()
#
#                 plt.title('Blocked req')
#                 plt.xlabel('Epi')
#                 plt.ylabel('# of req')
#                 plt.plot(log_avg_blk_req, 'b')
#                 plt.grid(True, axis='y')
#                 fig = plt.gcf()
#                 fig.savefig('{}/train blk req.png'.format(dirpath), facecolor='white', dpi=600)
#                 plt.clf()
#
#                 plt.title('blocking prob')
#                 plt.xlabel('Epi')
#                 plt.ylabel('prob')
#                 plt.plot(log_avg_BP_req, 'b')
#                 plt.grid(True, axis='y')
#                 fig = plt.gcf()
#                 fig.savefig('{}/train blocking prob.png'.format(dirpath), facecolor='white', dpi=600)
#                 plt.clf()
#
#                 plt.title('Entropy')
#                 plt.xlabel('Epi')
#                 plt.ylabel('entropy')
#                 plt.plot(log_avg_entropy, 'b')
#                 plt.grid(True, axis='y')
#                 fig = plt.gcf()
#                 fig.savefig('{}/train entropy.png'.format(dirpath), facecolor='white', dpi=600)
#                 plt.clf()
#
#             if cnt % 1000 == 0:
#                 file_name = "PPO_multi_{:05d}_{}.pth".format(cnt, avg_blk_req)
#                 file_path = os.path.join(pthpath, file_name)
#                 # model.save(file_path)
#                 torch.save(gnet.state_dict(), file_path)
#
#         else:
#             flag += 1
#             if flag == n_pro:
#                 break
#
#     [ag.join() for ag in agents]
#
#
#
#     elapsed_time = time.time() - start_time
#
#     print(num.value)
#     print(elapsed_time)
#
# def ppo_pppangyo_multi_new():
#
#     global_model = ActorCritic(100, N_A)
#     global_model.share_memory()
#
#     now_start = datetime.datetime.now()
#     resultdir = '{0:02}-{1:02} {2:02}-{3:02} {4:02}_PPO_npro {5}'.format(
#         now_start.month, now_start.day, now_start.hour, now_start.minute, now_start.second, n_pro)
#     dirpath = os.path.join(basepath, resultdir)
#     createFolder(dirpath)
#     print(dirpath)
#     pthpath = os.path.join(dirpath, 'model')
#     createFolder(pthpath)
#     res_queue = mp.Queue()
#
#     processes = []
#     for rank in range(n_pro):  # + 1 for test process
#         print(rank, ' Processor')
#         p = mp.Process(target=train, args=(global_model, rank, res_queue,))
#         p.start()
#         processes.append(p)
#
#     log_avg_ept_score = []
#     log_avg_succ_req = []
#     log_avg_blk_req = []
#     log_avg_BP_req = []
#     log_avg_entropy = []
#
#     log_tmp_avg_ept_score = []
#     log_tmp_avg_succ_req = []
#     log_tmp_avg_blk_req = []
#     log_tmp_vg_entropy = []
#
#     flag = 0
#     cnt = 0
#     while True:
#         result = res_queue.get()
#         cnt += 1
#         if result is not None:
#             ept_score, succ_req, blk_req, entropy = result
#
#             log_tmp_avg_ept_score.append(ept_score)
#             log_tmp_avg_succ_req.append(succ_req)
#             log_tmp_avg_blk_req.append(blk_req)
#             log_tmp_vg_entropy.append(entropy)
#
#             avg_blk = 0
#             if cnt % n_pro == 0:
#                 avg_ept_score = sum(log_tmp_avg_ept_score) / len(log_tmp_avg_ept_score)
#                 avg_succ_req = sum(log_tmp_avg_succ_req) / len(log_tmp_avg_succ_req)
#                 avg_blk_req = sum(log_tmp_avg_blk_req) / len(log_tmp_avg_blk_req)
#                 avg_avg_entropy = sum(log_tmp_vg_entropy) / len(log_tmp_vg_entropy)
#
#                 log_tmp_avg_ept_score.clear()
#                 log_tmp_avg_succ_req.clear()
#                 log_tmp_avg_blk_req.clear()
#                 log_tmp_vg_entropy.clear()
#
#                 log_avg_ept_score.append(avg_ept_score)
#                 log_avg_succ_req.append(avg_succ_req)
#                 log_avg_blk_req.append(avg_blk_req)
#                 log_avg_BP_req.append(avg_blk_req / timestep_max)
#                 log_avg_entropy.append(avg_avg_entropy)
#
#                 avg_blk = avg_blk_req
#
#                 print(cnt, 'Avg. Reward: ', avg_ept_score)
#
#                 training_time = datetime.datetime.now() - now_start
#
#                 plt.title('Training avg Reward plot: {}'.format(training_time))
#                 plt.xlabel('Epi')
#                 plt.ylabel('Reward')
#                 plt.plot(log_avg_ept_score, 'b')
#                 plt.grid(True, axis='y')
#                 fig = plt.gcf()
#                 fig.savefig('{}/train avg eptscores.png'.format(dirpath), facecolor='white', dpi=600)
#                 plt.clf()
#
#                 plt.title('Succ req')
#                 plt.xlabel('Epi')
#                 plt.ylabel('# of req')
#                 plt.plot(log_avg_succ_req, 'b')
#                 plt.grid(True, axis='y')
#                 fig = plt.gcf()
#                 fig.savefig('{}/train succ req.png'.format(dirpath), facecolor='white', dpi=600)
#                 plt.clf()
#
#                 plt.title('Blocked req')
#                 plt.xlabel('Epi')
#                 plt.ylabel('# of req')
#                 plt.plot(log_avg_blk_req, 'b')
#                 plt.grid(True, axis='y')
#                 fig = plt.gcf()
#                 fig.savefig('{}/train blk req.png'.format(dirpath), facecolor='white', dpi=600)
#                 plt.clf()
#
#                 plt.title('blocking prob')
#                 plt.xlabel('Epi')
#                 plt.ylabel('prob')
#                 plt.plot(log_avg_BP_req, 'b')
#                 plt.grid(True, axis='y')
#                 fig = plt.gcf()
#                 fig.savefig('{}/train blocking prob.png'.format(dirpath), facecolor='white', dpi=600)
#                 plt.clf()
#
#                 plt.title('Entropy')
#                 plt.xlabel('Epi')
#                 plt.ylabel('entropy')
#                 plt.plot(log_avg_entropy, 'b')
#                 plt.grid(True, axis='y')
#                 fig = plt.gcf()
#                 fig.savefig('{}/train entropy.png'.format(dirpath), facecolor='white', dpi=600)
#                 plt.clf()
#             if cnt % 1000 == 0:
#                 file_name = "PPO_npro_{}_{:05d}_{}.pth".format(n_pro, cnt, avg_blk)
#                 file_path = os.path.join(pthpath, file_name)
#                 # global_model.save(file_path)
#                 torch.save(global_model.state_dict(), file_path)
#
#         else:
#             flag += 1
#             if flag == n_pro:
#                 break
#
#     for p in processes:
#         p.join()

if __name__ == '__main__':
    # ppoTrainSingle_ppangyo_multi()
    ppoTrainSingle_ppangyo()
    # ppo_pppangyo_multi_new()
