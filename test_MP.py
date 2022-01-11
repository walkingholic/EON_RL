
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
import datetime
from simulation_v2 import Simulation
import os, time
import numpy as np
from PPO_ppangyo_EON import PPO as PPO_EON
import math
import torch
import torch.optim as optim
from torch.distributions import Categorical


################################## set device ##################################

print("============================================================================================")

# set device to cpu or cuda
device = torch.device('cpu')
#
# if (torch.cuda.is_available()):
#     device = torch.device('cuda:0')
#     torch.cuda.empty_cache()
#     print("Device set to : " + str(torch.cuda.get_device_name(device)))
# else:
#     print("Device set to : cpu")

print("============================================================================================")


basepath = 'result\\'

def createFolder(directory):
    try:
        if not os.path.isdir(directory):
            os.makedirs(directory)
    except OSError:
        print('error')

class SharedAdam(optim.Adam):
    """Implements Adam algorithm with shared states.
    """

    def __init__(self,
                 params,
                 lr=1e-3,
                 betas=(0.9, 0.999),
                 eps=1e-8,
                 weight_decay=0):
        super(SharedAdam, self).__init__(params, lr, betas, eps, weight_decay)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = torch.zeros(1)
                state['exp_avg'] = p.data.new().resize_as_(p.data).zero_()
                state['exp_avg_sq'] = p.data.new().resize_as_(p.data).zero_()

    def share_memory(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'].share_memory_()
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step'].item()
                bias_correction2 = 1 - beta2 ** state['step'].item()
                step_size = group['lr'] * math.sqrt(
                    bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)

        return loss


def train(i, shared_model, learning_rate, gamma, lmbda, eps_clip, K_epoch, optimizer=None):

    num_EP = 1000000
    timestep_max = 200
    num_of_req = 60000
    num_of_warmup = 10000
    ar = 4
    dif = 1
    avg_holding = 50
    range_band_s = 1
    range_band_e = 10
    TotalNumofSlots = 100
    num_kpath = 5
    rt_name = 'KSP'
    sa_name = 'DRL'
    i = 0
    N_A = 15
    N_S = 100

    update_timestep = timestep_max * 3
    model = PPO_EON(N_S, N_A).to(device)
    env = Simulation(num_of_req, num_kpath, num_of_warmup, (ar + (i) * dif), avg_holding, range_band_s,
                     range_band_e, TotalNumofSlots,
                     rt_name, sa_name)

    if optimizer is not None:
        model.optimizer = optimizer

    env.env_init()
    timestep = 0
    log_entropy = []
    log_rwd = []
    log_succ_req = []
    log_blk_req = []

    for e in range(num_EP):
        done = False
        state, req_info, req = env.env_reset()
        model.load_state_dict(shared_model.state_dict())

        ept_score = 0
        succ_req, blk_req = 0, 0
        entropy = 0
        actlist = []
        while done != 1:
            prob = model.pi(torch.from_numpy(state).float().to(device), torch.from_numpy(req_info).float().to(device))
            m = Categorical(prob)
            action = m.sample().item()
            actlist.append(action)
            entropy += m.entropy().item()
            next_state, next_req_info, next_req, reward, done = env.env_step(req, action)
            ept_score += reward

            if reward == 1:
                succ_req += 1
            else:
                blk_req += 1

            model.put_data((state, req_info, action, reward, next_state, next_req_info, prob[0, action].item(), done))

            state = next_state
            req_info = next_req_info
            req = next_req

            timestep += 1
            if timestep % timestep_max == 0:
                done = 1
                log_rwd.append(ept_score)
                log_succ_req.append(succ_req)
                log_blk_req.append(blk_req)
                log_entropy.append(entropy / timestep_max)
                print('Epi: ', e, '   Score: ', ept_score, '   BBP: ', blk_req / timestep_max, '  Epi avg Entropy: ',
                      entropy / timestep_max)
                print(actlist)

            if timestep % update_timestep == 0:
                model.train_net(gamma, lmbda, eps_clip, K_epoch)


def main():

    no_shared = False
    shared_model = PPO_EON(100, 15)
    shared_model.share_memory()

    learning_rate = 0.0001
    gamma = 0.99
    lmbda = 0.95
    eps_clip = 0.1
    K_epoch = 32

    if no_shared:
        optimizer = None
    else:
        optimizer = SharedAdam(shared_model.parameters(), lr=learning_rate)
        optimizer.share_memory()

    processes = [mp.Process(target=train, args=(i, shared_model, learning_rate, gamma, lmbda, eps_clip, K_epoch, optimizer)) for i in range(0, 2)]

    [p.start() for p in processes]

    [p.join() for p in processes]



if __name__ == '__main__':
    main()



