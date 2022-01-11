import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import numpy as np
from simulation_v2 import Simulation
import torch.multiprocessing as mp
import matplotlib.pyplot as plt


################################## set device ##################################

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


################################## PPO Policy ##################################


class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.req_infos = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.req_infos[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()


        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc1 = nn.Linear(1214, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc_pi = nn.Linear(256, action_dim)
        self.fc_v = nn.Linear(256, 1)

    def pi(self, state, req_info, softmax_dim=0):

        out = self.layer1(state)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.view(out.size(0), -1)
        in_data = torch.cat([out, req_info], dim=1)
        x = F.relu(self.fc1(in_data))
        x = F.relu(self.fc2(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob

    def v(self, state, req_info):

        out = self.layer1(state)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.view(out.size(0), -1)
        in_data = torch.cat([out, req_info], dim=1)
        x = F.relu(self.fc1(in_data))
        x = F.relu(self.fc2(x))
        v = self.fc_v(x)
        return v


    def forward(self):
        raise NotImplementedError

    def act(self, state, req_info):

        action_probs = self.pi(state, req_info)
        dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)

        return action.detach(), action_logprob.detach()

    def evaluate(self, state, req_info, action):


        action_probs = self.pi(state, req_info)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.v(state, req_info)

        return action_logprobs, state_values, dist_entropy


class PPO(mp.Process):
    def __init__(self, id, update_timestep, gmodel, state_dim, action_dim, lr_actor, gamma, K_epochs, eps_clip, queue):
        super(PPO, self).__init__()

        self.id = id
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.queue = queue


        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim).to(device)
        self.policy.load_state_dict(gmodel.state_dict())
        self.gmodel = gmodel

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr_actor)

        self.MseLoss = nn.MSELoss()
        self.num_EP = 500000
        self.update_timestep = update_timestep

    def run(self):
        print('run ')
        timestep = 0
        cnt = 0

        timestep_max=50

        num_of_senario = 1
        num_of_req = 60000
        num_of_warmup = 10000
        ar = 4
        dif = 1
        # avg_holding = 50
        avg_holding = 50
        range_band_s = 1
        range_band_e = 10
        TotalNumofSlots = 100
        num_kpath = 5
        rt_name='KSP'
        sa_name='DRL'
        i=0

        env = Simulation(num_of_req, num_kpath, num_of_warmup, (ar + (i) * dif), avg_holding, range_band_s,
                         range_band_e, TotalNumofSlots,
                         rt_name, sa_name)

        env.env_init()
        timestep = 0
        # log_avg_ep = []
        # log_avg_rwd = []
        # log_tmp_add_ep_rwd = []
        ep_entropy = []
        log_ep = []
        for e in range(self.num_EP):
            done = False
            state, req_info, req = env.env_reset()
            ept_score=0

            while done != 1:

                action = self.select_action(state, req_info)

                next_state, next_req_info, next_req, reward, done = env.env_step(req, action)
                ept_score += reward
                state = next_state
                req_info = next_req_info
                req = next_req

                timestep += 1
                if timestep%timestep_max==0:
                    done=1
                    # print('{}: {}'.format(e, ept_score))
                    self.queue.put((ept_score))

                self.buffer.rewards.append(reward)
                self.buffer.is_terminals.append(done)

                # print(timestep, done)
                if timestep % self.update_timestep == 0:
                    # print(self.id, 'len',len(self.buffer.states))
                    # log_ep.append(e)
                    self.update()
                    # ep_entropy.append(etrp)
                    # print(etrp)
                    self.policy.load_state_dict(self.gmodel.state_dict())


            # print(ep_entropy)
            # plt.xlabel('Epoch')
            # plt.ylabel('entropy')
            # plt.plot(log_ep, ep_entropy, 'b')
            # plt.grid(True, axis='y')
            # fig = plt.gcf()
            # fig.savefig('entropy.png', facecolor='white', dpi=600)
            # plt.clf()

        self.queue.put(None)

    def select_action(self, state, req_info):


        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            req_info = torch.FloatTensor(req_info).to(device)
            action, action_logprob = self.policy.act(state, req_info)


        # print(state.size())
        # print(req_info.size())

        self.buffer.states.append(state)
        self.buffer.req_infos.append(req_info)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)

        return action.item()


    def update(self):
        entropy=0
        print('Update')
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0), dim=1).detach().to(device)
        old_req_infos = torch.squeeze(torch.stack(self.buffer.req_infos, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)

        # print(old_states.size())
        # print(old_req_infos.size())

        # Optimize policy for K epochs
        for kth in range(self.K_epochs):
            # print('Update ', kth)
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_req_infos, old_actions)

            # print(dist_entropy.item())
            # entropy = sum(dist_entropy.detach())/len(dist_entropy.detach())
            # print('dist entropy: ', sum(dist_entropy.detach())/len(dist_entropy.detach()))

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy
        self.gmodel.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()
        # return entropy

    def save(self, checkpoint_path):
        torch.save(self.policy.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        # self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))

# state_dim = 4608
# action_dim = 5
# lr = 0.0002
# betas = (0.9, 0.999)
# gamma = 0.9998  # discount factor
# K_epochs = 16  # update policy for K epochs
# eps_clip = 0.2  # clip parameter for PPO
#
# model = PPO(state_dim, action_dim, lr, lr, gamma, K_epochs, eps_clip, 0)
#
# # x = torch.rand(1,100)
# # out = model.policy.pi(x)
#
# # print(out.size())
# # print(model.policy.state_dict())
#
#
# for param in model.policy.Covnet.parameters():
#     print(param, param.data)
# for param in model.policy.layer1.parameters():
#     print(param, param.data)


# for param in model.policy.critic.parameters():
#     print(param.data)
# for param_tensor in model.policy.state_dict():
#     print(param_tensor, "\t", model.policy.state_dict()[param_tensor].size())