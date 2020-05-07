import itertools
import numpy as np
import math
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
import logging
import os
import sys
import matplotlib.pyplot as plt
from datetime import datetime


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_tensor_type(torch.cuda.FloatTensor)

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_size2,hidden_size3, hidden_size4, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.drop1 = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size2)
        self.bn2 = nn.BatchNorm1d(hidden_size2)
        self.drop2 = nn.Dropout(0.1)
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.bn3 = nn.BatchNorm1d(hidden_size3)
        self.drop3 = nn.Dropout(0.1)
        self.fc4 = nn.Linear(hidden_size3, num_classes)

    def forward(self, x):
        x = self.relu((self.fc1(x)))
        x = self.relu((self.fc2(x)))
        x = self.relu((self.fc3(x)))
        out = self.fc4(x)
        return out

class Environment_HS:
    def __init__ (self, user_rate, C_total, R_n, R_C, num_time, state_dim, action_dim):
        self.user_rate = user_rate
        self.r = 0.2 # Input/output ratio
        self.C_total = C_total # Total CPU capacity of server [Hz]
        self.R_n = R_n # user->server datarate [bps]
        self.R_C = R_C # server->user datarate [bps]
        self.num_time = num_time # Total simulation time [s]
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.C0_usage = np.zeros(self.num_time)
        self.C0_usage2 = np.zeros(self.num_time)
        self.user_num = 0
        self.avg_completion_time = 0
        self.index=0
        self.queue = None
        self.queue2 = None
        self.queue3 = None
        self.user_set = None


    def generate_user(self): # get user set
        U = np.random.poisson(self.user_rate, self.num_time)
        user_set = np.zeros([sum(U),6])
        C_list = np.array([2.84e9, 2.39e9, 2.5e9])
        T_lim_list = np.array([0.1, 0.5, 1])
        beta_list = np.array([1/2,1,2])
        idx = 0
        t_task_origin = 0
        for i in range(self.num_time):
            if U[i]!=0:
                for j in range(U[i]):
                    ### [h,d,F_l,t_req,t,beta]
                    user_set[idx,0] = 2640  # h
                    user_set[idx,1] = 1000*8*np.random.randint(low=100, high=300+1) # d
                    user_set[idx,2] = C_list[np.random.randint(low=0, high=3)] # F_l
                    user_set[idx,3] = T_lim_list[np.random.randint(low=0,high=3)] # t_req
                    user_set[idx,4] = int(i)
                    user_set[idx,5] = beta_list[np.random.randint(low=0,high=3)] # beta
                    t_task_origin += np.ceil(user_set[idx,0]*user_set[idx,1]/user_set[idx,2])
                    idx = idx + 1
        del C_list, U
        self.user_set=user_set
        self.user_num=idx
        self.avg_completion_time=t_task_origin/len(user_set)
        print("######### new users are generated ########")
        print("total number of users = ", self.user_num)
        print("total completion time = ", self.avg_completion_time)
        print("##########################################")
        return user_set

    def add_user_to_queue2(self, time, user_set):
        new_user = user_set[user_set[:,4]==time,:]
        if self.queue2 is None:
            self.queue2 = new_user
        else:
            self.queue2 = np.append(self.queue2, new_user,axis=0)

    def step_uniform(self):
        t_available=0
        profit = 0
        total_time_uni = 0
        total_time_uni_off = 0
        idx=0
        avg_cost=0
        for t in range(num_time):
            self.add_user_to_queue2(t,self.user_set)

            if(t> t_available):
                if(len(self.queue2)!=0):
                    N = len(self.queue2)
                    A = (1/self.R_n) + (self.r/R_C) + (self.queue2[:,0]/(self.queue2[:,2]))
                    b_list = 1/self.queue2[:,2]
                    payment_b = np.zeros(N)
                    t_process_max_b = np.zeros(N)
                    t_total_mat = np.zeros(N)
                    for b_idx in range(N):
                        t_total_temp = 0
                        b_idx=int(b_idx)
                        b= b_list[b_idx]
                        I_opt = (self.queue2[:,1]*self.queue2[:,0]/self.queue2[:,2]) / ((self.queue2[:,0]/(self.C_total/N))+A)
                        I_opt[b>1/self.queue2[:,2]]=0
                        I_opt[I_opt<0]=0
                        payment_b[b_idx] = np.sum( b*I_opt*self.queue2[:,0])
                        t_process_mat = I_opt*self.queue2[:,0]/ (self.C_total/N)
                        t_process_max_b[b_idx] = np.amax(t_process_mat)
                        t_total_temp = (self.queue2[:,1]-I_opt)*self.queue2[:,0]/self.queue2[:,2]
                        t_total_mat[b_idx] =np.sum( t_total_temp )
                    payment_max_b = np.amax(payment_b)
                    b_max_idx = np.argmax(payment_b)
                    profit+=payment_max_b
                    t_available = int(t + (t_process_max_b[b_max_idx]*10))
                    total_time_uni += t_total_mat[b_max_idx]
                    total_time_uni_off += t_total_mat[b_max_idx]
                    self.queue2 = None
                    avg_cost += t_total_mat[b_max_idx]+payment_max_b
                    idx+=N
            else:
                self.queue2[:,1] = (self.queue2[:,1]-(self.queue2[:,2]/self.queue2[:,0]))
                queue_temp = self.queue2[self.queue2[:,1]<=0,:]
                time_temp = 1+queue_temp[:,1]/(queue_temp[:,2]/queue_temp[:,0])
                total_time_uni += np.sum(time_temp)
                avg_cost += np.sum(time_temp)
                self.queue2 = self.queue2[self.queue2[:,1]>0,:]
                total_time_uni += len(self.queue2)
                avg_cost += len(self.queue2)
        return profit, total_time_uni/len(self.user_set), total_time_uni_off/idx, avg_cost/len(self.user_set)



    def get_state(self):
        """state = [h, d, F_l, t_req, F_0, beta]"""
        state = [self.user_set[self.index,0]/2640,
                 self.user_set[self.index,1]/(1000*8*300),
                 (self.user_set[self.index,2]-2.39e9)/(2.84e9-2.39e9),
                 self.user_set[self.index,3],
                 self.C0_usage[int(self.user_set[self.index,4])],
                 self.user_set[self.index,5]/2]
        self.index+=1
        return np.array(state), self.user_set[self.index-1,4]


    def step(self,state,action,t):
        H = state[0]*2640
        r = self.r
        D = state[1]*(1000*8*300)
        C = state[2]*(2.84e9-2.39e9)+2.39e9
        T_lim = state[3]
        beta = state[5]*2
        C0 = state[4]*self.C_total
        Cr = self.C_total-C0
        t = int(t)
        a = action[0]
        b = action[1]
        t_task = 0
        t_task_off = 0
        payment=0
        """get solution"""

        # substitution
        A = (1/self.R_n) + (r/self.R_C) + (H/C)
        B = (b*beta*self.C_total) - (self.C_total**2)/C

        # solution
        if ( B>=0 ):
            C_opt = 0
        else:
            # clip

            C_b = (-(beta*a*H)+np.sqrt((beta**2)*(a**2)*(H**2)-beta*a*A*B*H) )/(beta*a*A)
            C_min = (C*H*(D*H-C*T_lim))/(D*(H**2)-A*C*(D*H-C*T_lim))
            C_max = ((self.C_total)/a)*( (self.C_total/(beta*C)) - b)

            if (C_max < C_min ):
                C_opt = 0
            else:
                C_opt= np.clip(C_b,C_min,C_max)
            if( C_opt >= Cr):
                    C_opt = Cr
                    C_min = (C*H*(D*H-C*T_lim))/(D*(H**2)-A*C*(D*H-C*T_lim))
                    if C_opt <= C_min:
                        C_opt = 0

            if C_opt == 0 :
                I_opt = 0
            else:
                I_opt = (D*H/C)/(A+(H/C_opt))
                Cost_offloading = np.maximum( (1/self.R_n + r/self.R_C + H/C_opt)*I_opt  ,  (D-I_opt)*H/C ) + beta*(a*C_opt+self.C_total*b)*(I_opt*H/((self.C_total)**2))
                Cost_non = D*H/C
                if Cost_offloading > Cost_non:
                    C_opt = 0
                    I_opt = 0



        # calculate completion time
        if C_opt ==0:
            I_opt = 0
            payment = 0
            t_task = D*H/C  # completion time
            t_process = 0
        else:
            I_opt = (D*H/C)/(A+(H/C_opt))
            payment = (a*C_opt+self.C_total*b)*(I_opt*H/((self.C_total)**2))
            t_task_off =  np.maximum( (1/self.R_n + r/self.R_C + H/C_opt)*I_opt  ,  (D-I_opt)*H/C )
            t_process = H*I_opt/C_opt

        t_process = int((t_process*10))

        # save F_0 info
        self.C0_usage[t:t+t_process] = self.C0_usage[t:t+t_process]+C_opt/self.C_total

        reward = payment
        cost = beta*payment+t_task+t_task_off

        return reward, t_task, t_task_off, cost

    
    def add_user_to_queue3(self, time, user_set):
        new_user = user_set[user_set[:,4]==time,:]
        if self.queue3 is None:
            self.queue3 = new_user
        else:
            self.queue3 = np.append(self.queue3, new_user,axis=0)
            
            

    def step_fool(self):
        t_available=0
        profit = 0
        total_time_uni = 0
        total_time_uni_off = 0
        idx=0
        avg_cost=0
        for t in range(num_time):
            self.add_user_to_queue3(t,self.user_set)

            if(t> t_available):
                if(len(self.queue3)!=0):
                    N = len(self.queue3)
                    A = (1/self.R_n) + (self.r/R_C) + (self.queue3[:,0]/(self.queue3[:,2]))
                    t_total_temp = 0
                    b= 1/(2.6e9)
                    I_opt = (self.queue3[:,1]*self.queue3[:,0]/self.queue3[:,2]) / ((self.queue3[:,0]/(self.C_total/N))+A)
                    I_opt[b>1/self.queue3[:,2]]=0
                    I_opt[I_opt<0]=0
                    payment_b = np.sum( b*I_opt*self.queue3[:,0])
                    t_process_mat = I_opt*self.queue3[:,0]/ (self.C_total/N)
                    t_process_max_b = np.amax(t_process_mat)
                    t_total_temp = (self.queue3[:,1]-I_opt)*self.queue3[:,0]/self.queue3[:,2]
                    t_total_mat =np.sum( t_total_temp )
                    profit+=payment_b
                    t_available = int(t + (t_process_max_b*10))
                    total_time_uni += t_total_mat
                    total_time_uni_off += t_total_mat
                    self.queue3 = None
                    avg_cost += t_total_mat+payment_b
                    idx+=N

                    # save F_0 info
                    self.C0_usage2[t:int(t+t_process_max_b*10)] = 110e9
                    
            else:
                self.queue3[:,1] = (self.queue3[:,1]-(self.queue3[:,2]/self.queue3[:,0]))
                queue_temp = self.queue3[self.queue3[:,1]<=0,:]
                time_temp = 1+queue_temp[:,1]/(queue_temp[:,2]/queue_temp[:,0])
                total_time_uni += np.sum(time_temp)
                avg_cost += np.sum(time_temp)
                self.queue3 = self.queue3[self.queue3[:,1]>0,:]
                total_time_uni += len(self.queue3)
                avg_cost += len(self.queue3)
        return profit, total_time_uni/len(self.user_set), total_time_uni_off/idx, avg_cost/len(self.user_set)
    
    
    def reset(self):
        self.C0_usage = np.zeros(self.num_time)
        self.index = 0
        self.user_num = 0
        self.avg_completion_time = 0

    



C_total = 110e9
R_n = 43.9e6
R_C = 150.6e6
num_time = 6000  # 10 min
state_dim = 6
action_dim = 300*200
max_episodes = 36


input_size = 6
hidden_size = 128
hidden_size2 = 64
hidden_size3 = 32
hidden_size4 = 16
num_classes = 2


model = NeuralNet(input_size, hidden_size, hidden_size2, hidden_size3, hidden_size4, num_classes).to(device)
model.load_state_dict(torch.load("trained_model.ckpt"))
model.eval()

time_step = 0

time_list =[]
time_list2 = []
time_list3 = []
offtime_list = []
offtime_list2 = []
offtime_list3 = []
profit_list= []
profit_list2 = []
profit_list3 = []
cost_list = []
cost_list2 = []
cost_list3 = []

time_list_1 =[]
time_list2_1 = []
time_list3_1 = []
offtime_list_1 = []
offtime_list2_1 = []
offtime_list3_1 = []
profit_list_1= []
profit_list2_1 = []
profit_list3_1 = []
cost_list_1 = []
cost_list2_1 = []
cost_list3_1 = []

time_list_2 =[]
time_list2_2 = []
time_list3_2 = []
offtime_list_2 = []
offtime_list2_2 = []
offtime_list3_2 = []
profit_list_2= []
profit_list2_2 = []
profit_list3_2 = []
cost_list_2 = []
cost_list2_2 = []
cost_list3_2 = []

time_list_3 =[]
time_list2_3 = []
time_list3_3 = []
offtime_list_3 = []
offtime_list2_3 = []
offtime_list3_3 = []
profit_list_3= []
profit_list2_3 = []
profit_list3_3 = []
cost_list_3 = []
cost_list2_3 = []
cost_list3_3 = []

time_list_4 =[]
time_list2_4 = []
time_list3_4 = []
offtime_list_4 = []
offtime_list2_4 = []
offtime_list3_4 = []
profit_list_4= []
profit_list2_4 = []
profit_list3_4 = []
cost_list_4 = []
cost_list2_4 = []
cost_list3_4 = []


""" Simulatation """

with torch.no_grad():
    for i_episode in range(1, max_episodes+1):
        if i_episode <= 12:
            user_rate = 1
        if 12<i_episode<=24:
            user_rate = 2
        if 24<i_episode<=36:
            user_rate = 3
        if 36<i_episode:
            user_rate = 4

        env = Environment_HS(user_rate, C_total, R_n, R_C, num_time, state_dim, action_dim)
        state_dim = env.state_dim
        action_dim = env.action_dim
        print("############################ episode {} ############################".format(i_episode))

        user_set = env.generate_user()
        completion_time_total = 0
        completion_time_off = 0
        running_reward = 0
        running_cost = 0
        idx2=0
        completion_time_total2 = 0
        completion_time_off2 = 0
        running_reward2 = 0
        running_cost2 = 0
        idx3=0
        for t in range(len(user_set)):
            state, time= env.get_state()
            state = torch.tensor(state).float().to(device)
            action = model(state)
            state = state.cpu().detach().numpy()
            action = action.cpu().detach().numpy()
            if action[0]<0:
                action[0]=1
            if action[1]<0:
                action[1]=1
            reward, comp_time, comp_time_off,cost = env.step(state, action, time)
            if comp_time_off != 0:
                idx2 +=1
            running_reward += reward
            running_cost += cost
            completion_time_total += comp_time+comp_time_off
            completion_time_off += comp_time_off
        if i_episode <= 12:
            profit_list_1.append(running_reward)
            time_list_1.append(completion_time_total/len(user_set))
            offtime_list_1.append(completion_time_off/idx2)
            cost_list_1.append(running_cost/len(user_set))
        if 12<i_episode<=24:
            profit_list_2.append(running_reward)
            time_list_2.append(completion_time_total/len(user_set))
            offtime_list_2.append(completion_time_off/idx2)
            cost_list_2.append(running_cost/len(user_set))
        if 24<i_episode<=36:
            profit_list_3.append(running_reward)
            time_list_3.append(completion_time_total/len(user_set))
            offtime_list_3.append(completion_time_off/idx2)
            cost_list_3.append(running_cost/len(user_set))
        if 36<i_episode:
            profit_list_4.append(running_reward)
            time_list_4.append(completion_time_total/len(user_set))
            offtime_list_4.append(completion_time_off/idx2)
            cost_list_4.append(running_cost/len(user_set))
        profit_list.append(running_reward)
        time_list.append(completion_time_total/len(user_set))
        offtime_list.append(completion_time_off/idx2)
        cost_list.append(running_cost/len(user_set))


        """uniform pricing simulation"""
        profit2, avg_time2, avg_time_off2, avg_cost2  = env.step_uniform()
        profit3, avg_time3, avg_time_off3, avg_cost3  = env.step_fool()

        if i_episode <= 12:
            profit_list2_1.append(profit2)
            time_list2_1.append(avg_time2)
            offtime_list2_1.append(avg_time_off2)
            cost_list2_1.append(avg_cost2)
            profit_list3_1.append(profit3)
            time_list3_1.append(avg_time3)
            offtime_list3_1.append(avg_time_off3)
            cost_list3_1.append(avg_cost3)
        if 12<i_episode<=24:
            profit_list2_2.append(profit2)
            time_list2_2.append(avg_time2)
            offtime_list2_2.append(avg_time_off2)
            cost_list2_2.append(avg_cost2)
            profit_list3_2.append(profit3)
            time_list3_2.append(avg_time3)
            offtime_list3_2.append(avg_time_off3)
            cost_list3_2.append(avg_cost3)
        if 24<i_episode<=36:
            profit_list2_3.append(profit2)
            time_list2_3.append(avg_time2)
            offtime_list2_3.append(avg_time_off2)
            cost_list2_3.append(avg_cost2)
            profit_list3_3.append(profit3)
            time_list3_3.append(avg_time3)
            offtime_list3_3.append(avg_time_off3)
            cost_list3_3.append(avg_cost3)
        if 36<i_episode:
            profit_list2_4.append(profit2)
            time_list2_4.append(avg_time2)
            offtime_list2_4.append(avg_time_off2)
            cost_list2_4.append(avg_cost2)
            profit_list3_4.append(profit3)
            time_list3_4.append(avg_time3)
            offtime_list3_4.append(avg_time_off3)
            cost_list3_4.append(avg_cost3)
        profit_list2.append(profit2)
        time_list2.append(avg_time2)
        offtime_list2.append(avg_time_off2)
        cost_list2.append(avg_cost2)
        profit_list3.append(profit3)
        time_list3.append(avg_time3)
        offtime_list3.append(avg_time_off3)
        cost_list3.append(avg_cost3)

        env.reset()

        print("    linear pricing  : revenue {}, total avg time {}".format(profit_list[-1], time_list[-1]))
        print("    uniform pricing : revenue {}, total avg time {}".format(profit_list2[-1], time_list2[-1]))
        print("    fool   pricing  : revenue {}, total avg time {}".format(profit_list3[-1], time_list3[-1]))




profit_sum_l=[]
profit_sum_m=[]
profit_sum_h=[]
profit_sum_hh = []
profit_sum2_l=[]
profit_sum2_m=[]
profit_sum2_h=[]
profit_sum2_hh=[]
profit_sum3_l=[]
profit_sum3_m=[]
profit_sum3_h=[]
profit_sum3_hh=[]
for i in range(len(profit_list_1)):
    profit_sum_l.append(np.sum(profit_list_1[:i+1]))
    profit_sum2_l.append(np.sum(profit_list2_1[:i+1]))
    profit_sum3_l.append(np.sum(profit_list3_1[:i+1]))
    profit_sum_m.append(np.sum(profit_list_2[:i+1]))
    profit_sum2_m.append(np.sum(profit_list2_2[:i+1]))
    profit_sum3_m.append(np.sum(profit_list3_2[:i+1]))
    profit_sum_h.append(np.sum(profit_list_3[:i+1]))
    profit_sum2_h.append(np.sum(profit_list2_3[:i+1]))
    profit_sum3_h.append(np.sum(profit_list3_3[:i+1]))
    profit_sum_hh.append(np.sum(profit_list_4[:i+1]))
    profit_sum2_hh.append(np.sum(profit_list2_4[:i+1]))
    profit_sum3_hh.append(np.sum(profit_list3_4[:i+1]))




avg_time_l = np.mean(time_list_1)
avg_time2_l =np.mean(time_list2_1)
avg_time3_l =np.mean(time_list3_1)
avg_time_m = np.mean(time_list_2)
avg_time2_m = np.mean(time_list2_2)
avg_time3_m = np.mean(time_list3_2)
avg_time_h = np.mean(time_list_3)
avg_time2_h = np.mean(time_list2_3)
avg_time3_h = np.mean(time_list3_3)
avg_time_hh = np.mean(time_list_4)
avg_time2_hh = np.mean(time_list2_4)
avg_time3_hh = np.mean(time_list3_4)

avg_cost_l = np.mean(cost_list_1)
avg_cost2_l =np.mean(cost_list2_1)
avg_cost3_l =np.mean(cost_list3_1)
avg_cost_m = np.mean(cost_list_2)
avg_cost2_m = np.mean(cost_list2_2)
avg_cost3_m = np.mean(cost_list3_2)
avg_cost_h = np.mean(cost_list_3)
avg_cost2_h = np.mean(cost_list2_3)
avg_cost3_h = np.mean(cost_list3_3)
avg_cost_hh = np.mean(cost_list_4)
avg_cost2_hh = np.mean(cost_list2_4)
avg_cost3_hh = np.mean(cost_list3_4)



print("avg time Low rate = {}, {}, {}    Medium rate = {}, {}, {}   High rate = {}, {}, {}  ".format(avg_time_l,avg_time2_l,avg_time3_l,avg_time_m,avg_time2_m,avg_time3_m,avg_time_h,avg_time2_h,avg_time3_h))
print("avg cost Low rate = {}, {}, {}    Medium rate = {}, {}, {}   High rate = {}, {}, {}  ".format(avg_cost_l,avg_cost2_l,avg_cost3_l,avg_cost_m,avg_cost2_m,avg_cost3_m,avg_cost_h,avg_cost2_h,avg_cost3_h))
