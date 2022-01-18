import itertools
import numpy as np
import math
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
import Rmatrix

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_tensor_type(torch.cuda.FloatTensor)
print("Device = ", device)

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

Bbconvert=8*1000
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
        self.C0_usage3 = np.zeros(self.num_time)
        self.C0_usage4 = np.zeros(self.num_time)
        self.user_num = 0
        self.avg_completion_time = 0
        self.index=0
        self.index2 = 0
        self.queue = None
        self.queue2 = None
        self.queue3 = None
        self.user_set = None
        self.time_diff = []
        self.time_uni = []
        self.time_det = []
        self.time_cheaper = []
        self.time_non = []
        self.cost_diff = []
        self.cost_uni =[]
        self.cost_det = []
        self.cost_cheaper = []
        self.cost_non = []


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
                    user_set[idx,1] = Bbconvert*np.random.randint(low=100, high=300+1) # d
                    user_set[idx,2] = C_list[np.random.randint(low=0, high=3)] # F_l
                    user_set[idx,3] = T_lim_list[np.random.randint(low=0,high=3)] # t_req
                    user_set[idx,4] = int(i)
                    user_set[idx,5] = beta_list[np.random.randint(low=0,high=3)] # beta
                    t_task_origin += np.ceil(user_set[idx,0]*user_set[idx,1]/user_set[idx,2])
                    self.time_non.append(np.ceil(user_set[idx,0]*user_set[idx,1]/user_set[idx,2]))
                    self.cost_non.append(np.ceil(user_set[idx,0]*user_set[idx,1]/user_set[idx,2]))
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
        user_idx = 0
        avg_cost=0
        for t in range(num_time):
            self.add_user_to_queue2(t,self.user_set)
            C_r = (self.C_total - self.C0_usage2[t]*self.C_total)
            epsilon = np.random.rand(1)
            if C_r == self.C_total:
                C_r = C_r * (0.85+epsilon*0.15)
           

            if(t> t_available):
                if(len(self.queue2)!=0 and C_r>0):
                    N = len(self.queue2)
                    A = (1/self.R_n) + (self.r/R_C) + (self.queue2[:,0]/(self.queue2[:,2]))
                    b_list = 1/self.queue2[:,2]
                    payment_b = np.zeros(N)
                    t_process_max_b = np.zeros(N)
                    t_process_median_b = np.zeros(N) ###
                    t_total_mat = np.zeros(N)
                    t_process_mat_all = np.zeros((N,N))
                    t_total_mat_all = np.zeros((N,N))
                    payment_mat_all = np.zeros((N,N))
                    for b_idx in range(N):
                        t_total_temp = 0
                        b_idx=int(b_idx)
                        b= b_list[b_idx]
                        I_opt = (self.queue2[:,1]*self.queue2[:,0]/self.queue2[:,2]) / ((self.queue2[:,0]/(C_r/N))+A)
                        I_opt[b>1/self.queue2[:,2]]=0
                        I_opt[I_opt<0]=0
                        payment_b[b_idx] = np.sum( b*I_opt*self.queue2[:,0])
                        t_process_mat = I_opt*self.queue2[:,0]/ (C_r/N)
                        t_process_max_b[b_idx] = np.amax(t_process_mat)

                        t_process_median_b[b_idx] = np.median(t_process_mat)
                        t_total_temp = (self.queue2[:,1]-I_opt)*self.queue2[:,0]/self.queue2[:,2] 
                        t_total_mat[b_idx] =np.sum( t_total_temp )
                        t_process_mat_all[b_idx,:]=t_process_mat ###
                        t_total_mat_all[b_idx,:]=t_total_temp ###
                        payment_mat_all[b_idx,:]=b*I_opt*self.queue2[:,0]
                    payment_max_b = np.amax(payment_b)
                    b_max_idx = np.argmax(payment_b)
                    profit+=payment_max_b


                    t_available = int(t + (t_process_median_b[b_max_idx]*10))
                    total_time_uni += t_total_mat[b_max_idx]
                    total_time_uni_off += t_total_mat[b_max_idx]

                    avg_cost += t_total_mat[b_max_idx]+payment_max_b

                    num_off = len(t_process_mat_all[b_max_idx,t_process_mat_all[b_max_idx,:]>0])

                    user_idx += len(self.queue2)
                    idx += len(t_total_mat_all[b_max_idx,t_total_mat_all[b_max_idx,:]<=self.queue2[:,3]]) 
                    self.queue2 = None
                
                    
                    
                    for k in range(N): 
                        self.time_uni.append(t_total_mat_all[b_max_idx,k])
                        self.cost_uni.append(t_total_mat_all[b_max_idx,k]+payment_mat_all[b_max_idx,k])
                        if t_process_mat_all[b_max_idx,k]>0 :
                            if int(t_process_mat_all[b_max_idx,k]*10) == 0:
                                self.C0_usage2[t:t+1]=self.C0_usage2[t:t+1]+(C_r/num_off)/self.C_total
                            else:
                                self.C0_usage2[t:t+int(t_process_mat_all[b_max_idx,k]*10)+1] = self.C0_usage2[t:t+int(t_process_mat_all[b_max_idx,k]*10)+1] + (C_r/num_off)/self.C_total
            else: 
                time_temp= self.queue2[:,0]*self.queue2[:,1]/self.queue2[:,2]
                for i in range(len(time_temp)):
                    self.time_uni.append(time_temp[i])
                    self.cost_uni.append(time_temp[i])
                total_time_uni += np.sum(time_temp)
                avg_cost += np.sum(time_temp)
                user_idx += len(self.queue2)
                idx += len(time_temp[time_temp<=self.queue2[:,3]])
                self.queue2 = None


        return profit, total_time_uni/len(self.user_set), total_time_uni_off/idx, avg_cost/len(self.user_set), idx/user_idx


    def get_state(self):
        """state = [h, d, F_l, t_req, F_0, beta]"""
        state = [self.user_set[self.index,0]/2640,
                 self.user_set[self.index,1]/(Bbconvert*300),
                 (self.user_set[self.index,2]-2.39e9)/(2.84e9-2.39e9),
                 self.user_set[self.index,3],
                 self.C0_usage[int(self.user_set[self.index,4])],
                 self.user_set[self.index,5]/2]
        self.index+=1
        return np.array(state), self.user_set[self.index-1,4]


    def step(self,state,action,t):
        H = state[0]*2640
        r = self.r
        D = state[1]*(Bbconvert*300)
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
        QoS_idx = 0
       
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
                execution_time = np.maximum( (1/self.R_n + r/self.R_C + H/C_opt)*I_opt  ,  (D-I_opt)*H/C )
                Cost_offloading = execution_time + beta*(a*C_opt+self.C_total*b)*(I_opt*H/((self.C_total)**2))
                Cost_non = D*H/C
                if Cost_offloading > Cost_non :
                    C_opt = 0
                    I_opt = 0



        # calculate completion time
        if C_opt ==0:
            I_opt = 0
            payment = 0
            t_task = D*H/C  # completion time
            t_task_off = 0
            t_process = 0
            t_process_scale = 0
        else:
            I_opt = (D*H/C)/(A+(H/C_opt))
            payment = (a*C_opt+self.C_total*b)*(I_opt*H/((self.C_total)**2))
            t_task_off =  np.maximum( (1/self.R_n + r/self.R_C + H/C_opt)*I_opt  ,  (D-I_opt)*H/C )
            t_task = 0
            t_process = H*I_opt/C_opt
            t_process_scale = int(t_process*10)
            if t_process_scale == 0:
                t_process_scale = 1

        # save F_0 info
        if t_process_scale>0:
            self.C0_usage[t:t+t_process_scale] = self.C0_usage[t:t+t_process_scale]+C_opt/self.C_total

        reward = payment
        cost = beta*payment+t_task+t_task_off
        
        self.cost_diff.append(cost)
        self.time_diff.append(t_task+t_task_off)
        
        if T_lim >= np.maximum(t_task, t_task_off):
            QoS_idx = 1

        return reward, t_task, t_task_off, cost , QoS_idx
    
    def get_state_cheaper(self):
        """state = [h, d, F_l, t_req, F_0, beta]"""
        state = [self.user_set[self.index2,0]/2640,
                 self.user_set[self.index2,1]/(Bbconvert*300),
                 (self.user_set[self.index2,2]-2.39e9)/(2.84e9-2.39e9),
                 self.user_set[self.index2,3],
                 self.C0_usage4[int(self.user_set[self.index2,4])],
                 self.user_set[self.index2,5]/2]
        self.index2+=1
        return np.array(state), self.user_set[self.index2-1,4]
    
    def step_cheaper(self,state,action,t):
        H = state[0]*2640
        r = self.r
        D = state[1]*(Bbconvert*300)
        C = state[2]*(2.84e9-2.39e9)+2.39e9
        T_lim = state[3]
        beta = state[5]*2
        C0 = state[4]*self.C_total
        Cr = self.C_total-C0
        t = int(t)
        a = action[0]
        if a*0.9 >= 0:
            a = a*0.9
        b = action[1]
        if b-5 >= 0 :
            b = b-5
        t_task = 0
        t_task_off = 0
        payment=0
        QoS_idx = 0
       
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
                execution_time = np.maximum( (1/self.R_n + r/self.R_C + H/C_opt)*I_opt  ,  (D-I_opt)*H/C )
                Cost_offloading = execution_time + beta*(a*C_opt+self.C_total*b)*(I_opt*H/((self.C_total)**2))
                Cost_non = D*H/C
                if Cost_offloading > Cost_non :
                    C_opt = 0
                    I_opt = 0



        # calculate completion time
        if C_opt ==0:
            I_opt = 0
            payment = 0
            t_task = D*H/C  # completion time
            t_task_off = 0
            t_process = 0
            t_process_scale = 0
        else:
            I_opt = (D*H/C)/(A+(H/C_opt))
            payment = (a*C_opt+self.C_total*b)*(I_opt*H/((self.C_total)**2))
            t_task_off =  np.maximum( (1/self.R_n + r/self.R_C + H/C_opt)*I_opt  ,  (D-I_opt)*H/C )
            t_task = 0
            t_process = H*I_opt/C_opt
            t_process_scale = int(t_process*10)
            if t_process_scale == 0:
                t_process_scale = 1


      

        # save F_0 info
        if t_process_scale>0:
            self.C0_usage4[t:t+t_process_scale] = self.C0_usage4[t:t+t_process_scale]+C_opt/self.C_total

        reward = payment
        cost = beta*payment+t_task+t_task_off
        
        self.cost_cheaper.append(cost)
        self.time_cheaper.append(t_task+t_task_off)
        
        if T_lim >= np.maximum(t_task, t_task_off):
            QoS_idx = 1


        return reward, t_task, t_task_off, cost , QoS_idx


    
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
        user_idx = 0
        for t in range(num_time):
            self.add_user_to_queue3(t,self.user_set)
            C_r = (self.C_total - self.C0_usage3[t]*self.C_total)
            epsilon = np.random.rand(1)
            if C_r == self.C_total :
                C_r = C_r * (0.85+epsilon*0.15)

            if(t> t_available and C_r>0):
                if(len(self.queue3)!=0):
                    N = len(self.queue3)
                    A = (1/self.R_n) + (self.r/R_C) + (self.queue3[:,0]/(self.queue3[:,2]))
                    t_total_temp = 0
                    b= 1/(2.6e9)
                    I_opt = (self.queue3[:,1]*self.queue3[:,0]/self.queue3[:,2]) / ((self.queue3[:,0]/(C_r/N))+A)
                    I_opt[b>1/self.queue3[:,2]]=0
                    I_opt[I_opt<0]=0
                    payment_temp = b*I_opt*self.queue3[:,0]
                    payment_b = np.sum( payment_temp)
                    t_process_mat = I_opt*self.queue3[:,0]/ (C_r/N)
                    t_process_max_b = np.amax(t_process_mat)
                    t_process_median_b = np.median(t_process_mat)
                    t_total_temp = (self.queue3[:,1]-I_opt)*self.queue3[:,0]/self.queue3[:,2] #+ ((1/self.R_n) + (self.r/R_C))*I_opt
                    t_total_mat =np.sum( t_total_temp )
                    profit+=payment_b
                    
                    
                    t_available = int(t + (t_process_median_b*10))
                    total_time_uni += t_total_mat
                    total_time_uni_off += t_total_mat
                    
                    avg_cost += t_total_mat+payment_b

                    num_off = len(t_process_mat[t_process_mat>0])

                    user_idx += len(self.queue3)

                    idx += len(t_total_temp[t_total_temp<=self.queue3[:,3]])
                    self.queue3 = None
                    for k in range(N):
                        self.cost_det.append(t_total_temp[k]+payment_temp[k])
                        self.time_det.append(t_total_temp[k])
                        if t_process_mat[k] >0:
                            if int(t_process_mat[k]*10) == 0 :
                                self.C0_usage3[t:t+1]=self.C0_usage3[t:t+1] + (C_r/num_off)/self.C_total

                            else:
                                self.C0_usage3[t:t+int(t_process_mat[k]*10)+1]=self.C0_usage3[t:t+int(t_process_mat[k]*10)+1] + (C_r/num_off)/self.C_total


                    
            else: 
                time_temp= self.queue3[:,0]*self.queue3[:,1]/self.queue3[:,2]
                total_time_uni += np.sum(time_temp)
                avg_cost += np.sum(time_temp)
                user_idx +=len(self.queue3)
                idx += len(time_temp[time_temp<=self.queue3[:,3]])
                self.queue3 = None

        return profit, total_time_uni/len(self.user_set), total_time_uni_off/idx, avg_cost/len(self.user_set), idx/user_idx
    
    
    def reset(self):
        self.C0_usage = np.zeros(self.num_time)
        self.C0_usage2 = np.zeros(self.num_time)
        self.C0_usage3 = np.zeros(self.num_time)
        self.C0_usage4 = np.zeros(self.num_time)
        self.index2 = 0
        self.index = 0
        self.user_num = 0
        self.avg_completion_time = 0
        self.time_diff = []
        self.time_uni = []
        self.time_det = []
        self.time_cheaper = []
        self.time_non = []
        self.cost_diff = []
        self.cost_uni =[]
        self.cost_det = []
        self.cost_cheaper = []
        self.cost_non = []

    



C_total = 110e9
R_n = 41.2e6 
R_C = 360.3e6 
num_time = 6000  # 10 min
state_dim = 6
action_dim = 300*200
max_episodes = 300


input_size = 6
hidden_size = 512 
hidden_size2 = 256 
hidden_size3 = 128 
hidden_size4 = 64 
num_classes = 2


model = NeuralNet(input_size, hidden_size, hidden_size2, hidden_size3, hidden_size4, num_classes).to(device)
model.load_state_dict(torch.load("trained_model.ckpt"))

model.eval()

time_step = 0

profit_list_cheaper_1 =[]
profit_list_cheaper_2 =[]
profit_list_cheaper_3 =[]
profit_list_cheaper_4 =[]
time_list_cheaper_1 = []
time_list_cheaper_2 = []
time_list_cheaper_3 = []
time_list_cheaper_4 = []
cost_list_cheaper_1 = []
cost_list_cheaper_2 = []
cost_list_cheaper_3 = []
cost_list_cheaper_4 = []
time_list_cheaper = []
offtime_list_cheaper = []
profit_list_cheaper = []
cost_list_cheaper = []
QoS_list_cheaper = []
QoS_list_cheaper_1 = []
QoS_list_cheaper_2 = []
QoS_list_cheaper_3 = []
QoS_list_cheaper_4 = []

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
QoS_list=[]
QoS_list2=[]
QoS_list3=[]

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
QoS_list_1 = []
QoS_list2_1 = []
QoS_list3_1 = []

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
QoS_list_2 = []
QoS_list2_2 = []
QoS_list3_2 = []

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
QoS_list_3 = []
QoS_list2_3 = []
QoS_list3_3 = []

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

time_list4_1=[]
time_list4_2=[]
time_list4_3=[]
time_list4_4=[]
""" Simulatation """

first_interval = 100
second_interval = 200
third_interval = 300

with torch.no_grad():
    reward_array_diff = np.zeros((max_episodes, num_time))
    for i_episode in range(1, max_episodes+1):
        if i_episode <= first_interval:
            user_rate = 1
        if first_interval<i_episode<=second_interval:
            user_rate = 2
        if second_interval<i_episode<=third_interval:
            user_rate = 3
        if third_interval<i_episode:
            user_rate = 4

        env = Environment_HS(user_rate, C_total, R_n, R_C, num_time, state_dim, action_dim)
        state_dim = env.state_dim
        action_dim = env.action_dim
        print("############################ episode {} ############################".format(i_episode))

        user_set = env.generate_user()
        if i_episode <=first_interval:
            time_list4_1.append(env.avg_completion_time)
        if first_interval<i_episode<=second_interval:
            time_list4_2.append(env.avg_completion_time)
        if second_interval<i_episode<=third_interval:
            time_list4_3.append(env.avg_completion_time)
        if third_interval<i_episode:
            time_list4_4.append(env.avg_completion_time)
            
        completion_time_total_diff = 0

        running_reward_diff = 0
        running_cost_diff = 0
        idx2=0
        
        completion_time_total_diff_cheaper = 0
        running_reward_diff_cheaper = 0
        running_cost_diff_cheaper = 0
        idx4=0
        
        completion_time_total_uni = 0

        running_reward_uni = 0
        running_cost_uni = 0
        idx3=0
        QoS_idx = 0
        QoS_idx_cheaper = 0
        
        with torch.no_grad():
            for t in range(len(user_set)):
                state, time= env.get_state()
                state_cheaper, time_cheaper = env.get_state_cheaper() 

                state = torch.tensor(state).float().to(device)
                state_cheaper = torch.tensor(state_cheaper).float().to(device) 
                action = model(state)
                action_cheaper = model(state_cheaper) 
                state = state.cpu().detach().numpy()
                action = action.cpu().detach().numpy()
                state_cheaper = state_cheaper.cpu().detach().numpy()
                action_cheaper = action_cheaper.cpu().detach().numpy()
                
                action = [int(action[0]),int(action[1])] 
                action_cheaper = [int(action_cheaper[0]), int(action_cheaper[1])]
                
                if action[0]<=0:
                    action[0]=1
                if action[1]<=0:
                    action[1]=1
                    
                if action_cheaper[0]<=0:
                    action_cheaper[0]=1
                if action_cheaper[1]<=0:
                    action_cheaper[1]=1
                    
                reward, comp_time, comp_time_off,cost, QoS = env.step(state, action, time)
                reward_cheaper, comp_time_cheaper, comp_time_off_cheaper, cost_cheaper, QoS_cheaper = env.step_cheaper(state_cheaper, action_cheaper, time_cheaper)
                QoS_idx += QoS
                QoS_idx_cheaper += QoS_cheaper
                if comp_time_off != 0:
                    idx2 +=1
                if comp_time_off != 0:
                    idx4 +=1
                
                running_reward_diff += reward
                running_cost_diff += cost
                completion_time_total_diff += comp_time+comp_time_off
                
                running_reward_diff_cheaper += reward_cheaper
                running_cost_diff_cheaper += cost_cheaper
                completion_time_total_diff_cheaper += comp_time_cheaper + comp_time_off_cheaper

            if i_episode <= first_interval:
                profit_list_1.append(running_reward_diff)
                time_list_1.append(completion_time_total_diff/len(user_set))
                cost_list_1.append(running_cost_diff/len(user_set))
                QoS_list_1.append(QoS_idx/len(user_set))
                profit_list_cheaper_1.append(running_reward_diff_cheaper)
                time_list_cheaper_1.append(completion_time_total_diff_cheaper/len(user_set))
                cost_list_cheaper_1.append(running_cost_diff_cheaper/len(user_set))
                QoS_list_cheaper_1.append(QoS_idx_cheaper/len(user_set))
            if first_interval<i_episode<=second_interval:
                profit_list_2.append(running_reward_diff)
                time_list_2.append(completion_time_total_diff/len(user_set))
                cost_list_2.append(running_cost_diff/len(user_set))
                QoS_list_2.append(QoS_idx/len(user_set))
                profit_list_cheaper_2.append(running_reward_diff_cheaper)
                time_list_cheaper_2.append(completion_time_total_diff_cheaper/len(user_set))
                cost_list_cheaper_2.append(running_cost_diff_cheaper/len(user_set))
                QoS_list_cheaper_2.append(QoS_idx_cheaper/len(user_set))
            if second_interval<i_episode<=third_interval:
                profit_list_3.append(running_reward_diff)
                time_list_3.append(completion_time_total_diff/len(user_set))
                cost_list_3.append(running_cost_diff/len(user_set))
                QoS_list_3.append(QoS_idx/len(user_set))
                profit_list_cheaper_3.append(running_reward_diff_cheaper)
                time_list_cheaper_3.append(completion_time_total_diff_cheaper/len(user_set))
                cost_list_cheaper_3.append(running_cost_diff_cheaper/len(user_set))
                QoS_list_cheaper_3.append(QoS_idx_cheaper/len(user_set))
            if third_interval<i_episode:
                profit_list_4.append(running_reward_diff)
                time_list_4.append(completion_time_total_diff/len(user_set))
                cost_list_4.append(running_cost_diff/len(user_set))
                profit_list_cheaper_4.append(running_reward_diff_cheaper)
                time_list_cheaper_4.append(completion_time_total_diff_cheaper/len(user_set))
                cost_list_cheaper_4.append(running_cost_diff_cheaper/len(user_set))
            if i_episode == first_interval-1:
                diff_C0_usage_1 = env.C0_usage
                diff_C0_usage_cheaper_1 = env.C0_usage4
            if i_episode == second_interval-1:
                diff_C0_usage_2 = env.C0_usage
                diff_C0_usage_cheaper_2 = env.C0_usage4
            if i_episode == third_interval-1:
                diff_C0_usage_3 = env.C0_usage
                diff_C0_usage_cheaper_3 = env.C0_usage4
                
            profit_list.append(running_reward_diff)
            time_list.append(completion_time_total_diff/len(user_set))
            cost_list.append(running_cost_diff/len(user_set))
            QoS_list.append(QoS_idx/len(user_set))
        
            profit_list_cheaper.append(running_reward_diff_cheaper)
            time_list_cheaper.append(completion_time_total_diff_cheaper/len(user_set))
            cost_list_cheaper.append(running_cost_diff_cheaper/len(user_set))
            QoS_list_cheaper.append(QoS_idx_cheaper/len(user_set))


        """uniform pricing simulation"""
        profit2, avg_time2, avg_time_off2, avg_cost2, QoS2  = env.step_uniform()
        profit3, avg_time3, avg_time_off3, avg_cost3, QoS3  = env.step_fool()

        if i_episode <= first_interval:
            profit_list2_1.append(profit2)
            time_list2_1.append(avg_time2)
            offtime_list2_1.append(avg_time_off2)
            cost_list2_1.append(avg_cost2)
            profit_list3_1.append(profit3)
            time_list3_1.append(avg_time3)
            offtime_list3_1.append(avg_time_off3)
            cost_list3_1.append(avg_cost3)
        if first_interval<i_episode<=second_interval:
            profit_list2_2.append(profit2)
            time_list2_2.append(avg_time2)
            offtime_list2_2.append(avg_time_off2)
            cost_list2_2.append(avg_cost2)
            profit_list3_2.append(profit3)
            time_list3_2.append(avg_time3)
            offtime_list3_2.append(avg_time_off3)
            cost_list3_2.append(avg_cost3)
        if second_interval<i_episode<=third_interval:
            profit_list2_3.append(profit2)
            time_list2_3.append(avg_time2)
            offtime_list2_3.append(avg_time_off2)
            cost_list2_3.append(avg_cost2)
            profit_list3_3.append(profit3)
            time_list3_3.append(avg_time3)
            offtime_list3_3.append(avg_time_off3)
            cost_list3_3.append(avg_cost3)
        if third_interval<i_episode:
            profit_list2_4.append(profit2)
            time_list2_4.append(avg_time2)
            offtime_list2_4.append(avg_time_off2)
            cost_list2_4.append(avg_cost2)
            profit_list3_4.append(profit3)
            time_list3_4.append(avg_time3)
            offtime_list3_4.append(avg_time_off3)
            cost_list3_4.append(avg_cost3)
        if i_episode == first_interval-2:
            uni_C0_usage_1 = env.C0_usage2
        if i_episode == second_interval-2:
            uni_C0_usage_2 = env.C0_usage2
        if i_episode == third_interval-2:
            uni_C0_usage_3 = env.C0_usage2
        profit_list2.append(profit2)
        time_list2.append(avg_time2)
        offtime_list2.append(avg_time_off2)
        cost_list2.append(avg_cost2)
        profit_list3.append(profit3)
        time_list3.append(avg_time3)
        offtime_list3.append(avg_time_off3)
        cost_list3.append(avg_cost3)
        QoS_list2.append(QoS2)
        QoS_list3.append(QoS3)

        env.reset()

        print("    linear pricing  : revenue {}, total avg time {}, QoS {}".format(profit_list[-1], time_list[-1], QoS_list[-1]))
        print("    uniform pricing : revenue {}, total avg time {}, QoS {}".format(profit_list2[-1], time_list2[-1], QoS_list2[-1]))
        print("    fool   pricing  : revenue {}, total avg time {}, QoS {}".format(profit_list3[-1], time_list3[-1], QoS_list3[-1]))

###############################################################################################################

data = [profit_list_1, profit_list2_1,profit_list3_1,
       profit_list_2, profit_list2_2, profit_list3_2,
       profit_list_3, profit_list2_3, profit_list3_3]

font_size = 10

plt.figure(figsize=(10,4))
plt.title("")
plt.subplot(1, 3, 1)
plt.xlabel(' Pricing scheme \n (a) R = 1 ',fontsize=font_size)
plt.ylabel('Revenue',fontsize=font_size)
# plt.title('(a) R = 1',fontsize=font_size)
plt.yscale('linear')
plt.grid(axis='both',alpha=0.5, linestyle='--')
plt.boxplot(data[0], labels=['Differential'], positions=[0], widths = 0.8, patch_artist=True, boxprops=dict(facecolor='red', color='k'), 
            medianprops=dict(color='white'), showmeans=True, meanprops={"marker":"o",
                       "markerfacecolor":"white", 
                       "markeredgecolor":"black",
                      "markersize":"7"}, showfliers=False )
plt.boxplot(data[1], labels=['Uniform'], positions=[1], widths = 0.8, patch_artist=True, boxprops=dict(facecolor='blue', color='k'), 
            medianprops=dict(color='white'), showmeans=True, meanprops={"marker":"o",
                       "markerfacecolor":"white", 
                       "markeredgecolor":"black",
                      "markersize":"7"}, showfliers=False )
plt.boxplot(data[2], labels=['Deterministic'], positions=[2], widths = 0.8, patch_artist=True, boxprops=dict(facecolor='gold', color='k'), 
            medianprops=dict(color='white'), showmeans=True, meanprops={"marker":"o",
                       "markerfacecolor":"white", 
                       "markeredgecolor":"black",
                      "markersize":"7"}, showfliers=False )

plt.subplot(1, 3, 2)
plt.xlabel(' Pricing scheme \n (b) R = 2 ',fontsize=font_size)
plt.ylabel('Revenue',fontsize=font_size)
# plt.title('(a) R = 2',fontsize=font_size)
plt.yscale('linear')
plt.grid(axis='both',alpha=0.5, linestyle='--')
plt.boxplot(data[3], labels=['Differential'], positions=[0], widths = 0.8, patch_artist=True, boxprops=dict(facecolor='red', color='k'), 
            medianprops=dict(color='white'), showmeans=True, meanprops={"marker":"o",
                       "markerfacecolor":"white", 
                       "markeredgecolor":"black",
                      "markersize":"7"}, showfliers=False )
plt.boxplot(data[4], labels=['Uniform'], positions=[1], widths = 0.8, patch_artist=True, boxprops=dict(facecolor='blue', color='k'), 
            medianprops=dict(color='white'), showmeans=True, meanprops={"marker":"o",
                       "markerfacecolor":"white", 
                       "markeredgecolor":"black",
                      "markersize":"7"}, showfliers=False )
plt.boxplot(data[5], labels=['Deterministic'], positions=[2], widths = 0.8, patch_artist=True, boxprops=dict(facecolor='gold', color='k'), 
            medianprops=dict(color='white'), showmeans=True, meanprops={"marker":"o",
                       "markerfacecolor":"white", 
                       "markeredgecolor":"black",
                      "markersize":"7"}, showfliers=False )

plt.subplot(1, 3, 3)
plt.xlabel(' Pricing scheme \n (c) R = 3 ',fontsize=font_size)
plt.ylabel('Revenue',fontsize=font_size)
# plt.title('(a) R = 3',fontsize=font_size)
plt.yscale('linear')
plt.grid(axis='both',alpha=0.5, linestyle='--')
plt.boxplot(data[6], labels=['Differential'], positions=[0], widths = 0.8, patch_artist=True, boxprops=dict(facecolor='red', color='k'), 
            medianprops=dict(color='white'), showmeans=True, meanprops={"marker":"o",
                       "markerfacecolor":"white", 
                       "markeredgecolor":"black",
                      "markersize":"7"}, showfliers=False )
plt.boxplot(data[7], labels=['Uniform'], positions=[1], widths = 0.8, patch_artist=True, boxprops=dict(facecolor='blue', color='k'), 
            medianprops=dict(color='white'), showmeans=True, meanprops={"marker":"o",
                       "markerfacecolor":"white", 
                       "markeredgecolor":"black",
                      "markersize":"7"}, showfliers=False )
plt.boxplot(data[8], labels=['Deterministic'], positions=[2], widths = 0.8, patch_artist=True, boxprops=dict(facecolor='gold', color='k'), 
            medianprops=dict(color='white'), showmeans=True, meanprops={"marker":"o",
                       "markerfacecolor":"white", 
                       "markeredgecolor":"black",
                      "markersize":"7"}, showfliers=False )
plt.tight_layout()

plt.savefig('./figure/rev_Average_revenue.png', format='png')
plt.savefig('./figure/rev_Average_revenue.eps', format='eps')

print("Diff = {}, {}, {}".format(np.mean(data[0]),np.mean(data[3]), np.mean(data[6])))
print("Uni  = {}, {}, {}".format(np.mean(data[1]),np.mean(data[4]), np.mean(data[7])))
print("Det  = {}, {}, {}".format(np.mean(data[2]),np.mean(data[5]), np.mean(data[8])))


###############################################################################################################


plt.figure(figsize=(10,5))
plt.title("")
font_size = 10
start_num = 1600 
end_num = start_num+100

plt.subplot(3,3,1)

plt.ylabel(" Differential \n pricing \n\n Server resource \n usage ratio", fontsize=font_size)

plt.grid(axis='y',alpha=0.5, linestyle='--')
plt.fill_between(np.linspace(0,10,100), diff_C0_usage_1[start_num:end_num], alpha=1, color='tab:red')
plt.xlim(0,10)
plt.ylim(0,1)


plt.subplot(3,3,2)

plt.grid(axis='y',alpha=0.5, linestyle='--')
plt.fill_between(np.linspace(0,10,100), diff_C0_usage_2[start_num:end_num], alpha=1, color='tab:red')
plt.xlim(0,10)
plt.ylim(0,1)

plt.subplot(3,3,3)
plt.grid(axis='y',alpha=0.5, linestyle='--')
plt.fill_between(np.linspace(0,10,100), diff_C0_usage_3[start_num:end_num], alpha=1, color='tab:red')
plt.xlim(0,10)
plt.ylim(0,1)

plt.subplot(3,3,4)
plt.ylabel(" Uniform \n pricing \n\n Server resource \n usage ratio", fontsize=font_size)
plt.grid(axis='y',alpha=0.5, linestyle='--')
plt.fill_between(np.linspace(0,10,100), uni_C0_usage_1[start_num:end_num], alpha=1, color='C0')
plt.xlim(0,10)
plt.ylim(0,1)

plt.subplot(3,3,5)
plt.grid(axis='y',alpha=0.5, linestyle='--')
plt.fill_between(np.linspace(0,10,100), uni_C0_usage_2[start_num:end_num], alpha=1, color='C0')
plt.xlim(0,10)
plt.ylim(0,1)

plt.subplot(3,3,6)
plt.grid(axis='y',alpha=0.5, linestyle='--')
plt.fill_between(np.linspace(0,10,100), uni_C0_usage_3[start_num:end_num], alpha=1, color='C0')
plt.xlim(0,10)
plt.ylim(0,1)

plt.subplot(3,3,7)
plt.xlabel(" Time [sec.] \n\n  R = 1 ", fontsize=font_size)
plt.ylabel(" Cheaper-\nDifferential \n pricing \n\n Server resource \n usage ratio", fontsize=font_size)
plt.grid(axis='y',alpha=0.5, linestyle='--')
plt.fill_between(np.linspace(0,10,100), diff_C0_usage_cheaper_1[start_num:end_num], alpha=1, color='tab:green')
plt.xlim(0,10)
plt.ylim(0,1)

plt.subplot(3,3,8)
plt.xlabel(" Time [sec.] \n\n  R = 2 ", fontsize=font_size)
plt.grid(axis='y',alpha=0.5, linestyle='--')
plt.fill_between(np.linspace(0,10,100), diff_C0_usage_cheaper_2[start_num:end_num], alpha=1, color='tab:green')
plt.xlim(0,10)
plt.ylim(0,1)

plt.subplot(3,3,9)
plt.xlabel(" Time [sec.] \n\n  R = 3 ", fontsize=font_size)
plt.grid(axis='y',alpha=0.5, linestyle='--')
plt.fill_between(np.linspace(0,10,100), diff_C0_usage_cheaper_3[start_num:end_num], alpha=1, color='tab:green')
plt.xlim(0,10)
plt.ylim(0,1)

plt.tight_layout()

plt.savefig('./figure/rev_Resource_usage.eps', format='eps')
plt.savefig('./figure/rev_Resource_usage.png', format='png')

###############################################################################################################

def create_x(t, w, n, d):
    return [t*x + w*n for x in range(d)]
font_size = 10
num_bar = 4
num_dataset = 3
bar_width = 0.8
x_val_non = create_x(num_bar,bar_width,1,num_dataset)
x_val_det = create_x(num_bar,bar_width,2,num_dataset)
x_val_uni = create_x(num_bar,bar_width,3,num_dataset)
x_val_diff = create_x(num_bar,bar_width,4,num_dataset)

plt.figure(figsize=(5,4))
plt.grid(axis='y',alpha=0.5, linestyle='--')
plt.xlabel("Average user arrival rate (R)", fontsize=font_size)
plt.ylabel("Average execution delay [sec.]", fontsize=font_size)

line_non = plt.bar(x_val_non, [np.mean(time_list4_1),np.mean(time_list4_2),np.mean(time_list4_3)],color='black',width=0.7)
line_det = plt.bar(x_val_det, [np.mean(time_list3_1),np.mean(time_list3_2),np.mean(time_list3_3)],color='gold',width=0.7)
line_uni = plt.bar(x_val_uni, [np.mean(time_list2_1),np.mean(time_list2_2),np.mean(time_list2_3)],color='C0',width=0.7)
line_diff = plt.bar(x_val_diff, [np.mean(time_list_1),np.mean(time_list_2),np.mean(time_list_3)], color='tab:red',width=0.7)
print(np.transpose([confidence_interval_diff_1,confidence_interval_diff_2,confidence_interval_diff_3]))
print("non = {}, {}, {}".format(np.mean(time_list4_1),np.mean(time_list4_2),np.mean(time_list4_3)))
print("det = {}, {}, {}".format(np.mean(time_list3_1),np.mean(time_list3_2),np.mean(time_list3_3)))
print("uni = {}, {}, {}".format(np.mean(time_list2_1),np.mean(time_list2_2),np.mean(time_list2_3)))
print("diff= {}, {}, {}".format(np.mean(time_list_1),np.mean(time_list_2),np.mean(time_list_3)))

middle_x = [(a+b+c+d)/4 for (a,b,c,d) in zip (x_val_non, x_val_det, x_val_uni, x_val_diff)]
plt.xticks(middle_x, labels = [1,2,3])
plt.ylim([0,3.5])

plt.legend(handles=(line_non,line_det,line_uni,line_diff), labels = ('Without offloading', 'Deterministic pricing', 'Uniform pricing', 'Differential pricing'))

plt.savefig('./figure/rev_Average_execution_delay.eps', format='eps')
plt.savefig('./figure/rev_Average_execution_delay.png', format='png')

###############################################################################################################


def create_x(t, w, n, d):
    return [t*x + w*n for x in range(d)]
font_size=10
num_bar = 4
num_dataset = 3
bar_width = 0.8
x_val_non = create_x(num_bar,bar_width,1,num_dataset)
x_val_det = create_x(num_bar,bar_width,2,num_dataset)
x_val_uni = create_x(num_bar,bar_width,3,num_dataset)
x_val_diff = create_x(num_bar,bar_width,4,num_dataset)

plt.figure(figsize=(5,4))
plt.grid(axis='y',alpha=0.5, linestyle='--')
plt.xlabel("Average user arrival rate (R)", fontsize=font_size)
plt.ylabel("Average cost", fontsize=font_size)

line_non = plt.bar(x_val_non, [np.mean(time_list4_1),np.mean(time_list4_2),np.mean(time_list4_3)],color='black',width=0.7)
line_det = plt.bar(x_val_det, [np.mean(cost_list3_1),np.mean(cost_list3_2),np.mean(cost_list3_3)],color='gold',width=0.7)
line_uni = plt.bar(x_val_uni, [np.mean(cost_list2_1),np.mean(cost_list2_2),np.mean(cost_list2_3)],color='C0',width=0.7)
line_diff = plt.bar(x_val_diff, [np.mean(cost_list_1),np.mean(cost_list_2),np.mean(cost_list_3)], color='tab:red',width=0.7)
print(np.transpose([confidence_interval_diff_1,confidence_interval_diff_2,confidence_interval_diff_3]))
print("non = {}, {}, {}".format(np.mean(time_list4_1),np.mean(time_list4_2),np.mean(time_list4_3)))
print("det = {}, {}, {}".format(np.mean(cost_list3_1),np.mean(cost_list3_2),np.mean(cost_list3_3)))
print("uni = {}, {}, {}".format(np.mean(cost_list2_1),np.mean(cost_list2_2),np.mean(cost_list2_3)))
print("diff= {}, {}, {}".format(np.mean(cost_list_1),np.mean(cost_list_2),np.mean(cost_list_3)))

middle_x = [(a+b+c+d)/4 for (a,b,c,d) in zip (x_val_non, x_val_det, x_val_uni, x_val_diff)]
plt.xticks(middle_x, labels = [1,2,3])
plt.ylim([0,3.5])

plt.legend(handles=(line_non,line_det,line_uni,line_diff), labels = ('Without offloading', 'Deterministic pricing', 'Uniform pricing', 'Differential pricing'))

plt.savefig('./figure/rev_Average_cost.eps', format='eps')
plt.savefig('./figure/rev_Average_cost.png', format='png')
