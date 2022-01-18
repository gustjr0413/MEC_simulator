import numpy as np
from Rmatrix import Rmatrix
from transform import transform

class Environment_search:
    def __init__ (self, num_user, C_total, R_n, R_C, state_dim, action_dim):
        self.num_user = num_user
        self.r = 0.2 # Input/output ratio
        self.C_total = C_total # Total CPU capacity of server [Hz]
        self.R_n = R_n # user->server datarate [bps]
        self.R_C = R_C # server->user datarate [bps]
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.avg_completion_time = 0
        self.index=0


    def generate_user(self): # get user set
        user_set = np.zeros([self.num_user,6])
        C_list = np.array([2.84e9, 2.39e9, 2.5e9])
        T_lim_list = np.array([0.1, 0.5, 1])
        beta_list = np.array([1/2,1,2])
        idx = 0
        t_task_origin = 0
        for i in range(self.num_user):
            ### [h,d,F_l,t_req,F_0,beta]
            user_set[idx,0] = 2640  # H
            user_set[idx,1] = 1024*8*np.random.randint(low=100, high=300+1) # D
            user_set[idx,2] = C_list[np.random.randint(low=0, high=3)] # C
            user_set[idx,3] = T_lim_list[np.random.randint(low=0,high=3)] # T_lim
            user_set[idx,4] = np.random.randint(low=0, high=(self.C_total/1e9)+1)*1e9
            user_set[idx,5] = beta_list[np.random.randint(low=0,high=3)] # beta
            t_task_origin += np.ceil(user_set[idx,0]*user_set[idx,1]/user_set[idx,2])
            idx = idx + 1
        self.user_set=user_set
        self.user_num=idx
        self.avg_completion_time=t_task_origin/len(user_set)
        print("######### new users are generated ########")
        print("total number of users = ", self.user_num)
        print("total completion time = ", self.avg_completion_time)
        print("##########################################")
        return user_set


    def get_state(self):
        """state = [h, d, F_l, t_req, F_0, beta]"""
        state = [self.user_set[self.index,0]/2640,
                 self.user_set[self.index,1]/(1024*8*300),
                 (self.user_set[self.index,2]-2.39e9)/(2.84e9-2.39e9),
                 self.user_set[self.index,3],
                 self.user_set[self.index,4]/(self.C_total),
                 self.user_set[self.index,5]/2]
        self.index+=1
        return np.array(state), self.user_set[self.index-1,4]


    def search(self, state):
        H = state[0]*2640
        r = self.r
        D = state[1]*(1024*8*300)
        C = state[2]*(2.84e9-2.39e9)+2.39e9
        T_lim = state[3]
        C0 = state[4]*(self.C_total)
        beta = state[5]*2
        Cr = self.C_total-C0
        M = Rmatrix(self.action_dim)
        for idx in range(self.action_dim):
            a = transform().to2dim(idx)[0]
            b = transform().to2dim(idx)[1]

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
                C_max = ((self.C_total)/a)*( (self.C_total/(C*beta)) - b)

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




            # calculate payment
            if C_opt ==0:
                I_opt = 0
                payment = 0
            else:
                I_opt = (D*H/C)/(A+(H/C_opt))
                payment =(a*C_opt+self.C_total*b)*(I_opt*H/((self.C_total)**2))


            M.update(idx, payment)

        action = M.select_action()
        max_payment = M.M[action]
        M.reset()
        return action, max_payment
