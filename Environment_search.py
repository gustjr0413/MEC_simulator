import numpy as np

class Environment_search:
    def __init__ (self, num_user, C_total, R_n, R_C, state_dim, action_dim):
        self.num_user = num_user
        self.r = 0.1 # Input/output ratio
        self.C_total = C_total # Total CPU capacity of server [Hz]
        self.R_n = R_n # user->server datarate [bps]
        self.R_C = R_C # server->user datarate [bps]
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.avg_completion_time = 0
        self.index=0


    def generate_user(self): # get user set
        user_set = np.zeros([self.num_user,5])
        C_list = np.linspace(0.1e9,1e9,10)
        T_lim_list = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
        idx = 0
        t_task_origin = 0
        for i in range(self.num_user):
            ### [H,D,C,T_req,C0]
            user_set[idx,0] = np.random.randint(low=500, high=1500+1) # H
            user_set[idx,1] = 1000*8*np.random.randint(low=100, high=500+1) # D
            user_set[idx,2] = C_list[np.random.randint(low=0, high=10)] # C
            user_set[idx,3] = T_lim_list[np.random.randint(low=0,high=10)] # T_lim
            user_set[idx,4] = np.random.randint(low=0, high=(self.C_total/1e9)+1)*1e9 # C0
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
        """state = [H, D, C, T_lim, C0]"""
        state = [self.user_set[self.index,0]/1500,
                 self.user_set[self.index,1]/(1000*8*500),
                 self.user_set[self.index,2]/(1e9),
                 self.user_set[self.index,3],
                 self.user_set[self.index,4]/(self.C_total)]
        self.index+=1
        return np.array(state)


    def search(self, state):
        H = state[0]*1500
        r = self.r
        D = state[1]*(1000*8*500)
        C = state[2]*1e9
        T_lim = (D*H/C)*state[3]
        C0 = state[4]*(self.C_total)
        Cr = self.C_total-C0
        M = Rmatrix(self.action_dim)
        for idx in range(self.action_dim):
            a = transform().to2dim(idx)[0]
            b = transform().to2dim(idx)[1]

             # substitution
            A = (1/self.R_n) + (r/self.R_C) + (H/C)
            B = (b*self.C_total) - (self.C_total**2)/C

            # solution
            if ( B>=0 ):
                C_opt = 0
            else:
                # clip

                C_b = (-(a*H)+np.sqrt((a**2)*(H**2)-2*a*A*B*H) )/(a*A)
                C_min = (C*H*(D*H-C*T_lim))/(D*(H**2)-A*C*(D*H-C*T_lim))
                C_max = ((2*self.C_total)/a)*( (self.C_total/C) - b)

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
                Cost_offloading = np.maximum( (1/self.R_n + r/self.R_C + H/C_opt)*I_opt  ,  (D-I_opt)*H/C ) + (a*C_opt+2*self.C_total*b)*(I_opt*H/(2*(self.C_total)**2))
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
                payment =(a*C_opt+2*self.C_total*b)*(I_opt*H/(2*(self.C_total)**2))


            M.update(idx, payment)

        action = M.select_action()
        max_payment = M.M[action]
        M.reset()
        return action, max_payment
