import numpy as np
import pandas as pd
import Environment_search as Env
import transform as tr


num_user = 1000000
C_total = 110e9
R_n = 43.9e6
R_C = 150.6e6
state_dim = 6
action_dim = 300*200



env = Environment_search(num_user, C_total, R_n, R_C, state_dim, action_dim)
state_dim = env.state_dim
action_dim = env.action_dim
state_action_list = []

user_set = env.generate_user()
for user in range(num_user):
    state, t = env.get_state()
    action, max_payment = env.search(state)
    if (user+1)%50==0:
        print(" user {} update is finished".format(user+1))
        print(" action {}, payment {}".format(transform().to2dim(action), max_payment))
    temp_file = state.tolist()
    temp_file.append(transform().to2dim(action)[0])
    temp_file.append(transform().to2dim(action)[1])
    state_action_list.append(temp_file)
    if (user+1)%10000 == 0:
        time_pd = pd.time_pd = pd.DataFrame(state_action_list)
        time_pd.to_csv("final_dataset_{}.csv".format(user+1), mode='w')

time_pd = pd.DataFrame(state_action_list)

time_pd.to_csv("final_dataset.csv", mode='w')


print("####################################################################")
print("dataset is generated")
print("####################################################################")
