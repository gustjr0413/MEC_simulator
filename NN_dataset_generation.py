import pandas as pd
from transform import transform
from Environment_search import Environment_search


num_user = 1000000
C_total = 110e9
R_n = 41.2e6 # uplink data rate
R_C = 360.3e6 # downlink data rate
state_dim = 6
action_dim = 300*200



for user in range(num_user):
    state, t = env.get_state()
    action, max_payment = env.search(state)
    if (user+1)%100==0:
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
