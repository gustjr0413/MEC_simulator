import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from tqdm import tqdm
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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

    
class TrainDataset(Dataset):
    def __init__(self, train_set, train_label):
        self.len = train_set.shape[0]
        self.x_data = torch.from_numpy(train_set)
        self.y_data = torch.from_numpy(train_label)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len
    
class TestDataset(Dataset):
    def __init__(self, test_set, test_label):
        self.len = test_set.shape[0]
        self.x_data = torch.from_numpy(test_set)
        self.y_data = torch.from_numpy(test_label)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len  
    

xy = pd.get_dummies(pd.read_csv('final_dataset_1000000.csv')).values[:,1:]
print("Dataset shape = {}".format(xy.shape))
x_data = xy[:,0:6]
y_data = xy[:,6:8]

# Test set
t_xy = pd.get_dummies(pd.read_csv('./test/testdata/final_dataset_100000.csv')).values[:,1:]
print("Test Dataset shape = {}".format(t_xy.shape))
t_x_data = xy[:,0:6]
t_y_data = xy[:,6:8]
test_dataset = TestDataset(t_x_data, t_y_data)
test_loader = DataLoader(dataset=test_dataset, batch_size = 1024, shuffle=False)


input_size = 6
hidden_size = 512
hidden_size2 = 256
hidden_size3 = 128
hidden_size4 = 64
num_classes = 2
num_sub_epochs = 1 # *5 = real epoch
num_steps = 4000
num_splits = 5

model = NeuralNet(input_size, hidden_size, hidden_size2, hidden_size3, hidden_size4, num_classes).to(device)
criterion = nn.L1Loss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.005)

kfold = KFold(n_splits=num_splits, random_state=0, shuffle=False)

t_loss = []
v_loss = []
test_loss = []
for step in tqdm(range(num_steps)):
    train_loss_list = []
    valid_loss_list = []
    test_loss_list = []
    for train_index, valid_index in kfold.split(x_data):
        X_train, X_valid = x_data[train_index], x_data[valid_index]
        Y_train, Y_valid = y_data[train_index], y_data[valid_index]
        train_dataset = TrainDataset(X_train, Y_train)
        valid_dataset = TestDataset(X_valid, Y_valid)
        train_loader = DataLoader(dataset=train_dataset, batch_size=1024, shuffle=True)
        valid_loader = DataLoader(dataset=valid_dataset, batch_size=1024, shuffle=False)

        for epoch in range(num_sub_epochs):
             for i, (states, actions) in enumerate(train_loader):
                    states = states.reshape(-1, input_size).float().to(device)
                    actions = actions.reshape(-1, num_classes).float().to(device)
                    outputs = model(states)
                    loss = criterion(outputs, actions)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    train_loss_list.append(loss.cpu().item())
        
        with torch.no_grad():
            for i, (states, actions) in enumerate(valid_loader):
                states = states.reshape(-1, input_size).float().to(device)
                actions = actions.reshape(-1, num_classes).float().to(device)
                outputs = model(states)
                loss = criterion(outputs, actions)
                valid_loss_list.append(loss.cpu().item())
                if (i+1) == 1:
                    print(outputs[0:4].data,"\n",actions[0:4].data)
                    print(" ------ ")
                
    t_loss_temp = np.round(np.array(train_loss_list).mean(),4)
    v_loss_temp = np.round(np.array(valid_loss_list).mean(),4)
    t_loss.append(t_loss_temp)
    v_loss.append(v_loss_temp)
    
    with torch.no_grad():
        for j , (test_states, test_actions) in enumerate(test_loader):
            test_states = test_states.reshape(-1,input_size).float().to(device)
            test_actions = test_actions.reshape(-1,num_classes).float().to(device)
            test_outputs = model(test_states)
            loss = criterion(test_outputs, test_actions)
            test_loss_list.append(loss.cpu().item())
    test_loss_temp = np.round(np.array(test_loss_list).mean(),4)
    test_loss.append(test_loss_temp)
    
    print(" ### epoch {} , train loss {}, valid loss {}, test loss {} ###".format(int(step+1)*num_sub_epochs*num_splits, t_loss_temp, v_loss_temp,test_loss_temp ))
    torch.save(model.state_dict(), 'model_{}_{}_{}.ckpt'.format(int(step+1)*num_sub_epochs*num_splits, v_loss_temp, test_loss_temp))
    
        
