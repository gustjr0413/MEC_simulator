import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
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
    def __init__(self):
        self.len = x_train.shape[0]
        self.x_data = torch.from_numpy(x_train)
        self.y_data = torch.from_numpy(y_train)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len
    
class TestDataset(Dataset):
    def __init__(self):
        self.len = x_test.shape[0]
        self.x_data = torch.from_numpy(x_test)
        self.y_data = torch.from_numpy(y_test)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len  
    

xy = pd.get_dummies(pd.read_csv('final_dataset_newparam300_50000.csv')).values[:,1:]
print("Dataset shape = {}".format(xy.shape))
x_data = xy[:,0:6]
y_data = xy[:,6:8]
print("    x_data shape = {}".format(x_data.shape))
print("    y_data shape = {}".format(y_data.shape))
x_train = x_data[0:int(xy.shape[0]*0.95)]
y_train = y_data[0:int(xy.shape[0]*0.95)]
x_test = x_data[int(xy.shape[0]*0.95):]
y_test = y_data[int(xy.shape[0]*0.95):]
print("Data separation")
print("    The number of training data = {}".format(x_train.shape[0]))
print("    The number of test data     = {}".format(x_test.shape[0]))
print("Data type = {}".format(x_data[0].dtype))


train_dataset = TrainDataset()
test_dataset = TestDataset()
train_loader = DataLoader(dataset=train_dataset, batch_size=256,shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=256,shuffle=False)




"""parameters"""
input_size = 6
hidden_size = 128
hidden_size2 = 64
hidden_size3 = 32
hidden_size4 = 16
num_classes = 2
num_epochs = 10000
prints_step = 100

model = NeuralNet(input_size, hidden_size, hidden_size2, hidden_size3, hidden_size4, num_classes).to(device)
criterion = nn.L1Loss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

loss_list =[]
# Train the model
total_step = len(train_loader)
for epoch in tqdm(range(num_epochs)):
    for i, (states, actions) in enumerate(train_loader):  
        # Move tensors to the configured device
        states = states.reshape(-1, input_size).float().to(device)
        actions = actions.reshape(-1, num_classes).float().to(device)
        # Forward pass
        outputs = model(states)
        loss = criterion(outputs, actions)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        

    print("################### Epoch {} done ###################".format(epoch+1))
    
    # Test the model
    # In test phase, we don't need to compute gradients (for memory efficiency)
    with torch.no_grad():
        loss_total = 0
        model_eval = NeuralNet(input_size, hidden_size, hidden_size2, hidden_size3, hidden_size4, num_classes).to(device)
        model_eval.load_state_dict(model.state_dict())
        model_eval.eval()
        for i, (states, actions) in enumerate(test_loader):
            states = states.reshape(-1, input_size).float().to(device)
            actions = actions.reshape(-1, num_classes).float().to(device)
            outputs = model_eval(states)
            loss = criterion(outputs, actions)
            loss_total += loss
            if (i+1)==1:
                print(outputs[0:20].data,"\n",actions[0:20].data)

        print('epoch = {}, total loss = {}, avg loss = {}'.format(epoch, loss_total, loss_total/256))

#     Save the model checkpoint
    loss_list.append(loss_total) 
    torch.save(model.state_dict(), 'model__{}_{}.ckpt'.format(epoch, np.round(loss_total.cpu().item()/256,4)))
