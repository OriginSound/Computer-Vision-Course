from model import * 
from dataset import Dataset, DataLoader
from utils import test 

import pdb 
import numpy as np 

# Dataset 
TrainDataloader = DataLoader(Dataset(train=True), batch_size=100)
TestDataLoader = DataLoader(Dataset(train=False), batch_size=100)

# model and hyper-parameters
epochs = 20
lr, decay = 2e-3, 0.05
gamma = 0.9
simpleModel = MyNetworkModel(hidden=200, lr=lr, decay=decay)


for epoch in range(epochs):
    for data, label in TrainDataloader:
        # Training Part
        # 1. forward : compute the loss and gradient
        # 2. step    : update the parameters using SGD 
        # 3. set_train_config: adjust lr  
        simpleModel.forward(data, label)
        simpleModel.step()

    lr, decay = gamma*lr, gamma*decay
    simpleModel.set_train_config(lr=lr, decay=decay)
    
    # print testing info
    loss, accuracy = test(TestDataLoader, simpleModel)
    print(f"Epoch: {epoch+1:2d}  Loss:{loss:1.4f}  Accuracy: {100*accuracy:3.2f}%")

simpleModel.save(f"checkpoint_{epochs}.pkl")




