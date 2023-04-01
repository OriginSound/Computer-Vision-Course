from model import * 
from dataset import Dataset, DataLoader

import pdb 
import numpy as np 

# Dataset 
TrainDataloader = DataLoader(Dataset(train=True), batch_size=100)
TestDataLoader = DataLoader(Dataset(train=False), batch_size=1)

# model
epochs = 20
simpleModel = MyNetworkModel()
simpleModel.set_train_config(lr=1e-3, decay=0)

for epoch in range(epochs):
    for data, label in TrainDataloader:
        
        ### For simple Model
        loss = simpleModel.train(data, label)
        simpleModel.step()
        # print(simpleModel.grad_W1[1, 2])
        # simpleModel.W1[1,2] += 1e-6
        # loss_new = simpleModel.train(data, label)
        # print((loss_new-loss)/1e-6)

    print(loss)




