from model import * 
from dataset import Dataset, DataLoader
from utils import test, plot

import pdb 
import numpy as np 
import pickle as pkl 

# Dataset 
TrainDataloader = DataLoader(Dataset(train=True), batch_size=100)
TestDataLoader = DataLoader(Dataset(train=False), batch_size=100)

# model and hyper-parameters
epochs = 20
lr, decay = 2e-3, 0.01
gamma = 0.95
simpleModel = MyNetworkModel(hidden=200, lr=lr, decay=decay)

# Records 
train_loss = []
test_loss = []
train_accuracy = []
test_accuracy = []


for epoch in range(epochs):
    for data, label in TrainDataloader:
        # Training Part
        # 1. forward : compute the loss and gradient
        # 2. step    : update the parameters using SGD 
        # 3. set_train_config: adjust lr  
        simpleModel.forward(data, label)
        simpleModel.step()

    lr = gamma*lr
    simpleModel.set_train_config(lr=lr, decay=decay)
    
    # record & print testing info
    loss, accuracy = test(TrainDataloader, simpleModel)
    train_loss.append(loss)
    train_accuracy.append(accuracy)

    loss, accuracy = test(TestDataLoader, simpleModel)
    test_loss.append(loss)
    test_accuracy.append(accuracy)
    print(f"Epoch: {epoch+1:2d}  Loss:{loss:1.4f}  Accuracy: {100*accuracy:3.2f}%")


# Save the model and info
simpleModel.save(f"checkpoint_{epochs}.pkl")

dic = {"train_loss":train_loss, "train_accuracy":train_accuracy,
        "test_loss": test_loss,  "test_accuracy": test_accuracy}
with open("loss_accuracy.pkl", "wb") as f:
    pkl.dump(dic,f)

plot(dic)



