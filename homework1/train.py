from model import * 
from dataset import Dataset, DataLoader
from utils import test, plot
import argparse 

import pdb 
import numpy as np 
import pickle as pkl 

# Args 
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--batch_size", type=int, default=100)
parser.add_argument("--lr", type=float, default=2e-3)
parser.add_argument("--gamma", type=float, default=0.95)
parser.add_argument("--decay", type=float, default=1e-2)
parser.add_argument("--hidden", type=int, default=200)
args = parser.parse_args()
print(args)

# Dataset 
TrainDataloader = DataLoader(Dataset(train=True), batch_size=args.batch_size)
TestDataLoader = DataLoader(Dataset(train=False), batch_size=args.batch_size)

# model and hyper-parameters
epochs = args.epochs
lr, decay = args.lr, args.decay
gamma = args.gamma
simpleModel = MyNetworkModel(hidden=args.hidden, lr=lr, decay=decay)

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
        simpleModel.forward(data, label)
        simpleModel.step()

    # 3. set_train_config: adjust lr  
    lr = gamma*lr
    simpleModel.set_train_config(lr=lr, decay=decay)
    
    # 4. record & print testing info
    loss, accuracy = test(TrainDataloader, simpleModel)
    train_loss.append(loss)
    train_accuracy.append(accuracy)
    print(f"[Train Set] Epoch: {epoch+1:2d}  Loss:{loss:1.4f}  Accuracy: {100*accuracy:3.2f}%")

    loss, accuracy = test(TestDataLoader, simpleModel)
    test_loss.append(loss)
    test_accuracy.append(accuracy)
    print(f"[Test Set] Epoch: {epoch+1:2d}  Loss:{loss:1.4f}  Accuracy: {100*accuracy:3.2f}%")


# Save the model and info
simpleModel.save(f"checkpoint_{epochs}.pkl")

dic = {"train_loss":train_loss, "train_accuracy":train_accuracy,
        "test_loss": test_loss,  "test_accuracy": test_accuracy}
with open("loss_accuracy.pkl", "wb") as f:
    pkl.dump(dic,f)

plot(dic)



