from model import MyNetworkModel
from dataset import Dataset, DataLoader
from utils import test

import os 
checkpoints = [file for file in os.listdir() if file.startswith("checkpoint")]
checkpoints.sort(key=lambda x: x[11:-4])


TestDataLoader = DataLoader(Dataset(train=False), batch_size=200)

simpleModel = MyNetworkModel(hidden=200)
simpleModel.load(checkpoints[-1])

loss, accuracy = test(TestDataLoader, simpleModel)
print(f"[Test Set] Loss:{loss:1.4f}  Accuracy: {100*accuracy:3.2f}%")

