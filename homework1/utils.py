import numpy as np 
import matplotlib.pyplot as plt 


def test(Loader, Model):
    # Compute the loss and accuracy
    num_sample = len(Loader)
    correct = 0 
    loss_total = 0

    for data, label in Loader:
        size = len(data)
        loss, _, pred = Model.forward(data, label)

        loss_total += loss * size 
        correct += np.sum(pred == label)
    
    return loss_total / num_sample, correct / num_sample
    

def visualization(Model):
    W1 = Model.W1 
    W2 = Model.W2 
    b1 = Model.b1 
    b2 = Model.b2 






