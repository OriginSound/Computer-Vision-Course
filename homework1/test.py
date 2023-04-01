import numpy as np 

def test(Loader, Model):
    # Compute the loss and accuracy
    num_sample = len(Loader)
    right = 0 
    loss = 0

    for data, label in Loader:
        size = len(data)


