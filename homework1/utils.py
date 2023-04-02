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
    

def plot(data):
    x = list(range(len(data["test_loss"])))

    fig, axs = plt.subplots(nrows=1, ncols=2)
    axs[0].plot(x, data["test_loss"], "o-", linewidth=2.0)
    axs[0].plot(x, data["train_loss"], "o-", linewidth=2.0)
    axs[0].set_title("Loss")
    axs[0].set_xlabel("epoch")
    axs[0].set_ylabel("loss")
    axs[0].legend(labels=["Test set", "Train set"])

    axs[1].plot(x, data["test_accuracy"], "o-", linewidth=2.0)
    axs[1].plot(x, data["train_accuracy"], "o-", linewidth=2.0)
    axs[1].set_title("Accuracy")
    axs[1].set_xlabel("epoch")
    axs[1].set_ylabel("accuracy")
    axs[1].legend(labels=["Test set", "Train set"])
    fig.savefig("plot.png")
    fig.show()

def visualization(Model):
    W1 = Model.W1 
    W2 = Model.W2 
    b1 = Model.b1 
    b2 = Model.b2 






