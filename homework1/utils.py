import numpy as np 
import matplotlib as mpl
import matplotlib.pyplot as plt 
from matplotlib.colors import ListedColormap


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


top = mpl.colormaps['Oranges_r'].resampled(128)
bottom = mpl.colormaps['Blues'].resampled(128)
newcolors = np.vstack((top(np.linspace(0, 1, 128)),
                       bottom(np.linspace(0, 1, 128))))
newcmp = ListedColormap(newcolors, name='OrangeBlue')

def visualization(Model, cmap=newcmp):
    W1 = Model.W1.T
    W2 = Model.W2.T
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(18, 10))
    # fig.tight_layout()

    psm = axs[0].pcolormesh(W1, cmap=cmap)
    fig.colorbar(psm, ax=axs[0])
    axs[0].set_xlabel("size of input")
    axs[0].set_ylabel("size of hidden")
    axs[0].set_title("First Layer")

    psm = axs[1].pcolormesh(W2, cmap=cmap)
    fig.colorbar(psm, ax=axs[1])
    axs[1].set_xlabel("size of hidden")
    axs[1].set_ylabel("size of output")
    axs[1].set_title("Second Layer")
    
    fig.subplots_adjust(left=0.1, right=0.95, hspace=0.5)
    fig.savefig("visualization.png")
    fig.show()








