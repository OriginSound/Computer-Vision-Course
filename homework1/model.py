import numpy as np 
import pickle as pkl 
import pdb 


class MyNetworkModel:
    def __init__(self, hidden=200, lr=1e-3, decay=0):
        self.hidden = hidden 
        self.lr = lr 
        self.decay = decay 

        # Parameters 
        self.W1 = np.random.normal(scale=np.sqrt(2/(28*28+hidden)), size=(28*28, hidden))
        self.b1 = np.zeros(hidden)

        self.W2 = np.random.normal(scale=np.sqrt(2/(hidden+10)), size=(hidden, 10))
        self.b2 = np.zeros(10)

    def forward(self, X, Y):
        """  X: (bs, 28*28)    Y: (bs, )  """

        # Forward Part: linear -> ReLU -> linear
        X = X*2 -1 
        out1 = X @ self.W1 + self.b1 

        out2 = out1.copy()
        mask = out2 < 0
        out2[mask] = 0

        out3 = out2 @ self.W2 + self.b2

        '''
        ### labels
        oneShot = np.zeros_like(out3)
        oneShot[np.arange(len(Y)), Y] = 1
        
        ### loss 
        loss = 0.5*np.sum((out3 - oneShot)**2) / len(Y) 

        grad_out3 = (out3 - oneShort) / len(Y)
        '''

        # raw output ==> probability
        out_shift = out3 - np.max(out3, axis=1, keepdims=True)
        prob = np.exp(out_shift) / np.sum(np.exp(out_shift), axis=-1, keepdims=True)
        pred = np.argmax(prob, axis=1)

        ### Loss
        loss = -np.mean(np.log(prob[np.arange(len(Y)), Y] + 1e-10))

        grad = prob.copy()
        grad[np.arange(len(Y)), Y] -= 1.0
        grad /= len(Y)
        
        ### compute gradients
        grad_out3 = grad       # (bs,10)

        grad_out2 = grad_out3 @ self.W2.T  # (bs, hidden)
        self.grad_W2 = out2.T @ grad_out3  # (hidden, 10)
        self.grad_b2 = grad_out3.sum(0)    # (10, )

        grad_out1 = grad_out2.copy()       # (bs, hidden)
        grad_out1[mask] = 0

        self.grad_W1 = X.T @ grad_out1     # (28*28, hidden)
        self.grad_b1 = grad_out1.sum(0)    # (hidden,)

        return loss, out3, pred  
    
    def predict(self, X):
        out = (2*X-1) @ self.W1 + self.b1 
        out[out < 0] = 0
        out = out @ self.W2 + self.b2
        pred = np.argmax(out, axis=1)

        return pred, out

    def set_train_config(self, lr, decay):
        self.lr=lr 
        self.decay=decay 

    def step(self):
        # SGD Optimize 
        self.W1 = self.W1 - self.lr*(self.grad_W1 + self.decay*self.W1)  
        self.W2 = self.W2 - self.lr*(self.grad_W2 + self.decay*self.W2)
        self.b1 = self.b1 - self.lr* self.grad_b1
        self.b2 = self.b2 - self.lr* self.grad_b2

    def save(self, file):
        dic = {"W1": self.W1, "W2": self.W2, 
               "b1": self.b1, "b2": self.b2}
        with open(file, "wb") as f:
            pkl.dump(dic,f)
        print("Save the model successfully!")
    
    def load(self, file):
        with open(file, "rb") as f:
            dic = pkl.load(f)
        try:
            self.W1 = dic["W1"]
            self.W2 = dic["W2"]
            self.b1 = dic["b1"]
            self.b2 = dic["b2"]
            print("Load the model successfully!")
        except:
            print("The checkpoint file is corrupted!")




