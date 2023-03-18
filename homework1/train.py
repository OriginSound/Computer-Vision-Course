import abc 
import numpy as np 
import pdb 

'========================   Model blocks   ========================'
class myModule(metaclass=abc.ABCMeta):
    def __call__(self, x):  pass 

    def forward(self, x):   pass 

    def backward(self, x):  pass 

class myReLU(myModule):
    def __init__(self):
        self.grad = None 

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        mask = x < 0
        x[mask] = 0
        return x 

    def backward(self):
        pass 

class myLinear(myModule):
    def __init__(self, in_channels, out_channels):
        self.in_channels = in_channels
        self.out_channels = out_channels
        std = np.sqrt(2/(in_channels+out_channels))

        # Parameters
        self.weight = np.random.normal(scale=std, size=(in_channels, out_channels))
        self.bias = np.zeros(out_channels)

        # gradients of Parameters
        self.weight_grad = np.zeros_like(self.weight)
        self.bias_grad = np.zeros_like(self.bias)


    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        x_shape = x.shape
        x = x.reshape(-1, self.in_channels)
        x = x @ self.weight + self.bias 
        x = x.reshape(x_shape[:-1]+(self.out_channels,))
        return x 

    def backward(self, x):
        pass 


'========================     Loss  part   ========================'
class CrossEntropyLoss(myModule):
    def __init__(self):
        pass 

    def __call__(self, *x):
        return self.forward(*x)

    def forward(self, pred, label):
        # (bs, 10)
        bs = pred.shape[0]
        pdb.set_trace()
        pred_exp = np.exp(pred)
        norm = pred_exp.sum(-1, keepdims=True)
        prob = pred_exp / norm 
        log_prob = np.log(prob+1e-6) 
        
        return -np.sum(log_prob[np.arange(bs), label])
    
    def backward(self, x):
        
        pass 

    
'========================      Network     ========================'
class myNetwork(myModule):
    def __init__(self, channels=[28*28, 200, 10]):
        a, b, c = channels
        self.layer1 = myLinear(a, b)
        self.layer2 = myReLU()
        self.layer3 = myLinear(b, c)

    @property
    def parameters(self):
        return [self.layer1.weight, self.layer1.bias,
                self.layer3.weight, self.layer3.bias]
    
    @property
    def parameters(self):
        return [self.layer1.weight_grad, self.layer1.bias_grad,
                self.layer3.weight_grad, self.layer3.bias_grad]

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        return  self.layer3(self.layer2(self.layer1(x)))

    def backward(self, x):

        pass 



class SGD:
    def __init__(self, model, lr=1e-3, decay=0):
        pass

    def zero_grad():
        pass 

    def step():
        pass 


if __name__ == "__main__":
    inputs = np.random.rand(1, 28*28)
    Model = myNetwork()
    outputs = Model(inputs)

    label = np.zeros(10)
    label[3] = 1
    label = label.astype(np.int)
    pdb.set_trace()
    Loss = CrossEntropyLoss()
    loss = Loss(outputs, label.reshape(1,10))
    pdb.set_trace()
 
