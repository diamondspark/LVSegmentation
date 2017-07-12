import torch
import torch.nn as nn

import torch.optim as optim
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torchvision import datasets,transforms
import scipy.misc
# n_in=121,n_h=100,n_out=121

import numpy 
class SAE(nn.Module):
    def __init__(self,n_in,n_h,n_out):
        super(SAE,self).__init__()
        
        
        self.fc1 = nn.Linear(n_in*n_in,n_h)
        self.fc2 = nn.Linear(n_h,n_out*n_out)
        
        self.f = nn.Sigmoid()
        
        self.n_in = n_in
        self.n_h = n_h
        self.n_out = n_out
        
    def encode(self,x):
        
        return self.f(self.fc1(x))
        #return (self.fc1(x))
        
    def decode(self,x):
        #return self.f(self.fc2(x))
        return (self.fc2(x))
        
    def forward(self,x):
        
        #if(len(x.numpy().shape)>1):
        
        x = self.encode(x.view(-1,self.n_in*self.n_in))
        
        #x = self.decode(x)
        
        return self.decode(x)
    

### TODO - add L2 regularzation and KL divergence
#def loss_function(model,inp,target):



import os
import shutil
def train(model,inp,epochs):
    torch.cuda.set_device(0)
    try:
        shutil.rmtree('../../saved/')
        os.makedirs('../../saved/')
    except:
        os.makedirs('../../saved/')
        
        
    optimizer = optim.SGD(model.parameters(),lr = 0.01)
    fig = plt.figure()
    model.train()
    ax = fig.add_subplot(1,1,1)  
    for epochs in range(0,epochs):
        #inp1 = torch.Tensor(inp)
        #fig = plt.figure()
        inp1 = inp
        optimizer.zero_grad()
        
        
        
        inp1 = Variable(inp1.cuda())
        model = model.cuda()
        out = model(inp1)
        loss = torch.norm(out - inp1)**2
        print(loss)
        loss.backward()
        optimizer.step()
        
        plt.imshow(out.cpu().data.numpy().reshape(121,121),cmap='gray'),plt.show()
        scipy.misc.imsave('../../saved/'+str(epochs)+'_saved.png',out.cpu().data.numpy().reshape(121,121))
        #ax.imshow(out.cpu().data.numpy().reshape(121,121))
        
        #ax.close()
        
        #del(out)
        
    
    
    
    
        
        
        
        
        
        