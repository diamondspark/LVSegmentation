import torch
import torch.nn as nn

import torch.optim as optim
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torchvision import datasets,transforms
import scipy.misc
# n_in=121,n_h=100,n_out=121
import random
import numpy 
class SAE(nn.Module):
    def __init__(self,n_in=11,n_h=100,n_out=11,data_path='./',b_size = 1000, patch_size = 100):
        super(SAE,self).__init__()
        
        
        self.fc1 = nn.Linear(n_in*n_in,n_h)
        self.fc2 = nn.Linear(n_h,n_out*n_out)
        self.data_path = data_path
        #self.f = nn.Sigmoid()
        self.b_size=b_size
        self.n_in = n_in
        self.n_h = n_h
        self.n_out = n_out
        
    def get_patches(self,save_path,count = 10**4):
        count1 = 0
        
        raise NotImplementedError 
        
        for i in os.listdir(self.data_path+'/Training'):
            for j in os.listdir(self.data_path+'/Training/'+i):
                print
                
                
        im_list = [plt.imread(self.data_path+'/Training/'+i) for i in os.listdir(self.data_path+'/Training') if not(i=='.') 
                  and not(i=='.DS_Store')]
        random.shuffle(im_list)
        
        for i in os.listdir(im_list[:count]):
            
            R = random.randint(11,i.shape[0])
            C = random.randint(11,i.shape[1])
            patch = i[R-11:R,C-11:C]
            scipy.io.imsave(save_path+'/'+count1+'.png',patch)
            count1+=1
        
    def transform(self,rand = False,test_only = False):
        
        test_sets = ['Training','Validation']
        data_transforms = {'Training':transforms.Compose([
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
               ,
               'Validation':transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]),

        }

        
        for i,j,k in os.walk(self.data_path):
            if '.ipynb_checkpoints' in i:
                print(i)
                shutil.rmtree(i)

                
        #return
        dsets = {x:datasets.ImageFolder(os.path.join(self.data_path,x),data_transforms[x])
            for x in test_sets}

        dset_loaders ={x: torch.utils.data.DataLoader(dsets[x],batch_size=self.b_size,shuffle=True,num_workers=4)
              for x in test_sets}

        dset_sizes = {x:len(dsets[x]) for x in test_sets}
        ### TODO maybe use os.listdir to get the made folders ?? or make the folders on the fly based on test_only input
        return dsets,dset_loaders,dset_sizes

        
    


    def encode(self,x):
        
        return nn.Sigmoid(self.fc1(x))
        #return (self.fc1(x))
        
    def decode(self,x):
        #return nn.Sigmoid(self.fc2(x))
        return (self.fc2(x))
        
    def forward(self,x):
        
        #if(len(x.numpy().shape)>1):
        
        x = self.encode(x.view(-1,self.n_in*self.n_in))
        
        #x = self.decode(x)
        
        return self.decode(x)
    

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
        inp1 = inp
        optimizer.zero_grad()

        inp1 = Variable(inp1.cuda())
        model = model.cuda()
        out = model(inp1)
        loss = torch.norm(out - inp1)**2
        print(loss)
        loss.backward()
        optimizer.step()        


class localnet(nn.Module):
    def __init__(self):
        super(localnet,self).__init__()
        
        self.conv = nn.Conv2d(1,100,11)
        self.avg_pool = nn.AvgPool2d(6,6)
        #self.relu = nn.functional.relu()
        self.classifier = nn.Linear(8100,1024)
        
    def forward(self,x):
        
        x = self.conv(x)
        x = nn.functional.relu(x)
        x = self.avg_pool(x)
        
        x = x.View(-1,8100)
        
        x = self.classifier(x)
        x = nn.functional.relu(x)
        return nn.Softmax(x)

import os
import shutil

        
        #ax.imshow(out.cpu().data.numpy().reshape(121,121))
        
        #ax.close()
        
        #del(out)
        

        
    
    
    
    
        
        
        
        
        
        