import torch
import torch.nn as nn

import torch.optim as optim
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torchvision import datasets,transforms
import scipy.misc
# n_in=121,n_h=100,n_out=121
import random
import numpy as np
import simplejson
import torchvision
from utils import get_series,find_stats,get_patches
### Train mean = 0.122231430457 
### Train sd = 0.162071986271
class SAE(nn.Module):
    def __init__(self,n_in=11,n_h=100,n_out=11,img_path='/data/gabriel/LVseg/dataset_img/img_256',
                 label_path = 0,b_size = 1000, patch_size = 0):
        
        ### patch_size to rescale the image input to auto encoder.
        
        super(SAE,self).__init__()
        
        self.patch_size=patch_size
        self.fc1 = nn.Linear(n_in*n_in,n_h)
        self.fc2 = nn.Linear(n_h,n_out*n_out)
        self.img_path = img_path
        self.label_path=label_path
        #self.f = nn.Sigmoid()
        self.b_size=b_size
        self.n_in = n_in
        self.n_h = n_h
        self.n_out = n_out
    
    
    

    def transform(self,rand = False,patch_path = './',mean=0,sd = 1):
        
        
        
        train_series = get_series(self.img_path,0)
        
        dim = plt.imread(self.img_path+'/'+train_series[0]+'.png').shape
        
        label_train = torch.zeros(len(train_series),dim[0],dim[1])
        img_train = torch.zeros(len(train_series),1,dim[0],dim[1])
        
        
        count = -1
        
        mean,sd = find_stats(self.img_path)
        
        for i in train_series:
            count+=1
            temp_im = plt.imread(self.img_path+'/'+i+'.png')
            temp_im.reshape(1,dim[0],dim[1])
            
            img_train[count,:,:,:] = transforms.ToTensor()(temp_im)
            
            img_train[count,:,:,:].sub(mean).div(sd)
            
            if not(self.label_path ==0):
                label_train[count,:,:] = transforms.ToTensor()(plt.imread(self.label_path+'/'+i+'.png'))
        
        if(patch_path=='./'):
            patch_path = self.img_path
        
        for i,j,k in os.walk(self.img_path):
            if '.ipynb_checkpoints' in i:
                print(i)
                shutil.rmtree(i)
        
        dsets = {'Training':torch.utils.data.TensorDataset(img_train,label_train)}
        
        #dsets = {'Training':datasets.ImageFolder(os.path.join(patch_path,'Training'),data_transforms['Training'])}

        dset_loaders ={'Training': torch.utils.data.DataLoader(dsets['Training'],batch_size=self.b_size,shuffle=False,num_workers=4)
              }

        dset_sizes = {'Training':len(dsets['Training'])}
        ### TODO maybe use os.listdir to get the made folders ?? or make the folders on the fly based on test_only input
        return dsets,dset_loaders,dset_sizes
        
    def forward(self,x):
        x = x.view(-1,self.n_in*self.n_in)
        #print(x.size())
        x = self.fc1(x.view(-1,self.n_in*self.n_in))
        
        x = torch.nn.functional.sigmoid(x)
        
        x = self.fc2(x)
        
        x = torch.nn.functional.sigmoid(x)
        
        return x    

    
def train_sae(model,epochs):
    torch.cuda.set_device(0)

    optimizer = optim.SGD(model.parameters(),lr = 0.001)
    #fig = plt.figure()
    model.train()
    #ax = fig.add_subplot(1,1,1) 
    
    data_set,dataset_loader,data_size = model.transform()
    #print(dataset_loader['Training'].next())
    #print(dataset_loader)
    #print(data_set['Training'])
    model = model.cuda()
    for epoch in range(0,epochs):

        for i in dataset_loader['Training']:
            #print(i)
            inp1,_ = i
            #print(inp1.size())
            
            optimizer.zero_grad()
            inp1 = Variable(inp1.cuda())
            #print(inp1.size())
            
            out = model(inp1)
            #print(out.size())
            
            loss = torch.norm(out - inp1.view(-1,model.n_in*model.n_in))**2
            #print(loss)
            loss.backward()
            optimizer.step()
            del(loss)
            
            del(inp1)
    model=model.cpu()     
    for p in model.modules():
        if isinstance(p,nn.Linear):
            print(p)
            return p.weight.data,p.bias.data

        
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
        

        
    
    
    
    
        
        
        
        
        
        