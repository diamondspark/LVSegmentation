import matplotlib.pyplot as plt

import  torch.utils.data as data_utils
import torch
import torch.nn as nn
#from torchvision.models import inception
#from torchvision.models import Inception3
import numpy as np
from torch.autograd import Variable
from collections import OrderedDict
import torch.nn.functional as F
from torchvision import datasets,models,transforms
import torch.optim as optim
import copy
from os import path
import skimage.io as io
from skimage import img_as_uint
import errno 
import numpy as np
import random
import gc
import scipy.misc as misc

import torch
import numpy as np
from torch.autograd import Variable
from torch.nn import DataParallel
import sys
import os
import scipy
import cv2
from PIL import Image
import shutil
import os

#shutil.mkdir('/storage/Normal_Images/')

class model_pip(object):

    def __init__(self,model_in,data_path = '/data/gabriel/OCR/OCR_data/',
                 batch_size=90,lr = 0.001, 
                 gpu=0,f_extractor = False,scale = False,op=optim.SGD,
               criterion=nn.CrossEntropyLoss(),use_gpu=True,lr_decay_epoch=7,
                 resume=False,rand=False,model_path_continue=None,verbose=True):
        self.rand=rand
        self.epochs = 0
        self.lr_decay_epoch=lr_decay_epoch
        self.data_path = data_path
        self.model = model_in
        self.b_size = batch_size
        self.scale = scale
        self.lr = lr
        self.criterion = criterion
        self.fe = f_extractor
        self.gpu=gpu
        self.use_gpu=use_gpu
        self.model_optimizer = op(model_in.parameters(),lr = self.lr,momentum=0.9)
        self.verbose = verbose
        self.resume = resume
        self.model_path_continue=model_path_continue
        #if(self.gpu in range(0,torch.cuda.device_count())):
        #    torch.cuda.set_device(self.gpu)
        #else:
        #    torch.cuda.set_device(self.gpu[0])

        ### TODO , correct below code, this is not optimal
        self.num_output = len(os.listdir(data_path+'test/'))
        torch.manual_seed(1)
        torch.cuda.manual_seed(1)
        #torch.manual_seed(1)
        #torch.cuda.manual_seed(1)
    def transform(self,rand = False,test_only = False):
        
        if(test_only):
            
            test_sets = ['test']
        
        else:
            test_sets = ['train','test','val']

        if(test_only and self.scale):
            
            if(rand):
                data_transforms = {
                       'test':transforms.Compose([transforms.Scale(350),
                transforms.RandomCrop(300),
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
                                                 
                ])
                }
            
            else:
                #print('here')
                data_transforms = {
                'test':transforms.Compose([transforms.Scale(300),
                transforms.RandomCrop(300),

                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
                                                 
                ])
                
 
                }

        if(test_only and not(self.scale)):
            
            if(rand):
                data_transforms = {
                       'test':transforms.Compose([transforms.Scale((646,434)),
                #transforms.RandomCrop(300),
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
                                                 
                ])
                }
            

        
        if(self.scale==True and not(test_only)):
            
            
            if(rand):
                data_transforms = {'train':transforms.Compose([transforms.Scale(300),
                                            transforms.CenterCrop(300),
                                                   #transforms.RandomHorizontalFlip(),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
                       ,
                       'val':transforms.Compose([transforms.Scale(350),
                transforms.RandomCrop(300),
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]),
                       'test':transforms.Compose([transforms.Scale(350),
                transforms.RandomCrop(300),
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
                                                 
                ])
                }
            
            else:
                #print('here')
                data_transforms = {'train':transforms.Compose([transforms.Scale(300),
                                            transforms.CenterCrop(300),
                                               #transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
                       ,
                       'val':transforms.Compose([transforms.Scale(300),
                transforms.RandomCrop(300),
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
                ]),
                'test':transforms.Compose([transforms.Scale(350),
                transforms.RandomCrop(300),
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
                                                 
                ])
                
 
                }
        if(self.scale==False and not(test_only)):
            #print('here2')
            data_transforms = {'train':transforms.Compose([transforms.Scale(300),transforms.CenterCrop(300),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
                   ,
                   'val':transforms.Compose([transforms.Scale(300),
            transforms.CenterCrop(300),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
            ]),                       'test':transforms.Compose([transforms.Scale(300),
                transforms.RandomCrop(300),
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
                                                 
                ])
                
 
            }
            
            
       
        #p = path(self.data_path)
        
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
    
    #@staticmethod
    def lr_scheduler(self,epoch):
        lr = self.lr*(0.1**(epoch//self.lr_decay_epoch))
        if epoch%self.lr_decay_epoch ==0:
            print('lr = ',lr)
        for param_group in self.model_optimizer.param_groups:
            param_group['lr'] = lr
        #return optimz
    
    def load_model(self,filename):
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epochs']
        
        
        # create new OrderedDict that does not contain `module.`
        if(len(self.gpu)>1):
            new_state_dict = OrderedDict()
            for k, v in checkpoint['state_dict'].items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            # load params
            self.model.load_state_dict(new_state_dict)
        
        if(self.gpu in range(0,torch.cuda.device_count())):
            
            self.model.load_state_dict(checkpoint['state_dict'])
            
        if(self.resume):
            self.model_optimizer.load_state_dict(checkpoint['optimizer'])
        #return model,optimz,start_epoch
    
    def train_model(self, epochs=30,n=None):
        dsets,dset_loaders,dset_sizes = self.transform() 
        print(dset_sizes)
        
        model = self.model
        #op = self.model_optimizer
        epoch_init = 0
        
        
        if(self.resume):
            
            try:
                self.load_model(filename=self.model_path_continue)
            except:
                print('invalid directory, starting from scratch')
        
        
        best_model=model
        
        criterion = self.criterion
        
        
        if(torch.cuda.is_available() and self.use_gpu and self.gpu in range(0,torch.cuda.device_count())):
            #torch.cuda.set_device(self.gpu)
            model=model.cuda()
            #criterion=criterion.cuda()
        
        elif(torch.cuda.is_available() and self.use_gpu and  len(self.gpu)>1):
            
            model = DataParallel(model,device_ids = self.gpu).cuda()
            #print('here')
            #criterion = DataParallel(model,device_ids=self.gpu)
        
        best_acc = 0.0
        best_epoch = 0
        
        
        if(~self.fe):
        
            for epoch in range(epoch_init,epochs):
                #print('Epoch = ',epoch)
                
                for phase in ['train','val']:
                    if(phase == 'train'):
                        model.train(True)

                        self.lr_scheduler(epoch)
                    else:
                        model.train(False)
                    c_mat = np.zeros((self.num_output,self.num_output)) 
                    running_loss = 0.0
                    running_corrects = 0.0
                    running_tp = 0.0
                    for data in dset_loaders[phase]:
                        inputs,labels = data
                        #print(inputs.size())
                        if(torch.cuda.is_available() and self.use_gpu):
                            inputs,labels = Variable(inputs.cuda(async=True)),Variable(labels.cuda(async=True))
                        else:
                            inputs,labels = Variable(inputs),Variable(labels)
                        
                        self.model_optimizer.zero_grad()
                        flag=0
                        
                        if(inputs.size(0)<self.b_size and  n == 'Inception'):
                            flag=1
                            
                            if(torch.cuda.is_available() and self.use_gpu):   
                                temp = Variable(torch.zeros((self.b_size,3,300,300)).cuda(async=True))
                                temp2 = Variable(torch.LongTensor(self.b_size).cuda(async=True))
                                             
                            else:
                                temp=Variable(torch.zeros((self.b_size,3,300,300)))
                                temp2=Variable(torch.LongTensor(self.b_size))
                                             
                            temp[0:inputs.size(0)]=inputs
                            inputs = temp
                            del(temp)
                            temp2[0:labels.size(0)] = labels
                            temp2[labels.size(0):] = 0
                            labels = temp2
                            del(temp2)
                        #print(epoch)
                        outputs = model(inputs)
                        #print(outputs.size())
                        #print(labels.size())
                        if(n=='Inception'):
                            if phase=='val':
                                #print('val')
                                _,preds = torch.max(outputs.data,1)
                                loss = criterion(outputs,labels)
                            else:
                                _,preds = torch.max(outputs[0].data,1)
                                loss = criterion(outputs[0],labels)

                        else:
                            _,preds = torch.max(outputs.data,1)
                            loss = criterion(outputs,labels)



                        if phase=='train':
                            loss.backward()
                            self.model_optimizer.step()
                        
                        running_loss+=loss.data[0]
                        running_corrects += torch.sum(preds == labels.data)
                        for i in range(0,labels.data.cpu().numpy().shape[0]):

                            c_mat[labels.data.cpu().numpy()[i],preds.cpu().numpy()[i]]+=1

                        self.epochs = epoch
                        del(inputs)
                        del(labels)
                        del(outputs)
                    epoch_loss = running_loss/dset_sizes[phase]
                    epoch_acc = running_corrects/dset_sizes[phase]
                    #epoch_tpr = running_tp/dset_sizes[phase]
                    print(phase + '{} Loss: {:.10f} \nAcc: {:.4f}'.format(phase,epoch_loss,epoch_acc))
                    #print(c_mat)
                    if(self.verbose):
                    
                        print(c_mat)
                    if phase=='val' and epoch_acc>best_acc:
                        best_acc=epoch_acc
                        best_model=copy.deepcopy(model)
                        best_epoch=epoch
                    #del(inp)
                    #del(label)
                #print()
        
        print(best_acc)
        print(best_epoch)
            
        self.model=best_model.cpu()
        
    def store_model(self,f_name):
        
        
        state={'epochs':self.epochs,'state_dict':self.model.state_dict(),
                          'optimizer':self.model_optimizer.state_dict()}
        torch.save(state,f_name)
    
    def test(self,model_dir,model_name,random_crop,test_on,n='None',save_miscl = False,folder = 'misc'):
        #TODO modify class to add a test path
        model = self.model
        model.eval()
        epoch_acc = 0
        try:
            shutil.rmtree(model_dir+'/'+folder+'/'+model_name[:model_name.find('.pth.tar')])
            os.makedirs(model_dir+'/'+folder+'/'+model_name[:model_name.find('.pth.tar')])
        except:
            os.makedirs(model_dir+'/'+folder+'/'+model_name[:model_name.find('.pth.tar')])
            
        dsets,dset_loaders,dset_sizes = self.transform(rand=random_crop,test_only=test_on)
        flag=False
        model_ind = model_name[:model_dir.find('.pth.tar')]
        multi_gpu = False
        
        if(torch.cuda.is_available() and self.use_gpu and (self.gpu in range(0,torch.cuda.device_count())) ):
            #print('here')
            #torch.cuda.set_device(self.gpu)
            model=model.cuda()
            
            #print('there')
            #criterion=criterion.cuda()
        
        elif(torch.cuda.is_available() and self.use_gpu and  len(self.gpu)>1):
            multi_gpu = True
            #model = model.cuda(self.gpu[0])
            model = DataParallel(model,device_ids = self.gpu).cuda()
            #criterion = DataParallel(model,device_ids=self.gpu)
        
        running_corrects = 0  
        c_mat = np.zeros((self.num_output,self.num_output))
        #print(c_mat.shape)
        for data in dset_loaders['test']:
            inp_img,labels = data
             
            if(torch.cuda.is_available()):
                inp_img,labels=inp_img.cuda(async=True),labels.cuda(async=True)
            
            inp,labels=Variable(inp_img),Variable(labels)
            
            if(inp.size(0)<self.b_size and  n == 'Inception'):
                #flag=1
                            
                if(flag):   
                    temp = Variable(torch.zeros((self.b_size,3,300,300)).cuda(async=True))
                    temp2 = Variable(torch.LongTensor(self.b_size).cuda(async=True))
                                             
                else:
                    temp=Variable(torch.zeros((self.b_size,3,300,300)))
                    temp2=Variable(torch.LongTensor(self.b_size))
                                             
                temp[0:inputs.size(0)]=inp
                inp = temp
                            
                temp2[0:labels.size(0)] = labels
                temp2[labels.size(0):] = 0
                labels = temp2
                        

            
            output = model(inp)
            if(n=='Inception'):
                

                _,preds = torch.max(output.data,1)
                
            else:
                
                _,preds = torch.max(output.data,1)
                #print(preds.size())
                #print(labels.data.size())
                #print(output.data.size())
                #print(labels.data)
                #print(preds)
                #print(labels)
            #preds = preds.cpu()
            running_corrects += torch.sum(preds.cpu() == labels.cpu().data)
            
            ### save misclassified ones
            #print(preds.cpu().numpy().shape)
            #print(preds.cpu().numpy().reshape(self.b_size,))
            #print(labels.data.cpu().numpy().reshape(self.b_size,))
            #print(preds.cpu().numpy().reshape(self.b_size,) == labels.data.cpu().numpy().reshape(self.b_size,))
            
            pred_np = preds.cpu().numpy()
            
            label_np = labels.cpu().data.numpy().reshape(pred_np.shape[0],1)
            #print(pred_np.shape)
            #print(label_np.shape)
            misc_arr = pred_np==label_np
            #print(misc)
            #print(misc.shape)
            misc_ind = [i for i in range(0,pred_np.shape[0]) if misc_arr[i] == False]
            #print(misc_ind)
            #print(misc_ind[0])
            #misc_ind = np.where(np.array((preds.cpu().numpy().reshape(self.b_size,)==labels.data.cpu().numpy().reshape(self.b_size,)))==False)[0]
            #print(misc_ind)
            #print(preds.cpu().numpy()==labels.data.cpu().numpy())
            #return
            for i in misc_ind:
                #print(preds[i])
                #print(labels.data[i])
                misc_img = inp_img.cpu()[i].numpy().transpose(1,2,0)*np.array([0.229,0.224,0.225]) +np.array([0.485,0.456,0.406])
                count_file = len(os.listdir(model_dir+'/'+folder+'/'+model_name[:model_name.find('.pth.tar')]))
                #plt.imshow(misc_img),plt.show()
                misc.imsave(model_dir+'/'+folder+'/'+model_name[:model_name.find('.pth.tar')]+'/'+str(count_file+1)+'_'+str(label_np[i])+'_as_'+str(pred_np[i])+'.jpg',misc_img)
            for i in range(0,labels.data.cpu().numpy().shape[0]):

                c_mat[labels.data.cpu().numpy()[i],preds.cpu().numpy()[i]]+=1
            #epoch_acc += running_corrects/dset_sizes['test']
            del(inp_img)
            del(inp)
            del(labels)
        t_acc = np.trace(c_mat)/np.sum(c_mat)
        if(self.verbose):
            print('test accuracy= ',t_acc)
            print(c_mat)
       
            
        np.savetxt(model_dir+'/'+folder+'/'+model_name[:model_name.find('.pth.tar')]+'/'+model_name+'c_mat.txt',c_mat.astype(int))
        np.savetxt(model_dir+'/'+folder+'/'+model_name[:model_name.find('.pth.tar')]+'/'+model_name+'accuracy.txt',np.array(t_acc).reshape(1,))
            
            
