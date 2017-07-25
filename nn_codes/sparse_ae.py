import torch
import torch.nn as nn
import os 
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torchvision import datasets,transforms
import scipy.misc
# n_in=121,n_h=100,n_out=121
import random
import numpy as np
import simplejson
import shutil
import torchvision
import pickle
from utils import find_stats,get_patches,split_im,get_series
### Train mean = 0.122231430457 
### Train sd = 0.162071986271
class SAE(nn.Module):
    def __init__(self,n_in=11,n_h=100,n_out=11,img_path='/data/gabriel/LVseg/dataset_img/img_256',
                 label_path = 0,b_size = 1000, patch_size = 0,lr=0.01,rho=0.1,gpu=1,lam = 10**-4,beta = 3,test_fraction=0):
        
        ### patch_size to rescale the image input to auto encoder.
        
        super(SAE,self).__init__()
        self.lam=lam
        self.lr = lr
        self.beta = beta
        self.gpu = gpu
        self.rho = rho
        self.patch_size=patch_size
        self.test_fraction=test_fraction
        self.fc1 = nn.Linear(n_in*n_in,n_h)
        self.fc2 = nn.Linear(n_h,n_out*n_out)
        self.img_path = img_path
        self.label_path=label_path
        #self.f = nn.Sigmoid()
        self.b_size=b_size
        self.n_in = n_in
        self.n_h = n_h
        self.n_out = n_out
    
    
    
    def out_at_layer(self,inp,L):

        count= -1
        for i in self.modules():
            count+=1
            try:
                if(count==L):
                    #print(tt)
                    return(torch.nn.functional.sigmoid(i(inp)))

            except:
                continue

    def transform(self,rand = False,patch_path = './',mean=0,sd = 1):
        
        
        if(self.test_fraction==0):
            
            train_series = get_series(self.img_path,0)
        else:
            train_series,test_series = get_series(self.img_path,self.test_fraction)
 
        dim = plt.imread(self.img_path+'/'+train_series[0]+'.png').shape
        
        label_train = torch.zeros(len(train_series),self.n_out*self.n_out)
        img_train = torch.zeros(len(train_series),1,self.n_in,self.n_in)
        
        
        count = -1
        
        mean,sd = find_stats(self.img_path)
        
        
        
        for i in train_series:
            count+=1
            
            if(not(dim[0]==self.n_in)):
                temp_im2 = plt.imread(self.img_path+'/'+i+'.png')
                temp_im2 -= mean
                temp_im2 /= sd
                temp_im = scipy.misc.imresize(temp_im2,(self.n_in,self.n_in)).astype(float)
            
            else:
                temp_im-=mean
                temp_im /=sd
                temp_im = plt.imread(self.img_path+'/'+i+'.png').astype(float)
                
            temp_im.reshape(1,self.n_in,self.n_in)
            
            img_train[count,:,:,:] = torch.Tensor(temp_im)/255.0
            
            #img_train[count,:,:,:].sub(mean).div(sd)
            
            if not(self.label_path ==0):
                label_train[count,:] = torch.Tensor(scipy.misc.imresize(plt.imread(self.label_path+'/'+i+'.png')))
        
        
        if(patch_path=='./'):
            patch_path = self.img_path
        
        
        for i,j,k in os.walk(self.img_path):
            if '.ipynb_checkpoints' in i:
                #print(i)
                shutil.rmtree(i)
        
        
        dsets = {'Training':torch.utils.data.TensorDataset(img_train,label_train)}
        
        
        dset_loaders ={'Training': torch.utils.data.DataLoader(dsets['Training'],batch_size=self.b_size,shuffle=False,num_workers=4)
              }

        dset_sizes = {'Training':len(dsets['Training'])}
        ### TODO maybe use os.listdir to get the made folders ?? or make the folders on the fly based on test_only input
        return dset_loaders,dset_sizes
        
    def forward(self,x):
        x = x.view(-1,self.n_in*self.n_in)
        #print(x.size())
        x = self.fc1(x.view(-1,self.n_in*self.n_in))
        
        x = torch.nn.functional.sigmoid(x)
        
        ### hidden layer activation
        #a = x
        
        x = self.fc2(x)
        
        
        x = torch.nn.functional.sigmoid(x)
        
        return x#,a    

    
def train_sae(model,epochs):
    torch.cuda.set_device(model.gpu)

    optimizer = optim.Adam(model.parameters(),lr = model.lr)
    #fig = plt.figure()
    model.train()
    #ax = fig.add_subplot(1,1,1) 
    
    dataset_loader,data_size = model.transform()
    dataset_loader2,data_size2 = model.transform()
    
    #print(dataset_loader['Training'].next())
    #print(dataset_loader)
    #print(data_set['Training'])
    model = model.cuda()
    for epoch in range(0,epochs):
        print(epoch)
        running_loss=0
        for i in dataset_loader['Training']:
            inp1,_ = i
            
            optimizer.zero_grad()
            inp1 = Variable(inp1.cuda())
            
            out = model(inp1)
            #print(model.out_at_layer(inp1.view(-1,model.n_in**2),1))
            a = torch.mean(model.out_at_layer(inp1.view(-1,model.n_in**2),1),0)
            
            kl = torch.sum(model.rho*torch.log(model.rho/a) + (1-model.rho)*torch.log((1-model.rho)/(1-a)))
            
            loss = model.beta*kl + ((torch.norm(out - inp1.view(-1,model.n_in*model.n_in))**2)/(2*model.b_size)) + 0.5*model.lam*(torch.norm(model.state_dict()['fc1.weight'])+ torch.norm(model.state_dict()['fc2.weight']))
            
            running_loss+=loss.cpu()
            loss.backward()
            optimizer.step()
            del(loss)
            
            del(inp1)
        print(running_loss)
    model=model.cpu()     
    for p in model.modules():
        if isinstance(p,nn.Linear):
            #print(p)
            return p.weight.data,p.bias.data


    
    
class localnet(nn.Module):
    def __init__(self,img_path,label_path,b_size,test_fraction=0,gpu=0,lr=0.01):
        super(localnet,self).__init__()
        self.test_fraction=test_fraction
        self.img_path=img_path
        self.lr = lr
        self.label_path=label_path
        self.b_size=b_size
        self.gpu=gpu
        self.conv = nn.Conv2d(1,100,11)
        self.avg_pool = nn.AvgPool2d(6,6)
        self.classifier = nn.Linear(8100,1024)
        self.smax = nn.Softmax()
    def transform(self,rand = False,patch_path = './',mean=0,sd = 1,test_on=0):
        if(self.test_fraction==0):
            train_series = get_series(self.label_path,self.test_fraction)

        else:
            train_series,test_series = get_series(self.label_path,self.test_fraction)
            
            label_test = torch.zeros(len(test_series),1024)

            img_test = torch.zeros(len(test_series),1,64,64)

        dim = plt.imread(self.img_path+'/'+train_series[0]+'.png').shape
        
        label_train = torch.zeros(len(train_series),1024)
        
        img_train = torch.zeros(len(train_series),1,64,64)
        
        count = -1
        
        mean,sd = find_stats(self.img_path)
        self.b_size = len(train_series)
        
        for i in test_series:
            count+=1
            
            temp_im = plt.imread(self.img_path+'/'+i+'.png')
            #print(i)
            #print(temp_im.min())
            temp_im-=mean
            temp_im/=sd
            #plt.imshow(scipy.misc.imresize(temp_im,(64,64))),plt.show()
            temp_im = scipy.misc.imresize(temp_im,(64,64))
            temp_im = temp_im.reshape(1,64,64).astype('float')
            
            temp_im = torch.Tensor(temp_im)
            #print(temp_im.size())
            img_test[count,:,:,:] = temp_im
            
            if not(self.label_path ==0):
                temp_im2 = scipy.misc.imresize(plt.imread(self.label_path+'/'+i+'.png'),(32,32)).astype(float)
                #plt.imshow(temp_im2),plt.show()
                
                temp_im2[np.where(np.abs(temp_im2-0)>0)]=1.0
                
                #plt.imshow(temp_im2),plt.show()
                temp_im2 = torch.Tensor(temp_im2)
                label_test[count,:] = (temp_im2).view(1024)
        
        count=-1
        for i in train_series:
            count+=1
            
            temp_im = plt.imread(self.img_path+'/'+i+'.png')
            
            #print(temp_im.min())
            temp_im-=mean
            temp_im/=sd
            #plt.imshow(scipy.misc.imresize(temp_im,(64,64))),plt.show()
            temp_im = scipy.misc.imresize(temp_im,(64,64))
            temp_im = temp_im.reshape(1,64,64).astype('float')
            
            #print(temp_im.shape)
            #print(type(temp_im))
            temp_im = torch.Tensor(temp_im)
            #print(temp_im.size())
            img_train[count,:,:,:] = temp_im
            
            if not(self.label_path ==0):
                temp_im2 = scipy.misc.imresize(plt.imread(self.label_path+'/'+i+'.png'),(32,32)).astype(float)
                #plt.imshow(temp_im2),plt.show()
                #print(temp_im2.min())
                #print(temp_im2.max())
                temp_im2[np.where(np.abs(temp_im2-0)>0)]=1
                
                #plt.imshow(temp_im2),plt.show()
                temp_im2 = torch.Tensor(temp_im2)
                label_train[count,:] = (temp_im2).view(1024)
            
        dsets = {'Training':torch.utils.data.TensorDataset(img_train,label_train)}
        
        dset_loaders ={'Training':torch.utils.data.DataLoader(dsets['Training'],batch_size=self.b_size,shuffle=False,num_workers=4)
              }
        
        dset_sizes = {'Training':len(dsets['Training'])}

        
        dsets1 = {'Test':torch.utils.data.TensorDataset(img_test,label_test)}
        dset_loaders1 ={'Test':torch.utils.data.DataLoader(dsets1['Test'],batch_size=self.b_size,shuffle=False,num_workers=4)
              }
        dset_sizes1 = {'Test':len(dsets1['Test'])}
        
        with open('/home/gam2018/cached_data/test.p','wb') as f:
            
        
            pickle.dump(dset_loaders1,f)    
        
        with open('/home/gam2018/cached_data/test_size.p','wb') as f:
            
            pickle.dump(dset_sizes1,f)    
            
        return dset_loaders,dset_sizes

    def store_model(self,f_name):

        
        state={'state_dict':self.state_dict()}
                          #,'optimizer':self.model_optimizer.state_dict()}
        torch.save(state,f_name)
    def out_at_layer(self,inp,L):

        count= -1
        for i in self.modules():
            count+=1
            try:
                if(count==L):
                    #print(tt)
                    return(torch.nn.functional.sigmoid(i(inp)))

            except:
                continue

    def load_model(self,filename):
        checkpoint = torch.load(filename)
        
        
        
        # create new OrderedDict that does not contain `module.`
        if not(isinstance(self.gpu,int)):
            new_state_dict = OrderedDict()
            for k, v in checkpoint['state_dict'].items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            # load params
            self.load_state_dict(new_state_dict)
        
        else:
            
            self.load_state_dict(checkpoint['state_dict'])
            
        #if(self.resume):
            #self.model_optimizer.load_state_dict(checkpoint['optimizer'])
        #return model,optimz,start_epoch

    def forward(self,x):
        
        x = self.conv(x)
        x = nn.functional.sigmoid(x)
        x = self.avg_pool(x)
        
        x = x.view(-1,8100)
        
        x = self.classifier(x)
        x = nn.functional.sigmoid(x)
        x =  self.smax(x)
        return x


    
def train_lnet(model,epochs,fname,save_dir,pretrain=True):
    torch.cuda.set_device(model.gpu)
    optimizer = optim.Adam(model.parameters(),lr = model.lr)
    #fig = plt.figure()
    model.train()
    
    dataset_loader,data_size = model.transform()
    dataset_loader2,data_size2 = model.transform()
    model = model.cuda()
    loss_arr = []
    ###pre trainining output layer - 
    if(pretrain):
    
        model.train()
        for p in model.conv.parameters():
            p.requires_grad=False

        #p.classifier = nn.Linear(8100,1024)
        for epoch in range(0,epochs):
            #print(epoch)
            running_loss= 0
            for i in dataset_loader['Training']:
                inp1,label = i
                label = Variable((label).cuda())
                optimizer.zero_grad()
                inp1 = Variable(inp1.cuda())

                out = model(inp1)
                #print(out.size())

                #print(label.size())

                #print(out)
                #print(label)
                #print(model.state_dict()['conv.weight'])
                #loss =((torch.norm(out - label)**2)/(2*data_size['Training'])) +torch.norm(model.state_dict()['classifier.weight'])/(2*(10**4))
                loss =torch.nn.functional.binary_cross_entropy(out,label) +torch.norm(model.state_dict()['classifier.weight'])/(2*(10**4))
                #print(out[0].cpu().numpy())
                running_loss+=loss.data
                loss.backward()
                optimizer.step()
                del(loss)

                del(inp1)
            #print(running_loss)
        for p in model.conv.parameters():

            p.requires_grad=True
        
    model.train()
    optimizer2 = optim.Adam(model.parameters(),lr = model.lr)
    loss_i = np.inf
    for epoch in range(0,epochs):
        #print(epoch)
        running_loss= 0
        for i in dataset_loader2['Training']:
            inp1,label = i
            label = Variable(label.cuda())
            optimizer2.zero_grad()
            inp1 = Variable(inp1.cuda())
            
            out = model(inp1)
            #print(out.size())
            
            #print(label.size())
            
            #print(out)
            #print(label)
            
            loss =(torch.norm(out - label)**2)/(2*data_size2['Training']) + (torch.norm(model.state_dict()['classifier.weight']) + torch.norm(model.state_dict()['conv.weight']))/(2*(10**4))
            #print(out[0].cpu().numpy())
            #scipy.misc.imsave(save_dir+'/'+str(epoch)+'.png',scipy.misc.imresize(out[0,:].cpu().data.numpy().reshape(32,32),(256,256)))
            
            running_loss+=loss.data
            loss.backward()
            optimizer2.step()
            del(loss)
            
            del(inp1)
        #print(running_loss)
        loss_arr.append(running_loss.cpu().numpy()[0])
        print(running_loss.cpu().numpy()[0])
        print(running_loss.cpu().numpy())
        if(running_loss.cpu().numpy()[0]<loss_i):
            print('loss_i')
            loss_i = running_loss.cpu().numpy()[0]
            best_model = model
    #plt.figure()    
    plt.plot(np.array(loss_arr))
    plt.savefig(save_dir+'/'+'train_loss.png')
    plt.show()
    
    model=model.cpu() 
    model.store_model(fname)
    best_model.store_model(fname[:fname.find('.pth.')]+'_best_loss.pth.tar')
    
def test_lnet(model,fname='0',save_dir='0'):
    torch.cuda.set_device(model.gpu)
    #optimizer = optim.SGD(model.parameters(),lr = 0.001)
    #fig = plt.figure()
    model.eval()
    
    if(not(fname=='0')):
        model.load_model(fname)
        
    with open('/home/gam2018/cached_data/test.p', 'rb') as pickle_file:
        dataset_loader = pickle.load(pickle_file)
    
    with open('/home/gam2018/cached_data/test_size.p', 'rb') as pickle_file:
        dset_size = pickle.load(pickle_file)
    
    model = model.cuda()
    

    for i in dataset_loader['Test']:
        #print(i)
        inp1,label = i
        print(label.size())
        inp1 = Variable(inp1.cuda())
        #print(inp1.size())
        label = Variable(label.cuda())
        out = model(inp1)
        
        for i in range(label.size()[0]):
            scipy.misc.imsave(save_dir+'/'+str(i)+'_img.png',
                              scipy.misc.imresize(inp1.data.cpu()[i,0,:,:].numpy().reshape(64,64),(256,256)))
            
            scipy.misc.imsave(save_dir+'/'+str(i)+'_test.png',
                              scipy.misc.imresize(out.data.cpu()[i,:].numpy().reshape(32,32),(256,256)))
            scipy.misc.imsave(save_dir+'/'+str(i)+'_label.png',
                              scipy.misc.imresize(label.data.cpu()[i,:].numpy().reshape(32,32),(256,256)))
        
        #plt.imshow(out.cpu().numpy().reshape(32,32)),plt.show()
        #plt.imshow(label.cpu().numpy().reshape(32,32)),plt.show()

        
        del(inp1)

    model=model.cpu() 
    model.store_model(fname)
    

class StackedAE(nn.Module):
    def __init__(self,img_path,label_path,gpu=0,n_in = 64,n_h =100 ,n_out = 64,test_fraction=0,lr=0.01,test_train_dst = '/data/gabriel/LVseg/dataset_img/data_seg',b_size=1000):
        super(StackedAE,self).__init__()
        self.img_path=img_path
        self.label_path=label_path
        self.gpu = gpu
        self.n_in = n_in
        self.n_h = n_h
        self.n_out = n_out
        self.test_fraction = test_fraction
        self.lr = lr
        self.fc1 = nn.Linear(n_in*n_in,n_h)
        self.fc2 = nn.Linear(n_h,n_h)
        self.fc3 = nn.Linear(n_h,n_out*n_out)
        self.test_train_dst = test_train_dst
        self.b_size=b_size
    def hidden1(self,x):
        x = self.fc1(x)
        x = nn.functional.sigmoid(x)
        return x
    
    def hidden2(self,x):
        x = self.fc2(x)
        x = nn.functional.sigmoid(x)
        return x
    
    def target_layer(self,x):
        x = self.fc3(x)
        x = nn.functional.sigmoid(x)
        return x
    
    def out_at_layer(self,inp,L):
    
        count= -1
        for i in self.modules():
            count+=1
            try:
                if(count==L):
                    #print(tt)
                    return(torch.nn.functional.sigmoid(i(inp)))
                    
            except:
                continue
    
    
    def forward(self,x):
        x = x.view(-1,self.n_in*self.n_in)
        
        x = self.hidden1(x)
        
        x = self.hidden2(x)
        
        x = self.target_layer(x)
        
        return x
       
    def transform(self,rand = False,mean=0,sd = 1):
        
        ### JUST ENTER THE PATH!
        dst = self.test_train_dst
        
        mean,sd = find_stats(self.img_path)
        
        
        train_l,test_l= split_im(src_img = self.img_path,src_label = self.label_path,
                 dst =dst)
        
        count=-1
        
        img_test = torch.Tensor(test_l,1,64,64)
        label_test = torch.Tensor(test_l,4096)
        
        for i in os.listdir(dst+'/'+'test_img'):
            count+=1

            temp_im = plt.imread(dst+'/test_img/'+i)
            temp_im-=mean
            temp_im/=sd

            img_test[count,:,:,:] = torch.Tensor(scipy.misc.imresize(temp_im,(64,64))/256.0)

            temp_im = scipy.misc.imresize(plt.imread(dst+'/'+'test_label'+'/'+i),(64,64))
            temp_im[temp_im>0]=1.0
            label_test[count,:] = torch.Tensor(temp_im.astype(float)).view(4096)
                
        dsets1 = {'Test':torch.utils.data.TensorDataset(img_test,label_test)}

        dset_loaders1 ={'Test': torch.utils.data.DataLoader(dsets1['Test'],batch_size=self.b_size,shuffle=False,num_workers=4)
              }
        dset_sizes1 = {'Test':len(dsets1['Test'])}

        with open(dst+'/test_loader_st.p','wb') as f:
            pickle.dump(dset_loaders1 ,f)

        with open(dst+'/test_size_st.p','wb') as f:
            pickle.dump(dset_sizes1 ,f)

        
        label_train = torch.zeros(train_l,4096)
        img_train = torch.zeros(train_l,1,64,64)
        
        count = -1
        
        for i in os.listdir(dst+'/'+'train_img'):
            count+=1

            temp_im = plt.imread(dst+'/train_img/'+i)
            temp_im-=mean
            temp_im/=sd

            img_train[count,:,:,:] = torch.Tensor(scipy.misc.imresize(temp_im,(64,64))/256.0)

            temp_im = scipy.misc.imresize(plt.imread(dst+'/'+'train_label'+'/'+i),(64,64))
            temp_im[temp_im>0]=1.0
            label_train[count,:] = torch.Tensor(temp_im.astype(float)).view(4096)
                
        dsets = {'Training':torch.utils.data.TensorDataset(img_train,label_train)}
        
        dset_loaders ={'Training': torch.utils.data.DataLoader(dsets['Training'],batch_size=self.b_size,shuffle=False,num_workers=4)
              }

        dset_sizes = {'Training':len(dsets['Training'])}

        with open(dst+'/train_loader_st.p','wb') as f:
            pickle.dump(dset_loaders1 ,f)

        with open(dst+'/train_size_st.p','wb') as f:
            pickle.dump(dset_sizes1 ,f)
      
        ### TODO maybe use os.listdir to get the made folders ?? or make the folders on the fly based on test_only input
        return dset_loaders,dset_sizes

def train_st_ae(model_st_ae,epochs,cache_dir):
    
    ### cache dir to store the in hidden layer activations
    torch.cuda.set_device(model_st_ae.gpu)
    
    
    optimizer = optim.Adam(model_st_ae.parameters(),lr = model_st_ae.lr)
    #num_
    model_st_ae.train()
    dataset_loader,data_size = model_st_ae.transform()
    #model = model_st_ae.cuda()
    loss_arr = []
    
    ### train the first weight
    
    #w1,b1 = train_sae(model=sae_1,epochs = 3000)
    count = 1
    cache = model_st_ae.img_path
    while(count<=3):
        print(count)
        try:
            #print('here')
            shutil.rmtree(cache_dir+'/res_at_'+str(count))
            #print('here2')
            shutil.rmtree(cache_dir+'/resin_at_'+str(count))
            os.makedirs(cache_dir+'/'+'res_at_'+str(count))
            os.makedirs(cache_dir+'/'+'resin_at_'+str(count))
        except:
            os.makedirs(cache_dir+'/'+'res_at_'+str(count))
            os.makedirs(cache_dir+'/'+'resin_at_'+str(count))

        if(count==3):
            sae_1 = SAE(n_in=64,n_h=100,n_out=64,img_path=model_st_ae.label_path,b_size = 1000, patch_size = 0,lr=0.001,rho=0.1,gpu=1,lam = 3*(10**-3),beta = 3,test_fraction=0)
        else:
            sae_1 = SAE(n_in=64,n_h=100,n_out=64,img_path=cache,b_size = 1000, patch_size = 0,lr=0.001,rho=0.1,gpu=1,lam = 3*(10**-3),beta = 3,test_fraction=0)
        
            l_count = 1
        
            w1,b1 = train_sae(model=sae_1,epochs = 3)

            for j in model_st_ae.modules():
                if isinstance(j,nn.Linear) and l_count == count:
                    j.weight.data,j.bias.data = w1,b1
                    break
                elif(isinstance(j,nn.Linear) and l_count <count):
                    l_count+=1

            ###cache hidden layer output as dataset for next pretraining
            temp_model = model_st_ae
            temp_model.cuda()


            b_count=-1
            for temp_inp in dataset_loader['Training']:
                inp,_ = temp_inp
                b_count+=1
                inp = Variable(inp.cuda())
                out = temp_model.out_at_layer(inp.view(-1,sae_1.n_in*sae_1.n_in),L=count)

                out = out.cpu().data.numpy()
                inp = inp.cpu().data.numpy()

                for i in range(0,out.shape[0]):
                    #print(np.sqrt(sae_1.n_h))
                    scipy.misc.imsave(cache_dir+'/'+'res_at_'+str(count)+'/'+'out_at_l'+str(count)+'_' +str(b_count)+'.png',out[int(i),:].reshape(int(np.sqrt(sae_1.n_h)),int(np.sqrt(sae_1.n_h))))    
                    scipy.misc.imsave(cache_dir+'/'+'resin_at_'+str(count)+'/'+'in_at_l'+str(count)+'_' +str(b_count)+'.png',inp[int(i),:].reshape(int(sae_1.n_in),int(sae_1.n_in)))
                #print(out.shape)

            cache = cache_dir+'/'+'res_at_'+str(count)+'/'



            count+=1
            
            #return
            #scipy.misc.imsave(cache_dir+'/'+'res_at_'+str(count)+'/'+'out_at_l'+str(count)+'_' +str(b_count),out.cpu().data.numpy().reshape(np.sqrt(sae_1.n_h),np.sqrt(sae_1.n_h)))
            #scipy.misc.imsave(cache_dir+'/'+'res_at_'+str(count)+'/'+'in_at_l'+str(count)+'_' +str(b_count),inp.cpu().data.numpy().reshape(np.sqrt(sae_1.n_h),np.sqrt(sae_1.n_h)))
            
            
            del temp_model
                
            
        #model = model_st_ae.cuda()    
    return    
    
    
def test_st_ae(model,fname='0',save_dir='0'):
    torch.cuda.set_device(model.gpu)
    #optimizer = optim.SGD(model.parameters(),lr = 0.001)
    #fig = plt.figure()
    model.eval()
    
    if(not(fname=='0')):
        model.load_model(fname)
        
    with open('/data/gabriel//LVseg/progress_box/test_box/test_loader_st.p', 'rb') as pickle_file:
        dataset_loader = pickle.load(pickle_file)
    
    with open('/data/gabriel//LVseg/progress_box/test_box/test_size_st.p', 'rb') as pickle_file:
        dset_size = pickle.load(pickle_file)
    
    model = model.cuda()
    

    for i in dataset_loader['Test']:
        #print(i)
        inp1,label = i
        print(label.size())
        inp1 = Variable(inp1.cuda())
        #print(inp1.size())
        label = Variable(label.cuda())
        out = model(inp1)
        
        for i in range(label.size()[0]):
            scipy.misc.imsave(save_dir+'/'+str(i)+'_test.png',
                              scipy.misc.imresize(out.data.cpu()[i,:].numpy().reshape(64,64),(100,100)))
            scipy.misc.imsave(save_dir+'/'+str(i)+'_label.png',
                              scipy.misc.imresize(label.data.cpu()[i,:].numpy().reshape(64,64),(100,100)))

        #plt.imshow(out.cpu().numpy().reshape(32,32)),plt.show()
        #plt.imshow(label.cpu().numpy().reshape(32,32)),plt.show()

        
        del(inp1)

    model=model.cpu() 
    model.store_model(fname)
            
 