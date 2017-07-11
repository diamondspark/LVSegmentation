import matplotlib.pyplot as plt

import  torch.utils.data as data_utils
import torch
import torch.nn as nn
from torchvision.models import inception
from torchvision.models import Inception3
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F

from torchvision import datasets,models,transforms
import torch.optim as optim


import numpy as np
import random
import gc

import copy
import torch
import numpy as np
from torch.autograd import Variable
import sys
import os
import scipy
import cv2
from PIL import Image
vc_nums = ['VC_10','VC_11','VC_12']

set_nums = ['SET1','SET2','SET3','SET4','SET5','SET6','SET7','SET8','SET9','SET10']
import os
import shutil
vc_num = 'VC_10'
import gc
from nndev import *
main_path =  '/data/Gurpreet/RUNS/'
try:
    os.makedirs('/data/gabriel/pat_accuracy')
except:
    shutil.rmtree('/data/gabriel/pat_accuracy')
    os.makedirs('/data/gabriel/pat_accuracy')
def find_sec(name):
    return [i for i,s in enumerate(name) if s=='_'][1]

for vc_num in vc_nums:
    for set_num in set_nums:
        model_names = [i for i in os.listdir(main_path+'/'+vc_num+'/'+set_num+'/') if '.pth.tar' in i and len(i)>5]
        model_path = main_path+'/'+vc_num+'/'+set_num
        test_path = main_path+'/'+vc_num+'/'+set_num+'/dataset/test'
        
        for model_name in model_names:    
            correct = 0
            wrong = 0
            #model_path = '/data/Gurpreet/RUNS/VC_10/SET1/'
            #model_name='ResNetModel_Pretrained_7_views_bs_100_e_10_28062017_0640.pth.tar'
            #test_path = '/data/Gurpreet/RUNS/VC_10/SET1/dataset/test'
            
            pat_vid = []
            classes = {}
            ord_label = []
            for i in os.listdir(test_path):
                if i not in classes:
                    ord_label.append(i)
                    classes[i]={}
                for j in os.listdir(test_path+'/'+i):
                    #print(j)
                    pat_vid_temp = j[:find_sec(j)]
                    #print(pat_vid_temp)
                    if(pat_vid_temp not in classes[i]):
                        classes[i][pat_vid_temp]=[]
                    classes[i][pat_vid_temp].append(test_path+'/'+i+'/'+j)


            
            temp_path = '/data/gabriel/dataset_temp'
            shutil.copy(model_path+'/'+model_name,'/data/gabriel/pat_acc/')
            
            m_name = model_name
            p2 = '/data/gabriel/pat_acc'
            res = models.resnet18(pretrained=True)
            res.fc = nn.Linear(res.fc.in_features,len(os.listdir(model_path+'/dataset/test'))) 


            for i in range(0,len(ord_label)):
                #print('i= ',i)
                #os.makedirs('/data/gabriel/temp/'+'/test/'+ord_label[i])
                for j in classes[ord_label[i]]:
                    #print('j ',j)
                    try:
                        os.makedirs('/data/gabriel/pat_acc/temp/test')
                    except:
                        shutil.rmtree('/data/gabriel/pat_acc/temp/test/')
                        os.makedirs('/data/gabriel/pat_acc/temp/test/')


                    for z in range(0,len(ord_label)):
                        try:

                            os.makedirs('/data/gabriel/pat_acc/temp/test/'+ord_label[z])
                            #print(ord_label[z])
                        except:

                            shutil.rmtree('/data/gabriel/pat_acc/temp/test/'+ord_label[z])
                            #s.makedirs('/data/gabriel/pat_acc/temp/test')
                            os.makedirs('/data/gabriel/pat_acc/temp/test/'+ord_label[z])
                            #print(ord_label[z])
                    #print(os.listdir('/data/gabriel/pat_acc/temp/test/'))
                    #reak
                    #print(classes[ord_label[i]].keys())
                    for k in classes[ord_label[i]][j]:
                        #print(k)


                        shutil.copy(k,'/data/gabriel/pat_acc/temp/'+'/test/'+ord_label[i])
                        #break
                    #break
                    obj1 = model_pip(model_in=res,data_path = '/data/gabriel/pat_acc/temp/',
                         batch_size=30, gpu=0,scale = True,use_gpu=True,resume=False,verbose=False)


                    obj1.load_model(p2+'/'+m_name)

                    obj1.test(model_dir=p2,model_name = m_name,test_on=True,random_crop=False,
                              save_miscl = False,folder = vc_num+'_'+set_num)
                    gc.collect()
                    arr = np.loadtxt(p2+'/'+vc_num+'_'+set_num+'/'+m_name
                                     [:m_name.find('.pth.tar')]+'/'+m_name+'accuracy.txt')
                    #print(arr)
                    del(obj1)
                    if(arr>0.5):
                        correct+=1
                    else:
                        wrong+=1
                    #print(correct,wrong)
            acc_arr = np.array([correct/(correct+wrong),correct,wrong,correct+wrong])
            np.savetxt('/data/gabriel/pat_accuracy/'+vc_num+'_'+set_num+'_'+model_name[:model_name.find('.pth.tar')]+
                       'pat_accuracy.txt',np.array(correct/(correct+wrong)).reshape(1,))    
            print(vc_num,set_num,model_name)
            print(correct/(correct+wrong))