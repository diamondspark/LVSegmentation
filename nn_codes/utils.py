import os
import matplotlib.pyplot as plt
import simplejson
import shutil

import matplotlib.pyplot as plt
import numpy as np
import os
import random
import scipy.misc

def get_box(label_path): 
    
    #print(os.listdir(label_path))
    img_gen = (i for i in os.listdir(label_path) if len(i)>5)

    for i in img_gen:
        img = plt.imread(label_path+'/'+i)
        if(img.max()==1):
            box = np.zeros_like(img)
            mean_pos = np.mean(np.where(img==1),axis=1).astype(int)
            box[mean_pos[0]-49:mean_pos[0]+50,mean_pos[1]-49:mean_pos[1]+50] = 1
        scipy.misc.imsave('/data/gabriel/LVseg/dataset_img/box_256/'+i,box)
    
def get_series(label_path,test_fraction = 0.05):
    
    series_name = [i[:-4] for i in os.listdir(label_path) if '.png' in i]
    
    random.shuffle(series_name)
    sz = len(series_name)
    ### returns train and test
    if (test_fraction>0):
        return series_name[test_fraction*sz//1 : ],series_name[:-1+test_fraction*sz//1]
    else:
        return series_name[test_fraction*sz//1 : ]

def find_stats(path):
    
    im_list = (plt.imread(path+'/'+i).reshape(1,-1) for i in os.listdir(path) if not(i=='.') and not(i=='.DS_Store'))
    count = 0
    Ex2 = 0
    mean = 0
    for i in im_list:
        mean += i.mean()
        y = i**2
        Ex2 += y.mean()
        count+=1
    mean/=count
    Ex2/=count
    Ex2-=mean**2
    Ex2 = np.sqrt(Ex2)
    return mean,Ex2


def get_patches(data_path,save_path,count = 10**4):
    count1 = 0

    #raise NotImplementedError 

    #for i in os.listdir(data_path+'/Training'):
    #    for j in os.listdir(data_path+'/Training/'+i):
    #        print


    try:
        shutil.rmtree(save_path)
        os.makedirs(save_path)
    except:
        os.makedirs(save_path)


    name_im_list1 = [i for i in os.listdir(data_path) if not(i=='.') 
              and not(i=='.DS_Store') and np.sum(plt.imread(data_path+'/'+i).shape) == 512 ]

    im_list1 = [scipy.misc.imresize(plt.imread(data_path+'/'+i),(64,64)) for i in os.listdir(data_path) if not(i=='.') 
              and not(i=='.DS_Store') and np.sum(plt.imread(data_path+'/'+i).shape) == 512 ]

    f = open('/data/gabriel/LVseg/name_im_list.txt', 'w')
    simplejson.dump(name_im_list1, f)
    f.close()
    
    random.shuffle(im_list1)
    num_images=len(im_list1)
    print(num_images)
    i=0
    while(count1<count):
        #print('here')
        if(i>=num_images-1):
            #print('here')
            i=0
        #plt.imshow(i),plt.show()

        R = random.randint(11,64)

        C = random.randint(11,64)
        #print(im_list1[i].shape)
        #print(im_list1[i][R-11:R,C-11:C].mean())
        print(i)
        print(R,C)
        while(im_list1[i][R-11:R,C-11:C].mean()<1):

            R = random.randint(11,64)

            C = random.randint(11,64)
            patch = im_list1[i][R-11:R,C-11:C]

        patch = im_list1[i][R-11:R,C-11:C]

        scipy.misc.imsave(save_path+'/'+str(count1)+'.png',patch)
        count1+=1
        i+=1
