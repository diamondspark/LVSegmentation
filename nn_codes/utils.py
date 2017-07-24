import os
import matplotlib.pyplot as plt
import simplejson
import shutil

import matplotlib.pyplot as plt
import numpy as np
import os
import random
import scipy.misc

'''
def compare_result(box_path,label_path,img_path):
    for i in os.listdir(box_path):
        if('.png' in i):
            img = plt.imread()
'''
def split_neg(label_src,img_src,dst):
    img_gen = (i for i in os.listdir(label_src) if '.png' in i)
    
    try:
        shutil.rmtree(dst+'/'+'neg')
        shutil.rmtree(dst+'/'+'pos')
        os.makedirs(dst+'/'+'neg')
        os.makedirs(dst+'/'+'pos')
        os.makedirs(dst+'/'+'neg/label')
        os.makedirs(dst+'/'+'pos/label')
        os.makedirs(dst+'/'+'neg/img')
        os.makedirs(dst+'/'+'pos/img')

    except:
        os.makedirs(dst+'/'+'neg')
        os.makedirs(dst+'/'+'pos')
        os.makedirs(dst+'/'+'neg/label')
        os.makedirs(dst+'/'+'pos/label')
        os.makedirs(dst+'/'+'neg/img')
        os.makedirs(dst+'/'+'pos/img')
        
                    
        
    for i in img_gen:
        label_img = plt.imread(label_src+'/'+i)
        img = plt.imread(img_src+'/'+i)
        if(label_img.max()==0):
            scipy.misc.imsave(dst+'/'+'neg/label/'+i,label_img)
            scipy.misc.imsave(dst+'/'+'neg/img/'+i,img)
        
        else:
            scipy.misc.imsave(dst+'/'+'pos/label/'+i,label_img)
            scipy.misc.imsave(dst+'/'+'pos/img/'+i,img)

def get_256_data(img_src,label_src,label_dst,img_dst):
    for i in os.listdir(img_src):
        img = plt.imread(img_src+'/'+i)
        if(i in os.listdir(label_src) and img.shape[0]==img.shape[1] and img.shape[0]==256):
            


            scipy.misc.imsave(label_dst+'/'+i,
                             plt.imread(label_src+'/'+i)
                             )

            scipy.misc.imsave(img_dst+'/'+i,
                             plt.imread(img_src+'/'+i)
                             )


def split_im(src_img,src_label,dst,split_fraction=0.8):
    
    train,test = get_series(label_path = src_label,test_fraction = 1-split_fraction)
    
    try:
        shutil.rmtree(dst)
        os.makedirs(dst)
        os.makedirs(dst+'/'+'train_label')
        os.makedirs(dst+'/'+'test_img')
        os.makedirs(dst+'/'+'train_img')
        os.makedirs(dst+'/'+'test_label')
    except:    
        os.makedirs(dst)
        os.makedirs(dst+'/'+'train_label')
        os.makedirs(dst+'/'+'test_img')
        os.makedirs(dst+'/'+'train_img')
        os.makedirs(dst+'/'+'test_label')

    img_train_gen = [k for k in os.listdir(src_img) if '.png' in k and k[:-4] in train]
    img_test_gen = [k for k in os.listdir(src_img) if '.png' in k and k[:-4] in test]
   
    for i in img_train_gen: 
        #print(i)
        scipy.misc.imsave(dst+'/'+'train_img/'+i+'.png',plt.imread(src_img+'/'+i))
        scipy.misc.imsave(dst+'/'+'train_label/'+i+'.png',plt.imread(src_label+'/'+i))
    
    for i in img_test_gen:
        scipy.misc.imsave(dst+'/'+'test_img/'+i+'.png',plt.imread(src_img+'/'+i))
        scipy.misc.imsave(dst+'/'+'test_label/'+i+'.png',plt.imread(src_label+'/'+i))
            
    return len(img_train_gen),len(img_test_gen)
    
def crop_roi(img_path,label_path,save_path='/data/gabriel/LVseg/dataset_img/cropped'):
    try:
        shutil.rmtree(save_path)
        os.makedirs(save_path)
        os.makedirs(save_path+'/'+'img')
        os.makedirs(save_path+'/'+'neg_img')
        os.makedirs(save_path+'/'+'label')
        os.makedirs(save_path+'/'+'neg_label')
    except:
        os.makedirs(save_path)
        os.makedirs(save_path+'/'+'img')
        os.makedirs(save_path+'/'+'neg_img')
        os.makedirs(save_path+'/'+'label')
        os.makedirs(save_path+'/'+'neg_label')
        
    label_name_gen = (i for i in os.listdir(label_path) if '.png' in i)
    zero = np.zeros((100,100))
    for i in label_name_gen:
    
        img = plt.imread(img_path+'/'+i)
        label = plt.imread(label_path+'/'+i)
        
        
        
        if((label**2).sum()==0):
            scipy.misc.imsave(save_path+'/'+'neg_label'+'/'+i,zero)
            scipy.misc.imsave(save_path+'/'+'neg_img'+'/'+i,img)
            
        else:
            

            mean_pos = np.mean(np.where(label>0),axis=1).astype(int)

            scipy.misc.imsave(save_path+'/'+'label'+'/'+i,label[mean_pos[0]-50 : mean_pos[0]+50,mean_pos[1]-50 : mean_pos[1]+50 ])
            scipy.misc.imsave(save_path+'/'+'img'+'/'+i,img[mean_pos[0]-50 : mean_pos[0]+50,mean_pos[1]-50 : mean_pos[1]+50 ])

        
        
def get_box(label_path): 
    
    #print(os.listdir(label_path))
    img_gen = (i for i in os.listdir(label_path) if len(i)>5)

    for i in img_gen:
        img = plt.imread(label_path+'/'+i)
        if(img.max()==1):
            box = np.zeros_like(img)
            mean_pos = np.mean(np.where(img==1),axis=1).astype(int)
            print(mean_pos)
            box[mean_pos[0]-50:mean_pos[0]+50,mean_pos[1]-50:mean_pos[1]+50] = 1
        scipy.misc.imsave('/data/gabriel/LVseg/dataset_img/box_256/'+i,box)
    
def get_series(label_path,test_fraction = 0.05):
    
    series_name = [i[:-4] for i in os.listdir(label_path) if '.png' in i]
    
    random.shuffle(series_name)
    sz = len(series_name)
    ### returns train and test
    train_end = int(np.ceil(sz - test_fraction*sz))
    test_start = int(train_end+1)
    #print(train_end)
    #print(test_start)
    if (test_fraction>0):
        return series_name[ :train_end ],series_name[test_start:]
    else:
        return series_name

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


def get_patches(data_path,save_path,count = 5*10**4,patch_resize=True,patch_size=64):
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

    if(patch_resize):
        im_list1 = [scipy.misc.imresize(plt.imread(data_path+'/'+i),(patch_size,patch_size)) for i in os.listdir(data_path) if not(i=='.') 
              and not(i=='.DS_Store') and np.sum(plt.imread(data_path+'/'+i).shape) == 512 ]

    else:
        patch_size = 256
        im_list1 = [plt.imread(data_path+'/'+i) for i in os.listdir(data_path) if not(i=='.') 
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
        
        
        R = random.randint(11,patch_size)

        C = random.randint(11,patch_size)
        
        #print(i)
        #print(R,C)
        #if(i==0):
            #print(im_list1[i].max()) 
        
        im_max =im_list1[i].max()/255 
        while(im_list1[i][R-11:R,C-11:C].mean()<im_max):

            R = random.randint(11,patch_size)

            C = random.randint(11,patch_size)
            patch = im_list1[i][R-11:R,C-11:C]

        patch = im_list1[i][R-11:R,C-11:C]

        scipy.misc.imsave(save_path+'/'+str(count1)+'.png',patch)
        count1+=1
        i+=1
