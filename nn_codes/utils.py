import os
import matplotlib.pyplot as plt
import simplejson
import shutil
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import scipy.misc
import cv2
from torch.autograd import Function
'''
def compare_result(box_path,label_path,img_path):
    for i in os.listdir(box_path):
        if('.png' in i):
            img = plt.imread()
'''

class DiceLoss(Function):
    def __init__(self):
        super(DiceLoss,self).__init__()
        pass
    
    def forward(self,out,target):
        eps = 1e-8
        self.out = out
        out_thresh = (torch.sign(out-0.5)+1)/2
        size0 = target.size(0)
        
        
        
        out_thresh = out_thresh.view(size0,64,64)
        
        target  = target.view(size0,64,64)
        intersect_map = out_thresh*target
        
        union_map = out_thresh*out_thresh + target*target
        
        intersect = torch.sum( intersect_map.view(size0,-1),1).squeeze()
        
        union = torch.sum(union_map.view(size0,-1),1).squeeze()
        
        
        self.target = target
        
        self.intersect = intersect
        self.union = union
        
        the_loss = torch.sum(2*intersect/(union+eps))/size0
        
        fin_loss = torch.FloatTensor(1).fill_(the_loss)
        
        #print(union.size())
        #print(intersect.size())
        
        #print(self.out.size())
        if(out.is_cuda):
            return fin_loss.cuda()
        else:
            return fin_loss
        
        
    def backward(self,grad_output):
        
        grad_input = torch.zeros(self.out.size())
        intersect = self.intersect
        union = self.union
        
        for i in range(self.target.size()[0]):
            #print(self.target.size())
            #print(self.out.size())
            
            dice_grad = 2*(self.target[i,:]*union[i] - 2*self.out[i,:]*intersect[i])/(union[i]**2)
            #print(grad_output.size())
            #print(dice_grad.size())
            grad_input[i,:] = dice_grad*grad_output[0]
        
        if self.out.is_cuda:
            grad_input = grad_input.cuda()
            
        
        return grad_input,None
    
    
        
def change_name(img_src,label_src,dst='/data/gabriel/LVseg/renamed/'):
    '''
    try:
        shutil.rmtree(dst)
        shutil.rmtree(dst+'/label')
        shutil.rmtree(dst+'/img')
        os.makedirs(dst)
        os.makedirs(dst+'/label')
        os.makedirs(dst+'/img')
    except:
        os.makedirs(dst)
        os.makedirs(dst+'/label')
        os.makedirs(dst+'/img')
    '''    
    pos = [ i for i,j in enumerate(img_src[::-1]) if j=='/' ][1]
    folder_name = img_src[::-1][1:pos][::-1]
    for i in os.listdir(img_src):
        if('.dcm' in i):
            shutil.copy(img_src+'/'+i,dst+'/img/'+folder_name+'_'+i)
        print(img_src)
    for i in os.listdir(label_src):
        print(label_src)
        if('.txt' in i ):
            shutil.copy(label_src+'/'+i,dst+'/label/'+folder_name+'_'+i)
            
    

def get_annot():
    labels = ['val_label','online_label','train_label']
    imgs = ['tr','on','val']
    for l in labels:
        
        for i in os.listdir('/data/gabriel/LVseg/MICCAI/'+l):
            for im in imgs:
                for j in os.listdir('/data/gabriel/LVseg/MICCAI/'+im):
                    for k in os.listdir('/data/gabriel/LVseg/MICCAI/'+l):
                        anno_img('/data/gabriel/LVseg/MICCAI/'+l+'/'+k,'/data/gabriel/LVseg/MICCAI/'+im)
                    
        
def anno_img(anno_src,img_src,dst = '/data/gabriel/LVseg/MICCAI/anno'):
    
    #print(img_src)
    
    img_list = [i[:i.find('.png')] for i in os.listdir(img_src+'/image/') if '.png' in i]
    #print(img_list[0])
    for i in os.listdir(anno_src):
        #print(i[:12])
        if(i[:12] in img_list):
            #print('here')
            img = plt.imread(img_src+'/image/'+i[:12]+'.png')
            anno = np.zeros_like(img)
            anno_arr = np.genfromtxt(anno_src+'/'+i)
            
            anno[anno_arr.astype(int)] = 1
            #plt.imshow(anno),plt.show()
            if('tr' in anno_src and 'tr' in img_src):
                scipy.misc.imsave(dst+'/tr/'+i[:12]+'.png',anno)
                
            elif('val' in anno_src and 'val' in img_src):
                scipy.misc.imsave(dst+'/val/'+i[:12]+'.png',anno)
                
            elif('on' in anno_src and 'on' in img_src):
                scipy.misc.imsave(dst+'/on/'+i[:12]+'.png',anno)
                
            
def get_contour(label_src='/data/gabriel/LVseg/dataset_img/cropped/label/',
                img_src='/data/gabriel/LVseg/dataset_img/cropped/img/',
                dst='/data/gabriel/LVseg/dataset_img/cropped/contour',
                dst1='/data/gabriel/LVseg/dataset_img/cropped/not_contour'):
    try:
        shutil.rmtree(dst)
        os.makedirs(dst)
        os.makedirs(dst+'/'+'label')
        os.makedirs(dst+'/'+'img')
        shutil.rmtree(dst1)
        os.makedirs(dst1)
        os.makedirs(dst1+'/'+'label')
        os.makedirs(dst1+'/'+'img')

    except:
        os.makedirs(dst)
        os.makedirs(dst+'/'+'img')
        os.makedirs(dst+'/'+'label')
        os.makedirs(dst1)
        os.makedirs(dst1+'/'+'img')
        os.makedirs(dst1+'/'+'label')

    for i in os.listdir(label_src):
        img = plt.imread(label_src+'/'+i)
        img2 = cv2.imread(label_src+'/'+i,0)
        im,cc,hi= cv2.findContours(img2.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        count=0
        res = []
        #plt.imshow(img2),plt.show()
        #for c in cc:
        #    res.append(cv2.isContourConvex(c))
        #print(res)
        
        
        if(len(cc)==2 and len(cc[0])/len(cc[1]) > 0.7 ):
            scipy.misc.imsave(dst+'/img/'+i,plt.imread(img_src+'/'+i))
            scipy.misc.imsave(dst+'/label/'+i,img)
            #print()
        elif(len(cc)==1 ):
            #plt.imshow(img),plt.show()
            #print(cc)
            M = cv2.moments(cc[0])
            #if(M['m00']==0):
            #    plt.imshow(img),plt.show()
                
            if(M['m00'] ==0):
                scipy.misc.imsave(dst+'/img/'+i,plt.imread(img_src+'/'+i))
                scipy.misc.imsave(dst+'/label/'+i,img)
            else:
                
                mean_r = int(M['m10']/(M['m00']))
                mean_c = int(M['m01']/(M['m00']))
                #print(cc[0])
                #return
                #print(mean_r,mean_c)


                if(img2[mean_r,mean_c]):
                    scipy.misc.imsave(dst+'/img/'+i,plt.imread(img_src+'/'+i))
                    scipy.misc.imsave(dst+'/label/'+i,img)
                    #print('here')

        else:
            scipy.misc.imsave(dst1+'/img/'+i,plt.imread(img_src+'/'+i))
            scipy.misc.imsave(dst1+'/label/'+i,img2)
            
            
            
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


def split_im(src_img,src_label,dst,split_fraction=0.7):
    
    train,val,test = get_series(label_path = src_label,test_fraction = 1-split_fraction)
    
    try:
        shutil.rmtree(dst)
        os.makedirs(dst)
        os.makedirs(dst+'/'+'val_label')
        os.makedirs(dst+'/'+'val_img')
        os.makedirs(dst+'/'+'train_label')
        os.makedirs(dst+'/'+'test_img')
        os.makedirs(dst+'/'+'train_img')
        os.makedirs(dst+'/'+'test_label')
    except:
        os.makedirs(dst)
        os.makedirs(dst+'/'+'val_label')
        os.makedirs(dst+'/'+'val_img')
        os.makedirs(dst+'/'+'train_label')
        os.makedirs(dst+'/'+'test_img')
        os.makedirs(dst+'/'+'train_img')
        os.makedirs(dst+'/'+'test_label')

    img_train_gen = [k for k in os.listdir(src_img) if '.png' in k and k[:-4] in train]
    img_val_gen = [k for k in os.listdir(src_img) if '.png' in k and k[:-4] in val]
    img_test_gen = [k for k in os.listdir(src_img) if '.png' in k and k[:-4] in test]
   
    for i in img_train_gen: 
        #print(i)
        if('.png' in i):
            i = i[:i.find('.png')]
        scipy.misc.imsave(dst+'/'+'train_img/'+i+'.png',plt.imread(src_img+'/'+i+'.png'))
        scipy.misc.imsave(dst+'/'+'train_label/'+i+'.png',plt.imread(src_label+'/'+i+'.png'))
    
    for i in img_test_gen:
        if('.png' in i):
            i = i[:i.find('.png')]

        scipy.misc.imsave(dst+'/'+'test_img/'+i+'.png',plt.imread(src_img+'/'+i+'.png'))
        scipy.misc.imsave(dst+'/'+'test_label/'+i+'.png',plt.imread(src_label+'/'+i+'.png'))

    for i in img_val_gen:
        if('.png' in i):
            i = i[:i.find('.png')]

        scipy.misc.imsave(dst+'/'+'val_img/'+i+'.png',plt.imread(src_img+'/'+i+'.png'))
        scipy.misc.imsave(dst+'/'+'val_label/'+i+'.png',plt.imread(src_label+'/'+i+'.png'))

        
    return len(img_train_gen),len(img_val_gen),len(img_test_gen)
    
def crop_roi(img_path,label_path,save_path='/data/gabriel/LVseg/dataset_img/cropped',few_images=False):
    try:
        shutil.rmtree(save_path)
        os.makedirs(save_path)
        os.makedirs(save_path+'/'+'img')
        os.makedirs(save_path+'/'+'neg_img')
        os.makedirs(save_path+'/'+'label')
        os.makedirs(save_path+'/'+'neg_label')
        os.makedirs(save_path+'/'+'test_images_corner')
        os.makedirs(save_path+'/'+'test_images_corner/img')
        os.makedirs(save_path+'/'+'test_images_corner/label')
    except:
        os.makedirs(save_path)
        os.makedirs(save_path+'/'+'img')
        os.makedirs(save_path+'/'+'neg_img')
        os.makedirs(save_path+'/'+'label')
        os.makedirs(save_path+'/'+'neg_label')
        os.makedirs(save_path+'/'+'test_images_corner')
        os.makedirs(save_path+'/'+'test_images_corner/img')
        os.makedirs(save_path+'/'+'test_images_corner/label')

    if(few_images):
        num_images = 3
    counter=0
        
    label_name_gen = (i for i in os.listdir(label_path) if '.png' in i)
    zero = np.zeros((100,100))
    for i in label_name_gen:
        if (few_images):
            
            print(i)
            print(i[:i.find('_img')])
            img = plt.imread(img_path+'/'+i[:2]+'_img.png')
            label = plt.imread(label_path+'/'+i[:2]+'_test.png')
           
        else:
            if(i in os.listdir(img_path)):
                img = plt.imread(img_path+'/'+i)
                label = plt.imread(label_path+'/'+i)


    
        #img = plt.imread(img_path+'/'+i)
        #label = plt.imread(label_path+'/'+i)
        
        
                if((label**2).sum()==0):
                    scipy.misc.imsave(save_path+'/'+'neg_label'+'/'+i,zero)
                    scipy.misc.imsave(save_path+'/'+'neg_img'+'/'+i,img)
            
                else:
                    mean_pos = np.mean(np.where(label>0),axis=1).astype(int)
                

                        
                    if(mean_pos[0] -50 >=0 and mean_pos[1] -50 >=0 and mean_pos[1] +50 <=label.shape[1] and mean_pos[0] +50 <= label.shape[0]):
                        scipy.misc.imsave(save_path+'/'+'label'+'/'+i,label[mean_pos[0]-50 : mean_pos[0]+50,mean_pos[1]-50 : mean_pos[1]+50 ])
                        scipy.misc.imsave(save_path+'/'+'img'+'/'+i,img[mean_pos[0]-50 : mean_pos[0]+50,mean_pos[1]-50 : mean_pos[1]+50 ])
                    else:

                        r_0 = max(mean_pos[0] -50,0)
                        c_0 = max(mean_pos[1] -50,0)
                        r_1 = min(mean_pos[0] +50,label.shape[0])
                        c_1 = min(mean_pos[1] +50,label.shape[1])

                        scipy.misc.imsave(save_path+'/'+'test_images_corner/label'+'/'+i,label[r_0 : r_1,c_0 : c_1 ])
                        scipy.misc.imsave(save_path+'/'+'test_images_corner/img'+'/'+i,img[r_0 : r_1,c_0 : c_1 ])

                            
       



                    #

                    #scipy.misc.imsave(save_path+'/'+'label'+'/'+i,label[mean_pos[0]-50 : mean_pos[0]+50,mean_pos[1]-50 : mean_pos[1]+50 ])
                    #scipy.misc.imsave(save_path+'/'+'img'+'/'+i,img[mean_pos[0]-50 : mean_pos[0]+50,mean_pos[1]-50 : mean_pos[1]+50 ])

        
        
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
    
    ### permanently 0.1 val fraction
    
    series_name = [i[:-4] for i in os.listdir(label_path) if '.png' in i]
    
    val_fraction=0.1
    train_fraction=0.9
    
    
    
    random.shuffle(series_name)
    sz = len(series_name)
    
        
    ### returns train and test
    
    val_start = 0
    val_end = np.ceil(sz*val_fraction).astype(int)
    #print(val_end)
    
    train_start = int(val_end+1)
    test_start = np.ceil(sz -test_fraction*sz).astype(int)
    train_end = int(test_start - 1)
    #print(train_start)
    #print(train_end+1)
    
    
    if (test_fraction>0):
        return series_name[ train_start:train_end ],series_name[val_start:val_end],series_name[test_start:]
    else:
        return series_name[train_start:train_end],series_name[:val_end]

def find_stats(path):
    
    im_list = (plt.imread(path+'/'+i).reshape(1,-1) for i in os.listdir(path) if not(i=='.') and not(i=='.DS_Store'))
    im_list2 = im_list
    count = 0
    Ex2 = 0
    mean = 0
    
    num_im = len([i for i in os.listdir(path) if '.png' in i])
    all_im = np.zeros((num_im,4096))
    
    count2=0
    
    for i in im_list2:
        all_im[count2,:] = i
        count2+=1
    
    Ex2 = np.std(all_im,0)
    mean = np.mean(all_im,0)
    
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

def calc_per_image_dice(output,label):
    output_thresh = np.zeros_like(output)
    output_thresh[output>0.5] = 1
    return 2.0*np.sum(output_thresh[label==1.0])/(np.sum(output_thresh) + np.sum(label))

    
    