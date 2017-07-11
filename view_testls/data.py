### this file is for generating data.
import shutil
import os
import numpy as np
def sample_data(path = '/storage/gabriel/VC/Normal_Images/',eq = False):
    data_dic = {}
    #test_arr = [i for i in os.listdir(path) if len(i)>2]
    #print(test_arr)
    len_arr = np.array([len(os.listdir(path+i)) for i in os.listdir(path) if '.ipynb' not in i])
    target_name = [i for i in os.listdir(path)]
    
    os.system('rm -rf /storage/gabriel/VC/dataset/*')
    
    if(eq):
        #print(len_arr)
        
        base_num = min(len_arr)
        train_ind = np.random.choice(base_num,base_num*7//10,replace=False)
        left_out_ind = np.array([i for i in np.arange(0,base_num) if i not in train_ind])
        val_ind = np.random.choice(left_out_ind,base_num//5,replace=False)
        test_ind = np.array([i for i in left_out_ind if i not in val_ind])

        data_dic_ind = {'train' : train_ind,'val' : val_ind, 'test' : test_ind}
        
        for i in data_dic_ind.keys():
            try:
                os.makedirs('/storage/gabriel/VC/dataset/'+i)
                for j in target_name:
                    os.makedirs('/storage/gabriel/VC/dataset/'+i+'/'+j)
                
            except:
                pass
        for i in os.listdir(path):
            count = 0
            #print(i)
            
            if('.ipynb' not in i):
                for j in os.listdir(path+i):


                    #print(j)
                    #print(path+'/'+i+'/'+j)
                    #return
                    if(count in data_dic_ind['train']):
                        src = path+'/'+i+'/'+j
                        dst = '/storage/gabriel/VC/dataset/'+'train/'+i
                        #!cp $path/$i/$j /storage/gabriel/VC/dataset/train/$i/
                        shutil.copy(path+'/'+i+'/'+j,'/storage/gabriel/VC/dataset/'+'train/'+i)

                    if(count in data_dic_ind['test']):
                        #!cp $path/$i/$j /storage/gabriel/VC/dataset/train/$i/
                        shutil.copy(path+'/'+i+'/'+j,'/storage/gabriel/VC/dataset/'+'test/'+i)

                    if(count in data_dic_ind['val']):

                        #!cp $path/$i/$j /storage/gabriel/VC/dataset/train/$i/
                        shutil.copy(path+'/'+i+'/'+j,'/storage/gabriel/VC/dataset/'+'val/'+i)
                    count+=1

    else:
        c_count=0
        for p in target_name:

            base_num = len(os.listdir(path+p))
            train_ind = np.random.choice(base_num,base_num*7//10,replace=False)
            left_out_ind = np.array([i for i in np.arange(0,base_num) if i not in train_ind])
            val_ind = np.random.choice(left_out_ind,base_num//5,replace=False)
            test_ind = np.array([i for i in left_out_ind if i not in val_ind])
            
            data_dic_ind = {'train' : train_ind,'val' : val_ind, 'test' : test_ind}
            for i in data_dic_ind.keys():
                try:
                    os.makedirs('/storage/gabriel/VC/dataset/'+i)
                    for j in target_name:
                        os.makedirs('/storage/gabriel/VC/dataset/'+i+'/'+j)
       
                except:
                    pass

            for i in os.listdir(path):
                count = 0
                for j in os.listdir(path+i):
                    if(count in data_dic_ind['train']):
                        shutil.copy(path+'/'+i+'/'+j,'/storage/gabriel/VC/dataset/'+'train/'+i+'/'+j)

                    if(count in data_dic_ind['test']):
                        shutil.copy(path+'/'+i+'/'+j,'/storage/gabriel/VC/dataset/'+'test/'+i+'/'+j)

                    if(count in data_dic_ind['val']):
                        shutil.copy(path+'/'+i+'/'+j,'/storage/gabriel/VC/dataset/'+'val/'+i+'/'+j)
                    count+=1





def make_pat_data(data_dir ='/storage/gabriel/VC/Normal_Images/' ,dest_dir ='/storage/gabriel/VC/Normal_Images_New/',eq = True ):
    
    try:
        shutil.rmtree(dest_dir)
        os.makedirs(dest_dir)
    except:
        
        #shutil.rmtree(dest_dir)
        os.makedirs(dest_dir)
        #pass
    
    #list containing patient number
    pat_num1 = []
    count=0
    all_im_name = []
    all_im_dir = []
    for i,j,k in os.walk(data_dir):
        count+=1
        if(len(k)>1):
            
            all_im_name+=k
            for l in k:
                all_im_dir.append(i+'/'+l)
                temp = l.find('EQo_')+4
                temp2 = l[temp:].find('_')+temp
                if(int(l[temp:temp2]) not in pat_num1):

                    pat_num1.append(int(l[temp:temp2]))

    #dictionary of form { patient number : number of frames }
    def find_last_two(s):
        return [k for k,q in enumerate(s) if q=='_'][:2]    
    
    num_class = {i:len(os.listdir(data_dir+i)) for i in os.listdir(data_dir) if '.ipynb' not in i}
    #print(pat_num1)
    pat_num = np.random.permutation(pat_num1)
    #print(pat_num)
    #return
    train_pat = [int(i) for i in pat_num[:len(pat_num)*6//10]]
    print('train  ',train_pat)
    test_pat = [int(i) for i in pat_num[1+ len(pat_num)*6//10:len(pat_num)*8//10]]
    print('test  ',test_pat)
    valid_pat = [int(i) for i in pat_num[1+ len(pat_num)*8//10:]]
    print('valid  ',valid_pat)
    table = {'train':train_pat,'test':test_pat,'val':valid_pat}
    
    for fol in ['train','val','test']:
        
        os.makedirs(dest_dir+'/'+fol)
        for i in num_class:
            os.makedirs(dest_dir+'/'+fol+'/'+i)        
            for l in all_im_dir:
                temp = l[l.find('EQo'):]
                if(i in l and int(temp[find_last_two(temp)[0]+1:find_last_two(temp)[1]]) in table[fol]):
                    shutil.copy(l,dest_dir+'/'+fol+'/'+i)


