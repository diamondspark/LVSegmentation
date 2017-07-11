
#-------------------------Usage--------------------------------------------------------------------------
#import sys
#sys.path.append('/data/Gurpreet/CODE/VC2/view_testls/')
#from Train_Resnet_Sets import Train_Resnet_Sets



#gpunum=1 #Change this number if encounter cuda error. or change the batch size
#pretrain=True  #Choose True for pretrained model and False to train model from sctratch
#mainfolder="/data/Gurpreet/RUNS/VC_12/"
#number_of_sets=10
#number_of_views=5
#lr_decay=10
#batch_size=100
#epochs=20
#Identifier="30062017_0425"
#lr=0.001
#Train_Resnet_Sets(gpunum,pretrain,mainfolder,number_of_sets,number_of_views,lr,lr_decay,batch_size,epochs,Identifier)

#--------------------------------------------------------------------------------------------------------

def Train_Resnet_Sets(gpunum,pretrain,mainfolder,number_of_sets,number_of_views,lr,lr_decay,batch_size,epochs,Identifier):
    from torchvision import datasets,models,transforms
    import torch.nn as nn
    from nndev import model_pip
    from nndev import model_pip
    from Conmat_TPR import calculate_APR as CAPR
    #from IPython.display import clear_output
    import numpy as np
    
    #-------------------------------------------------------------------------------------------------------#
    #print pretrain
    temp_Pr=np.zeros((number_of_views,1))
    temp_Re=np.zeros((number_of_views,1))
    Total_Accuracy=np.zeros((number_of_sets,1))
    for j in range(0,number_of_sets):
        k=j+1
        setnumber="SET"+str(k)
        print "============================================================================================"
        print "=============================="+setnumber+"=========================================================="
        
        if (pretrain.lower()=="true"):
            #print "inside true"
            pretrained_text="pretrained"
            res = models.resnet18(pretrained=True)
        else :
            pretrained_text="scratch"
            #print "inside false"
            res = models.resnet18(pretrained=False)
        
        Model_name="ResNetModel_"+str(pretrained_text)+"_"+str(number_of_views)+"_views_bs_"+str(batch_size)+"_e_"+str(epochs)+"_"+str(Identifier)+".pth.tar"
        datapath=mainfolder+setnumber+"/dataset/"
        savepath=mainfolder+setnumber+"/"+Model_name
        
        res.fc = nn.Linear(res.fc.in_features,number_of_views)
        obj = model_pip(model_in=res,scale=True,batch_size=batch_size,use_gpu=True,gpu=gpunum,data_path=datapath,lr=lr,lr_decay_epoch=lr_decay,verbose=True)
        model = obj.train_model(epochs=epochs)
        obj.store_model(f_name=savepath)
        print ("==============================Testing Started===============================================")
        mdir=mainfolder+str(setnumber)+"/"
        del (obj)
        obj = model_pip(model_in=res,scale=True,batch_size=batch_size,use_gpu=True,gpu=gpunum,data_path=datapath,lr=lr,lr_decay_epoch=lr_decay)
        #obj.test(self,model_dir,model_name,random_crop,test_on,n='None',save_miscl = False,folder = 'misc')
                
        obj.load_model(savepath)
        obj.test(model_dir=mdir,model_name=Model_name,save_miscl = True,test_on=True,n='None',random_crop=False,folder='misc')
        del (obj)
        
        
        folderpath=mainfolder+setnumber+"/misc/"+Model_name[:-8].strip().replace(" ","")+"/"+Model_name+"c_mat.txt"
        Confusion_mat=np.loadtxt(folderpath)
        TA,Pr,Re=CAPR(Confusion_mat)
        temp_Pr=np.concatenate((temp_Pr,Pr),axis=1)
        temp_Re=np.concatenate((temp_Re,Re),axis=1)
        Total_Accuracy[j,0]=TA
        print (temp_Pr.shape)
        print (Pr.shape)
        save_file=mainfolder+setnumber+"/Confusion_matrix"+Identifier+".txt"
        np.savetxt(save_file,Confusion_mat)

        save_file=mainfolder+setnumber+"/Precision"+Identifier+".txt"
        np.savetxt(save_file,Pr)

        save_file=mainfolder+setnumber+"/Recall"+Identifier+".txt"
        np.savetxt(save_file,Re)

        #clear_output()

    Precision=np.delete(temp_Pr, 0, 1)
    Recall=np.delete(temp_Re, 0, 1)
    Total_Accuracy=np.nan_to_num(Total_Accuracy)
    Precision=np.nan_to_num(Precision)
    Recall=np.nan_to_num(Recall)
    print (Total_Accuracy)
    Total_Accuracy_mean=np.average(Total_Accuracy,axis=0)
    Precision_mean=np.average(Precision,axis=1)
    Recall_mean=np.average(Recall,axis=1)

    Total_Accuracy_sd=np.std(Total_Accuracy,axis=0)
    Precision_sd=np.std(Precision,axis=1)
    Recall_sd=np.std(Recall,axis=1)

    save_file=mainfolder+setnumber+"TAM"+Identifier+".txt"
    np.savetxt(save_file,Total_Accuracy_mean)

    save_file=mainfolder+setnumber+"PRM"+Identifier+".txt"
    np.savetxt(save_file,Precision_mean)

    save_file=mainfolder+setnumber+"REM"+Identifier+".txt"
    np.savetxt(save_file,Recall_mean)

    save_file=mainfolder+setnumber+"TASD"+Identifier+".txt"
    np.savetxt(save_file,Total_Accuracy_sd)

    save_file=mainfolder+setnumber+"PRSD"+Identifier+".txt"
    np.savetxt(save_file,Precision_sd)

    save_file=mainfolder+setnumber+"RESD"+Identifier+".txt"
    np.savetxt(save_file,Recall_sd)

