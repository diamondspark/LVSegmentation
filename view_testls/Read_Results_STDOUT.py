##########Usage############################################

#Dir='/data/Gurpreet/RUNS/VC_12/'
#Fname='Scratch_07062017_0515_0.1.txt'

#Read_Results_STDOUT(Dir,Fname)

#######################################################
def Read_Results_STDOUT(Directory,Filename):
    import pandas as pd
    import numpy as np

    Fname=Directory+Filename
    f = open(Fname)
    Flines=f.readlines()
    lof=len(Flines)
    train_loss=[]
    train_acc=[]
    val_loss=[]
    val_acc=[]

    for i in xrange(0,lof):
        lineval=Flines[i]
        if "traintrain Loss:" in lineval:
            train_loss.append(float(lineval[17:-1]))
            train_acc.append(float(Flines[i+1][5:-1]))
        if "valval Loss:" in lineval:
            val_loss.append(float(lineval[14:-1]))
            val_acc.append(float(Flines[i+1][5:-1]))


    Data = pd.DataFrame(
        {'Train_loss': train_loss,
         'Train_accuracy': train_acc,
         'Val_loss': val_loss,
         'Val_accuracy': val_acc
        })
    xcelname=Directory+Filename[:-4]+".xls"
    writer = pd.ExcelWriter(xcelname, engine='xlsxwriter')
    Data.to_excel(writer, sheet_name='Sheet1')





