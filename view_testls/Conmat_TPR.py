
# coding: utf-8

# In[ ]:

def calculate_APR(Confusion_mat):

    import numpy as np
    Total_accuracy=((np.trace(Confusion_mat))/(Confusion_mat.sum()))*100
    Precision=np.zeros((len(Confusion_mat),1))
    Recall=np.zeros((len(Confusion_mat),1))
    runs=len(Confusion_mat)
    for i in xrange (0,runs):
        Precision[i,0]=(Confusion_mat[i,i])/(Confusion_mat[:,i].sum())
        Recall[i,0]=(Confusion_mat[i,i])/(Confusion_mat[i,:].sum())
    return Total_accuracy,Precision,Recall

