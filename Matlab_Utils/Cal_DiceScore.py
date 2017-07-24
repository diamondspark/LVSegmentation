
# coding: utf-8

# In[1]:

############################### Module to calculate Dice Score ###########################################


#please note: matlab file names Cal_DiceScore must be in the same directory as this python file.

#GTIm=str('/data/Gurpreet/Echo/Normal_Images/1/Images/TEE_1_1_1.jpg')
#SEGIm=str('/data/Gurpreet/Echo/Normal_Images/1/Images/TEE_1_1_30.jpg')
#path=1 #specify the path variabe as 1 if you are passing the images to matlab
#DS=calc_DiceScore(GTIm,SEGIm,path)
#print(DS)

##########################################################################################################

def Cal_DiceScore (GTIm,SEGIm,path):
    import matlab
    import matlab.engine
    eng = matlab.engine.start_matlab()
    import numpy as np
    import cv2
    import time

    if path == 0:
        GTIm=cv2.imread(GTIm)
        SEGIm=cv2.imread(SEGIm)
        GTIm=matlab.double(GTIm)
        SEGIm=matlab.double(SEGIm)

    DiceScore=eng.Cal_DiceScore(GTIm,SEGIm,path,nargout=1)
    return DiceScore

#Cal_DiceScore (GTIm,SEGIm,path)
#GTIm=str('/data/Gurpreet/Echo/Normal_Images/1/Images/TEE_1_1_1.jpg')
#SEGIm=str('/data/Gurpreet/Echo/Normal_Images/1/Images/TEE_1_1_30.jpg')
#path=1 #specify the path variabe as 1 if you are passing the images to matlab
#DS=calc_DiceScore(GTIm,SEGIm,path)
#print(DS)
