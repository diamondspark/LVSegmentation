from image import Image
from data import Data
import glob
import tensorflow as tf
from image import Image
import cv2

imageList_dcm=[]
imageList_png=[]

def LoadFromDir(path):
    for filename in glob.glob(str(path)+'/*.png'):
         im = Image(filename,'png')
         imageList_png.append(im)
    print "Loaded all png"
    for filename in glob.glob(str(path)+'/*.dcm'):
         im = Image(filename,'dicom')
         imageList_dcm.append(im)
         
#Load From Directory
#LoadFromDir('/Users/mop2014/Desktop/_Mohit/My Garbage/img_256')
LoadFromDir('/data/gabriel/LVseg/dataset_img/img_256')
images = imageList_png
imageList_png=[]
#LoadFromDir('/Users/mop2014/Desktop/_Mohit/My Garbage/box_256')
LoadFromDir('/data/gabriel/LVseg/dataset_img/box_256')
labels = imageList_png

#Make Data
x=[]
y=[]
for i in range(0,len(images)):
    x.append(images[i].resize(64,64).img)
    y.append(labels[i].resize(32,32).img)
    


## Batching
#b= data.batch(3)
##############################################################
#Labels preprocessing
##thresh_label=[]
##for i in range (0, len(y)):
##    ret, th1 = cv2.threshold(y[i].img,0.5,1,cv2.THRESH_BINARY)
##    thresh_label.append(Image('','',th1))

data = Data(x,y)
