import tensorflow as tf
import cv2
import dicom
import pylab

def readImage(path, file_format):
    if(file_format.lower()  == "dicom"):
        ds = dicom.read_file(path)
        return ds
    else:
        img = cv2.imread(path,0)
        return img
##        cv2.imshow('image',img)
##        cv2.waitKey(10)
##        cv2.destroyAllWindows()
    

def displayImage(image):
    pylab.imshow(image, cmap= pylab.cm.bone)
    pylab.show()

image = readImage('/Users/mop2014/Downloads/Training/DET0000101/DET0000101_LA1_ph0.png','png')
print (image)
displayImage(image)
