import tensorflow as tf
import cv2


def readImage(path):
    img = cv2.imread(path,0)
    #return img
