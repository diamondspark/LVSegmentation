import cv2
import dicom
import pylab

def readImage(path, file_format):
    if(file_format.lower()  == "dicom"):
        ds = dicom.read_file(path)
        return ds.pixel_array
    else:
        img = cv2.imread(path,0)
        return img

class Image:

    def __init__(self, path, fformat, img=None):
        self.path = path
        self.fformat = fformat
        if img is None:
            self.img = readImage(path,fformat)
        else:
            self.img = img
            
    def display(self):
        pylab.imshow(self.img, cmap= pylab.cm.bone)
        pylab.show()
        
    def resize(self, width, height):
        return Image('','',cv2.resize(self.img,(width,height)))
    
