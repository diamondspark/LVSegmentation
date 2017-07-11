from image import Image
import glob

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

LoadFromDir('/Users/mop2014/Downloads/Training/DET0000101')
