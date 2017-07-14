from image import readImage
import os

src = '/data/gabriel/LVseg/dataset/'
import scipy.misc


for i in os.listdir(src):
	
	for j in os.listdir(src+i):
		for k in os.listdir(src+i+'/'+j):

			if '.dcm' in k:
				I = readImage(src+i+'/'+j+'/'+k,'dicom')
	
				scipy.misc.imsave('/data/gabriel/LVseg/dataset_img/img/'+i+'/'+k[:k.find('.')-1]+'.png',I)
			elif '.png' in k:
				I = readImage(src+i+'/'+j+'/'+k,'png')
	
				scipy.misc.imsave('/data/gabriel/LVseg/dataset_img/label/'+i+'/'+k,I)
							

