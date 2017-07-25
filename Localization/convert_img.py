from image import readImage
import os

src = '/data/gabriel/LVseg/dataset/Training'
import scipy.misc


#for i in os.listdir(src):
#	if(i=='Training'):

for j in os.listdir(src):
	for k in os.listdir(src+'/'+j):
		if '.dcm' in k:
			I = readImage(src+'/'+j+'/'+k,'dicom')
	
			scipy.misc.imsave('/data/gabriel/LVseg/dataset_img/img/'+'/Training/'+k[:k.find('.')-1]+'.png',I)
		elif '.png' in k:
			I = readImage(src+'/'+j+'/'+k,'png')

			scipy.misc.imsave('/data/gabriel/LVseg/dataset_img/label/'+'Training/'+k,I)
						
