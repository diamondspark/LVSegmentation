import os
from image import readImage
from scipy.misc import imsave
Path = '/data/gabriel/LVseg/dataset/Validation/'
for i in os.listdir(Path):
	#print(i)
	for j in os.listdir(Path+'/'+i):
		#print(j)
		for k in os.listdir(Path+'/'+i+'/'+j):
			print(Path+'/'+i+'/'+j+'/'+k)
			#break
			if '.dcm' in k:
				I = readImage(Path+'/'+i+'/'+j+'/'+k,'dicom')
				#print(I)
				imsave('/data/gabriel/LVseg/dataset_img/img/Validation/'+'/'+k[:k.find('.')-1]+'.png',I)
