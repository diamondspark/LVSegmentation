import matplotlib.pyplot as plt
import numpy as np
import os
def find_stats(path):
	
	im_list = (plt.imread(path+'/Training/'+i).reshape(1,-1) for i in os.listdir(path+'/Training') if not(i=='.') and not(i=='.DS_Store'))
	count = 0
	Ex2 = 0
	mean = 0
	for i in im_list:
#		print(i.max())
#		print(i.mean())

#		break
		mean += i.mean()
		y = i**2
		Ex2 += y.mean()
		count+=1	
	mean/=count
	Ex2/=count
	Ex2-=mean**2
	Ex2 = np.sqrt(Ex2)
	
	print(mean,Ex2)
	
