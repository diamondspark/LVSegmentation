
# coding: utf-8

# In[ ]:

import sys, ast, getopt, types
total = len(sys.argv)
cmdargs = str(sys.argv)
IMG=eval(sys.argv[1])
MANUFACTURER=sys.argv[2]
WIDTH=sys.argv[3]
HEIGHT=sys.argv[4]
SAVE_ADDRESS=sys.argv[5]


# In[1]:

def get_echo(img,company,R,C,save_file):
    import matplotlib.pyplot as plt
    import numpy as np
    import skimage.io as io
    import dicom
    
    dic_m = {}
    
    dic_m["'GEMS Ultrasound' 636 422"] = [[0,316],[311,0],[411,186],[421,306],[421,586],[304,629],[0,316]]
    dic_m["'GEMS Ultrasound' 636 434"] = [[10,320],[320,10],[402,162],[400,600],[124,578],[10,320]]
    dic_m["'GE Vingmed Ultrasound' 636 434"] = [[7,319],[322,13],[401,241],[401,582],[7,577]]
    dic_m["'Philips Medical Systems' 800 600"]=[[98,426],[560,6],[568,636],[432,758],[426,98]]
    
    dic_m["'GE Healthcare Ultrasound' 1016 708"] = [[66,510],[505,84],[650,313],[663,524],[574,985],[60,510]]
    dic_m["'SIEMENS' 1024 768"] = [[30,474],[469,0],[673,0],[673,1024],[469,1023],[30,551]]
    
    if(company.lower()=='gems ultrasound' and R == 636 and C == 422):
        coords = np.array([np.arange(6,316),np.arange(316,6,-1)]).T
        for i in coords:
            #print(i)
            img[:i[0],:i[1]]=0

        coords = np.array([np.arange(350,420),np.arange(316,6,-1)]).T
        
        img[387:,:170]=0
        img[390:,603:]=0
        img[8:173,602:]=0
        
    elif (company.lower()=='siemens' and R == 1024 and C == 768):
        coords = np.array([np.arange(469,30,-1),np.arange(30,469)]).T
        for i in coords:
            img[:i[0],:i[1]]=0
        img[674:,:997]=0
        coords1 = np.array([np.arange(36,469),np.arange(561,993)]).T
        img[:258,755:] = 0
    
    elif company.lower()=='ge healthcare ultrasound' and R == 1016 and C == 708 :
        img[:148,:259] = 0
        ekg = img[650:,:]
        img[650:,:] = 0
        img[:,915:] = 0
        
    elif company.lower()=='gems ultrasound' and R == 636 and C == 434:
        img[:8,:] = 0
        img[0:42,:95]=0
        img[:63,:131] = 0
        img[:10,:] = 0
        img[:207,592:] = 0
        ekg = img[395:,:]
        img[393:,:] = 0
        
    elif(company.lower()=='ge vingmed ultrasound' and R == 636  and C == 434):
        img[:8,:] = 0
        img[0:42,:95]=0
        img[:63,:131] = 0
        img[:10,:] = 0
        img[:207,592:] = 0
        ekg = img[395:,:]
        img[393:,:] = 0
        
    elif(company.lower()=='philips medical systems' and R==800 and C ==600):
        img[:130,0:900]=0
        #print('here')
        img[:,:94]=0
        img[61:191,:220] = 0        
        img[:93,:431] = 0
        img[:380,746:,] = 0
        img[538:,679:] = 0
        img[555:,:]=0
    
    #imga = plt.imshow(img, interpolation='nearest',aspect='auto',cmap='gray')
    #plt.axis('off')
    #plt.tight_layout()
    #plt.savefig('/data/Gurpreet/VC/imgtemp.jpg', bbox_inches='tight')
    io.imsave(save_file,img)
    return img


# In[2]:

IM=get_echo(IMG,MANUFACTURER,WIDTH,HEIGHT,SAVE_ADDRESS)

