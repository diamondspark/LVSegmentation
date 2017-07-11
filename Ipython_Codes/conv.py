import matplotlib.pyplot as plt
import numpy as np
import scipy.io as io

def get_echo(img,company,R,C):
    #thresh = to(I)
    
    #binary = I>thresh
    
    #### img = the image object in grayscale 
    #### company = company tag in dicom file
    #### R = height
    #### C = width
    
    dic_m = {}
    
    dic_m["'SIEMENS' 1024 768"] = [[30,474],[469,0],[673,0],[673,1024],[469,1023],[30,551]]

    '/data/Gurpreet/Echo/Images/35/Images/TEE_35_24_1.jpg'
    dic_m["'INFINITT' 967 872"] = [[3,319],[306,21],[398,201],[415,380],[291,613],[3,319]]

    dic_m["'GEMS Ultrasound' 636 422"] = [[0,316],[311,0],[411,186],[421,306],[421,586],[304,629],[0,316]]

    dic_m["'Philips Medical Systems' 800 600"]=[[98,426],[560,6],[568,636],[432,758],[426,98]]

    '/data/Gurpreet/Echo/Images/35/Images/TEE_35_24_1.jpg'
    dic_m["'Philips Medical Systems' 1024 768"]=[[99,429],[433,100],[600,322],[599]]

    '/data/Gurpreet/Echo/Sorted_Images/A2C/EQo_78_42_1.jpg'
    dic_m["'GE Vingmed Ultrasound' 636 434"] = [[7,319],[322,13],[401,241],[401,582],[7,577]]

    dic_m["'GEMS Ultrasound' 636 434"] = [[10,320],[320,10],[402,162],[400,600],[124,578],[10,320]]

    dic_m["'NeXus-Community Medical Picture DEPT' 636 436"] = [[10,320],[320,10],[402,162],[400,600],[124,578],[10,320]]

    dic_m["'GE Healthcare Ultrasound' 1016 708"] = [[66,510],[505,84],[650,313],[663,524],[574,985],[60,510]]

    dic_m["'INFINITT' 966 873"] = [[112,482],[554,44],[721,350],[721,908],[480,908],[112,482]]

    dic_m["'TOSHIBA_MEC_US' 960 720"] = [[88,480],[477,100],[619,479],[465,867],[88,480]]

    '''
    Cannot process --

    'TOSHIBA_MEC' 512 512
    [["'48'", "'1'"], ["'48'", "'2'"], ["'48'", "'3'"], ["'48'", "'4'"], ["'48'", "'5'"], ["'48'", "'6'"], ["'48'", "'7'"], ["'48'", "'8'"], ["'48'", "'9'"]]

    "INFINITT' 967 832"] = [["'8'", "'46'"], ["'15'", "'59'"], ["'24'", "'76'"], ["'39'", "'39'"], ["'68'", "'24'"], ["'92'", "'34'"]]

    "'INFINITT' 1603 928"] = [["'14'", "'45'"], ["'14'", "'48'"], ["'14'", "'49'"], ["'14'", "'50'"], ["'14'", "'51'"], ["'28'", "'3'"], ["'28'", "'4'"], ["'93'", "'49'"]]

    "'INFINITT' 967 834"] = [[62,455],[526,50],[704,312],[622,950],[525,939],[62,455]]

    'INFINITT' 1024 1024
    [["'48'", "'10'"]]

    'GEMS Ultrasound' 640 458
    [["'69'", "'73'"], ["'69'", "'74'"], ["'77'", "'61'"], ["'77'", "'62'"], ["'77'", "'63'"]]

    'GE Vingmed Ultrasound' 640 480
    [["'73'", "'59'"], ["'73'", "'60'"]]

    'INFINITT' 967 808
    [["'72'", "'89'"], ["'72'", "'91'"]]

    '''
    if (company.lower()=='siemens' and R == 1024 and C == 768):
        coords = np.array([np.arange(469,30,-1),np.arange(30,469)]).T
        for i in coords:
            #print(i)
            img[:i[0],:i[1]]=0
        
        img[674:,:997]=0
        
        coords1 = np.array([np.arange(36,469),np.arange(561,993)]).T
        #print(coords1.shape)
#         for i in coords1:
#             #print(coords1[i])
#             #break
#             print(i)
#             img[0:i[0],i[1]]=0
        img[:258,755:] = 0
        plt.imshow(img),plt.show()
    
    elif(company.lower()=='gems ultrasound' and R == 636 and C == 422):
        coords = np.array([np.arange(6,316),np.arange(316,6,-1)]).T
        for i in coords:
            #print(i)
            img[:i[0],:i[1]]=0
       
        
        coords = np.array([np.arange(350,420),np.arange(316,6,-1)]).T
        
        img[387:,:170]=0
        img[390:,603:]=0
        img[8:173,602:]=0
        
        
    elif(company.lower()=='philips medical systems' and R==800 and C ==600):
        img[:130,0:900]=0
        #print('here')
        img[:,:94]=0
        img[61:191,:220] = 0
        img[:93,:431] = 0
        img[:380,746:,] = 0
        img[538:,679:] = 0
        img[555:,:]=0
    
    
    elif(company.lower()=='philips medical systems'and R== 1024 and C== 768):
        img[:133,:] = 0
        img[:280,:280] = 0
        img[:,:117] = 0
        ekg = img[716:,:]
        img[716:,:] = 0
        img[:345,770:] = 0
        
   
    
    elif(company.lower()=='ge vingmed ultrasound' and R == 636  and C == 434):
        img[:8,:] = 0
        img[0:42,:95]=0
        img[:63,:131] = 0
        img[:10,:] = 0
        img[:207,592:] = 0
        ekg = img[395:,:]
        img[393:,:] = 0
        
    #io.imsave('/data/gabriel/orig116.png',binary)
    
    elif company.lower()=='gems ultrasound' and R == 636 and C == 434:
        img[:8,:] = 0
        img[0:42,:95]=0
        img[:63,:131] = 0
        img[:10,:] = 0
        img[:207,592:] = 0
        ekg = img[395:,:]
        img[393:,:] = 0
    
    
    
    elif company.lower()=='ge healthcare ultrasound' and R == 1016 and C == 708 :
        img[:148,:259] = 0
        ekg = img[650:,:]
        img[650:,:] = 0
        img[:,915:] = 0
    
    
    
    elif company.upper()=='TOSHIBA_MEC_US' and R == 960 and C == 720:
        ekg = img[586:,:]
        img[586:,:] = 0
        img[:,:123]=0
        imf = np.fliplr(img)
        
        coords = np.array([np.arange(88,480),np.arange(480,88,-1)]).T
        for i in coords:
            #print(i)
            img[:i[0],:i[1]]=0
            imf[:i[0],:i[1]]=0
        
        #img=imf
        #img[674:,:997]=0
        
    #binary = img_as_uint(img)
    plt.imshow(img),plt.show()
    io.imsave('/data/gabriel/imgtemp.png',img)
    return img
