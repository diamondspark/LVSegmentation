###############################################################################################################################




##To overlay two Images

#from Overlay_Images import *
#Overlay_Images(2,0.2,'/data/Gurpreet/CODE/LVSegmentation','/data/Gurpreet/CODE/LVSegmentation/19_img.png','/data/Gurpreet/CODE/LVSegmentation/19_label.png','/data/Gurpreet/CODE/LVSegmentation/19.png')

##To overlay one Image

#from Overlay_Images import *
#Overlay_Images(1,0.2,'/data/Gurpreet/CODE/LVSegmentation','/data/Gurpreet/CODE/LVSegmentation/19_img.png','/data/Gurpreet/CODE/LVSegmentation/19_label.png','')

###############################################################################################################################


def Overlay_Images(num_of_overlays,alpha_val,work_path,p_g_image,p_f_overlay,p_s_overlay):
    
    import Image
    import numpy as np
    import cv2
    from PIL import Image

    ground = cv2.imread(p_g_image)
    F_overlay = Image.open(p_f_overlay)
    F_overlay= F_overlay.convert('RGBA')
    F_overlay_data = np.array(F_overlay)
    red, green, blue, alpha = F_overlay_data.T 
    white_areas = (red > 150) & (blue > 150) & (green > 150)
    F_overlay_data[..., :-1][white_areas.T] = (255, 0, 0) 
    F_overlay_im = Image.fromarray(F_overlay_data)
    save_F_overlay=str(work_path)+"/overlay1.jpg"
    F_overlay_im.save(save_F_overlay)
    overlay1 = cv2.imread(save_F_overlay)
    overlay1 = overlay1.copy()
    if num_of_overlays==2:
        S_overlay = Image.open(p_s_overlay)
        S_overlay = S_overlay.convert('RGBA')
        S_overlay_data = np.array(S_overlay)   
        red, green, blue, alpha = S_overlay_data.T
        white_areas = (red > 150) & (blue > 150) & (green > 150)
        S_overlay_data[..., :-1][white_areas.T] = (0, 0, 255)
        S_overlay_im = Image.fromarray(S_overlay_data)
        save_S_overlay=str(work_path)+"/overlay2.jpg"
        S_overlay_im.save(save_S_overlay)
        overlay2 = cv2.imread(save_S_overlay)
        overlay2 = overlay2.copy()
    alpha=alpha_val
    output =ground.copy()
    cv2.addWeighted(overlay1,alpha,output,1,1,output)
    if num_of_overlays==2:
        cv2.addWeighted(overlay2,alpha,output,1,1,output)
    final_frame = cv2.hconcat((ground, output))
    save_output=str(work_path)+"/output.jpg"
    print save_output
    cv2.imwrite(save_output,final_frame)


# In[ ]:



