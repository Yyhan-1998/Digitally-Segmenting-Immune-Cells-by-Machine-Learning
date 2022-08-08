import os
import cv2
import sklearn
import openslide
import numpy as np
from numba import jit
from skimage import io
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def model(patch_path): 
    
    '''
    patch_path: path of the patch that model needs to be fitted.
    '''

    model = KMeans(n_clusters=8, max_iter=20,
                  n_init=3, tol=1e-3)
    fit_patch = plt.imread(patch_path)       
    fit_patch_n = np.float32(fit_patch.reshape((-1, 3))/255.)
    model.fit(fit_patch_n)
    return model

def pred_and_cluster(model, dir_path):
    
    '''
    model: model that will be used to predict the patch.
    dir_path: path of the directory that has patches that needs to be predicted and clustered.
    
    '''
    for file in os.listdir(dir_path):
        
        try:
            os.mkdir(file[:-4])
        except:
            pass
            
        if len(model.cluster_centers_) == 8:
            pass
        else:
            raise ValueError("The model must be trained for 8 clusters")
        
        pred_patch = plt.imread(os.path.join(dir_path, file))
        pred_patch_n = np.float32(pred_patch.reshape((-1, 3))/255.)
        labels = model.predict(pred_patch_n)
        overlay_center = np.copy(model.cluster_centers_)
        back_img = np.uint8(np.copy(pred_patch))
        #Reassigning Overlay colors
        overlay_center[0] = np.array([255, 102, 102])/255. #Light Red
        overlay_center[1] = np.array([153, 255, 51])/255. #Light Green
        overlay_center[2] = np.array([0, 128, 255])/255. #Light Blue
        overlay_center[3] = np.array([0, 255, 255])/255. #Cyan
        overlay_center[4] = np.array([178, 102, 255])/255. #Light Purple
        overlay_center[5] = np.array([95, 95, 95])/255. #Grey
        overlay_center[6] = np.array([102, 0, 0])/255. #Maroon
        overlay_center[7] = np.array([255, 0, 127])/255. #Bright Pink
        #Ovrlaying each and every cluster
        for i in range(len(overlay_center)):
            seg_img = np.copy(pred_patch_n)
            seg_img[labels.flatten() == i] = overlay_center[i] 
            seg_img[labels.flatten() != i] = np.array([255, 255, 255])/255.
            seg_img = seg_img.reshape(pred_patch.shape)
            plt.imsave(os.path.join(file[:-4], 'segmented_'+str(i)+'.jpg'), seg_img, dpi=1000)
            seg_img = np.uint8(seg_img*255.)
            overlay_img = cv2.addWeighted(back_img, 0.4, seg_img, 0.6, 0)/255.
            plt.imsave(os.path.join(file[:-4], '_overlay_'+str(i)+'.jpg'), overlay_img, dpi=1000)
        #Make a complete cluster
        all_cluster = overlay_center[labels.flatten()].reshape(pred_patch.shape)
        plt.imsave(os.path.join(file[:-4], 'all_cluster.jpg'), all_cluster, dpi=1000)
        #Overlaying the complete cluster
        seg_img = np.uint8(np.copy(all_cluster)*255.)
        overlay_img = cv2.addWeighted(back_img, 0.6, seg_img, 0.4, 0)
        plt.imsave(os.path.join(file[:-4], 'full_overlay.jpg'), overlay_img, dpi=1000)

    return None

def remove_background(slide_img, x_tile_size, y_tile_size, color_delta=40):
    
    '''
    slide_img: whole slide image (WSI)
    x_tile_size: width of the desired patch
    y_tile_size: height of the desired patch
    color_delta: distance metric for tolerance between background color and white
    '''
    
    dim = slide_img.shape
    print(slide_img)
    
    if type(x_tile_size) == int and type(y_tile_size) == int:
        pass
    else:
        raise TypeError("The values of x_tile_size and y_tile_size must be type int")
     
    if any([dim[0] < y_tile_size, y_tile_size < 0]):
        raise ValueError("The value of y_tile_size must be a positive integer lower than the height of slide_img")
    else:
        pass
    
    if any([dim[1] < x_tile_size, x_tile_size < 0]):
        raise ValueError("The value of x_tile_size must be a positive integer lower than the width of slide_img")
    else:
        pass
    
    background_pixels = 0
   
    color_delta = 40 # Maxmium difference or distance between background color and white, can be adjusted manually
    
    white = [255, 255, 255]     
    r2, g2, b2 = white[0:3]
    
    # Iterating over image pixels in image and comparing whether they are background color
    for col in range(x_tile_size):
        for row in range(y_tile_size):
            
            # Calculating the distance between the positions of the background color and white in color space
            r1, g1, b1 = int(slide_img[row,col][0]),int(slide_img[row,col][1]),int(slide_img[row,col][2])
            rmean = int((r1 + r2) / 2)
            R = r1 - r2
            G = g1 - g2
            B = b1 - b2            
            color_dist = np.sqrt((((512+rmean)*R*R)>>8) + 4*G*G + (((767-rmean)*B*B)>>8))
            
            if color_dist < color_delta:
                background_pixels += 1
                
    return background_pixels

def extract_svs_img(slide_filename):
    
    '''
    slide_filename: name of the whole slide image in the current working directory 
    '''
    
    # Creating three new folders for output images
    all_images_file_path = os.getcwd() + "\\" + "All_Images"
    os.mkdir(all_images_file_path)
    tissue_images_file_path = os.getcwd() + "\\" + "Tissue_Images"
    os.mkdir(tissue_images_file_path)
    overlay_images_file_path = os.getcwd() + "\\" + "Overlay_Images" 
    os.mkdir(overlay_images_file_path)      
    
    # Opening the svs file and get the width and height of the image
    slide_file = openslide.OpenSlide(slide_filename)
    slide_width, slide_height = slide_file.dimensions 
    
    # Determining the size of the images to be extracted
    x_tile_size = 4000
    y_tile_size = 3000    
    
    slide_img = np.zeros((y_tile_size, x_tile_size, 3), np.uint8)
    
    # Calculating the number of extracted images according to the size of the original image
    x_tile_num = int(np.floor((slide_width - x_tile_size - 1) / (x_tile_size * 0.9))) + 2
    y_tile_num = int(np.floor((slide_height - y_tile_size - 1) / (y_tile_size * 0.9))) + 2        
    
    p_num, t_num = 1,1 #for naming files

    # Left to right and top to bottom
    for iy in range(y_tile_num):
        for ix in range(x_tile_num):
            
            # Coordinates of the upper left corner of each image
            start_x = int(ix * x_tile_size * 0.9) if (ix + 1) < x_tile_num else (slide_width - x_tile_size) 
            start_y = int(iy * y_tile_size * 0.9) if (iy + 1) < y_tile_num else (slide_height - y_tile_size) 
            
            # Reading the image to be extracted
            cur_tile = np.array(slide_file.read_region((start_x, start_y), 0, (x_tile_size, y_tile_size)))
            slide_img = np.array(cur_tile)[:,:,:3]
            
            # Saving as all_images
            slide_savename = os.path.splitext(slide_filename)[0] + '_P{p_num}'.format(p_num = p_num) + '.tif'
            io.imsave(all_images_file_path + "\\" + slide_savename, slide_img)
            
            # Selecting images with a background less than 70% by counting the number of white pixels 
            #and comparing with the total number of pixels
            background_pixels = remove_background(slide_img, x_tile_size, y_tile_size)
                                         
            # If the image background is less than 70%, save the image and call the overlay_image()
            if background_pixels < 0.7 * x_tile_size * y_tile_size:
                
                # Saving as tissue_images
                slide_savename = os.path.splitext(slide_filename)[0] + '_T{t_num}'.format(t_num = t_num) + '.tif'
                io.imsave(tissue_images_file_path + "\\" + slide_savename, slide_img)
            
                t_num += 1
            
            p_num += 1
    
    return None