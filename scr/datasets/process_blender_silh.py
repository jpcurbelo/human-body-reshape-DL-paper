import skimage
from skimage.color import rgb2gray
import cv2 as cv
import numpy as np

import sys
sys.path.append("..")
from utils import *

SIL_FILES_DIR_bw = os.path.join(DS_DIR , f"silhouettes_blender{IMG_SIZE_ORIG}_bw")
try:
    os.mkdir(SIL_FILES_DIR_bw) 
except OSError as error: 
    print(error)  

SIL_FILES_DIR_bw_rsz = os.path.join(DS_DIR , f"silhouettes_blender{IMG_SIZE_4NN}_bw")
try:
    os.mkdir(SIL_FILES_DIR_bw_rsz) 
except OSError as error: 
    print(error)  

SIL_FILES_DIR_npy = os.path.join(DS_DIR , f"silhouettes_blender{IMG_SIZE_4NN}_npy")
try:
    os.mkdir(SIL_FILES_DIR_npy) 
except OSError as error: 
    print(error)  


def main():

    for ds_name in ['SPRING'] + DATASETS:

        blender_files_dir = SIL_FILES_DIR_DICT [ds_name]

        # UNCOMMENT TO CREATE THE SILHOUETTES!!!
        # process_silhouettes(ds_name, blender_files_dir)

        # UNCOMMENT TO CREATE THE NPZ FILES!!!
        ## Save images to npz arrays
        imgs2npz(ds_name)

        # # ## Prepare csv files with measurements info
        # # prepare_measDB_files(ds_name)


def process_silhouettes(ds_name, blender_files_dir):


    silhoDB_files_dir = os.path.join(SIL_FILES_DIR_bw, f"silhouettes_{ds_name}_bw")
    try:
        os.mkdir(silhoDB_files_dir) 
    except OSError as error: 
        print(error)  

    silhoDB_files_dir_rsz = os.path.join(SIL_FILES_DIR_bw_rsz, f"silhouettes_{ds_name}_bw")
    try:
        os.mkdir(silhoDB_files_dir_rsz) 
    except OSError as error: 
        print(error)  

    for gender in GENDERS:
        dir_gender_src = os.path.join(blender_files_dir, gender)
        dir_gender_out = os.path.join(silhoDB_files_dir, gender)
        dir_gender_out_rsz = os.path.join(silhoDB_files_dir_rsz, gender)
    
        try: 
            os.mkdir(dir_gender_out) 
        except OSError as error: 
            print(error)  

        try: 
            os.mkdir(dir_gender_out_rsz) 
        except OSError as error: 
            print(error)  

        for view in VIEWS:
            dir_view_src = os.path.join(dir_gender_src, view)
            dir_view_out = os.path.join(dir_gender_out, view)
            dir_view_out_rsz = os.path.join(dir_gender_out_rsz, view)

            try: 
                os.mkdir(dir_view_out) 
            except OSError as error: 
                print(error) 

            try: 
                os.mkdir(dir_view_out_rsz ) 
            except OSError as error: 
                print(error) 

            ## UNCOMMENT TO CREATE THE SILHOUETTES!!!
            # createDBsilhouettes(dir_view_src, dir_view_out)
            # resizeDBsilhouettes(dir_view_out, dir_view_out_rsz)


def createDBsilhouettes(src_dir, files_dir):
        '''
        '''   
        
        file_list = sorted(os.listdir(src_dir))
        for model in file_list:

            mod_split = model.split('_')
            if len(mod_split) == 2:  ##is SPRING
                ID = mod_split[0][-4:] + '_' + mod_split[1]
                fname = 'silh_' + ID
            else:                    ##is ANSUR
                ID = mod_split[1:]
                ID = '_'.join(ID)
                fname = 'silh_' + ID

            # Load the image
            img = skimage.io.imread(os.path.join(src_dir, model))[:,:,:3]  # [:,:,:3] to remove alpha channel
            
            img = rgb2gray(img)
            img= skimage.transform.rotate(img, -90)

            # Obtain the optimal threshold value
            thresh = skimage.filters.threshold_mean(img)     #threshold_otsu(img)  threshold_mean(img)  
            # Apply thresholding to the image
            binary_global = img > thresh

            silpath = os.path.join(files_dir, fname)
            print(silpath)
            skimage.io.imsave(silpath, binary_global) 


def resizeDBsilhouettes(files_dir, files_dir_rsz):
    '''
    '''
    file_list = sorted(os.listdir(files_dir))

    print(files_dir_rsz)
    for model in file_list:
        ID = model.split('_')[1:]
        ID = '_'.join(ID)
        fname = 'silh_' + ID

        img = cv.imread(os.path.join(files_dir, model))
            
        # resize image
        img_size = (WIDTH, HEIGHT)
        img = cv.resize(img, img_size)

        cv.imwrite(os.path.join(files_dir_rsz, fname), img)


def imgs2npz(ds_name):
    '''
    '''

    silhoDB_src = os.path.join(SIL_FILES_DIR_bw_rsz, f"silhouettes_{ds_name}_bw")

    npz_files_dir = os.path.join(SIL_FILES_DIR_npy, f"silhouettes_{ds_name}_bw")
    try:
        os.mkdir(npz_files_dir) 
    except OSError as error: 
        print(error)  

    imgs_view_dir = list()

    # imgs_view_dir.append(os.path.join(silhoDB_src, GENDERS[0]))
    # imgs_view_dir.append(os.path.join(silhoDB_src, GENDERS[1]))

    for gender in GENDERS:

        img_dir = os.path.join(silhoDB_src,gender)
        img_data = load_imgs(img_dir)

        if TEST_FILES == True:
            npz_file_name = f"silh_Xarray{IMG_SIZE_4NN}_{ds_name}_{gender}_bw_{TEST_FILES_NUM}test.npz"
        else:
            npz_file_name = f"silh_Xarray{IMG_SIZE_4NN}_{ds_name}_{gender}_bw.npz"
            
        np.savez_compressed(os.path.join(npz_files_dir, npz_file_name), img_data)


def load_imgs(files_dir):
    '''
    files_dir is a list of string with the directory to the 
    silhouettes files (front and side)
    returns a list with front and side views [front, side]
    '''
    img_data = list()

    for view in VIEWS:

        view_dir = os.path.join(files_dir, view)
        files_list = sorted(os.listdir(view_dir))

        # initialize our list of input images
        inputImages = []

        if TEST_FILES == True:
            enum_files_list = enumerate(files_list[:TEST_FILES_NUM])
        else:
            enum_files_list = enumerate(files_list)
        
        for i, file in enum_files_list:
            
            img = cv.imread(os.path.join(view_dir, file), cv.IMREAD_GRAYSCALE)
            img = img.reshape(IMG_SIZE_4NN, IMG_SIZE_4NN, CHAN)
            
            inputImages.append(np.array(img).astype(float) / 255)

        img_data.append(inputImages)

    npy_imgs = np.array(img_data)

    return npy_imgs


# # def prepare_measDB_files(ds_name):

# #     silhoDB_src = os.path.join(SIL_FILES_DIR_bw_rsz, f"silhouettes_{ds_name}_bw")

# #     csv_files_dir = os.path.join(SIL_FILES_DIR_npy, f"silhouettes_{ds_name}_bw")

# ===========================================================================    

if __name__ == "__main__":
  
   main()