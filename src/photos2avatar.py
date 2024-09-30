# Extracts the silhouettes from input images (front and side views) and creates the avatar (obj file)

import os
import sys
import time
import random
import logging

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
import cv2 as cv
import PIL
import torch
from torchvision import transforms
from skimage import filters
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import gc

# Check if any GPUs are available
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if not gpus:
    # No GPUs available, force TensorFlow to use CPU only
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

from utils import ( 
    INPUT_FILES_DIR,
    FILE_ENCODING,
    DS_DIR,
    SCALER,
    MODEL_FILES_DIR,
    CONTINUOUS,
    GENDER_DICT,
    VIEWS,
    OUTPUT_FILES_DIR,
    IMG_SIZE_4NN,
    UK_MEAS,
    M_NUM,
)

# Add the parent directory to sys.path so we can use absolute imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Import Avatar class containing the 3D avatar generation methods (Reshaper and Imputer modules)
from reshaper.avatar import Avatar

# Constants
IMG_RESIZE = 448  # 512   #512   # 256
MODEL_NAME = "extractor_nn_model.h5"
NUM_AUG_INPUT = 20

#########################################################################################
def load_model(model_name):
    '''
    Loads a previously trained model.
    '''
    try:
        return keras.models.load_model(os.path.join(MODEL_FILES_DIR, model_name))
    except FileNotFoundError as file_error:
        print(f"Error: File not found - {file_error.filename}")
        sys.exit(1)
        
def create_measurements_array(extracted_measurements, weightkg_glob):
    '''
    Creates an array with the measurements to be used in the 3D avatar.
    The indexes in the array are to match the measurements indexes in utils.MEASUREMENTS
    '''
    measurements = np.zeros(M_NUM).transpose()
    measurements[0] = weightkg_glob
    measurements[1] = extracted_measurements[0]
    measurements[3] = extracted_measurements[1]
    measurements[4] = extracted_measurements[2]
    measurements[5] = extracted_measurements[3]
    measurements[7] = extracted_measurements[4]
    measurements[14] = extracted_measurements[5]
    measurements[17] = extracted_measurements[6]
    measurements[6] = extracted_measurements[7]
    measurements[16] = extracted_measurements[8]
    
    return measurements

def get_input_data():
    """Reads and processes input data.
    - Two images (front and side view) and extract the silhouettes.
    - Basic info: gender, height, and weight.

    Args:
        No arguments

    Returns:
        basic_info: a np array with 'gender', 'height', and 'weight' values (encoded and scaled)
        input_images (list): a list with two PIL images for the front and side view input images
        df_total: a DataFrame containing the data from SPRING, ANSUR I and II.
    """

    print("[1] Starting to load input data (2 photos and basic info)")
    start = time.time()

    ## Read the basic info (gender, height, weight)
    basic_info, df_total = import_basic_info()

    ## Read and process the input images
    input_images = load_input_images()

    print(f"[1] Finished loading input data (2 photos and basic info) in {(time.time() - start):.1f} s")
    return basic_info, input_images, df_total

def import_basic_info():
    """Imports input data (gender, height, weight)

    Args:
        No arguments

    Returns:
        basic_info: a np array with 'gender', 'height', and 'weight' values (encoded and scaled)
        df_total: a DataFrame containing the data from SPRING, ANSUR I and II.
    """

    input_db_dir = os.path.join(INPUT_FILES_DIR, "input_info_extractor.csv")
    try:
        input_df = pd.read_csv(input_db_dir, encoding=FILE_ENCODING)

        ###Save global variables not scaled
        global input_gender_glob
        input_gender_glob = input_df.iloc[0]["gender"]

    except FileNotFoundError:
        # The file does not exist, so we need to handle the exception
        print(
            f"Wrong file or file path --> '{input_db_dir}' file not found."
        )

    input_df = input_df.replace("male", 1)
    input_df = input_df.replace("female", 0)

    ## SCALE and gender to Categorical
    ## Load databases
    tot_db_dir = os.path.join(DS_DIR, f"measurements_TOTAL_{input_gender_glob}.csv")

    try:
        # Open the file for reading
        df_total = pd.read_csv(
            tot_db_dir, encoding=FILE_ENCODING, converters={"ID": str}
        )

    except FileNotFoundError:
        # The file does not exist, so we need to handle the exception
        print(f"Error: File not found - '{tot_db_dir}'. Please check the file path.")
        print("Tip: You may want to run ds_measurements_filter.py to filter the dataset.")
        # Optionally, you can raise the exception again to propagate it up the call stack
        raise

    except pd.errors.EmptyDataError:
        # Handle the case where the file exists but is empty
        print(f"Error: The file '{tot_db_dir}' is empty. Please check the file content.")

    except pd.errors.ParserError as parse_error:
        # Handle parsing errors (e.g., malformed CSV)
        print(f"Error: Unable to parse '{tot_db_dir}'. Reason: {parse_error}")

    except Exception as e:
        # Catch other unexpected errors
        print(f"Unexpected error: {e}")


    ## Load scaler
    if isinstance(SCALER, StandardScaler):
        tot_scaler_dir = os.path.join(MODEL_FILES_DIR, "scalerStd_img2ava.pkl")
    else:
        tot_scaler_dir = os.path.join(MODEL_FILES_DIR, "scalerMinMax_img2ava.pkl")

    try:
        # Open the file for reading
        data_measX_scaler = joblib.load(tot_scaler_dir)

    except FileNotFoundError:
        # The file does not exist
        logging.warning(f"Warning: Scaler file not found - '{tot_scaler_dir}'. Please check the file path.")
        data_measX_scaler = None

    except (joblib.exc.JoblibValueError, Exception) as e:
        # Handle other potential errors during loading
        logging.error(f"Error loading scaler file '{tot_scaler_dir}': {e}")
        data_measX_scaler = None

    if data_measX_scaler is None:
        ## Scales the data using the train/test split
        data_measX_scaler = scale_db_values()
        joblib.dump(data_measX_scaler, tot_scaler_dir)

    ###Save global variables not scaled
    global stature_glob
    stature_glob = input_df.iloc[0]["stature_cm"]

    global weightkg_glob
    weightkg_glob = input_df.iloc[0]["weight_kg"]

    # Scale the continuous values in the DF
    aux_df = data_measX_scaler.transform(input_df[CONTINUOUS])

    # Encode the categorical values in the DF
    gender_aux = np.array(input_df["gender"])
    gender_aux = gender_aux.reshape(gender_aux.shape[0], -1)
    gender_cat = keras.utils.to_categorical(gender_aux, len(GENDER_DICT.keys()))

    # Join the categorical and continuous values to be used as input data
    input_dataX = np.hstack([gender_cat, aux_df])

    return input_dataX, df_total

def scale_db_values():
    """
    Scale the continuous values in the DataFrame using the specified scaler.

    Args:
        df (DataFrame): The input DataFrame containing data to be scaled.

    Returns:
        StandardScaler: The fitted StandardScaler instance used for scaling.
    """
    
    tot_db_dir_fem = os.path.join(DS_DIR, "measurements_TOTAL_female.csv")
    tot_db_dir_mal = os.path.join(DS_DIR, "measurements_TOTAL_male.csv")

    # Open the file for reading
    df_fem = pd.read_csv(
        tot_db_dir_fem, encoding=FILE_ENCODING, converters={"ID": str}
    )
    df_mal = pd.read_csv(
        tot_db_dir_mal, encoding=FILE_ENCODING, converters={"ID": str}
    )
    df = pd.concat([df_fem, df_mal], ignore_index=True)

    # Split data into train and test sets
    (train_data, _) = train_test_split(
        df, train_size=0.8, shuffle=True, random_state=123
    )

    # Create and fit the scaler
    scaler = StandardScaler()
    scaler.fit(train_data[CONTINUOUS])

    return scaler

def load_input_images():
    """Imports input images (front and side views) and reduces them if needed.

    Args:
        No arguments

    Returns:
        input_images: a list with two PIL images for the front and side view input images
    """

    img_list = list()
    for view in VIEWS:

        filename = f"input_{view}.png"

        ## Load input image (as np array)
        img = cv.imread(os.path.join(INPUT_FILES_DIR, filename), cv.IMREAD_UNCHANGED)

        ## Resize image
        if img.shape[0] > 2048:
            scale_factor = 2048 / img.shape[0]
            width = int(img.shape[1] * scale_factor)
            height = int(img.shape[0] * scale_factor)
            dim = (width, height)
            resized = cv.resize(img, dim)
        else:
            resized = img

        ## NumPy array to PIL image
        ## https://stackoverflow.com/questions/10965417/how-to-convert-a-numpy-array-to-pil-image-applying-matplotlib-colormap
        img_list.append(PIL.Image.fromarray(np.uint8(resized)))

    return img_list

def extract_silhouette(input_images):
    """Extracts the silhouettes from input images (front and side views)
       and creates the silhouette files

    Args:
        input_images (list): a list with two PIL images for the front and side view input images

    Returns:
        sil_images: a list with two np arrays for the front and side view silhouette images
    """

    print(
        "[2] Starting to extract the silhouettes from input images (front and side views)"
    )
    start = time.time()

    sil_images = list()

    ## Adapted from: https://news.machinelearning.sg/posts/beautiful_profile_pics_remove_background_image_with_deeplabv3/
    # Load torchvision model
    deeplab_model = torch.hub.load(
        "pytorch/vision:v0.10.0", "deeplabv3_resnet101", weights="DEFAULT"
    )
    deeplab_model.eval()

    preprocess = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    for vi, img in enumerate(input_images):

        view = VIEWS[vi]

        print(f"** Starting {view} view")
        start_view = time.time()

        sil_name = f"silOrig_{view}_{input_gender_glob}.png"

        ## Check if silhouette does not exist to create it
        if sil_name not in os.listdir(OUTPUT_FILES_DIR):

            ## Preprocess the input image
            input_tensor = preprocess(img)
            # create a mini-batch as expected by the model
            input_batch = input_tensor.unsqueeze(0)

            ## Move the input and model to GPU for speed if available
            if torch.cuda.is_available():
                input_batch = input_batch.to("cuda")
                deeplab_model.to("cuda")

            with torch.no_grad():
                output = deeplab_model(input_batch)["out"][0]

            output_predictions = output.argmax(0)

            ## Create a color pallette, selecting a color for each class
            palette = torch.tensor([255, 255, 255])
            colors = torch.as_tensor(list(range(1, -1, -1)))[:, None] * palette
            colors = (colors).numpy().astype("uint8")

            fig = plt.figure()
            # Plot the semantic segmentation predictions of 21 classes in each color
            sil = PIL.Image.fromarray(output_predictions.byte().cpu().numpy()).resize(
                img.size
            )
            sil.putpalette(colors)

            ## Save Temporary ORIGINAL silhouette image
            plt.imshow(sil)
            plt.axis("off")
            fig.savefig(
                os.path.join(OUTPUT_FILES_DIR, sil_name),
                bbox_inches="tight",
                pad_inches=0,
                dpi=IMG_RESIZE,
            )

        ## Load image and remove transparency
        dim2 = (IMG_RESIZE, IMG_RESIZE)
        sil = PIL.Image.open(os.path.join(OUTPUT_FILES_DIR, sil_name))
        sil = PIL.ImageOps.pad(sil, dim2, color="white", centering=(0.5, 0.5))
        sil = remove_transparency(sil).convert("L")

        # sil = np.array(sil).astype(float)
        sil = np.array(sil).astype(np.uint8)  # Instead of float
        sil = sil.reshape(sil.shape[0], sil.shape[1], 1)

        ## Save ORIGINAL silhouette image
        orig_sil = np.array(sil).astype(float)
        sil_name = f"silOrig2_{view}_{input_gender_glob}.png"
        cv.imwrite(os.path.join(OUTPUT_FILES_DIR, sil_name), orig_sil)  ## np array

        ## Resize to (IMG_SIZE, IMG_SIZE, CHAN) - one channel: gray scale
        dim = (IMG_SIZE_4NN, IMG_SIZE_4NN)
        sil = cv.resize(sil, dim)
        sil = sil.reshape(sil.shape[0], sil.shape[1], 1)

        ## Obtain the optimal threshold value
        thresh = filters.threshold_mean(sil)  # threshold_otsu(img)
        ## Apply thresholding to the image
        binary_global = sil < thresh
        sil = np.array(binary_global).astype(float)
        sil = sil.reshape(sil.shape[0], sil.shape[1], 1)

        ## Save silhouette image
        sil_name = f"sil_{view}_{input_gender_glob}.png"
        cv.imwrite(
            os.path.join(OUTPUT_FILES_DIR, sil_name), sil.squeeze().astype("uint8") * 255
        )

        # sil_images.append(np.array(sil).astype(float) / 255)
        sil_images.append(sil)

        # Delete variables and manually invoke garbage collection
        # Safely delete variables only if they exist
        if 'input_batch' in locals():
            del input_batch
        if 'output' in locals():
            del output
        if 'output_predictions' in locals():
            del output_predictions
        if 'sil' in locals():
            del sil
        plt.close('all')  # Close figures to free memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Run garbage collection
        gc.collect()

        print(f"-- Finished {view} view in {(time.time() - start_view):.1f} s")

    print(f"[2] Finished extracting the silhouettes from input images (front and side) in {(time.time() - start):.1f} s"
    )
    return sil_images

def remove_transparency(im, bg_colour=(255, 255, 255)):
    """
    # https://stackoverflow.com/questions/44997339/convert-python-image-to-single-channel-from-rgb-using-pil-or-scipy
    Remove transparency from an image by filling transparent parts with a specified background color.

    Args:
        im (PIL.Image.Image): The input image.
        bg_colour (tuple): The RGB color of the background.

    Returns:
        PIL.Image.Image: The image with transparency removed.
    """
    
    # Only process if image has transparency
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):

        # Need to convert to RGBA if LA format due to a bug in PIL
        alpha = im.convert("RGBA").split()[-1]

        # Create a new background image of our matt color.
        # Must be RGBA because paste requires both images have the same format
        bg = PIL.Image.new("RGBA", im.size, bg_colour + (255,))
        bg.paste(im, mask=alpha)
    
        return bg
    
    return im

def measurements_from_sil(
    model,
    input_dataX,
    img_input,
    df,
    stature_cm,
):
    """Extracts 8 measurements from the silhouettes from input images (front and side views)

    Args:
        model (keras model): CNN-MLP model
        input_dataX (np array): 'gender', 'height', and 'weight' values (encoded and scaled)
        img_input (list of np arrays): front and side view silhouette images
        df (DataFrame): data from SPRING, ANSUR I and II.
        stature_cm (int): stature in cm

    Returns:
        extracted_measurements: a np array with 8 measurements (to complete LIST_10_MEAS)
    """

    ## CNN-MLP prediction
    input_img_front = np.array(img_input[0]).reshape(1, IMG_SIZE_4NN, IMG_SIZE_4NN, 1)
    input_img_side = np.array(img_input[1]).reshape(1, IMG_SIZE_4NN, IMG_SIZE_4NN, 1)

    ## Prediction (data augmentation)
    # https://stackoverflow.com/questions/57092637/how-to-fit-keras-imagedatagenerator-for-large-data-sets-using-batches
    # https://stackoverflow.com/questions/49404993/how-to-use-fit-generator-with-multiple-inputs

    # we create two instances with the same arguments
    data_gen_args_input = dict(
        featurewise_center = True,
        featurewise_std_normalization = True,
        # width_shift_range = 0.1,
        # height_shift_range = 0.25,
        # shear_range = 3,
        # zoom_range = [0.8, 1.2],
        # horizontal_flip = False,
        zoom_range = [0.8, 1.2],
    )

    ## compute quantities required for featurewise normalization
    ## (std, mean, and principal components if ZCA whitening is applied)
    datagen_input = ImageDataGenerator(**data_gen_args_input)

    datagen_input.fit(input_img_front, augment=True)

    ### Test-Time Augmentation to Make Predictions
    preds_input_aug_all = np.zeros((1, len(UK_MEAS)))

    # Initialize arrays to accumulate augmented images
    input_img_front_aug_accum = []
    input_img_side_aug_accum = []

    # Generate augmented images and accumulate
    for _ in range(NUM_AUG_INPUT):
        batch_img_front_aug = datagen_input.flow(
            input_img_front,
            batch_size=1,
            shuffle=False,
            seed=random.randint(0, 100),
        ).next()[0]
        
        # Generate augmented images and accumulate
        input_img_front_aug_accum.append(batch_img_front_aug)
        batch_img_side_aug = datagen_input.flow(
            input_img_side,
            batch_size=1,
            shuffle=False,
            seed=random.randint(0, 100),
        ).next()[0]
        input_img_side_aug_accum.append(batch_img_side_aug)

    # Stack the accumulated images into arrays
    input_img_front_aug = np.stack(input_img_front_aug_accum)
    input_img_side_aug = np.stack(input_img_side_aug_accum)

    # Repeat input_dataX along the first axis to match the shape of input_img_front_aug
    repeated_input_dataX = np.repeat(input_dataX, input_img_front_aug.shape[0], axis=0)

    try:
        preds_input_aug_all = model.predict(
                [repeated_input_dataX, input_img_front_aug, input_img_side_aug], verbose=0
        )
    except ValueError:
        print("Please, make sure the model file you are using is the correct one - double check the model architecture and the input data.")
        sys.exit(1)

    # Apply scaling
    mean_values = df[UK_MEAS].mean()
    std_values = df[UK_MEAS].std()

    preds_input_scaled = preds_input_aug_all * std_values.values + mean_values.values

    # Compute final prediction
    preds_input = np.round(np.mean(preds_input_scaled, axis=0), 2)

    ## Join info and extracted measurements (to complete 9 measurements) - Not included weight
    input9meas = np.hstack([np.array(stature_cm), preds_input])

    return input9meas

def main():
    """
    Main function to define all the steps from the photos to generating the 3D avatar.
    """

    try:
        input_info, input_images, df_total = get_input_data()
        sil_images = extract_silhouette(input_images)
    except (FileNotFoundError, IOError) as error:
        print(f"Error: {error}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)

    try:
        print("[3] Loading the previously trained extractor model.")
        start = time.time()
        model = load_model(MODEL_NAME)
        print(f"[3] Finished loading the previously trained extractor model in {(time.time() - start):.1f} s")
    except FileNotFoundError as file_error:
        print(f"Error: {file_error}")
        sys.exit(1)

    try:
        extracted_measurements = measurements_from_sil(
            model,
            input_info,
            sil_images,
            df_total,
            stature_glob,
        )
    except ValueError:
        print("Please, check the silhouettes images size - it must be 224x224 px (IMG_SIZE_4NN in utils.py). Check input_info.")
        sys.exit(1)

    print("[4] Starting to create the 3D avatar.")
    start = time.time()

    measurements = create_measurements_array(extracted_measurements, weightkg_glob)

    avatar = Avatar(measurements, input_gender_glob)
    # Impute missing measurements
    _ = avatar.predict()
    # Create the 3D avatar
    avatar.create_obj_file(ava_name=f'avatar_{input_gender_glob}_fromImg')
    # Measure the 3D avatar (to compare with the input measurements)
    _ = avatar.measure(out_meas_name=f"output_data_avatar_{input_gender_glob}_fromImg")

    print(f"[4] Finished creating the 3D avatar in {(time.time() - start):.1f} s")

#########################################################################################
if __name__ == "__main__":

    main()
    