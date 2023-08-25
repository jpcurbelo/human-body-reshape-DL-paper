import sys
import time

import pandas as pd
import joblib
import numpy as np
import keras
from sklearn.model_selection import train_test_split
import cv2 as cv
import PIL
import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from skimage import filters
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import random

from utils import *

# Add the parent directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
# Now you can use absolute imports
from reshaper.avatar import Avatar


## Global variables
IMG_RESIZE = 448  # 512   #512   # 256   #1024

MODEL_NAME = "extractor_stdScl_img224_inp2_out8_ep175_paper.h5"
NUM_AUG_INPUT = 20

def main():
    """
    Main function to define all the steps from the photos to generating the 3D avatar.
    """

    ## Read the input data
    try:
        ## Try to access the data
        input_info, input_images, df_total = get_input_data()

        ## Extract the silhouettes from the input images
        sil_images = extract_silhouette(input_images)

    except Exception as e:
        print("Please, check the input files.")
        sys.exit()

    else:

        model_name = MODEL_NAME
        model = keras.models.load_model(os.path.join(MODEL_FILES_DIR, model_name))

        try:
            extracted_measurements = measurements_from_sil(
                model,
                ## Only take the gender+stature as input - not weight
                input_info, 
                sil_images,
                df_total,
                model_name=model_name,
                gender=input_gender_glob,
                stature=stature,
            )

        except ValueError:
            print(
                "Please, check the silhouettes images size - it must be 224x224 px (IMG_SIZE_4NN in utils.py). Check input_info."
            )
            sys.exit()

        else:

            print("[3] Starting to create the 3D avatar.")
            start = time.time()

            measurements = np.zeros(M_NUM).transpose()

            ## Assign measurements from 9 to 21-sized array
            measurements[0]  = weightkg   # 'weight' To be used in case the user has informed the weight
            measurements[1]  = extracted_measurements[0]   # 'stature'
            measurements[3]  = extracted_measurements[1]   # 'chestcircumference'
            measurements[4]  = extracted_measurements[2]   # 'waistcircumference'
            measurements[5]  = extracted_measurements[3]   # 'buttockcircumference'
            measurements[7]  = extracted_measurements[4]   # 'thighcircumference'
            measurements[14] = extracted_measurements[5]   # 'sleeveoutseam'
            measurements[17] = extracted_measurements[6]   # 'waistbacklength'
            measurements[6]  = extracted_measurements[7]   # 'shouldercircumference'
            measurements[16] = extracted_measurements[8]   # 'crotchheight'

            ## Predictions - Real Photos
            body = Avatar(measurements, input_gender_glob)
            input_meas21 = body.predict()

            ## Create 3D avatar
            body.create_obj_file(ava_name=f'avatar_{input_gender_glob}_fromImg')

            ## Extract measurements from the 3D avatar
            output_meas21 = body.measure(out_meas_name=f"output_data_avatar_{input_gender_glob}_fromImg")

            print(
                "[3] Finished creating the 3D avatar in {:.1f} s".format(
                    time.time() - start
                )
            )

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

    print(
        "[1] Finished loading input data (2 photos and basic info) in {:.1f} s".format(
            time.time() - start
        )
    )
    return basic_info, input_images, df_total


def import_basic_info():
    """Imports input data (gender, height, weight)

    Args:
        No arguments

    Returns:
        ###basic_info: a tupple with 'gender', 'height', and 'weight' values
        ###basic_info: a DataFrame with 'gender', 'height', and 'weight' values
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

    except:
        # The file does not exist, so we need to handle the exception
        print(
            f"Wrong file or file path --> '{tot_db_dir}' file not found. Try filtering the DS with ds_measurements_filter.py"
        )


    ## Load scaler
    if isinstance(SCALER, StandardScaler):
        tot_scaler_dir = os.path.join(MODEL_FILES_DIR, "scalerStd_img2ava.pkl")
    else:
        tot_scaler_dir = os.path.join(MODEL_FILES_DIR, "scalerMinMax_img2ava.pkl")

    try:
        # Open the file for reading
        dataMeasX_scaler = joblib.load(tot_scaler_dir)

    except:
        # The file does not exist, so we need to handle the exception
        dataMeasX_scaler = None

    if dataMeasX_scaler is None:
        ## Scales the data using the train/test split
        dataMeasX_scaler = scale_db_values()
        joblib.dump(dataMeasX_scaler, tot_scaler_dir)


    ###Save global variables not scaled
    global stature
    stature = input_df.iloc[0]["stature_cm"]

    global weightkg
    weightkg = input_df.iloc[0]["weight_kg"]

    aux_df = dataMeasX_scaler.transform(input_df[CONTINUOUS])   

    gender_aux = np.array(input_df["gender"])
    gender_aux = gender_aux.reshape(gender_aux.shape[0], -1)
    gender_cat = keras.utils.to_categorical(gender_aux, len(GENDER_DICT.keys()))

    inputDataX = np.hstack([gender_cat, aux_df])

    return inputDataX, df_total 


def scale_db_values():
    """
    Scale the continuous values in the DataFrame using the specified scaler.

    Args:
        df (DataFrame): The input DataFrame containing data to be scaled.

    Returns:
        StandardScaler: The fitted StandardScaler instance used for scaling.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    tot_db_dir_fem = os.path.join(DS_DIR, f"measurements_TOTAL_female.csv")
    tot_db_dir_mal = os.path.join(DS_DIR, f"measurements_TOTAL_male.csv")

    # Open the file for reading
    df_fem = pd.read_csv(
        tot_db_dir_fem, encoding=FILE_ENCODING, converters={"ID": str}
    )
    df_mal = pd.read_csv(
        tot_db_dir_mal, encoding=FILE_ENCODING, converters={"ID": str}
    )
    df = pd.concat([df_fem, df_mal], ignore_index=True)

    # Split data into train and test sets
    (trainData, testData) = train_test_split(
        df, train_size=0.8, shuffle=True, random_state=123
    )

    # Create and fit the scaler
    scaler = StandardScaler()
    scaler.fit(trainData[CONTINUOUS]) 

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
            SCL = 2048 / img.shape[0]
            width = int(img.shape[1] * SCL)
            height = int(img.shape[0] * SCL)
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
            colors = torch.as_tensor([i for i in range(1, -1, -1)])[:, None] * palette
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

        sil = np.array(sil).astype(float)
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

        print("** Finished {} view in {:.1f} s".format(view, time.time() - start_view))

    print(
        "[2] Finished extracting the silhouettes from input images (front and side views) in {:.1f} s".format(
            time.time() - start
        )
    )
    return sil_images


def remove_transparency(im, bg_colour=(255, 255, 255)):
    """
    # https://stackoverflow.com/questions/44997339/convert-python-image-to-single-channel-from-rgb-using-pil-or-scipy
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

    else:
        return im


def measurements_from_sil(
    model,
    inputDataX,
    img_input,
    df,
    model_name="extractor_stdScl_img224_inp2_out8_ep175_paper.h5",
    gender="female",
    stature=170,
):
    """Extracts 8 measurements from the silhouettes from input images (front and side views)

    Args:
        inputDataX (np array): 'gender', 'height', and 'weight' values (encoded and scaled)
        img_input (list of np arrays): front and side view silhouette images
        df (DataFrame): data from SPRING, ANSUR I and II.

    Returns:
        extracted_measurements: a np array with 8 measurements (to complete LIST_10_MEAS)
    """

    ## CNN-MLP prediction
    input_imgFront = np.array(img_input[0]).reshape(1, IMG_SIZE_4NN, IMG_SIZE_4NN, 1)
    input_imgSide = np.array(img_input[1]).reshape(1, IMG_SIZE_4NN, IMG_SIZE_4NN, 1)

    ## Prediction (data augmentation)
    # https://stackoverflow.com/questions/57092637/how-to-fit-keras-imagedatagenerator-for-large-data-sets-using-batches
    # https://stackoverflow.com/questions/49404993/how-to-use-fit-generator-with-multiple-inputs

    # we create two instances with the same arguments
    data_gen_argsInput = dict(
        featurewise_center=True,
        featurewise_std_normalization=True,
        # width_shift_range=0.1,
        # height_shift_range=0.25,
        # shear_range=3,
        # # zoom_range=[0.8, 1.2],
        # horizontal_flip=False,
        zoom_range=[0.8, 1.2],
    )

    ## compute quantities required for featurewise normalization
    ## (std, mean, and principal components if ZCA whitening is applied)
    datagenInput = ImageDataGenerator(**data_gen_argsInput)

    datagenInput.fit(input_imgFront, augment=True)

    ### Test-Time Augmentation to Make Predictions
    predsInputAugSUM = np.zeros((1, len(UK_MEAS)))
    predsInputAugALL = np.zeros((1, len(UK_MEAS)))

    # Initialize arrays to accumulate augmented images
    input_imgFront_aug_accum = []
    input_imgSide_aug_accum = []

    # Generate augmented images and accumulate
    for _ in range(NUM_AUG_INPUT):
        batch_imgFront_aug = datagenInput.flow(
            input_imgFront,
            batch_size=1,
            shuffle=False,
            seed=random.randint(0, 100),
        ).next()[0]
        input_imgFront_aug_accum.append(batch_imgFront_aug)
        
        batch_imgSide_aug = datagenInput.flow(
            input_imgSide,
            batch_size=1,
            shuffle=False,
            seed=random.randint(0, 100),
        ).next()[0]
        input_imgSide_aug_accum.append(batch_imgSide_aug)

    # Stack the accumulated images into arrays
    input_imgFront_aug = np.stack(input_imgFront_aug_accum)
    input_imgSide_aug = np.stack(input_imgSide_aug_accum)

    # Repeat inputDataX along the first axis to match the shape of input_imgFront_aug
    repeated_inputDataX = np.repeat(inputDataX, input_imgFront_aug.shape[0], axis=0)

    predsInputAugALL = model.predict(
            [repeated_inputDataX, input_imgFront_aug, input_imgSide_aug], verbose=0
    )

    # Apply scaling
    mean_values = df[UK_MEAS].mean()
    std_values = df[UK_MEAS].std()

    predsInputScaled = predsInputAugALL * std_values.values + mean_values.values

    # Compute final prediction
    predsInput = np.round(np.mean(predsInputScaled, axis=0), 2)

    ## Join info and extracted measurements (to complete 9 measurements) - Not included weight
    input9meas = np.hstack([np.array(stature), predsInput])

    return input9meas

#########################################################################################
if __name__ == "__main__":

    main()