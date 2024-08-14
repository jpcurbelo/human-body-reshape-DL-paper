import os
import sys
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re  ## used to extract a substring of integers in a string
import joblib
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from scipy.interpolate import splprep, splev
import random

from utils import *
from main import load_model, measurements_from_sil, run_tflite_model_inference
from silhouettes2measurements import process_db_values, load_df_total
from avatar import Avatar

BM_DIR = (
    "/media/jesus/GSUS-DATA/CANADA_docs/3D_human_avatar_Mitacs/"
    "NN_silhouettes_SPRING-ANSUR/Papers_NN_measurements/"
    "SilhouetteBodyMeasurementBenchmarks"
)

SIL_DIR = os.path.join(BM_DIR, "rendered_rgb")  # female/front/silh"
FIT_ALG = "nicp"   #"lbfgsb"  # "nicp"  # "lbfgsb"
AVA_DIR = os.path.join(
    BM_DIR, f"smpl6890v_{FIT_ALG}_fits"
)  # "smpl6890v_lbfgsb_fits"    /female

NUM_CAT = False    ## whether the model was trained with numeric + categorical data

BENCHMARK_SILH_DIR = os.path.join(BENCHMARK_FILES, "benchmark_silh")

if NUM_CAT == True and AUGMENTATION_Model == True:
    BENCHMARK_AVA_DIR = os.path.join(BENCHMARK_FILES, f"benchmark_ava_{FIT_ALG}")
elif NUM_CAT == False and AUGMENTATION_Model == True:
    BENCHMARK_AVA_DIR = os.path.join(BENCHMARK_FILES, f"benchmark_ava_{FIT_ALG}_noNumCat")
elif NUM_CAT == False and AUGMENTATION_Model == False:
    BENCHMARK_AVA_DIR = os.path.join(BENCHMARK_FILES, f"benchmark_ava_{FIT_ALG}_noNumCat_noAug")

def main():
    """
        Main function to define all the steps to run test on the benchmark dataset.
    """

    print("MAIN *****************************************")

    # ## Load and process benchmark silh - save with Input silh format
    # bm2input_silh()

    ## Reconstruct the avatars benchmark
    create_benchmark_avatars()

    measure_save_avatars()


def bm2input_silh(benchmark_silh_dir=BENCHMARK_SILH_DIR):
    """Generate input silhouettes from benchmark silhouettes for all genders and views.

    Returns:
        None.
    """

    ## Create folders
    try:
        os.mkdir(BENCHMARK_SILH_DIR)
    except OSError as error:
        # print(error)
        print(f"Folder '{BENCHMARK_SILH_DIR}' already existed.")

    for gender in GENDERS:
        ## Create folder
        gender_silh_dir = os.path.join(BENCHMARK_SILH_DIR, gender)
        try:
            os.mkdir(gender_silh_dir)
        except OSError as error:
            # print(error)
            print(f"Folder '{gender_silh_dir}' already existed.")

        for view in VIEWS:
            ## Create folder
            view_silh_dir = os.path.join(gender_silh_dir, view)
            try:
                os.mkdir(view_silh_dir)
            except OSError as error:
                # print(error)
                print(f"Folder '{view_silh_dir}' already existed.")

            ## Load, transform, and save the silhs
            inp_silh_dir = os.path.join(SIL_DIR, f"{gender}/{view}/silh")

            inp_silh_file_list = sorted(os.listdir(inp_silh_dir))
            print(len(inp_silh_file_list))

            for inp_silh in inp_silh_file_list:

                match = re.search(r"\d+", inp_silh)
                idx = match.group()

                inp_silh_file_dir = os.path.join(inp_silh_dir, inp_silh)

                silh_file_name = f"model_{gender}_{view}_{idx}.png"
                out_silh_file_dir = os.path.join(view_silh_dir, silh_file_name)

                ## If exist the bm silh and not the one modified for us
                if os.path.exists(inp_silh_file_dir) and not os.path.exists(
                    out_silh_file_dir
                ):

                    try:
                        silh_img = load_save_silhouettes_bm(
                            gender=gender,
                            view=view,
                            idx=idx,
                            out_silh_dir=view_silh_dir,
                            silh_name=silh_file_name,
                        )
                    except Exception as e:
                        print(f"Failed to load silhouette image: {gender}/{view}/{idx}")
                        silh_img = None
                        break


def load_benchmark_ids(gender="female", view="front"):
    """
    Loads benchmark IDs for a given gender from an Excel file.

    Args:
        gender (str): Gender of the IDs to load. Default is 'female'.

    Returns:
        A list of benchmark IDs.
    """

    df = pd.read_excel(
        os.path.join(BENCHMARK_FILES, f"Transformed_{gender}_with_Index.xlsx")
    )

    # print(df)

    return df["Index"].tolist()


def load_save_silhouettes_bm(
    gender="female",
    view="front",
    idx="0000",
    out_silh_dir=os.path.join(BENCHMARK_FILES, "/female/front"),
    silh_name=f"model_female_front_0000.png",
):
    """Loads a silhouette benchmark image.

    Args:
        gender (str): Gender of the silhouette. Default is 'female'.
        view (str): View of the silhouette. Default is 'front'.
        idx (str): ID of the silhouette. Default is '0000'.

    Returns:
        A numpy array representing the loaded image.
    """

    inp_sil_dir = os.path.join(SIL_DIR, f"{gender}/{view}/silh")
    filename = f"{gender}_{idx}.png"

    ## Load input image (as np array)
    img = cv.imread(os.path.join(inp_sil_dir, filename), cv.IMREAD_UNCHANGED)

    if view == "side":
        # Flip image horizontally
        img = cv.flip(img, 1)

    ## Resize by scale
    if img.shape[0] > SIL_SIZE or img.shape[1] > SIL_SIZE:
        # Determine scaling factor
        scale_factor = min(SIL_SIZE / img.shape[0], SIL_SIZE / img.shape[1])

        # Scale image
        img = cv.resize(
            img, None, fx=scale_factor, fy=scale_factor, interpolation=cv.INTER_AREA
        )
    # else:
    #     # Determine padding values
    #     top_pad = (SIL_SIZE - img.shape[0]) // 2
    #     bottom_pad = SIL_SIZE - img.shape[0] - top_pad
    #     left_pad = (SIL_SIZE - img.shape[1]) // 2
    #     right_pad = SIL_SIZE - img.shape[1] - left_pad

    #     # Pad image
    #     img = np.pad(img, ((top_pad, bottom_pad), (left_pad, right_pad), (0, 0)), mode='constant')

    ## Pad scaled image
    ## Height
    if img.shape[0] < SIL_SIZE:
        dif = SIL_SIZE - img.shape[0]
        left_pad = dif // 2
        right_pad = dif - left_pad
    else:
        left_pad = 0
        right_pad = 0

    ## Width
    if img.shape[1] < SIL_SIZE:
        dif = SIL_SIZE - img.shape[1]
        top_pad = dif // 2
        bott_pad = dif - top_pad
    else:
        top_pad = 0
        bott_pad = 0

    pad_width = ((left_pad, right_pad), (top_pad, bott_pad))
    resized = np.pad(img, pad_width=pad_width, mode="constant")

    ## Save image
    cv.imwrite(os.path.join(out_silh_dir, silh_name), resized)

    return np.array(resized).astype(np.float32) / 255


def create_benchmark_avatars():
    """
    Create benchmark avatars for each gender and ID using the best available model.
    """

    ## Create folder
    try:
        os.mkdir(BENCHMARK_AVA_DIR)
    except OSError as error:
        # print(error)
        print(f"Folder '{BENCHMARK_AVA_DIR}' already existed.")

    ## model_name = "model_sil2meas_inp2.tflite"  ## best so far?
    nn_model_name = MODEL_NAME

    print(nn_model_name)
    nn_model = load_model(nn_model_name)

    for gender in GENDERS:

        ## Create folder
        gender_ava_dir = os.path.join(BENCHMARK_AVA_DIR, gender)
        try:
            os.mkdir(gender_ava_dir)
        except OSError as error:
            # print(error)
            print(f"Folder '{gender_ava_dir}' already existed.")

        ## List of IDs to be used/analyzed
        silh_dir = os.path.join(BENCHMARK_SILH_DIR, f"{gender}/front")

        id_list = list()
        for model in sorted(os.listdir(silh_dir)):
            ## Extract the (first) sequence of digits from the string representing the model name.
            match = re.search(r"\d+", model)
            idx = match.group()
            id_list.append(idx)

        measurements = np.zeros((len(LIST_10_MEAS) - 1, len(id_list)))

        for i, idx in enumerate(id_list):

            filename = os.path.join(
                BENCHMARK_AVA_DIR, f"{gender}/model_{gender}_{idx}.obj"
            )

            # ## Check if file exists and it is complete - (incomplete  has less vertices than expected)
            if os.path.exists(filename) and check_obj_complete(filename):
                print("File path:", filename, "exists")
            # if False:
            #     print("File path:", filename, "exists")
            else:
                # if True:
                sys.stdout.write(
                    "\r>> Extracting measurements from silh: %s body %d"
                    % (gender, i + 1)
                )
                sys.stdout.flush()

                ## Load input images
                silh_images = load_input_silh(gender, idx)

                ## Load height for avatar
                input_info, df_total = load_ava_data(gender, idx)

                try:

                    input_info = input_info.astype(np.float32)

                    if NUM_CAT == True and AUGMENTATION_Model == True:
                        if "inp3" in MODEL_NAME:
                            meas_dat = input_info[:, [0, 1, 2, 3]].astype(np.float32)
                        else:
                            meas_dat = input_info[:, [0, 1, 3]].astype(np.float32)

                        extracted_measurements = measurements_from_sil(
                            nn_model,
                            ## Only take the gender+stature as input - not weight
                            meas_dat,  # input_info[:, [0, 1, 2, 3]],  ##input_info    [0, 1, 3]
                            silh_images,
                            df_total,
                            augmentation=AUGMENTATION,
                            model_name=nn_model_name,
                            gender=gender,
                            stature=stature,
                            weightkg=weightkg,
                        )
                    elif NUM_CAT == False:    # and AUGMENTATION_Model == True:
                        extracted_measurements = measurements_from_sil_noNumCat(
                            nn_model,
                            ## Only images as inputs
                            silh_images,
                            df_total,
                            augmentation=AUGMENTATION,
                            model_name=nn_model_name,
                            gender=gender
                        )

                except ValueError:
                    print(
                        "Please, check the silhouettes images size - it must be 224x224 px (SIL_SIZE in utils.py). Check input_info."
                    )
                    sys.exit()

                else:

                    ## Save extracted data
                    measurements[:, i] = extracted_measurements

                    ## Create wavefront.obj files
                    # Predictions - Real Photos
                    body = Avatar(extracted_measurements, gender)
                    input_meas21 = body.predict()

                    ## Create 3D avatar
                    body.create_obj_file(
                        benchmark=True, ava_dir=gender_ava_dir, idx=idx
                    )

                    # ## Extract measurements from the 3D avatar
                    # output_meas21 = body.measure(save_file=False)
                    # print(output_meas21)

        ## Save extrated measurements
        save_data_csv(
            COLUMNS_NEW_NAMES.keys(),
            measurements,
            gender_ava_dir,
            label=gender,
            fileid="extracted",
        )  ## Do not consider weight here

        measurement = None


def measurements_from_sil_noNumCat(
    model,
    img_input,
    df,
    augmentation=False,
    model_name="model_sil2meas_inp3.tflite",
    gender="female"
):
    """Extracts 8 measurements from the silhouettes from input images (front and side views)

    Args:
        img_input (list of np arrays): front and side view silhouette images
        df (DataFrame): data from SPRING, ANSUR I and II.
        augmentation (bool, False): flag to use prediction with test-time data augmentation

    Returns:
        extracted_measurements: a np array with 8 measurements (to complete LIST_10_MEAS)
    """

    # print(
    #     "[3] Starting to extract body measurements from silhouettes images (front and side views)"
    # )
    # start = time.time()

    # model_dir = os.path.join(MODEL_DIR, model_name)
    model_ext = model_name.split(".")[-1]

    ## CNN-MLP prediction

    input_gender_id = GENDER_DICT[gender]
    gender_df = df[df["gender"] == input_gender_id]

    input_imgFront = np.array(img_input[0]).reshape(1, SIL_SIZE, SIL_SIZE, 1)
    input_imgSide = np.array(img_input[1]).reshape(1, SIL_SIZE, SIL_SIZE, 1)

    if augmentation == False:

        # print(f"input_imgFront {input_imgFront.shape} {np.max(input_imgFront)}")

        # ## Plot the silh using Matplotlib
        # plt.imshow(img_input[0], cmap='gray')
        # plt.show()

        ## Prediction (regular)
        if model_ext == "h5":
            predsInput = model.predict(
                [input_imgFront, input_imgSide], verbose=0
            )
        elif model_ext == "tflite":
            model.allocate_tensors()
            input_details = model.get_input_details()
            output_details = model.get_output_details()
            # print(input_details)
            predsInput = np.array(
                run_tflite_model_inference(
                    input_imgFront,
                    input_imgSide,
                    input_details,
                    output_details,
                    model,
                )
            )

        predsInput2 = predsInput.copy()
        for i in range(predsInput.shape[1]):
            ## To check this in the training step  **HEREEEEEEEEEEEEEEEEEEEEEEEEEEE
            predsInput2[:, i] = (
                predsInput[:, i] * df[UK_MEAS[i]].std() + df[UK_MEAS[i]].mean()
            )
            ## predsInput2[:,i] = predsInput[:,i] * gender_df[UK_MEAS[i]].std() + gender_df[UK_MEAS[i]].mean()

        # print(predsInput2)
        predsInput = np.round(predsInput2, 1)

    else:
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

        # for batch_i in range(NUM_AUG_INPUT):
        #     generatorInputF = datagenInput.flow(
        #         input_imgFront,
        #         batch_size=1,
        #         shuffle=False,
        #         seed=random.randint(0, 100),
        #     )
        #     generatorInputS = datagenInput.flow(
        #         input_imgSide,
        #         batch_size=1,
        #         shuffle=False,
        #         seed=random.randint(0, 100),
        #     )
        #     input_imgFront_aug = np.zeros(input_imgFront.shape)
        #     input_imgSide_aug = np.zeros(input_imgSide.shape)

        #     # for img in generatorInputF:
        #     #     input_imgFront_aug = img
        #     #     break

        #     # for img in generatorInputS:
        #     #     input_imgSide_aug = img
        #     #     break

        #     input_imgFront_aug = generatorInputF.next()[0]
        #     input_imgSide_aug = generatorInputS.next()[0]
        #     # Reshape
        #     input_imgFront_aug = input_imgFront_aug.reshape(-1, SIL_SIZE, SIL_SIZE, 1)
        #     input_imgSide_aug = input_imgSide_aug.reshape(-1, SIL_SIZE, SIL_SIZE, 1)

        #     # Prediction
        #     if model_ext == "h5":
        #         predsInputAug = model.predict(
        #             [input_imgFront_aug, input_imgSide_aug], verbose=0
        #         )
        #     elif model_ext == "tflite":
        #         model.allocate_tensors()
        #         input_details = model.get_input_details()
        #         output_details = model.get_output_details()
        #         predsInputAug = np.array(
        #             run_tflite_model_inference(
        #                 input_imgFront_aug,
        #                 input_imgSide_aug,
        #                 input_details,
        #                 output_details,
        #                 model,
        #             )
        #         )

        #     predsInput2 = predsInputAug.copy()
        #     for j in range(predsInputAug.shape[1]):
        #         predsInput2[:, j] = (
        #             predsInputAug[:, j] * df[UK_MEAS[j]].std() + df[UK_MEAS[j]].mean()
        #         )

        #     # Accumulate results
        #     predsInputAug = np.round(predsInput2, 1)
        #     predsInputAugSUM = np.add(predsInputAugSUM, predsInputAug)

        #     if batch_i == 0:
        #         predsInputAugALL = predsInputAug.copy()
        #     else:
        #         predsInputAugALL = np.concatenate(
        #             (predsInputAugALL, predsInputAug), axis=0
        #         )

        # predsInput = np.round(predsInputAugSUM / NUM_AUG_INPUT, 2)



        # input_imgFront_aug = np.zeros((NUM_AUG_INPUT, SIL_SIZE, SIL_SIZE, 1))
        # input_imgSide_aug = np.zeros((NUM_AUG_INPUT, SIL_SIZE, SIL_SIZE, 1))

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

        # Predict using TFLite for all images
        if model_ext == "h5":
            predsInputAugALL = model.predict(
                [input_imgFront_aug, input_imgSide_aug], verbose=0
        )
        elif model_ext == "tflite":
            model.allocate_tensors()
            input_details = model.get_input_details()
            output_details = model.get_output_details()
            predsInputAugALL = np.array(
                run_tflite_model_inference(
                    input_imgFront_aug,
                    input_imgSide_aug,
                    input_details,
                    output_details,
                    model,
                )
            )

        # Apply scaling
        mean_values = df[UK_MEAS].mean()
        std_values = df[UK_MEAS].std()

        predsInputScaled = predsInputAugALL * std_values.values + mean_values.values

        # Compute final prediction
        predsInput = np.round(np.mean(predsInputScaled, axis=0), 2)


    ## Join info and extracted measurements (to complete 10 measurements)
    # input10meas = np.hstack([np.array([stature, weightkg]), predsInput[0]])
    input10meas = np.hstack([np.array([0.0, 0.0]), predsInput])   ## in this case, weitght and stature are not known
    # input10meas = predsInput.copy()

    return input10meas


def check_obj_complete(filename, expected_vertices=V_NUM):
    """
    Check if a file has less vertices than expected.

    Parameters:
    - filename (str): The path to the file to check.
    - expected_vertices (int): The expected number of vertices in the file.

    Returns:
    - True if the file is complete (has the expected number of vertices), False otherwise.
    """
    with open(filename, "r") as modelf:
        vertices = 0
        for line in modelf:
            if "v " in line:
                vertices += 1

    if vertices != expected_vertices:
        return False
    else:
        return True


def load_input_silh(gender: str, idx: int):
    """
    Load input silhouette images for a given gender and index from benchmark silhouettes directory.

    Args:
        gender (str): Gender of the model. Either "male" or "female".
        idx (int): Index of the model.

    Returns:
        list: A list containing input silhouette images for different views.

    Raises:
        FileNotFoundError: If the input image file is not found.
    """

    silh_list = list()
    for view in VIEWS:

        view_silh_dir = os.path.join(BENCHMARK_SILH_DIR, f"{gender}/{view}")
        filename = f"model_{gender}_{view}_{idx}.png"

        ## Load input image (as np array)
        img = cv.imread(os.path.join(view_silh_dir, filename), cv.IMREAD_UNCHANGED)
        silh_list.append(img / 255)

    return silh_list


def load_ava_data(gender, idx):
    """
    Load avatar information from a given directory and file.

    Args:
        gender: A string specifying the gender of the avatar.
        id: An integer specifying the ID of the avatar.

    Returns:
        A NumPy array containing the avatar information.

    Raises:
        FileNotFoundError: If the specified file is not found.
    """

    # Load avatar from smpl6890v_lbfgsb_fits
    ava_dir = os.path.join(AVA_DIR, gender)

    ava_file = f"{gender}_{idx}.obj"

    with open(os.path.join(ava_dir, ava_file), "r") as modelf:
        # read vertices from model file
        vertices = []
        for line in modelf:
            if "v " in line:
                line.replace("\n", " ")
                tmp = list(map(float, line[1:].split()))
                vertices.append(tmp)

        vert_np = np.array(vertices)
        ## Difference on z (height)
        height = (np.max(vert_np[:, 1]) - np.min(vert_np[:, 1])) * 100  ## to cm

        ###Save global variable not scaled
        global stature
        stature = height
        global weightkg
        weightkg = 0.0

        d = {"weightkg": [0.0], "stature": [height]}
        df = pd.DataFrame(data=d)

        ## SCALE and gender to Categorical
        ## Load databases
        tot_db_dir = os.path.join(MODEL_DIR, "input_total_db_spring_ansurI_II.csv")
        try:
            # Open the file for reading
            df_total = pd.read_csv(
                tot_db_dir, encoding=file_encoding, converters={"ID": str}
            )

        except:
            # The file does not exist, so we need to handle the exception
            df_total = None

        if df_total is None:
            df_total = load_df_total()

        ## Load scaler
        if isinstance(SCALER, StandardScaler):
            tot_scaler_dir = os.path.join(MODEL_DIR, "scalerStd.pkl")
        else:
            tot_scaler_dir = os.path.join(MODEL_DIR, "scalerMinMax.pkl")

        try:
            # Open the file for reading
            dataMeasX_scaler = joblib.load(tot_scaler_dir)

        except:
            # The file does not exist, so we need to handle the exception
            dataMeasX_scaler = None

        if dataMeasX_scaler is None:
            ## Scales Train and Test values
            (
                trainMeasX,
                testMeasX,
            ), dataMeasX_scaler = process_db_values(df_total)
            joblib.dump(dataMeasX_scaler, tot_scaler_dir)

        # print(df)
        aux_df = dataMeasX_scaler.transform(df)

        gender_aux = np.array([GENDER_DICT[gender]])
        gender_cat = keras.utils.to_categorical(gender_aux, len(GENDER_DICT.keys()))

        inputDataX = np.hstack([gender_cat, aux_df])

        if "gender" not in LIST_3_MEAS:
            inputDataX = np.delete(inputDataX, 0, 1)

        return inputDataX, df_total


def measure_save_avatars():

    for gender in GENDERS:

        ## Create folder
        gender_ava_dir = os.path.join(BENCHMARK_AVA_DIR, gender)

        # print(gender_ava_dir)

        # facets_dir = os.path.join(MODEL_DIR, f"facets_template_3DHBSh.npy")
        # try:
        #     facets = np.load(open(facets_dir, "rb"), allow_pickle=True)
        # except FileNotFoundError:
        #     # The file does not exist, so we need to handle the exception
        #     print(f"Wrong file or file path --> '{facets_dir}' file not found.")

        cp_dir = os.path.join(MODEL_DIR, f"cp_perfit2023_{gender}.npy")
        try:
            cp = np.load(open(cp_dir, "rb"), allow_pickle=True)
        except FileNotFoundError:
            # The file does not exist, so we need to handle the exception
            print(f"Wrong file or file path --> '{cp_dir}' file not found.")

        vert_filename = os.path.join(
            BENCHMARK_AVA_DIR, f"benchmark_avatars_vertices_{FIT_ALG}_{gender}.npy"
        )
        if os.path.exists(vert_filename):
            vertices = np.load(open(vert_filename, "rb"), allow_pickle=True)
        else:
            vertices = obj2npy(gender_ava_dir, label=gender)

        # print(vertices.shape)

        measurements = measure_bodies(cp, vertices, gender=gender)
        # print(measurements.shape)

        save_data_csv(
            MEASUREMENTS[1:], measurements, gender_ava_dir, label=gender
        )  ## Do not consider weight here


def obj2npy(obj_file_dir, label="female"):
    """
    Loads data (vertices) from *.obj files in the database
    Returns a numpy array containing the vertices data
    """
    file_list = sorted(os.listdir(obj_file_dir))

    # load original data
    vertices = []
    for i, obj in enumerate(file_list):
        sys.stdout.write("\r>>  converting %s body %d" % (label, i + 1))
        sys.stdout.flush()
        f = open(os.path.join(obj_file_dir, obj), "r")
        # j = 0
        for line in f:
            if line[0] == "#":
                continue
            elif "v " in line:
                line.replace("\n", " ")
                tmp = list(map(float, line[1:].split()))
                # append vertices from every obj files
                vertices.append(tmp)
                # j += 1
            else:
                break

        f.close()

    # reshape vertices to an array of V_NUM rows * (x, y, z - 3 columns) for every .obj file
    # print("\nchecking: If you meant to do this, you must specify 'dtype=object' \
    #       when creating the ndarray.\n\n")
    # print(vertices.shape)
    vertices = np.array(vertices, dtype=np.float64).reshape(len(file_list), V_NUM, 3)

    # Normalize data
    for i in range(len(file_list)):
        # mean value of each of the 3 columns
        v_mean = np.mean(vertices[i, :, :], axis=0)
        vertices[i, :, :] -= v_mean

    vert_filename = os.path.join(
        BENCHMARK_AVA_DIR, f"benchmark_avatars_vertices_{FIT_ALG}_{label}.npy"
    )
    np.save(open(vert_filename, "wb"), vertices)

    # print(vertices)
    return vertices

    def calc_measurements(self, cp, facet):
        """Calculates measurement data from given vertices by control points
        Returns an array of M_NUM measurements
        """

        measurement_list = []

        # # calculate the person's weight
        # vol = 0.0
        # for i in range(0, F_NUM):
        #     f = [c - 1 for c in facet[i, :]]
        #     v1 = self.vertices[f[0], :]
        #     v2 = self.vertices[f[1], :]
        #     v3 = self.vertices[f[2], :]
        #     # the scalar triple product(axb).c
        #     vol += np.cross(v1, v2).dot(v3)
        # # volume of the tetrahedron
        # vol = abs(vol) / 6.0

        # weight = KHUMANBODY_DENSITY * vol
        # measurement_list.append(weight)  # * 10

        # calculate other measures
        for i, meas in enumerate(MEASUREMENTS[1:]):  # skip 0 - weight

            length = 0.0
            length = self.calc_length(MEAS_LABELS[meas], cp[i], meas)
            measurement_list.append(length * 100)  # meters to cm

        return np.array(measurement_list, dtype=np.float64)

    def calc_length(self, lab, cplist, meas):
        """ """

        length = 0.0
        p2 = self.vertices[int(cplist[0]) - 1]
        if lab == 1:
            p1 = self.vertices[int(cplist[1]) - 1]
            length = abs(p1[2] - p2[2])  # pos 2 (z-axis)

        elif lab == 3 or lab == 4:
            for i in range(1, len(cplist)):
                p1 = p2
                p2 = self.vertices[int(cplist[i]) - 1]
                # add point-to-point distance
                length += np.sqrt(np.sum((p1 - p2) ** 2.0))

        elif lab == 2:
            num_list = [int(x) - 1 for x in cplist]
            # define pts from the question
            pts = np.delete(self.vertices[num_list], 2, axis=1)  # delete pos 2 (z-axis)

            tck, u = splprep(pts.T, u=None, s=0.0, per=1)
            u_new = np.linspace(u.min(), u.max(), 1000)
            x_new, y_new = splev(u_new, tck, der=0)

            length = 0.0
            p2 = np.array([x_new[0], y_new[0]])
            for i in range(1, len(x_new)):
                p1 = p2
                p2 = np.array([x_new[i], y_new[i]])
                # add point-to-point distance
                length += np.sqrt(np.sum((p1 - p2) ** 2.0))

        return length  # / 10


def measure_bodies(cp, vertices, gender="female"):
    """
    Calculates the ANSUR obj file measurement data and convert to npy
    Returns a list of measurements
    """

    # load measures data
    measurements = np.zeros(
        (M_NUM - 1, vertices.shape[0])
    )  ## Do not consider weight here
    for i in range(vertices.shape[0]):
        sys.stdout.write(
            "\r>> calculating measurements of %s body %d" % (gender, i + 1)
        )
        sys.stdout.flush()
        measurements[:, i] = calc_measurements(cp, vertices[i, :, :], gender, i).flat

    return measurements


def calc_measurements(cp, vertices, label, i):
    """
    Calculates measurement data from given vertices by control points
    Returns an array of M_NUM measurements
    """
    measurement_list = []

    # # Set person's weight to 0
    # measurement_list.append(0)   #* 10

    # calculate other measures
    # for i, measurement in enumerate(cp):
    for i, meas in enumerate(MEASUREMENTS[1:]):  # skip 0 - weight

        length = 0.0
        length = calc_length(MEAS_LABELS[meas], cp[i], vertices, meas)
        measurement_list.append(length * 100)  # meters to cm

    meas_np = np.array(measurement_list, dtype=np.float64).reshape(
        M_NUM - 1, 1
    )  ## Do not consider weight here

    return meas_np


def calc_length(lab, cplist, vertices, meas):
    """ """
    length = 0.0
    p2 = vertices[int(cplist[0]) - 1]
    if lab == 1:
        p1 = vertices[int(cplist[1]) - 1]
        length = abs(p1[2] - p2[2])  # pos 2 (z-axis)

    elif lab == 3 or lab == 4:
        for i in range(1, len(cplist)):
            p1 = p2
            p2 = vertices[int(cplist[i]) - 1]
            # add point-to-point distance
            length += np.sqrt(np.sum((p1 - p2) ** 2.0))

    elif lab == 2:
        num_list = [int(x) - 1 for x in cplist]
        # define pts from the question
        pts = np.delete(vertices[num_list], 2, axis=1)  # delete pos 2 (z-axis)

        tck, u = splprep(pts.T, u=None, s=0.0, per=1)
        u_new = np.linspace(u.min(), u.max(), 1000)
        x_new, y_new = splev(u_new, tck, der=0)

        # if meas == 'chest_girth':
        #     plt.plot(pts[:,0] / 10, pts[:,1] / 10, 'ro')
        #     plt.plot(x_new / 10, y_new / 10, 'b--')
        #     # plt.title(meas)
        #     plt.xlabel('x-axis')
        #     plt.ylabel('y-axis')
        #     plt.axis('equal')
        #     # plt.savefig(f'{meas}.png', format='png')
        #     # plt.savefig(f'{meas}.eps', format='eps')
        #     plt.show()

        length = 0.0
        p2 = np.array([x_new[0], y_new[0]])
        # print(range(1, len(x_new)))
        for i in range(1, len(x_new)):
            p1 = p2
            p2 = np.array([x_new[i], y_new[i]])
            # add point-to-point distance
            length += np.sqrt(np.sum((p1 - p2) ** 2.0))

    return length  # / 10


def save_data_csv(
    meas_names, measurements, obj_file_dir, label="female", fileid="avatars"
):
    """
    Writes the nparray to disk
    """

    file_list = sorted(os.listdir(obj_file_dir))

    db_file_name = os.path.join(
        BENCHMARK_AVA_DIR, f"benchmark_{fileid}_measurements_{FIT_ALG}_{label}.csv"
    )
    with open(db_file_name, "w") as outfile:
        # I'm writing a header here just for the sake of readability
        outfile.write("index," + ",".join(meas_names) + "\n")

        for i in range(measurements.shape[1]):

            match = re.search(r"\d+", file_list[i])
            idx = match.group()

            meas_list = [measurements[j][i] for j in range(measurements.shape[0])]
            aux_list = [str(round(num, 2)) for num in meas_list]
            # aux_list = [str(num) for num in meas_list]
            outfile.write(idx + "," + ",".join(aux_list) + "\n")


#########################################################################################
if __name__ == "__main__":

    main()
