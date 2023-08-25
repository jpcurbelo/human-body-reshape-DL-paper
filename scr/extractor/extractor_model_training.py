# Import libraries and modules
import numpy as np
import pandas as pd
import gc  # Garbage Collector - use it like gc.collect()
import joblib
import matplotlib.pyplot as plt

import tensorflow as tf
if tf.test.gpu_device_name():
    print(f"\nDefault GPU Device: {tf.test.gpu_device_n2ame()}\n")
    ## Usual output: 'Default GPU Device: /device:GPU:0'
else:
    print("\nPlease install GPU version of TF\n")

from sklearn.model_selection import train_test_split

from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Dense,
    Conv2D,
    Flatten,
    Dropout,
    MaxPooling2D,
    concatenate,
)

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import sys
sys.path.append("..")
from utils import *

SIL_FILES_DIR_npy = os.path.join(DS_DIR , f"silhouettes_blender{IMG_SIZE_4NN}_npy")

## Load scaler
if TEST_FILES == True:
    if isinstance(SCALER, StandardScaler):
        TOT_SCALER_DIR = os.path.join(MODEL_FILES_DIR, f"scalerStd_{TEST_FILES_NUM}_extractor_test.pkl")
    else:
        TOT_SCALER_DIR = os.path.join(MODEL_FILES_DIR, f"scalerMinMax_{TEST_FILES_NUM}_extractor_test.pkl")
else:
    if isinstance(SCALER, StandardScaler):
        TOT_SCALER_DIR = os.path.join(MODEL_FILES_DIR, "scalerStd_extractor.pkl")
    else:
        TOT_SCALER_DIR = os.path.join(MODEL_FILES_DIR, "scalerMinMax_extractor.pkl")

def main():

    # Instantiate a model checkpoint callback
    try:
        os.mkdir(MODEL_FILES_DIR)
    except OSError as error:
        print(error)


    #############################################
    ### Load Databases
    #############################################
    df_total = load_databases()
    print(f"\ndf_total.shape = {df_total.shape}")

    #############################################
    ### Load Images
    #############################################
    # exit()
    imgX_front, imgX_side = load_images()
    print(f"imgX_front.shape = {imgX_front.shape}")
    print(f"imgX_side.shape = {imgX_side.shape}\n")

    #############################################
    ### Prepare Numerical and Categorical Data"""
    #############################################

    ## Train and Test splitting
    (trainData, testData, trainImgXf, testImgXf, trainImgXs, testImgXs) = train_test_split(
        df_total, imgX_front, imgX_side, train_size=0.8, shuffle=True, random_state=123
    )

    ## DELETE to free memory
    try:
        del imgX_front
    except NameError:
        print("imgX_front was already deleted")

    try:
        del imgX_side
    except NameError:
        print("imgX_side was already deleted")

    gc.collect()

    ## Normalize the UNKNOWN MEASUREMENTS (labels) - will lead to better training and convergence
    trainMeasY = (trainData[UK_MEAS] - df_total[UK_MEAS].mean()) / df_total[UK_MEAS].std()
    testMeasY = (testData[UK_MEAS] - df_total[UK_MEAS].mean()) / df_total[UK_MEAS].std()

    ## Scales Train and Test values
    (trainMeasX, testMeasX), dataMeasX_scaler = process_db_values(
        df_total, trainData, testData
    )

    global INP_SHAPE
    INP_SHAPE = trainMeasX.shape[1]

    if "gender" not in KN_MEAS:
        trainMeasX = np.delete(trainMeasX, 0, 1)
        testMeasX = np.delete(testMeasX, 0, 1)

    print(f"\ntrainMeasX.shape = {trainMeasX.shape}")
    print(f"testMeasX.shape = {testMeasX.shape}\n")

    #############################################
    ### MLP + CNN Model
    #############################################

    ### Data Augmentation
    # https://stackoverflow.com/questions/57092637/how-to-fit-keras-imagedatagenerator-for-large-data-sets-using-batches
    # https://stackoverflow.com/questions/49404993/how-to-use-fit-generator-with-multiple-inputs

    # we create two instances with the same arguments
    data_gen_args = dict(
        featurewise_center=True,
        featurewise_std_normalization=True,
        # rotation_range=0,
        width_shift_range=0.1,
        height_shift_range=0.25,
        shear_range=3,
        zoom_range=[0.8, 1.2],
        horizontal_flip=False,
    )

    ## compute quantities required for featurewise normalization
    ## (std, mean, and principal components if ZCA whitening is applied)
    datagen = ImageDataGenerator(**data_gen_args)
    datagen.fit(trainImgXf, augment=True)

    ##Create models
    MLP_model = createMLP_model(in_MLPlayers=2)
    print(MLP_model.summary())

    CNN_model = createCNN_model(
        in_CNNlayers=1,
        in_DENSElayers=0,
    )
    # CNN_model = createAlexNetModel()
    print(CNN_model.summary())

    Combined_model = createCombined_model(MLP_model, CNN_model)
    print(Combined_model.summary())

    #############################################
    ## Compile the model using MeanSquaredError as our loss,
    ## implying that we seek to minimize the squared difference - mse
    opt = Adam(1e-4)
    # opt = build_optimizer(NN_PARAMETERS["optimizer"], NN_PARAMETERS["learning_rate"])
    lo = "mse"
    met = ["mean_absolute_error"]
    Combined_model.compile(loss=lo, optimizer=opt, metrics=met)

    batch_size = 32    #NN_PARAMETERS["batch_size"] 

    """### Train and save the model"""

    training_data_gen = generator2imgsNumData(datagen, 
        trainMeasX, trainImgXf, trainImgXs, trainMeasY, batch_size
    )
    validation_data_gen = generator2imgsNumData(
        datagen, testMeasX, testImgXf, testImgXs, testMeasY, batch_size
    )

    ## Instantiate an early stopping callback
    early_stopping = EarlyStopping(
        monitor="val_loss", min_delta=1e-4, patience=10, restore_best_weights=True
    )

    if TEST_FILES == True:
        EPOCHS = 20
    else:
        EPOCHS = 500

    ## Train the model
    print("[INFO] training model...")
    Combined_history = Combined_model.fit(
        training_data_gen,
        validation_data=(validation_data_gen),
        validation_steps=testImgXf.shape[0] // batch_size,
        steps_per_epoch=trainImgXf.shape[0] // batch_size,
        verbose=2,
        epochs=EPOCHS ,
        callbacks=[early_stopping],  
    )

    gc.collect()

    final_epochs = len(Combined_history.history["val_loss"])

    if TEST_FILES == True:
        if isinstance(SCALER, StandardScaler):
            MODEL_NAME = f"extractor_stdScal_img{IMG_SIZE_4NN}_inp{len(KN_MEAS)}_out{len(UK_MEAS)}_ep{final_epochs}_{TEST_FILES_NUM}test"
        else:
            MODEL_NAME = f"extractor_minMax_img{IMG_SIZE_4NN}_inp{len(KN_MEAS)}_out{len(UK_MEAS)}_ep{final_epochs}_{TEST_FILES_NUM}test"
    else:
        if isinstance(SCALER, StandardScaler):
            MODEL_NAME = f"extractor_stdScal_img{IMG_SIZE_4NN}_inp{len(KN_MEAS)}_out{len(UK_MEAS)}_ep{final_epochs}"
        else:
            MODEL_NAME = f"extractor_minMax_img{IMG_SIZE_4NN}_inp{len(KN_MEAS)}_out{len(UK_MEAS)}_ep{final_epochs}"

    MODEL_NAME_SV = MODEL_NAME + ".keras"
    Combined_model.save(os.path.join(MODEL_FILES_DIR, MODEL_NAME_SV))

    ### Plot history
    histplot(Combined_history, MODEL_NAME_SV, "mean_absolute_error")


def load_databases():
    """
    Loads and prepares the databases for further processing.

    The function reads the csv files for each database and gender using the pandas library's pd.read_csv() function,
    assigns them to different DataFrames, and drops unnecessary columns. Additionally, it adds a "gender" column to
    each DataFrame and concatenates them into one DataFrame.
    """

    file_encoding = "ISO-8859-1"  # 'utf8'

    df_spring = list()
    df_ansuri = list()
    df_ansurii = list()

    for dbi, dbname in enumerate(DBNAMES):
        for i, gender in enumerate(GENDER_DICT.keys()):

            db_dir = os.path.join(DS_DIR, f"measurements_{dbname}_{gender}.csv")    

            df = pd.read_csv(db_dir, encoding=file_encoding, converters={"ID": str})

            if dbi == 0:
                df_spring.append(df)
                df_spring[i].drop(
                    labels=df_spring[i].columns.difference(UK_MEAS + KN_MEAS),
                    axis=1,
                    inplace=True,
                )
                df_spring[i]["gender"] = i
            elif dbi == 1:
                df_ansuri.append(df)
                df_ansuri[i].drop(
                    labels=df_ansuri[i].columns.difference(UK_MEAS + KN_MEAS),
                    axis=1,
                    inplace=True,
                )
                df_ansuri[i]["gender"] = i
            elif dbi == 2:
                df_ansurii.append(df)
                df_ansurii[i].drop(
                    labels=df_ansurii[i].columns.difference(UK_MEAS + KN_MEAS),
                    axis=1,
                    inplace=True,
                )
                df_ansurii[i]["gender"] = i

    ## Concatenate DataFrames
    df_female = pd.concat([df_spring[0], df_ansuri[0], df_ansurii[0]], axis=0)
    df_male = pd.concat([df_spring[1], df_ansuri[1], df_ansurii[1]], axis=0)

    if TEST_FILES == True:
        df_female = df_female.head(3 * TEST_FILES_NUM) ## 3 because there are 3 datasets (SPRING, ANSURI, ANSURII)
        df_male = df_male.head(3 * TEST_FILES_NUM) ## 3 because there are 3 datasets (SPRING, ANSURI, ANSURII)

    df_total = pd.concat([df_female, df_male], axis=0)

    return df_total


def load_images():
    """
    Loads images from .npz files and concatenates them.
    Images are loaded from 3 different datasets (SPRING2023, ANSURI2023, ANSURII2023) and for both genders.

    Returns:
        Tuple of numpy arrays :
            - imgX_front (numpy array) : concatenated front view images.
            - imgX_side (numpy array) : concatenated side view images.
    """

    ## NPZ files
    # Load a list of [image]
    img_spring = [[] for x in range(2)]  # 0-female, 1-male
    img_ansuri = [[] for x in range(2)]  # 0-female, 1-male
    img_ansurii = [[] for x in range(2)]  # 0-female, 1-male

    for g, gender in enumerate(GENDER_DICT.keys()):
        ## NPZ files

        if TEST_FILES == True:
            npz_file_name = f"silh_Xarray{IMG_SIZE_4NN}_SPRING_{gender}_bw_{TEST_FILES_NUM}test.npz"
        else:
            npz_file_name = f'silh_Xarray{IMG_SIZE_4NN}_SPRING_{gender}_bw.npz'

        img_spring_npz = np.load(
            open(
                os.path.join(
                    SIL_FILES_DIR_npy, 
                    f"silhouettes_SPRING_bw", 
                    npz_file_name
                ),
                "rb",
            ),
            allow_pickle=True,
        )

        if TEST_FILES == True:
            npz_file_name = f"silh_Xarray{IMG_SIZE_4NN}_ANSURI_{gender}_bw_{TEST_FILES_NUM}test.npz"
        else:
            npz_file_name = f'silh_Xarray{IMG_SIZE_4NN}_ANSURI_{gender}_bw.npz'

        img_ansuri_npz = np.load(
            open(
                os.path.join(
                    SIL_FILES_DIR_npy, 
                    f"silhouettes_ANSURI_bw", 
                    npz_file_name
                ),
                "rb",
            ),
            allow_pickle=True,
        )

        if TEST_FILES == True:
            npz_file_name = f"silh_Xarray{IMG_SIZE_4NN}_ANSURII_{gender}_bw_{TEST_FILES_NUM}test.npz"
        else:
            npz_file_name = f'silh_Xarray{IMG_SIZE_4NN}_ANSURII_{gender}_bw.npz'

        img_ansurii_npz = np.load(
            open(
                os.path.join(
                    SIL_FILES_DIR_npy, 
                    f"silhouettes_ANSURII_bw", 
                    npz_file_name
                ),
                "rb",
            ),
            allow_pickle=True,
        )

        for i, view in enumerate(VIEWS):
            img_spring[g].append(img_spring_npz["arr_0"][i, :, :, :])
            img_ansuri[g].append(img_ansuri_npz["arr_0"][i, :, :, :])
            img_ansurii[g].append(img_ansurii_npz["arr_0"][i, :, :, :])

    ## DELETE to free memory
    # Delete npz arrays
    try:
        del img_spring_npz
    except NameError:
        print("img_spring_npz was already deleted")

    try:
        del img_ansuri_npz
    except NameError:
        print("img_ansuri_npz was already deleted")

    try:
        del img_ansurii_npz
    except NameError:
        print("img_ansurii_npz was already deleted")

    gc.collect()

    ## Concatenate NPYarrays
    imgX_front_female = np.concatenate(
        (img_spring[0][0], img_ansuri[0][0], img_ansurii[0][0]), axis=0
    )
    imgX_front_male = np.concatenate(
        (img_spring[1][0], img_ansuri[1][0], img_ansurii[1][0]), axis=0
    )

    imgX_side_female = np.concatenate(
        (img_spring[0][1], img_ansuri[0][1], img_ansurii[0][1]), axis=0
    )
    imgX_side_male = np.concatenate(
        (img_spring[1][1], img_ansuri[1][1], img_ansurii[1][1]), axis=0
    )

    ## DELETE to free memory
    try:
        del img_spring
    except NameError:
        print("img_spring[0] was already deleted")

    try:
        del img_ansuri
    except NameError:
        print("img_spring[0] was already deleted")

    try:
        del img_ansurii
    except NameError:
        print("img_spring[0] was already deleted")

    gc.collect()

    imgX_front = np.concatenate((imgX_front_female, imgX_front_male), axis=0)
    imgX_side = np.concatenate((imgX_side_female, imgX_side_male), axis=0)

    ## DELETE to free memory
    try:
        del imgX_front_female
    except NameError:
        print("imgX_front_female was already deleted")

    try:
        del imgX_front_male
    except NameError:
        print("imgX_front_male was already deleted")

    ## DELETE to free memory
    try:
        del imgX_side_female
    except NameError:
        print("imgX_side_female was already deleted")

    try:
        del imgX_side_male
    except NameError:
        print("imgX_side_male was already deleted")

    return imgX_front, imgX_side


def process_db_values(df, train, test):
    """
    Performs min-max scaling each continuous feature column to the range [0, 1]

    Parameters:
        df (Pandas.DataFrame): Dataframe containing all features.
        train (Pandas.DataFrame): Dataframe containing the training subset of the data.
        test (Pandas.DataFrame): Dataframe containing the testing subset of the data.

    Returns:
        tuple: Tuple containing the processed training and testing data, and the MinMaxScaler (or StandardScaler) object.
    """

    cs = SCALER
    trainContinuous = cs.fit_transform(
        train[CONTINUOUS]
    )  
    testContinuous = cs.transform(test[CONTINUOUS])  # test[KN_MEAS - CATEGORICAL]

    # one-hot encode the GENDER categorical data (by definition of
    # one-hot encoding, all output features are now in the range [0, 1])
    trainCategorical = keras.utils.to_categorical(
        train[CATEGORICAL], len(GENDER_DICT.keys())
    )
    testCategorical = keras.utils.to_categorical(
        test[CATEGORICAL], len(GENDER_DICT.keys())
    )

    # construct our training and testing data points by concatenating
    # the categorical features with the continuous features
    trainX = np.hstack([trainCategorical, trainContinuous])
    testX = np.hstack([testCategorical, testContinuous])

    # Save the scaler
    joblib.dump(cs, TOT_SCALER_DIR)

    # return the concatenated training and testing data
    return (trainX, testX), cs


## MLP model
## Define the MLP_model
def createMLP_model(in_MLPlayers=1):
    """
    Creates an MLP model for the categorical and numerical inputs,
    which will be used for predicting the target variable

    Args:
        in_MLPlayers: Number of inner MLP layers to use. Default is 1.

    Returns:
        A Keras Model object.
    """

    mlp_input = Input(shape=INP_SHAPE, name="mlp_input")

    mlp_hidden = Dense(
        16, activation="relu", name="mlp_hidden1"
    )(mlp_input)

    for i in range(in_MLPlayers):
        mlp_hidden = Dense(
            64,
            activation="relu",
            name=f"mlp_hiddenInner{i+1}",
        )(mlp_hidden)

    mlp_hidden = Dense(
        64, activation="relu", name="mlp_hidden2"
    )(mlp_hidden)

    mlp_output = Dense(len(UK_MEAS), activation="linear", name="mlp_output")(mlp_hidden)

    ##returns model
    return Model(mlp_input, mlp_output)


KERNEL_SIZE = (3, 3)  #  (3,3)
POOL_SIZE = (3, 3)  #  (3,3)
def createCNN_model(in_CNNlayers=1, in_DENSElayers=0):
    """
    Creates the CNN model with (modified) AlexNet architecture for image inputs

    Args:
        in_CNNlayers : number of hidden convolutional layers
        in_DENSElayers : number of hidden dense layers

    Returns:
        A CNN model with AlexNet architecture
    """
    ## AlexNet-wise ##
    ## MLP-CNN_sil2meas_224_compcan-v2 / sweep_jesus_cnn_compcan / run_jesus_10 ##
    cnn_input = Input(shape=(IMG_SIZE_4NN, IMG_SIZE_4NN, CHAN), name="cnn_input")
    ######################################################################
    ## CNN1
    cnn_hidden = Conv2D(96, (5, 5), activation="relu", name="cnn_hidden1")(cnn_input)
    maxpool = MaxPooling2D(pool_size=POOL_SIZE)(cnn_hidden)

    cnn_hidden = maxpool

    ## CNN inner   
    for i in range(in_CNNlayers):
        cnn_hidden = Conv2D(
            128,
            KERNEL_SIZE,
            activation="relu",
            name=f"cnn_hiddenInner{i+1}",
        )(cnn_hidden)

        ## Inner maxpool
        cnn_hidden = MaxPooling2D(pool_size=POOL_SIZE)(cnn_hidden)

    ## CNN2    
    cnn_hidden = Conv2D(
        64,
        KERNEL_SIZE,
        activation="relu",
        name="cnn_hidden2",
    )(cnn_hidden)
    maxpool = MaxPooling2D(pool_size=POOL_SIZE)(cnn_hidden)

    flatten = Flatten()(maxpool)

    dense_hidden = Dense(
        500, activation="relu", name="dense_hidden1"
    )(flatten)
    dense_hidden = Dropout(0.3)(dense_hidden)

    for i in range(in_DENSElayers):
        dense_hidden = Dense(
            200,
            activation="relu",
            name=f"dense_hiddenInner{i+1}",
        )(dense_hidden)
        dense_hidden = Dropout(0.0)(dense_hidden)

    dense_hidden = Dense(
        200, activation="relu", name="dense_hidden2"
    )(dense_hidden)
    dense_hidden = Dropout(0.5)(dense_hidden)

    ######################################################################

    cnn_output = Dense(len(UK_MEAS), activation="linear", name="cnn_output")(
        dense_hidden
    )

    ##returns model
    return Model(cnn_input, cnn_output)


## Here is the function that merges our two generators
## We use the exact same generator with the same random seed for both the front and side images
def generator2imgsNumData(datagen, MeasX, ImgXf, ImgXs, MeasY, batch_size):  # MeasY,
    """
    Generates data for training the model by combining numerical input, front view images, and side view images.

    Args:
        datagen (ImageDataGenerator): An instance of ImageDataGenerator for data augmentation.
        MeasX (array-like): Numerical input data.
        ImgXf (array-like): Front view image data.
        ImgXs (array-like): Side view image data.
        MeasY (array-like): Target variable data.
        batch_size (int): Number of samples to yield per batch.

    Yields:
        tuple: A tuple containing input data [numerical input, front view images, side view images] and the target variable data.
    """

    genNum = datagen.flow(ImgXf, MeasX, batch_size=batch_size, shuffle=False, seed=123)
    genF = datagen.flow(ImgXf, MeasY, batch_size=batch_size, shuffle=False, seed=123)
    genS = datagen.flow(ImgXs, MeasY, batch_size=batch_size, shuffle=False, seed=321)

    while True:
        Xni = genNum.next()
        Xfi = genF.next()
        Xsi = genS.next()
        yield [Xni[1], Xfi[0], Xsi[0]], Xfi[1]


def createCombined_model(MLP_model, CNN_model):
    """
    Creates the final combined model by merging the outputs of the MLP model and the CNN model
    """

    input_numca = Input(shape=INP_SHAPE, name="input_numca")
    input_front = Input((IMG_SIZE_4NN, IMG_SIZE_4NN, CHAN), name="input_front")
    input_side = Input((IMG_SIZE_4NN, IMG_SIZE_4NN, CHAN), name="input_side")

    output_numca = MLP_model(input_numca)
    output_front = CNN_model(input_front)
    output_side = CNN_model(input_side)

    ## Create the input to our final set of layers as the *output* of the MLP nad both CNNs
    combinedInput = concatenate(
        [output_numca, output_front, output_side], name="combined_input"
    )

    ## Our final FC layer head will have X dense layers, the final one being our regression head
    combined_hidden = Dense(len(UK_MEAS) * 2, activation="relu", name="combined_hidden")(combinedInput)

    combinedOutput = Dense(len(UK_MEAS), activation="linear", name="combined_output")(combined_hidden)

    return Model(inputs=[input_numca, input_front, input_side], outputs=combinedOutput)


def histplot(history, model_name, acc_metric="mean_absolute_error"):
    """
    Plots training and validation loss and accuracy for a model.
    Args:
        history: A Keras training history object.
        model_name: The name of the model being plotted.
        acc_metric: The accuracy metric to be plotted. Defaults to 'mean_absolute_error'.
    """

    hist = pd.DataFrame(history.history)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    if TEST_FILES == True:
        fig.suptitle(f"Model accuracy - SPRING+ANSUR_imgInputs DataAugmentation (test{TEST_FILES_NUM}))")
    else:
        fig.suptitle(f"Model accuracy - SPRING+ANSUR_imgInputs DataAugmentation")

    hist.plot(y=["loss", "val_loss"], ax=ax1)

    min_loss = hist["loss"].min()
    ax1.hlines(
        min_loss,
        0,
        len(hist),
        linestyle="dashed",
        label="min(loss) = {:.3f}".format(min_loss),
    )

    min_val_loss = hist["val_loss"].min()
    ax1.hlines(
        min_val_loss,
        0,
        len(hist),
        linestyle="dotted",
        label="min(val_loss) = {:.3f}".format(min_val_loss),
    )

    ax1.legend(loc="upper right")
    ax1.legend(
        [
            "loss - mse",
            "val_loss - mse",
            f"min(loss - mse) = {round(min_loss, 3)}",
            f"min(val_loss - mse) = {round(min_val_loss, 3)}",
        ]
    )

    hist.plot(y=[f"{acc_metric}", f"val_{acc_metric}"], ax=ax2)

    min_acc = hist[f"{acc_metric}"].min()
    ax2.hlines(
        min_acc,
        0,
        len(hist),
        linestyle="dashed",
        label=f"min({acc_metric})" + " = {:.3f}".format(min_acc),
    )

    min_val_acc = hist[f"val_{acc_metric}"].min()
    ax2.hlines(
        min_val_acc,
        0,
        len(hist),
        linestyle="dotted",
        label=f"min(val_{acc_metric})" + " = {:.3f}".format(min_val_acc),
    )

    ax2.legend(loc="upper right")  

    fig_name = model_name.split('.')[0] + "_hist"
    fig.savefig(os.path.join(MODEL_FILES_DIR, f"{fig_name}.png"))



if __name__ == "__main__":
  
  main()