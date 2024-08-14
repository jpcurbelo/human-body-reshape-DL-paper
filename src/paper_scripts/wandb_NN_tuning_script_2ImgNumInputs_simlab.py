############################################################################
#Jesus, Oct 26th, 2022 
############################################################################

###Import modules###
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
#print(f"tensorflow version = {tf.__version__} \n")
from tensorflow import keras
from keras import backend as K   #tensorflow.keras
import pandas as pd
import numpy as np
import os
import random

from wandb_NN_tuning_utils_simlab import *

from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import (
    Input, 
    Conv2D, 
    MaxPooling2D, 
    Dense, 
    BatchNormalization,
    Flatten, 
    GlobalAveragePooling2D, 
    concatenate,
    Dropout
)

from tensorflow.keras.models import (
    Sequential, 
    Model
)

from tensorflow.keras.optimizers import (
    Adam,
    SGD
)

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error
)

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Import early stopping from keras callbacks
from tensorflow.keras.callbacks import EarlyStopping


# # https://www.youtube.com/watch?v=Hd94gatGMic
# # Set the random seeds
# os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
# random.seed(hash('setting random seeds') % 2**32 - 1)
# np.random.seed(hash('improves reproducibility') % 2**32 - 1)
# tf.random.set_seed(hash('by removing stochasticity') % 2**32 - 1)

# # The below is necessary for starting Numpy generated random numbers
# # in a well-defined initial state.
# np.random.seed(123)

# # The below is necessary for starting core Python generated random numbers
# # in a well-defined state.
# import random as python_random
# python_random.seed(123)

# # The below set_seed() will make random number generation
# # in the TensorFlow backend have a well-defined initial state.
# # For further details, see:
# # https://www.tensorflow.org/api_docs/python/tf/random/set_seed
# tf.random.set_seed(1234)


##Login to Weight&Biases
import wandb
#print(f"wandb version = {wandb.__version__} \n")
from wandb.keras import WandbCallback
wandb.login()    ##(create account in https://wandb.ai/site) - login with API = 68ceb21231b9f6e0d413aeb654070aeb7fcf8225


###############################Loading Data###############################################################
"""# Load data

SPRING DB
"""

## Load dataframe
file_encoding = "ISO-8859-1"    # 'utf8'
db_dir = f"{DATA_DIR}_SPRING/silhouettes_DB_SPRING.csv"
df_spring = pd.read_csv(db_dir, encoding = file_encoding, converters={'ID': str})

df_spring.drop(df_spring.columns.difference(UK_MEAS + KN_MEAS), 1, inplace=True)

"""ANSUR DB"""

## Load dataframe
df_ansur = list()
for i, gender in enumerate(GENDER_DICT.keys()):
  db_dir = f"{DATA_DIR}_ANSUR/silhouettes_DB_ANSUR_{gender}.csv"
  df_ansur.append(pd.read_csv(db_dir, encoding = file_encoding, converters={'ID': str}))
  df_ansur[i].drop(df_ansur[i].columns.difference(UK_MEAS + KN_MEAS), 1, inplace=True)

## Drop rows from "male" to have the same number of entries by gender
if DROP_EXTRA_MALE == True:
  remove_n = abs(len(df_ansur[1].index) - len(df_ansur[0].index)) - 14 # 14 female more than male in SPRING
  drop_indices = np.random.choice(df_ansur[1].index, remove_n, replace=False)
  df_ansur[1] = df_ansur[1].drop(drop_indices)

"""Replace gender key"""

## SPRING
for key in GENDER_DICT.keys():
  df_spring = df_spring.replace(key, GENDER_DICT[key])

## ANSUR
for i, gender in enumerate(GENDER_DICT.keys()):
  df_ansur[i]['gender'] = i

"""Concatenate DataFrames"""

df = pd.concat([df_spring, df_ansur[0], df_ansur[1]], axis=0)


"""SPRING Images"""

## NPZ files
img_spring_npz = np.load(open(os.path.join(f"{DATA_DIR}_SPRING", f'imagesXnparray{IMG_SIZE}_SPRING_bw_th.npz'), "rb"), allow_pickle=True)

img_spring = list()
for i, view in enumerate(VIEWS):
  img_spring.append(img_spring_npz['arr_0'][i,:,:,:])

## DELETE to free memory
# Delete npz arrays
try:
    del(img_spring_npz)
except NameError:
    print('img_spring_npz was already deleted')

"""ANSUR Images"""

# Load a list of [image]   
img_ansur = [[] for x in range(2)]  # 0-female, 1-male  

for g, gender in enumerate(GENDER_DICT.keys()):
  ## NPZ files
  img_ansur_npz = np.load(open(os.path.join(f"{DATA_DIR}_ANSUR", f'imagesXnparray{IMG_SIZE}_ANSUR_{gender}_bw_th.npz'), "rb"), allow_pickle=True)

  for i, view in enumerate(VIEWS):
    img_ansur[g].append(img_ansur_npz['arr_0'][i,:,:,:])

  ## DELETE to free memory
  # Delete npz arrays
  try:
      del(img_ansur_npz)
  except NameError:
      print('img_spring_npz was already deleted')

## Drop the images of subjects that are not in the ANSUR df 
## after dropping rows randomly - to match same number 
## of male and female in total.
if DROP_EXTRA_MALE == True:
  idx_to_drop = [x for x in range(len(img_ansur[1][0])) if x not in df_ansur[1].index]

  for i, view in enumerate(VIEWS):
    # for index in sorted(idx_to_drop, reverse=True):
    #   del img_ansur[1][i][index]
    img_ansur[1][i] = np.delete(img_ansur[1][i], idx_to_drop, 0)

"""Concatenate NPYarrays"""

imgX_front = np.concatenate((img_spring[0], img_ansur[0][0], img_ansur[1][0]), axis=0)

## DELETE to free memory
try:
    del(img_spring[0])
except NameError:
    print('img_spring[0] was already deleted')

try:
    del(img_ansur[0][0])
except NameError:
    print('img_spring[0] was already deleted')

try:
    del(img_ansur[1][0])
except NameError:
    print('img_spring[0] was already deleted')

imgX_side = np.concatenate((img_spring[0], img_ansur[0][0], img_ansur[1][0]), axis=0)

## DELETE to free memory
try:
    del(img_spring)
except NameError:
    print('img_spring[0] was already deleted')

try:
    del(img_ansur)
except NameError:
    print('img_spring[0] was already deleted')

try:
    del(img_ansur)
except NameError:
    print('img_spring[0] was already deleted')

## DELETE to free memory
try:
    del(df_spring)
except NameError:
    print('df_spring was already deleted')

try:
    del(df_ansur)
except NameError:
    print('df_ansur was already deleted')

"""# Prepare Numerical and Categorical Data"""

## Train and Test splitting
(trainData, testData, trainImgXf, testImgXf, trainImgXs, testImgXs) = train_test_split(df, imgX_front, imgX_side, train_size=0.85, shuffle='True', random_state=123)

## DELETE to free memory
try:
    del(imgX_front)
except NameError:
    print('imgX_front was already deleted')

try:
    del(imgX_side)
except NameError:
    print('imgX_side was already deleted')

## Normalize the UNKNOWN MEASUREMENTS (labels) - will lead to better training and convergence
trainMeasY = (trainData[UK_MEAS] - df[UK_MEAS].mean()) / df[UK_MEAS].std()   
testMeasY = (testData[UK_MEAS] - df[UK_MEAS].mean()) / df[UK_MEAS].std()

## Scales Train and Test values

(trainMeasX, testMeasX), dataMeasX_scaler = process_db_values(df, trainData, testData)

if 'gender' not in KN_MEAS:
  trainMeasX = np.delete(trainMeasX, 0, 1)
  testMeasX = np.delete(testMeasX, 0, 1)


#############################Building the model############################################
## Define the MLP network
def createMLP_model(in_MLPlayers=2):
    '''
    Creates the MLP model for the categorical+numerical inputs branch
    '''
    mlp_input = Input(shape=trainMeasX.shape[1])

    # mlp_hidden = Dense(wandb.config.mlp_nodes1, activation='relu')(mlp_input)

    # for i in range(in_MLPlayers):
    #   mlp_hidden = Dense(wandb.config.mlp_nodes_inner, activation='relu')(mlp_hidden)

    # mlp_hidden = Dense(wandb.config.mlp_nodes2, activation='relu')(mlp_hidden)

    # mlp_output = Dense(len(UK_MEAS), activation='linear')(mlp_hidden)

    
    mlp_hidden = Dense(16, activation='relu')(mlp_input)

    for i in range(in_MLPlayers):
      mlp_hidden = Dense(64, activation='relu')(mlp_hidden)

    mlp_hidden = Dense(64, activation='relu')(mlp_hidden)

    mlp_output = Dense(len(UK_MEAS), activation='linear')(mlp_hidden)

    ##returns model
    return Model(mlp_input, mlp_output)


##Create model
# MLP_model = createMLP_model(in_MLPlayers=wandb.config.mlp_inner_layers)


"""## CNN models (front + side)

### Data Augmentation
"""

#https://stackoverflow.com/questions/57092637/how-to-fit-keras-imagedatagenerator-for-large-data-sets-using-batches
#https://stackoverflow.com/questions/49404993/how-to-use-fit-generator-with-multiple-inputs

# we create two instances with the same arguments
data_gen_args = dict(
                    featurewise_center=True,
                    featurewise_std_normalization=True,
                    # rotation_range=0,   
                    width_shift_range=0.1,  
                    height_shift_range=0.25,  
                    shear_range=3,
                    zoom_range=[0.8, 1.2], 
                    horizontal_flip=False)

## compute quantities required for featurewise normalization
## (std, mean, and principal components if ZCA whitening is applied)
datagen = ImageDataGenerator(**data_gen_args)

datagen.fit(trainImgXf, augment=True)

### https://stackoverflow.com/questions/49404993/how-to-use-fit-generator-with-multiple-inputs
### https://stackoverflow.com/questions/231767/what-does-the-yield-keyword-do
### https://www.kaggle.com/code/sinkie/keras-data-augmentation-with-multiple-inputs/notebook

## Here is the function that merges our two generators
## We use the exact same generator with the same random seed for both the front and side images
def generator2imgsNumData(MeasX, ImgXf, ImgXs, MeasY, batch_size):  #MeasY, 

  genNum = datagen.flow(ImgXf, MeasX, batch_size=batch_size, shuffle=False, seed=123)
  genF = datagen.flow(ImgXf, MeasY, batch_size=batch_size, shuffle=False, seed=123)
  genS = datagen.flow(ImgXs, MeasY, batch_size=batch_size, shuffle=False, seed=321)

  # yield genF.next()

  while True:
    Xni = genNum.next()
    Xfi = genF.next()
    Xsi = genS.next()
    yield [Xni[1], Xfi[0], Xsi[0]], Xfi[1]


"""### CNN models"""

## Define the CNN_model

POOL_SIZE = (3,3)   #  (3,3)

def createCNN_model(in_CNNlayers=2, in_DENSElayers=1):
    '''
    '''
    ## AlexNet-wise ##
    cnn_input = Input(shape=(IMG_SIZE, IMG_SIZE, CHAN))
    ######################################################################
    cnn_hidden = Conv2D(wandb.config.cnn_filters1, (5,5), activation='relu')(cnn_input)
    maxpool = MaxPooling2D(pool_size=POOL_SIZE)(cnn_hidden)

    cnn_hidden = Conv2D(wandb.config.cnn_filters2, (3,3), activation='relu')(maxpool)
    maxpool = MaxPooling2D(pool_size=POOL_SIZE)(cnn_hidden)

    for i in range(in_CNNlayers):
        cnn_hidden = Conv2D(wandb.config.cnn_filters_inner, (3,3), activation='relu')(cnn_hidden)

    cnn_hidden = Conv2D(wandb.config.cnn_filters3, (3,3), activation='relu')(cnn_hidden)
    maxpool = MaxPooling2D(pool_size=POOL_SIZE)(cnn_hidden)

    flatten = Flatten()(maxpool)

    dense_hidden = Dense(wandb.config.dense_nodes1, activation='relu')(flatten)
    dense_hidden = Dropout(wandb.config.drop1)(dense_hidden)

    for i in range(in_DENSElayers):
        dense_hidden = Dense(wandb.config.dense_nodes_inner, activation='relu')(dense_hidden)
        dense_hidden = Dropout(wandb.config.drop_inner)(dense_hidden)

    dense_hidden = Dense(wandb.config.dense_nodes2, activation='relu')(dense_hidden)
    ######################################################################


    # ######################################################################
    # cnn_hidden = Conv2D(wandb.config.cnn_filters1, (11,11), activation='relu', strides=4)(cnn_input)
    # maxpool = MaxPooling2D(pool_size=POOL_SIZE, strides=2)(cnn_hidden)

    # cnn_hidden = Conv2D(wandb.config.cnn_filters2, (5,5), activation='relu', padding="same")(maxpool)
    # maxpool = MaxPooling2D(pool_size=POOL_SIZE, strides=2)(cnn_hidden)

    # for i in range(in_CNNlayers):
    #     cnn_hidden = Conv2D(wandb.config.cnn_filters_inner, (3,3), activation='relu', padding="valid")(cnn_hidden)

    # cnn_hidden = Conv2D(wandb.config.cnn_filters3, (3,3), activation='relu', padding="valid")(cnn_hidden)
    # maxpool = MaxPooling2D(pool_size=POOL_SIZE, strides=2)(cnn_hidden)

    # flatten = Flatten()(maxpool)

    # dense_hidden = Dense(wandb.config.dense_nodes1, activation='relu')(flatten)
    # # dense_hidden = Dropout(0.3)(dense_hidden)

    # for i in range(in_DENSElayers):
    #     dense_hidden = Dense(wandb.config.dense_nodes_inner, activation='relu')(dense_hidden)
    #     # dense_hidden = Dropout(0.3)(dense_hidden)

    # dense_hidden = Dense(wandb.config.dense_nodes2, activation='relu')(dense_hidden)


    # dense_hidden = Dense(200, activation='relu')(dense_hidden)
    # ######################################################################
    
    
    
    ######################################################################
    cnn_output = Dense(len(UK_MEAS), activation='linear')(dense_hidden)
    ######################################################################
    return Model(cnn_input, cnn_output)

##Create model
# CNN_model = createCNN_model(in_CNNlayers=wandb.config.cnn_inner_layers, in_DENSElayers=wandb.config.dense_inner_layers)


"""### Combine and Compile"""


def createCombined_model():
    '''
    '''
    # MLP_model = createMLP_model(in_MLPlayers=wandb.config.mlp_inner_layers)
    MLP_model = createMLP_model(in_MLPlayers=2)
    print(MLP_model.summary())

    CNN_model = createCNN_model(in_CNNlayers=wandb.config.cnn_inner_layers, 
                                in_DENSElayers=wandb.config.dense_inner_layers)
    print(CNN_model.summary())

    input_numca = Input(shape=trainMeasX.shape[1]) 
    input_front = Input((IMG_SIZE, IMG_SIZE, CHAN))
    input_side= Input((IMG_SIZE, IMG_SIZE, CHAN))

    output_numca = MLP_model(input_numca)
    output_front = CNN_model(input_front)
    output_side = CNN_model(input_side)

    ## Create the input to our final set of layers as the *output* of the MLP nad both CNNs
    combinedInput = concatenate([output_numca, output_front, output_side])

    ## Our final FC layer head will have X dense layers, the final one being our regression head
    combined_hidden = Dense(len(UK_MEAS) * 2, activation="relu")(combinedInput)

    combinedOutput = Dense(len(UK_MEAS), activation="linear")(combined_hidden)

    return Model(inputs=[input_numca, input_front, input_side], outputs=combinedOutput)



## Specify the hyperparameter to be tuned along with
## an initial value
config_defaults = {
    "name": "sweep_jesus_cnn1",
    # 'batch_size': 32,
    # 'learning_rate': 1e-3,
    # "epochs": 20,

    # "mlp_nodes1": 16,
    # "mlp_inner_layers": 2,
    # "mlp_nodes_inner": 64,
    # "mlp_nodes2": 64,

    "cnn_filters1": 32,
    "cnn_filters2": 64,
    "cnn_inner_layers": 1,
    "cnn_filters_inner": 64,
    "cnn_filters3": 128,

    "dense_nodes1": 1000,
    "drop1": 0.5,
    "dense_inner_layers": 1,
    "dense_nodes_inner": 2000,
    "drop_inner": 0.5,
    "dense_nodes2": 100,

    # "optimizer": "adam",
    # "loss": "mse"
}

run_number = 0
# ############################Trainning the model###########################################################
def train():

    reset_seeds() 

    # Initialize wandb with a sample project name
    with wandb.init(config=config_defaults) as run:

        global run_number
        run_number += 1
        # Overwrite the random run names chosen by wandb
        name_str = f'run_jesus_{run_number}'
        run.name = name_str

        # Initialize model with hyperparameters
        keras.backend.clear_session()
        model = createCombined_model()

        print(model.summary())

        # Train and validation generators
        BATCH_SIZE = 32
        # training_data_gen = generator2imgsNumData(trainMeasX, trainImgXf, trainImgXs, trainMeasY, wandb.config.batch_size)
        # validation_data_gen = generator2imgsNumData(testMeasX, testImgXf, testImgXs, testMeasY, wandb.config.batch_size)
        training_data_gen = generator2imgsNumData(trainMeasX, trainImgXf, trainImgXs, trainMeasY, BATCH_SIZE)
        validation_data_gen = generator2imgsNumData(testMeasX, testImgXf, testImgXs, testMeasY, BATCH_SIZE)
       
       
        # # Compile the model
        # opt= build_optimizer(wandb.config.optimizer, wandb.config.learning_rate)
        opt= build_optimizer('adam', 0.001)
        # lo = wandb.config.loss
        lo = 'mse'
        model.compile(loss=lo, optimizer=opt, metrics='mean_absolute_error')

        ## Instantiate an early stopping callback
        early_stopping = EarlyStopping(monitor='val_loss', min_delta = 1e-4, patience=5, restore_best_weights=True)

        # Define WandbCallback for experiment tracking
        wandb_callback = WandbCallback(monitor='val_loss', save_model=False)

        # callbacks
        callbacks = [early_stopping, wandb_callback]   # [early_stopping, wandb_callback]  #PrintLogs(wandb.config.epochs)
    
    
        ## Train the model
        print("[INFO] training model...")
        _ = model.fit(

                training_data_gen, 
                validation_data=(validation_data_gen),
                validation_steps=testImgXf.shape[0] // BATCH_SIZE,   #wandb.config.batch_size,
                steps_per_epoch = trainImgXf.shape[0] // BATCH_SIZE,   #wandb.config.batch_size,
                verbose = 2, 
                epochs = 100,   #wandb.config.epochs, 
                callbacks = callbacks
            )


sweep_config = {
    "name": "sweep_jesus_cnn_drop2",
    'method': 'bayes',  # 'random'
    'metric': {
        'name': 'mean_absolute_error',  # 'val_loss', 'val_accuracy' or others
        'goal': 'minimize'   #'minimize'
    },
    'early_terminate':{
        'type': 'hyperband',
        # 'min_iter': 2   # 5
        's': 2,
        'eta': 3,
        'max_iter': 27
    },
    'parameters': {
        # 'batch_size': {
        #         "values": [32]   # [16, 32]
        # },
        # 'learning_rate':{
        #         "values": [1e-3]  # [1e-4, 1e-3]   # 
        # },
        # "epochs":{
        #         "values": [100]  # [50, 100, 200, 300]
        # },
        # "mlp_nodes1":{
        #     "values": [16]   # [16]   # [16, 32, 64]
        # },
        # "mlp_inner_layers":{
        #     "values": [2]    # [1, 2, 3]
        # },
        # "mlp_nodes_inner":{
        #     "values": [64]   # [16, 32, 64]
        # },
        # "mlp_nodes2":{
        #     "values": [64]   # [16, 32, 64]
        # },
        "cnn_filters1":{
            "values": [64, 96, 128, 256]   #[64, 96, 128]   #[96]  # [32, 64, 128]
        },
        "cnn_filters2":{
            "values": [64, 96, 128, 256]   #[64, 96, 128]   #[256]   # [64, 128, 256]
        },
        "cnn_inner_layers":{
            "values": [1, 2]   #[1, 2]   #[2]   # [1, 2, 3]
        },
        "cnn_filters_inner":{
            "values": [64, 96, 128, 256]   #[64, 96, 128, 256]   #[384]   # [64, 128, 256, 512]
        },
        "cnn_filters3":{
            "values": [64, 96, 128]   #[64, 96, 128]   #[256]   # [64, 128, 256]
        },

        "dense_nodes1":{
            "values": [1000, 2000, 3000]  #[1000, 2000]   #[4096]   # [200, 500, 1000]
        },
        "drop1": {
            "values": [0.0, 0.2, 0.3, 0.5] 
        },
        "dense_inner_layers":{
            "values": [1, 2]   #[1, 2]   # [1]  # [1, 2, 3]
        },
        "dense_nodes_inner":{
            "values":  [1000, 2000, 3000]   #[1000, 2000, 3000]   #[4096]   # [200, 500, 1000]
        },
        "drop_inner": {
            "values": [0.0, 0.2, 0.3, 0.5] 
        },
        "dense_nodes2":{
            "values": [100, 200, 500]  #[100, 200]   #[1000]   # [50, 100, 200]
        },
        # "optimizer":{
        #     "values": ["adam"]   # ["adam"]   # ["adam", "sgd"]
        # },
        # "loss":{
        #     # "values": ["mean_squared_error", "mean_absolute_error", "mean_absolute_percentage_error"]
        #     "values": ["mse"]
        # }
  }
}


def main():
    '''
    '''
    sweep_id = wandb.sweep(sweep_config, project=f"MLP-CNN_silouettes2measurements_{IMG_SIZE}_simlab", entity="jpcurbelo")

    wandb.agent(sweep_id, function=train, count=500)



if __name__ == "__main__":
    
    main()