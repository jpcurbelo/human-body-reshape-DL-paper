import matplotlib.pyplot as plt
import numpy as np
np.random.seed(123)
import random
import os
import pandas as pd
import cv2 as cv

###Import modules###
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.layers import (
    Flatten, 
    GlobalAveragePooling2D
)

from tensorflow.keras.optimizers import (
    Adam,
    SGD
)

from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler,
    MinMaxScaler
)



### CONSTANTS ###
VIEWS = ['front', 'side']
IMG_SIZE = 128   #64   #224
CHAN = 1

DATA_DIR = "SilhouettesNN"
MODELS_DIR = "ModelsNN_tests2023"
INPUT_DIR = "SilhouettesNN_Input"

DBNAMES = ["SPRING2023", "ANSURI2023", "ANSURII2023"]

LIST_MEASUREMENTS = ['gender',
                     'crotchheight', 
                     'buttockcircumference',
                     'chestcircumference',
                     'thighcircumference',
                     'waistcircumference',
                     'weightkg',
                     'sleeveoutseam',
                     'stature',
                     'waistbacklength',
                     'shouldercircumference'
                    ]

KN_MEAS = ['gender',
           'weightkg', 
           'stature']

UK_MEAS = [
            'chestcircumference',
            'waistcircumference',
            'buttockcircumference',
            'thighcircumference',
            'sleeveoutseam',
            'waistbacklength',
            'shouldercircumference',
            'crotchheight'
          ]

CATEGORICAL = ['gender']
GENDER_DICT = {'female': 0, 'male': 1}
CONTINUOUS = [x for x in KN_MEAS if x not in CATEGORICAL]

COLUMNS_NEW_NAMES = {'weight_kg': 'weightkg', 'stature_cm': 'stature',
                      'chest_girth': 'chestcircumference', 'waist_girth': 'waistcircumference',
                      'hips_buttock_girth': 'buttockcircumference', 'thigh_girth': 'thighcircumference',
                      'waistback_length': 'waistbacklength', 'crotchheight_length': 'crotchheight', 
                      'sleeveoutseam_length': 'sleeveoutseam', 'shoulder_girth': 'shouldercircumference'
                    }

EPOCHS = 200
KERNEL_SIZE = 3

BITWISE = True
THRESHOLD = True
DROP_EXTRA_MALE = True


### FUNCTIONS ###
def plot_images(images, numbOfImg, view, gender='unknown'):
    '''
    Plots numbOfImg images from img_data
    '''
    plt.figure(figsize=(10,10))

    # for i, img in enumerate(random.sample(images, numbOfImg)):
    for i, img in enumerate(images[:numbOfImg]):
        plt.subplot(5, 5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.xlabel(f"{i+1}-{gender}-{view}")
        plt.imshow(img.reshape(IMG_SIZE, IMG_SIZE), cmap='gray')

    plt.show()

def build_optimizer(optimizer, learning_rate):
    
    if optimizer == "sgd":
        optimizer = SGD(learning_rate=learning_rate)
    elif optimizer == "adam":
        optimizer = Adam(learning_rate=learning_rate)
    return optimizer

def process_db_values(df, train, test):
    '''
    Performs min-max scaling each continuous feature column to the range [0, 1]
    '''

    cs = MinMaxScaler()
    trainContinuous = cs.fit_transform(train[CONTINUOUS])  #train[KN_MEAS - CATEGORICAL]
    testContinuous = cs.transform(test[CONTINUOUS])   #test[KN_MEAS - CATEGORICAL]

    # one-hot encode the GENDER categorical data (by definition of
    # one-hot encoding, all output features are now in the range [0, 1])
    trainCategorical = keras.utils.to_categorical(train[CATEGORICAL], len(GENDER_DICT.keys()))
    testCategorical = keras.utils.to_categorical(test[CATEGORICAL], len(GENDER_DICT.keys()))
  
    # construct our training and testing data points by concatenating
    # the categorical features with the continuous features
    trainX = np.hstack([trainCategorical, trainContinuous])
    testX = np.hstack([testCategorical, testContinuous])

    # return the concatenated training and testing data
    return (trainX, testX), cs

def plotModel(NN_model):
    '''
    '''
    tf.keras.utils.plot_model(NN_model, to_file ='model.png', show_shapes=True, show_layer_names=True)

    plt.figure(figsize=(10,10))

    img = plt.imread('model.png')
    plt.imshow(img)
    plt.show()

def histplot(history, MODEL_NAME, acc_metric = 'mean_absolute_error'):

    hist = pd.DataFrame(history.history)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(f'Model accuracy - SPRING+ANSUR_imgInputs DataAugmentation')
    

    hist.plot(y=['loss','val_loss'], ax=ax1)

    min_loss = hist['loss'].min()
    ax1.hlines(min_loss, 0, len(hist), linestyle='dashed',
                label='min(loss) = {:.3f}'.format(min_loss))

    min_val_loss = hist['val_loss'].min()
    ax1.hlines(min_val_loss, 0, len(hist), linestyle='dotted',
                label='min(val_loss) = {:.3f}'.format(min_val_loss))
  
    
    ax1.legend(loc='upper right')
    ax1.legend(['loss - mse', 'val_loss - mse', f'min(loss - mse) = {round(min_loss, 3)}', f'min(val_loss - mse) = {round(min_val_loss, 3)}'])
    

    hist.plot(y=[f'{acc_metric}', f'val_{acc_metric}'], ax=ax2)

    min_acc = hist[f'{acc_metric}'].min()
    ax2.hlines(min_acc, 0, len(hist), linestyle='dashed',
                label=f'min({acc_metric})' + ' = {:.3f}'.format(min_acc))

    min_val_acc = hist[f'val_{acc_metric}'].min()
    ax2.hlines(min_val_acc, 0, len(hist), linestyle='dotted',
                label=f'min(val_{acc_metric})' + ' = {:.3f}'.format(min_val_acc))
    
    ax2.legend(loc='upper right')  #, fontsize='large'


    fig_name = MODEL_NAME[:-3]
    fig.savefig(os.path.join(MODELS_DIR, f"{fig_name}.png"))

def reset_seeds():
    os.environ["PYTHONHASHSEED"] = str(123)
    # np.random.seed(123) 
    # random.seed(123)
    # tf.random.set_seed(1234)
    # os.environ['TF_CUDNN_DETERMINISTIC'] = str(1)
    np.random.seed(hash('setting random seeds') % 2**32 - 1)
    random.seed(hash('improves reproducibility') % 2**32 - 1)
    tf.random.set_seed(hash('by removing stochasticity') % 2**32 - 1)

def load_images(files_dir, resize = False):
  '''
  '''
  
  img_data = list()
  files_list = sorted(os.listdir(files_dir))  

  for i, file in enumerate(files_list):

    img = cv.imread(os.path.join(files_dir, file), cv.IMREAD_GRAYSCALE)
    if resize == True:
      img = cv.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.reshape(IMG_SIZE, IMG_SIZE, CHAN)

    img_data.append(np.array(img).astype(float) / 255)

  return img_data

class PrintLogs(tf.keras.callbacks.Callback):
    '''
    https://stackoverflow.com/questions/55422711/keras-training-progress-bar-on-one-line-with-epoch-number
    '''
    def __init__(self, epochs):
        self.epochs = epochs

    def set_params(self, params):
        params['epochs'] = 0

    def on_epoch_begin(self, epoch, logs=None):
        print('Epoch %d/%d' % (epoch + 1, self.epochs), end=' --> ')