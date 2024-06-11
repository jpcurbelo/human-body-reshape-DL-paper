import pandas as pd
import numpy as np


import sys
sys.path.append("..")
from utils import *


DS_DIR = "../../data/datasets/"
DS_ANSUR_DIR = os.path.join(DS_DIR, "ds_ansur_original")

ANSURI_MEAS = ['WEIGHT',
              'STATURE',
              'NECK_CIRC-BASE',
              'CHEST_CIRC',
              'WAIST_CIRC-OMPHALION',  #'WAIST_CIRC_NATURAL',
              'BUTTOCK_CIRC',
              'SHOULDER_CIRC',
              'THIGH_CIRC-PROXIMAL',
              'THIGH_CIRC-DISTAL',
              'CALF_CIRC',
              'ANKLE_CIRC',
              'FOREARM_CIRC-FLEXED',
              'WRIST_CIRC-STYLION',
              'SHOULDER_LNTH',
              'SLEEVE-OUTSEAM_LNTH',
              'RADIALE-STYLION_LNTH',
              'CROTCH_HT',
              'WAIST_NAT_LNTH',
              'THIGH_LINK',  ## needs to be added ((D37) THIGH_LINK = TROCHANTERION_HT - LATERAL_FEMORAL_EPICONDYLE_HT)
              'CHEST_DEPTH',
              'HEAD_CIRC']

ANSURII_MEAS = ['weightkg',
              'stature',
              'neckcircumferencebase', 
              'chestcircumference', 
              'waistcircumference',
              'buttockcircumference', 
              'shouldercircumference',
              'thighcircumference',
              'lowerthighcircumference',
              'calfcircumference', 
              'anklecircumference',
              'forearmcircumferenceflexed', 
              'wristcircumference', 
              'shoulderlength',
              'sleeveoutseam',
              'radialestylionlength',
              'crotchheight',
              'waistbacklength', 
              'thighlink', 
              'chestdepth',
              'headcircumference']

DS_MEAS = {
    DATASETS[0]: ANSURI_MEAS,
    DATASETS[1]: ANSURII_MEAS
}

NEW_NAMES_DICT_ansurI = {ANSURI_MEAS[i]: MEASUREMENTS[i] for i in range(len(ANSURI_MEAS))}
NEW_NAMES_DICT_ansurII = {ANSURII_MEAS[i]: MEASUREMENTS[i] for i in range(len(ANSURI_MEAS))}


def generateANSURfiles():

    for ds in DATASETS:
        for gender in GENDERS:
            dataset = load_ds(ds, gender)

            # Substitutions, renames and drops
            df = filter_ds(dataset, ds)

            # Save to csv
            file_path = os.path.join(DS_DIR, f"measurements_{ds}_{gender}.csv")
            df.to_csv(file_path, index=False)
            
            
def load_ds(ds, gender):

    ds_dir = os.path.join(DS_ANSUR_DIR, f"{ds}_{gender}.csv")
    df = pd.read_csv(ds_dir, encoding = FILE_ENCODING, converters={'ID': str})

    return df
    

def filter_ds(df, ds):

    # Thigh length
    if ds == 'ANSURI':
        # (D37) THIGH_LINK = TROCHANTERION_HT - LATERAL_FEMORAL_EPICONDYLE_HT
        df['THIGH_LINK'] = df.apply(lambda row: row.TROCHANTERION_HT - row.LATERAL_FEMORAL_EPICONDYLE_HT, axis=1)
        # Rename columns and only consider MEASUREMENTS
        df = df.rename(columns = NEW_NAMES_DICT_ansurI, inplace = False)
    else:
        # d29 = 'trochanterionheight' - 'lateralfemoralepicondyleheight'
        df['thigh_length'] = df.apply(lambda row: row.trochanterionheight - row.lateralfemoralepicondyleheight, axis=1)
        # Rename columns and only consider MEASUREMENTS
        df = df.rename(columns = NEW_NAMES_DICT_ansurII, inplace = False)

    # Drop columns that are not MEASUREMENTS
    # Divide by 10 to convert to cm
    df = df[MEASUREMENTS] / 10.0

    return df


def generateTOTALfiles():

    total_data = []  # List to store DataFrames for each gender
    for gender in GENDERS:
        
        #SPRING
        ds_dir = os.path.join(DS_DIR, f"measurements_SPRING_{gender}.csv")
        df_spring = pd.read_csv(ds_dir, encoding = FILE_ENCODING, converters={'ID': str})
        df_spring = df_spring[MEASUREMENTS]
        total_data.append(df_spring)
        # Save as npy file
        file_path_npy = os.path.join(DS_DIR, f"measurements_SPRING_{gender}.npy")
        np.save(file_path_npy, df_spring.to_numpy())

        #ANSURI
        ds_dir = os.path.join(DS_DIR, f"measurements_ANSURI_{gender}.csv")
        df_ansurI = pd.read_csv(ds_dir, encoding = FILE_ENCODING, converters={'ID': str})
        total_data.append(df_ansurI)
        # Save as npy file
        file_path_npy = os.path.join(DS_DIR, f"measurements_ANSURI_{gender}.npy")
        np.save(file_path_npy, df_ansurI .to_numpy())

        #ANSURII
        ds_dir = os.path.join(DS_DIR, f"measurements_ANSURII_{gender}.csv")
        df_ansurII = pd.read_csv(ds_dir, encoding = FILE_ENCODING, converters={'ID': str})
        total_data.append(df_ansurII)
        # Save as npy file
        file_path_npy = os.path.join(DS_DIR, f"measurements_ANSURII_{gender}.npy")
        np.save(file_path_npy, df_ansurII.to_numpy())

        # Concatenate all DataFrames into a single DataFrame
        total_dataframe = pd.concat(total_data, ignore_index=True)

        # Save the concatenated DataFrame to a new CSV file
        file_path = os.path.join(DS_DIR, f"measurements_TOTAL_{gender}.csv")
        total_dataframe.to_csv(file_path, index=False, encoding=FILE_ENCODING)

        # Save the concatenated DataFrame as npy file
        file_path_npy = os.path.join(DS_DIR, f"measurements_TOTAL_{gender}.npy")
        np.save(file_path_npy, total_dataframe.to_numpy())

# ===========================================================================    

if __name__ == "__main__":
  
#   generateANSURfiles()

  generateTOTALfiles()