import os

CP_FILES_DIR = os.path.join("../../data", "cp_blender_files")
RESHAPER_FILES_DIR = os.path.join("../../data", "body_reshaper_files") 
OBJ_FILES_DIR = os.path.join("../../data", "obj_files")

OBJ_FILES_SPRING = os.path.join(OBJ_FILES_DIR, "obj_database_SPRING")
OBJ_FILES_ANSURI = os.path.join(OBJ_FILES_DIR, "obj_database_ANSURI")
OBJ_FILES_ANSURII = os.path.join(OBJ_FILES_DIR, "obj_database_ANSURII")

OUTPUT_FILES_DIR = os.path.join("../../data", "output_files")
INPUT_FILES_DIR = os.path.join("../../data", "input_files")

DS_DIR = "../../data/datasets/"
FILE_ENCODING = "ISO-8859-1"    # 'utf8'

GENDERS = ["female", "male"]
DATASETS = ['ANSURI', 'ANSURII']

MEASUREMENTS = ['weight_kg',  # 'height_cm', 'weight_kg
                'stature_cm', 
                'neck_base_girth', 
                'chest_girth',
                'waist_girth',
                'hips_buttock_girth',
                'shoulder_girth',
                'thigh_girth',
                'thigh_low_girth',
                'calf_girth',
                'ankle_girth',
                'forearm_girth',
                'wrist_girth',
                'shoulder_length',
                'sleeveoutseam_length',
                'forearm_length',
                'crotchheight_length',
                'waistback_length',
                'thigh_length',
                'chest_depth_length',
                'head_girth']

M_NUM = len(MEASUREMENTS)

'''
labels:
    1 - vertical length
    2 - horizontal girth
    3 - point to point length
    4 - point to point girth    
'''
MEAS_LABELS = { 'stature_cm': 1, 
                'neck_base_girth': 2,   #4, 
                'chest_girth': 2,
                'waist_girth': 2,
                'hips_buttock_girth': 2,
                'shoulder_girth': 2,
                'thigh_girth': 2,
                'thigh_low_girth': 2,
                'calf_girth': 2,
                'ankle_girth': 2,
                'forearm_girth': 2,   #2,   #4,
                'wrist_girth': 2,
                'shoulder_length': 3,
                'sleeveoutseam_length': 3,
                'forearm_length': 3,
                'crotchheight_length': 1,
                'waistback_length': 3,
                'thigh_length': 3,   #3,
                'chest_depth_length': 3,
                'head_girth': 2}

# KHUMANBODY_DENSITY = 1026.0   # 3DHBSh
# 1 010 kilograms [kg] of human body fit into 1 cubic meter
# https://www.aqua-calc.com/calculate/weight-to-volume/substance/human-blank-body
KHUMANBODY_DENSITY = 1026.0   #1010.0	   # jesus

V_NUM = 12500
F_NUM = 25000
D_BASIS_NUM = 10
V_BASIS_NUM = 10

ROTATION_DICT = {"x": 0, "y": 0, "z": 52}

def load_input_data():
    '''
    Import input data (gender and measurments)
    Returns 'gender', abd a list of measurements
    '''
    f = open(os.path.join(INPUT_FILES_DIR, f'input_data_avatar.csv'))
    
    measurements = list()
    first_line = f.readline()
    gender = first_line.strip('\n').split(',')[1]   
    for line in f:
        measurements.append(float(line.split(',')[1]))
     
    f.close()
     
    return gender, measurements



##############################