import bpy
import os
import glob
import math

######################
GENDERS = ["female", "male"]
DATASETS = ['ANSURI', 'ANSURII']
VIEWS = ['front', 'side']

DATA_DIR = '/media/jesus/GSUS-DATA/CANADA_docs/'  \
                    'RaymondSpiteri_USASK_SK/Papers_SIMLAB_Jesus/'  \
                    'BodyMeasExtractionNN/human-body-reshape-DL-paper/data'

OBJ_FILES_DIR = os.path.join(DATA_DIR, "obj_files")

OBJ_FILES_DS_DIR_DICT = {
    name: os.path.join(OBJ_FILES_DIR, f"obj_database_{name}")
    for name in ['SPRING'] + DATASETS
}
######################

# Delete* Light!!!!!!!!
light = bpy.data.objects['Light']
light.location.x = 10
light.location.y = 0
light.location.z = -0.09
light.color = (0, 0, 0, 1)
light.data.energy = 0

# Set Camera!!!!!!!!
cam = bpy.data.objects['Camera']
cam.rotation_euler[0] = math.pi/2# math.pi/2
cam.rotation_euler[1] = math.pi/2 
cam.rotation_euler[2] = math.pi/2
cam.location.x = 3
cam.location.y = 0
cam.location.z = -0.09

RESOL = 400

DS_DIR_blender = os.path.join(DATA_DIR, "datasets")

SIL_FILES_DIR = os.path.join(DS_DIR_blender , f"silhouettes_blender{RESOL}")
try:
    os.mkdir(SIL_FILES_DIR) 
except OSError as error: 
    print(error)  

SIL_FILES_DIR_DICT = {
    name: os.path.join(SIL_FILES_DIR, f"silhouettes_{name}")
    for name in OBJ_FILES_DS_DIR_DICT.keys()
}

for name, sil_dir in SIL_FILES_DIR_DICT.items():
    if not os.path.exists(sil_dir):
        try:
            os.mkdir(sil_dir) 
        except OSError as error: 
            print(error)  

ROT_CAM_dict = {'front': (0, 0, math.radians(142)), 
                'side': (0, 0, math.radians(142) + math.pi/2)}


for ds_name in OBJ_FILES_DS_DIR_DICT.keys():

    for gender in GENDERS:

        obj_gender_dir = os.path.join(OBJ_FILES_DS_DIR_DICT[ds_name], gender)

        sil_gender_dir = os.path.join(SIL_FILES_DIR_DICT[ds_name], gender)
        try: 
            os.mkdir(sil_gender_dir) 
        except OSError as error: 
            print(error)

        # Specify OBJ files
        model_files = glob.glob(os.path.join(obj_gender_dir, "*.obj"))

        for view in VIEWS:

            sil_view_dir = os.path.join(sil_gender_dir, view)

            try: 
                os.mkdir(sil_view_dir) 
            except OSError as error: 
                print(error)

            for f in model_files:    ###[:5]
                head, tail = os.path.split(f)
                #print(head, tail)
                file_name = tail.split('.')[0]

                file_view_name = f'{file_name}_{view}.png'
                files_list = os.listdir(sil_view_dir)

                if file_view_name not in files_list:
            
                    # Call the obj import operator and pass the absolute file path
                    bpy.ops.import_scene.obj(filepath=f)
                    # Get all objects (create a copy of the entire list)
                    obj = bpy.context.selected_objects[0]
                    obj.rotation_euler = ROT_CAM_dict[view]

                    # Render Image
                    bpy.context.scene.render.resolution_x = RESOL
                    bpy.context.scene.render.resolution_y = RESOL
                    bpy.context.scene.render.film_transparent = True
                    bpy.context.scene.render.image_settings.color_mode = 'RGBA'

                    bpy.context.scene.render.filepath = os.path.join(sil_view_dir, file_view_name)
                    bpy.ops.render.render(write_still=True)
                    bpy.ops.object.delete(use_global=False, confirm=False)

                else:
                    print(f'File {file_view_name} already exists in {sil_view_dir}.')





