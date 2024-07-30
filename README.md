# human-body-reshape-DL-paper
Official Code for "Curbelo, J.P., Spiteri, R.J. A methodology for realistic human shape reconstruction from 2D images. Multimed Tools Appl (2024). https://doi.org/10.1007/s11042-023-17947-6"

![creating by deform-based global mapping](https://raw.githubusercontent.com/jpcurbelo/human-body-reshape-DL-paper/master/figures/Fig1.png)

If you want to explore this repo and eventually contribute, please, follow the instructions below.

## Getting Started

To get a local copy of the project, follow these steps:

### 1-Clone the repository:

`git clone https://github.com/jpcurbelo/human-body-reshape-DL-paper.git`

### 2-Navigate into the cloned directory:

`cd human-body-reshape-DL-paper/`

### 3-Create a virtual environment:

```bash
virtualenv venv
```
### 4-Activate the virtual environment:

```bash
source venv/bin/activate
```
### 5-Install the project dependencies:

```bash
pip install -r requirements.txt
```

## Folder Structure

Here's an overview of the folder structure after cloning the repository:

```plaintext
human-body-reshape-DL-paper/
├── data/
│   ├── body_reshaper_files/
│   │   └── [Files generated after training the Reshaper]
│   ├── cp_blender_files/
│   │   └── [*.obj and *.png files, and control-point-list files for each gender]
│   ├── datasets/
│   │   ├── ds_ansur_original/
│   │   └── [*.csv and *.npy files with measurements from the three datasets considered]
│   ├── input_files/
│   │   └── [Input files for the Extractor and to create the 3D avatar: measurements and full-body images]
│   ├── model_files/
│   │   ├── extractor_nn_model.h5
│   ├── output_files/
│   │   └── [Silhouettes/Avatar outputs files]
├── figures/
├── src/
│   ├── datasets/
│   │   ├── ansur2bodyfiles.py
│   │   └── ds_processer.py
│   ├── reshaper/
│   │   ├── avatar.py
│   │   ├── cp_handler.py
│   │   ├── tests_temp.py
│   │   ├── trainer.py
│   ├── extractor/
│   │   ├── extractor_model_training.py
│   └── utils.py
```

## Data files

To train and use the models, you will need the dataset files for avatars, silhouettes, and body measurements. These files can be downloaded from Zenodo: [3D Models and Silhouettes for Human Body Reshape with DL from the ANSUR Dataset (Version v3)](https://zenodo.org/records/11301099). Make sure to organize the downloaded files following the folder tree presented above.

### Organizing Data Files

Place the downloaded files in the `data/obj_files` directory following this structure:

```plaintext
data/
├── obj_files/
│   ├── obj_database_ANSURI/
│   │   ├── female/
│   │   ├── male/
│   ├── obj_database_ANSURII/
│   │   ├── female/
│   │   ├── male/
│   ├── obj_database_SPRING/
│   │   ├── female/
│   │   ├── male/
```

Note: The rest of the files to be downloaded include files generated after training the Reshaper model, *.obj files, and silhouettes generated to create the Benchmark dataset, and general files that might be helpful to reproduce the paper results.


## Running the Code

### 1 - Generating Avatars from Images and Measurements  

If you only want to generate the avatar using all the previously generated files and don't want/need to run the pretraining and other detailed steps, follow these instructions:

#### 1. Ensure Input Files are Correctly Placed:  

Place your input files in data/input_files:

```
input_info_extractor.csv
input_front.png
input_side.png
```

#### 2. Run the Avatar Generation Script:

Navigate to `src` and run:

```bash
python photos2avatar.py
```

#### 3. Check Output Files:

The generated output files, including silhouette images and the *wavefront.obj* file for the avatar, will be stored in `data/output_files`.


### 2 - Detailed Step-by-Step Instructions

If you want to run the full process, including pretraining and other detailed steps, follow these instructions:

#### 0. Clone the repo, download the data, and prepare your environment as described above.

#### 1. Reshaper Model Training: 

The `reshaper/` directory includes scripts to:

* Handle the control points on the *wavefront.obj* meshes to extract body measurements.
* Train the Reshaper model to generate 3D avatars from 21 body measurements (adapted from [3D-Human-Body-Shape](https://github.com/zengyh1900/3D-Human-Body-Shape)).
* The `Avatar` class is used in the main file. The `Imputer` is a method of the `Avatar` class.

**Folder Paths:**
* Control Points Handling: `src/reshaper/cp_handler.py`
* Reshaper Model Training: `src/reshaper/trainer.py`
* Avatar Class and Imputer: `src/reshaper/avatar.py`

Navigate to `src/reshaper` and run:

```bash
python trainer.py
```

By training the *Reshaper* model, the files needed to generated the 3D avatar from a set of 21 measurements will be created and stored in `data/body_reshaper_files/`. 


#### 2. Extractor Model Training.

This is the neural network explained in detail [in the paper](https://link.springer.com/article/10.1007/s11042-023-17947-6). The `extractor/` directory includes the script to build and train an MLP+CNN model to extract body measurements from two full-body images (front and side views).

**Folder Paths:**
* Model Building and Training: `src/extractor/extractor_model_training.py`

Navigate to `src/extractor` and run:

```bash
python extractor_model_training.py
```

By training the Extractor model, the neural network model will be created and stored in `data/model_files/`.


#### 3. Run the Avatar Generation Script:

Navigate to `src` and run:

```bash
python photos2avatar.py
```

## Error Handling and Troubleshooting

### Common Errors

- To be updated ...

### Seeking Help

- For additional help, you can open an issue on the [GitHub issues page](https://github.com/jpcurbelo/human-body-reshape-DL-paper/issues).