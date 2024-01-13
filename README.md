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

#### Folder tree
```plaintext
human-body-reshape-DL-paper/
├── data/
│ ├── body_reshaper_files/
│ │ └── [Files for the Body Reshaper]
│ ├── cp_blender_files/
│ │ └── [Files for CP Blender]
│ ├── datasets/
│ │ ├── ds_ansur_original/
│ │ │ └── [Original ANSUR datasets]
│ │ └── [Datasets files]
│ ├── input_files/
│ │ └── [Input files for processing]
│ ├── obj_files/
│ │ ├── obj_database_SPRING/
│ │ │ ├── female/
│ │ │ │ └── [Female OBJ files]
│ │ │ ├── male/
│ │ │ │ └── [Male OBJ files]
│ │ └── [Other OBJ files]
│ ├── output_files/
│ │ └── [Output files]
├── figures/
│ └── [Figures for the paper]
├── src/
│ ├── datasets/
│ │ ├── ansur2bodyfiles.py
│ │ └── ds_processer.py
│ ├── reshaper/
│ │ ├── avatar.py
│ │ ├── cp_handler.py
│ │ ├── tests_temp.py
│ │ ├── trainer.py
│ └── utils.py
```


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

## Reshaper

This directory includes scripts to:

* Handle the control points on the *wavefront.obj* meshes to extract body measurements.
* Train the Reshaper model to generate 3D avatars from 21 body measurements (adapted from [3D-Human-Body-Shape](https://github.com/zengyh1900/3D-Human-Body-Shape)).
* The `Avatar` class is used in the main file. The `Imputer` is a method of the `Avatar` class.

**Folder Paths:**
* Control Points Handling: `src/reshaper/cp_handler.py`
* Reshaper Model Training: `src/reshaper/trainer.py`
* Avatar Class and Imputer: `src/reshaper/avatar.py`

## Extractor

This directory includes the script to build and train an MLP+CNN model to extract body measurements from two (front and side views) full-body images.

**Folder Paths:**
* Model Building and Training: `src/extractor/extractor_model_training.py`

## Input and Output Files

Input files are to be located in `data/input_files` and consist of:

* `input_info_extractor.csv`
* `input_front.png`
* `input_side.png`
* `input_data_avatar.csv` (generated after the 'extraction' is performed)

Output files are to be stored in `data/output_files` and may consist of silhouette images extracted and the *wavefront.obj* file for the avatar.

## Generating Avatars from Images

To try the code, navigate to `src` and run:

```bash
python photos2avatar.py
