# human-body-reshape-DL-paper
Official Code for "A methodology for realistic human shape reconstruction from 2D images"

![creating by deform-based global mapping](https://raw.githubusercontent.com/jpcurbelo/human-body-reshape-DL-paper/master/figures/Fig1.png)

If you want to explore this repo and eventually contribute, please, follow the instructions below.

## Getting Started

To get a local copy of the project, follow these steps:

### 1-Clone the repository:

`git clone https://github.com/jpcurbelo/human-body-reshape-DL-paper.git`

### 2-Navigate into the cloned directory:

`cd human-body-reshape-DL-paper/`

#### Folder tree
human-body-reshape-DL-paper/
├── data/
│   ├── body_reshaper_files/
│   │   └── [Files for the Body Reshaper]
│   ├── cp_blender_files/
│   │   └── [files for CP Blender]
│   ├── datasets/
│   │   ├── ds_ansur_original/
│   │   │   └── [original ANSUR datasets]
│   │   └── [datasets files]
│   ├── input_files/
│   │   └── [input files for processing]
│   ├── obj_files/
│   │   ├── obj_database_SPRING/
│   │   │   ├── female/
│   │   │   │   └── [female OBJ files]
│   │   │   ├── male/
│   │   │   │   └── [male OBJ files]
│   │   └── [other OBJ files]
│   ├── output_files/
│   │   └── [output files]
├── figures/
│   └── [figures for the paper]
├── src/
│   ├── datasets/
│   │   ├── ansur2bodyfiles.py
│   │   └── ds_processer.py
│   ├── reshaper/
│   │   ├── avatar.py
│   │   ├── cp_handler.py
│   │   ├── tests_temp.py
│   │   ├── trainer.py
│   └── utils.py


### 3-Create a virtual environment:

`virtualenv venv`


### 4-Activate the virtual environment:

`source venv/bin/activate`


### 5-Install the project dependencies:

`pip install -r requirements.txt`


## The Reshaper

Adapted from [3D-Human-Body-Shape](https://github.com/zengyh1900/3D-Human-Body-Shape)

