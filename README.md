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
...
human-body-reshape-DL-paper/
├── data/
│   └── body_reshaper_files/
│   └── cp_blender_files/
│   └── obj_files/
├── figures/
├── src/
│   ├── reshaper/
│   │   └── reshaper.py
│   └── utils.py
...


### 3-Create a virtual environment:

`virtualenv venv`


### 4-Activate the virtual environment:

`source venv/bin/activate`


### 5-Install the project dependencies:

`pip install -r requirements.txt`


## The Reshaper

Adapted from [3D-Human-Body-Shape](https://github.com/zengyh1900/3D-Human-Body-Shape)

