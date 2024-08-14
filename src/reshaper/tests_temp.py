import numpy as np


import sys
sys.path.append("..")
from utils import *

from avatar import Avatar

def test_avatar():
   
     ## Import info: measurements data
    gender, measurements = load_input_data()
    measurements = np.array(measurements).transpose()

    body = Avatar(measurements, gender)

    input_meas21 = body.predict()

    ## Create 3D avatar
    body.create_obj_file()

    ## Extract measurements from the 3D avatar
    output_meas21 = body.measure()


if __name__ == "__main__":
    
    test_avatar()