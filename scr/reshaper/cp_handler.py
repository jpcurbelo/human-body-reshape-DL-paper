# -*- coding: utf-8 -*-
"""
Created on Thu Dec 8, 2022

@author: jpcurbelo
"""

import os
from natsort import natsorted 
from pathlib import Path

import copy
import math

import sys
sys.path.append("..")
from utils import *

TOL = 1e-5


def extract_cp():  
    """
    Read the *.obj files containing the CP points and generate a file with the number of rows
    in the mean_avatar_{gender}.obj file corresponding to each measurement.
    """  
    for gender in GENDERS:   
        modelname = f'mean_avatar_{gender}.obj'

        with open(os.path.join(CP_FILES_DIR, modelname), 'r')  as modelf:

            # List of *.obj files - only cp files (not mean_avatar file) and specific gender 
            obj_files = natsorted([str(x).split(os.sep)[1] for x in list(Path(CP_FILES_DIR).glob(r'**/*.obj')) \
                                    if 'mean_avatar' not in str(x).split(os.sep)[1] and \
                                        f'_{gender}' in str(x).split(os.sep)[1]
                                ])

            # read vertices from model file
            vertices = []
            for line in modelf:
                if "v " in line:
                    line.replace('\n', ' ')
                    tmp = list(map(float, line[1:].split()))
                    # append vertices from every obj files
                    vertices.append(tmp)

            # Create and prepare cp file 
            with open(os.path.join(CP_FILES_DIR, f'control_points_{gender}.txt'),mode="w",encoding="utf-8") as cpf:
                
                cpf.write(f'# Control Points (20 measurements)\n')

                # read info from measurement files
                for i, m in enumerate(MEASUREMENTS[1:]):   #  skip 0 - weight
                    cpf.write(f'# {i + 2}-{m}\n')
                    measname = f"{i + 2}_{m}_{gender}.obj"

                    with open(os.path.join(CP_FILES_DIR, measname), 'r') as measf:

                        # read measurement vertices
                        meas_vertices = {}
                        x = []
                        y = []
                        z = []

                        for line in measf:
                            if "v " in line:
                                line.replace('\n', ' ')
                                tmp = list(map(float, line[1:].split()))
                                # append vertices from every obj files
                                vertex_number = 1
                                for vertex in vertices:
                                    # check tolerance for coincidence
                                    if abs(vertex[0] - tmp[0]) < TOL and abs(vertex[1] - tmp[1]) < TOL \
                                        and abs(vertex[2] - tmp[2]) < TOL:
                                            
                                        reapeated = False
                                        for key in meas_vertices.keys():
                                            if tmp == meas_vertices[key]:
                                                print(tmp, m, "repeated!!!")
                                                reapeated = True
                                                break
                                        
                                        if reapeated == False:
                                            meas_vertices[vertex_number] = tmp
                                            x.append(tmp[0])
                                            y.append(tmp[1])
                                            z.append(tmp[2])
                                            # print(x, y, z)
                                    vertex_number += 1   

                    # Sort control points (from the first one, find the nearest one)
                    cp = dict()
                    aux_meas_vertices = copy.deepcopy(meas_vertices)   

                    if i == 13 or i == 16:
                        # Minimum z-value for sleeveoutseam and waistbacklength
                        key_list = list(meas_vertices.keys())
                        val_list = list(meas_vertices.values())
                        z_list = [z[1] for z in val_list]
                        min_z_pos = z_list.index(min(z_list))
                        current_key = key_list[min_z_pos]
                    else:
                        # Whichever point can be the first for girth measurements and vertical distances
                        current_key = min(meas_vertices.keys())

                    cp[current_key] = meas_vertices[current_key]
                    aux_meas_vertices.pop(current_key, None)
                    for num_p in range(1, len(x)):
                        next_key = nearest_point(cp[current_key], aux_meas_vertices, i)
                        cp[next_key] = meas_vertices[next_key]
                        aux_meas_vertices.pop(next_key, None)
                        current_key = next_key

                    # Write control points info for current measurement 
                    cpf.write(f'{len(cp.keys())}' + '\n')
                    p = 10  #forearm_girth
                    for key in cp.keys():   
                        x = cp[key][0]
                        y = cp[key][1]
                        z = cp[key][2]
                        
                        cpf.write(str(p)+ '\t' + str(key) + '\n')
                        p += 1 


def nearest_point(point, dic_of_points, meas_id):
    """
    Find the key containing the nearest point to the given 'point' from the dictionary of points.

    Args:
        point (list or tuple): The target point [x, y, z] or (x, y, z) to find the nearest point for.
        dic_of_points (dict): A dictionary containing keys as identifiers and values as [x, y, z] coordinate lists.
        meas_id: Identifier for the measurement, not used in this function.
        
    Returns:
        int: The key corresponding to the nearest point in the dictionary.
    """
    x0, y0, z0 = point
    min_dist = 1e5
    next_key = 0
    
    for key, coordinates in dic_of_points.items():
        x, y, z = coordinates
        dist = math.sqrt((x - x0)**2 + (y - y0)**2 + (z - z0)**2)
        if dist < min_dist:
            min_dist = dist
            next_key = key
    
    return next_key  


####################################################
if __name__ == "__main__":
    extract_cp()