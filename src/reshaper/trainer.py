# -*- coding: utf-8 -*-
"""
Created on Thu Dec 8, 2022

@author: jpcurbelo
"""

import time
import os
import numpy as np
from scipy.interpolate import splprep, splev
from multiprocessing import Pool
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
import scipy as sp
import pickle

import sys
sys.path.append("..")
from utils import *


def convert_cp(label = "female"):
    """
    Loads control points from a text file and saves them as a nested list using pickle.

    Args:
        label (str, optional): The label for the control points. Default is "female".

    Returns:
        list: A list of control points previously defined.
    """

    print('[1] starting to load cpoints from txt for %s'%(label))
    start = time.time()
    f = open(os.path.join(os.path.join(CP_FILES_DIR, f'control_points_{label}.txt')), 'r') 
    tmplist = []
    cp_list = []  
    
    for line in f:
        if '#' in line:
            if len(tmplist) != 0:
                cp_list.append(tmplist)
                tmplist = []
        elif len(line.split()) == 1:
            continue
        else:
            tmplist.append(list(map(float, line.strip().split()))[1]) 
    
    cp_list.append(tmplist)
    f.close()

    # Save the nested list using pickle
    with open(os.path.join(RESHAPER_FILES_DIR, f'cp_{label}.pkl'), 'wb') as f:
        pickle.dump(cp_list, f)
    
    return cp_list


def convert_template(label = "female"):
    """
    Converts facet information from a .txt file ('facets_template_3DHBS.txt') to a .npy file ('facets_template_3DHBS.npy').

    Args:
        label (str, optional): The label for the template. Defaults to "female".

    Returns:
        np.ndarray: A list of facets previously defined.
    """
    print('[2] starting to load facets from txt for %s'%(label))
    start = time.time()
    facets = np.zeros((F_NUM, 3), dtype=int)
    f = open(os.path.join(RESHAPER_FILES_DIR, 'facets_template_3DHBSh.txt'), 'r')
    
    i = 0
    for line in f:
        if line[0] == 'f':
            tmp = list(map(int, line[1:].split()))
            facets[i, :] = tmp
            i += 1
    
    f.close()
    
    np.save(open(os.path.join(RESHAPER_FILES_DIR, 'facets_template_3DHBSh.npy'), 'wb'), facets)
    print('[2] finished loading facets from txt for %s in %fs' %(label, time.time() - start))
    
    return facets


def obj2npy(label = "female"): 
    """
    Loads data (vertices) from *.obj files in the database and returns a numpy array containing the vertices data.

    Args:
        label (str, optional): The label for the template. Defaults to "female".

    Returns:
        np.ndarray: A numpy array containing the vertices data.
    """


    print('[3] starting to load vertices from .obj files for %s'%(label))
    start = time.time()    
    obj_file_dir = os.path.join(OBJ_FILES_SPRING, label)
    file_list = sorted(os.listdir(obj_file_dir))
 
    # load original data
    vertices = []
    for i, obj in enumerate(file_list):
        sys.stdout.write('\r>>  converting %s body %d'%(label, i + 1))
        sys.stdout.flush()
        f = open(os.path.join(obj_file_dir, obj), 'r')
        for line in f:
            if line[0] == '#':
                continue
            elif "v " in line:
                line.replace('\n', ' ')
                tmp = list(map(float, line[1:].split()))
                # append vertices from every obj files
                vertices.append(tmp)
            else:
                break 
            
        f.close()

    # reshape vertices to an array of V_NUM rows * (x, y, z - 3 columns) for every .obj file
    vertices = np.array(vertices, dtype=np.float64).reshape(len(file_list), V_NUM, 3)
    
    # Normalize data
    for i in range(len(file_list)):
            # mean value of each of the 3 columns
            v_mean = np.mean(vertices[i,:,:], axis=0)
            vertices[i,:,:] -= v_mean   

    np.save(open(os.path.join(RESHAPER_FILES_DIR, f"vertices_{label}.npy"), "wb"), vertices)
    
    print('\n[3] finished loading vertices from .obj files for %s in %fs' %(label, time.time() - start))
    
    return vertices 


def calculate_weights(cp, vertices, facets, label):
    """
    Calculates and stores weight-related information based on given vertices and facets.

    Args:
        cp (dict): A dictionary containing control points.
        vertices (np.ndarray): Numpy array containing vertices data.
        facets (np.ndarray): Numpy array containing facet information.
        label (str): The label for the template.

    Returns:
        volumes, weights (np.ndarray): Numpy arrays containing volume and weight information.
    """

    # load measures data
    weights = np.zeros((3, vertices.shape[0])) # store 4 values of weights to compare later
    volumes = np.zeros((1, vertices.shape[0])) 
    print(weights.shape, vertices.shape[0])
    for i in range(vertices.shape[0]):   #### 5 bodies for NOW ->  vertices.shape[0]
        sys.stdout.write('\r>> calculating weights of %s body %d'%(label, i + 1))
        sys.stdout.flush()
        
        vert = vertices[i, :, :]
        # calculate the person's weight
        vol = 0.0
        # area = 0.0
        for j in range(0, F_NUM):
            f = [c - 1 for c in facets[j, :]]
            v1 = vert[f[0], :]
            v2 = vert[f[1], :]
            v3 = vert[f[2], :]
            # the scalar triple product(axb).c
            vol += np.cross(v1, v2).dot(v3)    
            
        # volume of the tetrahedron
        vol = abs(vol) / 6.0
        
        ##0: 3DHBSh --> 1026.0
        ##1: https://www.aqua-calc.com/calculate/weight-to-volume/substance/human-blank-body
        ##   1 010 kilograms [kg] of human body fit into 1 cubic meter
        ##2: https://www.jstor.org/stable/26295002
        ##   V(L) = 1.015 * W(Kg) - 4.937
        ##3: https://www.livestrong.com/article/288241-how-to-calculate-your-total-body-volume/
        ##   V(L) = S(m2) * (51.44 * W(Kg) / H(cm))
         
        weights[0, i] = 1026.0 * vol
        weights[1, i] = 1010.0 * vol
        weights[2, i] = (1000 * vol + 4.937) / 1.015
        volumes[0, i] = vol       
    
    np.save(open(os.path.join(RESHAPER_FILES_DIR, f"volumes_{label}.npy"), "wb"), volumes)    

    obj_file_dir = os.path.join(OBJ_FILES_SPRING, label)
    file_list = sorted(os.listdir(obj_file_dir))
    
    weights_src = ['Volume', '3DHBSh', 'Aqua-Calc', 'Article 1987']
    with open(os.path.join(RESHAPER_FILES_DIR, f"weights_{label}.csv"), 'w') as outfile:
        # I'm writing a header here just for the sake of readability
        outfile.write('bodyID,' + ",".join(weights_src) + '\n')
        
        for i in range(weights.shape[1]):
            bodyID = file_list[i].split('.')[0][-4:]
            meas_list = [weights[j][i] for j in range(weights.shape[0])]
            aux_list = [str(round(num, 2)) for num in meas_list]
            outfile.write(bodyID + ',' + str(round(volumes[0, i], 8)) + ',' + ",".join(aux_list) + '\n') 

    print('\n')

    return(volumes, weights)


def measure_bodies(cp, vertices, vol, label = "female"):
    """
    Calculates measurement data based on given control points, vertices, and facets.

    Args:
        cp (dict): A dictionary containing control points.
        vertices (np.ndarray): Numpy array containing vertices data.
        facets (np.ndarray): Numpy array containing facet information.
        vol (np.ndarray): Numpy array containing subjects' volume information.
        label (str, optional): The label for the template. Defaults to "female".

    Returns:
        np.ndarray: A numpy array containing measurement data.
    """

    print('[4] starting to measure bodies from .obj data for %s'%(label))
    start = time.time()
    
    # load measures data
    measurements = np.zeros((M_NUM, vertices.shape[0]))
    for i in range(vertices.shape[0]):   
        sys.stdout.write('\r>> calculating measurements of %s body %d'%(label, i + 1))
        sys.stdout.flush()
        measurements[:, i] = calc_measurements(cp, vertices[i, :, :], vol[i]).flat
        
    np.save(open(os.path.join(RESHAPER_FILES_DIR, f"measurements_{label}.npy"), "wb"), measurements)
    
    # calculate t-value from mean and standard deviation
    mean_measurements = np.array(measurements.mean(axis=1), dtype=np.float64).reshape(M_NUM, 1)
    std_measurements = np.array(measurements.std(axis=1), dtype=np.float64).reshape(M_NUM, 1)
    t_measurements = (measurements - mean_measurements) / std_measurements
    
    np.save(open(os.path.join(RESHAPER_FILES_DIR, f'mean_measurements_{label}.npy'), "wb"), mean_measurements)
    np.save(open(os.path.join(RESHAPER_FILES_DIR, f"std_measurements_{label}.npy"), "wb"), std_measurements)
    np.save(open(os.path.join(RESHAPER_FILES_DIR, f"t_measurements_{label}.npy"), "wb"), t_measurements)
    
    print('\n[4] finished measuring bodies from .obj data for %s in %fs' %(label, time.time() - start))
    
    return measurements


def calc_measurements(cp, vertices, vol):
    """
    Calculates measurement data from given control points and vertices.

    Args:
        cp: list of control points.
        vertices (np.ndarray): Numpy array containing vertices data.
        vol (float): Subject's i volume value.

    Returns:
        np.ndarray: A numpy array containing measurement data.
    """

    measurement_list = []
    # Calculate the the person's weight
    weight = KHUMANBODY_DENSITY * vol
        
    measurement_list.append(weight)   #* 10
    # calculate other measures
    for j, meas in enumerate(MEASUREMENTS[1:]):   #  skip 0 - weight
        length = 0.0
        length = calc_length(MEAS_LABELS[meas], cp[j], vertices, meas)
        measurement_list.append(length * 100) # meters to cm   

    # Convert to numpy array
    return np.array(measurement_list, dtype=np.float64).reshape(M_NUM, 1)


def calc_length(lab, cplist, vertices, meas):
    """
    Calculates the length based on given points, control points, and vertices.

    Args:
        lab (int): The label indicating the type of length calculation.
        cplist (list): A list of control points.
        vertices (np.ndarray): Numpy array containing vertices data.
        meas (int): The measurement type.

    Returns:
        float: The calculated length.
    """

    length = 0.0
    p2 = vertices[int(cplist[0]) - 1]
    if lab == 1:
        p1 = vertices[int(cplist[1]) - 1]
        length = abs(p1[2] - p2[2])  # pos 2 (z-axis)
        
    elif lab == 3 or lab == 4:
        for i in range(1, len(cplist)):
            p1 = p2
            p2 = vertices[int(cplist[i]) - 1]   
            # add point-to-point distance
            length += np.sqrt(np.sum((p1 - p2)**2.0))
            
    elif lab == 2:
        num_list = [int(x) - 1 for x in cplist]  
        # define pts from the question
        pts = np.delete(vertices[num_list], 2, axis = 1)  #delete pos 2 (z-axis)
       
        tck, u = splprep(pts.T, u=None, s=0.0, per=1) 
        u_new = np.linspace(u.min(), u.max(), 1000)
        x_new, y_new = splev(u_new, tck, der=0)
        
        length = 0.0
        p2 = np.array([x_new[0], y_new[0]])
        for i in range(1, len(x_new)):
            p1 = p2
            p2 = np.array([x_new[i], y_new[i]])
            # add point-to-point distance
            length += np.sqrt(np.sum((p1 - p2)**2.0)) 
        
    return length   #/ 10


def save_data_csv(meas_names, measurements, label = 'female'):
    """
    Writes measurement data to a CSV file.

    Args:
        meas_names (list): List of measurement names.
        measurements (np.ndarray): Numpy array containing measurement data.
        label (str, optional): The label for the template. Defaults to 'female'.

    Returns:
        None
    """
    
    obj_file_dir = os.path.join(OBJ_FILES_SPRING, label)
    file_list = sorted(os.listdir(obj_file_dir))
    
    with open(os.path.join(DS_DIR, f'measurements_SPRING_{label}.csv'), 'w') as outfile:
        # I'm writing a header here just for the sake of readability
        outfile.write('bodyID,' + ",".join(MEASUREMENTS) + '\n')
        
        for i in range(measurements.shape[1]):
            bodyID = file_list[i].split('.')[0][:]
            meas_list = [measurements[j][i] for j in range(measurements.shape[0])]
            aux_list = [str(round(num, 2)) for num in meas_list]
            outfile.write(bodyID + ',' + ",".join(aux_list) + '\n')


def load_def_data(vertices, facet, label = "female"):
    """
    Loads deform-based data using vertices and facets.

    Args:
        vertices (np.ndarray): Numpy array containing vertices data.
        facet (np.ndarray): Numpy array containing facet information.
        label (str, optional): The label for the template. Defaults to 'female'.

    Returns:
        tuple: A tuple containing def_inv_mean, Qdeform, and Qdets arrays.
    """

    print('[5] starting to load deform-based data from vertices and facets for %s'%(label))
    start = time.time()
    Qdets = []

    # mean of vertices values for every .obj file (mean body)
    mean_vertices = np.array(vertices.mean(axis=0), dtype=np.float64).reshape(V_NUM, 3)
    # inverse of mean vertices matrix, V^-1
    def_inv_mean = get_inv_mean(mean_vertices, facet)
    # calculate deformation matrix of each body shape
    Qdeform = np.zeros((vertices.shape[0], F_NUM, 9))
    for i in range(0, F_NUM):
        sys.stdout.write('\r>>  loading %s deformation of facet %d'%(label, i))
        sys.stdout.flush()
        v = [k - 1 for k in facet[i, :]]
        for j in range(0, vertices.shape[0]):
            v1 = vertices[j, v[0], :]
            v2 = vertices[j, v[1], :]
            v3 = vertices[j, v[2], :]
            # See Eq. 4 in Sumner_Popovic_SIGGRAPH_2004
            Q = assemble_face(v1, v2, v3).dot(def_inv_mean[i])
            # determinant of Q
            Qdets.append(np.linalg.det(Q))
            Q.shape = (9, 1)
            Qdeform[j, i, :] = Q.flat

    # reshape dets of Q
    Qdets = np.array(Qdets, dtype=np.float64).reshape(F_NUM, vertices.shape[0])
        
    # save data
    np.save(open(os.path.join(RESHAPER_FILES_DIR, f"Qdets_{label}.npy"), "wb"), Qdets)
    np.save(open(os.path.join(RESHAPER_FILES_DIR, f"def_inv_mean_{label}.npy"), "wb"), def_inv_mean)
    np.save(open(os.path.join(RESHAPER_FILES_DIR, f"Qdeform_{label}.npy"), "wb"), Qdeform)
        
    mean_deform = np.array(Qdeform.mean(axis=0), dtype=np.float64)
    mean_deform.shape = (F_NUM, 9)
    std_deform = np.array(Qdeform.std(axis=0), dtype=np.float64)
    std_deform.shape = (F_NUM, 9)

    np.save(open(os.path.join(RESHAPER_FILES_DIR, f"Qdeform_mean_{label}.npy"), "wb"), mean_deform)
    np.save(open(os.path.join(RESHAPER_FILES_DIR, f"Qdeform_std_{label}.npy"), "wb"), std_deform)

    print('\n[5] finished loading deform-based data from vertices and facets for %s in %fs' %(label, time.time() - start))

    return def_inv_mean, Qdeform, Qdets 


def get_inv_mean(mean_vertices, facet):
    """
    Calculates the inverse of the mean vertices matrix V^-1.

    Args:
        mean_vertices (np.ndarray): Numpy array containing mean vertices data.
        facet (np.ndarray): Numpy array containing facet information.

    Returns:
        np.ndarray: The matrix of vertices (inverse of the 3x3 matrix V).
    """

    print("[*] starting to calculate the inverse of mean vertices matrix, V^-1")
    start = time.time()
    def_inv_mean = np.zeros((F_NUM, 3, 3))
    for i in range(0, F_NUM):
        v = [j - 1 for j in facet[i, :]]
        # x, y, z vertices
        v1 = mean_vertices[v[0], :]
        v2 = mean_vertices[v[1], :]
        v3 = mean_vertices[v[2], :]
        def_inv_mean[i] = assemble_face(v1, v2, v3)
        # inverse of the 3x3 matrix
        def_inv_mean[i] = np.linalg.inv(def_inv_mean[i])

    print('[*] finished calculating the inverse of mean vertices matrix, V^-1 in %fs' % (time.time() - start))
    
    return def_inv_mean        
        
            
def assemble_face(v1, v2, v3):
    """
    Assembles a face matrix based on input vertices.

    Args:
        v1 (np.ndarray): First vertex.
        v2 (np.ndarray): Second vertex.
        v3 (np.ndarray): Third vertex.

    Returns:
        np.ndarray: A 3x3 matrix where the columns are vectors v21, v31, and v41.
    """

    # See Eq. 1 in Sumner_Popovic_SIGGRAPH_2004
    # vectors v21 and v31
    v21 = np.array((v2 - v1), dtype=np.float64)
    v31 = np.array((v3 - v1), dtype=np.float64)
    # cross product
    v41 = np.cross(list(v21.flat), list(v31.flat))
    v41 /= np.sqrt(np.linalg.norm(v41))
    
    return np.column_stack((v21, np.column_stack((v31, v41))))


def rfe_local(Qdets, Qdeform, measurements, label = "female", k_features = 9):
    """
    Performs recursive feature elimination (RFE) for relationship calculation.

    Args:
        Qdets (np.ndarray): Numpy array containing Qdets data.
        Qdeform (np.ndarray): Numpy array containing Qdeform data.
        measurements (np.ndarray): Numpy array containing measurements data.
        label (str, optional): The label for the template. Defaults to 'female'.
        k_features (int, optional): Number of selected features in RFE. Defaults to 9.

    Returns:
        tuple: A tuple containing rfe_mats and masks arrays.
    """

    print('[7] starting recursive feature elimination (RFE) for %s'%(label))
    start = time.time()    

    body_num = Qdeform.shape[0]
    mean_measurements = np.array(measurements.mean(axis=1)).reshape(M_NUM, 1)
    std_measurements = np.array(measurements.std(axis=1)).reshape(M_NUM, 1)
    # calculate t-value from mean and standard deviation
    t_measurements = (measurements - mean_measurements) / std_measurements
    X = t_measurements.transpose()
        
    # (recursive feature elimination (RFE))
    # multiprocessing — Process-based parallelism
    pool = Pool(processes = 8)
    tasks = [(i, Qdets[i,:], Qdeform[:,i,:], body_num, X, measurements, 
                k_features, label) for i in range(F_NUM)]
    results = pool.starmap(rfe_multiprocess, tasks)
    pool.close()
    pool.join()         

    rfe_mats = np.array([elem[0] for elem in results]).reshape(F_NUM, 9, k_features)
    masks = np.array([elem[1] for elem in results]).reshape(F_NUM, M_NUM).transpose() 

    np.save(open(os.path.join(RESHAPER_FILES_DIR, f"rfemats_{label}.npy"), "wb"), rfe_mats)
    np.save(open(os.path.join(RESHAPER_FILES_DIR, f"rfemasks_{label}.npy"), "wb"), masks)

    print('[7] finished recursive feature elimination (RFE) for %s in %fs' % (label, time.time() - start))

    return rfe_mats, masks


def rfe_multiprocess(i, Qdets, Qdeform, body_num, X, measurements, k_features, label = "female"):
    '''
    Returns a list containing the rfe matrices and masks for each facet
    #X = t_measurements.transpose()#
    '''  
    # The determinants of transformation matrices of a facet for n body meshes is given by:
    Y = np.array(Qdets).reshape(body_num, 1)

    # Linear regression Model
    model = LinearRegression()

    # Recursive feature elimination
    # 'learns linear regression between X and Y to select the most important parameters from 19 (21) items
    #  by recursive feature elimination algorithm'
    rfe = RFE(model, n_features_to_select = k_features)
    # 'X' is 't_meas_transp'
    rfe.fit(X, Y.ravel())
            
    # 'If the parameter is selected for this facet, then the label in the vector of this facet will be
    #  true, otherwise it will be false'
    # 'support_' attribute provides a True or False for each input variable
    masks = rfe.support_
    # print('masks\n', masks)  
    # aux = input("enter...") 

    flag = np.array(masks).reshape(M_NUM, 1)
    flag = flag.repeat(body_num, axis=1)   

    # Calculte linear mapping matrices
    # S is a matrix formed from the i row (facet) of the j body (database .obj body)
    # source transformations???
    S = np.array(Qdeform)
    S.shape = (S.size, 1)

    # Form the m matrix by only considering the "true" measurements
    m = np.array(measurements[flag])
    m.shape = (k_features, body_num)

    M = build_sparse_matrix(m, 9)   

    MtM = M.transpose().dot(M)
    MtS = M.transpose().dot(S)

    mapMat = np.array(sp.sparse.linalg.spsolve(MtM, MtS)) 
    mapMat.shape = (9, k_features)

    return [mapMat, masks]


def build_sparse_matrix(m_dataset, basis_num):
    """
    Performs recursive feature elimination (RFE) for a specific facet.

    Args:
        i (int): Index of the current facet.
        Qdets (np.ndarray): Numpy array containing Qdets data.
        Qdeform (np.ndarray): Numpy array containing Qdeform data.
        body_num (int): Number of body meshes.
        X (np.ndarray): Numpy array containing transformed measurements.
        measurements (np.ndarray): Numpy array containing measurements data.
        k_features (int): Number of selected features in RFE.
        label (str, optional): The label for the template. Defaults to 'female'.

    Returns:
        list: A list containing the mapping matrix and mask for the current facet.
    """

    shape = (m_dataset.shape[1] * basis_num, m_dataset.shape[0] * basis_num)
    data = []
    rowid = []
    colid = []
    for i in range(0, m_dataset.shape[1]):  
        for j in range(0, basis_num):  
            # increase lists
            data += [c for c in m_dataset[:, i].flat]
            rowid += [basis_num * i + j for a in range(m_dataset.shape[0])]
            colid += [a for a in range(j * m_dataset.shape[0], (j + 1) * m_dataset.shape[0])]

    return sp.sparse.coo_matrix((data, (rowid, colid)), shape)


def get_def2vert_matrix(def_inv_mean, facets, label="female"):
    """
    Constructs the related matrix A to change deformation into vertices.

    Args:
        def_inv_mean (np.ndarray): Numpy array containing the inverse mean vertices matrix.
        facets (np.ndarray): Numpy array containing facets data.
        label (str, optional): The label for the template. Defaults to 'female'.

    Returns:
        sp.sparse.coo_matrix: A sparse COO matrix representing the transformation from deformation to vertices.
    """

    print('[8] starting deformation matrices')
    start = time.time()    

    data = []
    rowidx = []
    colidx = []
    r = 0
    off = V_NUM * 3
    shape = (F_NUM * 9, (V_NUM + F_NUM) * 3)

    for i in range(0, F_NUM):
        coeff = construct_coeff_mat(def_inv_mean[i])
        v = [c - 1 for c in facets[i, :]]
        v1 = range(v[0] * 3, v[0] * 3 + 3)
        v2 = range(v[1] * 3, v[1] * 3 + 3)
        v3 = range(v[2] * 3, v[2] * 3 + 3)
        v4 = range(off + i * 3, off + i * 3 + 3)  
        
        for j in range(0, 3):
            data += [c for c in coeff.flat]
            rowidx += [r, r, r, r, r + 1, r + 1, r + 1, r + 1, r + 2, r + 2, r + 2, r + 2]
            colidx += [v1[j], v2[j], v3[j], v4[j], v1[j], v2[j], v3[j], v4[j], v1[j], v2[j], v3[j], v4[j]]
            r += 3

    def2vert = sp.sparse.coo_matrix((data, (rowidx, colidx)), shape=shape)
    np.savez(os.path.join(RESHAPER_FILES_DIR, f"def2vert_{label}"), row = def2vert.row, col = def2vert.col,
                data = def2vert.data, shape = def2vert.shape)

    print(f'[8] finished deformation maxtrices in {round(time.time() - start, 1)} s')

    return def2vert


def construct_coeff_mat(mat):
    """
    Constructs the coefficient matrix for deformation calculations.

    Args:
        mat (np.ndarray): The input matrix.

    Returns:
        np.ndarray: The constructed coefficient matrix.
    """

    tmp = -mat.sum(0)
    
    return np.row_stack((tmp, mat)).transpose()


# ===========================================================================
def train():
    '''
    Trains all data
    Extracts information from bodies database
    '''
    
    facets_file = os.path.join(RESHAPER_FILES_DIR, f'facets_template_3DHBSh.npy')
    
    for gender in GENDERS:
        print('gender = ', gender)
   
        ## generate and load control point from txt to pkl file
        cp_gender_file = os.path.join(RESHAPER_FILES_DIR, f'cp_{gender}.pkl')
        if not os.path.exists(cp_gender_file):
            cp = convert_cp(label = gender)
        else:
            cp = np.load(open(cp_gender_file, "rb"), allow_pickle=True)

        ## generate and load facets from txt to npy file
        if not os.path.exists(facets_file):
            facets = convert_template(label = gender)
        else:
            facets = np.load(open(facets_file, "rb"), allow_pickle=True)

        ## generate and load vertices from obj to npy file
        vert_gender_file = os.path.join(RESHAPER_FILES_DIR, f'vertices_{gender}.npy')
        if not os.path.exists(vert_gender_file):
            vertices = obj2npy(label = gender)
        else:
            vertices = np.load(open(vert_gender_file, "rb"), allow_pickle=True)

        ## generate and load volumes from obj to npy file
        vol_gender_file = os.path.join(RESHAPER_FILES_DIR, f'volumes_{gender}.npy')
        if not os.path.exists(vol_gender_file):
            vol, _ = calculate_weights(cp, vertices, facets, label = gender)
        else:
            vol = np.load(open(vol_gender_file, "rb"), allow_pickle=True)[0, :]

        ## generate and load measurements from obj to npy file
        meas_gender_file = os.path.join(RESHAPER_FILES_DIR, f'measurements_{gender}.npy')
        if not os.path.exists(meas_gender_file):
            measurements = measure_bodies(cp, vertices, vol, label = gender)
        else:
            measurements = np.load(open(meas_gender_file, "rb"), allow_pickle=True)

        ## save measurements to csv file
        save_data_csv(MEASUREMENTS, measurements, label=gender)

        ## generate and deformation matrices to npy file
        def2vert_gender_file = os.path.join(RESHAPER_FILES_DIR, f'def2vert_{gender}.npz')
        if not os.path.exists(def2vert_gender_file):

            ## generate and load def_inv_mean, Qdeform, Qdets from obj to npy file
            def_inv_mean_gender_file = os.path.join(RESHAPER_FILES_DIR, f'def_inv_mean_{gender}.npy')
            Qdeform_gender_file = os.path.join(RESHAPER_FILES_DIR, f'Qdeform_{gender}.npy')
            Qdets_gender_file = os.path.join(RESHAPER_FILES_DIR, f'Qdets_{gender}.npy')
            if not os.path.exists(def_inv_mean_gender_file) or   \
                not os.path.exists(Qdeform_gender_file) or  \
                not os.path.exists(Qdets_gender_file):

                def_inv_mean, Qdeform, Qdets = load_def_data(vertices, facets, label = gender)

            else:
                def_inv_mean = np.load(open(def_inv_mean_gender_file, "rb"), allow_pickle=True)
                Qdeform = np.load(open(Qdeform_gender_file, "rb"), allow_pickle=True)
                Qdets = np.load(open(Qdets_gender_file, "rb"), allow_pickle=True)
                    
            ## generate and load rfe_mats from obj to npy file
            rfe_mats_gender_file = os.path.join(RESHAPER_FILES_DIR, f'rfemats_{gender}.npy')
            if not os.path.exists(rfe_mats_gender_file):
                rfe_mats, _ = rfe_local(Qdets, Qdeform, measurements, label = gender, k_features = 9)
            else:
                rfe_mats = np.load(open(rfe_mats_gender_file, "rb"), allow_pickle=True)

            def2vert = get_def2vert_matrix(def_inv_mean, facets, label = gender)

        else:
            def2vert = np.load(open(def2vert_gender_file, "rb"), allow_pickle=True)        

        
# ===========================================================================    

if __name__ == "__main__":
  
  train()