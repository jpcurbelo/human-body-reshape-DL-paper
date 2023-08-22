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



# NEW_FILES_DIR = utils.NEW_FILES_DIR
# FILES_DIR = utils.FILES_DIR
# OBJ_DIR = os.path.join(FILES_DIR, "obj_database")


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
    '''
    Loads facet information from .txt ('template_mitacs.txt') to .npy ('facet_mitacs.npy')
    Returns a list of facets previously defined
    '''
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
    '''
    Loads data (vertices) from *.obj files in the database
    Returns a numpy array containing the vertices data
    '''
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
    
    # print(vertices)
    return vertices 


def calculate_weights(cp, vertices, facets, label):
    # load measures data
    weights = np.zeros((3, vertices.shape[0])) # store 4 values of weights to compare later
    volumes = np.zeros((1, vertices.shape[0])) 
    print(weights.shape, vertices.shape[0])
    for i in range(5):   #### 5 bodies for NOW ->  vertices.shape[0]
        sys.stdout.write('\r>> calculating weights of %s body %d'%(label, i + 1))
        sys.stdout.flush()
        
        vert = vertices[i, :, :]
        # calculate the person's weight
        vol = 0.0
        # area = 0.0
        for j in range(0, utils.F_NUM):
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
    
    np.save(open(os.path.join(NEW_FILES_DIR, f"volumes_perfit2023_{label}.npy"), "wb"), volumes)    

    obj_file_dir = os.path.join(OBJ_DIR, label)
    file_list = sorted(os.listdir(obj_file_dir))
    
    weights_scr = ['Volume', '3DHBSh', 'Aqua-Calc', 'Article 1987']
    with open(os.path.join(NEW_FILES_DIR, f"weights_perfit2023_{label}.csv"), 'w') as outfile:
        # I'm writing a header here just for the sake of readability
        outfile.write('bodyID,' + ",".join(weights_scr) + '\n')
        
        for i in range(weights.shape[1]):
            bodyID = file_list[i].split('.')[0][-4:]
            meas_list = [weights[j][i] for j in range(weights.shape[0])]
            aux_list = [str(round(num, 2)) for num in meas_list]
            outfile.write(bodyID + ',' + str(round(volumes[0, i], 8)) + ',' + ",".join(aux_list) + '\n') 

    print('\n')


def measure_bodies(cp, vertices, facets, label = "female"):
    '''
    Calculates the measurement data and convert to npy
    Returns a list of measurements
    '''
    print('[4] starting to measure bodies from .obj data for %s'%(label))
    start = time.time()
    
    # load measures data
    measurements = np.zeros((utils.M_NUM, vertices.shape[0]))
    for i in range(vertices.shape[0]):   #### 5 bodies for NOW  vertices.shape[0]
        sys.stdout.write('\r>> calculating measurements of %s body %d'%(label, i + 1))
        sys.stdout.flush()
        measurements[:, i] = calc_measurements(cp, vertices[i, :, :], facets, label, i).flat
        
    np.save(open(os.path.join(NEW_FILES_DIR, f"measurements_perfit2023_{label}.npy"), "wb"), measurements)
    
    # calculate t-value from mean and standard deviation
    mean_measurements = np.array(measurements.mean(axis=1), dtype=np.float64).reshape(utils.M_NUM, 1)
    std_measurements = np.array(measurements.std(axis=1), dtype=np.float64).reshape(utils.M_NUM, 1)
    t_measurements = (measurements - mean_measurements) / std_measurements
    
    np.save(open(os.path.join(NEW_FILES_DIR, f'mean_measurements_perfit2023_{label}.npy'), "wb"), mean_measurements)
    # save_data_csv(mean_measurements, 'mean_measurements', 'measurement', label)
    np.save(open(os.path.join(NEW_FILES_DIR, f"std_measurements_perfit2023_{label}.npy"), "wb"), std_measurements)
    np.save(open(os.path.join(NEW_FILES_DIR, f"t_measurements_perfit2023_{label}.npy"), "wb"), t_measurements)
    
    # save_data_csv(measurements, 'measurements', 'measurement', label)
    
    print('\n[4] finished measuring bodies from .obj data for %s in %fs' %(label, time.time() - start))
    
    return measurements


def calc_measurements(cp, vertices, facet, label, i):
    '''
    Calculates measurement data from given vertices by control points
    Returns an array of M_NUM measurements
    '''
    measurement_list = []
    # Read the the person's weight
    vol = np.load(open(os.path.join(NEW_FILES_DIR, f'volumes_perfit2023_{label}.npy'), "rb"), 
                        allow_pickle=True)[0, i]

    weight = KHUMANBODY_DENSITY * vol
        
    measurement_list.append(weight)   #* 10
    # calculate other measures
    # for i, measurement in enumerate(cp):
    for i, meas in enumerate(MEASUREMENTS[1:]):   # skip 0 - weight

        length = 0.0
        length = calc_length(MEAS_LABELS[meas], cp[i], vertices, meas)
        measurement_list.append(length * 100) # meters to cm   

    return np.array(measurement_list, dtype=np.float64).reshape(M_NUM, 1)


def calc_length(lab, cplist, vertices, meas):
    '''

    '''

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
        
        if meas == 'hips_buttock_girth':   #'chest_girth':   #'hips_buttock_girth':    #'waist_girth':

            # Rotate the points by -53 degrees
            theta = np.radians(-52)
            c, s = np.cos(theta), np.sin(theta)
            R = np.array(((c, -s), (s, c)))
            pts2 = np.dot(pts, R)
            # x_new2 = np.dot(x_new, R)

            xy = np.vstack((x_new, y_new)).T
            rotated_xy = np.dot(xy, R)

            plt.figure(figsize=(8, 6))

            plt.plot(pts2[:,0] * 100, pts2[:,1] * 100, 'ro')
            plt.plot(rotated_xy[:,0] * 100, rotated_xy[:,1] * 100, 'b--')
            # plt.title(meas)
            plt.xlabel('x-axis ($cm$)', fontsize=20)
            plt.ylabel('y-axis ($cm$)', fontsize=20)
            plt.axis('equal')
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            # plt.savefig(f'{meas}.png', format='png')
            plt.savefig(f'{meas}.eps', format='eps')
            plt.show()  
        
        length = 0.0
        p2 = np.array([x_new[0], y_new[0]])
        # print(range(1, len(x_new)))
        for i in range(1, len(x_new)):
            p1 = p2
            p2 = np.array([x_new[i], y_new[i]])
            # add point-to-point distance
            length += np.sqrt(np.sum((p1 - p2)**2.0)) 
        
    return length   #/ 10


def save_data_csv(meas_names, measurements, label = 'female'):
    '''
    Writes the nparray to disk
    '''
    
    obj_file_dir = os.path.join(OBJ_DIR, label)
    file_list = sorted(os.listdir(obj_file_dir))
    
    with open(os.path.join(NEW_FILES_DIR, f'measurements_perfit2023_{label}.csv'), 'w') as outfile:
        # I'm writing a header here just for the sake of readability
        outfile.write('bodyID,' + ",".join(MEASUREMENTS) + '\n')
        
        for i in range(measurements.shape[1]):
            bodyID = file_list[i].split('.')[0][:]
            meas_list = [measurements[j][i] for j in range(measurements.shape[0])]
            aux_list = [str(round(num, 2)) for num in meas_list]
            # aux_list = [str(num) for num in meas_list]
            outfile.write(bodyID + ',' + ",".join(aux_list) + '\n')


def load_def_data(vertices, facet, label = "female"):
    '''
    Loads deform-based data (See Sumner_Popovic_SIGGRAPH_2004)
    Returns a list containing [def_inv_mean, Qdeform, Qdets]
    "The goal of deformation transfer is to transfer the change in shape
        exhibited by the source deformation onto the target." Sumner_Popovic_SIGGRAPH_2004
    '''
    print('[5] starting to load deform-based data from vertices and facets for %s'%(label))
    start = time.time()
    Qdets = []

    # mean of vertices values for every .obj file (mean body)
    mean_vertices = np.array(vertices.mean(axis=0), dtype=np.float64).reshape(utils.V_NUM, 3)
    # inverse of mean vertices matrix, V^-1
    def_inv_mean = get_inv_mean(mean_vertices, facet)
    # calculate deformation matrix of each body shape
    Qdeform = np.zeros((vertices.shape[0], utils.F_NUM, 9))
    for i in range(0, utils.F_NUM):
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
    Qdets = np.array(Qdets, dtype=np.float64).reshape(utils.F_NUM, vertices.shape[0])
        
    # save data
    #f'cp_perfit2022_{label}.npy'
    np.save(open(os.path.join(NEW_FILES_DIR, f"Qdets_perfit2023_{label}.npy"), "wb"), Qdets)
    np.save(open(os.path.join(NEW_FILES_DIR, f"def_inv_mean_perfit2023_{label}.npy"), "wb"), def_inv_mean)
    np.save(open(os.path.join(NEW_FILES_DIR, f"Qdeform_perfit2023_{label}.npy"), "wb"), Qdeform)
        
    mean_deform = np.array(Qdeform.mean(axis=0), dtype=np.float64)
    mean_deform.shape = (utils.F_NUM, 9)
    std_deform = np.array(Qdeform.std(axis=0), dtype=np.float64)
    std_deform.shape = (utils.F_NUM, 9)

    np.save(open(os.path.join(NEW_FILES_DIR, f"Qdeform_mean_perfit2023_{label}.npy"), "wb"), mean_deform)
    np.save(open(os.path.join(NEW_FILES_DIR, f"Qdeform_std_perfit2023_{label}.npy"), "wb"), std_deform)

    print('\n[5] finished loading deform-based data from vertices and facets for %s in %fs' %(label, time.time() - start))

    return def_inv_mean, Qdeform, Qdets 


def get_inv_mean(mean_vertices, facet):
    '''
    Calculates (and Return) the inverse of mean vertices matrix, V^-1
    Returns the matrix of vertices (inverse of the 3x3 matrix V)
    '''    
    print("[*] starting to calculate the inverse of mean vertices matrix, V^-1")
    start = time.time()
    def_inv_mean = np.zeros((utils.F_NUM, 3, 3))
    for i in range(0, utils.F_NUM):
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
    '''
    Imports the '4th point' of the triangle (cross product), and calculate the deformation
    (fourth vertex in the direction perpendicular to the triangle - Sumner_Popovic_SIGGRAPH_2004)
    Returns an array (3x3 matrix) where the 3 columns are vectors v21, v31 and v41 
    ''' 
    # See Eq. 1 in Sumner_Popovic_SIGGRAPH_2004
    # vectors v21 and v31
    v21 = np.array((v2 - v1), dtype=np.float64)
    v31 = np.array((v3 - v1), dtype=np.float64)
    # cross product
    v41 = np.cross(list(v21.flat), list(v31.flat))
    v41 /= np.sqrt(np.linalg.norm(v41))
    
    return np.column_stack((v21, np.column_stack((v31, v41))))


def rfe_local(Qdets, Qdeform, measurements, label = "female", k_features = 9):
    '''
    Calculates relationship directly (recursive feature elimination (RFE))
    Returns a list containing the rfe matrices and masks
    '''    
    print('[7] starting recursive feature elimination (RFE) for %s'%(label))
    start = time.time()    

    body_num = Qdeform.shape[0]
    mean_measurements = np.array(measurements.mean(axis=1)).reshape(utils.M_NUM, 1)
    std_measurements = np.array(measurements.std(axis=1)).reshape(utils.M_NUM, 1)
    # calculate t-value from mean and standard deviation
    t_measurements = (measurements - mean_measurements) / std_measurements
    X = t_measurements.transpose()
        
    # (recursive feature elimination (RFE))
    # multiprocessing — Process-based parallelism
    pool = Pool(processes = 8)
    tasks = [(i, Qdets[i,:], Qdeform[:,i,:], body_num, X, measurements, 
                k_features, label) for i in range(utils.F_NUM)]
    results = pool.starmap(rfe_multiprocess, tasks)
    # results = pool.map(rfe_multiprocess, tasks)
    pool.close()
    pool.join()         

    rfe_mats = np.array([elem[0] for elem in results]).reshape(utils.F_NUM, 9, k_features)
    masks = np.array([elem[1] for elem in results]).reshape(utils.F_NUM, utils.M_NUM).transpose() 

    np.save(open(os.path.join(NEW_FILES_DIR, f"rfemats_perfit2023_{label}.npy"), "wb"), rfe_mats)
    np.save(open(os.path.join(NEW_FILES_DIR, f"rfemasks_perfit2023_{label}.npy"), "wb"), masks)

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

    flag = np.array(masks).reshape(utils.M_NUM, 1)
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
    '''
    Builds sparse matrix
    '''
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
    '''
    Constructs the related matrix A to change deformation into vertices
    '''
    print('[8] starting deformation maxtrices')
    start = time.time()    

    data = []
    rowidx = []
    colidx = []
    r = 0
    off = utils.V_NUM * 3
    shape = (utils.F_NUM * 9, (utils.V_NUM + utils.F_NUM) * 3)

    for i in range(0, utils.F_NUM):
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
    np.savez(os.path.join(NEW_FILES_DIR, f"def2vert_perfit2023_{label}"), row = def2vert.row, col = def2vert.col,
                data = def2vert.data, shape = def2vert.shape)

    print(f'[8] finished deformation maxtrices in {round(time.time() - start, 1)} s')

    return def2vert


def construct_coeff_mat(mat):
    '''
    Constructs the matrix = v_mean_inv.dot(the matrix consists of 0 -1...)

    '''
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
            facets = np.load(open(facets_file, "rb"), allow_pickle=True)
            vertices = np.load(open(vert_gender_file, "rb"), allow_pickle=True)

        print(vertices.shape)






        # 
        # weights = calculate_weights(cp, vertices, facets, label = gender)
        # measurements = measure_bodies(cp, vertices, facets, label = gender)
        # save_data_csv(MEASUREMENTS, measurements, label=gender)
        # def_inv_mean, Qdeform, Qdets = load_def_data(vertices, facets, label = gender)
        # rfe_mats, masks = rfe_local(Qdets, Qdeform, measurements, label = gender, k_features = 9)
        # def2vert = get_def2vert_matrix(def_inv_mean, facets, label = gender)

        # 
        # facets = np.load(open(os.path.join(NEW_FILES_DIR, 'facets_template_3DHBSh.npy'), "rb"), allow_pickle=True)
        # vertices = np.load(open(os.path.join(NEW_FILES_DIR, f'vertices_perfit2023_{gender}.npy'), "rb"),   \
        #                         allow_pickle=True)
        

        # measurements = measure_bodies(cp, vertices, facets, label = gender)
        
        # measurements = np.load(open(os.path.join(NEW_FILES_DIR, f'measurements_perfit2023_{gender}.npy'), "rb"),   \
        #                         allow_pickle=True)
        # def_inv_mean = np.load(open(os.path.join(NEW_FILES_DIR, f'def_inv_mean_perfit2023_{gender}.npy'), "rb"),   \
        #                   allow_pickle=True)
        # Qdeform = np.load(open(os.path.join(NEW_FILES_DIR, f'Qdeform_perfit2023_{gender}.npy'), "rb"),    \
        #                   allow_pickle=True)
        # Qdets = np.load(open(os.path.join(NEW_FILES_DIR, f'Qdets_perfit2023_{gender}.npy'), "rb"),    \
        #                   allow_pickle=True)
        # # rfe_mats = np.load(open(os.path.join(NEW_FILES_DIR, 'rfemats_perfit2023_{gender}.npy'), "rb"),    \
        # #                    allow_pickle=True)


        # rfe_mats, masks = rfe_local(Qdets, Qdeform, measurements, label = gender, k_features = 9)
        # def2vert = get_def2vert_matrix(def_inv_mean, facets, label = gender)
        

# ===========================================================================    

if __name__ == "__main__":
  
  train()