# https://gist.github.com/eug/a3dce98d4c88b0934970ee7de8bb1ec5

from scipy.io import loadmat
import pandas as pd

def mat2csv(file_mat, file_csv, index=False):
    mat = loadmat(file_mat)
    
    data = {}
    for col_id in range(len(mat['X'][0])):
        data[col_id] = []

    for row in mat['X']:
        for col_id, value in enumerate(row):
            data[col_id].append(value)

    data['class'] = []
    for row in mat['y']:
        for value in row:
            data['class'].append(value)

    df = pd.DataFrame(data)
    df.to_csv(file_csv, index=index)
    
    
if __name__ == "__main__":
    
    # matfile = loadmat('mpii_human_pose_v1_u12_2/mpii_human_pose_v1_u12_1.mat')
    # # open the file in the write mode
    # csvfile = open('mpii_human_pose_v1_u12_2/mpii_human_pose_v1_u12_1', 'w')    
    
    mat2csv('mpii_human_pose_v1_u12_2/mpii_human_pose_v1_u12_1.mat', 
            'mpii_human_pose_v1_u12_2/mpii_human_pose_v1_u12_1', index=False)