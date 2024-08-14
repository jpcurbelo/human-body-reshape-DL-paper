
import os
import scipy.io
import numpy as np


# FILES_DIR = "caesar-fitted-meshes"
# FILES_DIR = "caesar-norm-nh-fitted-meshes"
# FILES_DIR = "caesar-norm-wsx-fitted-meshes"

FILES_DIR = ["caesar-fitted-meshes", "caesar-norm-nh-fitted-meshes", "caesar-norm-wsx-fitted-meshes"]


def nparray2obj_save(file_path: str, vertices: np.ndarray, indices: np.ndarray) -> None:
    '''
    https://gist.github.com/ando-takahiro/426ec5399cdd6bb43c9a06d610b0602e
    '''

    with open(file_path, 'w') as out:
        for v in vertices:
            out.write('v %f %f %f\n' % tuple(v))
        # for f in indices:
        #     out.write('f %d %d %d\n' % tuple(f + 1))



def main():
    # pass

    for mat_files_dir in FILES_DIR:
        obj_files_dir = f'{mat_files_dir}_OBJ'
        try: 
            os.mkdir(obj_files_dir) 
        except OSError as error: 
            print(error)  

        file_list = [file for file in os.listdir(mat_files_dir) if '.mat' in file and 'missing' not in file]
        # print(file_list)

        for file in file_list:

            # print(file)

            mat = scipy.io.loadmat(os.path.join(mat_files_dir, file))
            
            # print(mat)
            # print(mat['points'])

            nparray2obj_save(os.path.join(obj_files_dir, f'{file[:-4]}.obj'), mat['points'], mat['points']) 

            # break


#########################################################################################
if __name__ == '__main__':
    
    main()