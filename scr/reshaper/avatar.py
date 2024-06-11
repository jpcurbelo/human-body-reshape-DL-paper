import numpy as np
import scipy as sp
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import splprep, splev

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

import sys
sys.path.append("..")
from utils import *


class Avatar:
    """
    Class for creating 3D avatars from input measurements.
    Imputation for missing data and creates wavefront.obj file from the measurements.
    """

    def __init__(self, input_data, gender="female"):
        """
        Initialize the class with input measurements.

        Args:
            input_data (numpy ndarray): input measurements of size 21.
            gender (str): "female" or "male".
        """

        self.gender = gender
        self.input_data = np.zeros(M_NUM).transpose()
        self.output_data = np.zeros(M_NUM).transpose()
        self.vertices = np.zeros((V_NUM, 3))

        ## Assign measurements to 20-sized array
        self.input_data = input_data
        self.imputed_data = self.input_data.copy()

    # Imputation for missing data
    def predict(self, db_name="TOTAL"):
        """Impute the missing values to complete the 21 input measurements

        Args:
            self
            db_name: string to determine which dataset to be used to impute the missing values
                     ("SPRING2023", "ANSURI2023", "ANSURII2023", or "TOTAL")

        Returns:
            self.imputed_data: 21 measurements without missing values
        """

        scaler = StandardScaler()

        meas_dir = os.path.join(DS_DIR, f"measurements_{db_name}_{self.gender}.npy")

        try:
            measurements = np.load(open(meas_dir, "rb"), allow_pickle=True)

        except FileNotFoundError:

            # The file does not exist, so we need to handle the exception
            print(f"Wrong file or file path --> '{meas_dir}' file not found.")

        else:
            
            input_data_aux = self.input_data.copy()
            input_data_aux[input_data_aux == 0.0] = np.nan

            measurements = scaler.fit_transform(
                np.vstack([measurements, input_data_aux])
            )

            solver = IterativeImputer(
                random_state=0,
                estimator=LinearRegression(),
                initial_strategy="mean",
                max_iter=100,
                tol=1e-4,
                n_nearest_features=20,
                verbose=0,
            )

            output_data = solver.fit_transform(measurements)
            self.imputed_data = scaler.inverse_transform(
                np.array(output_data[-1, :]).reshape(1, -1)
            ).transpose()

            return self.imputed_data
        
    def create_obj_file(self, ava_dir=OUTPUT_FILES_DIR, ava_name=None):
        """Creates a wavefront.obj file from the measurements (input) and data from training.
        First, updates [vertices, normals, facets] by mapping the input data with the rfe_matrices

        Args:
            ava_dir: string to determine the directory where the .obj file will be saved
            ava_name: string to determine the name of the .obj file

        Returns:
            No arguments
        """

        ## Load vertices
        vertices = self.mapping_rfemat()

        # ## Rotate vertices
        # vertices = self.rotate_vertices(vertices)

        self.vertices = vertices

        if ava_name == None:
            out_dir = os.path.join(ava_dir, f"avatar_{self.gender}.obj")
        else:
            out_dir = os.path.join(ava_dir, f"{ava_name}.obj")

        with open(out_dir, "w") as file:
            for i in range(0, vertices.shape[0]):
                file.write(
                    "v %.12f %.12f %.12f\n"
                    % (self.vertices[i][0], self.vertices[i][1], self.vertices[i][2])
                )

            facets_dir = os.path.join(RESHAPER_FILES_DIR, f"facets_template_3DHBSh.npy")
            try:
                facets = np.load(open(facets_dir, "rb"), allow_pickle=True)

            except FileNotFoundError:

                # The file does not exist, so we need to handle the exception
                print(f"Wrong file or file path --> '{facets_dir}' file not found.")

            else:

                normals = self.compute_normals(vertices, facets)

                for i in range(0, len(normals)):
                    file.write(
                        "vn %.12f %.12f %.12f\n"
                        % (normals[i][0], normals[i][1], normals[i][2])
                    )

                for i in range(0, facets.shape[0]):
                    file.write(
                        "f %d//%d %d//%d %d//%d\n"
                        % (
                            facets[i][0],
                            facets[i][0],
                            facets[i][1],
                            facets[i][1],
                            facets[i][2],
                            facets[i][2],
                        )
                    )

    def mapping_rfemat(self):
        """Local mapping using measurements + rfe_mat

        Args:
            self

        Returns:
            vertices (np array): all the vertices in the SPRING database - for the specific gender
        """

        # print("** Loading the vertices")

        data = np.array(self.imputed_data).reshape(M_NUM, 1)
        d = []

        try:
            maksks_dir = os.path.join(
                RESHAPER_FILES_DIR, f"rfemasks_{self.gender}.npy"
            )
            rfemasks = np.load(open(maksks_dir, "rb"), allow_pickle=True)

            mats_dir = os.path.join(RESHAPER_FILES_DIR, f"rfemats_{self.gender}.npy")
            rfemats = np.load(open(mats_dir, "rb"), allow_pickle=True)

        except FileNotFoundError:

            # The file does not exist, so we need to handle the exception
            print(
                f"Wrong file or file path --> '{maksks_dir}' and/or '{mats_dir}' file not found."
            )

        for i in range(0, F_NUM):
            mask = np.array(rfemasks[:, i]).reshape(M_NUM, 1)
            alpha = np.array(data[mask])
            alpha.shape = (alpha.size, 1)
            s = rfemats[i].dot(alpha)
            d += [a for a in s.flat]

        d = np.array(d).reshape(F_NUM * 9, 1)

        vertices = self.d_synthesize(d)

        return vertices
    
    def d_synthesize(self, deform):
        """Synthesizes a body by deform-based, given deform, output vertices

        Args:
            self
            deform: a np array with the deformations

        Returns:
            x (np array): all the 'synthesized' vertices in the SPRING database - for the specific gender
        """

        d = np.array(deform.flat).reshape(deform.size, 1)

        def2vert_dir = os.path.join(RESHAPER_FILES_DIR, f"def2vert_{self.gender}.npz")
        try:
            loader = np.load(def2vert_dir)
        except FileNotFoundError:

            # The file does not exist, so we need to handle the exception
            print(f"Wrong file or file path --> '{def2vert_dir}' file not found.")

        def2vert = sp.sparse.coo_matrix(
            (loader["data"], (loader["row"], loader["col"])), shape=loader["shape"]
        )
        LU = sp.sparse.linalg.splu(def2vert.transpose().dot(def2vert).tocsc())

        Atd = def2vert.transpose().dot(d)
        x = LU.solve(Atd)
        x = x[: V_NUM * 3]

        # move to center
        x.shape = (V_NUM, 3)
        x_mean = np.mean(x, axis=0)
        x -= x_mean

        return x  # 10 * x

    def rotate_vertices(self, vertices):
        """Rotates all the 'synthesized' vertices in the SPRING database - for the specific gender
           (to be loaded better on Blender)

        Args:
            self
            vertices: a np array

        Returns:
            rotated_vec (np array): all the rotated vertices in the SPRING database - for the specific gender
        """

        # print("** Rotating the vertices")

        rotated_vec = vertices

        if ROTATION_DICT["x"] != 0:
            rotation_degrees = ROTATION_DICT["x"]
            rotation_radians = np.radians(rotation_degrees)
            rotation_axis = np.array([1, 0, 0])
            rotation_vector = rotation_radians * rotation_axis
            rotation = R.from_rotvec(rotation_vector)
            for i in range(0, rotated_vec.shape[0]):
                vec = [rotated_vec[i][0], rotated_vec[i][1], rotated_vec[i][2]]
                rotated_vec[i] = rotation.apply(vec)

        if ROTATION_DICT["y"] != 0:
            rotation_degrees = ROTATION_DICT["y"]
            rotation_radians = np.radians(rotation_degrees)
            rotation_axis = np.array([0, 1, 0])
            rotation_vector = rotation_radians * rotation_axis
            rotation = R.from_rotvec(rotation_vector)
            for i in range(0, rotated_vec.shape[0]):
                vec = [rotated_vec[i][0], rotated_vec[i][1], rotated_vec[i][2]]
                rotated_vec[i] = rotation.apply(vec)

        if ROTATION_DICT["z"] != 0:
            rotation_degrees = ROTATION_DICT["z"]
            rotation_radians = np.radians(rotation_degrees)
            rotation_axis = np.array([0, 0, 1])
            rotation_vector = rotation_radians * rotation_axis
            rotation = R.from_rotvec(rotation_vector)
            for i in range(0, rotated_vec.shape[0]):
                vec = [rotated_vec[i][0], rotated_vec[i][1], rotated_vec[i][2]]
                rotated_vec[i] = rotation.apply(vec)

        return rotated_vec

    def compute_normals(self, vertices, facets):
        """Computes the normals on the 3D body surface

        Args:
            self
            vertices: a np array
            facets: a np array

        Returns:
            normals (np array): the normals to every facet
        """

        # print("** Calculating the normals")

        normals = []
        vertNormalLists = [[] for i in range(0, len(vertices))]

        for idf, facet in enumerate(facets):
            AB = np.array(vertices[facet[0] - 1]) - np.array(vertices[facet[1] - 1])
            AC = np.array(vertices[facet[0] - 1]) - np.array(vertices[facet[2] - 1])

            n = np.cross(AB, -AC)
            n /= np.linalg.norm(n)

            for i in range(0, 3):
                vertNormalLists[facet[i] - 1].append(n)

        for idx, normalList in enumerate(vertNormalLists):
            normalSum = np.zeros(3)
            for normal in normalList:
                normalSum += normal

            normal = normalSum / float(len(normalList))
            normal /= np.linalg.norm(normal) * (
                -1
            )  # (-1) to compute "exiting normals" -> light avatar load in blender
            normals.append(normal.tolist())

        return normals

    def measure(self, save_file=True, out_meas_name=None):
        """Extract measurements from the 3D avatar
        Args:
            self

        Returns:
            self.output_data (np array): 21 measurements extracted from the 3D avatar
        """

        facets_dir = os.path.join(RESHAPER_FILES_DIR, f"facets_template_3DHBSh.npy")
        try:
            facets = np.load(open(facets_dir, "rb"), allow_pickle=True)
        except FileNotFoundError:
            # The file does not exist, so we need to handle the exception
            print(f"Wrong file or file path --> '{facets_dir}' file not found.")

        cp_dir = os.path.join(RESHAPER_FILES_DIR, f"cp_{self.gender}.pkl")
        try:
            cp = np.load(open(cp_dir, "rb"), allow_pickle=True)
        except FileNotFoundError:
            # The file does not exist, so we need to handle the exception
            print(f"Wrong file or file path --> '{cp_dir}' file not found.")

        self.output_data = self.calc_measurements(cp, facets)


        if save_file == True:
            
            if out_meas_name == None:
                meas_dir = os.path.join(OUTPUT_FILES_DIR, f"output_data_avatar_{self.gender}.csv")
            else:
                meas_dir = os.path.join(OUTPUT_FILES_DIR, f"{out_meas_name}.csv")
            
            ## Create output file
            with open(meas_dir, "w") as file:

                file.write(
                    "_______________________________________________________________________________\n"
                )
                file.write(
                    f"%-21s|%-16s|%-16s|%-16s\n"
                    % (
                        "Measurement",
                        "Basic-Input",
                        "Predicted-Input",
                        f"3D Avatar-Output ({chr(0x03B4)}%)",
                    )  # \u03B4
                )
                file.write(
                    "_______________________________________________________________________________\n"
                )

                for i, meas in enumerate(MEASUREMENTS):

                    if abs(self.output_data[i]) < 1e-10:
                        rel_err = None  # Or any other value you prefer
                    else:
                        rel_err = (
                            abs(self.imputed_data[i] - self.output_data[i])
                            / abs(self.output_data[i])
                            * 100
                        )

                    file.write(
                        "%-21s|%-16.1f|%-16.1f|%-7.1f (%-5.1f%%) \n"
                        % (
                            meas,
                            self.input_data[i],
                            self.imputed_data[i],
                            self.output_data[i],
                            rel_err,
                        )
                    )

                file.write(
                    "_______________________________________________________________________________"
                )

        return self.output_data

    def calc_measurements(self, cp, facet):
        """Calculates measurement data from given vertices by control points
        Returns an array of M_NUM measurements
        """

        measurement_list = []

        # calculate the person's weight
        vol = 0.0
        for i in range(0, F_NUM):
            f = [c - 1 for c in facet[i, :]]
            v1 = self.vertices[f[0], :]
            v2 = self.vertices[f[1], :]
            v3 = self.vertices[f[2], :]
            # the scalar triple product(axb).c
            vol += np.cross(v1, v2).dot(v3)
        # volume of the tetrahedron
        vol = abs(vol) / 6.0

        weight = KHUMANBODY_DENSITY * vol
        measurement_list.append(weight)  

        # calculate other measures
        for i, meas in enumerate(MEASUREMENTS[1:]):  # skip 0 - weight

            length = 0.0
            length = self.calc_length(MEAS_LABELS[meas], cp[i], meas)
            measurement_list.append(length * 100)  # meters to cm

        return np.array(measurement_list, dtype=np.float64)

    def calc_length(self, lab, cplist, meas):
        """ """

        length = 0.0
        p2 = self.vertices[int(cplist[0]) - 1]
        if lab == 1:
            p1 = self.vertices[int(cplist[1]) - 1]
            length = abs(p1[2] - p2[2])  # pos 2 (z-axis)

        elif lab == 3 or lab == 4:
            for i in range(1, len(cplist)):
                p1 = p2
                p2 = self.vertices[int(cplist[i]) - 1]
                # add point-to-point distance
                length += np.sqrt(np.sum((p1 - p2) ** 2.0))

        elif lab == 2:
            num_list = [int(x) - 1 for x in cplist]
            # define pts from the question
            pts = np.delete(self.vertices[num_list], 2, axis=1)  # delete pos 2 (z-axis)

            tck, u = splprep(pts.T, u=None, s=0.0, per=1)
            u_new = np.linspace(u.min(), u.max(), 1000)
            x_new, y_new = splev(u_new, tck, der=0)

            length = 0.0
            p2 = np.array([x_new[0], y_new[0]])
            for i in range(1, len(x_new)):
                p1 = p2
                p2 = np.array([x_new[i], y_new[i]])
                # add point-to-point distance
                length += np.sqrt(np.sum((p1 - p2) ** 2.0))

        return length 

###########################