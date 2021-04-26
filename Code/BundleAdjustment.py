import numpy as np
import cv2
from scipy.sparse import lil_matrix
import time
from scipy.optimize import least_squares
from Utils.MiscUtils import *

##########################################################################################
########################### Helper Functions for BA ######################################
##########################################################################################
def getObservationsIndexAndVizMat(X_found, filtered_feature_flag, nCam):
    # find the 3d points such that they are visible in either of the cameras < nCam
    bin_temp = np.zeros((filtered_feature_flag.shape[0]), dtype = int)
    for n in range(nCam + 1):
        bin_temp = bin_temp | filtered_feature_flag[:,n]

    X_index = np.where((X_found.reshape(-1)) & (bin_temp))
    
    visiblity_matrix = X_found[X_index].reshape(-1,1)
    for n in range(nCam + 1):
        visiblity_matrix = np.hstack((visiblity_matrix, filtered_feature_flag[X_index, n].reshape(-1,1)))

    o, c = visiblity_matrix.shape
    return X_index, visiblity_matrix[:, 1:c]

def get2DPoints(X_index, visiblity_matrix, feature_x, feature_y):
    pts2D = []
    visible_feature_x = feature_x[X_index]
    visible_feature_y = feature_y[X_index]
    h, w = visiblity_matrix.shape
    for i in range(h):
        for j in range(w):
            if visiblity_matrix[i,j] == 1:
                pt = np.hstack((visible_feature_x[i,j], visible_feature_y[i,j]))
                pts2D.append(pt)
    return np.array(pts2D).reshape(-1, 2)             

def getCameraPointIndices(visiblity_matrix):
    camera_indices = []
    point_indices = []
    h, w = visiblity_matrix.shape
    for i in range(h):
        for j in range(w):
            if visiblity_matrix[i,j] == 1:
                camera_indices.append(j)
                point_indices.append(i)

    return np.array(camera_indices).reshape(-1), np.array(point_indices).reshape(-1)


##########################################################################################
##########################################################################################
##########################################################################################

def bundle_adjustment_sparsity(X_found, filtered_feature_flag, nCam):
    
    """
    To create the Sparsity matrix
    """
    number_of_cam = nCam + 1
    X_index, visiblity_matrix = getObservationsIndexAndVizMat(X_found.reshape(-1), filtered_feature_flag, nCam)
    n_observations = np.sum(visiblity_matrix)
    n_points = len(X_index[0])

    m = n_observations * 2
    n = number_of_cam * 6 + n_points * 3
    A = lil_matrix((m, n), dtype=int)
    print(m, n)


    i = np.arange(n_observations)
    camera_indices, point_indices = getCameraPointIndices(visiblity_matrix)

    for s in range(6):
        A[2 * i, camera_indices * 6 + s] = 1
        A[2 * i + 1, camera_indices * 6 + s] = 1

    for s in range(3):
        A[2 * i, (nCam)* 6 + point_indices * 3 + s] = 1
        A[2 * i + 1, (nCam) * 6 + point_indices * 3 + s] = 1

    return A

# def project(points, camera_params, K):
#     """Convert 3-D points to 2-D by projecting onto images."""
#     points_proj = rotate(points, camera_params[:, :3])
#     points_proj += camera_params[:, 3:]
#     points_proj = -points_proj[:, :2] / points_proj[:, 2, np.newaxis]
#     return points_proj

def project(points_3d, camera_params, K):
    def projectPoint_(R, C, pt3D, K):
        P2 = np.dot(K, np.dot(R, np.hstack((np.identity(3), -C.reshape(3,1)))))
        x3D_4 = np.hstack((pt3D, 1))
        x_proj = np.dot(P2, x3D_4.T)
        x_proj /= x_proj[-1]
        return x_proj

    x_proj = []
    for i in range(len(camera_params)):
        R = getRotation(camera_params[i, :3], 'e')
        C = camera_params[i, 3:].reshape(3,1)
        pt3D = points_3d[i]
        pt_proj = projectPoint_(R, C, pt3D, K)[:2]
        x_proj.append(pt_proj)    
    return np.array(x_proj)

def rotate(points, rot_vecs):
    """Rotate points by given rotation vectors.
    
    Rodrigues' rotation formula is used.
    """
    theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
    with np.errstate(invalid='ignore'):
        v = rot_vecs / theta
        v = np.nan_to_num(v)
    dot = np.sum(points * v, axis=1)[:, np.newaxis]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    return cos_theta * points + sin_theta * np.cross(v, points) + dot * (1 - cos_theta) * v

def fun(x0, nCam, n_points, camera_indices, point_indices, points_2d, K):
    """Compute residuals.
    
    `params` contains camera parameters and 3-D coordinates.
    """
    number_of_cam = nCam + 1
    camera_params = x0[:number_of_cam * 6].reshape((number_of_cam, 6))
    points_3d = x0[number_of_cam * 6:].reshape((n_points, 3))
    points_proj = project(points_3d[point_indices], camera_params[camera_indices], K)
    error_vec = (points_proj - points_2d).ravel()
    
    return error_vec

def BundleAdjustment(X_all,X_found, feature_x, feature_y, filtered_feature_flag, R_set_, C_set_, K, nCam):
    
    X_index, visiblity_matrix = getObservationsIndexAndVizMat(X_found, filtered_feature_flag, nCam)
    points_3d = X_all[X_index]
    points_2d = get2DPoints(X_index, visiblity_matrix, feature_x, feature_y)

    RC_list = []
    for i in range(nCam+1):
        C, R = C_set_[i], R_set_[i]
        Q = getEuler(R)
        RC = [Q[0], Q[1], Q[2], C[0], C[1], C[2]]
        RC_list.append(RC)
    RC_list = np.array(RC_list).reshape(-1, 6)

    x0 = np.hstack((RC_list.ravel(), points_3d.ravel()))
    n_points = points_3d.shape[0]

    camera_indices, point_indices = getCameraPointIndices(visiblity_matrix)
    
    A = bundle_adjustment_sparsity(X_found, filtered_feature_flag, nCam)
    t0 = time.time()
    res = least_squares(fun, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-10, method='trf',
                        args=(nCam, n_points, camera_indices, point_indices, points_2d, K))
    t1 = time.time()
    print('time to run BA :', t1-t0, 's \nA matrix shape: ' ,  A.shape, '\n############')
    
    x1 = res.x
    number_of_cam = nCam + 1
    optimized_camera_params = x1[:number_of_cam * 6].reshape((number_of_cam, 6))
    optimized_points_3d = x1[number_of_cam * 6:].reshape((n_points, 3))

    optimized_X_all = np.zeros_like(X_all)
    optimized_X_all[X_index] = optimized_points_3d

    optimized_C_set, optimized_R_set = [], []
    for i in range(len(optimized_camera_params)):
        R = getRotation(optimized_camera_params[i, :3], 'e')
        C = optimized_camera_params[i, 3:].reshape(3,1)
        optimized_C_set.append(C)
        optimized_R_set.append(R)
    
    return optimized_R_set, optimized_C_set, optimized_X_all


