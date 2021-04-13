import numpy as np
from EstimateFundamentalMatrix import *
import cv2

def errorF(feature, F): 
    """
    check the epipolar constraint
    """
    x1,x2 = feature[3:5], feature[5:7]
    x1tmp=np.array([x1[0], x1[1], 1])
    x2tmp=np.array([x2[0], x2[1], 1]).T

    error = np.dot(x2tmp, np.dot(F, x1tmp))
    
    return np.abs(error)

def meanErrorF(features, F):
    total_error = 0
    for n in range(features.shape[0]):
        feature = features[n, :]
        # x1,x2 = feature[3:5], feature[5:7]
        total_error = total_error + errorF(feature,F)

    mean_error = total_error / features.shape[0] 
    return mean_error


def getInliers(features):
    n_iterations = 2000
    error_thresh = 0.0002
    inliers_thresh = 0
    chosen_indices = []
    chosen_f = None

    for i in range(0, n_iterations):
  
        #select 8 points randomly
        n_rows = features.shape[0]
        random_indices = np.random.choice(n_rows, size=8)
        features_8 = features[random_indices, :] 
        f_8 = EstimateFundamentalMatrix(features_8)
        # f_8, _ = cv2.findFundamentalMat(np.int32(features_8[:, 3:5]), np.int32(features_8[:, 5:7]),cv2.FM_LMEDS)
        indices = []
        if f_8 is not None:
            for j in range(n_rows):
                feature = features[j]
                error = errorF(feature, f_8)
                
                if error < error_thresh:
                    indices.append(j)

        if len(indices) > inliers_thresh:
            inliers_thresh = len(indices)
            chosen_indices = indices
            chosen_f = f_8

    filtered_features = features[chosen_indices, :]
    return chosen_f, filtered_features



