from LinearPnP import PnP
import numpy as np
from Utils.MiscUtils import ProjectionMatrix, homo


def PnPError(x, X, R, C, K):
    u,v = x
    X = homo(X.reshape(1,-1)).reshape(-1,1) # make X it a column of homogenous vector
    C = C.reshape(-1, 1)
    P = ProjectionMatrix(R,C,K)
    p1, p2, p3 = P
        
    u_proj = np.divide(p1.dot(X) , p3.dot(X))
    v_proj =  np.divide(p2.dot(X) , p3.dot(X))

    x_proj = np.hstack((u_proj, v_proj))
    x = np.hstack((u, v))
    e = np.linalg.norm(x - x_proj)
#     e = np.sqrt(np.square(u - u_proj) + np.square(v - v_proj))
    return  e

def PnPRANSAC(K, features, x3D, n_iterations = 1000, error_thresh = 5):

    inliers_thresh = 0
    chosen_indices = []
    chosen_R, chosen_t = None, None
    n_rows = x3D.shape[0]
    
    for i in range(0, n_iterations):
        
        #select 6 points randomly
        random_indices = np.random.choice(n_rows, size=6)
        X_set, x_set = x3D[random_indices], features[random_indices]
        
        R,C = PnP(X_set, x_set, K)
        
        indices = []
        errors = []
        if R is not None:
            for j in range(n_rows):
                feature = features[j]
                X = x3D[j]
                error = PnPError(feature, X, R, C, K)

                if error < error_thresh:
                    indices.append(j)
                    
        if len(indices) > inliers_thresh:
            inliers_thresh = len(indices)
            chosen_indices = indices
            chosen_R = R
            chosen_t = C
            
    #     filtered_features = features[chosen_indices, :]
    return chosen_R, chosen_t
