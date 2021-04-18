import numpy as np
import cv2
import scipy.optimize as optimize

def NonLinearTriangulation(K, pts1, pts2, x3D, R1, C1, R2, C2):
    """    
    K : Camera Matrix
    pts1, pts2 : Point Correspondences
    x3D :  initial 3D point 
    R2, C2 : relative camera pose

    Returns:
        x3D : optimized 3D points
    """
    
    P1 = ProjectionMatrix(R1,C1,K) 
    P2 = ProjectionMatrix(R2,C2,K)
    pts1, pts2, x3D = pts1, pts2, x3D
    
    if pts1.shape[0] != pts2.shape[0] != x3D.shape[0]:
        raise 'Check point dimensions - level nlt'

    x3D_ = []
    for i in range(len(x3D)):
        optimized_params = optimize.least_squares(
            fun=ReprojectionLoss,
            x0=x3D[i],
            method="trf",
            args=[pts1[i], pts2[i], P1,P2])
        X1 = optimized_params.x
#         X1 = X1/X1[-1]
        x3D_.append(X1[:3])
    return np.array(x3D_)

def ReprojectionLoss(X, pts1, pts2, P1, P2):
    
    X = homo(X.reshape(1,-1)).reshape(-1,1) # make X it a column of homogenous vector
    
    p1_1T, p1_2T, p1_3T = P1 # rows of P1
    p1_1T, p1_2T, p1_3T = p1_1T.reshape(1,-1), p1_2T.reshape(1,-1),p1_3T.reshape(1,-1)

    p2_1T, p2_2T, p2_3T = P2 # rows of P2
    p2_1T, p2_2T, p2_3T = p2_1T.reshape(1,-1), p2_2T.reshape(1,-1), p2_3T.reshape(1,-1)

    ## reprojection error for reference camera points - j = 1
    u1,v1 = pts1[0], pts1[1]
    u1_proj = np.divide(p1_1T.dot(X) , p1_3T.dot(X))
    v1_proj =  np.divide(p1_2T.dot(X) , p1_3T.dot(X))
    E1= np.square(v1 - v1_proj) + np.square(u1 - u1_proj)

    
    ## reprojection error for second camera points - j = 2    
    u2,v2 = pts2[0], pts2[1]
    u2_proj = np.divide(p2_1T.dot(X) , p2_3T.dot(X))
    v2_proj =  np.divide(p2_2T.dot(X) , p2_3T.dot(X))
    
    E2= np.square(v2 - v2_proj) + np.square(u2 - u2_proj)
    error = E1 + E2
    return error.squeeze()



###################################################### 
########### Reprojection Error Functions #############
######################################################

def ProjectionMatrix(R,C,K):
    C = np.reshape(C, (3, 1))        
    I = np.identity(3)
    P = np.dot(K, np.dot(R, np.hstack((I, -C))))
    return P


def meanReprojectionError(x3D, pts1, pts2, R1, C1, R2, C2, K ):    
    Error = []
    for pt1, pt2, X in zip(pts1, pts2, x3D):
        e1,e2 = ReprojectionError(X, pt1, pt2, R1, C1, R2, C2, K )
        Error.append(e1+e2)
        
    return np.mean(Error)

def ReprojectionError(X, pt1, pt2, R1, C1, R2, C2, K ):
    
    P1 = ProjectionMatrix(R1,C1,K) 
    P2 = ProjectionMatrix(R2,C2,K)

    X = homo(X.reshape(1,-1)).reshape(-1,1) # make X it a column of homogenous vector
    
    p1_1T, p1_2T, p1_3T = P1 # rows of P1
    p1_1T, p1_2T, p1_3T = p1_1T.reshape(1,-1), p1_2T.reshape(1,-1),p1_3T.reshape(1,-1)

    p2_1T, p2_2T, p2_3T = P2 # rows of P2
    p2_1T, p2_2T, p2_3T = p2_1T.reshape(1,-1), p2_2T.reshape(1,-1), p2_3T.reshape(1,-1)

    ## reprojection error for reference camera points - j = 1
    u1,v1 = pt1[0], pt1[1]
    u1_proj = np.divide(p1_1T.dot(X) , p1_3T.dot(X))
    v1_proj =  np.divide(p1_2T.dot(X) , p1_3T.dot(X))
    E1= np.square(v1 - v1_proj) + np.square(u1 - u1_proj)
    
    ## reprojection error for second camera points - j = 2    
    u2,v2 = pt2[0], pt2[1]
    u2_proj = np.divide(p2_1T.dot(X) , p2_3T.dot(X))
    v2_proj =  np.divide(p2_2T.dot(X) , p2_3T.dot(X))
    
    E2= np.square(v2 - v2_proj) + np.square(u2 - u2_proj)
    
    return E1, E2

def projectPts(R, C, x3D, K):
    I  = np.identity(3)
    P2 = np.dot(K, np.dot(R, np.hstack((I, -C.reshape(3,1)))))
    x3D_4 = homo(x3D)
    x_proj = np.dot(P2, x3D_4.T)
    x_proj = (x_proj/x_proj[2,:]).T
    return x_proj

def homo(pts):
    return np.hstack((pts, np.ones((pts.shape[0], 1))))


##########################################################################################################################
#################################### Nothing to see here #################################################################
##########################################################################################################################
# def ReprojectionError(x3D, pts1, pts2, R1, C1, R2, C2, K ):

#     X3D = homo(x3D)

#     P1 = ProjectionMatrix(R1,C1,K) 
#     P2 = ProjectionMatrix(R2,C2,K)
#     p1_1T, p1_2T, p1_3T = P1 # rows of P1
#     p1_1T, p1_2T, p1_3T = p1_1T.reshape(1,-1), p1_2T.reshape(1,-1),p1_3T.reshape(1,-1)

#     p2_1T, p2_2T, p2_3T = P2 # rows of P2
#     p2_1T, p2_2T, p2_3T = p2_1T.reshape(1,-1), p2_2T.reshape(1,-1), p2_3T.reshape(1,-1)

#     Error1 = []
#     Error2 = []
#     for i in range(len(X3D)):
#         X = X3D[i].reshape(-1,1)
#         u1,v1 = pts1[i,0], pts1[i,1]
#         UV1_ = np.hstack([u1,v1])    
#         u2,v2 = pts2[i,0], pts2[i,1]
#         UV2_ = np.hstack([u2,v2])

#         u1_proj = p1_1T.dot(X) / p1_3T.dot(X)
#         v1_proj =  p1_2T.dot(X) / p1_3T.dot(X)
#         UV1_proj = np.hstack([u1_proj,v1_proj])

#         u2_proj = p2_1T.dot(X) / p2_3T.dot(X)
#         v2_proj =  p2_2T.dot(X) / p2_3T.dot(X)
#         UV2_proj = np.hstack([u2_proj,v2_proj])

#         E1_vec= UV1_ - UV1_proj
#         E1 = np.sum(E1_vec**2)    
#         E2_vec= UV2_ - UV2_proj
#         E2 = np.sum(E2_vec**2)

#         # E = np.linalg.norm((UV_ - UV_proj), 2)
#         Error1.append(E1)
#         Error2.append(E2)

# #     error = np.sum(Error1) + np.sum(Error1)
#     mean_error = (np.mean(Error1) + np.mean(Error1))/2
#     return mean_error

# def ReprojectionLoss_LM(X, pts1, pts2, P1, P2):
    
#     X = homo(X.reshape(1,-1)).reshape(-1,1) # make X it a column of homogenous vector
    
#     p1_1T, p1_2T, p1_3T = P1 # rows of P1
#     p1_1T, p1_2T, p1_3T = p1_1T.reshape(1,-1), p1_2T.reshape(1,-1),p1_3T.reshape(1,-1)

#     p2_1T, p2_2T, p2_3T = P2 # rows of P2
#     p2_1T, p2_2T, p2_3T = p2_1T.reshape(1,-1), p2_2T.reshape(1,-1), p2_3T.reshape(1,-1)

#     ## reprojection error for reference camera points - j = 1
#     u1,v1 = pts1[0], pts1[1]
#     u1_proj = np.divide(p1_1T.dot(X) , p1_3T.dot(X))
#     v1_proj =  np.divide(p1_2T.dot(X) , p1_3T.dot(X))
#     E1= np.square(v1 - v1_proj) + np.square(u1 - u1_proj)

    
#     ## reprojection error for second camera points - j = 2    
#     u2,v2 = pts2[0], pts2[1]
#     u2_proj = np.divide(p2_1T.dot(X) , p2_3T.dot(X))
#     v2_proj =  np.divide(p2_2T.dot(X) , p2_3T.dot(X))
    
#     E = np.array([np.square(u1 - u1_proj), np.square(v1 - v1_proj), np.square(u2 - u2_proj), np.square(v2 - v2_proj)]).squeeze()

#     return E
