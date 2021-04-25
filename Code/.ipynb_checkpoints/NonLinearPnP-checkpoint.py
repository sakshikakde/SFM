
import scipy.optimize as optimize
import numpy as np
from Utils.MiscUtils import *


def NonLinearPnP(K, pts, x3D, R0, C0):
    """    
    K : Camera Matrix
    pts1, pts2 : Point Correspondences
    x3D :  initial 3D point 
    R2, C2 : relative camera pose - estimated from PnP
    Returns:
        x3D : optimized 3D points
    """

    Q = getQuaternion(R0)
    X0 = [Q[0] ,Q[1],Q[2],Q[3], C0[0], C0[1], C0[2]] 

    optimized_params = optimize.least_squares(
        fun = PnPLoss,
        x0=X0,
        method="trf",
        args=[x3D, pts, K])
    X1 = optimized_params.x
    Q = X1[:4]
    C = X1[4:]
    R = getRotation(Q)
    return R, C

def PnPLoss(X0, x3D, pts, K):
    
    Q, C = X0[:4], X0[4:].reshape(-1,1)
    R = getRotation(Q)
    P = ProjectionMatrix(R,C,K)
    
    Error = []
    for X, pt in zip(x3D, pts):

        p_1T, p_2T, p_3T = P# rows of P
        p_1T, p_2T, p_3T = p_1T.reshape(1,-1), p_2T.reshape(1,-1), p_3T.reshape(1,-1)


        X = homo(X.reshape(1,-1)).reshape(-1,1) # make X it a column of homogenous vector
        ## reprojection error for reference camera points 
        u, v = pt[0], pt[1]
        u_proj = np.divide(p_1T.dot(X) , p_3T.dot(X))
        v_proj =  np.divide(p_2T.dot(X) , p_3T.dot(X))

        E = np.square(v - v_proj) + np.square(u - u_proj)

        Error.append(E)

    sumError = np.mean(np.array(Error).squeeze())
    return sumError