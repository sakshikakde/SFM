from scipy.spatial.transform import Rotation 

def getQuaternion(R2):
    Q = Rotation.from_matrix(R2)
    return Q.as_quat()

def getRotation(Q):
    R = Rotation.from_quat(Q)
    return R.as_matrix()
def homo(pts):
    return np.hstack((pts, np.ones((pts.shape[0], 1))))

def ProjectionMatrix(K,R,C):
    I  = np.identity(3)
    return np.dot(K, np.dot(R, np.hstack((I, -C))))


def NonLinearPnP(K, pts1, pts2, x3D, R2, C2):
    """    
    K : Camera Matrix
    pts1, pts2 : Point Correspondences
    x3D :  initial 3D point 
    R2, C2 : relative camera pose - estimated from PnP

    Returns:
        x3D : optimized 3D points
    """
    if pts1.shape[0] != pts2.shape[0] != x3D.shape[0]:
        raise 'Check point dimensions - level nlt'
    
    Q = getQuaternion(R2)
    X0 = [Q[0] ,Q[1],Q[2],Q[3], C2[0], C2[1], C2[2]] 

    optimized_params = optimize.least_squares(
        fun = PnPLoss,
        x0=X0,
        method="trf",
        args=[x3D, pts1, pts2, K])
    X1 = optimized_params.x
    Q = X1[:4]
    C = X1[4:]
    R = getRotation(Q)
    return R, C


def PnPLoss(X0, x3D, pts1, pts2, K):
    
    Q, C2 = X0[:4], X0[4:].reshape(-1,1)
    R2 = getRotation(Q)
    R1, C1 = np.identity(3), np.zeros((3,1))
    P1 = ProjectionMatrix(K,R1,C1) 
    P2 = ProjectionMatrix(K,R2,C2)
    
    Error = []
    for X, pt1, pt2 in zip(x3D, pts1, pts2):

        p1_1T, p1_2T, p1_3T = P1 # rows of P1
        p1_1T, p1_2T, p1_3T = p1_1T.reshape(1,-1), p1_2T.reshape(1,-1),p1_3T.reshape(1,-1)

        p2_1T, p2_2T, p2_3T = P2 # rows of P2
        p2_1T, p2_2T, p2_3T = p2_1T.reshape(1,-1), p2_2T.reshape(1,-1), p2_3T.reshape(1,-1)

        X = homo(X.reshape(1,-1)).reshape(-1,1) # make X it a column of homogenous vector
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
        error = E1 + E2
        Error.append(error)
        
#     meanError = np.mean(np.array(Error).squeeze())
#     return meanError
    sumError = np.mean(np.array(Error).squeeze())
    return sumError