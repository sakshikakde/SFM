import cv2
import numpy as np 
from scipy.spatial.transform import Rotation 
import os

def foldercheck(Savepath):
    if(not (os.path.isdir(Savepath))):
        print(Savepath, "  was not present, creating the folder...")
        os.makedirs(Savepath)
        
def project3DPoints(K, R, C, X, image, points):
    I = np.identity(3)
    P = np.dot(K, np.dot(R, np.hstack((I, -C.reshape(3,1)))))
    if X.shape[1] != 4:
        X = homo(X)
    x = np.dot(P, X.T)
    # x01 = np.dot(P1, X01.T)

    x = x/x[2,:]

    xi = x[0, :].T
    yi = x[1, :].T
    im = image.copy()
    for i in range(points.shape[0]):

        x1, y1 = xi[i], yi[i]
        x2, y2 = points[i, 0], points[i, 1]
        cv2.circle(im, (int(x1), int(y1)), 3, (255,0,0), -1)
        cv2.circle(im, (int(x2), int(y2)), 3, (0,255,0), -1)

    cv2.imshow("im", im)
    cv2.waitKey() 
    cv2.destroyAllWindows()
    
        
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

    # X = homo(X.reshape(1,-1)).reshape(-1,1) # make X it a column of homogenous vector
    
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


########################################################################################################

def getQuaternion(R2):
    Q = Rotation.from_matrix(R2)
    return Q.as_quat()

def getEuler(R2):
    euler = Rotation.from_matrix(R2)
    return euler.as_rotvec()

def getRotation(Q, type_ = 'q'):
    if type_ == 'q':
        R = Rotation.from_quat(Q)
        return R.as_matrix()
    elif type_ == 'e':
        R = Rotation.from_rotvec(Q)
        return R.as_matrix()