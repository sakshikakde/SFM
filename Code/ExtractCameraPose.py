import numpy as np

def ExtractCameraPose(E):

    W = np.array([[0,-1,0],[1,0,0],[0,0,-1]])
    Z = np.array([[0,1,0],[-1,0,0],[0,0,0]])

    U,S,VT = np.linalg.svd(E)
    S = np.dot(U, np.dot(Z, U.T))
    R1 = np.dot(U, np.dot(W, VT))
    R2 = np.dot(U, np.dot(W.T, VT))

    t = np.array([-S[1,2],S[0,2],-S[0,1]]).reshape(3,1)
    
    P1 = np.hstack((R1, t))
    P2 = np.hstack((R1, -t))
    P3 = np.hstack((R2, t))
    P4 = np.hstack((R2, -t))
    return [P1, P2, P3, P4]

