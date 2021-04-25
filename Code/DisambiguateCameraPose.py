import numpy as np

def DisambiguatePose(r_set, c_set, x3D_set):
    best_i = 0
    max_positive_depths = 0
    
    for i in range(len(r_set)):
        R, C = r_set[i],  c_set[i].reshape(-1,1) 
        r3 = R[2, :].reshape(1,-1)
        x3D = x3D_set[i]
        x3D = x3D / x3D[:,3].reshape(-1,1)
        x3D = x3D[:, 0:3]
        n_positive_depths = DepthPositivityConstraint(x3D, r3,C)
        if n_positive_depths > max_positive_depths:
            best_i = i
            max_positive_depths = n_positive_depths
#         print(n_positive_depths, i, best_i)

    R, C, x3D = r_set[best_i], c_set[best_i], x3D_set[best_i]

    return R, C, x3D 

def DepthPositivityConstraint(x3D, r3, C):
    # r3(X-C) alone doesnt solve the check positivity. z = X[2] must also be +ve 
    n_positive_depths=  0
    for X in x3D:
        X = X.reshape(-1,1) 
        if r3.dot(X-C)>0 and X[2]>0: 
            n_positive_depths+=1
    return n_positive_depths

# def DisambiguatePose_cv2(E, x1,x2, K):
#     _, R,t,_ = cv2.recoverPose(E, x1,x2, K)    
#     return R,t