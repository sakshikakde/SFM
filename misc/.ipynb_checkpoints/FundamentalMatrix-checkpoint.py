import numpy as np
from misc.RANSAC import *

def EssentialMatrix(K1,K2, F):
    E = K2.T.dot(F).dot(K1)
    U,s,V = np.linalg.svd(E)
    s = [1,1,0]
    E_ = np.dot(U,np.dot(np.diag(s),V))
    return E_

def FundamentalMatrix(data ,s = 8, thresh = 0.01,n_iterations = 100):
    model_F = FMatrix()
    ransac = RANSAC(model_F)
    F, inlier_mask = ransac.fit(data,s ,thresh , n_iterations) 
    return F, inlier_mask

def cv2_FundamentalMatrix(pts1,pts2, a = 1, b= 0.90):
    bestF,mask =  cv2.findFundamentalMat(pts1,pts2,cv2.RANSAC, a,b)
#     bestF,mask =  cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)
    if bestF is None or len(bestF)==0:
        print('F bad :( ')
        return None, None
    U,s,V = np.linalg.svd(bestF)
    s[2] = 0.0
    F =  U.dot(np.diag(s).dot(V))
    return F, mask.ravel()

class FMatrix():
    #### helper functions of class ###
    def errorCond(self,x1,x2,F): 
        """
        check the epipolar constraint
        """
        x1tmp=np.array([x1[0], x1[1], 1]).T
        x2tmp=np.array([x2[0], x2[1], 1])
        
        return abs(np.squeeze(np.matmul((np.matmul(x2tmp,F)),x1tmp)))
    def normalize(self,uv):
        """
        https://www.cc.gatech.edu/classes/AY2016/cs4476_fall/results/proj3/html/sdai30/index.html
        """
        uv_ = np.mean(uv, axis=0)
        u_,v_ = uv_[0], uv_[1]
        u_cap, v_cap = uv[:,0] - u_, uv[:,1] - v_

        s = (2/np.mean(u_cap**2 + v_cap**2))**(0.5)
        T_scale = np.diag([s,s,1])
        T_trans = np.array([[1,0,-u_],[0,1,-v_],[0,0,1]])
        T = T_scale.dot(T_trans)

        x_ = np.column_stack((uv, np.ones(len(uv))))
        x_norm = (T.dot(x_.T)).T

        return  x_norm, T
    
    #### main functions of the class ###
    def fit(self,data):
        normalised = True
        
        x1,x2 = data[0], data[1]
        if normalised == True:
            x1_norm, T1 = self.normalize(x1)
            x2_norm, T2 = self.normalize(x2)
        else:
            x1_norm,x2_norm = x1,x2
            
        A = np.zeros((len(x1_norm),9))
        for i in range(0, len(x1_norm)):
            x_1,y_1 = x1_norm[i][0], x1_norm[i][1]
            x_2,y_2 = x2_norm[i][0], x2_norm[i][1]
            A[i] = np.array([x_1*x_2, x_2*y_1, x_2, y_2*x_1, y_2*y_1, y_2, x_1, y_1, 1])

        # Ai = [ui'ui, ui'vi, ui', vi'ui, vi'vi, vi', ui, vi, 1]
        U, S, VT = np.linalg.svd(A, full_matrices=True)
        F = VT.T[:, -1]
        F = F.reshape(3,3)
        
        u, s, vt = np.linalg.svd(F)
        s = np.diag(s)
        s[2,2] = 0
        F = u @ s @ vt
        
        if normalised:
            F = (T2.T) @ F @ T1
        return F
    
    def check(self,data, F):
        """
        check epipolarConstraint
        """
        x1pts, x2pts =  data[0], data[1]
        E = []
        for i,(pt1, pt2) in enumerate(zip(x1pts,x2pts)):
            error = self.errorCond(pt1,pt2,F)
            E.append(error)
        return E
    
####################################################################################
#     def check_(self,data, thresh, F, mask):
#         """
#         check epipolarConstraints
#         """
#         x1pts, x2pts =  data[0], data[1]        
#         for i,(pt1, pt2) in enumerate(zip(x1pts,x2pts)):
# #             error = self.errorCond(pt1,pt2,F)
#             error = self.FmatrixCond(pt1,pt2,F)
#             if error < thresh:
#                 mask[i] = 1
#         return mask
    
#     def errorCond(self,x1,x2,F):
#         # To satisfy epipolar constraint
#         x1 = x1.T
#         err = np.matmul(x2,F)
#         err = np.matmul(err,x1)
#         err  =abs(np.squeeze(err))
#         return err