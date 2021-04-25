import numpy as np
import cv2
from EstimateFundamentalMatrix import *
from GetInliersRANSAC import *
from EssentialMatrixFromFundamentalMatrix import *
from ExtractCameraPose import *
from LinearTriangulation import *
from NonLinearTriangulation import *
from DisambiguateCameraPose  import *
from PnPRansac import *
from NonLinearPnP import *
from BundleAdjustment import *
from Utils.ImageUtils import *
from Utils.DataLoader import *
from Utils.MiscUtils import *
from matplotlib import pyplot as plt
import scipy.optimize as optimize
from scipy.sparse import lil_matrix
import time
from scipy.optimize import least_squares


K = np.array([[568.996140852,             0,  643.21055941],
              [            0, 568.988362396, 477.982801038],
              [            0,             0,             1]]).reshape(3,3)

def main():
    
    load_data = True
    BA = True
    
    folder_name = "../Data/"
    total_images = 6
    images = readImageSet(folder_name, total_images)

    """
     read the feature correspondencs as n x n_images matrix, 
     if image 1 and image 2 have correspondences, then column 1 and column 2 has data in it

     In every row, the non zero column positions signify that there are point correspondences between those images
    """ 
    feature_x, feature_y,  feature_flag, feature_descriptor = extractMatchingFeaturesFromFileNew(folder_name, total_images)
    
    
    
    if load_data:
        filtered_feature_flag = np.load('./tmp_files/filtered_feature_flag.npy', allow_pickle=True)
        f_matrix = np.load('./tmp_files/f_matrix.npy', allow_pickle=True)
    else:        
        filtered_feature_flag = np.zeros_like(feature_flag)
        f_matrix = np.empty(shape=(total_images, total_images), dtype=object)

        for i in range(0, total_images - 1):
            # filtered_feature_flag[:, i] = feature_descriptor[:,i]
            for j in range(i + 1, total_images):

                idx = np.where(feature_flag[:,i] & feature_flag[:,j])
                pts1 = np.hstack((feature_x[idx, i].reshape((-1, 1)), feature_y[idx, i].reshape((-1, 1))))
                pts2 = np.hstack((feature_x[idx, j].reshape((-1, 1)), feature_y[idx, j].reshape((-1, 1))))
                # showMatches(images[i], images[j], pts1, pts2, (0,255,0), None)
                idx = np.array(idx).reshape(-1)
                # print(len(idx))
                if len(idx) > 8:
                    F_best, chosen_idx = getInliers(pts1, pts2, idx)
                    print('At image : ',  i,j, '|| Number of inliers: ', len(chosen_idx), '/', len(idx) )            
                    f_matrix[i, j] = F_best
                    filtered_feature_flag[chosen_idx, j] = 1
                    filtered_feature_flag[chosen_idx, i] = 1
                    
    ######################################## Obtained Feature points  ##########################################################
        
    """
    Register First two Images
    """      
    print('Registering images 1 and 2 ...... ')
    n,m = 0,1
    F12 = f_matrix[n,m]
    E12 = getEssentialMatrix(K, F12)
    R_set, C_set = ExtractCameraPose(E12)

    idx = np.where(filtered_feature_flag[:,n] & filtered_feature_flag[:,m])
    pts1 = np.hstack((feature_x[idx, n].reshape((-1, 1)), feature_y[idx, n].reshape((-1, 1))))
    pts2 = np.hstack((feature_x[idx, m].reshape((-1, 1)), feature_y[idx, m].reshape((-1, 1))))

    R1 = np.identity(3)
    C1 = np.zeros((3,1))
    I = np.identity(3)
    pts3D_4 = []
    for i in range(len(C_set)):
        pts3D = []
        x1 = pts1
        x2 = pts2
        X = LinearTriangulation(K, C1, R1, C_set[i], R_set[i], x1, x2)
        X = X/X[:,3].reshape(-1,1)
        pts3D_4.append(X)

    R_chosen, C_chosen, X = DisambiguatePose(R_set, C_set, pts3D_4)
    X = X/X[:,3].reshape(-1,1)
    print(' Poses Estimated')
    X_refined = NonLinearTriangulation(K, pts1, pts2, X, R1, C1, R_chosen, C_chosen)
    X_refined = X_refined / X_refined[:,3].reshape(-1,1)
    print(' 3D points Estimated')
    mean_error1 = meanReprojectionError(X, pts1, pts2, R1, C1, R_chosen, C_chosen, K )
    mean_error2 = meanReprojectionError(X_refined, pts1, pts2, R1, C1, R_chosen, C_chosen, K )

    print('Before optimization: ', mean_error1, 'After optimization :', mean_error2)


    ############################################################################################################
    ######################################## Register Camera 1 and 2  ##########################################

    X_all = np.zeros((feature_x.shape[0], 3))
    camera_indices = np.zeros((feature_x.shape[0], 1), dtype = int) 
    X_found = np.zeros((feature_x.shape[0], 1), dtype = int)

    X_all[idx] = X[:, :3]
    X_found[idx] = 1
    camera_indices[idx] = 1

    # print(np.nonzero(X_found[idx])[0].shape)
    X_found[np.where(X_all[:,2] < 0)] = 0
    # print(len(idx[0]), '--' ,np.nonzero(X_found[idx])[0].shape)

    C_set_ = []
    R_set_ = []

    C0 = np.zeros(3)
    R0 = np.identity(3)
    C_set_.append(C0)
    R_set_.append(R0)

    C_set_.append(C_chosen)
    R_set_.append(R_chosen)
    print(' #####################  Registered Cameras 1 and 2 #####################' )
    ############################################################################################################
    ######################################## Registering Remaining images  #####################################
    print('Registering remaining Images ......')
    
    for i in range(2, total_images):
        print('Registering Image: ', str(i+1) ,'......')
        feature_idx_i = np.where(X_found[:, 0] & filtered_feature_flag[:, i])
        if len(feature_idx_i[0]) < 8:
            print("Found ", len(feature_idx_i), "common points between X and ", i, " image")
            continue

        pts_i = np.hstack((feature_x[feature_idx_i, i].reshape(-1,1), feature_y[feature_idx_i, i].reshape(-1,1)))
        X = X_all[feature_idx_i, :].reshape(-1,3)
        #PnP
        R_init, C_init = PnPRANSAC(K, pts_i, X, n_iterations = 1000, error_thresh = 5)
        Ri, Ci = NonLinearPnP(K, pts_i, X, R_init, C_init)

        C_set_.append(Ci)
        R_set_.append(Ri)

        #trianglulation
        for j in range(0, i):
            # idx_X_pts = np.where(X_found[:, 0] & filtered_feature_flag[:, j] & filtered_feature_flag[:, i])
            idx_X_pts = np.where(filtered_feature_flag[:, j] & filtered_feature_flag[:, i])
            if (len(idx_X_pts[0]) < 8):
                continue

            x1 = np.hstack((feature_x[idx_X_pts, j].reshape((-1, 1)), feature_y[idx_X_pts, j].reshape((-1, 1))))
            x2 = np.hstack((feature_x[idx_X_pts, i].reshape((-1, 1)), feature_y[idx_X_pts, i].reshape((-1, 1))))

            X = LinearTriangulation(K, C_set_[j], R_set_[j], Ci, Ri, x1, x2)
            X = X/X[:,3].reshape(-1,1)
            X = NonLinearTriangulation(K, x1, x2, X, R_set_[j], C_set_[j], Ri, Ci)
            X = X/X[:,3].reshape(-1,1)


            X_all[idx_X_pts] = X[:,:3]
            X_found[idx_X_pts] = 1

            print("    - appended ", len(idx_X_pts[0]), " points between ", j+1 ," and ", i+1)

        if BA:
            print( 'Performing Bundle Adjustment  for image : ', i  )
            R_set_, C_set_, X_all = BundleAdjustment(X_all,X_found, feature_x, feature_y, filtered_feature_flag, R_set_, C_set_, K, nCam = i)
        print('##################### Registered Camera : ', i+1, '######################')
        
    np.save('./tmp_files/optimized_C_set_', C_set_)
    np.save('./tmp_files/optimized_R_set_', R_set_)
    np.save('./tmp_files/optimized_X_all', X_all)
    np.save('./tmp_files/optimized_X_found', X_found)
    
    
    feature_idx = np.where(X_found[:, 0])
    X = X_all[feature_idx]

    x = X[:,0]
    y = X[:,1]
    z = X[:,2]

    x[(x < -500) | (x > 500)] = 0 
    y[(y < -500) | (y > 500)] = 0 
    z[(z <= 0) | (z > 500)] = 0

    plt.xlim(-250,  250)
    plt.ylim(0,  500)
    plt.set_xlabel('x')
    plt.set_ylabel('z')
    plt.scatter(x, z, marker='.',linewidths=0.5, color = 'blue')
    plt.show()
    plt.savefig('2D.png')

    # For 3D plotting
    ax = plt.axes(projection='3d')

    # Data for three-dimensional scattered points
    ax.scatter3D(x,y,z, s=1)  # cmap='viridis',
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim([-250,  250])
    ax.set_ylim([-250,  250])
    ax.set_zlim([0,  500])
    plt.show()
    plt.savefig('3D.png')
    
    
if __name__ == '__main__':
    main()
 