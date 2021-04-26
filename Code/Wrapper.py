import numpy as np
import cv2

from EstimateFundamentalMatrix import *
from GetInliersRANSAC import *
from EssentialMatrixFromFundamentalMatrix import *
from ExtractCameraPose import *
from LinearTriangulation import *
from NonLinearTriangulation import *
from DisambiguateCameraPose  import *
from LinearPnP import *
from PnPRansac import *
from NonLinearPnP import *
from BundleAdjustment import *
from Utils.ImageUtils import *
from Utils.DataLoader import *
from Utils.MiscUtils import *

from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d

import scipy.optimize as optimize
from scipy.sparse import lil_matrix
import time
from scipy.optimize import least_squares
import argparse

import csv

K = np.array([[568.996140852,             0,  643.21055941],
              [            0, 568.988362396, 477.982801038],
              [            0,             0,             1]]).reshape(3,3)

def main():
    
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--DataPath', default="../Data/", help='base path where data files exist')
    Parser.add_argument('--savepath', default="../outputs/", help='Save files here')
    Parser.add_argument('--load_data', default= True, type = lambda x: bool(int(x)), help='load data from files or not')
    Parser.add_argument('--BA', default= True, type = lambda x: bool(int(x)), help='Do bundle adjustment or not')
    
    Args = Parser.parse_args()
    folder_name = Args.DataPath
    savepath = Args.savepath
    load_data = False #bool(int(Args.load_data))
    BA = bool(int(Args.BA))
    
    print(load_data, BA)
#     load_data= True
#     BA = False    
#     folder_name = "../Data/"
#     savepath = "../outputs_noBA/"

    #for error chart
    f = open('error_chart.csv', mode='w')
    error_chart = csv.writer(f)
    ##
    
    foldercheck(savepath)
    
    total_images = 6
    images = readImageSet(folder_name, total_images)

    """
     read the feature correspondencs as n x n_images matrix, 
     if image 1 and image 2 have correspondences, then column 1 and column 2 has data in it

     In every row, the non zero column positions signify that there are point correspondences between those images
    """ 
    feature_x, feature_y,  feature_flag, feature_descriptor = extractMatchingFeaturesFromFileNew(folder_name, total_images)
    
    
    
    if load_data:
        print('Bypassing RANSAC inlier extraction, since filtered data is available')
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
    ############################################################################################################################
    
    ##################### Compute Essential Matrix, Estimate Pose, Triangulate ###############
    """
    Register First two Images
    """      
    print('Registering images 1 and 2 ...... ')

    n,m = 0,1
    F12 = f_matrix[n,m]
    E12 = getEssentialMatrix(K, F12)
    print('Estimating poses of Camera 2')
    R_set, C_set = ExtractCameraPose(E12)

    idx = np.where(filtered_feature_flag[:,n] & filtered_feature_flag[:,m])
    pts1 = np.hstack((feature_x[idx, n].reshape((-1, 1)), feature_y[idx, n].reshape((-1, 1))))
    pts2 = np.hstack((feature_x[idx, m].reshape((-1, 1)), feature_y[idx, m].reshape((-1, 1))))

    R1_ = np.identity(3)
    C1_ = np.zeros((3,1))
    I = np.identity(3)
    pts3D_4 = []
    for i in range(len(C_set)):
        pts3D = []
        x1 = pts1
        x2 = pts2
        X = LinearTriangulation(K, C1_, R1_, C_set[i], R_set[i], x1, x2)
        X = X/X[:,3].reshape(-1,1)
        pts3D_4.append(X)

    R_chosen, C_chosen, X = DisambiguatePose(R_set, C_set, pts3D_4)
    X = X/X[:,3].reshape(-1,1)
    print('Done ### ')
    print('Performing NonLinear Triangulation...')
    X_refined = NonLinearTriangulation(K, pts1, pts2, X, R1_, C1_, R_chosen, C_chosen)
    X_refined = X_refined / X_refined[:,3].reshape(-1,1)
    
    mean_error1 = meanReprojectionError(X, pts1, pts2, R1_, C1_, R_chosen, C_chosen, K )
    mean_error2 = meanReprojectionError(X_refined, pts1, pts2, R1_, C1_, R_chosen, C_chosen, K )
    print(n+1,m+1, 'Before optimization LT: ', mean_error1, 'After optimization nLT:', mean_error2)
    print('Done ### ')

    error_row_for_chart = np.zeros((20))
    # # camera 0
    # error_row_for_chart[2] = 0
    # error_row_for_chart[8] = 0
    # error_chart.writerow(error_row_for_chart)

    # camera 0
    error_row_for_chart[3] = mean_error1
    error_row_for_chart[9] = mean_error2
    error_chart.writerow(list(error_row_for_chart))

    ########################################################################################
    ###################### Register Camera 1 and 2  #########################################

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
    ########################################################################################
    ###################### Register Remaining Cameras  #####################################
    print('Registering remaining Images ......')
    for i in range(2, total_images):
        # for chart
        error_row_for_chart = np.zeros((20))
        #for chart

        print('Registering Image: ', str(i+1) ,'......')
        feature_idx_i = np.where(X_found[:, 0] & filtered_feature_flag[:, i])
        if len(feature_idx_i[0]) < 8:
            print("Found ", len(feature_idx_i), "common points between X and ", i, " image")
            continue

        pts_i = np.hstack((feature_x[feature_idx_i, i].reshape(-1,1), feature_y[feature_idx_i, i].reshape(-1,1)))
        X = X_all[feature_idx_i, :].reshape(-1,3)
        #PnP
        R_init, C_init = PnPRANSAC(K, pts_i, X, n_iterations = 1000, error_thresh = 5)
        errorLinearPnP = reprojectionErrorPnP(X, pts_i, K, R_init, C_init)
        
        Ri, Ci = NonLinearPnP(K, pts_i, X, R_init, C_init)
        errorNonLinearPnP = reprojectionErrorPnP(X, pts_i, K, Ri, Ci)
        print("Error after linear PnP: ", errorLinearPnP, " Error after non linear PnP: ", errorNonLinearPnP)

        error_row_for_chart[0] = errorLinearPnP
        error_row_for_chart[1] = errorNonLinearPnP

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
            
            LT_error = meanReprojectionError(X, x1, x2, R_set_[j], C_set_[j], Ri, Ci, K)
            
            X = NonLinearTriangulation(K, x1, x2, X, R_set_[j], C_set_[j], Ri, Ci)
            X = X/X[:,3].reshape(-1,1)
            
            nLT_error = meanReprojectionError(X, x1, x2, R_set_[j], C_set_[j], Ri, Ci, K)
            print("Error after linear triangulation: ", LT_error, " Error after non linear triangulation: ", nLT_error)
            
            error_row_for_chart[2 + j] = LT_error
            error_row_for_chart[8 + j] = nLT_error

            X_all[idx_X_pts] = X[:,:3]
            X_found[idx_X_pts] = 1
            
            print("appended ", len(idx_X_pts[0]), " points between ", j ," and ", i)

        if BA:
            print( 'Performing Bundle Adjustment  for image : ', i  )
            R_set_, C_set_, X_all = BundleAdjustment(X_all,X_found, feature_x, feature_y,
                                                     filtered_feature_flag, R_set_, C_set_, K, nCam = i)
           
        
            for k in range(0, i+1):
                idx_X_pts = np.where(X_found[:,0] & filtered_feature_flag[:, k])
                x = np.hstack((feature_x[idx_X_pts, k].reshape((-1, 1)), feature_y[idx_X_pts, k].reshape((-1, 1))))
                X = X_all[idx_X_pts]
                BA_error = reprojectionErrorPnP(X, x, K, R_set_[k], C_set_[k])
                print("Error after BA :", BA_error)
                error_row_for_chart[14+k] = BA_error
        
        print('##################### Registered Camera : ', i+1, '######################')
        error_chart.writerow(list(error_row_for_chart))

    X_found[X_all[:,2]<0] = 0    
    print('##########################################################################')
    
    # np.save(savepath+'optimized_C_set_', C_set_)
    # np.save(savepath+'optimized_R_set_', R_set_)
    # np.save(savepath+'optimized_X_all', X_all)
    # np.save(savepath+'optimized_X_found', X_found)
    
    
    feature_idx = np.where(X_found[:, 0])
    X = X_all[feature_idx]
    x = X[:,0]
    y = X[:,1]
    z = X[:,2]
    
    # 2D plotting
    fig = plt.figure(figsize = (10, 10))
    plt.xlim(-250,  250)
    plt.ylim(-100,  500)
    plt.scatter(x, z, marker='.',linewidths=0.5, color = 'blue')
    for i in range(0, len(C_set_)):
        R1 = getEuler(R_set_[i])
        R1 = np.rad2deg(R1)
        plt.plot(C_set_[i][0],C_set_[i][2], marker=(3, 0, int(R1[1])), markersize=15, linestyle='None')
        
    plt.savefig(savepath+'2D.png')
    plt.show()
    
    # For 3D plotting
    fig = plt.figure(figsize = (10, 10))
    ax = plt.axes(projection ="3d")
    # Creating plot
    ax.scatter3D(x, y, z, color = "green")
    plt.show()
    plt.savefig(savepath+'3D.png')

    f.close()
    
if __name__ == '__main__':
    main()
         

        
