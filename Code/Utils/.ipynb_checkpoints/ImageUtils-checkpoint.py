import cv2
import numpy as np

def readImageSet(folder_name, n_images):
    print("Reading images from ", folder_name)
    images = []
    for n in range(1, n_images+1):
        image_path = folder_name + "/" + str(n) + ".jpg"
        image = cv2.imread(image_path)
        if image is not None:
            images.append(image)			
        else:
            print("Error in loading image ", image)

    return images

def makeImageSizeSame(imgs):
    images = imgs.copy()
    sizes = []
    for image in images:
        x, y, ch = image.shape
        sizes.append([x, y, ch])

    sizes = np.array(sizes)
    x_target, y_target, _ = np.max(sizes, axis = 0)
    
    images_resized = []

    for i, image in enumerate(images):
        image_resized = np.zeros((x_target, y_target, sizes[i, 2]), np.uint8)
        image_resized[0:sizes[i, 0], 0:sizes[i, 1], 0:sizes[i, 2]] = image
        images_resized.append(image_resized)

    return images_resized
def showMatches(image_1, image_2, pts1, pts2, color, file_name):
#     file_name =  None
    # image_1 = img_1
    # image_2 = img_2

    image_1, image_2 = makeImageSizeSame([image_1, image_2])
    concat = np.concatenate((image_1, image_2), axis = 1)

    if pts1 is not None:
        corners_1_x = pts1[:,0].copy().astype(int)
        corners_1_y = pts1[:,1].copy().astype(int)
        corners_2_x = pts2[:,0].copy().astype(int)
        corners_2_y = pts2[:,1].copy().astype(int)
        corners_2_x += image_1.shape[1]

        for i in range(corners_1_x.shape[0]):
            cv2.line(concat, (corners_1_x[i], corners_1_y[i]), (corners_2_x[i] ,corners_2_y[i]), color, 1)
    cv2.imshow(file_name, concat)
    cv2.waitKey() 
    if file_name is not None:    
        cv2.imwrite(file_name, concat)
    cv2.destroyAllWindows()
    return concat

def showAllMatches(images, feature_x, feature_y, feature_flag, total_images):
    for i in range(0, total_images-1):
        for j in range(i+1, total_images):
            idx = np.where(feature_flag[:,i] & feature_flag[:,j])
            pts1 = np.hstack((feature_x[idx, i].reshape((-1, 1)), feature_y[idx, i].reshape((-1, 1))))
            pts2 = np.hstack((feature_x[idx, j].reshape((-1, 1)), feature_y[idx, j].reshape((-1, 1))))
            filename = '../Results/' + str(i) + str(j) + ".png"
            concat = showMatches(images[i], images[j], pts1, pts2, (0,0,255), filename)
            
############################################################################################################            
def showMatches_filtered(image_1, image_2, pts1, pts2, color1, pts1_filt, pts2_filt, color2, file_name):
    image_1, image_2 = makeImageSizeSame([image_1, image_2])
    concat = np.concatenate((image_1, image_2), axis = 1)

    if pts1 is not None:
        corners_1_x = pts1[:,0].copy().astype(int)
        corners_1_y = pts1[:,1].copy().astype(int)
        corners_2_x = pts2[:,0].copy().astype(int)
        corners_2_y = pts2[:,1].copy().astype(int)
        corners_2_x += image_1.shape[1]

        for i in range(corners_1_x.shape[0]):
            cv2.line(concat, (corners_1_x[i], corners_1_y[i]), (corners_2_x[i] ,corners_2_y[i]), color1, 1)

    if pts1_filt is not None:
        corners_1_x = pts1_filt[:,0].copy().astype(int)
        corners_1_y = pts1_filt[:,1].copy().astype(int)
        corners_2_x = pts2_filt[:,0].copy().astype(int)
        corners_2_y = pts2_filt[:,1].copy().astype(int)
        corners_2_x += image_1.shape[1]

        for i in range(corners_1_x.shape[0]):
            cv2.line(concat, (corners_1_x[i], corners_1_y[i]), (corners_2_x[i] ,corners_2_y[i]), color2, 1)
    
    
    cv2.imshow(file_name, concat)
    cv2.waitKey() 
    if file_name is not None:    
        cv2.imwrite(file_name, concat)
    cv2.destroyAllWindows()


def showFilteredMatches(images, feature_x, feature_y, filtered_feature_flag, feature_flag, total_images):
    for i in range(0, total_images-1):
        for j in range(i+1, total_images):
                        
            idx = np.where(feature_flag[:,i] & feature_flag[:,j])
            pts1 = np.hstack((feature_x[idx, i].reshape((-1, 1)), feature_y[idx, i].reshape((-1, 1))))
            pts2 = np.hstack((feature_x[idx, j].reshape((-1, 1)), feature_y[idx, j].reshape((-1, 1))))

            idx = np.where(filtered_feature_flag[:,i] & filtered_feature_flag[:,j])
            pts1_filt = np.hstack((feature_x[idx, i].reshape((-1, 1)), feature_y[idx, i].reshape((-1, 1))))
            pts2_filt = np.hstack((feature_x[idx, j].reshape((-1, 1)), feature_y[idx, j].reshape((-1, 1))))
            
            filename = '../Results/filtered/' + str(i) + str(j) + ".png"
            showMatches_filtered(images[i], images[j], pts1, pts2,(0,0,255), pts1_filt, pts2_filt, (0,255,0), filename)

            
############################################################################################################            