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

def showMatches(image_1, image_2, matched_pairs, color, file_name):

    # image_1 = img_1
    # image_2 = img_2

    image_1, image_2 = makeImageSizeSame([image_1, image_2])
    concat = np.concatenate((image_1, image_2), axis = 1)

    if matched_pairs is not None:
        corners_1_x = matched_pairs[:,0].copy().astype(int)
        corners_1_y = matched_pairs[:,1].copy().astype(int)
        corners_2_x = matched_pairs[:,2].copy().astype(int)
        corners_2_y = matched_pairs[:,3].copy().astype(int)
        corners_2_x += image_1.shape[1]

        for i in range(corners_1_x.shape[0]):
            cv2.line(concat, (corners_1_x[i], corners_1_y[i]), (corners_2_x[i] ,corners_2_y[i]), color, 1)
    
    if file_name is not None:      
        cv2.imshow(file_name, concat)
        cv2.waitKey() 
        cv2.destroyAllWindows()
        cv2.imwrite(file_name, concat)
    
