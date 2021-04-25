import numpy as np

def extractMatchingFeaturesFromFile(folder_name, total_images):

    feature_matrix = np.empty(shape=(total_images, total_images), dtype=object)
    for i in range(total_images):
        for j in range(total_images):
            feature_matrix[i,j] = list()

    for n in range(1, total_images):
        # print(n)
        matching_file_name = folder_name + "matching" + str(n) + ".txt"
        file_object = open(matching_file_name,"r")
        nFeatures = 0

        features_list = []
        for i, line in enumerate(file_object):
            if i == 0:#nFeatures
                line_elements = line.split(':')
                nFeatures = int(line_elements[1])
            else:
                line_elements = line.split()
                features = [float(x) for x in line_elements]
                features = np.array(features)
                features_list.append(features.T)

                n_matches = features[0]
                r = features[1]
                g = features[2]
                b = features[3]
                src_x = features[4]
                src_y = features[5]
                m = 1
                while n_matches > 1:
                    # print(n_matches)
                    image_id = int(features[5+m])
                    image_id_x = features[6+m]
                    image_id_y = features[7+m]
                    m = m+3
                    n_matches = n_matches - 1
                    feature_pair = np.array([r, g, b, src_x, src_y, image_id_x, image_id_y]).reshape(1,-1)
                    # print(i-1, image_id - 1)
                    feature_matrix[n - 1, image_id - 1].append(feature_pair) 
    return feature_matrix     


def extractMatchingFeaturesFromFileNew(folder_name, total_images):
   
    feature_descriptor = []
    feature_x = []
    feature_y = []
    feature_flag = []


    for n in range(1, total_images):
        # print(n)
        matching_file_name = folder_name + "matching" + str(n) + ".txt"
        file_object = open(matching_file_name,"r")
        nFeatures = 0

        for i, line in enumerate(file_object):
            if i == 0:#nFeatures
                line_elements = line.split(':')
                nFeatures = int(line_elements[1])
            else:
                x_row = np.zeros((1, total_images))
                y_row = np.zeros((1, total_images))
                flag_row = np.zeros((1, total_images), dtype = int)

                line_elements = line.split()
                features = [float(x) for x in line_elements]
                features = np.array(features)
       
                n_matches = features[0]
                r = features[1]
                g = features[2]
                b = features[3]

                feature_descriptor.append([r,g,b])

                src_x = features[4]
                src_y = features[5]

                x_row[0, n-1] = src_x
                y_row[0, n-1] = src_y
                flag_row[0, n-1] = 1

                m = 1
                while n_matches > 1:
                    image_id = int(features[5+m])
                    image_id_x = features[6+m]
                    image_id_y = features[7+m]
                    m = m+3
                    n_matches = n_matches - 1

                    x_row[0, image_id - 1] = image_id_x
                    y_row[0, image_id - 1] = image_id_y
                    flag_row[0, image_id - 1] = 1

                feature_x.append(x_row)
                feature_y.append(y_row)
                feature_flag.append(flag_row)

    return np.array(feature_x).reshape(-1, total_images), np.array(feature_y).reshape(-1, total_images), np.array(feature_flag).reshape(-1, total_images), np.array(feature_descriptor).reshape(-1, 3)