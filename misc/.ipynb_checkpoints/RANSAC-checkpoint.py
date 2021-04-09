import numpy as np
import cv2

class RANSAC:
    def __init__(self, model):
        self.model = model
        
    def fit(self, data, s, thresh, n_iterations = 100):
        self.data=  data
        self.s = s
        self.N = len(data[0]) # data is a tuple of x,y values to be fit, so N= len(x)
        
        mask = np.zeros(self.N)
        i = 0
        n_inliers, max_inliers = 0, 0
        best_model = None
#         p_outlier = 0.5
#         acc = 0.99    
        while n_iterations > i:
            
            sample_data = self.RandomSample()
            model_out = self.model.fit(sample_data)

            # count the inliers within the threshold
            E = self.model.check(self.data, model_out)
            for idx in range(len(E)):
                if E[idx] < thresh:
                    n_inliers+=1
                    mask[idx] = 1
                    
            # check for the best model 
            if n_inliers > max_inliers:
                max_inliers = n_inliers
                best_model = model_out
                best_mask = mask 
            
            i += 1
            n_inliers = 0
#             if inlier_count >= 20:
#                 prob_outlier = 1 - inlier_count/data_size
                #print((1 - prob_outlier)**num_sample, prob_outlier, inlier_count)
                #if math.log(1 - (1 - prob_outlier)**num_sample) != 0:
                #    num_iterations = math.log(1 - desired_prob)/math.log(1 - (1 - prob_outlier)**num_sample)
            
        return best_model, best_mask

    def RandomSample(self):
        s = self.s
        N = self.N
        data = self.data
        random_idxs = np.random.choice(N, s,replace=True)
        data = np.array(data)
        sample_data = np.zeros((data.shape[0], s, data.shape[2]), dtype=np.float32)
        for i in range(data.shape[0]):
            sample_data[i] =  data[i][random_idxs]
        return sample_data