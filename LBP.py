import cv2
import numpy as np
from skimage import feature

def lbp_extract(train_data, point_num, radius):
    # label column
    labels = train_data['label']
    # image size / pixel number
    pixel_number = len(train_data.columns) - 1
    # sample number
    data_number = len(train_data.index)

    hist_matrix = np.empty((data_number, radius+2 ))
    for i in range(0, data_number):
        print(str(i))
        image_vecotr = train_data.iloc[i, 1:pixel_number+1]
        image = np.asarray(image_vecotr, dtype=np.uint8).reshape((28, 28))
        # cv2.namedWindow('image')
        # cv2.imshow('image', image)
        # cv2.waitKey(20)
        lbp_matrix = feature.local_binary_pattern(image, point_num, radius, 'uniform')
        (hist,_) = np.histogram(lbp_matrix)
        hist_matrix[i] = hist
    np.savetxt('lbp_feature.txt', hist_matrix)