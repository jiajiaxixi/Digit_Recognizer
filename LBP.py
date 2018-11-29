import numpy as np
from skimage import feature


def lbp_extract(data, point_num=24, radius=8):
    rows = data.shape[0]
    cols = data.shape[1]
    data = data.values

    hist_matrix = np.empty((rows, radius+2 ))
    for i in range(0, rows):
        print(str(i))
        image_vector = data[i, 0:cols]
        image = image_vector.reshape(28, 28)
        # cv2.namedWindow('image')
        # cv2.imshow('image', image)
        # cv2.waitKey(20)
        lbp_matrix = feature.local_binary_pattern(image, point_num, radius, 'uniform')
        (hist, _) = np.histogram(lbp_matrix)
        hist_matrix[i] = hist
    return hist_matrix
