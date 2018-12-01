import numpy as np
from skimage import feature


def lbp_extract(data, point_num=8, radius=2):
    rows = data.shape[0]
    cols = data.shape[1]
    data = data.values

    hist_matrix = np.empty((rows, 10))
    for i in range(0, rows):
        image_vector = data[i, 0:cols]
        image = image_vector.reshape(28, 28)
        lbp_matrix = feature.local_binary_pattern(image, point_num, radius, 'uniform')
        (hist, _) = np.histogram(lbp_matrix)
        hist_matrix[i] = hist
    return hist_matrix
