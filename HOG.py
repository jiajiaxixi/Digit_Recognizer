from skimage.feature import hog
import numpy as np

def hog_extract(data, point_num=24, radius=3):
    rows = data.shape[0]
    cols = data.shape[1]
    data = data.values

    hist_matrix = np.empty((rows, 200))
    for i in range(0, rows):
        image_vector = data[i, 0:cols]
        image = image_vector.reshape(28, 28)
        hog_vector = hog(image, 8, (5, 5), (1,1))
        hist_matrix[i]=hog_vector
    return hist_matrix