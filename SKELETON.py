from skimage import morphology
from skimage.filters import threshold_isodata
import numpy as np


def skeleton_extract(data):
    rows = data.shape[0]
    cols = data.shape[1]
    data = data.values
    skeleton_matrix = np.empty((rows, cols))
    for i in range(0, rows):
        image_vector = data[i, 0:cols]
        image = image_vector.reshape(28, 28)
        thresh = threshold_isodata(image)
        for n in range(0, 28):
            for m in range(0, 28):
                if image[m, n] > thresh:
                    image[m, n] = 1
                else:
                    image[m, n] = 0
        skeleton = morphology.skeletonize(image)
        skeleton_matrix[i] = skeleton.reshape(1, 28*28)
    return skeleton_matrix
