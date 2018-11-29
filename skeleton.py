from skimage import morphology
import numpy as np

def skeletion_extract(data):
    rows = data.shape[0]
    cols = data.shape[1]
    skeletion_matrix = np.empty((rows, cols))
    for i in range(0, rows):
        image_vector = data[i, 0:cols]
        image = image_vector.reshape(28, 28)
        skeleton = morphology.skeletonize(image)
        skeletion_matrix[i] = skeleton.reshape(1, 28*28)
    return skeletion_matrix
