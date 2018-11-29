import numpy as np


def grid_extract(data):
    rows = data.shape[0]
    cols = data.shape[1]
    data = data.values
    grid_matrix = np.empty((rows, 16))

    for i in range(0, rows):
        image_vector = data[i, 0:cols]
        image = image_vector.reshape(28, 28)
        feature = np.empty((1, 16))
        for n in range(0, 4):
            for m in range(0, 4):
                sub_area = image[n*4:n*4+7, m*4:m*4+7]
                count = 0
                for p in range(0, 7):
                    for q in range(0, 7):
                        if sub_area[p, q] > 0:
                            count += 1
                feature[0, n*4 + m] = count
        grid_matrix[i] = feature
    return grid_matrix
