from sklearn.decomposition import PCA
import numpy as np


def PCA_extract(train_data, test_data, n=0.75):
    train_data = train_data.values/255.0
    test_data = test_data.values/255.0
    pca = PCA(n_components=n)
    pca.fit(train_data)
    X_train_pca = pca.transform(train_data)
    X_test_pca = pca.transform(test_data)

    return X_train_pca, X_test_pca