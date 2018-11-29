from sklearn import svm
from sklearn.decomposition import PCA
from sklearn import metrics
import pandas as pd
import numpy as np

def trainAndPredict(X_train, y_train, X_test, y_test):
    # X_train = train_data.iloc[:, 1:]
    # y_train = train_data['label']
    # X_train = X_train.values / 255.0
    # X_test = test_data.values / 255.0
    # pca = PCA(n_components=n)
    # pca.fit(X_train)
    # X_train_pca = pca.transform(X_train)
    # X_test_pca = pca.transform(X_test)
    svmClassifier = svm.SVC()
    svmClassifier.fit(X_train, y_train)
    # y_score = svmClassifier.decision_function(X_test_pca)
    # fpr, tpr, thresholds = metrics.roc_curve(y, y_score, pos_label=2)
    labels_predict = svmClassifier.predict(X_test)
    return y_test, labels_predict
    # data_frame = pd.DataFrame(labels_predict)
    #
    # # Add one to all the index to match Kaggle submission requirement
    # data_frame.index = np.arange(1, len(data_frame) + 1)
    # data_frame.to_csv('predict_naive_bayes_SVM.csv', index=True, index_label='ImageId', header=['Label'])
