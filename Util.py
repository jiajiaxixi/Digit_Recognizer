from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier

import LBP
import skeleton
import PCA
import grid

def preprocess(feature_abstract_method):
    # X_raw = raw_data.iloc[:, 1:]
    # y_raw = raw_data['label']
    # X_train, X_test, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.2)
    # X_train.to_csv('x_train.csv')
    # X_test.to_csv('x_test.csv')
    # y_train.to_csv('y_train.csv')
    # y_test.to_csv('y_test.csv')
    X_train = pd.read_csv('x_train.csv', index_col=0)
    X_test = pd.read_csv('x_test.csv', index_col=0)
    y_train = pd.read_csv('y_train.csv', index_col=0, header=None)
    y_test = pd.read_csv('y_test.csv', index_col=0, header=None)
    if (feature_abstract_method == 'LBP'):
        X_train = LBP.lbp_extract(X_train)
        X_test = LBP.lbp_extract(X_test)
    elif (feature_abstract_method == 'PCA'):
        X_train, X_test = PCA.PCA_extract(X_train, X_test)
    elif(feature_abstract_method == 'skeleton'):
        X_train = skeleton.skeleton_extract(X_train)
        X_test = skeleton.skeleton_extract(X_test)
    elif (feature_abstract_method == 'grid'):
        X_train = grid.grid_extract(X_train)
        X_test = grid.grid_extract(X_test)
    return X_train, X_test, y_train, y_test

def score(y_true, y_predict):
    print(classification_report(y_true, y_predict))
    print("Detailed confusion matrix:")
    print(confusion_matrix(y_true, y_predict))
    print("Accuracy Score: \n")
    print(accuracy_score(y_true, y_predict))

def ROC(y_true, y_predict, X_test, classifier):
    # Binarize the output
    y_true = label_binarize(y_true, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    y_predict = label_binarize(y_predict, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    n_classes = y_true.shape[1]

    # # Add noisy features to make the problem harder
    # random_state = np.random.RandomState(0)
    # n_samples, n_features = X.shape
    # X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

    # # shuffle and split training and test sets
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5,
    #                                                     random_state=0)

    # # Learn to predict each class against the other
    # classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True,
    #                                          random_state=random_state))

    # y_score = classifier.fit(X_train, y_train).decision_function(X_test)
    y_score = classifier.decision_function(X_test)
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plt.figure()
    lw = 2
    plt.plot(fpr[2], tpr[2], color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()