from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plot

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier

import LBP
import SKELETON
import PCA
import GRID
import HOG

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
        X_train = SKELETON.skeleton_extract(X_train)
        X_test = SKELETON.skeleton_extract(X_test)
    elif (feature_abstract_method == 'grid'):
        X_train = GRID.grid_extract(X_train)
        X_test = GRID.grid_extract(X_test)
    elif (feature_abstract_method == 'hog'):
        X_train = HOG.hog_extract(X_train)
        X_test = HOG.hog_extract(X_test)
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

    plot.figure()
    lw = 2
    plot.plot(fpr[2], tpr[2], color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
    plot.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plot.xlim([0.0, 1.0])
    plot.ylim([0.0, 1.05])
    plot.xlabel('False Positive Rate')
    plot.ylabel('True Positive Rate')
    plot.title('Receiver operating characteristic example')
    plot.legend(loc="lower right")
    plot.show()