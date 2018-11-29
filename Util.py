from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier

import LBP
import skeleton
import PCA

def preprocess(raw_data, feature_abstract_method):
    X_raw = raw_data.iloc[:, 1:]
    y_raw = raw_data['label']
    X_train, X_test, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.2)
    if (feature_abstract_method == 'LBP'):
        X_train = LBP.lbp_extract(X_train)
        X_test = LBP.lbp_extract(X_test)
    elif (feature_abstract_method == 'PCA'):
        return PCA()
    elif(feature_abstract_method == 'skeleton'):
        X_train = skeleton.skeleton_extract(X_train)
        X_test = skeleton.skeleton_extract(X_test)
    return X_train, X_test, y_train, y_test

def score(y_true, y_predict):
    print(classification_report(y_true, y_predict))
    print("Detailed confusion matrix:")
    print(confusion_matrix(y_true, y_predict))
    print("Accuracy Score: \n")
    print(accuracy_score(y_true, y_predict))

def ROC():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # Binarize the output
    y = label_binarize(y, classes=[0, 1, 2])
    n_classes = y.shape[1]

    # Add noisy features to make the problem harder
    random_state = np.random.RandomState(0)
    n_samples, n_features = X.shape
    X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

    # shuffle and split training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5,
                                                        random_state=0)

    # Learn to predict each class against the other
    classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True,
                                             random_state=random_state))
    y_score = classifier.fit(X_train, y_train).decision_function(X_test)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
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