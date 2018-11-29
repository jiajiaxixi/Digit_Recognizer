from sklearn import svm
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

def SVM_train_and_predict(X_train, y_train, X_test, y_test):
    svmClassifier = svm.SVC()
    svmClassifier.fit(X_train, y_train)
    labels_predict = svmClassifier.predict(X_test)
    return y_test, labels_predict

def XGB_train_and_predict(X_train, y_train, X_test, y_test):
    XGBoostClassifier = XGBClassifier(booster='gbtree',
                      learning_rate=0.5,
                      n_estimators=200,
                      max_depth=4,
                      seed=5)
    XGBoostClassifier.fit(X_train, y_train)
    labels_predict = XGBoostClassifier.predict(X_test)
    return y_test, labels_predict

def AdaBoost_train_and_predict(X_train, y_train, X_test, y_test):
    AdaBoostClf = AdaBoostClassifier()
    AdaBoostClf.fit(X_train, y_train)
    labels_predict = AdaBoostClf.predict(X_test)
    return y_test, labels_predict

def GradientBoosting_train_and_predict(X_train, y_train, X_test, y_test):
    GradientBoostingClf = GradientBoostingClassifier()
    GradientBoostingClf.fit(X_train, y_train)
    labels_predict = GradientBoostingClf.predict(X_test)
    return y_test, labels_predict

def GradientBoosting_train_and_predict(X_train, y_train, X_test, y_test):
    GradientBoostingClf = GradientBoostingClassifier()
    GradientBoostingClf.fit(X_train, y_train)
    labels_predict = GradientBoostingClf.predict(X_test)
    return y_test, labels_predict