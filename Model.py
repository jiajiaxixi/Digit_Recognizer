from sklearn import svm
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

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

def Bagging_train_and_predict(X_train, y_train, X_test, y_test):
    BaggingClf = BaggingClassifier()
    BaggingClf.fit(X_train, y_train)
    labels_predict = BaggingClf.predict(X_test)
    return y_test, labels_predict

def CNN_train_and_predict(X_train, y_train, X_test, y_test):
    MLPClf = MLPClassifier()
    MLPClf.fit(X_train, y_train)
    labels_predict = MLPClf.predict(X_test)
    return y_test, labels_predict

def DecisionTree_train_and_predict(X_train, y_train, X_test, y_test):
    DecisionTreeClf = DecisionTreeClassifier()
    DecisionTreeClf.fit(X_train, y_train)
    labels_predict = DecisionTreeClf.predict(X_test)
    return y_test, labels_predict

def LogisticRegression_train_and_predict(X_train, y_train, X_test, y_test):
    LogisticRegressionClf = LogisticRegression()
    LogisticRegressionClf.fit(X_train, y_train)
    labels_predict = LogisticRegressionClf.predict(X_test)
    return y_test, labels_predict

def KNN_train_and_predict(X_train, y_train, X_test, y_test):
    KNeighborsClf = KNeighborsClassifier()
    KNeighborsClf.fit(X_train, y_train)
    labels_predict = KNeighborsClf.predict(X_test)
    return y_test, labels_predict

def RandomForest_train_and_predict(X_train, y_train, X_test, y_test):
    RandomForestClf = RandomForestClassifier()
    RandomForestClf.fit(X_train, y_train)
    labels_predict = RandomForestClf.predict(X_test)
    return y_test, labels_predict