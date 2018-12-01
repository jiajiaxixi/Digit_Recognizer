import pandas as pd
import Util
import Model


# Read data from train set
train_data_frame = pd.read_csv('train.csv', header=0)
# Data abstraction(hog, pca, grid)
X_train, X_test, y_train, y_test = Util.preprocess("hog")
# Train the data base on different models
y_test, labels_predict, classifier = Model.SVM_train_and_predict(X_train, y_train, X_test, y_test)
# Get the score and roc curve of the result
Util.score(y_test, labels_predict)
Util.ROC(y_test, labels_predict, X_test, classifier)



