import pandas as pd
import NaiveBayes
import SVM
import XGBoost
# import LBP


# Read data from train set
train_data_frame = pd.read_csv('train.csv', header=0)
test_data_frame = pd.read_csv('test.csv', header=0)
# SVM.trainAndPredict(train_data_frame, test_data_frame, 0.75)
XGBoost.trainAndPredict(train_data_frame, test_data_frame, 0.75)
# train_data_frame, test_data_frame = PCA.reduce_dimension(train_data_frame, test_data_frame, 0.75)
# print(train_data_frame)
# counts = NaiveBayes.train(train_data_frame)
# NaiveBayes.predict(test_data_frame)

# LBP.lbp_extract(train_data_frame)


