import data_handling
import pandas as pd
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
import numpy as np
import evaluation

"""
Model class to predict the total amount in a receipt given annotations of the receipt.
"""


class Model:

    def __init__(self, train_path="data/train/", test_path=input("enter test path")):
        """
        Creates a model class, which collects data from the given paths. The model can be trained
        on the data, using a logistic regression.
        :param train_path: path of the training data
        :param test_path: path of the testing data
        """
        self.train_df = data_handling.load_data(train_path)
        self.test_df = data_handling.load_data(test_path)
        self.logistic_regression = LogisticRegression(solver='liblinear')
        self.x_train = self.train_df.drop(['name', 'id', 'path', 'label'], axis='columns')
        self.y_train = self.train_df['label']
        self.train_path = "data/train/"
        self.test_path = test_path
        self.model = None
        self.predictions = None

    def fit(self):
        """
        Fits a logistic regression on the data in the model. Extra samples are generated if the
        classes in the data are not distributed equally.
        """

        os = SMOTE(random_state=0)
        os_data_x, os_data_y = os.fit_resample(self.x_train, self.y_train)
        os_data_x = pd.DataFrame(data=os_data_x, columns=self.x_train.columns)
        os_data_y = pd.DataFrame(data=os_data_y, columns=['label'])
        self.model = self.logistic_regression.fit(os_data_x, np.ravel(os_data_y))

    def predict(self):
        """
        Predicts the total amounts of the receipts in the test data.
        """
        self.predictions = evaluation.predict(self.test_df, self.model, self.test_path)

    def print_predictions(self):
        """
        Prints the predictions made by the model on the testing data.
        """

        for i in range(len(self.predictions[1])):
            print(str(self.predictions[0][i]) + "," + str(self.predictions[1][i]))
