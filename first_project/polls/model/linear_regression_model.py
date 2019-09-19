import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import *
import pickle


class Regression:

    def data_load(self):
        # Import dataset
        dataset = pd.read_csv("polls/model/Dataset/Salary_Data.csv")
        x = dataset.iloc[:, :-1].values
        y = dataset.iloc[:, 1].values
        # dataset.head()

        # check for describe
        dataset.describe()

        # check for info
        dataset.info()

        # check for num of rows and cols
        print('The train data has {0} rows and {1} columns'.format(dataset.shape[0], dataset.shape[1]))

        # check for null

        dataset.isnull()
        return x, y, dataset

    def split(self, x, y):
        # split train and test data
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.35, random_state=0)
        return x_train, x_test, y_train, y_test

    def model(self, x_train, y_train):
        # fitting sample linear regresion to the training set

        regressor = LinearRegression()
        return regressor.fit(x_train, y_train)

    def prediction(self, regressor, x_test):
        # predicting the test set result and train set result
        prediction = regressor.predict(x_test)
        return prediction

    def accuracy(self, y_test, prediction):
        # Accuracy for train and test
        Accuracy = r2_score(y_test, prediction) * 100
        return Accuracy

    def save_model(self, regressor):
        # dump train model pickle file
        file = open('model.pkl', 'wb')
        pickle.dump(regressor, file)
        file.close()
        print("Pickle file create")


object_Regression = Regression()
x, y, dataset = object_Regression.data_load()
x_train, x_test, y_train, y_test = object_Regression.split(x, y)
regressor = object_Regression.model(x_train, y_train)
prediction = object_Regression.prediction(regressor,x_test)
object_Regression.accuracy(y_test, prediction)
object_Regression.save_model(regressor)
