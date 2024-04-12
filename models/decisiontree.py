import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics 


class DecisionTree:
    def __init__(self):
        self.X = None
        self.y = None
        self.model = DecisionTreeRegressor()

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.model.fit(X, y) 

    def predict(self, X_pred):
        y_pred = self.model.predict(X_pred)
        return y_pred
    
    def get_score(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        return metrics.accuracy_score(y_test, y_pred)
    