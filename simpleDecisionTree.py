import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics 


class DecisionTree:
    def __init__(self):

        self.X = None
        self.y = None
        self.model = None

    def fit(self, X, y):
        self.model = DecisionTreeClassifier()

    def predict(self, X_pred):
        y_pred = self.model.predict(X_pred)
        return y_pred
    
    def get_score(self, X_test, y_test):

        y_pred = self.model.predict(X_test)
        score = metrics.accuracy_score(y_test, y_pred)

        return score 