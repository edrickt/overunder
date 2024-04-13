import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn import metrics 


class DecisionTree:
    def __init__(self, ccp_alpha=0, max_depth=None):
        self.ccp_alpha = ccp_alpha
        self.max_depth = max_depth
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
    
    # ccp_alpha, max_depth
    def output_optimized_parameters(self):
        parameters = {
            'ccp_alpha':[0],
            'max_depth':[None]
        }
        model = DecisionTreeRegressor()
        grid=GridSearchCV(model, parameters, cv=10)
        grid.fit(self.X, self.y)

        file = open("models/decisiontree_optimize.txt", "w+")
        parameter_string = ",".join("{}={}".format(*i) for i in grid.best_params_.items())
        file.write(parameter_string)