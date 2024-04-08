from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
import numpy as np

class NNMLPRegressor:
    def __init__(self):     
        self.X = None
        self.y = None
        self.model = None
    
    def fit(self, X, y):
        self.model = RandomForestRegressor()
        self.model.fit(X, y)
        self.X = X
        self.y = y
        
    def predict(self, X_pred):
        y_pred = self.model.predict(X_pred)
        return y_pred
    
    def get_score(self):
        score = cross_val_score(self.model, self.X, self.y, scoring="neg_mean_squared_error")
        return abs(score.mean())
    
    def output_optimized_parameters(self):
        parameters = {

        } 
        model = RandomForestRegressor()
        grid = GridSearchCV(model, parameters, cv=10)
        grid.fit(self.X, self.y)

        file = open("randomforest_optimize.txt", "w+")
        parameter_string = ",".join("{}={}".format(*i) for i in grid.best_params_.items())
        file.write(parameter_string)
