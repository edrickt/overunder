import numpy as np
from datahandler import DataHandler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV

class NNMLPRegressor:
    def __init__(self, hidden_layer_sizes=(50,50),
                 activation="relu",
                 solver="adam",
                 alpha=0.0001):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.model = None
    
    def fit(self, X, y):
        self.model = MLPRegressor(hidden_layer_sizes=self.hidden_layer_sizes,
                                  activation=self.activation,
                                  solver=self.solver,
                                  alpha=self.alpha,)
        self.model.fit(X, y)
        
    def predict(self, X_pred):
        y_pred = self.model.predict(X_pred)
        return y_pred
    
    def optimize(self, X, y):
        parameters = {
            'hidden_layer_sizes': [(50,), (50, 50), (100,), (100, 100), (100, 50), (50, 100)],
        } 
        model = MLPRegressor(max_iter=1000)
        grid = GridSearchCV(model, parameters)
        grid.fit(X, y)
        print(grid.best_params_)
