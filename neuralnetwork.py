from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

class NNMLPRegressor:
    def __init__(self, hidden_layer_sizes=(50,50),
                 activation="relu",
                 solver="adam",
                 alpha=0.0001):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        
        self.X = None
        self.y = None
        self.model = None
    
    def fit(self, X, y):
        self.model = MLPRegressor(hidden_layer_sizes=self.hidden_layer_sizes,
                                  activation=self.activation,
                                  solver=self.solver,
                                  alpha=self.alpha,)
        self.model.fit(X, y)
        self.X = X
        self.y = y
        
    def predict(self, X_pred):
        y_pred = self.model.predict(X_pred)
        return y_pred
    
    def get_score(self):
        score = cross_val_score(self.model, self.X, self.y, scoring="neg_mean_squared_error")
        return abs(score.mean())
    
    def optimize(self):
        parameters = {
            'hidden_layer_sizes': [(50,), (50, 50), (100,), (100, 100), (100, 50), (50, 100)],
        } 
        model = MLPRegressor(max_iter=1000)
        grid = GridSearchCV(model, parameters)
        grid.fit(self.X, self.y)
        print(grid.best_params_)
