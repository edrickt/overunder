from sklearn.linear_model import LinearRegression
from sklearn import metrics 


class LinReg:
    def __init__(self):
        self.X = None
        self.y = None
        self.model = LinearRegression()

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
