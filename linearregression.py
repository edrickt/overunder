from sklearn.linear_model import LinearRegression
from sklearn import metrics 
import statsmodels.api as sm


class LinReg:
    def __init__(self):
        self.X = None
        self.y = None
        self.model = LinearRegression()
        self.est2 = None

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.model.fit(X, y) 
        # Adding p-values
        X2 = sm.add_constant(X)
        est = sm.OLS(y, X2)
        self.est2 = est.fit()
        

    def predict(self, X_pred):
        y_pred = self.model.predict(X_pred)
        return y_pred

    def get_score(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        return metrics.accuracy_score(y_test, y_pred)
    def get_pvalues(self):
        if self.model is None:
            raise ValueError("Model not trained. Call fit() before get_pvalues()")
        return self.est2