from models.elasticnetregressor import ENet
from models.neuralnetworkregressor import NNMLPRegressor
from models.linearregressor import LinearRegressor
from models.randomforestregressor import RandForestRegressor
from objects.datahandler import DataHandler

if __name__ == "__main__":
    dh = DataHandler()
    dh.load_data()
    dh.set_X_y()
    
    X, y = dh.X, dh.y
    
    enet = ENet()
    mlpreg = NNMLPRegressor()
    linreg = LinearRegressor()
    randforestreg = RandForestRegressor()
    
    enet.fit(X, y)
    mlpreg.fit(X, y)
    linreg.fit(X, y)
    randforestreg.fit(X, y)
    
    enet.output_optimized_parameters()
    mlpreg.output_optimized_parameters()
    randforestreg.output_optimized_parameters()
    
    enet_score = enet.get_score()
    mlpreg_score = mlpreg.get_score()
    linreg_score = linreg.get_score()
    randforestreg_score = randforestreg.get_score()
    
    print(enet_score, mlpreg_score, linreg_score, randforestreg_score)
    