from models.elasticnetregressor import ENet
from models.neuralnetworkregressor import NNMLPRegressor
from models.linearregressor import LinearRegressor
from models.randomforestregressor import RandForestRegressor
from models.decisiontree import DecisionTree
from objects.datahandler import DataHandler
import pickle

if __name__ == "__main__":
    dh = DataHandler()
    dh.load_data()
    dh.set_X_y()
    
    X, y = dh.X, dh.y
    
    enet = ENet()
    mlpreg = NNMLPRegressor()
    linreg = LinearRegressor()
    randforestreg = RandForestRegressor()
    decisiontree = DecisionTree()
    
    enet.fit(X, y)
    linreg.fit(X, y)
    randforestreg.fit(X, y)
    mlpreg.fit(X, y)
    decisiontree.fit(X, y)
    
    # UNCOMMENT IF YOU WANT TO CREATE A NEW PICKLE FILE FOR REGRESSOR MODEL
    with open("modelspickle/team_overunder_regressor.pkl", "wb") as f:
        pickle.dump(enet, f)
    
    # enet.output_optimized_parameters()
    # randforestreg.output_optimized_parameters()
    # mlpreg.output_optimized_parameters()
    # decisiontree.output_optimized_parameters()
    
    enet_score = enet.get_score()
    linreg_score = linreg.get_score()
    randforestreg_score = randforestreg.get_score()
    mlpreg_score = mlpreg.get_score()
    decisiontree_score = decisiontree.get_score()
    
    print("MSE SCORES")
    print(f"ElasticNet: {enet_score}")
    print(f"Ordinary Least Squares: {linreg_score}")
    print(f"Random Forest: {randforestreg_score}")
    print(f"MLPRegressor: {mlpreg_score}")
    print(f"Decsion Tree: {decisiontree_score}")
    