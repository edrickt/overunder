from models.elasticnetregressor import ENet
from objects.datahandler import DataHandler
from sklearn.linear_model import LogisticRegression
from misc.helperfunctions import get_seasons
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, f1_score
from objects.team import Team
import pandas as pd
import pickle
import numpy as np


def build_csv():
    df = pd.read_csv("csvs/predicted_points_empty.csv", index_col=False)

    dh = DataHandler()
    dh.load_data(update_team=False)
    dh.set_X_y()

    X = dh.X
    y = dh.y

    model = ENet()
    model.fit(X, y)

    for i in range(len(df)):
        cur_game = df.iloc[[i]]
        cur_season = get_seasons()[0]
        
        team_away_name = cur_game["TEAM_AWAY"].values[0]
        team_home_name = cur_game["TEAM_HOME"].values[0]
        
        team_away = Team().set_team(season=cur_season, name=team_away_name)
        team_home = Team().set_team(season=cur_season, name=team_home_name)
        
        combined_stats = pd.concat([team_away.stats.add_suffix("_A"), team_home.stats.add_suffix("_H")], axis=1)
        
        X_pred = dh.get_X_pred(team_away_name, team_home_name)
        y_pred = model.predict(X_pred)
        
        df.loc[i, "PREDICTED"] = y_pred
        df.loc[i, combined_stats.columns] = combined_stats.iloc[0].values
        
    df = df.drop(["TEAM_AWAY", "TEAM_HOME", "TEAM_NAME_A", "TEAM_ID_A", "GP_A", "W_A", "L_A", "SEASON_A",
             "TEAM_NAME_H", "TEAM_ID_H", "GP_H", "W_H", "L_H", "SEASON_H", "TOTAL_POINTS"], axis=1)
            
    # df = df.drop(["TEAM_AWAY", "TEAM_HOME", "TOTAL_POINTS"], axis=1)
        
    df.to_csv("csvs/predicted_points.csv", index=False)


if __name__ == "__main__":
    build_csv()
    df = pd.read_csv("csvs/predicted_points.csv", index_col=False)
    
    dh = DataHandler()
    dh.load_data()
    
    # X = df.drop(["OVER"], axis=1)
    X = df[["VEGAS_LINE", "PREDICTED"]]
    y = df["OVER"]
    
    model = LogisticRegression(C=0.5, penalty="l1", solver="saga").fit(X, y)
    
    # FOR OPTIMIZING CLASSIFICATION MODEL
    # OPTIMIZED PARAMETERS: {'C': 0.5, 'penalty': 'l1', 'solver': 'saga'}
    # parameters = {
    #     "penalty" : ["l1", "l2", "elasticnet", "none"],
    #     "C" : [.5, 1, 1.5, 2, 2.5, 3],
    #     "solver" : ["lbfgs","newton-cg","liblinear","sag","saga"],
    # }

    # grid = GridSearchCV(model, parameters)
    # grid.fit(X, y)
    # print(grid.best_params_)
    
    # model = LogisticRegression(max_iter=100000, **grid.best_params_)
    # OPTIMIZED PARAMETERS: {'C': 0.5, 'penalty': 'l1', 'solver': 'saga'}
    
    accuracy = cross_val_score(model, X, y, scoring="accuracy")
    precision = cross_val_score(model, X, y, scoring="precision")
    f1 = cross_val_score(model, X, y, scoring="f1")
    
    print("Classification Cross Validation Results:")
    print(f"Accuracy: {accuracy.mean()*100:.2f}%")
    print(f"Precision: {precision.mean()*100:.2f}%")
    print(f"F1 Score: {f1.mean()*100:.2f}%")
    
    # UNCOMMENT IF YOU WANT TO CREATE NEW PICKLE FILE FOR CLASSIFIER MODEL
    # with open("modelspickle/team_overunder_classifier.pkl", "wb") as f:
    #     pickle.dump(model, f)
