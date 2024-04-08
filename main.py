from datahandler import DataHandler
#from elasticnet import ENet
from linearregression import LinReg


if __name__ == "__main__":
    # Load data. Will create the game_logs.csv, game_logs_metrics.csv, and team_stats.csv for up the past num_years
    # Has the X and y dataframes can be accessed by dh.X and dh.y, for fitting the model. Will not create the a csv
    # file if it is already present.
    dh = DataHandler()
    dh.load_data(update_team=False)
    dh.set_X_y()
    
    while True:
        # Get the dataframe for two teams with the dataframe formatted for prediction
        X = dh.X
        y = dh.y
        
        team_away_name = input("Away team: ")
        team_home_name = input("Home team: ")
        try:
            overunder_line = float(input("Vegas line: "))
        except:
            print("Bad Input\n")
            continue
        
        X_pred = dh.get_X_pred(team_away_name=team_away_name, team_home_name=team_home_name)
        
        # dh.get_X_pred() will return false if bad input
        if (X_pred is False):
            print("Bad Input\n")
            continue
        
        model = LinReg()
        model.fit(X, y)
        
        y_pred = model.predict(X_pred)
        
        est = model.get_pvalues()        
    
        print(f"{dh.team_away.info.nickname[0].capitalize()} vs {dh.team_home.info.nickname[0].capitalize()}: {y_pred[0]:.2f} points")
        print(f"Percent dif: {(y_pred[0]/overunder_line-1)*100:.2f}%\n")
        print("P-Value Analysis: \n")
        print(est.summary())
        
from game import Game
from team import Team
from helperfunctions import get_seasons
import pandas as pd
from sklearn.linear_model import LinearRegression


if __name__ == "__main__":
    num_years = 5
    
    Team.team_stats_to_csv(num_years=num_years)
    game_logs = Game.get_game_logs(num_years=num_years)
    game_metrics = Game.get_team_metrics_for_games(game_logs)
    
    while True:
            cur_season = get_seasons()[0]
            
            team_away = Team.set_team(season=cur_season, name=input("Away Team: ").lower())
            team_home = Team.set_team(season=cur_season, name=input("Home Team: ").lower())

            columns_to_drop = ["TEAM_ID_A", "TEAM_ID_H", "GP_A", "GP_H", "W_A", "W_H", "L_A", "L_H", "SEASON_A", "SEASON_H", "TOTAL_POINTS"]

            X = game_metrics.drop(columns=columns_to_drop)
            y = game_metrics["TOTAL_POINTS"]

            model = LinearRegression().fit(X, y)

            X_pred = pd.concat([team_away.stats.add_suffix("_A"), team_home.stats.add_suffix("_H")], axis=1).drop(columns=columns_to_drop[:-1])
            y_pred = model.predict(X_pred)

            print(f"{team_away.info.nickname[0].capitalize()} vs {team_home.info.nickname[0].capitalize()}: {y_pred[0]:.2f} points\n")
            
            # p value analysis and look at simple decision tree, if around 3000 datapoints then can experiment neural network

