from datahandler import DataHandler
from models.elasticnet import ENet


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
        
        model = ENet()
        model.fit(X, y)
        
        y_pred = model.predict(X_pred)        
    
        print(f"{dh.team_away.info.nickname[0].capitalize()} vs {dh.team_home.info.nickname[0].capitalize()}: {y_pred[0]:.2f} points")
        print(f"Percent dif: {(y_pred[0]/overunder_line-1)*100:.2f}%\n")
