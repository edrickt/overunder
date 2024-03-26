from datahandler import DataHandler
from sklearn.linear_model import LinearRegression


if __name__ == "__main__":
    # Load data. Will create the game_logs.csv, game_logs_metrics.csv, and team_stats.csv for up the past num_years
    # Has the X and y dataframes can be accessed by dh.X and dh.y, for fitting the model. Will not create the a csv
    # file if it is already present.
    dh = DataHandler(num_years=5)
    
    while True:
        # Get the dataframe for two teams with the dataframe formatted for prediction
        X = dh.X
        y = dh.y
        X_pred = dh.input_teams_get_X_pred()
        
        # dh.input_teams_get_X_pred() will return false if bad input
        if (X_pred is False):
            print("Bad Input\n")
            continue

        model = LinearRegression().fit(X, y)
        y_pred = model.predict(X_pred)

        print(f"{dh.team_away.info.nickname[0].capitalize()} vs {dh.team_home.info.nickname[0].capitalize()}: {y_pred[0]:.2f} points\n")
            