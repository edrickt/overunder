from objects.datahandler import DataHandler
import pickle
import pandas as pd


if __name__ == "__main__":
    # Load data. Will create the game_logs.csv, game_logs_metrics.csv, and team_stats.csv for up the past num_years
    # Has the X and y dataframes can be accessed by dh.X and dh.y, for fitting the model. Will not create the a csv
    # file if it is already present.
    dh = DataHandler()
    dh.load_data(update_team=False)
    dh.set_X_y()
    
    with open("modelspickle/team_overunder_regressor.pkl", "rb") as f:
        regressor = pickle.load(f)
    with open("modelspickle/team_overunder_classifier.pkl", "rb") as f:
        classifier = pickle.load(f)
    
    while True:        
        team_away_name = input("      Away team: ")
        team_home_name = input("      Home team: ")
        try:
            overunder_line = float(input("     Vegas line: "))
        except:
            print("Bad Input\n")
            continue
        
        X_pred = dh.get_X_pred(team_away_name=team_away_name, team_home_name=team_home_name)
        
        # dh.get_X_pred() will return false if bad input
        if (X_pred is False):
            print("Bad Input\n")
            continue
        
        y_pred_regressor = regressor.predict(X_pred)
        
        X_pred_classifier = pd.DataFrame({"VEGAS_LINE": [overunder_line], "PREDICTED": [y_pred_regressor]})
        y_pred_probability = classifier.predict_proba(X_pred_classifier)[0]
        
        print(f"{dh.team_away.info.nickname[0].capitalize()} vs {dh.team_home.info.nickname[0].capitalize()}: {y_pred_regressor[0]:.2f} points")
        print(f"    Percent dif: {(y_pred_regressor[0]/overunder_line-1)*100:.2f}%")
        print(f"           Over: {y_pred_probability[0]*100:.2f}%\n          Under: {y_pred_probability[1]*100:.2f}%\n")
