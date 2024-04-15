from models.elasticnetregressor import ENet
from objects.datahandler import DataHandler
import pandas as pd


if __name__ == "__main__":
    df = pd.read_csv("csvs/predicted_points.csv")

    dh = DataHandler()
    dh.load_data(update_team=False)
    dh.set_X_y()

    X = dh.X
    y = dh.y

    model = ENet()
    model.fit(X, y)

    for i in range(len(df)):
        cur_game = df.iloc[[i]]
        team_away = cur_game["TEAM_AWAY"].values[0]
        team_home = cur_game["TEAM_HOME"].values[0]
        
        X_pred = dh.get_X_pred(team_away, team_home)
        y_pred = model.predict(X_pred)
        
        df.loc[i, "PREDICTED"] = y_pred
    
    df.to_csv("csvs/predicted_points.csv")