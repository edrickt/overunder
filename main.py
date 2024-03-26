from datahandler import DataHandler
from sklearn.tree import DecisionTreeRegressor


if __name__ == "__main__":
    dh = DataHandler()
    
    while True:
            X_pred = dh.input_teams_get_X_pred()

            model = DecisionTreeRegressor(max_depth=3).fit(dh.X, dh.y)
            y_pred = model.predict(X_pred)

            print(f"{dh.team_away.info.nickname[0].capitalize()} vs {dh.team_home.info.nickname[0].capitalize()}: {y_pred[0]:.2f} points\n")
            