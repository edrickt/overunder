from datahandler import DataHandler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR


if __name__ == "__main__":
    dh = DataHandler(num_years=5)
    
    while True:
            X_pred = dh.input_teams_get_X_pred()
            if X_pred is False:
                print("Bad Input\n")
                continue

            model = LinearRegression().fit(dh.X, dh.y)
            y_pred = model.predict(X_pred)

            print(f"{dh.team_away.info.nickname[0].capitalize()} vs {dh.team_home.info.nickname[0].capitalize()}: {y_pred[0]:.2f} points\n")
            