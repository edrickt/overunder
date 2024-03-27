from simpleDecisionTree import DecisionTree
from datahandler import DataHandler


if __name__ == "__main__":

    dh = DataHandler()
    dh.load_data()
    dh.set_X_y()


    while True:

        # Get the dataframe for two teams with the dataframe formatted for prediction
        X = dh.X
        y = dh.y

        team_away_name = input("Away team: ")
        team_home_name = input("Home team: ")

        X_pred = dh.get_X_pred(team_away_name=team_away_name, team_home_name=team_home_name)

        # dh.get_X_pred() will return false if bad input

        if (X_pred is False):
            print("Bad Input\n")
            continue
        
        decisionTree = DecisionTree()
        decisionTree.fit(dh.X, dh.y)

        decisionTree_score = decisionTree.get_score()

        y_pred = decisionTree.predict(X_pred)

        print(f"{dh.team_away.info.nickname[0].capitalize()} vs {dh.team_home.info.nickname[0].capitalize()}: {y_pred[0]:.2f} points")
        print(f"Simple Decision Tree MSE: {decisionTree_score:.2f}\n")        
        
