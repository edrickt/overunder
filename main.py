from game import Game
from team import Team
import pandas as pd
from sklearn.linear_model import LinearRegression


if __name__ == "__main__":
    while True:
        team_a_input = input("team a: ")
        team_b_input = input("team b: ")

        try:
            team_a = Team.set_team(team_a_input)
            team_b = Team.set_team(team_b_input)

            game_logs = Game.get_game_logs()
            game_metrics = Game.get_team_metrics_for_games(game_logs)

            columns_to_drop = ["TEAM_NAME_A", "TEAM_NAME_B", "TEAM_ID_A", "TEAM_ID_B", "GP_A", "GP_B", "W_A", "W_B", "L_A", "L_B", "TOTAL_POINTS"]

            X = game_metrics.drop(columns=columns_to_drop)
            y = game_metrics["TOTAL_POINTS"]

            model = LinearRegression().fit(X, y)

            X_pred = pd.concat([team_a.stats.add_suffix("_A"), team_b.stats.add_suffix("_B")], axis=1).drop(columns=columns_to_drop[:-1])
            y_pred = model.predict(X_pred)

            print(f"{team_a_input} vs {team_b_input}: {y_pred[0]:2f} points\n")
        except:
            print("bad input")
            continue