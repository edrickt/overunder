from game import Game
from team import Team
import pandas as pd
from sklearn.linear_model import LinearRegression


if __name__ == "__main__":
    while True:
        team_away = Team.set_team(input("Away Team: ").lower())
        team_home = Team.set_team(input("Home Team: ").lower())

        game_logs = Game.get_game_logs()
        game_metrics = Game.get_team_metrics_for_games(game_logs)

        columns_to_drop = ["TEAM_NAME_A", "TEAM_NAME_H", "TEAM_ID_A", "TEAM_ID_H", "GP_A", "GP_H", "W_A", "W_H", "L_A", "L_H", "TOTAL_POINTS"]

        X = game_metrics.drop(columns=columns_to_drop)
        y = game_metrics["TOTAL_POINTS"]

        model = LinearRegression().fit(X, y)

        X_pred = pd.concat([team_away.stats.add_suffix("_A"), team_home.stats.add_suffix("_H")], axis=1).drop(columns=columns_to_drop[:-1])
        y_pred = model.predict(X_pred)

        print(f"{team_away.info.nickname[0].capitalize()} vs {team_home.info.nickname[0].capitalize()}: {y_pred[0]:.2f} points\n")
