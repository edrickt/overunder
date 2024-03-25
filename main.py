from game import Game
from team import Team
from helperfunctions import get_seasons
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV


if __name__ == "__main__":
    num_years = 5
    
    Team.team_stats_to_csv(num_years=num_years)
    game_logs = Game.get_game_logs(num_years=num_years)
    game_metrics = Game.get_team_metrics_for_games(game_logs)
    
    while True:
            cur_season = get_seasons()[0]
            
            team_away = Team.set_team(season=cur_season, name=input("Away Team: ").lower())
            team_home = Team.set_team(season=cur_season, name=input("Home Team: ").lower())

            columns_to_drop = ["TEAM_NAME_A", "TEAM_NAME_H", "TEAM_ID_A", "TEAM_ID_H", "GP_A", "GP_H", "W_A", "W_H", "L_A", "L_H", "SEASON_A", "SEASON_H", "TOTAL_POINTS"]

            X = game_metrics.drop(columns=columns_to_drop)
            y = game_metrics["TOTAL_POINTS"]

            model = DecisionTreeRegressor(max_depth=3).fit(X, y)

            X_pred = pd.concat([team_away.stats.add_suffix("_A"), team_home.stats.add_suffix("_H")], axis=1).drop(columns=columns_to_drop[:-1])
            y_pred = model.predict(X_pred)

            print(f"{team_away.info.nickname[0].capitalize()} vs {team_home.info.nickname[0].capitalize()}: {y_pred[0]:.2f} points\n")
            