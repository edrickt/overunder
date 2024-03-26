from game import Game
from team import Team
from helperfunctions import get_seasons
import pandas as pd
from sklearn.tree import DecisionTreeRegressor


class DataHandler:
    def __init__(self, num_years):
        self.num_years = num_years
        self.game_metrics = None
        self.X = None
        self.y = None
        self.team_away = None
        self.team_home = None
        self.cur_season = get_seasons()[0]
        self.columns_to_drop = ["TEAM_NAME_A", "TEAM_NAME_H", "TEAM_ID_A", "TEAM_ID_H", "GP_A", "GP_H", "W_A", "W_H", "L_A", "L_H", "SEASON_A", "SEASON_H", "TOTAL_POINTS"]
        
        self._load_data()
        self._set_X_y()
        
    def input_teams_get_X_pred(self):
        self.team_away = Team.set_team(season=self.cur_season, name=input("Away Team: ").lower())
        self.team_home = Team.set_team(season=self.cur_season, name=input("Home Team: ").lower())
        
        if (not self.team_away or not self.team_home):
            return False
        
        X_pred = pd.concat([self.team_away.stats.add_suffix("_A"), self.team_home.stats.add_suffix("_H")], axis=1).drop(columns=self.columns_to_drop[:-1])
        return X_pred
        
    def _load_data(self):
        Team.team_stats_to_csv(num_years=self.num_years)
        game_logs = Game.get_game_logs(num_years=self.num_years)
        self.game_metrics = Game.get_team_metrics_for_games(game_logs)
        
    def _set_X_y(self):
        self.X = self.game_metrics.drop(columns=self.columns_to_drop)
        self.y = self.game_metrics["TOTAL_POINTS"]
            