from nba_api.stats.endpoints import leaguegamelog
from helperfunctions import get_years
from team import Team
import pandas as pd


class Game:
    @staticmethod
    def get_game_logs(num_games=164):
        try:
            game_log = pd.read_csv("game_log.csv", index_col=False)
            return game_log
        except:
            years = get_years(num_games//82+2)
            game_log = []

            num_games = num_games*2
            for year in years:
                games = leaguegamelog.LeagueGameLog(season=year, season_type_all_star="Regular Season").get_data_frames()[0].reset_index(drop=True)
                game_log.append(games)

            game_log = pd.concat(game_log).reset_index(drop=True).head(num_games)
            game_log.to_csv("game_log.csv", index=False)
            return game_log
    
    @staticmethod
    def get_team_metrics_for_games(game_log):
        try:
            game_metrics = pd.read_csv("game_logs_metrics.csv", index_col=False)
            return game_metrics
        except:
            game_metrics = []

            for game_id, game in game_log.groupby("GAME_ID"):
                if (len(game)) < 2: continue

                team_a = Team().set_team(team_id=game.iloc[0].TEAM_ID)
                team_b = Team().set_team(team_id=game.iloc[1].TEAM_ID)
                total_points = game.iloc[0].PTS + game.iloc[1].PTS
                
                combined_stats = pd.concat([team_a.stats.add_suffix("_A"), team_b.stats.add_suffix("_B")], axis=1)
                combined_stats["TOTAL_POINTS"] = total_points
                game_metrics.append(combined_stats)
                
            game_metrics = pd.concat(game_metrics, axis=0).reset_index(drop=True)
            game_metrics.to_csv("game_logs_metrics.csv", index=False)
            return game_metrics
