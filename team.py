from nba_api.stats.static.teams import find_team_by_abbreviation, find_team_name_by_id, find_teams_by_full_name, get_teams
from nba_api.stats.endpoints import teamestimatedmetrics
from helperfunctions import get_years
import time
import pandas as pd

class Team:
    def __init__(self, name=None, team_id=None):
        self.name = name
        self.team_id = team_id
        self.info = None
        self.stats = None

    def _get_team_info(self):
        if (self.name):
            if (len(self.name) == 3):
                info = pd.Dataframe([find_team_by_abbreviation(self.name)])
            else:
                info = pd.DataFrame.from_dict(find_teams_by_full_name(self.name))
        else:
            info = pd.DataFrame([find_team_name_by_id(self.team_id)])
        return info

    def _get_team_stats_by_info(self, num_years=3):
        years = get_years(num_years)

        team_id = self.info["id"][0]
        stats_log = []

        for year in years:
            stats = teamestimatedmetrics.TeamEstimatedMetrics(season=year).get_data_frames()[0]
            stats.insert(loc=7, column="SEASON", value=year)
            stats = stats[stats["TEAM_ID"] == team_id].reset_index(drop=True)
            stats_log.append(stats)

        stats_log = pd.concat(stats_log).reset_index(drop=True)

        return stats_log

    @staticmethod
    def set_team(name=None, team_id=None):
        try:
            all_teams = pd.read_csv("team_stats.csv", index_col=False)
        except:
            Team._team_stats_to_csv()
            all_teams = pd.read_csv("team_stats.csv", index_col=False)

        if (name):
            if (len(name) == 3):
                name = name.lower()
                team_df = all_teams.loc[all_teams["abbreviation"] == name]
            else:
                team_df = all_teams.loc[all_teams["nickname"] == name]
        else:
            team_df = all_teams.loc[all_teams["id"] == team_id]
        
        info = pd.DataFrame([team_df.iloc[0, 0:7]]).reset_index(drop=True)
        stats = pd.DataFrame([team_df.iloc[0, 8:-1]]).reset_index(drop=True)
        name = info.full_name[0]
        team_id = info.id[0]

        team = Team(name=name, team_id=team_id)
        team.info, team.stats = info, stats

        return team

    @staticmethod
    def _get_team(name=None, team_id=None):
        team = Team(name=name, team_id=team_id)

        team.info = team._get_team_info()
        time.sleep(0.6)
        team.stats = team._get_team_stats_by_info()
        time.sleep(0.6)

        team.team_id = team.info.id[0]
        team.name = team.info.full_name[0]

        return team

# NOTE: ISSUE MULTIPLE ROWS OF STATS FOR TEAM BUT ONE ROW FOR INPUT RIGHT NOW
    # FIXED ABOVE ISSUE, NOW MAKE SURE TO GET THE RIGHT TEAM GIVEN THE SEASON
        # Last thing added is print statement to get stats as dataframe instead of series

    @staticmethod
    def _team_stats_to_csv():
        teams = get_teams()
        team_stats = []

        for team in teams:
            cur_team = Team()._get_team(team_id=team["id"])
            team_stats.append(pd.concat([cur_team.info.reset_index(drop=True), cur_team.stats.reset_index(drop=True)], axis=1))
            print(team_stats)
            # team_stats.append(pd.concat([cur_team.info.reset_index(drop=True), cur_team.stats.reset_index(drop=True)], axis=1))
        print(team_stats)

        team_stats = pd.concat(team_stats).reset_index(drop=True).map(lambda s: s.lower() if type(s) == str else s)
        team_stats.to_csv("team_stats.csv", index=False)
