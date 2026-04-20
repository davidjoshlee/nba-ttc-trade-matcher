"""
Fetch NBA player stats and roster data via nba_api.
Caches results locally to avoid repeated API calls.
"""

import os
import time
import pandas as pd
from nba_api.stats.endpoints import leaguedashplayerstats

from src.config import CURRENT_SEASON, MIN_MINUTES_THRESHOLD

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
CACHE_FILE = os.path.join(CACHE_DIR, f"player_stats_{CURRENT_SEASON}.csv")

# Columns we care about
STAT_COLUMNS = [
    "PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION",
    "GP", "MIN", "PTS", "REB", "AST", "STL", "BLK",
    "FGA", "FG_PCT", "FG3A", "FG3_PCT", "TOV",
]

RENAME_MAP = {
    "TEAM_ABBREVIATION": "team_abbr",
    "PLAYER_ID": "player_id",
    "PLAYER_NAME": "player_name",
    "TEAM_ID": "team_id",
    "GP": "gp",
    "MIN": "min",
    "PTS": "pts",
    "REB": "reb",
    "AST": "ast",
    "STL": "stl",
    "BLK": "blk",
    "FGA": "fga",
    "FG_PCT": "fg_pct",
    "FG3A": "fg3a",
    "FG3_PCT": "fg3_pct",
    "TOV": "tov",
}


def fetch_player_stats(force_refresh: bool = False) -> pd.DataFrame:
    """
    Fetch per-game player stats for the current season.
    Returns a DataFrame filtered to rotation-level players (>= MIN_MINUTES_THRESHOLD mpg).
    Uses cached data if available unless force_refresh is True.
    """
    if not force_refresh and os.path.exists(CACHE_FILE):
        df = pd.read_csv(CACHE_FILE)
        return df

    # Fetch from NBA.com via nba_api
    stats = leaguedashplayerstats.LeagueDashPlayerStats(
        season=CURRENT_SEASON,
        per_mode_detailed="PerGame",
    )
    time.sleep(1)  # Rate limiting

    df = stats.get_data_frames()[0]
    df = df[STAT_COLUMNS].rename(columns=RENAME_MAP)

    # Filter to players with meaningful minutes
    df = df[df["min"] >= MIN_MINUTES_THRESHOLD].copy()
    df = df.reset_index(drop=True)

    # Cache
    os.makedirs(CACHE_DIR, exist_ok=True)
    df.to_csv(CACHE_FILE, index=False)

    return df


def get_team_rosters(player_stats: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Group player stats by team. Returns {team_abbr: DataFrame of players}.
    """
    return {
        team: group.reset_index(drop=True)
        for team, group in player_stats.groupby("team_abbr")
    }
