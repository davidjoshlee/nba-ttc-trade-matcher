"""
Configuration constants for the NBA TTC Trade Matching Platform.
"""

CURRENT_SEASON = "2024-25"

# Minimum minutes per game to be considered a meaningful rotation player
MIN_MINUTES_THRESHOLD = 10.0

# Number of players each team makes available for trade
# (selected as the lowest-minutes rotation players on the roster)
# Adjustable in the UI — higher = more trade options but less realistic
PLAYERS_AVAILABLE_PER_TEAM = 7

# Ideal roster composition: minimum count of each role a balanced team should have
IDEAL_ROSTER_COMPOSITION = {
    "Floor Spacer": 3,
    "Rim Protector": 1,
    "Playmaker": 2,
    "Paint Scorer": 1,
    "Two-Way Wing": 2,
    "Stretch Big": 1,
    "Defensive Anchor": 1,
    "Volume Scorer": 1,
}

# NBA team abbreviations (all 30 teams)
NBA_TEAMS = [
    "ATL", "BOS", "BKN", "CHA", "CHI",
    "CLE", "DAL", "DEN", "DET", "GSW",
    "HOU", "IND", "LAC", "LAL", "MEM",
    "MIA", "MIL", "MIN", "NOP", "NYK",
    "OKC", "ORL", "PHI", "PHX", "POR",
    "SAC", "SAS", "TOR", "UTA", "WAS",
]
