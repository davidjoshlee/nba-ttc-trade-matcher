"""
Role definitions for NBA players.

Each role is defined by a set of stat thresholds. A player's "fit score" for a
role is the average of how far above each threshold they are (in standard
deviations), giving us a continuous measure rather than a binary yes/no.

Design decision: We use rule-based classification rather than ML because:
1. It's transparent — we can explain exactly why a player got a role
2. It's fast to iterate on thresholds
3. 10-day project timeline doesn't justify training a model
The tradeoff is less nuance (e.g., can't capture playstyle from tracking data).
"""

# Each role: name -> dict of stat_name -> minimum threshold
# These thresholds represent "above average for that role" benchmarks
ROLE_DEFINITIONS = {
    "Floor Spacer": {
        "fg3a": 5.0,
        "fg3_pct": 0.36,
    },
    "Rim Protector": {
        "blk": 1.2,
        "reb": 7.0,
    },
    "Playmaker": {
        "ast": 5.5,
        "ast_to_tov": 1.6,  # computed field: ast / tov
    },
    "Paint Scorer": {
        "fg_pct": 0.54,
        "reb": 6.0,
        "pts": 12.0,
    },
    "Two-Way Wing": {
        "stl": 1.0,
        "pts": 14.0,
        "reb": 4.0,
    },
    "Stretch Big": {
        "reb": 7.0,
        "fg3a": 2.0,
        "fg3_pct": 0.32,
    },
    "Defensive Anchor": {
        "stl_plus_blk": 2.2,  # computed field: stl + blk
        "min": 27.0,
    },
    "Volume Scorer": {
        "pts": 22.0,
        "fga": 17.0,
    },
}

# Default role for players who don't strongly fit any archetype
DEFAULT_ROLE = "Rotation Player"
