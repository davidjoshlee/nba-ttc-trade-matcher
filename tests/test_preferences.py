"""Tests for preference generation."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from src.preferences.gap_analysis import analyze_team_gaps
from src.preferences.ranker import identify_available_players, generate_preferences


def _make_test_roster():
    """Create a simple test roster with known role composition."""
    players = []
    # Team AAA: 3 Floor Spacers, 0 Rim Protectors
    for i in range(3):
        players.append({
            "player_id": i + 1, "player_name": f"Shooter {i+1}",
            "team_id": 100, "team_abbr": "AAA",
            "gp": 50, "pts": 12, "reb": 2, "ast": 2, "stl": 0.5, "blk": 0.1,
            "fg3a": 6, "fg3_pct": 0.38, "fg_pct": 0.44, "fga": 10,
            "min": 25 - i * 3, "tov": 1,
            "primary_role": "Floor Spacer", "secondary_role": None, "role_scores": {},
        })
    # Team BBB: 2 Rim Protectors, 0 Floor Spacers
    for i in range(2):
        players.append({
            "player_id": 10 + i, "player_name": f"Blocker {i+1}",
            "team_id": 200, "team_abbr": "BBB",
            "gp": 50, "pts": 8, "reb": 10, "ast": 1, "stl": 0.3, "blk": 2.0,
            "fg3a": 0, "fg3_pct": 0, "fg_pct": 0.55, "fga": 6,
            "min": 28 - i * 5, "tov": 1,
            "primary_role": "Rim Protector", "secondary_role": None, "role_scores": {},
        })
    return pd.DataFrame(players)


def test_gap_analysis_finds_missing_roles():
    """A team with no Rim Protectors should have high urgency for that role."""
    df = _make_test_roster()
    gaps = analyze_team_gaps(df)

    # AAA has 0 Rim Protectors — should have urgency 1.0
    aaa_gaps = dict(gaps["AAA"])
    assert aaa_gaps["Rim Protector"] == 1.0, f"Expected 1.0, got {aaa_gaps['Rim Protector']}"

    # BBB has 0 Floor Spacers — should have high urgency
    bbb_gaps = dict(gaps["BBB"])
    assert bbb_gaps["Floor Spacer"] > 0, f"Expected positive urgency for Floor Spacer"


def test_gap_analysis_identifies_surplus():
    """A team with surplus Floor Spacers should have negative urgency."""
    df = _make_test_roster()
    gaps = analyze_team_gaps(df)

    # AAA has 3 Floor Spacers, ideal is 3 — urgency should be 0
    aaa_gaps = dict(gaps["AAA"])
    assert aaa_gaps["Floor Spacer"] <= 0, f"Expected <= 0, got {aaa_gaps['Floor Spacer']}"


def test_preferences_rank_needed_roles_higher():
    """Team AAA (needs Rim Protector) should prefer BBB's blockers."""
    df = _make_test_roster()
    gaps = analyze_team_gaps(df)

    # Make all players available
    df_with_avail = df.copy()
    df_with_avail["available_for_trade"] = True

    prefs = generate_preferences(df, gaps)

    # AAA's top preference should be from BBB (Rim Protectors)
    if prefs.get("AAA"):
        top_pid = prefs["AAA"][0][0]
        top_player = df[df.player_id == top_pid].iloc[0]
        assert top_player["team_abbr"] == "BBB", \
            f"Expected AAA to prefer BBB player, got {top_player['team_abbr']}"
        assert top_player["primary_role"] == "Rim Protector", \
            f"Expected Rim Protector, got {top_player['primary_role']}"


if __name__ == "__main__":
    test_gap_analysis_finds_missing_roles()
    test_gap_analysis_identifies_surplus()
    test_preferences_rank_needed_roles_higher()
    print("All preference tests passed!")
