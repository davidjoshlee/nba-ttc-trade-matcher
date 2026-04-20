"""Tests for role classification."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from src.roles.classifier import classify_player, classify_all_players


def test_floor_spacer():
    """A high-volume 3-point shooter should be classified as Floor Spacer."""
    player = pd.Series({
        "pts": 16.0, "reb": 3.0, "ast": 2.0, "stl": 0.5, "blk": 0.2,
        "fg3a": 8.0, "fg3_pct": 0.40, "fg_pct": 0.44, "fga": 12.0,
        "min": 30.0, "tov": 1.5,
    })
    result = classify_player(player)
    assert result["primary_role"] == "Floor Spacer", f"Expected Floor Spacer, got {result['primary_role']}"


def test_rim_protector():
    """A big man with blocks and rebounds should be Rim Protector."""
    player = pd.Series({
        "pts": 12.0, "reb": 11.0, "ast": 1.5, "stl": 0.5, "blk": 2.5,
        "fg3a": 0.0, "fg3_pct": 0.0, "fg_pct": 0.60, "fga": 8.0,
        "min": 30.0, "tov": 1.5,
    })
    result = classify_player(player)
    assert result["primary_role"] == "Rim Protector", f"Expected Rim Protector, got {result['primary_role']}"


def test_playmaker():
    """A high-assist player should be Playmaker."""
    player = pd.Series({
        "pts": 20.0, "reb": 3.0, "ast": 10.0, "stl": 1.0, "blk": 0.1,
        "fg3a": 7.0, "fg3_pct": 0.35, "fg_pct": 0.44, "fga": 18.0,
        "min": 35.0, "tov": 4.0,
    })
    result = classify_player(player)
    assert result["primary_role"] == "Playmaker", f"Expected Playmaker, got {result['primary_role']}"


def test_rotation_player():
    """A low-stats player should default to Rotation Player."""
    player = pd.Series({
        "pts": 5.0, "reb": 2.0, "ast": 1.0, "stl": 0.3, "blk": 0.1,
        "fg3a": 1.0, "fg3_pct": 0.30, "fg_pct": 0.42, "fga": 5.0,
        "min": 12.0, "tov": 0.5,
    })
    result = classify_player(player)
    assert result["primary_role"] == "Rotation Player", f"Expected Rotation Player, got {result['primary_role']}"


def test_classify_all_adds_columns():
    """classify_all_players should add role columns to the DataFrame."""
    df = pd.DataFrame([{
        "player_id": 1, "player_name": "Test", "team_id": 1, "team_abbr": "TST",
        "gp": 50, "pts": 20.0, "reb": 5.0, "ast": 5.0, "stl": 1.0, "blk": 0.5,
        "fg3a": 6.0, "fg3_pct": 0.38, "fg_pct": 0.46, "fga": 16.0,
        "min": 30.0, "tov": 2.0,
    }])
    result = classify_all_players(df)
    assert "primary_role" in result.columns
    assert "secondary_role" in result.columns
    assert "role_scores" in result.columns


if __name__ == "__main__":
    test_floor_spacer()
    test_rim_protector()
    test_playmaker()
    test_rotation_player()
    test_classify_all_adds_columns()
    print("All role tests passed!")
