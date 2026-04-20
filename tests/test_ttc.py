"""Tests for TTC algorithm."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from src.matching.ttc import run_ttc, _find_all_cycles


def test_find_cycles_simple():
    """A -> B -> A should produce one 2-node cycle."""
    graph = {"A": "B", "B": "A"}
    cycles = _find_all_cycles(graph)
    assert len(cycles) == 1
    assert set(cycles[0]) == {"A", "B"}


def test_find_cycles_three():
    """A -> B -> C -> A should produce one 3-node cycle."""
    graph = {"A": "B", "B": "C", "C": "A"}
    cycles = _find_all_cycles(graph)
    assert len(cycles) == 1
    assert set(cycles[0]) == {"A", "B", "C"}


def test_find_cycles_multiple():
    """Two separate 2-cycles."""
    graph = {"A": "B", "B": "A", "C": "D", "D": "C"}
    cycles = _find_all_cycles(graph)
    assert len(cycles) == 2


def test_ttc_simple_3team():
    """
    3 teams, each wants what another has:
    Team A has a Playmaker, wants a Rim Protector (from B)
    Team B has a Rim Protector, wants a Floor Spacer (from C)
    Team C has a Floor Spacer, wants a Playmaker (from A)
    Should produce one 3-team cycle.
    """
    players = pd.DataFrame([
        {"player_id": 1, "player_name": "Player A", "team_id": 100, "team_abbr": "AAA",
         "gp": 50, "pts": 15, "reb": 3, "ast": 8, "stl": 1, "blk": 0.2,
         "fg3a": 2, "fg3_pct": 0.3, "fg_pct": 0.45, "fga": 12, "min": 30, "tov": 3,
         "primary_role": "Playmaker", "secondary_role": None, "role_scores": {}},
        {"player_id": 2, "player_name": "Player B", "team_id": 200, "team_abbr": "BBB",
         "gp": 50, "pts": 10, "reb": 10, "ast": 1, "stl": 0.5, "blk": 2.5,
         "fg3a": 0, "fg3_pct": 0, "fg_pct": 0.55, "fga": 8, "min": 28, "tov": 1,
         "primary_role": "Rim Protector", "secondary_role": None, "role_scores": {}},
        {"player_id": 3, "player_name": "Player C", "team_id": 300, "team_abbr": "CCC",
         "gp": 50, "pts": 12, "reb": 3, "ast": 2, "stl": 0.8, "blk": 0.1,
         "fg3a": 7, "fg3_pct": 0.4, "fg_pct": 0.44, "fga": 10, "min": 25, "tov": 1,
         "primary_role": "Floor Spacer", "secondary_role": None, "role_scores": {}},
    ])

    available = players.copy()
    available["available_for_trade"] = True

    # Preferences: A wants B's player, B wants C's player, C wants A's player
    preferences = {
        "AAA": [(2, 10.0), (3, 5.0)],  # A wants Rim Protector (B) most
        "BBB": [(3, 10.0), (1, 5.0)],  # B wants Floor Spacer (C) most
        "CCC": [(1, 10.0), (2, 5.0)],  # C wants Playmaker (A) most
    }

    gaps = {
        "AAA": [("Rim Protector", 1.0), ("Floor Spacer", 0.5)],
        "BBB": [("Floor Spacer", 1.0), ("Playmaker", 0.5)],
        "CCC": [("Playmaker", 1.0), ("Rim Protector", 0.5)],
    }

    cycles = run_ttc(players, preferences, available, gaps)

    assert len(cycles) == 1, f"Expected 1 cycle, got {len(cycles)}"
    assert cycles[0]["num_teams"] == 3, f"Expected 3-team cycle, got {cycles[0]['num_teams']}"

    # Every team should receive the player they most wanted
    for trade in cycles[0]["trades"]:
        team = trade["team"]
        received_pid = trade["receives"]["player_id"]
        # Check that the received player was their top preference
        assert received_pid == preferences[team][0][0], \
            f"{team} received player {received_pid}, expected {preferences[team][0][0]}"


def test_ttc_no_trades_when_no_preferences():
    """If no team has preferences, should return no cycles."""
    players = pd.DataFrame([
        {"player_id": 1, "player_name": "P1", "team_id": 100, "team_abbr": "AAA",
         "gp": 50, "pts": 5, "reb": 2, "ast": 1, "stl": 0.3, "blk": 0.1,
         "fg3a": 1, "fg3_pct": 0.3, "fg_pct": 0.42, "fga": 5, "min": 12, "tov": 0.5,
         "primary_role": "Rotation Player", "secondary_role": None, "role_scores": {}},
    ])
    available = players.copy()
    available["available_for_trade"] = True

    cycles = run_ttc(players, {}, available, {})
    assert len(cycles) == 0


if __name__ == "__main__":
    test_find_cycles_simple()
    test_find_cycles_three()
    test_find_cycles_multiple()
    test_ttc_simple_3team()
    test_ttc_no_trades_when_no_preferences()
    print("All TTC tests passed!")
