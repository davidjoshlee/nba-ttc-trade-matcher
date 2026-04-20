"""
Generate preference rankings for each team over available players from other teams.
"""

import pandas as pd
from src.config import PLAYERS_AVAILABLE_PER_TEAM


def identify_available_players(classified_players: pd.DataFrame) -> pd.DataFrame:
    """
    For each team, mark the bottom N players by minutes as available for trade.
    Returns the full DataFrame with an added 'available_for_trade' boolean column.
    """
    result = classified_players.copy()
    result["available_for_trade"] = False

    for team_abbr, roster in result.groupby("team_abbr"):
        # Sort by minutes ascending — lowest-minute players are most tradeable
        team_idx = roster.sort_values("min").index
        # Mark bottom N as available
        available_idx = team_idx[:PLAYERS_AVAILABLE_PER_TEAM]
        result.loc[available_idx, "available_for_trade"] = True

    return result


def _player_quality_score(row: pd.Series) -> float:
    """
    Simple composite quality score based on per-game stats.
    Weighted sum of key stats to rank players within a role.
    """
    return (
        row["pts"] * 1.0
        + row["reb"] * 0.8
        + row["ast"] * 1.0
        + row["stl"] * 2.0
        + row["blk"] * 2.0
        - row["tov"] * 1.5
    )


def generate_preferences(
    classified_players: pd.DataFrame,
    team_gaps: dict[str, list[tuple[str, float]]],
) -> dict[str, list[tuple[int, float]]]:
    """
    For each team, rank all available players from other teams based on
    how well they fill the team's needs.

    Returns {team_abbr: [(player_id, preference_score), ...]} sorted by preference descending.

    Preference score = urgency_of_needed_role * player_quality_score
    This means a player who fills a critical gap AND is high quality ranks highest.
    """
    players_with_availability = identify_available_players(classified_players)
    available = players_with_availability[players_with_availability["available_for_trade"]]

    preferences = {}

    for team_abbr, gaps in team_gaps.items():
        # Build urgency lookup for this team's needs
        urgency_by_role = {role: max(urgency, 0) for role, urgency in gaps}

        scores = []
        for _, player in available.iterrows():
            # Skip players on this team (can't trade with yourself)
            if player["team_abbr"] == team_abbr:
                continue

            # How well does this player's role match our needs?
            role_urgency = urgency_by_role.get(player["primary_role"], 0)
            # Also consider secondary role at half weight
            if player["secondary_role"]:
                secondary_urgency = urgency_by_role.get(player["secondary_role"], 0)
                role_urgency = max(role_urgency, 0.5 * secondary_urgency)

            # Rotation Players get a baseline urgency scaled by their quality,
            # so better bench players are meaningfully preferred over worse ones.
            # This creates differentiation in the preference graph.
            if role_urgency == 0:
                quality_preview = _player_quality_score(player)
                role_urgency = 0.05 + 0.15 * max(0, quality_preview / 30.0)

            quality = _player_quality_score(player)
            pref_score = role_urgency * quality

            if pref_score > 0:
                scores.append((int(player["player_id"]), round(pref_score, 2)))

        # Sort by preference score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        preferences[team_abbr] = scores

    return preferences
