"""
Identify each team's roster gaps by comparing their role composition
to an ideal roster template.
"""

import pandas as pd
from src.config import IDEAL_ROSTER_COMPOSITION


def analyze_team_gaps(classified_players: pd.DataFrame) -> dict[str, list[tuple[str, float]]]:
    """
    For each team, compare roster role counts to the ideal composition.
    Returns {team_abbr: [(role_name, urgency_score), ...]} sorted by urgency descending.

    Urgency = (ideal_count - actual_count) / ideal_count
    A score of 1.0 means the team has zero players in a role where they need some.
    A score of 0.0 means the team meets the ideal. Negative means surplus.
    """
    team_gaps = {}

    for team_abbr, roster in classified_players.groupby("team_abbr"):
        # Count how many players fill each role (primary or secondary)
        role_counts = {}
        for role in IDEAL_ROSTER_COMPOSITION:
            primary_count = (roster["primary_role"] == role).sum()
            secondary_count = (roster["secondary_role"] == role).sum()
            # Secondary counts for half — a secondary-role player is a partial fit
            role_counts[role] = primary_count + 0.5 * secondary_count

        # Compute urgency for each role
        gaps = []
        for role, ideal in IDEAL_ROSTER_COMPOSITION.items():
            actual = role_counts.get(role, 0)
            urgency = (ideal - actual) / ideal
            gaps.append((role, round(urgency, 3)))

        # Sort by urgency descending (biggest needs first)
        gaps.sort(key=lambda x: x[1], reverse=True)
        team_gaps[team_abbr] = gaps

    return team_gaps
