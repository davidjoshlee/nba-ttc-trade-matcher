"""
Identify each team's roster gaps by comparing their role composition
to a team-specific ideal roster template.

Key design choice: instead of a single static ideal for all teams, we
adjust each team's ideal based on their current statistical profile.
Teams weak in scoring prioritize offensive roles; teams weak in defense
prioritize defensive roles. This creates diverse preferences across
teams, which is critical for generating multi-team trade cycles.
"""

import pandas as pd
import numpy as np
from src.config import IDEAL_ROSTER_COMPOSITION


def _compute_team_profile(roster: pd.DataFrame) -> dict[str, float]:
    """
    Compute a team's statistical profile as percentile ranks.
    Returns dict of stat category -> z-score relative to league average.
    """
    return {
        "scoring": roster["pts"].mean(),
        "rebounding": roster["reb"].mean(),
        "playmaking": roster["ast"].mean(),
        "defense": (roster["stl"].mean() + roster["blk"].mean()),
        "shooting": roster["fg3_pct"].mean() if roster["fg3_pct"].mean() > 0 else 0,
    }


def _adjust_ideal_for_team(base_ideal: dict, profile: dict,
                            league_profiles: dict[str, dict]) -> dict[str, float]:
    """
    Adjust the ideal roster composition based on team needs.
    Teams below league median in a category get higher ideal counts
    for roles that address that weakness.
    """
    # Compute league medians
    medians = {}
    for stat in profile:
        values = [p[stat] for p in league_profiles.values()]
        medians[stat] = np.median(values)

    adjusted = dict(base_ideal)

    # Teams weak in scoring -> want more Volume Scorers and Floor Spacers
    if profile["scoring"] < medians["scoring"]:
        adjusted["Volume Scorer"] = adjusted.get("Volume Scorer", 1) + 1
        adjusted["Floor Spacer"] = adjusted.get("Floor Spacer", 3) + 1

    # Teams weak in rebounding -> want more Rim Protectors and Paint Scorers
    if profile["rebounding"] < medians["rebounding"]:
        adjusted["Rim Protector"] = adjusted.get("Rim Protector", 1) + 1
        adjusted["Paint Scorer"] = adjusted.get("Paint Scorer", 1) + 1

    # Teams weak in playmaking -> want more Playmakers
    if profile["playmaking"] < medians["playmaking"]:
        adjusted["Playmaker"] = adjusted.get("Playmaker", 2) + 1

    # Teams weak in defense -> want more Defensive Anchors and Two-Way Wings
    if profile["defense"] < medians["defense"]:
        adjusted["Defensive Anchor"] = adjusted.get("Defensive Anchor", 1) + 1
        adjusted["Two-Way Wing"] = adjusted.get("Two-Way Wing", 2) + 1

    # Teams weak in shooting -> want more Stretch Bigs and Floor Spacers
    if profile["shooting"] < medians["shooting"]:
        adjusted["Stretch Big"] = adjusted.get("Stretch Big", 1) + 1
        adjusted["Floor Spacer"] = adjusted.get("Floor Spacer", 3) + 1

    return adjusted


def analyze_team_gaps(classified_players: pd.DataFrame) -> dict[str, list[tuple[str, float]]]:
    """
    For each team, compare roster role counts to a team-specific ideal composition.
    Returns {team_abbr: [(role_name, urgency_score), ...]} sorted by urgency descending.

    Urgency = (ideal_count - actual_count) / ideal_count
    A score of 1.0 means the team has zero players in a role where they need some.
    A score of 0.0 means the team meets the ideal. Negative means surplus.
    """
    # First pass: compute all team profiles
    league_profiles = {}
    for team_abbr, roster in classified_players.groupby("team_abbr"):
        league_profiles[team_abbr] = _compute_team_profile(roster)

    team_gaps = {}

    for team_abbr, roster in classified_players.groupby("team_abbr"):
        # Get team-specific ideal
        team_ideal = _adjust_ideal_for_team(
            IDEAL_ROSTER_COMPOSITION,
            league_profiles[team_abbr],
            league_profiles,
        )

        # Count how many players fill each role (primary or secondary)
        role_counts = {}
        for role in team_ideal:
            primary_count = (roster["primary_role"] == role).sum()
            secondary_count = (roster["secondary_role"] == role).sum()
            role_counts[role] = primary_count + 0.5 * secondary_count

        # Compute urgency for each role
        gaps = []
        for role, ideal in team_ideal.items():
            actual = role_counts.get(role, 0)
            urgency = (ideal - actual) / ideal
            gaps.append((role, round(urgency, 3)))

        # Sort by urgency descending (biggest needs first)
        gaps.sort(key=lambda x: x[1], reverse=True)
        team_gaps[team_abbr] = gaps

    return team_gaps
