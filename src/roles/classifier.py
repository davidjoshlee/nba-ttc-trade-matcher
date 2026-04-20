"""
Classify NBA players into roles based on their stats.
"""

import pandas as pd
from src.roles.taxonomy import ROLE_DEFINITIONS, DEFAULT_ROLE


def _compute_derived_stats(row: pd.Series) -> dict:
    """Add computed fields needed by role definitions."""
    stats = row.to_dict()
    stats["ast_to_tov"] = stats["ast"] / stats["tov"] if stats["tov"] > 0 else stats["ast"]
    stats["stl_plus_blk"] = stats["stl"] + stats["blk"]
    return stats


def _score_role(player_stats: dict, role_name: str) -> float:
    """
    Score how well a player fits a role.
    Uses the MINIMUM ratio across all stat thresholds — a player must
    meet every requirement, not just dominate one. This prevents, e.g.,
    a guard with high rebounds-ratio from being classified as a big.
    The min ratio is the primary score; we add a small bonus from the
    average to break ties.
    """
    thresholds = ROLE_DEFINITIONS[role_name]
    ratios = []
    for stat, threshold in thresholds.items():
        value = player_stats.get(stat, 0)
        if threshold > 0:
            ratios.append(value / threshold)
        else:
            ratios.append(1.0)
    min_ratio = min(ratios)
    avg_ratio = sum(ratios) / len(ratios)
    # Primary signal is the bottleneck stat; small tiebreaker from average
    return min_ratio + 0.01 * avg_ratio


def classify_player(row: pd.Series) -> dict:
    """
    Classify a single player into primary and secondary roles.

    Returns:
        {
            "primary_role": str,
            "secondary_role": str | None,
            "role_scores": {role_name: float, ...}
        }
    """
    stats = _compute_derived_stats(row)

    scores = {}
    for role_name in ROLE_DEFINITIONS:
        scores[role_name] = _score_role(stats, role_name)

    # Sort by score descending
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # Primary role: must score >= 0.85 to qualify (otherwise "Rotation Player")
    primary_role = DEFAULT_ROLE
    secondary_role = None

    if ranked[0][1] >= 0.85:
        primary_role = ranked[0][0]
        # Secondary role: must also score >= 0.85 and be different
        if ranked[1][1] >= 0.85:
            secondary_role = ranked[1][0]

    return {
        "primary_role": primary_role,
        "secondary_role": secondary_role,
        "role_scores": scores,
    }


def classify_all_players(player_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Classify all players and add role columns to the DataFrame.

    Returns a new DataFrame with added columns:
        - primary_role
        - secondary_role
        - role_scores (dict)
    """
    classifications = player_stats.apply(classify_player, axis=1, result_type="expand")
    result = pd.concat([player_stats, classifications], axis=1)
    return result
