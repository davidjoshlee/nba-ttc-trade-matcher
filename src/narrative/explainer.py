"""
Template-based narrative generation for trade cycles.

Generates human-readable explanations of why each trade makes sense,
using structured templates that pull from trade data. No external API needed.

Design tradeoff: Template-based narratives are deterministic and faithful
to the underlying data (zero hallucination risk), but lack the nuance and
variety of LLM-generated text.
"""

import random

# Templates for individual team explanations within a trade
_TEAM_TEMPLATES = [
    "{team} addresses their need at {needed_role} by acquiring {receives_name} "
    "({receives_role}, {receives_pts}/{receives_reb}/{receives_ast}) from {source_team}, "
    "while parting with {gives_name} ({gives_role}) — a position where they had depth to spare.",

    "{team} upgrades at {needed_role} with the addition of {receives_name} "
    "({receives_pts} PPG, {receives_reb} RPG, {receives_ast} APG), "
    "sending {gives_name} ({gives_role}) to {dest_team} in return.",

    "For {team}, this deal fills a gap at {needed_role}. They land {receives_name} "
    "({receives_role}) and move {gives_name} ({gives_role}), where they were deep.",
]

# Templates for cycle-level summary
_CYCLE_SUMMARY_TEMPLATES_2TEAM = [
    "A straightforward swap: {team_a} and {team_b} exchange players to address "
    "complementary roster needs.",

    "{team_a} and {team_b} both improve by trading from positions of strength "
    "to fill gaps on their rosters.",
]

_CYCLE_SUMMARY_TEMPLATES_MULTI = [
    "A {n}-team trade cycle that unlocks value no bilateral deal could: "
    "{team_list} each send a player and receive one that better fits their needs.",

    "This {n}-team deal shows the power of multi-party coordination — "
    "{team_list} each give up depth at one position to fill a gap at another.",
]


def explain_cycle(
    cycle: dict,
    team_gaps: dict[str, list[tuple[str, float]]],
) -> dict:
    """
    Generate a narrative explanation for a trade cycle.

    Args:
        cycle: A cycle dict from run_ttc
        team_gaps: {team_abbr: [(role, urgency), ...]} from gap analysis

    Returns:
        {
            "summary": str,        # Cycle-level summary
            "team_details": [str],  # Per-team explanations
        }
    """
    trades = cycle["trades"]
    teams = cycle["cycle"]
    n = len(teams)

    # Generate cycle summary
    if n == 2:
        template = random.choice(_CYCLE_SUMMARY_TEMPLATES_2TEAM)
        summary = template.format(team_a=teams[0], team_b=teams[1])
    else:
        template = random.choice(_CYCLE_SUMMARY_TEMPLATES_MULTI)
        team_list = ", ".join(teams[:-1]) + f" and {teams[-1]}"
        summary = template.format(n=n, team_list=team_list)

    # Generate per-team explanations
    team_details = []
    for trade in trades:
        team = trade["team"]
        gives = trade["gives"]
        receives = trade["receives"]

        # Find what this team needed most
        gaps = team_gaps.get(team, [])
        top_need = gaps[0][0] if gaps else "depth"

        # Find which team they're sending their player to
        # (the team that receives what this team gives)
        dest_team = None
        source_team = None
        for other_trade in trades:
            if other_trade["receives"]["player_id"] == gives["player_id"]:
                dest_team = other_trade["team"]
            if other_trade["gives"]["player_id"] == receives["player_id"]:
                source_team = other_trade["team"]

        template = random.choice(_TEAM_TEMPLATES)
        detail = template.format(
            team=team,
            needed_role=top_need,
            receives_name=receives["player_name"],
            receives_role=receives["primary_role"],
            receives_pts=receives["pts"],
            receives_reb=receives["reb"],
            receives_ast=receives["ast"],
            gives_name=gives["player_name"],
            gives_role=gives["primary_role"],
            source_team=source_team or "another team",
            dest_team=dest_team or "another team",
        )
        team_details.append(detail)

    return {
        "summary": summary,
        "team_details": team_details,
    }


def explain_all_cycles(
    cycles: list[dict],
    team_gaps: dict[str, list[tuple[str, float]]],
) -> list[dict]:
    """Generate explanations for all trade cycles."""
    return [explain_cycle(c, team_gaps) for c in cycles]
