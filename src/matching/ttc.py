"""
Top Trading Cycles (TTC) algorithm for multi-team NBA trades.

In this variant, each team offers a pool of available players (not just one).
Each team "points" to the team owning their most-preferred available player.
When a cycle is found, each team in the cycle sends a player and receives one.

The sending player is selected as the player the NEXT team in the cycle most
wants from the sending team's available pool. This ensures trades are mutually
beneficial: each team gives what others value and gets what they need.

Key property: No team in a completed cycle would prefer to undo the trade.
"""

import pandas as pd


def _find_all_cycles(graph: dict[str, str]) -> list[list[str]]:
    """
    Find all cycles in a functional graph (each node has at most one outgoing edge).
    Returns a list of cycles, each a list of nodes.
    """
    cycles = []
    visited_global = set()

    for start_node in graph:
        if start_node in visited_global:
            continue

        path = []
        path_set = set()
        current = start_node

        while current not in visited_global and current not in path_set:
            if current not in graph:
                break
            path.append(current)
            path_set.add(current)
            current = graph[current]

        if current in path_set:
            # Found a cycle — extract it
            cycle_start_idx = path.index(current)
            cycle = path[cycle_start_idx:]
            if len(cycle) >= 2:
                cycles.append(cycle)

        visited_global.update(path_set)

    return cycles


def run_ttc(
    classified_players: pd.DataFrame,
    preferences: dict[str, list[tuple[int, float]]],
    available_players: pd.DataFrame,
    team_gaps: dict[str, list[tuple[str, float]]],
) -> list[dict]:
    """
    Run the Top Trading Cycles algorithm.

    Args:
        classified_players: Full player DataFrame with role classifications
        preferences: {team_abbr: [(player_id, score), ...]} sorted by preference
        available_players: DataFrame of players marked available for trade
        team_gaps: {team_abbr: [(role, urgency), ...]} from gap analysis

    Returns list of trade cycles.
    """
    # Build lookup structures
    player_lookup = {}
    for _, row in classified_players.iterrows():
        player_lookup[int(row["player_id"])] = row.to_dict()

    # Map each available player to their team
    player_to_team = {}
    for _, row in available_players.iterrows():
        player_to_team[int(row["player_id"])] = row["team_abbr"]

    # Track which players are still available
    remaining_player_ids = set(player_to_team.keys())
    remaining_teams = set(available_players["team_abbr"].unique())
    all_cycles = []

    while remaining_teams:
        # Build pointing graph: each team -> team owning their top-preferred
        # available player
        graph = {}
        team_wants_player = {}

        for team in remaining_teams:
            if team not in preferences:
                continue

            # Find this team's most-preferred player still available
            best_player = None
            for pid, score in preferences[team]:
                if pid in remaining_player_ids and player_to_team.get(pid) != team:
                    best_player = pid
                    break

            if best_player is None:
                continue

            target_team = player_to_team[best_player]
            if target_team not in remaining_teams:
                continue

            graph[team] = target_team
            team_wants_player[team] = best_player

        if not graph:
            break

        # Find all cycles
        cycles = _find_all_cycles(graph)

        if not cycles:
            break

        for cycle in cycles:
            # For each team in the cycle, determine what they give and receive
            trades = []
            for i, team in enumerate(cycle):
                # This team receives the player they pointed at
                receives_pid = team_wants_player[team]

                # This team gives a player to the PREVIOUS team in the cycle.
                # The previous team wants a player from this team.
                # Find which of this team's available players the previous team
                # values most.
                prev_team = cycle[i - 1]  # Python negative indexing handles wrap

                gives_pid = _select_player_to_give(
                    team, prev_team, remaining_player_ids, player_to_team,
                    preferences, team_gaps
                )

                gives_info = player_lookup.get(gives_pid, {})
                receives_info = player_lookup.get(receives_pid, {})

                trades.append({
                    "team": team,
                    "gives": {
                        "player_id": gives_pid,
                        "player_name": gives_info.get("player_name", "Unknown"),
                        "primary_role": gives_info.get("primary_role", "Unknown"),
                        "pts": gives_info.get("pts", 0),
                        "reb": gives_info.get("reb", 0),
                        "ast": receives_info.get("ast", 0),
                    },
                    "receives": {
                        "player_id": receives_pid,
                        "player_name": receives_info.get("player_name", "Unknown"),
                        "primary_role": receives_info.get("primary_role", "Unknown"),
                        "pts": receives_info.get("pts", 0),
                        "reb": receives_info.get("reb", 0),
                        "ast": receives_info.get("ast", 0),
                    },
                })

            all_cycles.append({
                "cycle": cycle,
                "num_teams": len(cycle),
                "trades": trades,
            })

            # Remove traded teams and their players from the pool
            for team in cycle:
                remaining_teams.discard(team)
                # Remove all of this team's available players
                pids_to_remove = [
                    pid for pid, t in player_to_team.items()
                    if t == team and pid in remaining_player_ids
                ]
                remaining_player_ids -= set(pids_to_remove)

    return all_cycles


def _select_player_to_give(
    giving_team: str,
    receiving_team: str,
    remaining_player_ids: set[int],
    player_to_team: dict[int, str],
    preferences: dict[str, list[tuple[int, float]]],
    team_gaps: dict[str, list[tuple[str, float]]],
) -> int:
    """
    Select which player the giving team sends to the receiving team.
    Picks the player from giving_team's available pool that the receiving_team
    values most (highest in their preference ranking).
    """
    giving_team_pids = [
        pid for pid in remaining_player_ids
        if player_to_team.get(pid) == giving_team
    ]

    if not giving_team_pids:
        return giving_team_pids[0] if giving_team_pids else 0

    # Find which of these players the receiving team ranks highest
    receiving_prefs = preferences.get(receiving_team, [])
    giving_set = set(giving_team_pids)

    for pid, score in receiving_prefs:
        if pid in giving_set:
            return pid

    # Fallback: give the lowest-minutes player
    return giving_team_pids[0]
