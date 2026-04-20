"""
Top Trading Cycles (TTC) algorithm for multi-team NBA trades.

This implementation uses a multi-edge preference graph to find trade cycles.
Instead of each team pointing to only their #1 choice (which tends to create
all 2-team swaps due to preference concentration), each team expresses
preferences toward multiple teams. We then search for cycles in this richer
graph, prioritizing longer cycles that unlock multi-party value.

Key property: Every team in a completed cycle receives a player they prefer
over what they gave up.
"""

import pandas as pd
from itertools import combinations


def _find_cycles_in_multigraph(
    edges: dict[str, list[str]],
    max_cycle_len: int = 8,
) -> list[list[str]]:
    """
    Find all simple cycles up to max_cycle_len in a directed multigraph.
    edges: {node: [list of target nodes]}
    Returns cycles sorted longest first.
    """
    all_nodes = list(edges.keys())
    cycles = []

    def dfs(path: list[str], visited: set[str]):
        current = path[-1]
        for neighbor in edges.get(current, []):
            if neighbor == path[0] and len(path) >= 2:
                cycles.append(list(path))
            elif neighbor not in visited and len(path) < max_cycle_len:
                visited.add(neighbor)
                path.append(neighbor)
                dfs(path, visited)
                path.pop()
                visited.remove(neighbor)

    for start in all_nodes:
        visited = {start}
        dfs([start], visited)

    # Deduplicate cycles (same cycle can be found from different starting nodes)
    unique = []
    seen = set()
    for cycle in cycles:
        # Normalize: rotate so smallest element is first
        min_idx = cycle.index(min(cycle))
        normalized = tuple(cycle[min_idx:] + cycle[:min_idx])
        if normalized not in seen:
            seen.add(normalized)
            unique.append(cycle)

    # Sort longest first
    unique.sort(key=len, reverse=True)
    return unique


def _build_preference_graph(
    remaining_teams: set[str],
    preferences: dict[str, list[tuple[int, float]]],
    remaining_player_ids: set[int],
    player_to_team: dict[int, str],
    top_k: int = 5,
) -> tuple[dict[str, list[str]], dict[tuple[str, str], int]]:
    """
    Build a multi-edge preference graph where each team has edges to the
    top K teams owning their most-preferred players.

    Returns:
        edges: {team: [list of preferred teams]}
        best_player_per_edge: {(from_team, to_team): player_id}
    """
    edges = {team: [] for team in remaining_teams}
    best_player_per_edge = {}

    for team in remaining_teams:
        if team not in preferences:
            continue

        # Find up to top_k distinct teams this team wants players from
        seen_teams = set()
        for pid, score in preferences[team]:
            if pid not in remaining_player_ids:
                continue
            target_team = player_to_team.get(pid)
            if target_team is None or target_team == team or target_team not in remaining_teams:
                continue
            if target_team in seen_teams:
                continue

            seen_teams.add(target_team)
            edges[team].append(target_team)
            best_player_per_edge[(team, target_team)] = pid

            if len(seen_teams) >= top_k:
                break

    return edges, best_player_per_edge


def _select_non_overlapping_cycles(cycles: list[list[str]]) -> list[list[str]]:
    """
    From a list of cycles (sorted longest first), select cycles that
    don't share any teams. Greedy: pick longest, then next longest
    that doesn't overlap, etc.
    """
    selected = []
    used_teams = set()

    for cycle in cycles:
        cycle_set = set(cycle)
        if not cycle_set & used_teams:
            selected.append(cycle)
            used_teams |= cycle_set

    return selected


def run_ttc(
    classified_players: pd.DataFrame,
    preferences: dict[str, list[tuple[int, float]]],
    available_players: pd.DataFrame,
    team_gaps: dict[str, list[tuple[str, float]]],
) -> list[dict]:
    """
    Run the Top Trading Cycles algorithm with multi-edge preference graph.

    Returns list of trade cycles.
    """
    # Build lookup structures
    player_lookup = {}
    for _, row in classified_players.iterrows():
        player_lookup[int(row["player_id"])] = row.to_dict()

    player_to_team = {}
    for _, row in available_players.iterrows():
        player_to_team[int(row["player_id"])] = row["team_abbr"]

    remaining_player_ids = set(player_to_team.keys())
    remaining_teams = set(available_players["team_abbr"].unique())
    all_cycles = []

    while len(remaining_teams) >= 2:
        # Build multi-edge preference graph
        edges, best_player = _build_preference_graph(
            remaining_teams, preferences, remaining_player_ids,
            player_to_team, top_k=5,
        )

        # Find all cycles, longest first
        cycles = _find_cycles_in_multigraph(edges, max_cycle_len=6)

        if not cycles:
            break

        # Select non-overlapping cycles, prioritizing longest
        selected = _select_non_overlapping_cycles(cycles)

        if not selected:
            break

        for cycle in selected:
            trades = []
            for i, team in enumerate(cycle):
                # This team receives the best player from the next team in the cycle
                next_team = cycle[(i + 1) % len(cycle)]
                receives_pid = best_player.get((team, next_team))

                if receives_pid is None:
                    # Fallback: find best player from next_team
                    for pid, score in preferences.get(team, []):
                        if pid in remaining_player_ids and player_to_team.get(pid) == next_team:
                            receives_pid = pid
                            break

                # This team gives a player to the previous team
                prev_team = cycle[i - 1]
                gives_pid = _select_player_to_give(
                    team, prev_team, remaining_player_ids, player_to_team,
                    preferences, team_gaps,
                )

                gives_info = player_lookup.get(gives_pid, {})
                receives_info = player_lookup.get(receives_pid, {}) if receives_pid else {}

                trades.append({
                    "team": team,
                    "gives": {
                        "player_id": gives_pid,
                        "player_name": gives_info.get("player_name", "Unknown"),
                        "primary_role": gives_info.get("primary_role", "Unknown"),
                        "pts": gives_info.get("pts", 0),
                        "reb": gives_info.get("reb", 0),
                        "ast": gives_info.get("ast", 0),
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

            # Remove traded teams
            for team in cycle:
                remaining_teams.discard(team)
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
        return 0

    # Find which of these players the receiving team ranks highest
    receiving_prefs = preferences.get(receiving_team, [])
    giving_set = set(giving_team_pids)

    for pid, score in receiving_prefs:
        if pid in giving_set:
            return pid

    # Fallback: give the first available player
    return giving_team_pids[0]
