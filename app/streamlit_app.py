"""
NBA Trade Matcher — Top Trading Cycles
Streamlit web UI for the OIT 277 group project.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from openai import OpenAI

from src.data.fetch_rosters import fetch_player_stats
from src.roles.classifier import classify_all_players
from src.preferences.gap_analysis import analyze_team_gaps
from src.preferences.ranker import generate_preferences, identify_available_players
from src.matching.ttc import run_ttc
from src.narrative.explainer import explain_all_cycles
from src.config import IDEAL_ROSTER_COMPOSITION


def generate_ai_narrative(cycle: dict, api_key: str) -> str:
    """Call OpenAI gpt-4o to generate a narrative explanation for a trade cycle."""
    trades = cycle["trades"]
    lines = []
    for t in trades:
        g, r = t["gives"], t["receives"]
        lines.append(
            f"- {t['team']} sends {g['player_name']} ({g['primary_role']}, "
            f"{g['pts']} PPG/{g['reb']} RPG/{g['ast']} APG) and receives "
            f"{r['player_name']} ({r['primary_role']}, {r['pts']} PPG/{r['reb']} RPG/{r['ast']} APG)"
        )
    trade_summary = "\n".join(lines)
    n = len(trades)

    prompt = (
        f"You are an NBA analyst. Explain in 3-4 sentences why this {n}-team trade makes sense "
        f"for each team. Be specific about roster fit, what each team gains, and why they give up "
        f"what they do. Keep it conversational and insightful — like an ESPN breakdown.\n\n"
        f"Trade details:\n{trade_summary}"
    )

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()

# --- Page config ---
st.set_page_config(
    page_title="NBA Trade Matcher — TTC",
    page_icon="🏀",
    layout="wide",
)

# --- Sidebar ---
st.sidebar.title("NBA Trade Matcher")
st.sidebar.markdown("**Leveraging the Top Trading Cycles Algorithm**")
st.sidebar.markdown("OIT 277 — Digital Platforms in AI")
st.sidebar.markdown("---")

players_available = st.sidebar.slider(
    "Players available per team",
    min_value=3, max_value=10, value=7,
    help="How many players each team puts on the trade block (bottom N by minutes played)"
)

min_minutes = st.sidebar.slider(
    "Minimum minutes/game",
    min_value=5.0, max_value=20.0, value=10.0, step=1.0,
    help="Only include players averaging at least this many minutes per game"
)

st.sidebar.markdown("---")
openai_api_key = st.sidebar.text_input(
    "OpenAI API Key",
    type="password",
    placeholder="sk-...",
    help="Enter your OpenAI key to enable AI-generated trade narratives",
)

# --- Data loading & pipeline ---
@st.cache_data
def load_player_data():
    """Fetch and cache raw player stats (only hits API once)."""
    return fetch_player_stats()


@st.cache_data
def run_pipeline(players_per_team: int, min_min: float):
    """Run the full pipeline with the given parameters."""
    df = load_player_data()
    # Filter by minutes threshold
    df = df[df["min"] >= min_min].reset_index(drop=True)
    classified = classify_all_players(df)
    gaps = analyze_team_gaps(classified)
    prefs = generate_preferences(classified, gaps, players_per_team=players_per_team)
    avail_df = identify_available_players(classified, players_per_team=players_per_team)
    available_only = avail_df[avail_df.available_for_trade]
    cycles = run_ttc(classified, prefs, available_only, gaps)
    explanations = explain_all_cycles(cycles, gaps)

    return classified, gaps, prefs, avail_df, cycles, explanations


# Run pipeline with current slider values, with a visible spinner
with st.spinner("Running trade matching algorithm..."):
    classified, gaps, prefs, avail_df, cycles, explanations = run_pipeline(
        players_available, min_minutes
    )

# Show current config in sidebar after pipeline runs
avail_count = int(avail_df["available_for_trade"].sum())
st.sidebar.markdown("---")
st.sidebar.markdown("**Current Run**")
st.sidebar.markdown(
    f"- **{len(classified)}** players in pool\n"
    f"- **{avail_count}** available for trade\n"
    f"- **{len(cycles)}** trade cycles found"
)

# --- Tabs ---
tab_overview, tab_dashboard, tab_teams, tab_trades, tab_eval = st.tabs([
    "App Overview", "Dashboard", "Team Explorer", "Trade Cycles", "AI Evaluation"
])

# --- DASHBOARD TAB ---
with tab_dashboard:
    st.header("Dashboard")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Players", len(classified))
    with col2:
        st.metric("Teams", classified["team_abbr"].nunique())
    with col3:
        st.metric("Trade Cycles Found", len(cycles))
    with col4:
        multi = sum(1 for c in cycles if c["num_teams"] > 2)
        st.metric("Multi-Team Trades", multi)

    st.markdown("---")

    # Trade cycle network graph
    st.subheader("Trade Network")
    if cycles:
        fig, ax = plt.subplots(figsize=(12, 8))
        G = nx.DiGraph()

        colors_list = plt.cm.Set3.colors
        for i, cycle in enumerate(cycles):
            color = colors_list[i % len(colors_list)]
            teams = cycle["cycle"]
            for j, team in enumerate(teams):
                next_team = teams[(j + 1) % len(teams)]
                G.add_edge(team, next_team, cycle=i + 1)
                G.nodes[team]["color"] = color

        node_colors = [G.nodes[n].get("color", "#cccccc") for n in G.nodes()]
        pos = nx.spring_layout(G, k=2, seed=42)
        nx.draw(G, pos, ax=ax, with_labels=True, node_color=node_colors,
                node_size=1200, font_size=9, font_weight="bold",
                edge_color="#666666", arrows=True, arrowsize=20,
                edgecolors="black", linewidths=1.5)
        ax.set_title("Trade Cycles — Each color = one trade cycle", fontsize=14)
        st.pyplot(fig)
        plt.close()
    else:
        st.info("No trade cycles found. Try adjusting the parameters.")

    # Role distribution
    st.subheader("League-Wide Role Distribution")
    role_counts = classified["primary_role"].value_counts()
    st.bar_chart(role_counts)

# --- TEAM EXPLORER TAB ---
with tab_teams:
    st.header("Team Explorer")

    team_list = sorted(classified["team_abbr"].unique())
    selected_team = st.selectbox("Select a team", team_list)

    if selected_team:
        team_roster = classified[classified["team_abbr"] == selected_team].copy()
        team_avail = avail_df[
            (avail_df["team_abbr"] == selected_team) & avail_df["available_for_trade"]
        ]

        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader(f"{selected_team} Roster")
            display_cols = ["player_name", "primary_role", "secondary_role",
                           "pts", "reb", "ast", "stl", "blk", "min"]
            roster_display = team_roster[display_cols].sort_values("min", ascending=False)
            roster_display = roster_display.reset_index(drop=True)
            st.dataframe(roster_display, use_container_width=True, hide_index=True)

        with col2:
            st.subheader("Roster Gaps")
            team_gap = gaps.get(selected_team, [])
            for role, urgency in team_gap:
                if urgency > 0:
                    st.markdown(f"- **{role}**: urgency {urgency:.2f}")
                elif urgency == 0:
                    st.markdown(f"- {role}: balanced")
                else:
                    st.markdown(f"- ~~{role}~~: surplus ({urgency:.2f})")

            st.subheader("Role Composition")
            team_roles = team_roster["primary_role"].value_counts()
            st.bar_chart(team_roles)

        # Available players
        st.subheader(f"Players Available for Trade ({len(team_avail)})")
        if not team_avail.empty:
            avail_display = team_avail[["player_name", "primary_role", "pts", "reb", "ast", "min"]]
            st.dataframe(avail_display.reset_index(drop=True), use_container_width=True, hide_index=True)

        # Top preferences
        st.subheader("Top 10 Preferred Available Players (from other teams)")
        team_prefs = prefs.get(selected_team, [])[:10]
        if team_prefs:
            pref_rows = []
            for pid, score in team_prefs:
                player = classified[classified["player_id"] == pid]
                if not player.empty:
                    p = player.iloc[0]
                    pref_rows.append({
                        "Player": p["player_name"],
                        "Team": p["team_abbr"],
                        "Role": p["primary_role"],
                        "PTS": p["pts"],
                        "REB": p["reb"],
                        "AST": p["ast"],
                        "Pref Score": score,
                    })
            st.dataframe(pd.DataFrame(pref_rows), use_container_width=True, hide_index=True)

# --- TRADE CYCLES TAB ---
with tab_trades:
    st.header("Trade Cycles")

    if not cycles:
        st.info("No trade cycles found. Try adjusting the parameters.")
    else:
        # Summary
        two_team = sum(1 for c in cycles if c["num_teams"] == 2)
        multi_team = sum(1 for c in cycles if c["num_teams"] > 2)
        st.markdown(f"**{len(cycles)} total trades**: {two_team} two-team swaps, {multi_team} multi-team deals")

        # Show multi-team trades first
        sorted_cycles = sorted(
            zip(cycles, explanations),
            key=lambda x: x[0]["num_teams"],
            reverse=True,
        )

        if not openai_api_key:
            st.info("Enter your OpenAI API key in the sidebar to enable AI-generated trade narratives.")

        for i, (cycle, explanation) in enumerate(sorted_cycles):
            teams = cycle["cycle"]
            label = f"{'⭐ ' if cycle['num_teams'] > 2 else ''}{cycle['num_teams']}-Team Trade: {' ↔ '.join(teams)}"

            with st.expander(label, expanded=cycle["num_teams"] > 2):
                # Trade breakdown
                for trade in cycle["trades"]:
                    col1, col2, col3 = st.columns([1, 0.5, 1])
                    with col1:
                        g = trade["gives"]
                        st.markdown(f"**{trade['team']}** sends:")
                        st.markdown(f"**{g['player_name']}** ({g['primary_role']})")
                        st.caption(f"{g['pts']} PPG / {g['reb']} RPG / {g['ast']} APG")
                    with col2:
                        st.markdown("<div style='text-align:center; padding-top:20px; font-size:24px'>→</div>",
                                    unsafe_allow_html=True)
                    with col3:
                        r = trade["receives"]
                        st.markdown(f"**{trade['team']}** receives:")
                        st.markdown(f"**{r['player_name']}** ({r['primary_role']})")
                        st.caption(f"{r['pts']} PPG / {r['reb']} RPG / {r['ast']} APG")
                    st.markdown("")

                st.markdown("---")

                # AI narrative
                cache_key = f"ai_narrative_{i}"
                if openai_api_key:
                    if cache_key not in st.session_state:
                        with st.spinner("Generating AI analysis..."):
                            try:
                                st.session_state[cache_key] = generate_ai_narrative(cycle, openai_api_key)
                            except Exception as e:
                                st.session_state[cache_key] = f"⚠️ Could not generate narrative: {e}"
                    st.markdown(f"**AI Analysis**\n\n{st.session_state[cache_key]}")
                else:
                    st.markdown(f"*{explanation['summary']}*")
                    for detail in explanation["team_details"]:
                        st.markdown(f"> {detail}")
                    st.markdown("")

# --- APP OVERVIEW TAB ---
with tab_overview:
    st.header("App Overview")
    st.markdown("This app identifies multi-team NBA trade opportunities using a 6-stage data pipeline. Here's what happens under the hood.")
    st.markdown("---")

    st.subheader("Step 1 — Data Collection")
    st.markdown("""
    - Pulls live per-game stats for every NBA player from NBA.com
    - Filters out players with **<10** minutes per game
    - Result: ~458 players across all 30 teams
    """)

    team_list_overview = sorted(classified["team_abbr"].unique())
    selected_team_overview = st.selectbox("Select a team to explore its data", team_list_overview, index=team_list_overview.index("TOR"), key="overview_team")

    team_data = classified[classified["team_abbr"] == selected_team_overview][
        ["player_name", "pts", "reb", "ast", "stl", "blk", "min", "gp"]
    ].sort_values("min", ascending=False).reset_index(drop=True)

    st.markdown(f"**{selected_team_overview}** — {len(team_data)} players in dataset")
    st.dataframe(team_data, use_container_width=True, hide_index=True)

    st.markdown("---")

    st.subheader("Step 2 — Role Classification")
    st.markdown("""
    - Each player is assigned one of 8 archetypes based on their per-game stats
    - Players must meet **all** stat thresholds for a role — one standout stat isn't enough
    - Players who don't fit any archetype are labeled **Rotation Player**
    """)

    st.markdown("**Role Classification Rubric** — minimum thresholds a player must meet to qualify for each role:")
    rubric_data = {
        "Role": ["Floor Spacer", "Rim Protector", "Playmaker", "Paint Scorer", "Two-Way Wing", "Stretch Big", "Defensive Anchor", "Volume Scorer"],
        "PTS": ["", "", "", "12", "14", "", "", "22"],
        "REB": ["", "7", "", "6", "4", "7", "", ""],
        "AST": ["", "", "6", "", "", "", "", ""],
        "BLK": ["", "1", "", "", "", "", "", ""],
        "STL": ["", "", "", "", "1", "", "", ""],
        "3PA": ["5", "", "", "", "", "2", "", ""],
        "3P%": ["36%", "", "", "", "", "32%", "", ""],
        "FG%": ["", "", "", "54%", "", "", "", ""],
        "FGA": ["", "", "", "", "", "", "", "17"],
        "AST/TOV": ["", "", "2", "", "", "", "", ""],
        "STL+BLK": ["", "", "", "", "", "", "2", ""],
        "MIN": ["", "", "", "", "", "", "27", ""],
    }
    rubric_df = pd.DataFrame(rubric_data)
    st.dataframe(
        rubric_df.style.set_properties(**{"text-align": "center"}).set_properties(subset=["Role"], **{"text-align": "left"}),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("---")
    st.markdown("**Example Players** — select a role to see an example player that fits it:")

    ALL_COLS = ["PTS", "REB", "AST", "BLK", "STL", "3PA", "3P%", "FG%", "FGA", "AST/TOV", "STL+BLK", "MIN"]

    example_players = {
        "Floor Spacer": {
            "player": "Steph Curry (GSW)",
            "standard": {"3PA": "5", "3P%": "36%"},
            "performance": {"3PA": "12", "3P%": "40%"},
        },
        "Rim Protector": {
            "player": "Rudy Gobert (MIN)",
            "standard": {"REB": "7", "BLK": "1"},
            "performance": {"REB": "12", "BLK": "2"},
        },
        "Playmaker": {
            "player": "LeBron James (LAL)",
            "standard": {"AST": "6", "AST/TOV": "2"},
            "performance": {"AST": "8", "AST/TOV": "2"},
        },
        "Paint Scorer": {
            "player": "Giannis Antetokounmpo (MIL)",
            "standard": {"PTS": "12", "REB": "6", "FG%": "54%"},
            "performance": {"PTS": "28", "REB": "11", "FG%": "58%"},
        },
        "Two-Way Wing": {
            "player": "Kawhi Leonard (LAC)",
            "standard": {"PTS": "14", "REB": "4", "STL": "1"},
            "performance": {"PTS": "24", "REB": "7", "STL": "2"},
        },
        "Stretch Big": {
            "player": "Kevin Durant (PHX)",
            "standard": {"REB": "7", "3PA": "2", "3P%": "32%"},
            "performance": {"REB": "7", "3PA": "5", "3P%": "38%"},
        },
        "Defensive Anchor": {
            "player": "Draymond Green (GSW)",
            "standard": {"STL+BLK": "2", "MIN": "27"},
            "performance": {"STL+BLK": "2", "MIN": "30"},
        },
        "Volume Scorer": {
            "player": "Luka Dončić (DAL)",
            "standard": {"PTS": "22", "FGA": "17"},
            "performance": {"PTS": "33", "FGA": "22"},
        },
        "Rotation Player": None,
    }

    selected_role = st.selectbox("Select a role", list(example_players.keys()), key="overview_role")

    if example_players[selected_role] is None:
        st.info("Rotation Player is the default label for players who don't meet the thresholds for any specific role. They still contribute but don't fit a clear archetype.")
    else:
        p = example_players[selected_role]
        st.markdown(f"**Example: {p['player']} — {selected_role}**")
        def build_row(label, stats):
            row = {"": label}
            for col in ALL_COLS:
                row[col] = stats.get(col, "")
            return row
        example_df = pd.DataFrame([
            build_row("Standard", p["standard"]),
            build_row("Performance", p["performance"]),
        ])
        st.dataframe(
            example_df.style.set_properties(**{"text-align": "center"}).set_properties(subset=[""], **{"text-align": "left"}),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("---")

    st.subheader("Step 3 — Identify Team Needs")
    st.markdown("""
    - Every team starts with the same baseline role targets
    - Targets are adjusted **up** for roles that address that team's statistical weaknesses
    - The result is a customized set of needs unique to each team
    - **Urgency score** = (target − actual) / target — higher means a bigger need
    """)

    _team_list_s3 = sorted(classified["team_abbr"].unique())
    selected_team_step3 = st.selectbox("Select a team", _team_list_s3, index=_team_list_s3.index("TOR"), key="step3_team")

    # a) Baseline needs
    st.markdown("**a) Baseline needs — same for every team:**")
    baseline_df = pd.DataFrame([
        {"Role": role, "Base Target": count}
        for role, count in IDEAL_ROSTER_COMPOSITION.items()
    ])
    st.dataframe(baseline_df, use_container_width=True, hide_index=True)

    # b) Team performance vs league median
    st.markdown("**b) Team performance vs. league median:**")
    import numpy as np
    league_profiles = {}
    for team, roster in classified.groupby("team_abbr"):
        league_profiles[team] = {
            "Scoring (PTS)": roster["pts"].mean(),
            "Rebounding (REB)": roster["reb"].mean(),
            "Playmaking (AST)": roster["ast"].mean(),
            "Defense (STL+BLK)": (roster["stl"] + roster["blk"]).mean(),
            "Shooting (3P%)": roster["fg3_pct"].mean(),
        }
    medians = {stat: np.median([p[stat] for p in league_profiles.values()]) for stat in list(league_profiles.values())[0]}
    team_profile = league_profiles[selected_team_step3]

    adjustment_map = {
        "Scoring (PTS)": "+1 Volume Scorer, +1 Floor Spacer",
        "Rebounding (REB)": "+1 Rim Protector, +1 Paint Scorer",
        "Playmaking (AST)": "+1 Playmaker",
        "Defense (STL+BLK)": "+1 Defensive Anchor, +1 Two-Way Wing",
        "Shooting (3P%)": "+1 Stretch Big, +1 Floor Spacer",
    }
    perf_rows = []
    for stat, value in team_profile.items():
        median = medians[stat]
        below = value < median
        perf_rows.append({
            "Stat": stat,
            "Team Value": round(value, 2),
            "League Median": round(median, 2),
            "Below Median?": "Yes" if below else "No",
            "Adjustment": adjustment_map[stat] if below else "—",
        })
    st.dataframe(pd.DataFrame(perf_rows), use_container_width=True, hide_index=True)

    # c) Customized targets
    st.markdown("**c) Customized targets after adjustments:**")
    adjusted = dict(IDEAL_ROSTER_COMPOSITION)
    stat_to_role_bumps = {
        "Scoring (PTS)": ["Volume Scorer", "Floor Spacer"],
        "Rebounding (REB)": ["Rim Protector", "Paint Scorer"],
        "Playmaking (AST)": ["Playmaker"],
        "Defense (STL+BLK)": ["Defensive Anchor", "Two-Way Wing"],
        "Shooting (3P%)": ["Stretch Big", "Floor Spacer"],
    }
    bumped_roles = set()
    for stat, value in team_profile.items():
        if value < medians[stat]:
            for role in stat_to_role_bumps[stat]:
                adjusted[role] = adjusted.get(role, 0) + 1
                bumped_roles.add(role)
    custom_rows = []
    for role, base in IDEAL_ROSTER_COMPOSITION.items():
        final = adjusted[role]
        custom_rows.append({
            "Role": role,
            "Base Target": base,
            "Adjustment": f"+{final - base}" if final > base else "—",
            "Final Target": final,
        })
    st.dataframe(pd.DataFrame(custom_rows), use_container_width=True, hide_index=True)

    # d) Gaps
    st.markdown("**d) Gaps — actual roster vs. target:**")
    team_roster_s3 = classified[classified["team_abbr"] == selected_team_step3]
    gap_rows = []
    for role, target in adjusted.items():
        actual = (
            (team_roster_s3["primary_role"] == role).sum()
            + 0.5 * (team_roster_s3["secondary_role"] == role).sum()
        )
        urgency = round((target - actual) / target, 2)
        gap_rows.append({
            "Role": role,
            "Target": target,
            "Actual": round(actual, 1),
            "Urgency Score": urgency,
        })
    gap_rows.sort(key=lambda x: x["Urgency Score"], reverse=True)
    st.dataframe(pd.DataFrame(gap_rows), use_container_width=True, hide_index=True)

    st.markdown("---")

    st.subheader("Step 4 — Preference Ranking")
    st.markdown("""
    - For each team, all players available for trade from other teams are ranked by how well they fill that team's gaps
    - **Preference score** = role urgency × quality score
    - **Role urgency** — how badly does this team need the player's role? (from Step 3)
    - **Quality score** — a weighted sum of per-game stats (see table below)
    - The result is an ordered **wish list** for every team — the players they most want to acquire
    """)

    st.markdown("**Quality Score Formula:**")
    quality_weights = pd.DataFrame([
        {"Stat": "PTS", "Weight": "1.0"},
        {"Stat": "REB", "Weight": "0.8"},
        {"Stat": "AST", "Weight": "1.0"},
        {"Stat": "STL", "Weight": "2.0"},
        {"Stat": "BLK", "Weight": "2.0"},
        {"Stat": "TOV", "Weight": "−1.5"},
    ])
    st.dataframe(quality_weights, use_container_width=False, hide_index=True)
    st.markdown("*Note: this formula favors physical players (high BLK, REB, STL) and may undervalue shooters and playmakers — a known limitation of the current design.*")

    st.markdown("**Full league preference ranking:**")
    st.markdown("*Own-team players are highlighted — the algorithm guarantees every trade gives a team a player ranked higher than anyone on their own roster.*")
    _team_list_s4 = sorted(classified["team_abbr"].unique())
    selected_team_step4 = st.selectbox("Select a team", _team_list_s4, index=_team_list_s4.index("TOR"), key="step4_team")

    def quality_score(row):
        return row["pts"] * 1.0 + row["reb"] * 0.8 + row["ast"] * 1.0 + row["stl"] * 2.0 + row["blk"] * 2.0 - row["tov"] * 1.5

    urgency_lookup = {role: max(u, 0) for role, u in gaps.get(selected_team_step4, [])}

    # Score every player in the league
    combined = []
    for _, row in classified.iterrows():
        is_own = row["team_abbr"] == selected_team_step4
        role_urgency = urgency_lookup.get(row["primary_role"], 0)
        if row.get("secondary_role"):
            sec_urgency = urgency_lookup.get(row["secondary_role"], 0)
            role_urgency = max(role_urgency, 0.5 * sec_urgency)
        if role_urgency == 0:
            q = quality_score(row)
            role_urgency = 0.05 + 0.15 * max(0, q / 30.0)
        score = round(role_urgency * quality_score(row), 2)
        combined.append((int(row["player_id"]), score, is_own))

    combined.sort(key=lambda x: x[1], reverse=True)

    team_prefs_step4 = combined
    if team_prefs_step4:
        pref_rows = []
        for rank, (pid, score, is_own) in enumerate(team_prefs_step4, 1):
            player = classified[classified["player_id"] == pid]
            if not player.empty:
                p = player.iloc[0]
                pref_rows.append({
                    "Rank": rank,
                    "Player": p["player_name"],
                    "Current Team": p["team_abbr"],
                    "Role": p["primary_role"],
                    "PTS": p["pts"],
                    "REB": p["reb"],
                    "AST": p["ast"],
                    "Preference Score": score,
                    "_own": is_own,
                })
        pref_df = pd.DataFrame(pref_rows).drop(columns=["_own"])
        pref_df["Preference Score"] = pref_df["Preference Score"].apply(lambda x: round(x, 2) if isinstance(x, float) else x)
        for col in ["PTS", "REB", "AST"]:
            pref_df[col] = pref_df[col].apply(lambda x: round(x, 1) if isinstance(x, float) else x)

        st.markdown("🟡 **Highlighted rows** — players on this team's own roster")

        def highlight_s4(row):
            if row["Current Team"] == selected_team_step4:
                return ["background-color: #3a3a1a; color: #dddd99"] * len(row)
            return [""] * len(row)

        st.dataframe(
            pref_df.style.apply(highlight_s4, axis=1),
            use_container_width=True,
            hide_index=True,
        )

        # AI-style summary
        team_gap_lookup = {role: urgency for role, urgency in gaps.get(selected_team_step4, [])}
        top_roles_needed = [role for role, urgency in sorted(gaps.get(selected_team_step4, []), key=lambda x: x[1], reverse=True) if urgency > 0][:3]
        top_players = pref_rows[:3]

        summary_lines = [f"**Why these players are attractive to {selected_team_step4}:**\n"]
        if top_roles_needed:
            summary_lines.append(f"{selected_team_step4}'s biggest needs are **{top_roles_needed[0]}**" +
                (f", **{top_roles_needed[1]}**" if len(top_roles_needed) > 1 else "") +
                (f", and **{top_roles_needed[2]}**" if len(top_roles_needed) > 2 else "") + ".")

        for i, p in enumerate(top_players):
            urgency = team_gap_lookup.get(p["Role"], 0)
            if urgency > 0.5:
                need_str = "a critical need"
            elif urgency > 0:
                need_str = "a moderate need"
            else:
                need_str = "a position of depth"
            summary_lines.append(
                f"- **{p['Player']}** ({p['Current Team']}) is a **{p['Role']}** — {selected_team_step4} has {need_str} at this role (urgency: {round(urgency, 2)}). "
                f"With {p['PTS']} PPG, {p['REB']} RPG, and {p['AST']} APG, they score {p['Preference Score']} on the preference scale."
            )

        st.markdown("\n".join(summary_lines))

    st.markdown("---")

    st.subheader("Step 5 — TTC Matching")
    st.markdown("""
    - Each team's preference list feeds into the Top Trading Cycles algorithm
    - A preference graph is built — each team points to its **top 5** preferred trade partners
    - A cycle detection algorithm finds all valid trading cycles up to **6 teams** long
    - **Longest cycles are selected first** — every team in a cycle receives a player they ranked higher than what they gave up
    - No team appears in more than one cycle
    """)

    # Build team → cycle index and team → received player_id lookups
    team_to_cycle_idx = {}
    team_to_received_pid = {}
    for ci, cycle in enumerate(cycles):
        for trade in cycle["trades"]:
            team_to_cycle_idx[trade["team"]] = ci
            team_to_received_pid[trade["team"]] = trade["receives"]["player_id"]

    if cycles:
        ci = 0
        cycle = cycles[0]
        teams = cycle["cycle"]
        cycle_label = f"Example Cycle:  {' → '.join(teams)} → {teams[0]}  ({len(teams)}-team trade)"

        with st.expander(cycle_label, expanded=True):

            # Visual gives/receives display
            st.markdown("####")
            cols = st.columns(len(teams))
            for col, trade in zip(cols, cycle["trades"]):
                with col:
                    next_team = teams[(teams.index(trade["team"]) + 1) % len(teams)]
                    st.markdown(f"""
                    <div style='background:#1e1e2e;border-radius:10px;padding:14px;text-align:center;border:1px solid #444'>
                        <div style='font-size:18px;font-weight:bold;color:#e0e0ff'>{trade['team']}</div>
                        <div style='font-size:11px;color:#aaa;margin:6px 0'>gives ↓</div>
                        <div style='font-size:13px;color:#ff9999'>{trade['gives']['player_name']}</div>
                        <div style='font-size:10px;color:#888'>{trade['gives']['primary_role']}</div>
                        <div style='font-size:11px;color:#aaa;margin:6px 0'>receives ↓</div>
                        <div style='font-size:13px;color:#99ffcc'>{trade['receives']['player_name']}</div>
                        <div style='font-size:10px;color:#888'>{trade['receives']['primary_role']}</div>
                        <div style='font-size:10px;color:#666;margin-top:6px'>from {next_team}</div>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("---")

            # Dropdown: team preference ranking
            selected_pref_team = st.selectbox(
                "View preference ranking for a team in this cycle",
                teams,
                key="step5_cycle_0",
            )

            selected_pid = team_to_received_pid.get(selected_pref_team)
            gives_info = next((t["gives"] for t in cycle["trades"] if t["team"] == selected_pref_team), None)
            gives_pid = gives_info["player_id"] if gives_info else None

            urgency_lookup_s5 = {role: max(u, 0) for role, u in gaps.get(selected_pref_team, [])}

            # Score every player in the league
            all_scored = []
            for _, row in classified.iterrows():
                is_own = row["team_abbr"] == selected_pref_team
                role_urgency = urgency_lookup_s5.get(row["primary_role"], 0)
                if row.get("secondary_role"):
                    sec_urgency = urgency_lookup_s5.get(row["secondary_role"], 0)
                    role_urgency = max(role_urgency, 0.5 * sec_urgency)
                if role_urgency == 0:
                    q = quality_score(row)
                    role_urgency = 0.05 + 0.15 * max(0, q / 30.0)
                score = round(role_urgency * quality_score(row), 2)
                all_scored.append((int(row["player_id"]), score, is_own))

            all_scored.sort(key=lambda x: x[1], reverse=True)

            pref_rows_s5 = []
            for rank, (pid, score, is_own) in enumerate(all_scored, 1):
                player = classified[classified["player_id"] == pid]
                if player.empty:
                    continue
                p = player.iloc[0]
                if pid == selected_pid:
                    status = "Receives"
                elif pid == gives_pid:
                    status = "Gives"
                else:
                    status = ""
                pref_rows_s5.append({
                    "Rank": rank,
                    "Player": p["player_name"],
                    "From": p["team_abbr"],
                    "Role": p["primary_role"],
                    "Preference Score": score,
                    "Status": status,
                    "_own": is_own,
                })

            pref_df_s5 = pd.DataFrame(pref_rows_s5).drop(columns=["_own"])
            pref_df_s5["Preference Score"] = pref_df_s5["Preference Score"].apply(lambda x: round(x, 2) if isinstance(x, float) else x)

            st.markdown(
                "🟢 **Receives** — player this team acquires in the trade &nbsp;&nbsp; 🔴 **Gives** — player this team sends away",
                unsafe_allow_html=True,
            )

            def highlight_s5(row):
                if row["Status"] == "Receives":
                    return ["background-color: #1a4731; color: white"] * len(row)
                if row["Status"] == "Gives":
                    return ["background-color: #4a1a1a; color: white"] * len(row)
                return [""] * len(row)

            st.dataframe(
                pref_df_s5.style.apply(highlight_s5, axis=1),
                use_container_width=True,
                hide_index=True,
            )

    st.markdown("---")

    st.subheader("Step 6 — Trade Narratives (AI-Generated)")
    st.markdown("""
    - For every completed cycle, the app generates a plain-English explanation of each trade
    - Each narrative explains **who the team gives up**, **who they receive**, and **why it fills their gap**
    - Narratives are generated by **GPT-4o** — given the trade details, it writes an ESPN-style analyst breakdown
    - The tradeoff: AI narratives are more natural and varied, but require an API key (enter yours in the sidebar)
    """)

    st.markdown("**How it works:**")
    st.markdown("""
    1. For each completed cycle, the app passes the full trade details to GPT-4o — each team, who they give, who they receive, and the players' stats and roles
    2. GPT-4o is prompted to write a 3-4 sentence analyst breakdown explaining the roster fit logic for each team
    3. The result is cached for the session so the API is only called once per cycle
    """)

    if explanations and cycles:
        cache_key_s6 = "step6_ai_narrative"
        if openai_api_key:
            if cache_key_s6 not in st.session_state:
                with st.spinner("Generating AI narrative..."):
                    try:
                        st.session_state[cache_key_s6] = generate_ai_narrative(cycles[0], openai_api_key)
                    except Exception as e:
                        st.session_state[cache_key_s6] = f"⚠️ Error: {e}"
            st.markdown("**Live AI-generated narrative — first trade cycle:**")
            st.markdown(st.session_state[cache_key_s6])
        else:
            st.info("Enter your OpenAI API key in the sidebar to see a live AI-generated narrative here.")

    st.markdown("---")

    st.subheader("Step 7 — Explore the Results")
    st.markdown("Use the tabs at the top of the page to explore the full output:")

    st.markdown("""
    | Tab | What it shows |
    |---|---|
    | **Dashboard** | High-level summary — total players, teams, cycles found, and a visual network graph of all trades |
    | **Team Explorer** | Select any team to see their full roster, role composition, gaps, and top trade targets |
    | **Trade Cycles** | Every trade cycle in detail — who each team gives and receives, with narrative explanations |
    | **AI Evaluation** | Honest assessment of where the algorithm works well and where it falls short |
    """)

    st.markdown("---")

# --- AI EVALUATION TAB ---
with tab_eval:
    st.header("AI Evaluation: Where It Works and Where It Fails")

    st.markdown("""
    This project uses AI and algorithmic methods at multiple stages. Here we evaluate
    where these automated approaches add value and where they fall short.
    """)

    st.subheader("1. Role Classification (Rule-Based AI)")
    st.markdown("""
    **What it does**: Assigns each player an archetype (Floor Spacer, Rim Protector, etc.)
    based on statistical thresholds.

    **Where it works well**:
    - Clear archetypes are identified accurately (elite shooters → Floor Spacer, shot blockers → Rim Protector)
    - Transparent and explainable — you can see exactly why each classification was made
    - Fast to iterate on thresholds when results don't pass the eye test

    **Where it fails**:
    - Players who don't fit neat archetypes get classified as "Rotation Player" (~50% of the league)
    - Can't capture playstyle nuance from tracking data (e.g., a player who doesn't score much but spaces the floor by gravity)
    - Thresholds are set manually, not learned from data — a more sophisticated approach would use clustering or ML
    """)

    # Show some example classifications for discussion
    st.markdown("**Sample Classifications (verify these pass the eye test):**")
    sample_players = classified.nlargest(20, "pts")[
        ["player_name", "team_abbr", "primary_role", "secondary_role", "pts", "reb", "ast"]
    ]
    st.dataframe(sample_players.reset_index(drop=True), use_container_width=True, hide_index=True)

    st.subheader("2. Preference Simulation")
    st.markdown("""
    **What it does**: Automatically generates team preferences by identifying roster gaps
    and ranking available players by how well they fill those gaps.

    **Where it works well**:
    - Correctly identifies obvious needs (team with no rim protector values shot blockers)
    - Team-specific ideal rosters (adjusted by aggregate stats) create diverse preferences — teams weak in scoring prioritize offensive roles, teams weak in defense prioritize defensive roles
    - Quality-scaled Rotation Player scoring ensures meaningful differentiation among bench players

    **Where it fails**:
    - Real GMs consider factors we don't model: chemistry, age, contract, injury history, scheme fit
    - Team-specific adjustments are based on aggregate stats, not coaching philosophy or front-office strategy
    - The urgency scoring is simplistic (linear gap from ideal) — real trade value is nonlinear
    """)

    st.subheader("3. Trade Narratives (Template-Based)")
    st.markdown("""
    **What it does**: Generates human-readable explanations of each trade using
    structured templates filled with trade data.

    **Design choice**: We chose template-based over LLM-generated narratives.

    | | Template-Based (our approach) | LLM-Generated |
    |---|---|---|
    | **Accuracy** | Always faithful to data | Risk of hallucination |
    | **Variety** | Limited by template count | Highly varied |
    | **Nuance** | Mechanical, formulaic | Can capture subtle logic |
    | **Cost** | Free, no API key | Requires paid API |
    | **Reliability** | Always works | API can be down |

    This is a genuine tradeoff — for a demo prototype where accuracy matters more than
    literary quality, templates are the right call.
    """)

    st.subheader("4. TTC Algorithm (Multi-Edge Variant)")
    st.markdown("""
    **What it does**: Builds a multi-edge preference graph (each team has edges to their top 5 preferred teams) and finds stable trading cycles up to 6 teams, prioritizing longer cycles.

    **Where it works well**:
    - Guarantees individual rationality — no team would undo their trade
    - Multi-edge graph + longest-first cycle selection reliably surfaces multi-team deals (5-6 teams each)
    - Computationally efficient (runs in milliseconds for 30 teams)

    **Where it fails**:
    - Does not guarantee global optimality — a different configuration might create more total value
    - Single-player-per-team constraint limits trade complexity (real NBA trades involve multiple players + picks)
    - Longest-first cycle prioritization means some teams may get a slightly less-preferred outcome than strict classical TTC

    **Key iteration**: Our initial single-pointer graph produced almost exclusively 2-team swaps because 17 of 30 teams pointed at the same target (SAC). The multi-edge approach was critical to unlocking multi-party trades.
    """)
