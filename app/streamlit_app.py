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

from src.data.fetch_rosters import fetch_player_stats
from src.roles.classifier import classify_all_players
from src.preferences.gap_analysis import analyze_team_gaps
from src.preferences.ranker import generate_preferences, identify_available_players
from src.matching.ttc import run_ttc
from src.narrative.explainer import explain_all_cycles
from src.config import IDEAL_ROSTER_COMPOSITION

# --- Page config ---
st.set_page_config(
    page_title="NBA Trade Matcher — TTC",
    page_icon="🏀",
    layout="wide",
)

# --- Sidebar ---
st.sidebar.title("NBA Trade Matcher")
st.sidebar.markdown("**Top Trading Cycles Algorithm**")
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
tab_dashboard, tab_teams, tab_trades, tab_eval = st.tabs([
    "Dashboard", "Team Explorer", "Trade Cycles", "AI Evaluation"
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

        for cycle, explanation in sorted_cycles:
            teams = cycle["cycle"]
            label = f"{'⭐ ' if cycle['num_teams'] > 2 else ''}{cycle['num_teams']}-Team Trade: {' ↔ '.join(teams)}"

            with st.expander(label, expanded=cycle["num_teams"] > 2):
                st.markdown(f"*{explanation['summary']}*")
                st.markdown("---")

                for trade, detail in zip(cycle["trades"], explanation["team_details"]):
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

                    st.markdown(f"> {detail}")
                    st.markdown("")

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
