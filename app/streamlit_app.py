"""
NBA Trade Matcher — Top Trading Cycles
Streamlit web UI for the OIT 277 group project.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
import numpy as np
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
    trades = cycle["trades"]
    lines = []
    for t in trades:
        g, r = t["gives"], t["receives"]
        lines.append(
            f"- {t['team']} sends {g['player_name']} ({g['primary_role']}, "
            f"{g['pts']} PPG/{g['reb']} RPG/{g['ast']} APG) and receives "
            f"{r['player_name']} ({r['primary_role']}, {r['pts']} PPG/{r['reb']} RPG/{r['ast']} APG)"
        )
    n = len(trades)
    prompt = (
        f"You are an NBA analyst. Explain in 3-4 sentences why this {n}-team trade makes sense "
        f"for each team. Be specific about roster fit, what each team gains, and why they give up "
        f"what they do. Keep it conversational and insightful — like an ESPN breakdown.\n\n"
        f"Trade details:\n" + "\n".join(lines)
    )
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()


def quality_score(row):
    return row["pts"] * 1.0 + row["reb"] * 0.8 + row["ast"] * 1.0 + row["stl"] * 2.0 + row["blk"] * 2.0 - row["tov"] * 1.5


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="NBA Trade Matcher", page_icon="🏀", layout="wide")

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem !important; padding-bottom: 2rem !important; }

/* Background */
[data-testid="stAppViewContainer"] { background: #0B0E14; }
.main .block-container { background: #0B0E14; }

/* Sidebar */
[data-testid="stSidebar"] { background: #0E1118 !important; border-right: 1px solid #1A2235; }
[data-testid="stSidebar"] .stMarkdown p { color: #94A3B8; font-size: 13px; }
[data-testid="stSidebar"] label { color: #94A3B8 !important; font-size: 12px !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #0E1118;
    border-radius: 10px;
    padding: 5px;
    gap: 3px;
    border: 1px solid #1A2235;
    margin-bottom: 24px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 7px;
    color: #94A3B8;
    font-weight: 600;
    font-size: 12px;
    letter-spacing: 0.6px;
    text-transform: uppercase;
    padding: 9px 20px;
    background: transparent;
    border: none;
}
.stTabs [aria-selected="true"] {
    background: #C89B3C !important;
    color: #0B0E14 !important;
}
.stTabs [data-baseweb="tab"]:hover { color: #C89B3C !important; }

/* Metrics */
[data-testid="metric-container"] {
    background: #0E1118;
    border: 1px solid #1A2235;
    border-radius: 12px;
    padding: 18px 22px;
}
[data-testid="stMetricLabel"] p {
    color: #94A3B8 !important;
    font-size: 11px !important;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    font-weight: 600;
}
[data-testid="stMetricValue"] {
    color: #F1F5F9 !important;
    font-size: 32px !important;
    font-weight: 800 !important;
}

/* Expanders */
[data-testid="stExpander"] {
    background: #0E1118 !important;
    border: 1px solid #1A2235 !important;
    border-radius: 12px !important;
    margin-bottom: 12px !important;
}
[data-testid="stExpander"] summary {
    background: #D1D5DB !important;
    color: #111827 !important;
    font-weight: 700 !important;
    font-size: 14px !important;
    border-radius: 10px !important;
    padding: 12px 16px !important;
}

/* Info/warning boxes */
[data-testid="stInfo"] {
    background: #0C1828 !important;
    border: 1px solid #1C3A5F !important;
    border-radius: 8px !important;
    color: #7BAFD4 !important;
}

/* Dividers */
hr { border-color: #1A2235 !important; margin: 20px 0 !important; }

/* Headings */
h1 { color: #F1F5F9 !important; font-weight: 800 !important; letter-spacing: -0.5px; }
h2 { color: #E2E8F0 !important; font-weight: 700 !important; }
h3 { color: #CBD5E1 !important; font-weight: 600 !important; }
p { color: #CBD5E1; }

/* Body text — markdown bullets and paragraphs */
.stMarkdown p, .stMarkdown li { color: #CBD5E1 !important; font-size: 14px; line-height: 1.7; }
.stMarkdown strong { color: #F1F5F9 !important; }
.stMarkdown em { color: #94A3B8 !important; }

/* Selectbox */
[data-testid="stSelectbox"] label { color: #94A3B8 !important; font-size: 12px !important; }

/* Caption */
[data-testid="stCaptionContainer"] p { color: #64748B !important; font-size: 12px !important; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.markdown("""
<div style='padding:12px 0 20px;'>
  <div style='font-size:20px;font-weight:800;color:#C89B3C;letter-spacing:-0.3px;'>NBA Trade Matcher</div>
  <div style='font-size:10px;color:#64748B;text-transform:uppercase;letter-spacing:2px;margin-top:5px;'>Top Trading Cycles · OIT 277</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("<div style='font-size:11px;color:#94A3B8;text-transform:uppercase;letter-spacing:1px;margin-bottom:10px;'>Parameters</div>", unsafe_allow_html=True)

min_minutes = st.sidebar.slider(
    "Min. minutes per game",
    min_value=5.0, max_value=20.0, value=10.0, step=1.0,
    help="Exclude players below this playing time threshold",
)

st.sidebar.markdown("---")
_default_key = st.secrets.get("OPENAI_API_KEY", "") if hasattr(st, "secrets") else ""
openai_api_key = st.sidebar.text_input(
    "OpenAI API Key",
    value=_default_key,
    type="password",
    placeholder="sk-...",
    help="Required for AI-generated trade narratives",
)

# ── Pipeline ──────────────────────────────────────────────────────────────────
@st.cache_data
def load_player_data():
    return fetch_player_stats()

@st.cache_data
def run_pipeline(min_min: float):
    df = load_player_data()
    df = df[df["min"] >= min_min].reset_index(drop=True)
    classified = classify_all_players(df)
    gaps = analyze_team_gaps(classified)
    avail_df = classified.copy()
    avail_df["available_for_trade"] = True
    prefs = generate_preferences(classified, gaps, players_per_team=len(classified))
    cycles = run_ttc(classified, prefs, avail_df, gaps)
    explanations = explain_all_cycles(cycles, gaps)
    return classified, gaps, prefs, avail_df, cycles, explanations

with st.spinner("Loading trade intelligence..."):
    classified, gaps, prefs, avail_df, cycles, explanations = run_pipeline(min_minutes)

multi_count = sum(1 for c in cycles if c["num_teams"] > 2)
st.sidebar.markdown("---")
st.sidebar.markdown(
    f"<div style='font-size:11px;color:#64748B;text-transform:uppercase;letter-spacing:1px;margin-bottom:10px;'>Current Run</div>"
    f"<div style='font-size:13px;line-height:2;color:#6B7A99;'>"
    f"<span style='color:#C89B3C;font-weight:700;'>{len(classified)}</span> players analyzed<br>"
    f"<span style='color:#C89B3C;font-weight:700;'>{len(cycles)}</span> trade cycles found<br>"
    f"<span style='color:#C89B3C;font-weight:700;'>{multi_count}</span> multi-team deals</div>",
    unsafe_allow_html=True,
)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_overview, tab_dashboard, tab_teams, tab_trades, tab_eval = st.tabs([
    "How It Works", "Dashboard", "Team Explorer", "Trade Cycles", "AI Evaluation",
])


# ══════════════════════════════════════════════════════════════════════════════
# DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
with tab_dashboard:
    st.markdown("<h1>League Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#64748B;margin-top:-8px;margin-bottom:28px;'>2024–25 NBA Season</p>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Players Analyzed", len(classified))
    with c2: st.metric("Teams", classified["team_abbr"].nunique())
    with c3: st.metric("Trade Cycles", len(cycles))
    with c4: st.metric("Multi-Team Deals", multi_count)

    st.markdown("---")

    col_graph, col_roles = st.columns([3, 2])

    with col_graph:
        st.markdown("<h3 style='color:#CBD5E1;margin-bottom:16px;'>Trade Network</h3>", unsafe_allow_html=True)
        if cycles:
            CYCLE_COLORS = ["#C89B3C", "#22C55E", "#60A5FA", "#F87171", "#A78BFA", "#FB923C", "#34D399"]
            plt.style.use("dark_background")
            fig, ax = plt.subplots(figsize=(10, 7))
            fig.patch.set_facecolor("#0E1118")
            ax.set_facecolor("#0E1118")

            G = nx.DiGraph()
            edge_colors = []
            for i, cycle in enumerate(cycles):
                color = CYCLE_COLORS[i % len(CYCLE_COLORS)]
                teams = cycle["cycle"]
                for j, team in enumerate(teams):
                    nt = teams[(j + 1) % len(teams)]
                    G.add_edge(team, nt, col=color)
                    G.nodes[team]["color"] = color

            pos = nx.spring_layout(G, k=2.5, seed=42)
            node_colors = [G.nodes[n].get("color", "#3A4560") for n in G.nodes()]
            edge_colors = [G[u][v]["col"] for u, v in G.edges()]

            nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_colors,
                                   arrows=True, arrowsize=18, width=2, alpha=0.8,
                                   connectionstyle="arc3,rad=0.08")
            nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                                   node_size=1100, alpha=1.0)
            nx.draw_networkx_labels(G, pos, ax=ax, font_size=8,
                                    font_weight="bold", font_color="#0B0E14")
            ax.axis("off")
            fig.tight_layout(pad=0)
            st.pyplot(fig)
            plt.close()

            st.markdown("<div style='font-size:11px;color:#64748B;text-transform:uppercase;letter-spacing:1px;margin-top:16px;margin-bottom:8px;'>Each color = one trade cycle</div>", unsafe_allow_html=True)
            legend_html = "<div style='display:flex;flex-direction:column;gap:6px;'>"
            for i, cycle in enumerate(cycles):
                color = CYCLE_COLORS[i % len(CYCLE_COLORS)]
                teams_str = " → ".join(cycle["cycle"])
                legend_html += (
                    f"<div style='display:flex;align-items:center;gap:10px;'>"
                    f"<span style='width:12px;height:12px;border-radius:50%;background:{color};flex-shrink:0;display:inline-block;'></span>"
                    f"<span style='font-size:12px;color:#CBD5E1;font-weight:600;'>{teams_str}</span>"
                    f"<span style='font-size:11px;color:#64748B;'>({cycle['num_teams']}-team)</span>"
                    f"</div>"
                )
            legend_html += "</div>"
            st.markdown(legend_html, unsafe_allow_html=True)

    with col_roles:
        st.markdown("<h3 style='color:#CBD5E1;margin-bottom:16px;'>Role Distribution</h3>", unsafe_allow_html=True)
        role_counts = classified["primary_role"].value_counts()
        max_count = role_counts.max()
        bars = "<div style='display:flex;flex-direction:column;gap:10px;'>"
        for role, count in role_counts.items():
            pct = int(count / max_count * 100)
            bars += f"""
            <div>
              <div style='display:flex;justify-content:space-between;margin-bottom:4px;'>
                <span style='font-size:12px;color:#94A3B8;'>{role}</span>
                <span style='font-size:12px;color:#94A3B8;font-weight:600;'>{count}</span>
              </div>
              <div style='background:#131926;border-radius:4px;height:6px;'>
                <div style='background:#C89B3C;width:{pct}%;height:6px;border-radius:4px;'></div>
              </div>
            </div>"""
        bars += "</div>"
        st.markdown(bars, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TEAM EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
with tab_teams:
    st.markdown("<h1>Team Explorer</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#64748B;margin-top:-8px;margin-bottom:24px;'>Roster breakdown, role composition, and trade targets for any team</p>", unsafe_allow_html=True)

    team_list = sorted(classified["team_abbr"].unique())
    selected_team = st.selectbox("Select a team", team_list)

    if selected_team:
        team_roster = classified[classified["team_abbr"] == selected_team].copy()

        st.markdown(f"<h3 style='color:#C89B3C;border-bottom:1px solid #1A2235;padding-bottom:8px;margin-top:8px;'>{selected_team} Roster</h3>", unsafe_allow_html=True)
        display_cols = ["player_name", "primary_role", "secondary_role", "pts", "reb", "ast", "stl", "blk", "min"]
        roster_display = team_roster[display_cols].sort_values("min", ascending=False).reset_index(drop=True)
        st.dataframe(roster_display, use_container_width=True, hide_index=True)

        st.markdown("<h3 style='color:#C89B3C;border-bottom:1px solid #1A2235;padding-bottom:8px;margin-top:24px;'>Role Composition</h3>", unsafe_allow_html=True)
        team_roles = team_roster["primary_role"].value_counts()
        max_r = team_roles.max()
        rc = "<div style='display:flex;flex-direction:column;gap:8px;'>"
        for role, count in team_roles.items():
            pct = int(count / max_r * 100)
            rc += f"""
            <div>
              <div style='display:flex;justify-content:space-between;margin-bottom:3px;'>
                <span style='font-size:13px;color:#94A3B8;'>{role}</span>
                <span style='font-size:12px;color:#94A3B8;font-weight:600;'>{count}</span>
              </div>
              <div style='background:#131926;border-radius:4px;height:7px;'>
                <div style='background:#C89B3C;width:{pct}%;height:7px;border-radius:4px;'></div>
              </div>
            </div>"""
        rc += "</div>"
        st.markdown(rc, unsafe_allow_html=True)

        st.markdown(f"<h3 style='color:#C89B3C;border-bottom:1px solid #1A2235;padding-bottom:8px;margin-top:28px;'>Preference Ranking — Players {selected_team} Would Most Want</h3>", unsafe_allow_html=True)
        st.caption("Ranked by role urgency × quality score. 🟡 Own-team players highlighted — any trade must land a player ranked above these.")

        urgency_t = {role: max(u, 0) for role, u in gaps.get(selected_team, [])}
        pref_rows_t = []
        for _, row in classified.iterrows():
            is_own = row["team_abbr"] == selected_team
            role_urgency = urgency_t.get(row["primary_role"], 0)
            if row.get("secondary_role"):
                role_urgency = max(role_urgency, 0.5 * urgency_t.get(row["secondary_role"], 0))
            if role_urgency == 0:
                q = quality_score(row)
                role_urgency = 0.05 + 0.15 * max(0, q / 30.0)
            pref_rows_t.append({
                "Player": row["player_name"],
                "Team": row["team_abbr"],
                "Role": row["primary_role"],
                "PTS": round(row["pts"], 1),
                "REB": round(row["reb"], 1),
                "AST": round(row["ast"], 1),
                "Pref Score": round(role_urgency * quality_score(row), 2),
                "_own": is_own,
            })
        pref_rows_t.sort(key=lambda x: x["Pref Score"], reverse=True)
        pref_df_t = pd.DataFrame(pref_rows_t).drop(columns=["_own"])

        def highlight_own_t(row):
            if row["Team"] == selected_team:
                return ["background-color: #2E2A0A; color: #D4C44A"] * len(row)
            return [""] * len(row)

        st.dataframe(pref_df_t.style.apply(highlight_own_t, axis=1), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TRADE CYCLES
# ══════════════════════════════════════════════════════════════════════════════
with tab_trades:
    st.markdown("<h1>Trade Cycles</h1>", unsafe_allow_html=True)

    if not cycles:
        st.info("No trade cycles found. Try lowering the minimum minutes threshold.")
    else:
        two_team = sum(1 for c in cycles if c["num_teams"] == 2)
        st.markdown(
            f"<p style='color:#6B7A99;margin-top:-8px;margin-bottom:24px;'>"
            f"<span style='color:#C89B3C;font-weight:700;'>{len(cycles)}</span> trade opportunities — "
            f"<span style='color:#C89B3C;font-weight:700;'>{multi_count}</span> multi-team, "
            f"<span style='color:#C89B3C;font-weight:700;'>{two_team}</span> bilateral</p>",
            unsafe_allow_html=True,
        )

        if not openai_api_key:
            st.info("Add your OpenAI key in the sidebar to unlock AI-generated trade analysis.")

        sorted_cycles = sorted(zip(cycles, explanations), key=lambda x: x[0]["num_teams"], reverse=True)

        for i, (cycle, explanation) in enumerate(sorted_cycles):
            teams = cycle["cycle"]
            n_teams = cycle["num_teams"]
            star = "⭐  " if n_teams > 2 else ""
            label = f"{star}{n_teams}-Team Trade  ·  {' → '.join(teams)}"

            with st.expander(label, expanded=(n_teams > 2)):

                # Player cards
                cols = st.columns(n_teams)
                for col, trade in zip(cols, cycle["trades"]):
                    g, r = trade["gives"], trade["receives"]
                    with col:
                        st.markdown(f"""
                        <div style='background:#0E1118;border:1px solid #1A2235;border-radius:14px;padding:18px 14px;text-align:center;'>
                          <div style='font-size:17px;font-weight:800;color:#C89B3C;letter-spacing:1px;margin-bottom:16px;'>{trade['team']}</div>
                          <div style='font-size:9px;color:#64748B;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:5px;'>Sends</div>
                          <div style='font-size:14px;font-weight:700;color:#EF4444;'>{g['player_name']}</div>
                          <div style='font-size:10px;color:#94A3B8;margin-top:3px;margin-bottom:14px;'>{g['primary_role']} &nbsp;·&nbsp; {g['pts']} / {g['reb']} / {g['ast']}</div>
                          <div style='border-top:1px solid #1A2235;margin:0 auto 14px;width:60%;'></div>
                          <div style='font-size:9px;color:#64748B;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:5px;'>Receives</div>
                          <div style='font-size:14px;font-weight:700;color:#22C55E;'>{r['player_name']}</div>
                          <div style='font-size:10px;color:#94A3B8;margin-top:3px;'>{r['primary_role']} &nbsp;·&nbsp; {r['pts']} / {r['reb']} / {r['ast']}</div>
                        </div>
                        """, unsafe_allow_html=True)

                st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

                # AI narrative
                cache_key = f"ai_narrative_{i}"
                if openai_api_key:
                    if cache_key not in st.session_state:
                        with st.spinner("Generating analysis..."):
                            try:
                                st.session_state[cache_key] = generate_ai_narrative(cycle, openai_api_key)
                            except Exception as e:
                                st.session_state[cache_key] = f"⚠️ {e}"
                    st.markdown(
                        f"<div style='background:#0C1828;border-left:3px solid #C89B3C;border-radius:0 8px 8px 0;"
                        f"padding:14px 18px;font-size:14px;line-height:1.8;color:#CBD5E1;font-style:italic;'>"
                        f"{st.session_state[cache_key]}</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"<div style='background:#0C1828;border-left:3px solid #C89B3C;border-radius:0 8px 8px 0;"
                        f"padding:14px 18px;font-size:14px;line-height:1.8;color:#CBD5E1;font-style:italic;'>"
                        f"{explanation['summary']}</div>",
                        unsafe_allow_html=True,
                    )


# ══════════════════════════════════════════════════════════════════════════════
# HOW IT WORKS (App Overview)
# ══════════════════════════════════════════════════════════════════════════════
with tab_overview:
    st.markdown("<h1>How It Works</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#64748B;margin-top:-8px;margin-bottom:24px;'>A 6-stage data pipeline that surfaces multi-team NBA trade opportunities.</p>", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("<h3 style='color:#C89B3C;'>Step 1 — Data Collection</h3>", unsafe_allow_html=True)
    st.markdown("""
- Pulls live per-game stats for every NBA player from NBA.com
- Filters out players averaging **<10** minutes per game
- All remaining players are eligible to be traded — no artificial trade block restriction
- Result: ~458 players across all 30 teams, all available
""")
    team_list_overview = sorted(classified["team_abbr"].unique())
    selected_team_overview = st.selectbox("Select a team to explore its data", team_list_overview,
                                           index=team_list_overview.index("TOR"), key="overview_team")
    team_data = classified[classified["team_abbr"] == selected_team_overview][
        ["player_name", "pts", "reb", "ast", "stl", "blk", "min", "gp"]
    ].sort_values("min", ascending=False).reset_index(drop=True)
    st.caption(f"{selected_team_overview} — {len(team_data)} players in dataset")
    st.dataframe(team_data, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("<h3 style='color:#C89B3C;'>Step 2 — Role Classification</h3>", unsafe_allow_html=True)
    st.markdown("""
- Each player is assigned one of 8 archetypes based on their per-game stats
- Players must meet **all** stat thresholds for a role — one standout stat isn't enough
- Players who don't fit any archetype are labeled **Rotation Player**
""")
    rubric_data = {
        "Role": ["Floor Spacer", "Rim Protector", "Playmaker", "Paint Scorer",
                 "Two-Way Wing", "Stretch Big", "Defensive Anchor", "Volume Scorer"],
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
    st.dataframe(
        pd.DataFrame(rubric_data).style
          .set_properties(**{"text-align": "center"})
          .set_properties(subset=["Role"], **{"text-align": "left"}),
        use_container_width=True, hide_index=True,
    )

    st.markdown("**Example player by role:**")
    ALL_COLS = ["PTS", "REB", "AST", "BLK", "STL", "3PA", "3P%", "FG%", "FGA", "AST/TOV", "STL+BLK", "MIN"]
    example_players = {
        "Floor Spacer":      {"player": "Steph Curry (GSW)",              "standard": {"3PA": "5", "3P%": "36%"},                    "performance": {"3PA": "12", "3P%": "40%"}},
        "Rim Protector":     {"player": "Rudy Gobert (MIN)",              "standard": {"REB": "7", "BLK": "1"},                      "performance": {"REB": "12", "BLK": "2"}},
        "Playmaker":         {"player": "LeBron James (LAL)",             "standard": {"AST": "6", "AST/TOV": "2"},                  "performance": {"AST": "8", "AST/TOV": "2"}},
        "Paint Scorer":      {"player": "Giannis Antetokounmpo (MIL)",    "standard": {"PTS": "12", "REB": "6", "FG%": "54%"},       "performance": {"PTS": "28", "REB": "11", "FG%": "58%"}},
        "Two-Way Wing":      {"player": "Kawhi Leonard (LAC)",            "standard": {"PTS": "14", "REB": "4", "STL": "1"},         "performance": {"PTS": "24", "REB": "7", "STL": "2"}},
        "Stretch Big":       {"player": "Kevin Durant (PHX)",             "standard": {"REB": "7", "3PA": "2", "3P%": "32%"},        "performance": {"REB": "7", "3PA": "5", "3P%": "38%"}},
        "Defensive Anchor":  {"player": "Draymond Green (GSW)",           "standard": {"STL+BLK": "2", "MIN": "27"},                 "performance": {"STL+BLK": "2", "MIN": "30"}},
        "Volume Scorer":     {"player": "Luka Dončić (DAL)",              "standard": {"PTS": "22", "FGA": "17"},                    "performance": {"PTS": "33", "FGA": "22"}},
        "Rotation Player":   None,
    }
    selected_role = st.selectbox("Select a role", list(example_players.keys()), key="overview_role")
    if example_players[selected_role] is None:
        st.info("Rotation Player is the default for players who don't meet any role's thresholds.")
    else:
        p = example_players[selected_role]
        st.caption(f"Example: {p['player']} — {selected_role}")
        def build_row(label, stats):
            return {"": label, **{col: stats.get(col, "") for col in ALL_COLS}}
        st.dataframe(
            pd.DataFrame([build_row("Threshold", p["standard"]), build_row("Actual", p["performance"])])
              .style.set_properties(**{"text-align": "center"}).set_properties(subset=[""], **{"text-align": "left"}),
            use_container_width=True, hide_index=True,
        )

    st.markdown("---")
    st.markdown("<h3 style='color:#C89B3C;'>Step 3 — Identify Team Needs</h3>", unsafe_allow_html=True)
    st.markdown("""
- Every team starts with the same baseline role targets
- Targets are adjusted **up** for roles that address each team's statistical weaknesses
- **Urgency score** = (target − actual) / target — higher means a bigger need
""")
    _team_list_s3 = sorted(classified["team_abbr"].unique())
    selected_team_step3 = st.selectbox("Select a team", _team_list_s3,
                                        index=_team_list_s3.index("TOR"), key="step3_team")

    st.markdown("**a) Baseline targets — same for every team:**")
    st.dataframe(pd.DataFrame([{"Role": r, "Base Target": c} for r, c in IDEAL_ROSTER_COMPOSITION.items()]),
                 use_container_width=True, hide_index=True)

    st.markdown("**b) Team stats vs. league median:**")
    league_profiles = {}
    for team, roster in classified.groupby("team_abbr"):
        league_profiles[team] = {
            "Scoring (PTS)":     roster["pts"].mean(),
            "Rebounding (REB)":  roster["reb"].mean(),
            "Playmaking (AST)":  roster["ast"].mean(),
            "Defense (STL+BLK)": (roster["stl"] + roster["blk"]).mean(),
            "Shooting (3P%)":    roster["fg3_pct"].mean(),
        }
    medians = {s: np.median([p[s] for p in league_profiles.values()]) for s in list(league_profiles.values())[0]}
    team_profile = league_profiles[selected_team_step3]
    adjustment_map = {
        "Scoring (PTS)":     "+1 Volume Scorer, +1 Floor Spacer",
        "Rebounding (REB)":  "+1 Rim Protector, +1 Paint Scorer",
        "Playmaking (AST)":  "+1 Playmaker",
        "Defense (STL+BLK)": "+1 Defensive Anchor, +1 Two-Way Wing",
        "Shooting (3P%)":    "+1 Stretch Big, +1 Floor Spacer",
    }
    st.dataframe(pd.DataFrame([{
        "Stat": s, "Team Value": round(v, 2), "League Median": round(medians[s], 2),
        "Below Median?": "Yes" if v < medians[s] else "No",
        "Adjustment": adjustment_map[s] if v < medians[s] else "—",
    } for s, v in team_profile.items()]), use_container_width=True, hide_index=True)

    st.markdown("**c) Adjusted targets:**")
    adjusted = dict(IDEAL_ROSTER_COMPOSITION)
    stat_to_role_bumps = {
        "Scoring (PTS)":     ["Volume Scorer", "Floor Spacer"],
        "Rebounding (REB)":  ["Rim Protector", "Paint Scorer"],
        "Playmaking (AST)":  ["Playmaker"],
        "Defense (STL+BLK)": ["Defensive Anchor", "Two-Way Wing"],
        "Shooting (3P%)":    ["Stretch Big", "Floor Spacer"],
    }
    for s, v in team_profile.items():
        if v < medians[s]:
            for role in stat_to_role_bumps[s]:
                adjusted[role] = adjusted.get(role, 0) + 1
    st.dataframe(pd.DataFrame([{
        "Role": role, "Base Target": base,
        "Adjustment": f"+{adjusted[role] - base}" if adjusted[role] > base else "—",
        "Final Target": adjusted[role],
    } for role, base in IDEAL_ROSTER_COMPOSITION.items()]), use_container_width=True, hide_index=True)

    st.markdown("**d) Urgency gaps:**")
    team_roster_s3 = classified[classified["team_abbr"] == selected_team_step3]
    gap_rows = []
    for role, target in adjusted.items():
        actual = ((team_roster_s3["primary_role"] == role).sum()
                  + 0.5 * (team_roster_s3["secondary_role"] == role).sum())
        gap_rows.append({"Role": role, "Target": target,
                         "Actual": round(actual, 1), "Urgency Score": round((target - actual) / target, 2)})
    gap_rows.sort(key=lambda x: x["Urgency Score"], reverse=True)
    st.dataframe(pd.DataFrame(gap_rows), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("<h3 style='color:#C89B3C;'>Step 4 — Preference Ranking</h3>", unsafe_allow_html=True)
    st.markdown("""
- Every player from every other team is ranked by how well they fill the team's gaps
- **Preference score** = role urgency × quality score
- **Quality score** = PTS×1.0 + REB×0.8 + AST×1.0 + STL×2.0 + BLK×2.0 − TOV×1.5
""")
    _team_list_s4 = sorted(classified["team_abbr"].unique())
    selected_team_step4 = st.selectbox("Select a team", _team_list_s4,
                                        index=_team_list_s4.index("TOR"), key="step4_team")
    urgency_lookup = {role: max(u, 0) for role, u in gaps.get(selected_team_step4, [])}
    combined = []
    for _, row in classified.iterrows():
        is_own = row["team_abbr"] == selected_team_step4
        ru = urgency_lookup.get(row["primary_role"], 0)
        if row.get("secondary_role"):
            ru = max(ru, 0.5 * urgency_lookup.get(row["secondary_role"], 0))
        if ru == 0:
            ru = 0.05 + 0.15 * max(0, quality_score(row) / 30.0)
        combined.append((int(row["player_id"]), round(ru * quality_score(row), 2), is_own))
    combined.sort(key=lambda x: x[1], reverse=True)

    pref_rows = []
    for rank, (pid, score, is_own) in enumerate(combined, 1):
        pl = classified[classified["player_id"] == pid]
        if not pl.empty:
            p = pl.iloc[0]
            pref_rows.append({"Rank": rank, "Player": p["player_name"], "Current Team": p["team_abbr"],
                              "Role": p["primary_role"], "PTS": round(p["pts"], 1),
                              "REB": round(p["reb"], 1), "AST": round(p["ast"], 1),
                              "Preference Score": score, "_own": is_own})

    pref_df = pd.DataFrame(pref_rows).drop(columns=["_own"])
    st.caption("🟡 Highlighted = own-team players. The algorithm guarantees every team receives a player ranked above their own.")

    def highlight_s4(row):
        if row["Current Team"] == selected_team_step4:
            return ["background-color: #2E2A0A; color: #D4C44A"] * len(row)
        return [""] * len(row)

    st.dataframe(pref_df.style.apply(highlight_s4, axis=1), use_container_width=True, hide_index=True)

    team_gap_lookup = dict(gaps.get(selected_team_step4, []))
    top_needs = [r for r, u in sorted(gaps.get(selected_team_step4, []), key=lambda x: x[1], reverse=True) if u > 0][:3]
    if top_needs and pref_rows:
        lines = [f"**{selected_team_step4}'s top needs:** {', '.join(f'**{r}**' for r in top_needs)}\n"]
        for p in pref_rows[:3]:
            u = team_gap_lookup.get(p["Role"], 0)
            need = "critical need" if u > 0.5 else ("moderate need" if u > 0 else "depth position")
            lines.append(f"- **{p['Player']}** ({p['Current Team']}, {p['Role']}) — {need} for {selected_team_step4}, scores {p['Preference Score']}")
        st.markdown("\n".join(lines))

    st.markdown("---")
    st.markdown("<h3 style='color:#C89B3C;'>Step 5 — TTC Matching</h3>", unsafe_allow_html=True)
    st.markdown("""
- Each team's preference list feeds into the Top Trading Cycles algorithm
- A multi-edge preference graph is built — each team points to its **top 5** preferred teams
- Cycles up to **6 teams** long are detected; **longest cycles selected first**
- Every team in a cycle receives a player ranked higher than what they gave up
""")
    team_to_received_pid = {}
    for ci, cycle in enumerate(cycles):
        for trade in cycle["trades"]:
            team_to_received_pid[trade["team"]] = trade["receives"]["player_id"]

    if cycles:
        cycle = cycles[0]
        teams = cycle["cycle"]
        with st.expander(f"Example Cycle: {' → '.join(teams)} → {teams[0]}  ({len(teams)}-team trade)", expanded=True):
            cols = st.columns(len(teams))
            for col, trade in zip(cols, cycle["trades"]):
                nt = teams[(teams.index(trade["team"]) + 1) % len(teams)]
                with col:
                    st.markdown(f"""
                    <div style='background:#0E1118;border:1px solid #1A2235;border-radius:12px;padding:14px;text-align:center;'>
                      <div style='font-size:17px;font-weight:800;color:#C89B3C;margin-bottom:12px;'>{trade['team']}</div>
                      <div style='font-size:9px;color:#64748B;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:4px;'>Sends</div>
                      <div style='font-size:13px;font-weight:700;color:#EF4444;'>{trade['gives']['player_name']}</div>
                      <div style='font-size:10px;color:#94A3B8;margin-bottom:10px;'>{trade['gives']['primary_role']}</div>
                      <div style='font-size:9px;color:#64748B;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:4px;'>Receives</div>
                      <div style='font-size:13px;font-weight:700;color:#22C55E;'>{trade['receives']['player_name']}</div>
                      <div style='font-size:10px;color:#94A3B8;'>{trade['receives']['primary_role']}</div>
                      <div style='font-size:10px;color:#64748B;margin-top:8px;'>from {nt}</div>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
            selected_pref_team = st.selectbox("View preference ranking for a team in this cycle", teams, key="step5_cycle_0")
            selected_pid = team_to_received_pid.get(selected_pref_team)
            gives_info = next((t["gives"] for t in cycle["trades"] if t["team"] == selected_pref_team), None)
            gives_pid = gives_info["player_id"] if gives_info else None
            ul_s5 = {role: max(u, 0) for role, u in gaps.get(selected_pref_team, [])}
            all_scored = []
            for _, row in classified.iterrows():
                ru = ul_s5.get(row["primary_role"], 0)
                if row.get("secondary_role"):
                    ru = max(ru, 0.5 * ul_s5.get(row["secondary_role"], 0))
                if ru == 0:
                    ru = 0.05 + 0.15 * max(0, quality_score(row) / 30.0)
                all_scored.append((int(row["player_id"]), round(ru * quality_score(row), 2)))
            all_scored.sort(key=lambda x: x[1], reverse=True)
            rows_s5 = []
            for rank, (pid, score) in enumerate(all_scored, 1):
                pl = classified[classified["player_id"] == pid]
                if pl.empty: continue
                p = pl.iloc[0]
                rows_s5.append({"Rank": rank, "Player": p["player_name"], "From": p["team_abbr"],
                                "Role": p["primary_role"], "Preference Score": score,
                                "Status": "Receives" if pid == selected_pid else ("Gives" if pid == gives_pid else "")})
            pref_df_s5 = pd.DataFrame(rows_s5)

            def highlight_s5(row):
                if row["Status"] == "Receives": return ["background-color: #0D2B1A; color: #22C55E"] * len(row)
                if row["Status"] == "Gives":    return ["background-color: #2D0A0A; color: #EF4444"] * len(row)
                return [""] * len(row)

            st.caption("🟢 Receives — player acquired  ·  🔴 Gives — player sent away")
            st.dataframe(pref_df_s5.style.apply(highlight_s5, axis=1), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("<h3 style='color:#C89B3C;'>Step 6 — AI Trade Narratives</h3>", unsafe_allow_html=True)
    st.markdown("""
- For every completed cycle, **GPT-4o** generates an ESPN-style analyst breakdown
- Explains what each team gains, what they give up, and why it makes sense for their roster
- Cached per session — API called once per cycle, not on every page load
""")
    if explanations and cycles:
        if openai_api_key:
            if "step6_ai_narrative" not in st.session_state:
                with st.spinner("Generating AI narrative..."):
                    try:
                        st.session_state["step6_ai_narrative"] = generate_ai_narrative(cycles[0], openai_api_key)
                    except Exception as e:
                        st.session_state["step6_ai_narrative"] = f"⚠️ Error: {e}"
            st.caption("Live AI-generated narrative — first trade cycle:")
            st.markdown(
                f"<div style='background:#0C1828;border-left:3px solid #C89B3C;border-radius:0 8px 8px 0;"
                f"padding:14px 18px;font-size:14px;line-height:1.8;color:#94A3B8;font-style:italic;margin-top:8px;'>"
                f"{st.session_state['step6_ai_narrative']}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.info("Enter your OpenAI API key in the sidebar to see a live AI-generated narrative.")

    st.markdown("---")
    st.markdown("<h3 style='color:#C89B3C;'>Step 7 — Explore the Results</h3>", unsafe_allow_html=True)
    st.markdown("""
| Tab | What it shows |
|---|---|
| **Dashboard** | High-level metrics and the full trade network graph |
| **Team Explorer** | Roster, role composition, and ranked trade targets for any team |
| **Trade Cycles** | Every trade in detail with AI-generated analyst breakdowns |
| **AI Evaluation** | Honest assessment of where the algorithm works and where it falls short |
""")
    st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
# AI EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
with tab_eval:
    st.markdown("<h1>AI Evaluation</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#64748B;margin-top:-8px;margin-bottom:24px;'>Where the algorithm works well — and where it falls short.</p>", unsafe_allow_html=True)

    st.markdown("<h3 style='color:#C89B3C;'>1. Role Classification</h3>", unsafe_allow_html=True)
    st.markdown("""
**What it does**: Assigns each player one of 8 archetypes using statistical thresholds.

**Works well**: Clear archetypes are accurately identified. Transparent and explainable — you can see exactly why each classification was made.

**Falls short**: ~50% of players fall into "Rotation Player" because they don't fit neat archetypes. Can't capture playstyle nuance (e.g. gravity shooters, connective passers). Thresholds are hand-tuned, not learned.
""")
    st.caption("Sample classifications — top 20 scorers:")
    st.dataframe(
        classified.nlargest(20, "pts")[["player_name", "team_abbr", "primary_role", "secondary_role", "pts", "reb", "ast"]].reset_index(drop=True),
        use_container_width=True, hide_index=True,
    )

    st.markdown("<h3 style='color:#C89B3C;'>2. Preference Simulation</h3>", unsafe_allow_html=True)
    st.markdown("""
**What it does**: Ranks every player on every other team by how well they fill a team's roster gaps.

**Works well**: Correctly identifies obvious needs. Team-specific ideals create diverse preferences, which is critical for multi-team cycles to emerge.

**Falls short**: Doesn't model chemistry, age, contract status, injury history, or scheme fit. Urgency scoring is linear — real trade value is not.
""")

    st.markdown("<h3 style='color:#C89B3C;'>3. AI Trade Narratives</h3>", unsafe_allow_html=True)
    st.markdown("""
**What it does**: Uses GPT-4o to generate analyst-style breakdowns for each trade cycle.

**Works well**: Natural, varied language that explains roster logic clearly to a non-technical audience.

**Falls short**: Requires an API key and has a small risk of hallucination — it may add context not strictly in the data.
""")
    st.markdown("""
| | Template-Based | GPT-4o (our approach) |
|---|---|---|
| **Accuracy** | Always faithful to data | May add context beyond data |
| **Variety** | Limited templates | Highly varied |
| **Nuance** | Mechanical | Can capture subtle logic |
| **Cost** | Free | Requires API key |
""")

    st.markdown("<h3 style='color:#C89B3C;'>4. TTC Algorithm (Multi-Edge Variant)</h3>", unsafe_allow_html=True)
    st.markdown("""
**What it does**: Builds a multi-edge preference graph and finds stable trading cycles up to 6 teams long.

**Works well**: Guarantees individual rationality. Multi-edge + longest-first reliably surfaces multi-team deals. Runs in milliseconds for 30 teams.

**Falls short**: No global optimality guarantee. Single-player-per-team constraint limits complexity. Longest-first prioritization may give some teams a slightly less-preferred outcome vs. classical TTC.

**Key iteration**: Our initial single-pointer graph produced almost exclusively 2-team swaps because 17/30 teams pointed at the same target (SAC). Multi-edge was the fix.
""")
