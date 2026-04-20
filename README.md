# NBA Trade Matcher — Top Trading Cycles

**OIT 277: Digital Platforms in the Age of AI — Group Project**
David Lee, Tyson Fenay, Ryley Mehta | Stanford GSB, Spring 2026

A platform prototype that surfaces multi-team NBA trade opportunities using the Top Trading Cycles (TTC) algorithm. Teams submit roster data, the system infers player roles and team needs, then identifies stable trading cycles where every participant improves.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app/streamlit_app.py
```

The app will open in your browser. NBA data is fetched from NBA.com on first run and cached locally — subsequent launches are instant.

## How It Works

1. **Data** — Pull current NBA rosters and per-game stats via `nba_api`
2. **Role Classification** — Assign each player an archetype (Floor Spacer, Rim Protector, Playmaker, etc.) based on stat thresholds
3. **Gap Analysis** — Compare each team's role composition to an ideal roster template to identify needs
4. **Preference Ranking** — Rank available players from other teams by how well they fill each team's gaps
5. **TTC Algorithm** — Find stable trading cycles where every team in the cycle receives a player they prefer
6. **Narrative Generation** — Produce plain-language explanations of why each trade makes sense

## Project Structure

```
group_project/
├── app/
│   └── streamlit_app.py           # Web UI (4 tabs: Dashboard, Team Explorer, Trade Cycles, AI Evaluation)
├── src/
│   ├── config.py                  # Tunable parameters (players per team, minutes threshold, ideal roster)
│   ├── data/fetch_rosters.py      # NBA data fetching + caching
│   ├── roles/
│   │   ├── taxonomy.py            # Role definitions and stat thresholds
│   │   └── classifier.py          # Assigns roles to players
│   ├── preferences/
│   │   ├── gap_analysis.py        # Identifies team needs
│   │   └── ranker.py              # Ranks trade targets per team
│   ├── matching/ttc.py            # TTC algorithm
│   └── narrative/explainer.py     # Template-based trade explanations
├── design-decisions/              # Docs explaining every major design choice (read these!)
├── tests/                         # Unit tests for roles, TTC, and preferences
├── project-docs/                  # Proposal and syllabus
└── requirements.txt
```

## Design Decisions

See the `design-decisions/` folder for detailed write-ups on:

1. **Role Taxonomy** — Why rule-based classification, how thresholds were set
2. **TTC Algorithm** — Why TTC over stable matching or auctions
3. **Preference Simulation** — How team needs drive automated rankings
4. **Tech Stack** — Why Streamlit, nba_api, and no external AI API
5. **AI Evaluation** — Where automation helps vs. where it falls short

## Key Parameters (in `src/config.py`)

| Parameter | Default | What it does |
|-----------|---------|-------------|
| `PLAYERS_AVAILABLE_PER_TEAM` | 7 | How many players each team puts on the trade block |
| `MIN_MINUTES_THRESHOLD` | 10.0 | Minimum minutes/game to be included |
| `IDEAL_ROSTER_COMPOSITION` | (see file) | Target role counts for a balanced roster |

These are also adjustable via sliders in the Streamlit sidebar.

## Running Tests

```bash
python tests/test_roles.py
python tests/test_ttc.py
python tests/test_preferences.py
```

## Scope Decisions

**In scope**: Single-player-per-team trades, role-based gap analysis, all 30 NBA teams, template-based narratives.

**Out of scope** (by design): Salary cap, draft picks, contract length, multi-player packages, LLM-generated narratives.
