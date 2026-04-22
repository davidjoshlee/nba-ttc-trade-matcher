# NBA Trade Matcher — TTC

OIT 277 group project (Stanford GSB, Spring 2026). Team: David Lee, Tyson Fenay, Ryley Mehta.

## What this is

A platform prototype that surfaces multi-team NBA trade opportunities using the Top Trading Cycles (TTC) algorithm. It solves a coordination failure: multi-team trades create value but rarely happen because bilateral phone calls can't align 3-6 parties simultaneously.

## Deadlines

- **Wed April 29, 9am**: 2-3 min pitch video + ~1 page project description
- **Thu April 30, 8am**: 4-5 slide deck + working demo (Streamlit app)

## How to run

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

Tests: `python tests/test_roles.py && python tests/test_ttc.py && python tests/test_preferences.py`

## Architecture (6-stage pipeline)

```
nba_api → Role Classification → Gap Analysis → Preference Ranking → TTC Matching → Narrative
```

1. `src/data/fetch_rosters.py` — pulls per-game stats via nba_api, caches as CSV
2. `src/roles/taxonomy.py` + `classifier.py` — 8 archetypes + "Rotation Player" default, rule-based with min-ratio scoring
3. `src/preferences/gap_analysis.py` — team-specific ideal roster (adjusted by team stats, not one-size-fits-all)
4. `src/preferences/ranker.py` — ranks available players by role_urgency × quality_score
5. `src/matching/ttc.py` — multi-edge preference graph (top 5 teams per team), DFS cycle detection up to length 6, longest-first greedy selection
6. `src/narrative/explainer.py` — template-based trade explanations (no LLM API)

UI: `app/streamlit_app.py` — Streamlit, 4 tabs (Dashboard, Team Explorer, Trade Cycles, AI Evaluation)

## Key design decisions (don't undo these without discussion)

- **Multi-edge TTC, not single-pointer**: The original single-pointer graph produced only 2-team swaps because 17/30 teams pointed at the same target (SAC). Multi-edge (top 5 preferences per team) + longest-cycle-first was the fix. This is documented in `design-decisions/02_ttc_algorithm.md`.
- **Min-ratio role scoring, not average**: Average ratios let one extreme stat dominate (Curry was classified as Stretch Big). Min-ratio ensures a player must meet ALL thresholds. See `design-decisions/01_role_taxonomy.md`.
- **Team-specific ideal rosters**: Gap analysis adjusts each team's ideal based on their stats (bad scorers want offense, bad defenders want defense). Without this, all teams have identical needs and the preference graph collapses.
- **Template-based narratives**: No API key available. Templates are deterministic and can't hallucinate — this is a feature for the evaluation section.
- **Quality-scaled Rotation Player urgency**: Baseline urgency scales with player quality (0.05–0.20) instead of flat 0.1, creating meaningful differentiation among bench players.

## Out of scope (by design, do not add)

- Salary cap matching
- Draft picks as tradeable assets
- Contract length / expiring deals
- Multi-player packages per team per cycle
- LLM-generated narratives (no API key)

## Key parameters (adjustable in UI sidebar)

- `PLAYERS_AVAILABLE_PER_TEAM` (default 7) — how many players each team puts on the trade block
- `MIN_MINUTES_THRESHOLD` (default 10.0) — minimum minutes/game to be included
- Both are passed explicitly through the pipeline — do not use config module overrides

## Documentation

- `design-decisions/00_marketplace_overview.md` — start here, full platform summary
- `design-decisions/01-05_*.md` — detailed docs on each design choice
- `README.md` — setup and project structure

## Current results (default settings)

- 458 players across 30 teams
- 5 trade cycles, all multi-team (four 6-team, one 5-team)
- All 30 teams participate
- Example: BOS→SAC→POR→PHI→UTA→DEN 6-team cycle

## Style notes

- Python 3.11+, pandas DataFrames throughout
- No external AI API calls anywhere
- Data cached in `src/data/cache/` (gitignored)
- networkx is for visualization only — TTC uses custom DFS in `_find_cycles_in_multigraph`
