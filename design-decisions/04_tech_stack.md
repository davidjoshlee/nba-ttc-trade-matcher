# Design Decision: Tech Stack

## Stack
- **Python 3.11+** — single language for everything
- **nba_api** — NBA data from NBA.com
- **pandas/numpy** — data manipulation
- **networkx** — trade cycle network visualization in the dashboard
- **matplotlib** — chart and graph rendering
- **Streamlit** — web UI
- **Git/GitHub** — collaboration

## Why Streamlit over Flask/React?
- **Zero frontend code**: No HTML, CSS, or JavaScript needed
- **Built-in data display**: Tables, charts, metrics out of the box
- **Rapid iteration**: Change Python code, page hot-reloads
- **Free deployment**: Streamlit Cloud for demo day
- **3 people can run it locally** with one command

The tradeoff is less layout control and no persistent state between sessions. For a class demo, this doesn't matter.

## Why nba_api over other data sources?
- **Free, no API key** — wraps NBA.com's internal endpoints
- **Comprehensive**: Per-game stats, rosters, all current season data
- **Python-native**: `pip install nba_api`, clean DataFrame output
- **Rate limiting**: We cache results as CSV on first pull to avoid hitting NBA.com repeatedly

Alternative considered: balldontlie API (requires free API key, simpler but less data). We chose nba_api for richer stat coverage.

## Why no external AI API?
No Claude/GPT API key available. We use template-based narratives instead. This is actually a feature for evaluation purposes — templates are deterministic and can't hallucinate, making it easier to validate accuracy.

## Data caching
API results are cached as CSV in `src/data/cache/`. The app reads from cache by default, only re-fetching if explicitly requested. This makes the demo reliable regardless of network conditions.
