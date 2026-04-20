# Marketplace Overview: NBA Trade Matcher

## The Problem

Professional sports leagues systematically under-produce multi-team trades. The core coordination failure: Team A wants what Team B has, but B has no interest in what A offers. A third party could complete the deal, but finding that third party through bilateral phone calls is slow, information-leaky, and doesn't scale. Value-creating trades go unrealized, leaving teams worse off and the league product weaker.

## The Platform

NBA Trade Matcher is a centralized matching platform that surfaces multi-party trade opportunities by running a Top Trading Cycles (TTC) algorithm across team preference data. It solves the coordination problem that bilateral negotiation cannot: simultaneously aligning the interests of 3, 4, 5, or 6 teams around a single deal.

### Marketplace Participants
- **30 NBA teams**, each with a roster of ~12-16 rotation players
- Each team acts as both buyer and seller: they offer players from positions of surplus and seek players to fill roster gaps

### What Gets Traded
- Individual players (single player per team per cycle)
- Out of scope: draft picks, salary cap matching, multi-player packages

## Platform Architecture

The system runs as a six-stage pipeline:

```
NBA Data (nba_api) → Role Classification → Gap Analysis → Preference Ranking → TTC Matching → Narrative
```

### Stage 1: Data Ingestion
Pull current-season per-game stats for all NBA players via `nba_api`. Filter to rotation-level players (≥10 min/game). Cache locally to avoid repeated API calls. This gives us ~458 players across 30 teams.

### Stage 2: Role Classification
Assign each player one of 8 archetypes (Floor Spacer, Rim Protector, Playmaker, Paint Scorer, Two-Way Wing, Stretch Big, Defensive Anchor, Volume Scorer) based on statistical thresholds. Players who don't strongly fit any archetype default to "Rotation Player." Rule-based, not ML — transparent and fast to iterate.

### Stage 3: Gap Analysis
For each team, compute a team-specific ideal roster composition based on their statistical profile. Teams weak in scoring get higher ideal counts for offensive roles; teams weak in defense prioritize defensive roles. Compare actual roster to ideal to produce an urgency-ranked list of needs per team.

### Stage 4: Preference Ranking
For each team, rank all available players on other teams by how well they address the team's top needs. Preference score = role urgency × player quality. Available players are the bottom N by minutes on each roster (simulating what a GM would realistically trade). Each team ends up with a ranked preference list over ~140 available players from other teams.

### Stage 5: TTC Matching
Build a multi-edge preference graph where each team has directed edges to the top 5 teams owning their most-preferred players. Search for all simple cycles up to length 6 using DFS. Select non-overlapping cycles greedily, longest first, to maximize multi-team trade discovery. Each team in a cycle sends a player and receives one.

### Stage 6: Narrative Generation
For each trade cycle, generate a plain-language explanation using structured templates that reference the specific roles gained, roles lost, and team needs addressed. Template-based (no LLM API) — deterministic, zero hallucination risk.

## Key Design Principles

### 1. Stability Over Optimality
TTC guarantees that no team in a completed cycle would prefer to undo the trade (individual rationality). It does not guarantee the globally optimal configuration. We chose stability because in a voluntary market, a deal every party will actually agree to is more valuable than a theoretically superior deal that falls apart.

### 2. Privacy by Design
In the proposed real-world deployment, teams submit preferences privately. No team sees another team's full trade block or preference list. Teams only learn what others offered once a valid cycle is identified. This prevents strategic manipulation (withholding assets, distorting preferences to block rivals).

### 3. Coordination as the Value Proposition
The platform doesn't tell teams what to want — it takes their preferences as given and finds mutually beneficial cycles they couldn't discover through bilateral negotiation. A 6-team trade requires all six parties to simultaneously agree on terms. No amount of phone calls scales to that.

### 4. Multi-Edge Graph for Multi-Party Trades
Our initial single-pointer TTC (each team points to only their #1 choice) produced almost exclusively 2-team swaps because preferences were concentrated — 17 of 30 teams pointed at the same target. The multi-edge approach (top 5 preferences per team) with longest-cycle-first selection was critical to surfacing the multi-party trades that are the platform's core value.

## Current Results

- **458 players** classified across 30 teams
- **5 trade cycles** found, all multi-team (four 6-team, one 5-team)
- **All 30 teams** participate in at least one trade
- Example: BOS sends Horford (Floor Spacer) → DEN, SAC sends Monk (Playmaker) → BOS, POR sends Clingan (Rim Protector) → SAC — a 6-team cycle where each team fills a genuine roster gap

## What Makes This a Platform, Not Just an Algorithm

The TTC algorithm is one component. The platform value comes from:

1. **Aggregating private information** — each team's needs and available players, which no single team has visibility into
2. **Reducing coordination costs** — turning an O(n²) bilateral search into a single centralized matching run
3. **Enabling trades that couldn't exist otherwise** — multi-party cycles require simultaneous agreement from all participants, which bilateral negotiation cannot efficiently produce
4. **Preserving competitive confidentiality** — teams reveal preferences to the platform, not to each other

This is the same value proposition that powers matching markets in other domains: kidney exchange (where donor-recipient cycles mirror our player-team cycles), school choice, and residency matching.
