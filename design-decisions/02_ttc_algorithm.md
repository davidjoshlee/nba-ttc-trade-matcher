# Design Decision: Top Trading Cycles (TTC) Algorithm

## Decision
We use the Top Trading Cycles algorithm to identify stable multi-party trade opportunities across NBA teams.

## Why TTC over alternatives?

| Mechanism | Pros | Cons | Why not? |
|-----------|------|------|----------|
| **TTC** (our choice) | Stability guarantee, finds multi-party cycles naturally | Not globally optimal | Best fit for voluntary trading |
| Stable matching (Gale-Shapley) | Well-studied, optimal for one side | Two-sided only, doesn't handle multi-party | NBA trades aren't a two-sided market |
| Auction / market clearing | Price discovery, efficient | Requires defining "price" for players | Adds complexity we scoped out (salary cap) |
| Brute force (enumerate all trades) | Globally optimal | Computationally infeasible for 30 teams | O(n!) complexity |

## Key Property: Individual Rationality
Every team in a completed cycle receives a player they prefer over what they gave up. No team would unilaterally undo their trade. This is critical for a voluntary market — a theoretically optimal trade that any team would veto is worthless.

## Implementation Details
- Each team offers a pool of available players (bottom N by minutes)
- Each team "points" to the team owning their most-preferred available player
- We find all cycles in the resulting directed graph
- When a team sends a player, we pick the one the receiving team values most from the sender's available pool

## Tradeoff: Stability vs. Global Optimality
TTC guarantees no team regrets their trade, but it does NOT guarantee the maximum total value across all possible trade configurations. We chose stability because in a voluntary setting, a deal every party agrees to is worth more than a theoretically superior deal that falls apart.

## Structural Observation
Most trades found are 2-team swaps. Multi-team trades (3+) are rarer. This mirrors reality — the NBA sees far more bilateral trades than multi-party deals, because alignment across 3+ parties is hard. The algorithm surfaces multi-team trades when they exist, but doesn't force them.
