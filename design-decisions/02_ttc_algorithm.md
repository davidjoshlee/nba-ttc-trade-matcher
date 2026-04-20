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
- We build a **multi-edge preference graph**: each team has directed edges to the top 5 teams owning their most-preferred available players. This is a key departure from classic TTC, where each agent has exactly one outgoing edge.
- We search for all simple cycles up to length 6 in this graph using DFS
- Cycles are sorted longest first, and we greedily select non-overlapping cycles (longest priority) to maximize multi-team trade discovery
- When a team sends a player, we pick the one the receiving team values most from the sender's available pool

### Why multi-edge instead of single-pointer?
Our initial implementation used a functional graph (1 edge per team). This produced almost exclusively 2-team swaps because preference concentration meant most teams pointed at the same popular target (e.g., 17 of 30 teams pointed at SAC). That target's team pointed back at one other team, forming an immediate 2-cycle that consumed the most popular players. The multi-edge approach creates a richer graph structure where longer cycles can form.

## Tradeoff: Stability vs. Global Optimality
TTC guarantees no team regrets their trade, but it does NOT guarantee the maximum total value across all possible trade configurations. We chose stability because in a voluntary setting, a deal every party agrees to is worth more than a theoretically superior deal that falls apart.

## Tradeoff: Cycle Length Priority vs. Classical TTC
By prioritizing longer cycles, we bias toward multi-team trades. Classical TTC makes no such distinction — it finds and executes cycles in whatever order they appear. Our approach better demonstrates the platform's value proposition (surfacing multi-party deals that bilateral negotiation would miss), but it means some teams might get a slightly less-preferred player than they would in a strict first-come-first-served TTC.

## Results
The current implementation produces 5 trade cycles involving all 30 NBA teams, all multi-team (four 6-team trades and one 5-team trade). Example: BOS sends Horford to DEN, SAC sends Monk to POR, POR sends Clingan to SAC — a 6-team deal where each team fills a genuine roster gap.
