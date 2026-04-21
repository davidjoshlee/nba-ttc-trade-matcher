# Design Decision: Role Taxonomy

## Decision
We classify NBA players into 8 archetypes (Floor Spacer, Rim Protector, Playmaker, Paint Scorer, Two-Way Wing, Stretch Big, Defensive Anchor, Volume Scorer) using rule-based stat thresholds, with a "Rotation Player" default for players who don't strongly fit any archetype.

## Why rule-based over ML?
- **Transparency**: Every classification can be explained by pointing to specific stats vs. thresholds. This matters for the evaluation rubric — we need to show where AI works and fails.
- **Iteration speed**: When Steph Curry was initially classified as "Stretch Big" (wrong), we could immediately see why (the scoring function averaged ratios, letting one extreme stat inflate the score) and fix it by switching to a min-ratio approach.
- **Scope**: Training a clustering or classification model requires labeled data or careful validation. With 10 days and 3 people, a rule-based system gets us to "good enough" faster.

## Tradeoff: Simplicity vs. Nuance
- **What we gain**: Deterministic, explainable classifications. Zero risk of overfitting.
- **What we lose**: ~50% of players get "Rotation Player" because they don't hit any threshold set. A more sophisticated approach (e.g., k-means clustering on stat vectors) could find more nuanced groupings. We also can't capture playstyle from tracking data (gravity, defensive versatility).

## Scoring Method
We use the minimum ratio across all stat thresholds for a role, with a small tiebreaker from the average. This ensures a player must meet ALL of a role's requirements — not just dominate one stat. This fixed a bug where high-rebound guards were being classified as bigs.

## Thresholds

Current values (defined in `src/roles/taxonomy.py`):

| Role | Stat Thresholds |
|------|----------------|
| Floor Spacer | fg3a >= 5.0, fg3_pct >= 0.36 |
| Rim Protector | blk >= 1.2, reb >= 7.0 |
| Playmaker | ast >= 5.5, ast/tov >= 1.6 |
| Paint Scorer | fg_pct >= 0.54, reb >= 6.0, pts >= 12.0 |
| Two-Way Wing | stl >= 1.0, pts >= 14.0, reb >= 4.0 |
| Stretch Big | reb >= 7.0, fg3a >= 2.0, fg3_pct >= 0.32 |
| Defensive Anchor | stl+blk >= 2.2, min >= 27.0 |
| Volume Scorer | pts >= 22.0, fga >= 17.0 |

A player must score >= 0.85 on the minimum ratio across all thresholds to qualify for a role. Players below this get "Rotation Player."

Thresholds were set by examining league-wide stat distributions and calibrating against known players (Curry → Floor Spacer, Gobert → Rim Protector, etc.). They are not learned from data — this is a limitation we acknowledge.
