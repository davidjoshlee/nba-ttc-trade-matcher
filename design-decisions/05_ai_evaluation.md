# Design Decision: AI Evaluation

## Where AI / automation adds value

### 1. Role Classification
The rule-based classifier correctly identifies clear archetypes: elite shooters become Floor Spacers, rim protectors are tagged, playmakers are found by assist numbers. It processes 458 players instantly and produces consistent, explainable results.

### 2. Gap Analysis
Automated roster analysis correctly identifies team weaknesses, and the team-specific ideal roster adjustment adds meaningful diversity. Teams weak in scoring are steered toward offensive roles; teams weak in defense prioritize defensive upgrades. This prevents the "everyone wants the same player" problem that plagued our initial implementation, where 17 of 30 teams pointed at the same target.

### 3. Trade Cycle Discovery
The multi-edge TTC algorithm finds trades that would be extremely difficult to coordinate manually. The current implementation surfaces 5 trade cycles involving all 30 NBA teams, all multi-team (5-6 teams each). A 6-team trade requires all six parties to simultaneously agree — something phone-call-based negotiation handles poorly. This is the core value proposition of the platform.

## Where AI / automation falls short

### 1. "Rotation Player" black hole
~50% of players can't be classified into any role. These players still have value in real basketball, but our system treats them as interchangeable.

### 2. Simplified team strategy model
While we adjust each team's ideal roster based on their aggregate stats (scoring, rebounding, defense), this is still a crude proxy for real coaching philosophy. A team running a switch-everything defense values different player profiles than one running a drop-coverage scheme — nuances our stat-based adjustment can't capture.

### 3. Missing context
Real trade decisions depend on factors we don't model: salary cap implications, player age and contract years, injury history, locker room fit, draft pick compensation, and coaching scheme. Our system operates in a simplified world.

### 4. Narrative limitations
Template-based narratives are accurate but mechanical. They can state *what* happened but lack the analytical depth to explain *why* at a strategic level. An LLM could add richer context (e.g., "This addresses Portland's perimeter scoring drought that has persisted all season") but at the cost of potential hallucination.

## Summary
The system works well as a coordination tool — it surfaces trades that human negotiators would struggle to find. But it should augment, not replace, human judgment. A real deployment would need human review of every suggested trade before execution.
