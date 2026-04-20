# Design Decision: AI Evaluation

## Where AI / automation adds value

### 1. Role Classification
The rule-based classifier correctly identifies clear archetypes: elite shooters become Floor Spacers, rim protectors are tagged, playmakers are found by assist numbers. It processes 458 players instantly and produces consistent, explainable results.

### 2. Gap Analysis
Automated roster analysis correctly identifies team weaknesses. Teams lacking shot blockers are flagged as needing Rim Protectors; teams with no ball handlers are flagged as needing Playmakers. This drives preference generation that makes intuitive sense.

### 3. Trade Cycle Discovery
The TTC algorithm finds trades that would be extremely difficult to coordinate manually. Even a simple 3-team trade requires all three parties to simultaneously agree — something phone-call-based negotiation handles poorly.

## Where AI / automation falls short

### 1. "Rotation Player" black hole
~50% of players can't be classified into any role. These players still have value in real basketball, but our system treats them as interchangeable.

### 2. One-size-fits-all team strategy
Every team is evaluated against the same "ideal roster composition." In reality, a team running a specific system (e.g., small-ball, switch-everything defense) values completely different player profiles.

### 3. Missing context
Real trade decisions depend on factors we don't model: salary cap implications, player age and contract years, injury history, locker room fit, draft pick compensation, and coaching scheme. Our system operates in a simplified world.

### 4. Narrative limitations
Template-based narratives are accurate but mechanical. They can state *what* happened but lack the analytical depth to explain *why* at a strategic level. An LLM could add richer context (e.g., "This addresses Portland's perimeter scoring drought that has persisted all season") but at the cost of potential hallucination.

## Summary
The system works well as a coordination tool — it surfaces trades that human negotiators would struggle to find. But it should augment, not replace, human judgment. A real deployment would need human review of every suggested trade before execution.
