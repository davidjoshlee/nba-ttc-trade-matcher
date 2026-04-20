**OIT 277 Project Proposal: Sports Trade Matching via Top Trading Cycles**  
David Lee | [dlee13@stanford.edu](mailto:dlee13@stanford.edu), Tyson Fenay | [tfenay@stanford.edu](mailto:tfenay@stanford.edu),   
Ryley Mehta | ryley@stanford.edu

**THE PROBLEM**

Professional sports leagues see relatively few multi-team trades, even when these multi-party deals would create positive value for all stakeholders involved. The core coordination failure is the following: Team A may want what Team B has, but B may have no interest in what A offers in return. Without a third (or fourth) party in the deal the trade does not occur and potential value goes unrealized. The conventional approach to player trading (managers picking up the phone and cold-calling counterparts) is too slow, too bilateral focused, and too exposed to information leakage to facilitate multi-team trades. The result is a market that systematically under-produces multi-team trades, leaving teams worse off and the league product weaker than it could be.

**PLATFORM CONCEPT**

We are proposing to develop a platform that surfaces multi-party trade opportunities by running a Top Trading Cycles (TTC) algorithm across anonymized team preference data.

The mechanism works as follows:

* Each team's manager submits: players on their roster they are willing to trade ("available assets"), and players on other rosters they would want to acquire ("targets")  
* Managers then rank the combined list by preference. To support this, we will explore using AI to parse a plain-language input of positional needs, salary constraints and contract status preferences to generate a suggested ranking  
* The platform then constructs, for each team, a filtered version of this data on the back-end only using players who were actually made available by their teams. This preserves a degree of competitive confidentiality: no team sees the complete list of players another team has put on the trading block  
* The TTC algorithm then identifies stable trading cycles \- sets of teams where every participant receives a player they prefer over what they currently hold. Cycles may involve two teams or many and will only involve players who were actually made available for trading.

The central design challenge is matching: identifying which potential trades would create mutually beneficial value for all teams involved. 

**PROTOTYPE SCOPE**

We will build a functional TTC matching engine simulated over a realistic NBA roster dataset.

*Inputs:* NBA player data (sourced online or synthetically generated), Team needs: position, role, and salary band, Player availability (team manager input), Target rankings (Suggested by AI, finalized by team-manager)

*Outputs:* Identified trading cycles, with a plain-language explanation of the strategic logic

*Tools:* Google Colab (TTC algorithm), Airtable (team and player data), Claude/ChatGPT (preference inference and narrative generation)

The AI layer serves two functions: inferring preference orderings from natural-language descriptions of team needs, and producing human-readable explanations of why each suggested trade makes sense.

**DESIGN TRADEOFFS**

**1\. Stability vs. optimality.** TTC guarantees that no team in a cycle prefers its starting position but it does not guarantee the globally highest-value configuration across all possible trades. We are leaning toward stability. In a voluntary market, a deal that every party will actually agree to is more valuable than a theoretically superior deal that falls apart in reality.

**2\. Transparency vs. strategic manipulation.** If teams could see which players other teams have made available, they could game the system withholding assets or distorting preferences to block a rival from completing a favorable cycle. Our design addresses this by keeping availability and preference data private: teams only learn what others offered once a valid cycle is identified. The tradeoff is reduced transparency, but we view that as the right call given the competitive dynamics of the setting.

**EVALUATION PLAN**

"Good" means two things: the engine identifies cycles that are genuinely value-additive given the stated preferences, and the AI-generated explanations accurately reflect the logic of each deal \-  not just that a cycle was found. We will validate by:

* Manually auditing engine outputs against input rankings to confirm the algorithm is surfacing cycles that are preference-consistent  
* Evaluating AI explanations for accuracy: do they correctly describe the trade logic, or do they hallucinate justifications disconnected from the underlying data?  
* If real player data is available, backtesting against historical multi-team trades to assess whether the engine would have flagged similar cycles  
* Assessing the number of two party vs. multi team trades suggested by our engine