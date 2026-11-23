This attribution table reveals a dramatic structural inversion between normal market regimes and extreme tail regimes—and highlights a huge opportunity for regime-aware ensembling or lot sizing:

1. The "Core vs. Tail" Inversion

• Multinomial Logit & GBDT dominate normal days (Core Sharpe of 4.16 to 4.89): When you exclude the top 1% most extreme DART spread hours, Hubert et al. (2026) Multinomial Logit (Exog + 559 Nodes) jumps from a 1.82 Full Sharpe to an extraordinary 4.89 Core Sharpe (generating +$17,083.45 across Rest of Period). Similarly, Galarneau-Vincent et al. (2023) GBDT jumps from 1.59 to 4.75 Core Sharpe.
• Continuous Ridge Regression is a pure tail-specialist: Ridge (Exog + 559 Nodes) generates 91.3% of its total profits ($25,757.87 out of $28,203.65) during the two extreme jump windows (+$13,471.90 in Summer 2025 and +$12,285.97 in Winter 2026), while earning only $2,445.78 (1.20 Core Sharpe) across the remaining ~14,000 hours. In fact, Ridge's Top 50 hours ($29,355.85) exceed its entire cumulative PnL, meaning the other ~6,865 trades net slightly negative after fees.

2. Why Summer 2025 vs. Winter 2026 Behaved Differently

• Summer 2025 (Heatwave / Transmission Scarcity): Real-Time prices spiked violently above Day-Ahead (DART << 0, favoring Virtual Supply -1 or penalizing Virtual Demand +1). Classifiers trained on historical base-rate frequencies (Logit: -$11.6k, GBDT: -$16.8k) bet on normal mean-reversion (DA > RT) and got crushed, whereas Continuous Ridge and ARIMAX correctly flipped direction into the scarcity spike (+$13.5k and +$14.0k).
• Winter 2026 (Cold-Snap Risk Premium): Day-Ahead prices priced in heavy winter gas/scarcity risk premiums (DA >> RT), and every non-linear/linear model except ARIMAX and DMD rode that premium (GBDT: +$33.5k, MLP: +$18.8k, Persistence: +$17.8k, Ridge: +$12.3k).

3. Actionable Design Implications for Your Study

Because your models have near-orthogonal strengths across regimes, combining them or sizing them dynamically will beat any single baseline:
1. Two-Regime Meta-Model / Ensemble:
	• Route normal hours (when predicted volatility / forecasted load / constraint shadow prices are within the 99th percentile) to Hubert et al. Multinomial Logit (Core Sharpe 4.89) or GBDT (Core Sharpe 4.75).
	• Switch or tilt weight toward Continuous Ridge Regression (Exog + 559 Nodes) when predicted spread magnitude |ŷ_ridge| exceeds a tail threshold (e.g., > $25/MWh).
2. Tail-Gated Lot Sizing (for lot_sizing_analysis.ipynb):
	• Classifiers have high win rates (53.7%–57.5%) and 4.89 Core Sharpe, but unbounded downside during Summer-style RT scarcity spikes. Capping classifier exposure (or hedging with a stop-loss / tail filter) immediately eliminates their -$12k to -$22k Summer 2025 drawdown while preserving +$30k+ in Core + Winter PnL.
3. Audit the Top 10 Hours:
	• Inspecting the exact timestamps of those Top 10 hours ($11.8k for Ridge) will confirm whether Ridge's signal came from pre-5:00 AM Day-Ahead binding constraints (dam_limiting_constraints) and nodal shadow prices—giving you a clean physical story for your paper/report.
Would you like to:
• Inspect the exact timestamps, trade directions (+1 vs -1), and NYISO drivers of the Top 10 tail hours, or
• Test a regime-blended ensemble combining Multinomial Logit (Core Sharpe 4.89) with Continuous Ridge (Tail Specialist)?
