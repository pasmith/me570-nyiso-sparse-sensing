# Sparse Sensing and Discriminant Analysis for Interpretable NYISO DART Spread Trading Literature Review

This literature review anchors Narrative Arc 3 ("From Forecasting to Regime Detection & Liquidity-Aware Execution"), tracing why traditional continuous forecasting and unforced spectral models break down in two-settlement electricity markets, how sparse spatial classification (`SSPOC`) bridges grid physics with algorithmic virtual trading, and how calibrated reject options and bid-stack elasticity govern net trading profitability.

### Narrative Arc

- **Paragraph 1** — Two-settlement wholesale market mechanics, virtual bidding (INC/DEC), and why DART spreads exhibit clustered, congestion-driven tail spikes [1–5].
- **Paragraph 2** — Traditional continuous electricity price forecasting (SARIMA, VAR, Ridge, ML/DL surrogates), direct spread modeling vs. subtraction, and the curse of dimensionality across 559 collinear generator nodes [6–11].
- **Paragraph 3** — Dynamical systems & modal decomposition lineage: POD/SVD → Exact DMD & Bagging-Optimized DMD (BOP-DMD) → why unforced spectral extrapolation suffers from 48-hour eigenvalue damping and inertia overshoot at exogenous regime turning points [12–16].
- **Paragraph 4** — Sparse sensing & classification lineage: Compressed Sensing → SSPOC → predictive/time-delay sparse sensing → glass-box interpretability mapped onto physical NYISO transmission bottlenecks [17–21].
- **Paragraph 5** — Distributional & spike-regime classification in power trading: reframing point regression as 3-class tail-spike detection (INC/DEC/Abstain) and modern ML benchmarks (Logit, GBDT, Transformers, MDNs) [22–28].
- **Paragraph 6** — Cost-sensitive probability thresholding & liquidity-aware execution: Bayes-optimal reject rules to prevent class-imbalance over-trading, quadratic Day-Ahead bid-stack self-impact lot sizing, and leak-free chronological validation [29–35].

---

#### Paragraph 1 — Market Microstructure & Virtual Arbitrage Mechanics

| # | Paper | Key idea | Why included |
| --- | --- | --- | --- |
| [1] | Borenstein et al. (2002), *American Economic Review* | Empirical analysis of two-settlement power markets showing how transmission congestion and forecast errors drive systematic DA–RT price divergence. | Foundational economics reference establishing why DART spreads exist physically and structurally. |
| [2] | Longstaff & Wang (2004), *Journal of Finance* | Documents forward risk premia, INC/DEC asymmetry, and volatility clustering in day-ahead versus real-time electricity prices. | Establishes the financial theory behind virtual bidding risk premia and limits to arbitrage. |
| [3] | Christensen et al. (2012), *International Journal of Forecasting* | Demonstrates that extreme electricity price spikes are short-lived, clustered, and triggered by binding transmission constraints. | Proves that DART volatility is driven by discrete structural bottlenecks rather than smooth Gaussian noise. |
| [4] | NYISO Market Participants User's Guide / Virtual Trading Manual (2024) | Defines virtual supply (INC) and virtual demand (DEC) bidding rules, collateral requirements, and the 05:00 EST Day-Ahead market cutoff. | Grounds the strict causal filtration ($\mathcal{F}_{D-1,05:00}$) and transaction-cost hurdle ($c = \$0.50$/MWh) in real NYISO exchange rules. |
| [5] | Ioannidis et al. (2025), *Energy Economics* | Analyzes cross-commodity spillovers (natural gas, weather, load) and structural bid-stack steepness governing LMP sensitivity. | Explains why exogenous weather and load forecasts interact non-linearly with grid congestion during tight supply regimes. |

#### Paragraph 2 — Traditional Continuous Price Forecasting & High-Dimensional Collinearity

| # | Paper | Key idea | Why included |
| --- | --- | --- | --- |
| [6] | Weron (2014), *International Journal of Forecasting* | Comprehensive benchmark review of statistical (SARIMA, VAR) and computational electricity price forecasting (EPF) models. | Canonical EPF survey establishing conventional time-series regression baselines. |
| [7] | Lago et al. (2021), *Applied Energy* | Establishes rigorous open-source benchmarks and statistical testing protocols for Day-Ahead electricity price forecasting. | Provides the academic standard for avoiding look-ahead bias and evaluating out-of-sample forecasting performance. |
| [8] | Afrasiabi et al. (2020), *IEEE Access* | Applies deep neural networks (CNN/LSTM) to continuous Locational Marginal Price (LMP) forecasting across grid nodes. | Represents modern black-box regression approaches, illustrating their high parameter count and lack of physical transparency. |
| [9] | Das et al. (2022), *IEEE PES General Meeting* | Evaluates univariate time-series point forecasts for hourly DART spreads across PJM grid nodes. | Highlights why direct spread modeling outperforms subtracting separate DA and RT forecasts, while showing point regression's failure near zero. |
| [10] | Jami et al. (2023), *IEEE ISGT Europe* | Uses machine learning surrogates to approximate Optimal Power Flow (OPF) LMPs 4–5 orders of magnitude faster than physical solvers. | Bridges physical grid optimization with data-driven nodal modeling while showing why full-grid surrogates remain opaque. |
| [11] | Borenstein et al. (2008), *Journal of Industrial Economics* | Analyzes market power, price-elasticity, and persistent arbitrage inefficiencies across nodal electricity markets. | Explains why L2-regularized linear models (e.g., Ridge) capture smooth directional risk premia with low drawdown, whereas unregularized full-grid models (559 collinear Pnodes) suffer variance inflation and out-of-sample Sharpe decay. |

#### Paragraph 3 — Modal Decomposition & Unforced Spectral Failure Modes

| # | Paper | Key idea | Why included |
| --- | --- | --- | --- |
| [12] | Schmid (2010), *Journal of Fluid Mechanics* | Introduces Dynamic Mode Decomposition (DMD) to extract coherent spatiotemporal modes from high-dimensional sequential snapshots. | Foundational algorithm reference for decomposing tall-and-skinny ($N \times T$) nodal state-space matrices. |
| [13] | Mann & Kutz (2016), *Quantitative Finance* | Adapts Exact DMD to financial time series to isolate coherent trading modes and execute systematic arbitrage strategies. | Direct methodological precedent connecting modal decomposition to quantitative trading signals. |
| [14] | Sashidhar & Kutz (2022), *Philosophical Transactions of the Royal Society A* | Introduces Bagging, Optimized DMD (BOP-DMD) using variable projection and statistical bagging to stabilize eigenvalue estimation under noise. | Benchmark spectral baseline showing that even stabilized, noise-robust Koopman eigenvalues remain unforced. |
| [15] | Dylewsky et al. (2020), *arXiv:2010.04248* | Develops stochastically forced ensemble DMD in time-delay coordinates to forecast near-periodic power grid load dynamics. | Explains why autonomous DMD decays toward the trailing mean across a 48-hour auction lag ($\text{Re}(\omega_k) \le 0$) and suffers inertia overshoot during abrupt exogenous mean-reversions. |
| [16] | Mohan et al. (2018), *IEEE Transactions on Power Systems* | Applies DMD and Koopman operators to wide-area power grid disturbance monitoring using limited measurements. | Connects modal decomposition directly to wide-area power grid bottleneck detection and sets up sparse sensing. |

#### Paragraph 4 — Sparse Sensor Placement (SSPOC) & Glass-Box Grid Physics

| # | Paper | Key idea | Why included |
| --- | --- | --- | --- |
| [17] | Candès & Wakin (2008), *IEEE Signal Processing Magazine* | Establishes Compressed Sensing theory: high-dimensional sparse signals can be recovered from far fewer measurements via $\ell_1$ optimization. | Mathematical bedrock explaining why full-state grid reconstruction is unnecessary when underlying congestion patterns are low-rank. |
| [18] | Brunton, Brunton, Proctor & Kutz (2016), *SIAM Journal on Applied Mathematics* | Introduces SSPOC: combines POD/SVD, LDA discriminant projection, and $\ell_1$ sparse optimization to place minimal sensors for classification. | Centerpiece algorithm paper—proves classification requires orders-of-magnitude fewer sensors than full-state reconstruction, eliminating 559-node collinearity. |
| [19] | Manohar, Brunton, Kutz & Brunton (2018), *IEEE Control Systems Magazine* | Extends data-driven sparse sensor placement via QR pivoting and time-delay coordinates for dynamic state estimation and forecasting. | Provides the spatiotemporal mathematical extension linking instantaneous point measurements to forward-looking horizons (T+24 to T+48). |
| [20] | Manohar, Hogan, Kutz & Brunton (2018), *arXiv:1804.07884* | Demonstrates neural-inspired sparse sensor architectures that preserve classification accuracy under severe dimensionality reduction (>95%). | Supports experimental comparisons pairing `SSPOC` sensors with linear (`LDA`, `Logit`) and non-linear (`GBDT`, `MLP`) classifiers. |
| [21] | Rudin (2019), *Nature Machine Intelligence* | Argues for inherently interpretable ("glass-box") models over post-hoc explanations in high-stakes physical and financial systems. | Grounds the dual-audience pitch: mapping $p \in [5, 20]$ sensor weights directly onto physical NYISO bottlenecks (Central-East, UPNY-SENY, PJM borders). |

#### Paragraph 5 — Tail-Spike Regime Classification & Quantitative Trading Benchmarks

| # | Paper | Key idea | Why included |
| --- | --- | --- | --- |
| [22] | Galarneau-Vincent et al. (2023), *Energy Economics* | Models hourly DART spread spike probabilities in NYISO's Long Island zone using GBDT, RF, MLP, and Logit with load, weather, and calendar features. | Direct NYISO empirical benchmark confirming ~55–62% hit rates and warning that raw nodal features degrade tree/linear generalization. |
| [23] | Deschatre & Gruet (2022), *Quantitative Finance* | Uses Gaussian–Generalized Pareto mixture and regime-switching models to separate normal market hours from extreme price tails. | Theoretical justification for splitting the continuous DART distribution into a central neutral zone (`Class 0`) and extreme tails (`±1`). |
| [24] | Hubert, Lolas, & Sircar (2026), *Energy Economics* / *arXiv:2601.05085* | Frames INC/DEC virtual bidding across NYISO, ISO-NE, and ERCOT as a joint spike classification and optimal bid-sizing problem net of transaction costs. | Primary literature benchmark (~53–58% win rate, 1.4–3.2 Sharpe) validating multi-zone spike classification and bid-stack sizing. |
| [25] | Lee et al. (2024), *arXiv:2412.00062* | Evaluates temporal Transformer architectures that quantize DART spreads into discrete categorical bins using multi-horizon load, wind, and solar forecasts. | Demonstrates how discretizing continuous spreads into categorical regimes filters low-edge hours during peak congestion ramps. |
| [26] | Yu et al. (2019), *IEEE Transactions on Power Systems* | Introduces Mixture Density Networks (MDNs) paired with Conditional Value-at-Risk (CVaR) to optimize virtual bid portfolios across correlated nodes. | Contextualizes generative probabilistic approaches for tail-risk control across multi-node electricity portfolios. |

#### Paragraph 6 — Cost-Sensitive Reject Thresholding, Bid-Stack Sizing & Backtest Hygiene

| # | Paper | Key idea | Why included |
| --- | --- | --- | --- |
| [27] | Chow (1970), *IEEE Transactions on Information Theory* | Derives the Bayes-optimal classification rule with a reject (abstention) option when posterior confidence falls below a cost-determined threshold $\tau$. | Mathematical foundation for abstaining (`Class 0`) when tail posterior probabilities fail to clear the \$0.50/MWh fee hurdle. |
| [28] | Fumera, Roli, & Giacinto (2000), *Pattern Recognition* | Generalizes Bayes-optimal reject rules to class-specific asymmetric thresholds ($\tau_{-1}, \tau_{+1}$). | Captures INC vs. DEC structural asymmetry in NYISO (consistent positive INC risk premia vs. severe summer heatwave DEC tail risk). |
| [29] | Elkan (2001) / Zadrozny & Elkan (2001), *IJCAI* & *ACM SIGKDD* | Proves that distorting class priors (`class_weight="balanced"`) biases posterior probabilities and requires probability calibration + cost-sensitive thresholding. | Explains why uncalibrated `class_weight="balanced"` classifiers over-trade (~69% of hours vs. 30% true tail rate), eroding net PnL per trade. |
| [30] | Sheng & Ling (2006), *AAAI* | Establishes empirical threshold tuning on held-out validation splits to align classifier decision boundaries with asymmetric misclassification costs. | Justifies tuning ($\tau_{-1}, \tau_{+1}$) strictly inside the inner validation window (`2024`) before freezing on `2025–2026` out-of-sample tests. |
| [31] | Xiao et al. (2021), *arXiv:2104.02754* | Pairs sequence models with monotonically constrained GBDTs to estimate self-induced Day-Ahead price impact across PJM, ISO-NE, and CAISO. | Demonstrates why price-taker assumptions overestimate Sharpe ratios compared to models accounting for virtual bid price elasticity. |
| [32] | Almgren & Chriss (2001), *Journal of Risk* | Formulates quadratic market-impact execution costs and closed-form optimal position sizing under linear price-impact schedules. | Provides the execution-theory foundation for the quadratic Day-Ahead bid-stack slope lot sizing formula $q_t^* = \min(q_{\max}, \max(0, |\hat{S}_t| - c) / 2\Lambda_t)$. |
| [33] | Bailey et al. (2014) / López de Prado (2018), *Advances in Financial Machine Learning* | Establishes strict chronological walk-forward validation, purged cross-validation gaps, and meta-labeling / abstention principles. | Justifies the 48-hour causal purge gap, `2020–2024` train / `2025–2026` test split, and vintage-filtered feature pipeline. |

---

### Notes & Potential Pitfalls to Flag in Prose

- **Separating Reconstruction from Classification ([17] vs. [18]):** Be explicit in Paragraph 4 that standard compressed sensing [17] optimizes sensor placement for full-state reconstruction ($\mathbf{x} \approx \boldsymbol{\Psi}\mathbf{a}$), whereas `SSPOC` [18] optimizes strictly for discriminant classification ($\boldsymbol{\Psi}_r^\top \mathbf{S} \approx \mathbf{W}$). This distinction explains why $p = 5$ to $20$ bottleneck generator nodes avoid the 559-node collinearity trap while preserving spatial congestion signals.
- **Why Autonomous DMD Fails Across the 48-Hour Auction Lag ([13–15]):** Emphasize that while Exact DMD and BOP-DMD cleanly extract spatial-temporal Koopman modes, modeling NYISO DART spreads as an *unforced* dynamical system across a 48-hour causal cutoff (`D-1 05:00 EST`) forces stable eigenvalues ($\text{Re}(\omega_k) \le 0$) to decay toward the trailing mean during calm regimes and overshoot during abrupt weather-driven mean-reversions.
- **Calibration Before Thresholding ([27–30]):** Note that tree ensembles (`GBDT`) and deep nets (`MLP`) trained on natural priors often produce overconfident probabilities, whereas regularized `Multinomial Logit` produces conservative probabilities. Pairing natural-prior classifiers with isotonic/Platt calibration before applying symmetric or asymmetric reject thresholds ($\tau_{-1}, \tau_{+1}$) ensures disciplined trade selectivity (~20–35% of hours).
- **Decoupling Signal Generation from Lot Sizing ([24, 31–32]):** Highlight the two-stage architecture: Stage 1 evaluates unit-1 MWh directional classification (`SSPOC` + classifier), and Stage 2 scales position size $q_t^*$ via causal Day-Ahead `P-27` bid-stack slopes ($\Lambda_{t-24}$) so size automatically contracts when tight peaker stacks steepen price impact.


---

### Notes & Potential Pitfalls to Flag in Prose

- Separating Reconstruction from Classification ([15] vs. [16]): Be explicit in Paragraph 4 that standard compressed sensing [15] optimizes sensor placement for full-state reconstruction ($\mathbf{x} \approx \boldsymbol{\Psi}\mathbf{a}$), whereas SSPOC [16] optimizes strictly for discriminant classification ($\boldsymbol{\Psi}_r^\top \mathbf{S} \approx \mathbf{W}$). This distinction is why p=15 or 20 nodes suffice to detect NYC spikes without reconstructing all 559 generator prices.
- Preprint vs. Peer-Reviewed Weighting ([13], [18], [22]): Note that [22] (arXiv:2601.05085) was published in Energy Economics in 2026—cite the peer-reviewed Energy Economics version in your final bibliography. Keep [13] and [18] paired alongside peer-reviewed foundational papers ([11, 16, 17]) so your empirical claims rest on published venues.
- Bridging Your Proposal to Your Code: Your initial proposal mentioned DMD regression as a baseline; pairing [12] and [13] with [6] and [9] seamlessly explains why you tested DMD regression first and why structural tail-spike clustering [3, 21] motivated your pivot to SSPOC classification.
  - [1] [Sparse Sensor Placement Optimization for Classification](https://www.researchgate.net/publication/309810376_Sparse_Sensor_Placement_Optimization_for_Classification)
  - [2] [Trading Electrons: Predicting DART Spread Spikes in ISO Electricity Markets](https://arxiv.org/html/2601.05085v3)


Would you like me to:
- Draft the full prose Literature Review section woven directly from these 5 paragraphs and citations so it flows seamlessly into your Section 3 Methodology?
- Build the PnL & Sharpe Ratio Backtest + Regression Baseline code in evaluate_models.py so you can generate your final experimental results table?

## Unused References

Removed from the final review because their intended role (establishing predecessor lineage for PPO and TD3) was not carried through to the prose.

| # | Paper | Key idea | Reason removed |
|---|-------|----------|----------------|
| | | *NONE* |