# Sparse Sensing and Discriminant Analysis for Interpretable NYISO DART Spread Trading — Literature Review Outline

This literature review anchors the project's core narrative ("From Continuous Forecasting & Spectral Decomposition to Sparse Spatial Regime Classification"), tracing why traditional continuous forecasting and unforced spectral models break down in two-settlement electricity markets, how separating lagged grid states from Day-Ahead exogenous controls and applying 30-day scheduled refitting stabilizes `DMDc`, how sparse spatial classification (`SSPOC`) bridges grid physics with algorithmic virtual trading using zero exogenous covariates, and how cost-sensitive decision rules and leak-free chronological validation govern net trading profitability.

### Narrative Arc

- **Paragraph 1** — Two-settlement wholesale market mechanics, virtual bidding (INC/DEC), and why DART spreads exhibit clustered, congestion-driven tail spikes [1–5].
- **Paragraph 2** — Traditional continuous electricity price forecasting (SARIMA, VAR, Ridge, ML/DL surrogates), direct spread modeling vs. subtraction, and the curse of dimensionality across 559 collinear generator nodes [6–11].
- **Paragraph 3** — Dynamical systems & modal decomposition lineage: POD/SVD → Exact DMD, Bagging-Optimized DMD (BOP-DMD), and DMD with Control (`DMDc`) → why unforced spectral extrapolation suffers from 48-hour eigenvalue damping, why Day-Ahead weather/load/outage forecasts belong in exogenous control ($\mathbf{u}_{t}$) rather than internal state ($\mathbf{x}_t$), and why static `DMDc` suffers winter regime-drift drawdowns that motivate 1-hour iterative rollout and 30-day scheduled expanding-window refitting [12–17].
- **Paragraph 4** — Sparse sensing & classification lineage: Compressed Sensing → `SSPOC` → predictive/time-delay sparse sensing → glass-box interpretability mapped onto physical NYISO transmission bottlenecks (`0/20` nodes inside Zone J) and high 30-day Jaccard sensor stability (`0.864`) without exogenous covariates [18–22].
- **Paragraph 5** — Distributional & spike-regime classification in power trading: reframing point regression as 3-class tail-spike detection (INC/DEC/Abstain) and modern ML benchmarks (Logit, GBDT, Transformers, MDNs) [23–27].
- **Paragraph 6** — Cost-sensitive thresholding, class-imbalance trade-offs, market-impact execution context, and leak-free chronological validation: Bayes-optimal reject rules, empirical threshold tuning on validation splits, unit-size (`1 MWh`) backtesting net of `$0.50/MWh` fees, and 48-hour causal purge gaps [28–34].

---

#### Paragraph 1 — Market Microstructure & Virtual Arbitrage Mechanics

| # | Paper | Key idea | Why included |
| --- | --- | --- | --- |
| [1] | Borenstein et al. (2002), *American Economic Review* | Empirical analysis of two-settlement power markets showing how transmission congestion and forecast errors drive systematic DA–RT price divergence. | Foundational economics reference establishing why DART spreads exist physically and structurally. |
| [2] | Longstaff & Wang (2004), *Journal of Finance* | Documents forward risk premia, INC/DEC asymmetry, and volatility clustering in day-ahead versus real-time electricity prices. | Establishes the financial theory behind virtual bidding risk premia and limits to arbitrage. |
| [3] | Christensen et al. (2012), *International Journal of Forecasting* | Demonstrates that extreme electricity price spikes are short-lived, clustered, and triggered by binding transmission constraints. | Proves that DART volatility is driven by discrete structural bottlenecks rather than smooth Gaussian noise. |
| [4] | NYISO Market Participants User's Guide / Virtual Trading Manual (2024) | Defines virtual supply (INC) and virtual demand (DEC) bidding rules, collateral requirements, and the 05:00 EST Day-Ahead market cutoff. | Grounds the strict causal filtration ($\mathcal{F}_{D-1,05:00}$ via a 48-hour lag) and transaction-cost hurdle ($c = \$0.50/\text{MWh}$) in real NYISO exchange rules. |
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

#### Paragraph 3 — Modal Decomposition, Exogenous Control (`DMDc`) & Scheduled Refitting

| # | Paper | Key idea | Why included |
| --- | --- | --- | --- |
| [12] | Schmid (2010), *Journal of Fluid Mechanics* | Introduces Dynamic Mode Decomposition (DMD) to extract coherent spatiotemporal modes from high-dimensional sequential snapshots. | Foundational algorithm reference for decomposing tall-and-skinny ($N \times T$) nodal/zonal state-space matrices. |
| [13] | Mann & Kutz (2016), *Quantitative Finance* | Adapts Exact DMD to financial time series to isolate coherent trading modes and execute systematic arbitrage strategies. | Direct methodological precedent connecting modal decomposition to quantitative trading signals. |
| [14] | Sashidhar & Kutz (2022), *Philosophical Transactions of the Royal Society A* | Introduces Bagging, Optimized DMD (BOP-DMD) using variable projection and statistical bagging to stabilize eigenvalue estimation under noise. | Benchmark spectral baseline showing that even stabilized Koopman eigenvalues remain unforced (`1.6%` trade frequency, `-$985.77` Cum PnL). |
| [15] | Proctor, Brunton, & Kutz (2016), *SIAM Journal on Applied Dynamical Systems* | Introduces Dynamic Mode Decomposition with Control (`DMDc`: $\mathbf{z}_{k+1} = \mathbf{A}_r\mathbf{z}_k + \mathbf{B}_r\mathbf{u}_k$) to disambiguate autonomous system dynamics from external actuation/forcing. | Mathematically grounds separating lagged delay-embedded zonal states ($\mathbf{x}_t$) from 424 Day-Ahead exogenous forecasts ($\mathbf{u}_{t}$), preserving unidirectional causality, causal filtration, and rank-12 POD congestion modes. |
| [16] | Dylewsky et al. (2020), *arXiv:2010.04248* | Develops stochastically forced ensemble DMD in time-delay coordinates to forecast near-periodic power grid load dynamics. | Explains why autonomous DMD decays toward the trailing mean across a 48-hour auction lag ($\text{Re}(\omega_k) \le 0$) and overshoots during abrupt reversals, and why static `DMDc` fails during winter regime shifts unless rolled iteratively and refit on a 30-day schedule. |
| [17] | Mohan et al. (2018), *IEEE Transactions on Power Systems* | Applies DMD and Koopman operators to wide-area power grid disturbance monitoring using limited measurements. | Connects modal decomposition directly to wide-area power grid bottleneck detection and sets up sparse sensing. |

#### Paragraph 4 — Sparse Sensor Placement (`SSPOC`) & Glass-Box Grid Physics

| # | Paper | Key idea | Why included |
| --- | --- | --- | --- |
| [18] | Candès & Wakin (2008), *IEEE Signal Processing Magazine* | Establishes Compressed Sensing theory: high-dimensional sparse signals can be recovered from far fewer measurements via $\ell_1$ optimization. | Mathematical bedrock explaining why full-state grid reconstruction (`QR-Pivoting`) is unnecessary when underlying congestion patterns are low-rank. |
| [19] | Brunton, Brunton, Proctor & Kutz (2016), *SIAM Journal on Applied Mathematics* | Introduces `SSPOC`: combines POD/SVD, LDA discriminant projection, and $\ell_1$ sparse optimization to place minimal sensors for classification. | Centerpiece algorithm paper—proves classification requires orders-of-magnitude fewer sensors ($p=20$ of 559 buses, `3.6%`) than full-state reconstruction, achieving `3.56` Sharpe (`+$35,246.10` Cum PnL) with zero exogenous features. |
| [20] | Manohar, Brunton, Kutz & Brunton (2018), IEEE Control Systems Magazine | Extends data-driven sparse sensor placement via QR pivoting and time-delay coordinates for dynamic state estimation and forecasting. | Provides the mathematical baseline (QR-Pivoting) contrasted against discriminative SSPOC in our whole-grid reconstruction vs. trading Sharpe Pareto ablation. |
| [21] | Manohar, Hogan, Kutz & Brunton (2018), arXiv:1804.07884 | Demonstrates neural-inspired sparse sensor architectures that preserve classification accuracy under severe dimensionality reduction (>95%). | Supports experimental comparisons pairing SSPOC, QR-Pivoting, and Mutual-Info sensors with linear (Multinomial Logit, LDA) and non-linear (GBDT, Balanced MLP) classifiers. |
| [22] | Rudin (2019), Nature Machine Intelligence | Argues for inherently interpretable ("glass-box") models over post-hoc explanations in high-stakes physical and financial systems. | Grounds our physical grid interpretability findings: SSPOC places 0/20 nodes inside N.Y.C. and instead isolates invariant upstream bottlenecks (HUNTINGTON_RES_REC, NINE_MILE_1 & 2, CRICKET___VALLEY_CC3, PJM_GEN_VFT_PROXY) with 0.864 30-day Jaccard stability. |


#### Paragraph 5 — Tail-Spike Regime Classification & Quantitative Trading Benchmarks


| # | Paper | Key idea | Why included |
| --- | --- | --- | --- |
|[23] | Galarneau-Vincent et al. (2023), Energy Economics | Models hourly DART spread spike probabilities in NYISO's Long Island zone using GBDT, RF, MLP, and Logit with load, weather, and calendar features. | Direct NYISO empirical benchmark confirming ~53–58% win rates and warning that raw full-grid nodal features degrade linear/tree generalization.|
|[24] | Deschatre & Gruet (2022), Quantitative Finance | Uses Gaussian–Generalized Pareto mixture and regime-switching models to separate normal market hours from extreme price tails. | Theoretical justification for splitting the continuous DART distribution into a central neutral zone (Class 0) and extreme tails (±1, 15th/85th percentiles).|
|[25] | Hubert, Lolas, & Sircar (2026), Energy Economics / arXiv:2601.05085 | Frames INC/DEC virtual bidding across NYISO, ISO-NE, and ERCOT as a joint spike classification and optimal bid-sizing problem net of transaction costs. | Primary literature benchmark (~53–58% win rate, 1.4–3.2 Sharpe) validating multi-zone spike classification (N.Y.C., LONGIL, HUD VL) and cost-aware evaluation.|
|[26] | Lee et al. (2024), arXiv:2412.00062 | Evaluates temporal Transformer architectures that quantize DART spreads into discrete categorical bins using multi-horizon load, wind, and solar forecasts. | Demonstrates how discretizing continuous spreads into categorical regimes filters low-edge hours during peak congestion ramps.|
|[27] | Yu et al. (2019), IEEE Transactions on Power Systems | Introduces Mixture Density Networks (MDNs) paired with Conditional Value-at-Risk (CVaR) to optimize virtual bid portfolios across correlated nodes. | Contextualizes generative probabilistic approaches for tail-risk control across multi-node electricity portfolios.|


#### Paragraph 6 — Cost-Sensitive Thresholding, Class Imbalance, Execution Context & Backtest Hygiene


| # | Paper | Key idea | Why included |
| --- | --- | --- | --- |
|[28] | Chow (1970), IEEE Transactions on Information Theory | Derives the Bayes-optimal classification rule with a reject (abstention) option when posterior confidence falls below a cost-determined threshold $\tau$. | Mathematical foundation for abstaining (Class 0) when tail confidence fails to clear the $0.50/\text{MWh}$ fee hurdle.|
|[29] | Fumera, Roli, & Giacinto (2000), Pattern Recognition | Generalizes Bayes-optimal reject rules to class-specific asymmetric thresholds ($\tau_{-1}, \tau_{+1}$). | Captures INC vs. DEC structural asymmetry in NYISO and motivates inner-validation threshold tuning (tau).|
|[30] | Elkan (2001) / Zadrozny & Elkan (2001), IJCAI & ACM SIGKDD | Proves that distorting class priors (class_weight="balanced") shifts decision boundaries toward minority tail classes and interacts with posterior probability calibration. | Explains why class_weight="balanced" increases tail sensitivity (48.9%–68.6% trade frequency on orthogonalized SSPOC sensors) whereas natural-prior models require explicit threshold tuning (tau).|
|[31] | Sheng & Ling (2006), AAAI | Establishes empirical threshold tuning on held-out validation splits to align classifier decision boundaries with asymmetric misclassification costs. | Justifies tuning reject thresholds and DMDc dollar hurdles strictly inside the inner validation window (2024) before freezing on 2025–2026 out-of-sample tests.|
|[32] | Xiao et al. (2021), arXiv:2104.02754 | Pairs sequence models with monotonically constrained GBDTs to estimate self-induced Day-Ahead price impact across PJM, ISO-NE, and CAISO. | Contextualizes price-taker vs. price-maker execution limits in wholesale auctions, explaining why our backtests standardize on a 1 MWh unit position ($0.50/MWh fee) to isolate signal quality from bid-stack price impact.|
|[33] | Almgren & Chriss (2001), Journal of Risk | Formulates quadratic market-impact execution costs and closed-form optimal position sizing under linear price-impact schedules. | Provides the theoretical framework for future multi-MW scaling via Day-Ahead P-27 bid-stack slopes without confounding Stage 1 sensor evaluation.|
|[34] | Bailey et al. (2014) / López de Prado (2018), Advances in Financial Machine Learning | Establishes strict chronological walk-forward validation, purged cross-validation gaps, and expanding-window retraining protocols. | Justifies the 48-hour causal purge gap, 2020–2024 train / 2025–2026 OOS split, vintage-filtered feature pipeline, and 30-day (720h) scheduled expanding-window refits.|


---

### Key Methodological Distinctions Verified Across Notebooks 01–04

- **Internal Grid State ($\mathbf{x}_t$) vs. Exogenous Control ($\mathbf{u}_{t}$) in DMDc** [15]: Day-Ahead weather, load, solar, and scheduled outage forecasts (424 features) are modeled as external controls ($\mathbf{u}_{t}$) rather than stacked into the internal state vector ($\mathbf{x}_t$):
	1. *Asymmetric Causal Filtration ($\mathcal{F}_{D-1, 05:00}$)*: Realized zonal/nodal spreads ($\mathbf{x}_t$) are only observable up to $t - 48\text{h}$ prior to the 05:00 EST Day-Ahead market cutoff, whereas Day-Ahead weather, load, solar, and outage schedules ($\mathbf{u}_{t}$) are already published for every intermediate hour $k \in [t-48, t-1]$ up to target delivery hour $t$.
	2. *Unidirectional Physical Causality*: Weather fronts, solar irradiance, and calendar demand force the power grid without price feedback.
	3. *Subspace & Spectral Rank Preservation*: Keeping 424 high-variance diurnal weather/load features out of $\mathbf{x}_t$ ensures the rank-12 POD basis captures spatial grid dynamics rather than smooth 24-hour temperature cycles.

- **Why Static DMDc Fails & How 1-Hour Iterative Rollout + 30-Day Refitting Solves It** [13–16]: Unforced Exact DMD and BOP-DMD suffer from 48-hour eigenvalue damping (1.6% trade frequency, -$985.77 Cum PnL). Static 48h-jump DMDc (frozen in Dec 2024) captures 2025 premia (+$35.8k peak) but suffers a -$37,920.72 regime-drift drawdown during the Jan–Feb 2026 winter cold snap. Switching to a 48-step 1-hour iterative DMDc rollout ($\mathbf{z}_{k+1} = \mathbf{A}_r\mathbf{z}_k + \mathbf{B}_r\mathbf{u}_k$) with 30-day (720h) scheduled expanding-window refits cuts Max Drawdown to $5,181.87 and achieves 3.12 Annual Sharpe (+$24,111.80 Cum PnL).

- **Separating Reconstruction from Classification & Zero-Exogenous Sparse Sensing** ([18] vs. [19]): Standard compressed sensing / QR-Pivoting [18, 20] optimizes sensor placement for whole-grid reconstruction (Grid R² = 0.945 at $p=40$ vs. 0.196 for SSPOC), whereas SSPOC [19] optimizes strictly for discriminant tail-regime separation. Using only $p=20$ sparse generator buses (3.6% of the 559-bus grid) and zero exogenous variables, Scheduled 30-Day Refit SSPOC achieves 3.56 Annual Sharpe (+$35,246.10 Cum PnL, 0.4167 Macro F1), outperforming both Scheduled 30-Day Refit DMDc and full-grid baselines.

- **Invariant Physical Sensors + Scheduled Classifier Refitting** [19, 22, 34]: Across 23 monthly refits in 2025–2026, SSPOC ($p=20$) exhibits 0.864 mean step-to-step Jaccard stability and places 0/20 sensors inside N.Y.C., instead locking onto upstream bottlenecks (HUNTINGTON_RES_REC, NINE_MILE_1 & 2, CANDIGU_WT_PWR at 100% persistence; CRICKET___VALLEY_CC3 and PJM_GEN_VFT_PROXY at 91.3%). Keeping the 20 upstream bellwether sensors fixed while retraining the Multinomial Logit (class_weight="balanced") classifier every 30 days (3.56 Sharpe) outperforms dynamic monthly sensor re-selection (2.21 Sharpe) by preventing short-term noise from pulling collinear in-zone N.Y.C. units into the active sensor set.

- **Unit-Size (1 MWh) Signal Isolation** [25, 32, 33]: All backtests evaluate standardized 1 MWh positions net of the $0.50/MWh NYISO transaction fee to isolate directional signal quality and sensor sparsity from Day-Ahead bid-stack price impact.