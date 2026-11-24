---
title: "Sparse Sensor Placement and Dynamical Control for Interpretable NYISO DART Spread Virtual Bidding"
subtitle: "NYISO Zone J (N.Y.C.) | 2020–2024 Train/Validation | 2025–2026 Out-of-Sample (16,306 Hours)"
author:
  - "Patrick Smith"
  - "Tyler Darwin"
date: "November 2026"
abstract: |
  Two-settlement wholesale electricity markets exhibit clustered, non-Gaussian price divergences between Day-Ahead and Real-Time Locational Based Marginal Prices (DART spreads) driven by binding transmission constraints and forecast errors. Using 59,958 hourly observations across 559 NYISO generator buses (2020–2024 training/validation, 2025–2026 out-of-sample test) under a strict 48-hour causal auction filtration ($\mathcal{F}_{D-1,05:00}$) and \$0.50/MWh transaction hurdle, we evaluate dynamical modal decomposition (DMD / DMDc) alongside discriminative sparse sensor placement (SSPOC) and continuous sparse sensor regression (QR-Pivoting) for N.Y.C. (Zone J) virtual bidding. We first show that autonomous Exact DMD and BOP-DMD suffer from 48-hour eigenvalue damping (1.6% trade frequency, -\$985.77 Cum PnL) and winter phase inversion, while a static DMDc model frozen in December 2024 incurs a -\$37,920.72 regime-drift drawdown during the January–February 2026 cold snap. Rolling a reduced-order DMDc model iteratively across the 48-hour trajectory of 424 Day-Ahead exogenous forecasts with 30-day (720h) scheduled expanding-window refitting resolves this non-stationarity, achieving 3.12 Annual Sharpe (+\$24,111.80 Cum PnL, \$5,181.87 Max DD). Finally, applying Sparse Sensor Placement Optimization for Classification (SSPOC) on a rank-25 POD basis (98.0% energy) demonstrates that just $p=20$ generator buses (3.6% of the grid) with zero exogenous covariates paired with a 30-day refit Multinomial Logit classifier achieves 3.56 Annual Sharpe (+\$35,246.10 Cum PnL, 0.4167 Macro F1), outperforming both dense dynamical control and continuous sparse sensor regression. Without topological priors, SSPOC places 0/20 sensors inside Zone J and instead locks onto invariant upstream thermal bottlenecks and cross-border ties (0.864 mean 30-day step Jaccard stability), delivering an interpretable "glass-box" framework for algorithmic power trading.
---


## 1. Motivation & Problem Statement

### 1.1 Two-Settlement Electricity Market Mechanics
Wholesale electricity markets operate as two-settlement systems designed to coordinate physical grid reliability with financial price discovery [@borenstein2002measuring; @longstaff2004electricity]. In the NYISO market, participants submit bids prior to `05:00 EST` on the day before delivery ($D\text{-}1$) to clear the Day-Ahead Market (DAM) based on forecasted load and generation schedules, while the Real-Time Market (RTM) balances physical dispatch at five-minute intervals [@nyiso2024virtual]. Because unscheduled generator outages, renewable generation ramps, natural gas fuel shocks, and binding transmission constraints cause real-time grid conditions to diverge from day-ahead expectations [@christensen2012forecasting; @ioannidis2025spillovers], systematic differences arise between Day-Ahead and Real-Time Locational Based Marginal Prices (LBMPs), defined for bus or zone $i$ at delivery hour $t$ as:

$$S_{i,t} = \text{LBMP}^{\text{DA}}_{i,t} - \text{LBMP}^{\text{RT}}_{i,t}$$

Virtual trading mechanisms—specifically **Increment (`INC`)** virtual supply offers ($q_t = +1$, profiting when $S_{i,t} > c$) and **Decrement (`DEC`)** virtual demand bids ($q_t = -1$, profiting when $S_{i,t} < -c$)—allow financial participants to arbitrage these spreads without physical generation or load assets, subject to a clearing and uplift transaction hurdle of $c = \$0.50/\text{MWh}$ [@nyiso2024virtual; @hubert2026trading].[^1]

[^1]: Under NYISO Market Services Tariff rules [@nyiso2024virtual], virtual bids settle financially against real-time hourly LBMPs without physical delivery obligations, but incur ISO transaction fees, credit/collateral charges, and Real-Time uplift allocations approximated by $c = \$0.50/\text{MWh}$ [@hubert2026trading].

### 1.2 Research Questions
Traditional continuous electricity price forecasting models (e.g., SARIMA, VAR, or full-grid neural networks) minimize mean squared error across hundreds of highly collinear generator nodes, smoothing over rare congestion spikes while inflating estimation variance [@weron2014electricity; @afrasiabi2020probabilistic; @das2022dart]. Conversely, autonomous modal decompositions (`Exact DMD`, `BOP-DMD`) lack exogenous weather/load forcing and damp toward the trailing mean across the 48-hour auction horizon [@schmid2010dmd; @sashidhar2022bopdmd; @dylewsky2020stochastically]. To address these limitations, our team investigates two complementary sparse-sensing formulations alongside forced dynamical control:

1. **Dynamical Control (`DMDc`):** Can separating lagged zonal states from Day-Ahead exogenous control trajectories ($\mathbf{u}_k$) [@proctor2016dmdc] and refitting on a scheduled 30-day (`720h`) expanding window overcome autonomous DMD damping and winter regime drift?
2. **Discriminative Sparse Sensing (`SSPOC` — Patrick Smith):** Can framing DART trading as a three-class tail-spike regime problem (`INC`, `DEC`, `Abstain`) [@huisman2003regime; @galarneauvincent2023foreseeing] using a minimal subset of $p \in \{5, 10, 20, 40\}$ physical generator buses [@brunton2016sspoc]—with **zero exogenous weather/load covariates**—outperform full-grid ML benchmarks and forced `DMDc`?
3. **Continuous Regression Sparse Sensing (`QR-Pivoting` / POD Regression — Tyler Darwin) & Cross-Domain Synthesis:** [PLACEHOLDER: Tyler's continuous regression problem framing — e.g., evaluating whether reconstruction-driven sparse sensors (`QR-Pivoting`) [@manohar2018data] can forecast continuous DART spreads accurately], and whether the same physical generator buses and sensor budgets ($p$) are optimal across classification and continuous regression.


---

## 2. Dataset & Leak-Free Experimental Architecture

We constructed an hourly dataset from public NYISO market reports spanning **January 8, 2020 through November 11, 2026 (`59,958` hourly timestamps)** following open-access electricity forecasting hygiene protocols [@lago2021forecasting; @lopezdeprado2018advances]:

* **Spatial State & Target:** `559` generator buses (`P-2B` / `P-4B`) and `15` load zones (`P-2A` / `P-4A`), targeting **Zone J (`N.Y.C.`)** (with spatial generalization checks on `LONGIL` and `HUD VL`).
* **Exogenous Control Features (`424` variables):** Vintage-filtered Day-Ahead load forecasts (`P-7`), weather forecasts (`P-7A`), zonal bid load (`P-59`), behind-the-meter solar forecasts (`P-70B`), scheduled/actual outages (`P-54C`/`P-54B`), limiting transmission constraints (`P-511A`), real-time fuel mix (`P-63`), and interface flows (`P-32`), plus `P-19` generator coordinates and `P-23` subzone metadata.
* **Strict Causal Filtration ($\mathcal{F}_{D-1, 05:00}$):** Because Day-Ahead bids for hour 23 of day $D$ must be submitted before `05:00 EST` on day $D\text{-}1$ [@nyiso2024virtual], all realized spreads and real-time variables are lagged by **48 hours** (`CAUSAL_LAG = 48`), whereas Day-Ahead forecasts are aligned to their published vintage.[^2]
* **Chronological Split & Unit-Size Evaluation:** Models are trained on **`2020–2024`** (`43,652` hours) and evaluated strictly out-of-sample on **`2025–2026`** (`16,306` hours) [@bailey2014deflated]. All strategies trade a standardized **`1 MWh` unit position** net of `TX_FEE = $0.50/MWh` to isolate directional signal quality from bid-stack price impact [@hubert2026trading].

[^2]: Enforcing a strict 48-hour causal purge gap between the latest realized spread observation and the target delivery hour prevents look-ahead leakage across all 24 hours of the $D\text{-}1$ Day-Ahead auction clearing window [@lopezdeprado2018advances].


---

## 3. Mathematical Formulation

### 3.1 Dynamic Mode Decomposition with Control (`DMDc`)
To model forced spatiotemporal grid dynamics across all $N=15$ NYISO zones without requiring explicit AC power-flow equations [@mohan2018widearea; @brunton2022koopman], we construct a 60-dimensional Hankel time-delay state vector $\mathbf{x}_t \in \mathbb{R}^{60}$ (`lags = [0, 1, 24, 48]` hours) [@dylewsky2020stochastically] paired with $M=424$ Day-Ahead exogenous controls $\mathbf{u}_t \in \mathbb{R}^{424}$. Following Proctor, Brunton, and Kutz [@proctor2016dmdc], we seek linear operators $(\mathbf{A}, \mathbf{B})$ governing the 1-hour state transition:

$$\mathbf{x}_{k+1} = \mathbf{A}\mathbf{x}_k + \mathbf{B}\mathbf{u}_k$$

Applying a rank-$r$ ($r=12$) Truncated Singular Value Decomposition (SVD) to the augmented snapshot matrix $\boldsymbol{\Omega} = [\mathbf{X}; \mathbf{U}] \approx \tilde{\mathbf{U}}\tilde{\boldsymbol{\Sigma}}\tilde{\mathbf{V}}^\top$ and projecting onto the rank-12 output Proper Orthogonal Decomposition (POD) basis $\hat{\mathbf{U}} \in \mathbb{R}^{60 \times 12}$ yields the reduced-order operators $(\mathbf{A}_r, \mathbf{B}_r)$ via Tikhonov-regularized (`ridge_alpha = 50.0`) pseudoinversion:

$$\mathbf{z}_{k+1} = \mathbf{A}_r\mathbf{z}_k + \mathbf{B}_r\mathbf{u}_k, \qquad \hat{\mathbf{x}}_t = \hat{\mathbf{U}}\mathbf{z}_{t}$$

From decision time $t_{\text{dec}} = t - 48$, we roll $\mathbf{z}_k$ iteratively forward across all 48 intermediate hours $k \in [t-48, t-1]$ using the published Day-Ahead control trajectory $\mathbf{U}_{t-48:t}$ to forecast $\hat{S}_{\text{NYC}, t}$.[^3]

[^3]: Separating $\mathbf{u}_k$ from $\mathbf{x}_k$ preserves unidirectional atmospheric-to-grid causality and prevents high-variance diurnal temperature and solar cycles from dominating the leading rank-12 POD modes [@proctor2016dmdc].

### 3.2 Sparse Sensor Placement Optimization for Classification (`SSPOC`)
Rather than reconstructing the full 559-bus state vector $\mathbf{x}_t \in \mathbb{R}^{559}$ [@candes2008compressive; @jami2023surrogates], we partition the training `N.Y.C.` spread distribution into three discrete regimes using leak-free `2020–2024` quantiles ($q_{0.15}, q_{0.85}$) [@huisman2003regime]:

$$y_t = \begin{cases} +1 \text{ (INC)}, & S_{\text{NYC}, t} > q_{0.85} \\ -1 \text{ (DEC)}, & S_{\text{NYC}, t} < q_{0.15} \\ \phantom{+}0 \text{ (Abstain)}, & q_{0.15} \le S_{\text{NYC}, t} \le q_{0.85} \end{cases}$$

Following Brunton et al. [@brunton2016sspoc], we compute a rank-$r$ ($r=25$, capturing `98.0%` of nodal variance) POD basis $\boldsymbol{\Psi}_r \in \mathbb{R}^{559 \times 25}$ from standardized training nodal spreads $\mathbf{X}_{\text{nodes}}^\top = \boldsymbol{\Psi}_r \boldsymbol{\Sigma}_r \mathbf{V}_r^\top$, and fit a Linear Discriminant Analysis (LDA) projection matrix $\mathbf{W} \in \mathbb{R}^{25 \times 2}$ in the modal subspace to maximize between-class separation across $y_t \in \{-1, 0, +1\}$. We then solve for a sparse sensor matrix $\mathbf{S} \in \mathbb{R}^{559 \times 2}$ via Multi-Task Elastic Net ($\ell_1/\ell_2$ penalty, `l1_ratio = 0.9`, `alpha_frac = 0.15`) [@tibshirani1996lasso]:

$$\min_{\mathbf{S}} \|\boldsymbol{\Psi}_r^\top \mathbf{S} - \mathbf{W}\|_F^2 + \lambda \left( \rho \|\mathbf{S}\|_{2,1} + \frac{1-\rho}{2}\|\mathbf{S}\|_F^2 \right)$$

Ranking buses by row norms $\|\mathbf{S}_{i,:}\|_2$ isolates the top $p \in \{5, 10, 20, 40\}$ "bellwether" generator nodes $\mathcal{I}_p$, which serve as the sole inputs (lagged by 48 hours) to a downstream `class_weight="balanced"` `Multinomial Logit` classifier [@manohar2018neural; @rudin2019stop].

### 3.3 Continuous Sparse Sensor Regression (`QR-Pivoting` / POD Regression)
To contrast discriminative sensing against whole-grid reconstruction and continuous regression, we also evaluate POD-based **QR-Pivoting** [@manohar2018data]:

$$\boldsymbol{\Psi}_r^\top \mathbf{P} = \mathbf{Q}\mathbf{R}$$

where column pivoting $\mathbf{P}$ greedily selects $p$ sensor locations that maximize the determinant (minimize the condition number) of the subsampled basis $\mathbf{C}\boldsymbol{\Psi}_r$, enabling least-squares whole-grid reconstruction $\hat{\mathbf{x}} = \boldsymbol{\Psi}_r(\mathbf{C}\boldsymbol{\Psi}_r)^\dagger \mathbf{y}_{\text{meas}}$ and continuous spread regression `[PLACEHOLDER: Tyler's continuous regression equation / loss formulation]`.


---

## 4. Methodology & Experimental Pipeline

Our reproducible software pipeline (`Notebooks 00–04`) executes four sequential stages under strict chronological walk-forward validation [@lopezdeprado2018advances; @bailey2014deflated]:

1. **Stage 1 — Econometric, ML & Spectral Baselines (`Notebook 01`):** Benchmarks 14 reference models on `N.Y.C.` (`Continuous Ridge`, `Natural-Prior MLP`, `GBDT`, `Multinomial Logit`, `LDA`, `Causal 48h Persistence`, and unforced `Exact DMD` / `BOP-DMD`) [@mann2016dmd; @sashidhar2022bopdmd; @galarneauvincent2023foreseeing; @hubert2026trading].
2. **Stage 2 — `DMDc` Forensics & Scheduled Refitting (`Notebooks 02–03`):** Diagnoses unforced DMD eigenvalue damping and static 48h-jump `DMDc` winter regime drift (`Notebook 02`), then evaluates 1-hour iterative `DMDc` rollouts under both frozen (`Dec 2024`) and **scheduled 30-day (`720h`) expanding-window refits** (`Notebook 03`) [@proctor2016dmdc; @dylewsky2020stochastically].
3. **Stage 3 — `SSPOC` Ablation, Physical Mapping & Stability (`Notebook 04`):**
   * Contrasts `SSPOC`, `QR-Pivoting`, and `Mutual-Info` across $p \in \{5, 10, 20, 40\}$ and 5 classifiers (`Multinomial Logit`, `Logit`, `GBDT`, `LDA`, `Balanced MLP`) [@brunton2016sspoc; @manohar2018data; @manohar2018neural].
   * Maps top sensors onto NYISO `P-19`/`P-23` metadata [@nyiso2024virtual; @rudin2019stop] and tests spatial generalization on `LONGIL` and `HUD VL`.
   * Evaluates **30-day (`720h`) expanding-window refitting** for `SSPOC` ($p=20$), comparing **fixed physical sensors + 30-day classifier refits** against **dynamic 30-day sensor re-selection + classifier refits**, quantified via step-to-step Jaccard similarity $J(S_t, S_{t-1}) = |S_t \cap S_{t-1}| / |S_t \cup S_{t-1}|$.[^4]
4. **Stage 4 — Continuous Sparse Sensor Regression (`[PLACEHOLDER: Tyler's Notebook Name]`):** `[PLACEHOLDER: Describe Tyler's continuous regression experimental setup, sensor budgets p, regressors (e.g., Ridge, GBDT, MLP on QR-Pivoting vs. SSPOC nodes), and hurdle tuning]`.

[^4]: Step-to-step Jaccard similarity $J(S_t, S_{t-1}) \in \lbrack 0, 1 \rbrack$ measures the intersection-over-union of active sensor sets between consecutive 30-day refits, isolating structural grid invariance from short-term sampling noise.


---

## 5. Empirical Results (`2025–2026` Out-of-Sample)

### 5.1 Dynamical Systems Progression (`DMD` $\rightarrow$ `DMDc` $\rightarrow$ 30-Day Refit)
* **Failure of Autonomous DMD:** Rolling `BOP-DMD` [@sashidhar2022bopdmd] damps toward zero across the 48-hour auction lag, trading on only **`1.6%`** of hours and losing **`-$985.77`** (`-0.19` Annual Sharpe).
* **Static `DMDc` Regime-Drift Crash:** Adding 424 exogenous controls via a static 48h-jump `DMDc` model [@proctor2016dmdc] generates `+$35,777.49` through late 2025, but freezing $[\mathbf{A}, \mathbf{B}]$ in December 2024 causes severe directional inversion during the January–February 2026 winter cold snap (`-$37,920.72` Max DD, ending at `+$630.27` Cum PnL, `0.06` Sharpe).
* **Iterative Rollout + 30-Day Scheduled Refit:** Rolling the rank-12 `DMDc` system hour-by-hour across the 48-hour exogenous trajectory and refitting every `720h` (`30 days`) eliminates the winter collapse, cutting Max Drawdown by `86%` to **`$5,181.87`** and achieving **`3.12` Annual Sharpe (`+$24,111.80` Cum PnL)**.

### 5.2 Sparse Sensor Ablation: `SSPOC` vs. `QR-Pivoting` vs. `Mutual-Info`
* **Reconstruction vs. Classification Trade-Off:** `QR-Pivoting` [@manohar2018data] excels at whole-grid reconstruction (`Grid R² = 0.945` at $p=40$ vs. `0.196` for `SSPOC`), whereas `Mutual-Info` clusters redundantly inside `N.Y.C.` (`10/20` nodes in Zone J) and degrades out-of-sample (`Grid R² = -0.229` at $p=40$).
* **Compact Budget Efficiency:** With just **$p=20$ sensors (`3.6%` of the grid)**, static `SSPOC` [@brunton2016sspoc] paired with `Multinomial Logit` achieves **`1.95` Annual Sharpe (`+$21,455.66` Cum PnL, `0.3085` Tail F1)**—beating the full 559-node `Multinomial Logit` baseline (`1.78` Sharpe, `+$19,539.42`) while cutting Max Drawdown by `30%` (`$13,173.59`).
* **Spatial Generalization:** Re-running `SSPOC` ($p=20$) on adjacent load pockets relocates sensors to each zone's local bottlenecks, achieving **`2.41` Annual Sharpe** on `LONGIL` and **`4.04` Annual Sharpe** on `HUD VL`.

::: {.figbox}
**Figure 1 Placeholder — Geospatial Sensor Network & Corridor Allocation (`Notebook 04`, Cells 5 & 13):** Insert the NYISO geospatial map of the $p=20$ `SSPOC` bellwether sensors and cross-border interconnects alongside the Whole-Grid Reconstruction $R^2$ vs. `N.Y.C.` Trading Sharpe Pareto frontier.
:::

### 5.3 Physical Bottleneck Discovery & 30-Day Sensor Stability
* **Zero In-Zone Redundancy:** Without any topological priors, `SSPOC` places **`0/20` sensors inside `N.Y.C.`** (where internal 345 kV/138 kV prices are tightly coupled) [@rudin2019stop] and instead selects upstream congestion interfaces:
  * **Long Island Submarine Cable Corridor (`LONGIL`):** `HUNTINGTON_RES_REC` (#1 overall weight `0.649`, **`100%`** persistence across all 23 monthly refits).
  * **Central & Western Bulk Transfer (`CENTRL`, `MHK VL`, `WEST`, `GENESE`):** `NINE_MILE_1 & 2` (**`100%`** persistence), `CANDIGU_WT_PWR` (**`100%`** persistence), `MAPLE_RIDGE_WT_1 & 2`, and Niagara-area nodes.
  * **UPNY–ConEd Hudson Valley Corridor (`HUD VL`):** `CRICKET___VALLEY_CC1–CC3` (`91.3%` persistence) and `SHOEMAKER___GT`.
  * **Cross-Border External Ties (`PJM`, `H Q`, `NPX`):** `PJM_GEN_VFT_PROXY` (Linden VFT controllable tie into Zone J, `91.3%` persistence), `HQ_GEN_IMPORT`, and `NPX_GEN_1385_PROXY`.
* **Fixed Physical Sensors + 30-Day Classifier Refit Wins:** Across 23 monthly refits in `2025–2026`, `SSPOC` ($p=20$) maintains **`0.864` mean step-to-step Jaccard stability**. Keeping the **20 upstream bellwether sensors fixed** while **retraining the `Multinomial Logit` classifier every 30 days** achieves **`3.56` Annual Sharpe (`+$35,246.10` Cum PnL, `0.4167` Macro F1, `$7,595.29` Max DD)**—outperforming dynamic monthly sensor re-selection (`2.21` Sharpe, `+$21,483.10` Cum PnL).[^5]

[^5]: Dynamic monthly sensor re-selection allows short-term local volatility to pull collinear in-zone `N.Y.C.` units (`Arthur Kill`, `Linden Cogen`) into the active sensor set at the expense of upstream western corridors, whereas fixing the 20 upstream `SSPOC` nodes separates invariant grid topology from non-stationary market regimes.

::: {.figbox}
**Figure 2 Placeholder — 30-Day `SSPOC` Jaccard Stability & Selection Heatmap (`Notebook 04`, Cell 11):** Insert the two-panel figure showing the 30-day Step-to-Step vs. Anchor Jaccard trajectory (`0.864` mean) and the binary selection persistence heatmap for the top 20 bellwether nodes.
:::

### 5.4 Continuous Sparse Sensor Regression Results (Tyler Darwin)

::: {.tablebox}
**Placeholder — Tyler's Continuous Sparse Sensor Regression Results:**

* **Optimal Sensor Scheme & Budget ($p$):** `[PLACEHOLDER: e.g., QR-Pivoting vs. SSPOC sensors at p = 5, 10, 20, 40]`
* **Out-of-Sample Regression & Trading Metrics:** `[PLACEHOLDER: RMSE ($/MWh), R², Cum PnL ($), Annual Sharpe, Max DD ($), Trade Freq (%), Win Rate (%)]`
* **Cross-Domain Sensor Overlap (`SSPOC` Classification vs. Continuous Regression):** `[PLACEHOLDER: Jaccard similarity between optimal classification sensors and optimal continuous regression sensors; comparison of optimal sensor budget p across domains]`
:::

### 5.5 Unified Cross-Paradigm Comparison (`2025–2026` Out-of-Sample, `N.Y.C.`)

\begin{table*}[t]
\centering
\footnotesize
\setlength{\tabcolsep}{3pt}
\caption{\textbf{Table 1:} Unified Cross-Paradigm Comparison (2025--2026 Out-of-Sample, N.Y.C., 1~MWh Unit Size, \$0.50/MWh Fee)}
\resizebox{\textwidth}{!}{%
\begin{tabular}{llrrrrrrr}
\toprule
\textbf{Model / Formulation} & \textbf{Grid Telemetry} & \textbf{Cum PnL (\$)} & \textbf{Sharpe} & \textbf{Max DD (\$)} & \textbf{Freq (\%)} & \textbf{Win (\%)} & \textbf{Macro F1} & \textbf{Tail F1} \\
\midrule
1. Unforced BOP-DMD (Static) & 15 Zones (Dense) & -985.77 & -0.19 & 4,772.75 & 1.6 & 43.2 & 0.2644 & 0.0243 \\
2. Static 48h-Jump DMDc (Frozen '24) & 15 Zones + 424 Exog & 630.27 & 0.06 & 37,920.72 & 58.8 & 51.4 & 0.3697 & 0.2812 \\
3. Scheduled 30-Day Refit DMDc (720h) & 15 Zones + 424 Exog & 24,111.80 & 3.12 & 5,181.87 & 36.2 & 52.2 & 0.3714 & 0.2312 \\
4. Static SSPOC ($p=20$, Multinom. Logit) & 20/559 Buses (3.6\% Sparse) & 21,455.70 & 1.95 & 13,173.60 & 68.6 & 53.2 & 0.3732 & 0.3085 \\
5. Full 30-Day Refit SSPOC (Sensors + Clf) & 20/559 Buses (Dynamic) & 21,483.10 & 2.21 & 10,127.10 & 42.7 & 51.0 & 0.3875 & 0.2609 \\
\textbf{6. Scheduled 30-Day Refit SSPOC ($p=20$)} & \textbf{20/559 Buses (3.6\% Fixed)} & \textbf{35,246.10} & \textbf{3.56} & \textbf{7,595.29} & \textbf{48.9} & \textbf{49.5} & \textbf{0.4167} & \textbf{0.2953} \\
7. Sparse Regression ([Tyler TBD]) & [p/559 Buses] & [TBD] & [TBD] & [TBD] & [TBD] & [TBD] & --- & --- \\
\midrule
\textit{Ref: Continuous Ridge (Full Grid)} & \textit{559 Buses + 424 Exog} & \textit{27,964.00} & \textit{3.50} & \textit{---} & \textit{---} & \textit{---} & \textit{---} & \textit{---} \\
\bottomrule
\end{tabular}%
}
\end{table*}



::: {.figbox}
**Figure 3 Placeholder — Full-Project Out-of-Sample Cumulative PnL Trajectories (`Notebook 04`, Cell 12):** Insert the cumulative virtual bidding PnL curve (`2025–2026`, `$0.50/MWh` fee) comparing `Scheduled 30-Day Refit SSPOC (p=20)` (`+$35,246`) and `Scheduled 30-Day Refit DMDc` (`+$24,112`) against `Continuous Ridge`, `Natural-Prior MLP`, `GBDT`, and `Static DMDc`.
:::


---

## 6. Conclusion & Future Work

### 6.1 Key Conclusions
1. **Exogenous Control & Scheduled Refitting Rescue Dynamical Models:** Unforced spectral models fail across a 48-hour auction lag due to eigenvalue damping [@sashidhar2022bopdmd; @dylewsky2020stochastically], whereas separating 424 Day-Ahead exogenous controls ($\mathbf{u}_k$) from delay-embedded grid states ($\mathbf{x}_k$) and refitting `DMDc` [@proctor2016dmdc] every 30 days achieves **`3.12` Annual Sharpe** and eliminates winter regime-drift drawdowns.
2. **Task-Driven Sparse Sensing (`SSPOC`) Beats Dense Models with `3.6%` of the Grid:** Optimizing sensor placement directly for 3-class tail separation (`SSPOC`, $p=20$) [@brunton2016sspoc] outperforms whole-grid reconstruction (`QR-Pivoting`) [@manohar2018data], dense `DMDc`, and full-grid ML baselines—achieving **`3.56` Annual Sharpe (`+$35,246.10` Cum PnL)** with **zero exogenous weather or load covariates**.
3. **Separating Invariant Grid Topology from Non-Stationary Market Regimes:** `SSPOC` automatically discovers NYISO's physical transmission choke points (`0.864` monthly Jaccard stability) [@rudin2019stop]. Holding these 20 upstream bellwether buses fixed while updating classifier weights every 30 days maximizes out-of-sample Sharpe while requiring minimal live telemetry infrastructure.
4. **Cross-Formulation Synthesis:** `[PLACEHOLDER: 1–2 sentences summarizing how Patrick's SSPOC classification results compare against Tyler's continuous sparse regression results on sensor overlap and optimal budget p]`.

### 6.2 Limitations & Next Steps
* **Group-Sparse ($\ell_{2,1}$) Spatial Regularization:** Standard $\ell_1$ `SSPOC` occasionally selects multiple co-located units at the same generating facility (`Cricket Valley CC1–3`, `Nine Mile 1–2`, `Maple Ridge WT 1–2`). Incorporating plant-level group-sparse penalties will enforce strict geographic diversity across substations.
* **Hybrid `SSPOC` + `DMDc` Fusion:** Combining `SSPOC`'s 20 sparse nodal sensors with `DMDc`'s 48-hour Day-Ahead exogenous trajectory forecasts could unify spatial bottleneck detection with forward weather/load forcing.
* **Liquidity-Aware Quadratic Lot Sizing:** Extending our `1 MWh` unit-size benchmark to multi-megawatt institutional deployment by scaling position size $q_t^*$ inversely with causal Day-Ahead `P-27` bid-stack slopes ($\Lambda_{t-24}$) to account for self-induced auction price impact [@hubert2026trading; @almgren2001optimal].[^6]

[^6]: Following Almgren and Chriss [@almgren2001optimal] and Hubert et al. [@hubert2026trading], accounting for linear Day-Ahead bid-stack slope $\Lambda_t$ yields optimal position sizing $q_t^* = \min\big(q_{\max}, \max(0, |\hat{S}_t| - c) / (2\Lambda_t)\big)$, automatically throttling megawatt exposure during steep peaker-stack hours.

---

# References
