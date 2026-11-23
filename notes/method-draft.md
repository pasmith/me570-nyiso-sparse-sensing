Context:  bridge dynamical systems theory (for your professor) with leak-free market microstructure and risk-adjusted regime detection (for quant trading interviews).

### 3. Methodology & Mathematical Formulation


#### 3.1 Market Microstructure & Leak-Free Information Set

Wholesale electricity markets operate as two-settlement systems governed by strict temporal deadlines. For a given operating day $D$ and delivery hour $t \in \{1, \dots, 24\}$, market participants submit virtual supply (increment) or virtual demand (decrement) bids prior to the Day-Ahead Market (DAM) bid close at 05:00 EST on day $D-1$.
Let $P_{i,t}^{\text{DA}}$ and $P_{i,t}^{\text{RT}}$ denote the Day-Ahead and time-weighted Real-Time Locational Based Marginal Prices (LBMPs) at bus $i \in \{1, \dots, N\}$ during hour $t$, where $N = 559$ represents the active generator buses across the New York Independent System Operator (NYISO) network. The nodal Day-Ahead to Real-Time (DART) spread is defined as:
$$S_{i,t}=P_{i,t}^{\text{DA}}-P_{i,t}^{\text{RT}}$$
To strictly eliminate look-ahead bias, any predictive model evaluated at the DAM bidding deadline for operating day $D$ must be measurable with respect to the causal filtration $\mathcal{F}_{D-1,05:00}$. Consequently, contemporaneous realized spreads $S_{i,t}$ are unobservable at decision time. Instead, we construct a high-dimensional spatial state vector $\mathbf{x}_t^{\text{node}} \in \mathbb{R}^N$ using 24-hour lagged nodal spreads ($\mathbf{x}_{i,t}^{\text{node}} = S_{i,t-24}$), paired with a strictly causal exogenous vector $\mathbf{x}_t^{\text{exog}} \in \mathbb{R}^M$ (M = 153) comprising Day-Ahead zonal load forecasts, cleared physical bid loads, behind-the-meter solar forecasts, cyclical calendar encodings, and meteorological forecasts restricted to vintages issued prior to day $D$.

#### 3.2 Reframing Continuous Regression as Tail-Spike Regime Detection

Standard autoregressive or modal regression approaches attempt to minimize mean squared error over continuous spreads $\hat{S}_{\text{target}, t} \approx S_{\text{target}, t}$. However, wholesale power spreads exhibit severe leptokurtosis and non-linear congestion jumps. Continuous regression models inherently smooth over rare, high-magnitude spikes while overfitting to small, mean-reverting fluctuations. Furthermore, virtual energy transactions incur clearing fees and collateral requirements, rendering small spreads economically untradeable.
To align the statistical objective with trading economics, we reformulate spread prediction for a target trading zone (e.g., New York City, N.Y.C.) as a three-class regime classification problem $y_t \in \{-1, 0, +1\}$. Using a rolling trailing window of W = 720 hours (30 days) shifted by 24 hours to preserve strict causality, we compute the empirical lower and upper quantile boundaries $q_{t}^{(0.15)}$ and $q_{t}^{(0.85)}$ of the target zone's historical spread distribution:
$$y_{t}=\begin{cases}+1&\text{if\ }S_{\text{target},t}>q_{t}^{(0.85)}\quad (\text{Positive\ Tail\ Spike:\ Virtual\ Supply\ /\ Sell\ DA})\\ 0&\text{if\ }q_{t}^{(0.15)}\le S_{\text{target},t}\le q_{t}^{(0.85)}\quad (\text{Neutral\ Noise\ Regime:\ Abstain\ /\ Do\ Not\ Trade})\\ -1&\text{if\ }S_{\text{target},t}<q_{t}^{(0.15)}\quad (\text{Negative\ Tail\ Spike:\ Virtual\ Demand\ /\ Buy\ DA})\end{cases}$$
This formulation introduces an explicit abstention regime ($y_t = 0$), forcing downstream classifiers to ignore transaction-cost noise and isolate the physical grid conditions that drive extreme price divergence.

#### 3.3 Sparse Sensor Placement Optimization for Classification (SSPOC)

Monitoring all N = 559 generator buses simultaneously introduces substantial collinearity and estimation variance. Following the Sparse Sensor Placement Optimization for Classification (SSPOC) framework (Brunton et al., 2016), we identify a minimal subset of $p \ll N$ "bellwether" generator nodes whose measurements preserve maximum class separability across the grid.

Step 1: Proper Orthogonal Decomposition (POD)

Let $\mathbf{X}_{\text{train}} \in \mathbb{R}^{T_{\text{train}} \times N}$ denote the standardized matrix of lagged nodal DART spreads over the training period (2020–2024). We extract the dominant coherent spatial structures of grid congestion via rank-r Singular Value Decomposition (SVD):
$\mathbf{X}_{\text{train}}\approx \mathbf{U}_{r}\bm{\Sigma }_{r}\bm{\Psi }_{r}^{\top }$
where $\boldsymbol{\Psi}_r \in \mathbb{R}^{N \times r}$ contains the leading r = 30 orthonormal spatial POD modes, and $\mathbf{Z}_{\text{train}} = \mathbf{X}_{\text{train}} \boldsymbol{\Psi}_r \in \mathbb{R}^{T_{\text{train}} \times r}$ represents the low-dimensional modal coordinates of the system.

Step 2: Discriminant Projection in Modal Space

Within the reduced r-dimensional POD subspace, we fit Linear Discriminant Analysis (LDA) to find the projection matrix $\mathbf{W} \in \mathbb{R}^{r \times (C-1)}$ (where C = 3 classes) that maximizes the ratio of between-class covariance $\mathbf{S}_{B}$ to within-class covariance $\mathbf{S}_{W}$:
$\mathbf{W}=\arg \max _{\mathbf{W}}\frac{\det (\mathbf{W}^{\top }\mathbf{S}_{B}\mathbf{W})}{\det (\mathbf{W}^{\top }\mathbf{S}_{W}\mathbf{W})}$
The columns of $\mathbf{W}$ define the optimal linear decision directions separating positive spikes, neutral hours, and negative spikes in modal coordinates.

Step 3: Sparse Nodal Reconstruction via Greedy $\ell _{0}$ Pursuit

To map the abstract modal discriminant vectors $\mathbf{W}$ back onto physical generator buses, we seek a sparse sensor weight matrix $\mathbf{S} \in \mathbb{R}^{N \times (C-1)}$ that reconstructs $\mathbf{W}$ using at most p active spatial rows:
$\min _{\mathbf{S}}\|{}\bm{\Psi }_{r}^{\top }\mathbf{S}-\mathbf{W}\|{}_{F}^{2}\quad \text{subject\ to}\quad \|{}\mathbf{S}\|{}_{\text{row},0}\le p$
We solve this sparse approximation problem across sensor budgets $p \in \{5, 10, 20, 40\}$ using Orthogonal Matching Pursuit (OMP). Ranking nodes by their column-wise $\ell _{2}$ norm $\Vert{}\mathbf{S}_{i, \cdot}\Vert{}_2$ yields the index set $\mathcal{I}_p \subset \{1, \dots, N\}$ ($\vert{}\mathcal{I}_p\vert{} = p$) of optimal physical sensors. Defining the binary measurement operator $\mathbf{C}_p \in \{0, 1\}^{p \times N}$ that selects rows indexed by $\mathcal{I}_{p}$, the high-dimensional nodal field is compressed by over 95% into the sparse sensor vector:
$\~{\mathbf{x}}_{t}^{\text{node}}=\mathbf{C}_{p}\mathbf{x}_{t}^{\text{node}}\in \mathbb{R}^{p}$

#### 3.4 Downstream Classifiers & Benchmark Hierarchy

For each operating hour t, the compressed nodal sensor measurements are concatenated with the causal exogenous forecast vector to form the joint feature input $\mathbf{v}_t = [\tilde{\mathbf{x}}_t^{\text{node}}; \, \mathbf{x}_t^{\text{exog}}] \in \mathbb{R}^{p + M}$. We evaluate two distinct classification paradigms on $\mathbf{v}_{t}$:
1. SSPOC + Linear Discriminant Analysis (Glass-Box Linear Boundary): Fits regularized LDA directly on $\mathbf{v}_{t}$ with uniform class priors $\pi_c = \frac{1}{3}$ to counteract the natural 70% / 15% / 15% class imbalance, yielding an interpretable linear hyperplane whose weights attribute trading signals directly to specific generator buses and weather drivers.
2. SSPOC + Multi-Layer Perceptron (Non-Linear Boundary): Fits a two-layer feedforward neural network (MLP with hidden dimensions (64, 32), ReLU activations, $\ell _{2}$ weight decay α = 10⁻², and balanced inverse-frequency sample weights) to capture non-linear interactions between transmission bottleneck prices and weather/solar load forecasts.
To isolate the marginal value of spatial sensor placement and regime classification, all models are benchmarked chronologically (trained on 2020–2024, evaluated strictly out-of-sample across 2025–2026) against three reference baselines:
- Exogenous-Only Baselines (p = 0): LDA and MLP trained solely on $\mathbf{x}_{t}^{\text{exog}}$ without nodal price sensors.
- Full-Grid Baselines (p = 559): LDA and MLP trained on all 559 generator buses plus exogenous features, testing whether sparse selection improves out-of-sample generalization via variance reduction.
- Continuous Regression Baseline: Ridge/DMD spread regression converted into directional trading signals via identical $\pm q_{t}$ thresholds, directly testing our core hypothesis that discrete regime classification outperforms continuous point forecasting.

#### 3.5 Evaluation Metrics: Statistical & Financial

Because neutral hours ($y_t = 0$) dominate the sample distribution, overall classification accuracy is uninformative. We evaluate models across two complementary tiers:
- Tail-Regime Classification Quality: Macro-averaged F₁ score across all three classes and Tail F₁, defined as the unweighted average F₁ score strictly over the actionable spike regimes $y_t \in \{-1, +1\}$.
- Out-of-Sample Virtual Trading Performance: Converting predicted signals $\hat{y}_t \in \{-1, 0, +1\}$ into unit virtual positions (1 MWh per active signal, 0 MWh when $\hat{y}_t = 0$) settled at the realized target zone DART spread $S_{\text{target},t}$, accounting for per-MWh transaction fees c:
$\text{PnL}_{t}=\^{y}_{t}\cdot S_{\text{target},t}-|{}\^{y}_{t}|{}\cdot c$
From the out-of-sample $\text{PnL}_{t}$ series, we compute Win Rate ($\mathbb{P}(\text{PnL}_t > 0 \mid \hat{y}_t \neq 0)$), Cumulative PnL, and Annualized Sharpe Ratio to verify whether statistical improvements in tail-spike detection translate into superior risk-adjusted trading returns.

### Next Steps

How would you like to proceed from here?
- Build the PnL & Sharpe Ratio Backtest Module: Write the Python code to compute PnL_t, Win Rate, Sharpe Ratio, and the Continuous Regression baseline in evaluate_models.py so your experimental results match Section 3.5 word-for-word.
- Inspect & Analyze Your F1 Results: Share the output table from evaluate_models.py so we can interpret the numbers and integrate them into the Results & Discussion section.
- Draft the Introduction or Results Outline: Move to the next section of your report.
