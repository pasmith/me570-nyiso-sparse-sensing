### Project Progress Report: NYISO DART Spread Forecasting & Virtual Bidding

**Team Member:** Patrick Smith  

#### Question Under Investigation
Can sparse sensor placement and reduced-order modal representations of nodal electricity prices outperform standard econometric and machine-learning baselines in predicting extreme Day-Ahead to Real-Time (DART) locational marginal price (LMP) spreads for profitable virtual bidding in the NYISO `N.Y.C.` zone? Specifically, I am investigating whether capturing spatial transmission congestion across generator nodes overcomes the regime-shift and tail-event failures observed in unforced spectral and standard tabular models under strict Day-Ahead auction timing constraints ($D-1$ 05:00 EST cutoff, requiring a 48-hour causal lag and a `$0.50/MWh` transaction fee hurdle).

#### Dataset
I constructed a leak-free, aligned hourly panel spanning **January 2020 to November 2026 (59,958 hourly timestamps)** from 15 NYISO public data feeds:
* **Nodal & Zonal DART Spreads:** Hourly Day-Ahead minus Real-Time LMP spreads across **559 generator nodes** and **15 NYISO zones** (`N.Y.C.` target plus 14 surrounding zones), paired with leak-free 3-class tail regime labels (`-1`, `0`, `+1`) defined by rolling 30-day 15th/85th percentiles.
* **Exogenous & Market Features:** **424 vintage-filtered features** strictly available prior to the Day-Ahead market close, including zonal load forecasts, weather forecasts, behind-the-meter solar forecasts, scheduled/actual outages, fuel mix, interface flows, limiting transmission constraints, and Day-Ahead bid stacks.
* **Validation Protocol:** Chronological split using **2020–2024** for training and hyperparameter tuning (`2024` inner validation year for reject-option threshold $\tau$ calibration) and **2025–2026** (16,306 hours) for strict out-of-sample evaluation.

#### Baseline Reproduction & Initial Findings
I have implemented, cached, and evaluated **14 benchmark models** on the 2025–2026 out-of-sample test set (`N.Y.C.`, `$0.50/MWh` fee), reproducing prior literature baselines alongside spectral and econometric benchmarks:

| Reproduced Baseline / Reference Model | Feature Set | Macro F1 | Tail F1 | Win Rate (%) | Cum PnL ($) | Annual Sharpe |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Continuous Ridge Regression** | Exog + 559 Nodes | 0.3799 | 0.2385 | 47.8% | $28,203.65 | **3.53** |
| **Causal 48h Persistence** | Target Lags | 0.4163 | 0.2547 | 50.6% | $22,388.24 | **2.31** |
| **Hubert et al. (2026) Multinomial Logit** | Exog + 559 Nodes | 0.3672 | 0.3041 | 53.7% | $19,974.56 | **1.82** |
| **Galarneau-Vincent et al. (2023) GBDT** | Exog-Only | 0.4510 | 0.3096 | 57.5% | $17,385.77 | **1.59** |
| **Hourly ARIMAX (24-hr)** | Exog + AR(48,72,168) | 0.3499 | 0.1238 | 39.6% | $9,355.99 | **1.25** |
| **Exact DMD (Mann & Kutz, 2016)** | 15 Zones + Delay Lags | 0.3238 | 0.0899 | 42.4% | -$6,023.68 | **-0.85** |
| **BOP-DMD (Sashidhar & Kutz, 2022)** | 15 Zones + Bagged VarPro | 0.3187 | 0.0826 | 42.3% | -$15,901.89 | **-2.28** |

**Key Diagnostic Insights:**
1. **Value of Nodal Spatial Information:** Incorporating the 559 nodal spreads boosts Continuous Ridge Regression to the top baseline (`3.53` Sharpe, `$28,203.65` PnL), whereas tree and neural models degrade when fed all 559 nodes without dimensionality reduction.
2. **Failure Mode of Unforced Spectral Models:** Forensic diagnostics on Exact DMD and BOP-DMD revealed that unforced 48-hour eigenvalue extrapolation suffers from calm-regime damping (`~90%` abstention) and severe phase inversion during winter/summer weather shocks (e.g., Jan–Feb 2026). Testing **DMD with Control (`DMDc`, Proctor et al., 2016)** confirmed that exogenous forcing restores positive PnL (`+$630.27`), motivating my next phase: **Sparse Sensor Placement Optimization for Classification (`SSPOC`)** to select an optimal subset of high-signal generator nodes.
