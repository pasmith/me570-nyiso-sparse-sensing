# Project Progress Update: Sparse Sensor Placement for NYISO DART Spread Trading

**Team Members:** Patrick Smith and Tyler Darwin

**Question Under Investigation:**  
We investigate whether sparse sensor placement can accurately forecast Day-Ahead Real-Time (DART) locational marginal price (LMP) spreads for virtual bidding in wholesale electricity markets. Specifically, Patrick Smith evaluates whether Sparse Sensor Placement Optimization for Classification (SSPOC) can frame DART spike trading as a three-class regime problem and outperform literature benchmarks, while Tyler Darwin investigates sparse sensing for continuous DART spread regression. Together, we compare both formulations against benchmarks and each other testing whether the same physical generator buses serve as optimal sensors across classification and regression, and how the required sensor budget ($p$) differs between domains.

**Dataset:**  
We constructed an hourly dataset from public NYISO market reports spanning **January 2020 through November 2026 (59,958 hourly timestamps)**, strictly lagged by 48 hours to respect the 5:00 AM $D\text{-}1$ (day-ahead) auction cutoff without look-ahead leakage. The spatial state comprises **559 generator nodes** and **15 load zones**, paired with **424 engineered exogenous features** (load/solar forecasts, weather, outages, binding constraints, fuel mix, and interface flows) and `P-27` Day-Ahead bid-stack curves. Models are trained on **2020–2024** and evaluated out-of-sample on **2025–2026** (`N.Y.C.` target zone, net of a `$0.50/MWh` transaction fee).

**Baseline Reproduction:**  
We reproduced NYISO DART benchmarks (**Galarneau-Vincent et al., 2023**, Gradient Boosted Decision Trees [GBDT]; **Hubert, Lolas, & Sircar, 2026**, Multinomial Logit) alongside adapted spectral baselines (**Exact DMD**, Mann & Kutz, 2015; **BOP-DMD**, Sashidhar & Kutz, 2021; **DMD with Control [DMDc]**, Proctor, Brunton, & Kutz, 2016). Our out-of-sample GBDT and Multinomial Logit baselines achieved win rates of **57.7%–58.1%** and **53.5%–54.1%** with annualized Sharpe ratios of **1.24–2.17** and **1.59–1.78** (and Continuous Ridge Regression reaching **3.50**), matching the ~53%–62% win rates and ~1.4–3.2 unit-size Sharpe ranges reported across NYISO zones by Galarneau-Vincent et al. (2023) and Hubert et al. (2026). Unforced DMD variants yielded negative Sharpe ratios (**-0.85 to -2.28**) due to 48-hour eigenvalue damping and phase inversion, and while adding Day-Ahead exogenous forcing via DMDc recovered positive PnL (**0.06** Sharpe; **51.4%** win rate), it still suffered a **$37.9k** drawdown during the Jan–Feb 2026 winter regime shift.

**Implication:**  
This failure directly motivates **SSPOC**: zonal inputs hide localized congestion that bellwether generator nodes surface, while **3-class regime classification** decouples tail-spike probability from a linear continuous regressor's inability to recognize regime shifts.
