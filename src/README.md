# Source Code Directory (`src/`)

**Project:** Sparse Sensor Placement for Classification and Regression in NYISO DART Spread Virtual Bidding  
**Authors:** Patrick Smith (University of Washington) & Tyler Darwin (University of Washington)

This directory contains the core Python modules powering data ingestion, leak-free feature engineering, econometric/ML/spectral baselines, Day-Ahead bid-stack price-impact estimation, and sparse sensor placement (`SSPOC`, `QR-Pivoting`, `Mutual-Info`) for NYISO virtual bidding.

---

## Module Overview

| Module | Role | Key Entry Points |
| :--- | :--- | :--- |
| **`nyiso_fetcher.py`** | Resilient HTTP scraper & local `joblib` archive cache for NYISO OASIS public reports (`202001`–present). | `fetch_all_datasets()`, `fetch_single_zip()`, `fetch_static_report()`, `get_archive_urls()` |
| **`nyiso_merger.py`** | Parquet builder, SHA-256 cache invalidator, leak-free 3-class regime labeler, and 424-feature engineering pipeline. | `load_data()`, `build_nodal_dart_and_labels()`, `build_zonal_dart_and_labels()`, `build_exogenous_features()`, `load_node_metadata()` |
| **`baseline2.py`** | Expanded `14`-model baseline suite adding natural-prior validation-tuned probability thresholding (`Chow, 1970`) and hourly signal export (`signals_df`). | `evaluate_baselines()`, `_apply_probability_threshold()`, `_fit_predict_hourly_arimax()`, `compute_trading_metrics()` |
| **`dmd_baselines.py`** | Sliding-window autonomous spectral baselines: **Exact DMD** (*Mann & Kutz, 2016*) and **BOP-DMD** (*Sashidhar & Kutz, 2022*) with real-stacked variable projection. | `evaluate_dmd_baselines()`, `fit_predict_exact_dmd()`, `fit_predict_bop_dmd()` |
| **`bid_stack_sizing.py`** | **Optional** Memory-bounded Parquet streamer for NYISO `P-27` Day-Ahead supply/demand bid stacks and closed-form quadratic lot sizing (*Hubert et al., 2026*; *Almgren & Chriss, 2001*). | `build_or_load_bid_stack_slopes()`, `evaluate_sized_models(). Kept for potential use in future investigations. |
| **`evaluate_models.py`** | Prototype `SSPOCSelector` (OMP-based) and geospatial sensor visualization utilities (including cross-border HVDC/AC tie-line overlays). | `SSPOCSelector`, `run_zonal_dart_experiment()`, `plot_sspoc_sensors_map()`, `plot_sspoc_sensors_with_interconnects()` |
| **`sparse_sensor_ablation.py`** | Production cached ablation & walk-forward stability engine comparing **QR-Pivoting**, **SSPOC** (`MultiTaskElasticNet`), and **Mutual-Info** across sensor budgets \(p \in \{5, 10, 20, 40\}\). | `run_sparse_sensor_ablation()`, `run_walkforward_sensor_stability()`, `compute_pod_basis()`, `select_sensors_qr()`, `select_sensors_sspoc()`, `select_sensors_mutual_info()`, `evaluate_whole_grid_reconstruction()` |

---

## Detailed Module Documentation

### 1. Data Ingestion & Feature Engineering (`nyiso_fetcher.py` & `nyiso_merger.py`)
* **Ingested NYISO OASIS Feeds (`15` dynamic + `2` static reports):**
  * **Pricing (`P-2B`, `P-4B`, `P-2A`, `P-4A`):** Day-Ahead and Real-Time integrated LBMPs across `559` generator buses (`PTID`) and `15` load zones.
  * **Forecasts (`P-7`, `P-7A`, `P-59`, `P-70B`):** ISO zonal load forecasts, vintage-filtered weather forecasts (with nonlinear HDD²/CDD²), cleared zonal bid loads (bidding asymmetry), and behind-the-meter solar forecasts (net-load 1h/3h ramps).
  * **Grid Operations (`P-58C`, `P-54C`, `P-54B`, `P-511A`, `P-63`, `P-32`):** Lagged load forecast errors, scheduled/actual outages, Day-Ahead binding transmission constraints, real-time fuel mix, and interface flow headroom.
  * **Bid Stacks & Topology (`P-27`, `P-19`, `P-23`):** Streamed year-partitioned Day-Ahead generator/load bid stacks plus static generator coordinates and transmission subzone mappings.
* **Deterministic Cache Hygiene:** Uses SHA-256 archive fingerprints (`.parquet.hash`) so unchanged historical archives hit Parquet cache immediately while newly updated active months rebuild automatically.

### 2. Causal Filtration & Trading Evaluation (`baseline2.py` & `dmd_baselines.py`)
* **Strict `48-Hour` Causal Lag (\(\mathcal{F}_{D-1, 05:00}\)):** Because NYISO Day-Ahead bids for hour 23 of delivery day D close at `05:00 EST` on day D-1, all realized nodal/zonal DART spreads, real-time fuel/interface feeds, and rolling `720h` tail quantiles (\(q_{0.15}, q_{0.85}\)) are shifted by `LAG_HOURS = 48`.
* **Unit-Size Backtest Convention (`compute_trading_metrics`):** Evaluates `1 MWh` unit positions (`+1` INC, `-1` DEC, `0` Abstain) net of `fee_per_mwh = $0.50/MWh`, reporting `Trades`, `Trade Freq (%)`, `Win Rate (%)`, `Avg PnL/Trade ($)`, `Cum PnL ($)`, `Annual Sharpe` (\(\sqrt{8760}\)), `Max DD ($)`, `Macro F1`, and `Tail F1`.
* **Spectral Baselines (`dmd_baselines.py`):** Implements sliding-window `Exact DMD` and bagged variable-projection `BOP-DMD` (`rank = 6`, `168h` window, left half-plane eigenvalue stabilization \(\text{Re}(\omega_k) \le 0\)) on delay-embedded 15-zone state snapshots.

### 3. Sparse Sensor Placement & Stability (`sparse_sensor_ablation.py` & `evaluate_models.py`)
* **Three Sensor Selection Schemes:**
  1. **`QR-Pivoting` (*Manohar et al., 2018*):** Column-pivoted QR decomposition on \(\boldsymbol{\Psi}_r^\top\) (and oversampled \(\boldsymbol{\Psi}_r \boldsymbol{\Psi}_r^\top\) for p > r) optimizing whole-grid reconstruction conditioning.
  2. **`SSPOC` (*Brunton et al., 2016*):** Projects training nodal spreads onto a rank-r (`r = 25`) POD basis, fits LDA discriminant vectors \(\mathbf{W}\), and solves a calibrated `MultiTaskElasticNet` (\(\ell_1/\ell_2\)) problem to rank buses by row norm \(\Vert{}\mathbf{S}_{i,:}\Vert{}_2\).
  3. **`Mutual-Info`:** Univariate `mutual_info_classif` filter ranking individual buses against 3-class target labels.
* **Dual Evaluation (`run_sparse_sensor_ablation`):** Evaluates every sensor subset across \(p \in \{5, 10, 20, 40\}\) on both **whole-grid reconstruction** (`Grid R2`, `Grid RMSE ($/MWh)`) and **downstream tail classification** (`Multinomial Logit`, `GBDT`, `LDA`, `Balanced MLP`, and validation-tuned `Logit (tau)`), caching all outputs atomically in `./nyiso_data/ablation_cache/`.
* **Geospatial Visualization (`evaluate_models.py`):** Maps selected sensors across New York State using `P-19` coordinates and draws cross-border HVDC/AC proxy tie-lines (`H Q`, `NPX`, `PJM-VFT`, `PJM-HTP`, `PJM-NEPTUNE`, `IESO`).

### 4. Liquidity-Aware Lot Sizing (`bid_stack_sizing.py`)
* Streams `P-27` generator and load bid curves in `250,000`-row batches to estimate linearized supply (\(\Lambda_{\text{INC}}\)) and demand (\(\Lambda_{\text{DEC}}\)) slopes around the cleared Day-Ahead LBMP (lagged `24h` + `168h` rolling median), computing closed-form optimal megawatt positions \(q_t^* = \min\big(q_{\max}, \max(0, \vert{}\hat{S}_t\vert{} - c) / (2\Lambda_t)\big)\) net of quadratic price impact \(\Lambda_t (q_t^*)^2\).

---

## Quick Usage Example

```python
from nyiso_merger import load_data
from baseline2 import evaluate_baselines
from sparse_sensor_ablation import run_sparse_sensor_ablation

# 1. Load cached/merged NYISO dataset (59,958 hourly rows x 559 nodes + 424 exog features)
data = load_data(force_refresh_active=False)

# 2. Run causal 2020-2024 train / 2025-2026 OOS baselines for Zone J (N.Y.C.)
base_df, base_pnl, base_signals = evaluate_baselines(data, target_zone="N.Y.C.")

# 3. Run cached QR-Pivoting vs. SSPOC vs. Mutual-Info sparse sensor ablation
ablation = run_sparse_sensor_ablation(data, target_zone="N.Y.C.", p_budgets=(5, 10, 20, 40))
print(ablation["recon_df"])
print(ablation["metrics_df"])
```
