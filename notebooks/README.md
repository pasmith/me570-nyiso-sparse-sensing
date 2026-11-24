# NYISO DART Spread Virtual Bidding — Notebooks

This folder contains the Jupyter notebooks for our NYISO Day-Ahead to Real-Time (DART) electricity spread virtual bidding pipeline (`2020–2024` train/validation, `2025–2026` out-of-sample test, `48h` causal auction lag, `$0.50/MWh` transaction fee).

## Folder Contents

* **Orchestrator Notebook (one-click execution):**
  * `00_run_all_pipeline.ipynb` — Runs notebooks `01` through `04` sequentially in isolated kernels and updates all outputs in-place (~3.8 minutes total runtime).
* **Research Notebooks (run in the numbered order below):**
  1. `01_baseline_evaluation.ipynb` — Econometric, ML, and unforced DMD baseline evaluation
  2. `02_dmdc_static_forensics.ipynb` — Forensic DMD diagnostics & static `DMDc` baseline
  3. `03_dmdc_iterative_scheduled_refit.ipynb` — Iterative rollout & 30-day scheduled refit `DMDc`
  4. `04_sspoc_sparse_sensor_placement.ipynb` — `SSPOC` sparse sensor ablation & cross-paradigm comparison
* **Supporting Python Modules:**
  * `data_loader.py` — Downloads, aligns, and caches NYISO P-2A/2B/4A/4B/7/19/23/27/32/54/58/59/63/70/511 datasets into `nyiso_data/merged_parquet/`
  * `baseline.py` — Econometric and ML reference models (`ARIMAX`, `Ridge`, `GBDT`, `Multinomial Logit`, `LDA`, `MLP`)
  * `dmd.py` — Unforced `Exact DMD` and `BOP-DMD` baselines
  * `evaluate_models.py` & `sparse_sensor.py` — `SSPOC`, `QR-Pivoting`, `Mutual-Info`, POD reconstruction, and geospatial sensor plotting

---

## Execution Order & Data Flow

You can either run **`00_run_all_pipeline.ipynb`** to execute the entire pipeline automatically, or run the four research notebooks manually in strict numerical order (**1 to 4**). Each notebook loads preprocessed data or cached `.parquet` artifacts saved by earlier notebooks in `nyiso_data/merged_parquet/`:

1. **`01_baseline_evaluation.ipynb`**
   * **Loads:** Raw NYISO ZIP archives via `data_loader.load_data()` (or loads cached `nodal_dart_spread.parquet`, `zonal_dart_spread.parquet`, `exogenous_features.parquet`, `stage1_baseline_results_NYC.parquet`, `stage1_baseline_unit_pnl_NYC.parquet`, and `stage1_signals_NYC.parquet`).
   * **What it does:** Evaluates 14 reference econometric, machine learning, and unforced DMD baselines for `N.Y.C.`, tunes Chow reject-option thresholds (`tau`) on `2024` validation data, computes block-bootstrap Sharpe confidence intervals, and plots initial `SSPOC` sensor maps.
   * **Saves to disk:**
     * `nyiso_data/merged_parquet/stage1_baseline_results_NYC.parquet` (and `stage1_baseline_results.parquet`)
     * `nyiso_data/merged_parquet/stage1_baseline_unit_pnl_NYC.parquet` (and `stage1_baseline_unit_pnl.parquet`)
     * `nyiso_data/merged_parquet/stage1_signals_NYC.parquet` (and `stage1_signals.parquet`)

2. **`02_dmdc_static_forensics.ipynb`**
   * **Loads:**
     * `zonal_dart_spread.parquet`, `zonal_dart_3class_labels.parquet`, and `exogenous_features.parquet` (built by `data_loader.py` in Notebook 1).
     * `stage1_baseline_results_NYC.parquet` (saved by **Notebook 1**).
   * **What it does:** Audits why unforced `Exact DMD` and `BOP-DMD` fail (calm-regime eigenvalue damping and winter phase inversion), fits a static 48-hour-jump `DMDc` model frozen on `2020–2024`, and diagnoses its `-$37,920.72` winter regime-drift drawdown in Jan–Feb 2026 to motivate scheduled retraining.
   * **Saves to disk:**
     * `nyiso_data/merged_parquet/dmdc_static_oos_results.parquet` (`16,306` out-of-sample rows of predictions, signals, and PnL).

3. **`03_dmdc_iterative_scheduled_refit.ipynb`**
   * **Loads:**
     * `zonal_dart_spread.parquet` and `exogenous_features.parquet`.
     * `dmdc_static_oos_results.parquet` (saved by **Notebook 2**).
   * **What it does:** Implements 1-hour iterative `DMDc` rollouts on rank-12 delay-embedded POD modes across the full 48-hour Day-Ahead exogenous trajectory, benchmarks frozen static `DMDc` against a **30-day (`720h`) scheduled expanding-window refit**, and motivates `SSPOC` for sparse nodal sensing and tail discrimination.
   * **Saves to disk:**
     * `nyiso_data/merged_parquet/dmdc_scheduled_oos_results.parquet` (`16,306` out-of-sample rows comparing static and 30-day refit `DMDc`).

4. **`04_sspoc_sparse_sensor_placement.ipynb`**
   * **Loads:**
     * Full dataset and metadata (`P-19`, `P-23`, `nodal_dart_spread.parquet`, `zonal_dart_spread.parquet`) via `data_loader.load_data()`.
     * `dmdc_scheduled_oos_results.parquet` (saved by **Notebook 3**).
   * **What it does:** Benchmarks `SSPOC` against `QR-Pivoting` and `Mutual-Info` across sensor budgets $p \in \{5, 10, 20, 40\}$ out of `559` buses, maps selected bellwether sensors to physical NYISO transmission corridors, verifies spatial generalization (`LONGIL`, `HUD VL`) and quarterly walk-forward Jaccard persistence, and evaluates **30-day scheduled refit `SSPOC` ($p=20$)** head-to-head against Notebook 3's `DMDc` results.
   * **Saves to disk:** Cached ablation artifacts in `nyiso_data/merged_parquet/` (managed automatically by `run_sparse_sensor_ablation`).

---

## Notes
* `baselines.ipynb` is an earlier exploratory draft superseded by `01_baseline_evaluation.ipynb` and does not need to be run.
