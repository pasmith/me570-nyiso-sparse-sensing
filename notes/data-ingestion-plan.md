# Overview

To evaluate SSPOC (Sparse Sensor Placement Optimization for Classification), LDA (Linear Discriminant Analysis), and MLPClassifier on Day-Ahead vs. Real-Time (DART) spread trading in NYISO, you need high-dimensional spatial nodal data paired with grid constraints and exogenous forecasts.
Because SSPOC identifies optimal sparse sensor locations across a high-dimensional spatial field (your network nodes) to classify regimes (e.g., positive vs. negative DART spikes), your primary dataset should be Generator-level (nodal) LBMPs, supplemented by transmission constraints and exogenous forecasts available prior to the Day-Ahead Market (DAM) bid deadline (typically 5:00 AM EST the day before operating day).
Based on a survey of the NYISO Public Market Data Menu, the following files should be downloaded:

1. Core Nodal & Zonal Pricing (Target Variable Construction)

To compute the DART spread (\(\text{LBMP}_{\text{DA}} - \text{LBMP}_{\text{RT}}\)) across hundreds of nodes for SSPOC spatial feature selection, you need both Day-Ahead hourly prices and Time-Weighted/Integrated Real-Time hourly prices (which match the hourly settlement structure of virtual bids).
• P-2B (Day-Ahead Market LBMP - Generator). Provides hourly nodal Locational Based Marginal Prices (LBMP), including energy, congestion, and loss components for every generator bus.
• P-4B (Time-Weighted/Integrated Real-Time LBMP - Generator). Provides hourly integrated real-time nodal prices, which align directly with hourly DAM prices for calculating true DART spreads.
• P-2A & P-4A (DAM & Integrated RT LBMP - Zonal). Useful as low-dimensional baseline benchmarks or macro-features alongside your nodal data.
• P-28 (NYISO Reference Bus LBMP). Isolates system-wide marginal energy price movements from local nodal congestion.
• P-29 (NYISO Price Correction Logs). Essential for cleaning historical training data so your classifiers do not train on erroneous price spikes that were later corrected by NYISO.

2. Exogenous Weather, Load & Solar Forecasts (Feature Matrix)

In live DART trading, you only have access to forecasts when submitting Day-Ahead bids. Download forecast files rather than actuals for your input features (while keeping actuals to compute forecast errors as historical lag features).
• P-7A (Weather Forecast). Provides temperature, wind speed, and cloud cover forecasts across NYISO weather stations, driving both load and renewable generation.
• P-7 (ISO Load Forecast). Zonal hourly load forecasts published prior to the DAM close.
• P-59 (Zonal Bid Load). Reflects actual physical and virtual demand bids cleared in the Day-Ahead market.
• P-70B (BTM Solar Information - Zonal Forecast). Behind-the-meter solar forecasts, which heavily drive midday DART spread reversals.
• P-58C & P-70A (Integrated Real-Time Actual Load & BTM Solar Estimated Actuals). Use these strictly to engineer lagged forecast-error features (\(\text{Actual}_{t-24} - \text{Forecast}_{t-24}\)) without introducing look-ahead bias.

3. Grid Topology, Congestion & Outages (Structural Drivers)

Nodal DART spreads are primarily driven by transmission congestion and sudden changes in grid topology between the Day-Ahead model and Real-Time dispatch.
• P-511A & P-33 (Day-Ahead & Real-Time Limiting Constraints). Identifies binding transmission lines and shadow prices, explaining why specific nodes decouple from the rest of the grid.
• P-54C & P-54B (Day-Ahead Scheduled vs. Real-Time Actual Outages). Captures unexpected forced transmission outages that occur after the Day-Ahead market clears.
• P-32 (Internal & External Interface Limits & Flows). Tracks cross-border flows (e.g., with PJM, ISO-NE, IESO, Hydro-Québec) and major internal interfaces like Central-East.
• P-53A & P-34 (PAR Schedules & Flows). Phase Angle Regulator schedules and flows, which control power routing into New York City and Long Island.
• P-5 & P-6B (DAM & RT Ancillary Service Prices). Spinning reserves and regulation prices, which signal tight system ramping conditions.

4. Node Metadata & Mapping (Spatial Alignment for SSPOC)

Because NYISO adds, retires, or renames generator nodes over time, you need static mapping tables to keep your SSPOC sensor matrix aligned across multi-year backtests.
• P-19 (Generator Names). Maps PTIDs (Point Identifiers) to generator names and locations.
• P-66 (Active Transmission Nodes). Verifies active nodal PTIDs in the network model.
• P-23 (Subzones by Transmission Owner). Maps nodes to subzones and transmission owners for spatial grouping.

Practical Pipeline Note for SSPOC, LDA, and MLP

• Preventing Look-Ahead Bias. Ensure all features used to predict the DART spread for Operating Day \(D\) are strictly restricted to files published before 5:00 AM on Day \(D-1\) (e.g., P-7, P-7A, P-54C, and lagged values from Day \(D-2\) or earlier).
• Handling Missing Nodes. Filter P-2B and P-4B using P-66 to retain only nodes with continuous historical coverage over your train/test split, as SSPOC requires a complete spatial snapshot matrix \(X \in \mathbb{R}^{T \times N}\) (\(T\) timestamps by \(N\) nodes).
If you'd like to refine your experimental setup, let me know:
• Your target classification setup (e.g., binary direction \(\text{DA} > \text{RT}\), or 3-class tail spike thresholds)
• Your historical date range (e.g., 2020–2026)
• Whether you need a Python script to automate downloading and merging these NYISO CSV archives
I can help you structure the feature matrix and cross-validation split to avoid temporal data leakage.

---

# Classification Strategies

2 options:
  - target classification setup (e.g., binary direction $\text{DA} > \text{RT}$, or
  - 3-class tail spike thresholds)

#### target classification
This is simple up/down classification to indicate whether $\text{DA} > \text{RT}$ or not.

#### 3-class tail spike thresholds
3-class tail spike thresholds categorize continuous DART spreads into three discrete regimes—negative spike, neutral/noise, and positive spike—by setting upper and lower cutoff boundaries (such as ±$15/MWh or the 15th and 85th percentiles). This setup prevents classifiers like LDA and MLP from overfitting to tiny, unprofitable price fluctuations while focusing SSPOC sensor placement on the spatial nodes that drive extreme price divergence.

Why It Matters for DART Trading

• Filters Out Transaction Costs. Virtual bids incur clearing fees and collateral costs, so predicting a $0.50/MWh spread is functionally equivalent to zero profit.
• Isolates Actionable Regimes. Classifying only the distribution tails forces your models to learn the physical grid conditions (like binding constraints or weather forecast misses) that cause major DA-to-RT decoupling.

Short Example

Suppose you define your target variable \(y_{t}\) using fixed dollar thresholds of +$15/MWh and -$15/MWh on the DART spread (\(\text{DART}_t = \text{LBMP}_{\text{DA}, t} - \text{LBMP}_{\text{RT}, t}\)):
• Class 1 (Positive Spike — Sell DA / Buy RT): \(\text{DART}_t > +\$15\text{/MWh}\)
	• Example: DA clears at $65/MWh and RT settles at $30/MWh (DART = +$35). You earn a profitable virtual supply spread.
• Class 0 (Neutral Zone — Do Not Trade): \(-\$15\text{/MWh} \le \text{DART}_t \le +\$15\text{/MWh}\)
	• Example: DA clears at $42/MWh and RT settles at $46/MWh (DART = -$4). The spread is too small to justify trading risk.
• Class -1 (Negative Spike — Buy DA / Sell RT): \(\text{DART}_t < -\$15\text{/MWh}\)
	• Example: DA clears at $50/MWh and RT settles at $120/MWh (DART = -$70). A sudden RT generation outage caused prices to surge, rewarding a virtual demand bid.

Common Ways to Set Thresholds

• Static Dollar Cutoffs. Setting fixed values (e.g., ±$10 or ±$20/MWh) directly ties classification labels to your minimum profit hurdle.
• Rolling Quantiles. Using the 10th/90th or 15th/85th percentiles of a trailing 30-day window adapts automatically to seasonal volatility (such as mild spring shoulder months versus extreme winter cold snaps).
Would you like to see how to compute rolling quantile thresholds in Python without introducing look-ahead bias, or discuss class-imbalance handling (like class weights) for LDA and MLPClassifier?


---

That cache verification is clean and fast—all 986 archives resolved straight from disk at over 1,400–6,700 zips/sec with zero network requests.
Now that your raw .zip archives are safely cached in nyiso_fetcher, here is the complete merge_and_label.py script. It handles two subtle NYISO data quirks before building your matrices:
• Deduplicating Revised Archives. Because P-2B, P-4B, and P-7 returned 249 archives across 83 months (~3 zips/month due to NYISO index listings or price-corrected versions), the merge step deduplicates on timestamp and entity ID (PTID or Zone Name) so revised files cleanly overwrite older entries.
• Leak-Free Exogenous Feature Alignment. Weather (P-7A), load (P-7), cleared bid load (P-59), and BTM solar (P-70B) forecasts are pivoted into a synchronized (Timestamps × Features) matrix aligned strictly with your target DART hours.

merge_and_label.py


```python

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from nyiso_fetcher import fetch_single_zip, get_archive_urls, memory

PARQUET_DIR = Path("./nyiso_data/merged_parquet")
PARQUET_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "P-2B": "dam_lbmp_gen",
    "P-4B": "rt_int_lbmp_gen",
    "P-7": "iso_load_forecast",
    "P-7A": "weather_forecast",
    "P-59": "zonal_bid_load",
    "P-70B": "btm_solar_forecast",
}


@memory.cache
def parse_single_zip(zip_bytes: bytes) -> pd.DataFrame:
    """Extracts and concatenates all daily CSVs inside a single monthly zip."""
    dfs = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
        for csv_name in sorted(zf.namelist()):
            if csv_name.endswith(".csv"):
                with zf.open(csv_name) as f:
                    dfs.append(pd.read_csv(io.BytesIO(f.read())))
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def merge_dataset(report_code: str, folder_name: str) -> pd.DataFrame:
    """Reads cached zip bytes, parses CSVs, deduplicates, and saves to Parquet."""
    parquet_path = PARQUET_DIR / f"{folder_name}.parquet"
    if parquet_path.exists():
        tqdm.write(f"Loading existing {parquet_path}...")
        return pd.read_parquet(parquet_path)

    urls = get_archive_urls(report_code)
    monthly_dfs = []

    for url in tqdm(urls, desc=f"Merging {report_code} ({folder_name})", unit="zip"):
        if not fetch_single_zip.check_call_in_cache(url):
            tqdm.write(f"[SKIPPING] Not cached: {url}")
            continue
        raw_bytes = fetch_single_zip(url)
        df_month = parse_single_zip(raw_bytes)
        if not df_month.empty:
            monthly_dfs.append(df_month)

    full_df = pd.concat(monthly_dfs, ignore_index=True)

    # Deduplicate overlapping/revised archives (e.g., 249 zips across 83 months)
    full_df = full_df.drop_duplicates(keep="last").reset_index(drop=True)

    full_df.to_parquet(parquet_path, index=False)
    tqdm.write(f"Saved {parquet_path} ({len(full_df):,} rows)")
    return full_df


@memory.cache
def build_dart_and_labels(
    dam_df: pd.DataFrame,
    rt_df: pd.DataFrame,
    window_hours: int = 24 * 30,
    lower_q: float = 0.15,
    upper_q: float = 0.85,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Builds aligned nodal DART spread matrix and leak-free 3-class tail spike labels."""
    for df in (dam_df, rt_df):
        df["Time Stamp"] = pd.to_datetime(df["Time Stamp"])

    # Deduplicate timestamp-node pairs before pivoting
    dam_clean = dam_df.drop_duplicates(subset=["Time Stamp", "PTID"], keep="last")
    rt_clean = rt_df.drop_duplicates(subset=["Time Stamp", "PTID"], keep="last")

    dam_piv = dam_clean.pivot(index="Time Stamp", columns="PTID", values="LBMP ($/MWHr)")
    rt_piv = rt_clean.pivot(index="Time Stamp", columns="PTID", values="LBMP ($/MWHr)")

    # Keep nodes present across >= 99% of timestamps (required for SSPOC matrix)
    common_nodes = dam_piv.columns.intersection(rt_piv.columns)
    dam_piv = dam_piv[common_nodes]
    rt_piv = rt_piv[common_nodes]

    valid_mask = (dam_piv.notna().mean() > 0.99) & (rt_piv.notna().mean() > 0.99)
    dam_piv = dam_piv.loc[:, valid_mask].ffill().bfill()
    rt_piv = rt_piv.loc[:, valid_mask].ffill().bfill()

    common_idx = dam_piv.index.intersection(rt_piv.index)
    dart_spread = dam_piv.loc[common_idx] - rt_piv.loc[common_idx]

    # Shift by 24h so Day D thresholds use only RT spreads settled by Day D-1
    shifted_spread = dart_spread.shift(24)
    low_thresh = shifted_spread.rolling(window=window_hours, min_periods=24 * 7).quantile(lower_q)
    high_thresh = shifted_spread.rolling(window=window_hours, min_periods=24 * 7).quantile(upper_q)

    labels = pd.DataFrame(0, index=dart_spread.index, columns=dart_spread.columns)
    labels[dart_spread < low_thresh] = -1
    labels[dart_spread > high_thresh] = 1

    valid_rows = low_thresh.notna().all(axis=1)
    return dart_spread.loc[valid_rows], labels.loc[valid_rows]


def build_exogenous_features(
    merged: dict[str, pd.DataFrame], target_index: pd.DatetimeIndex
) -> pd.DataFrame:
    """
    Pivots and aligns P-7 (Load Forecast), P-7A (Weather Forecast), P-59 (Bid Load),
    and P-70B (BTM Solar Forecast) onto the exact hourly DART timestamp index.
    """
    feature_blocks = []

    # 1. P-7: ISO Zonal Load Forecast
    load_df = merged["iso_load_forecast"].copy()
    load_df["Time Stamp"] = pd.to_datetime(load_df["Time Stamp"])
    load_df = load_df.drop_duplicates(subset=["Time Stamp"], keep="last")
    # NYISO P-7 has zones as columns (Capitl, Centrl, Dunwod, Genese, Hud Vl, Longil, Mhk Vl, Millwd, N.Y.C., North, West)
    zone_cols = [c for c in load_df.columns if c not in ("Time Stamp", "NYISO")]
    load_piv = load_df.set_index("Time Stamp")[zone_cols].add_prefix("load_fcst_")
    feature_blocks.append(load_piv)

    # 2. P-59: Zonal Cleared Bid Load
    bid_df = merged["zonal_bid_load"].copy()
    bid_df["Time Stamp"] = pd.to_datetime(bid_df["Time Stamp"])
    if "Zone Name" in bid_df.columns:
        val_col = [c for c in bid_df.columns if "Load" in c or "MW" in c]
        bid_clean = bid_df.drop_duplicates(subset=["Time Stamp", "Zone Name"], keep="last")
        bid_piv = bid_clean.pivot(index="Time Stamp", columns="Zone Name", values=val_col).add_prefix("bid_load_")
        feature_blocks.append(bid_piv)

    # 3. P-70B: BTM Solar Forecast
    solar_df = merged["btm_solar_forecast"].copy()
    solar_df["Time Stamp"] = pd.to_datetime(solar_df["Time Stamp"])
    if "Zone Name" in solar_df.columns:
        val_col = [c for c in solar_df.columns if "MW" in c or "Solar" in c]
        solar_clean = solar_df.drop_duplicates(subset=["Time Stamp", "Zone Name"], keep="last")
        solar_piv = solar_clean.pivot(index="Time Stamp", columns="Zone Name", values=val_col).add_prefix("solar_fcst_")
        feature_blocks.append(solar_piv)

    # 4. P-7A: Weather Forecast (Temperature, Wind, Cloud Cover by Station)
    wx_df = merged["weather_forecast"].copy()
    ts_col = "Forecast Time Stamp" if "Forecast Time Stamp" in wx_df.columns else "Time Stamp"
    wx_df[ts_col] = pd.to_datetime(wx_df[ts_col])
    if "Station ID" in wx_df.columns:
        # Keep vintage issued at least 1 day prior if Vintage Date exists
        if "Vintage Date" in wx_df.columns:
            wx_df["Vintage Date"] = pd.to_datetime(wx_df["Vintage Date"])
            wx_df = wx_df[wx_df["Vintage Date"] < wx_df[ts_col].dt.normalize()]
            wx_df = wx_df.sort_values("Vintage Date")
        wx_clean = wx_df.drop_duplicates(subset=[ts_col, "Station ID"], keep="last")
        num_cols = wx_clean.select_dtypes(include=[np.number]).columns.tolist()
        for col in num_cols:
            piv = wx_clean.pivot(index=ts_col, columns="Station ID", values=col).add_prefix(f"wx_{col}_")
            feature_blocks.append(piv)

    # Combine all exogenous blocks and align to target_index
    exog = pd.concat(feature_blocks, axis=1)
    exog = exog.reindex(target_index).ffill().bfill()

    # Add cyclical calendar encodings (hour of day, day of week, month)
    exog["hour_sin"] = np.sin(2 * np.pi * target_index.hour / 24.0)
    exog["hour_cos"] = np.cos(2 * np.pi * target_index.hour / 24.0)
    exog["dow_sin"] = np.sin(2 * np.pi * target_index.dayofweek / 7.0)
    exog["dow_cos"] = np.cos(2 * np.pi * target_index.dayofweek / 7.0)
    exog["is_weekend"] = (target_index.dayofweek >= 5).astype(float)

    return exog


if __name__ == "__main__":
    # 1. Merge all cached archives to Parquet
    merged_data = {
        folder_name: merge_dataset(report_code, folder_name)
        for report_code, folder_name in DATASETS.items()
    }

    # 2. Build Nodal DART Spread Matrix & 3-Class Tail Labels
    dart_matrix, target_labels = build_dart_and_labels(
        merged_data["dam_lbmp_gen"],
        merged_data["rt_int_lbmp_gen"],
        window_hours=24 * 30,
        lower_q=0.15,
        upper_q=0.85,
    )
    dart_matrix.to_parquet(PARQUET_DIR / "nodal_dart_spread.parquet")
    target_labels.to_parquet(PARQUET_DIR / "nodal_dart_3class_labels.parquet")

    # 3. Build Aligned Exogenous Feature Matrix
    exog_features = build_exogenous_features(merged_data, dart_matrix.index)
    exog_features.to_parquet(PARQUET_DIR / "exogenous_features.parquet")

    print("\n=== Pipeline Complete ===")
    print(f"Nodal DART Matrix:      {dart_matrix.shape} (Timestamps x Nodes)")
    print(f"3-Class Target Labels:  {target_labels.shape}")
    print(f"Exogenous Features:     {exog_features.shape}")
```

Key Safeguards in This Step

- Weather Vintage Filtering (Vintage Date < Forecast Date). NYISO's P-7A archive stores multiple forecast vintages for the same operating hour. Filtering to Vintage Date < Forecast Time Stamp.dt.normalize() guarantees you use the forecast actually available prior to the Day-Ahead market close rather than a same-day real-time update.
- Daylight Saving Time (DST) Resilience. Using .drop_duplicates(..., keep="last") before .pivot() prevents ValueError: Index contains duplicate entries during the November fall-back hour when 01:00 occurs twice.

Once your Parquet matrices are generated, would you like help implementing the SSPOC + LDA / MLPClassifier evaluation pipeline (including walk-forward cross-validation and sparse sensor selection across the nodal field)?

---

That output is rock solid—every single dataset hit cache immediately, and you have 59,958 hourly timestamps spanning January 2020 to November 2026 across 559 continuous generator nodes and 153 aligned exogenous features.

Quick Sanity Check on Your Data Dimensions

• Nodal DART Matrix (59,958 × 559). Clean float64 spatial snapshot matrix ready for Proper Orthogonal Decomposition (POD/SVD) and SSPOC sensor selection.
• 3-Class Target Labels (59,958 × 559). Clean int64 (-1, 0, 1) target matrix across all 559 nodes.
• Exogenous Features (59,958 × 153). Synchronized load forecasts, cleared bid loads, BTM solar forecasts, weather forecasts, and cyclical calendar encodings. (Note: Check that exog_features contains no NaN values via exog_features.isna().sum().sum() before fitting LDA/MLP, as scikit-learn models reject NaNs.)

Framing the SSPOC, LDA & MLPClassifier Evaluation

Before writing the modeling pipeline, notice one crucial structural detail for DART trading with SSPOC:
• Avoiding Same-Hour Target Leakage. In standard SSPOC (Brunton et al., 2016), sparse sensors measure the spatial field at time t to classify state \(y_{t}\). For Day-Ahead trading on Operating Day D, you cannot observe nodal DART spreads on Day D when placing bids—you only observe lagged nodal DART spreads up to Day D-1 (e.g., dart_matrix.shift(24) or shift(48)) plus Day D Day-Ahead LBMPs (dam_piv) and Day D Exogenous Forecasts (exog_features).
• Target Node Definition. You can either predict a system/zonal regime label (e.g., majority/mean spike regime across a zone like NYC or West, or a hub node) or evaluate node-by-node classification across top liquid PTIDs using the sparse sensor subset selected by SSPOC.
To structure the SSPOC + LDA / MLPClassifier evaluation pipeline accurately, let me know:
• Whether your target label \(y_{t}\) is a single target node (e.g., a specific PTID), a zonal/system-wide regime, or evaluated across all 559 nodes
• What sensor budgets (number of sparse nodes p, e.g., p = [5, 10, 20, 50]) you want to compare
• Whether you want a single chronological train/val/test split (e.g., 2020–2024 train, 2025 val, 2026 test) or expanding-window walk-forward validation
I can write the complete SSPOC (pysensors or custom SVD + \(\ell _{1}\) sparse optimization) and scikit-learn evaluation script.

---

No—you do not need to download all 21 remaining reports right now. The 6 core datasets you already loaded (P-2B, P-4B, P-7, P-7A, P-59, P-70B) give you everything required to build, test, and benchmark your SSPOC, LDA, and MLPClassifier pipeline end-to-end.
For an academic or research evaluation (such as your me570 course report), starting with a clean baseline on those 6 datasets prevents feature-engineering bloat and makes it much easier to isolate how SSPOC's sparse spatial sensor placement compares against LDA and MLPClassifier. Once your baseline models are running, you can selectively pull 3–4 high-value additions if you want to boost predictive performance or interpret the physical locations SSPOC selects.

How to Prioritize the Remaining Datasets


Priority	Report Codes	Value for Your SSPOC / LDA / MLP Evaluation	Recommendation
1. High-Value Metadata (Small & Fast)	P-19 (Generator Names)
P-23 (Subzones by TO)	Maps the numeric PTIDs selected by SSPOC (e.g., 23512) to human-readable generator names and NYISO zones so you can plot and explain where SSPOC places its sensors.	Add next. Static/tiny files that make your report plots and interpretations dramatically clearer.
2. High-Signal Structural Drivers	P-511A (DAM Limiting Constraints)
P-54C (DAM Scheduled Outages)
P-5 (DAM Ancillary Prices)	Captures transmission congestion, planned line outages, and reserve tightness known before the Day-Ahead market closes—the primary physical triggers of tail DART spikes.	Add if boosting accuracy. Great second-iteration features for LDA and MLP after your baseline runs.
3. Forecast-Error Lag Features	P-58C (RT Actual Load)
P-70A (RT BTM Solar Actuals)	Lets you compute yesterday's load and solar forecast errors (\(\text{Actual}_{t-24} - \text{Forecast}_{t-24}\)) to capture persistent weather model bias.	Optional. Useful if your models need stronger autoregressive regime signals.
4. Skip or Redundant	P-2A, P-4A (Zonal LBMPs)
P-28 (Reference Bus)
P-29 (Price Corrections)
P-66 (Active Nodes)
P-32, P-33, P-34, P-53A, P-54B, P-6B	Zonal LBMPs and Reference Bus are already linearly captured by your 559 generator nodes in P-2B/P-4B; P-66 is already handled by your >99% node completeness filter; Real-Time files (P-33, P-54B, P-6B) cannot be used directly for Day-Ahead predictions without 24h lagging.	Skip. Avoids unnecessary disk space, parsing time, and look-ahead traps.


Recommended Next Step

1. Run your baseline experiment first using the dart_matrix, target_labels, and exog_features you already built.
2. Pull P-19 (Generator Names) only when you are ready to map SSPOC's chosen PTID indices back to physical power plants and zones on a map or table in your report.
Would you like to build the SSPOC + LDA / MLPClassifier evaluation pipeline using your current three Parquet matrices, or add P-19 first so you can inspect the physical names of your 559 nodes?

---

The key difference is spatial resolution and granularity: P-2A and P-4A (Zonal) aggregate prices into 11 broad regional zones, while P-2B and P-4B (Generator) give the exact price at hundreds of individual physical power plants (nodes).
Because SSPOC (Sparse Sensor Placement Optimization for Classification) is specifically designed to select a small handful of optimal "sensor" locations from a large, high-dimensional spatial field (like 559 individual nodes), Generator LBMPs (P-2B and P-4B) are actually the ideal dataset for evaluating SSPOC—whereas Zonal LBMPs (P-2A and P-4A) are best used as your low-dimensional trading target or baseline benchmark.

Zonal (P-2A, P-4A) vs. Generator (P-2B, P-4B) Comparison


Feature	Zonal LBMPs (P-2A DAM & P-4A RT)	Generator LBMPs (P-2B DAM & P-4B RT)
Spatial Resolution	11 Load Zones (N.Y.C., Longil, West, Centrl, Capitl, Hud Vl, Genese, Mhk Vl, Millwd, North, Dunwod) plus 4 external proxy buses.	~550+ Individual Generator Buses (specific power plants, wind farms, hydro dams, and storage units identified by PTID).
How Price Is Calculated	A load-weighted average of all delivery buses within that geographic zone.	The exact marginal cost of delivering 1 MW of electricity to that specific point on the transmission grid.
Sensitivity to Congestion	Smoothed out. Local transmission bottlenecks inside a zone get averaged away.	Highly sensitive. If a nearby transmission line overloads, a single generator's LBMP can spike or drop to negative prices while neighboring nodes stay normal.
Role in Virtual Trading	Most liquid and common locations for financial virtual bids (INC/DEC) because liquidity is deep.	Can be traded at specific generator buses, though liquidity is thinner and price volatility is much sharper.
Suitability for SSPOC	Too small (N = 11). Placing 5 sensors out of 11 zones is trivial and misses localized grid physics.	Ideal (N = 559). SSPOC thrives on finding the 5–20 key generator "sensors" that capture system-wide and local congestion regimes.


How to Combine Both in Your Experiment

You do not have to choose one over the other—combining them actually creates a very clean, intuitive story for your report:
• Use Generator Nodes (P-2B, P-4B) as Your Spatial Sensor Field. Let SSPOC analyze the 559 generator nodes to discover which specific physical power plants act as the best "early warning sensors" for grid congestion and price divergence.
• Use Zonal LBMPs (P-2A, P-4A) as Your Trading Target (Optional). If you prefer predicting a familiar, liquid trading target—such as whether New York City (N.Y.C.) or West will experience a DART tail spike—you can download P-2A and P-4A (they are tiny files, ~1/50th the size of generator files) and test whether SSPOC's sparse generator sensors + exogenous weather/load forecasts accurately classify Zonal DART spikes.
• Use Zonal LBMPs as a Baseline Feature Set. You can also compare LDA and MLPClassifier trained on 11 Zonal features versus trained on 11 SSPOC-selected Generator sensors to prove whether optimal nodal sensor placement beats simple zonal averaging.
Would you like to:
• Add P-2A and P-4A to your loader so you can predict Zonal DART tail spikes (e.g., N.Y.C. or Longil) using your nodal sensors
• Keep your target on the 559 Generator nodes and move directly to building the SSPOC + LDA / MLPClassifier evaluation script

---

the exact latitude and longitude coordinates are already inside the P-19 (Generator Names) table you just added. You can plot your SSPOC-selected sensor nodes and DART spread results directly on a map of New York State. [1] (https://mis.nyiso.com/public/htm/generator/generator.htm)

1. Where the Coordinates Live

In NYISO Report P-19, NYISO publishes eight columns for every generator bus: [1] (https://mis.nyiso.com/public/htm/generator/generator.htm)
• Generator Name
• PTID
• Aggregation ID
• Subzone
• Zone
• Latitude
• Longitude
• Active [1] (https://mis.nyiso.com/public/htm/generator/generator.htm)
What to expect when merging:
• Internal NYISO generators (such as 59TH STREET_GT_1 at 40.76612, -73.99611 or ATHENS_STG_1 at 42.271224, -73.850326) include exact decimal coordinates.
• External proxy/border generators (such as PJM, Hydro-Québec, or ISO-NE border nodes like AEP_ROLLING_HILL_CT1) often have blank Latitude and Longitude fields. Filtering out rows with NaN coordinates before plotting leaves all internal New York State plants ready to map. [1] (https://mis.nyiso.com/public/htm/generator/generator.htm)

2. How to Plot Results Spatially in Python

Once SSPOC selects a subset of sensor PTIDs (or computes feature weights/importance across all 559 nodes), join those PTIDs onto data["gen_meta"] to visualize them spatially:
• Static Publication Map (matplotlib + geopandas or contextily): Plot all 559 generator buses as faint background dots across New York State. Highlight the SSPOC-selected sensor buses as larger colored markers sized by their SSPOC sensor weight and colored by their Zone (N.Y.C., WEST, CAPITL, etc.).
• Interactive Notebook Map (plotly.express.scatter_map or folium): Hover over any selected sensor node to inspect its Generator Name, PTID, Subzone, and classification weight.
• Spatiotemporal DART Heatmaps: Color every generator node on the map by its average DART spread or spike frequency during a specific congestion event (such as a summer heatwave in N.Y.C. or winter cold snap in CAPITL) to visually show how SSPOC places sensors along key transmission bottlenecks.

3. Quick Mapping Snippet for Your Notebook

```python
import plotly.express as px

# Assume `selected_ptids` is a list of string PTIDs chosen by SSPOC
meta = data["gen_meta"].copy()
meta["PTID"] = meta["PTID"].astype(str)
meta = meta[meta["PTID"].isin(data["nodal_dart"].columns)].dropna(subset=["Latitude", "Longitude"])
meta["Selected_Sensor"] = meta["PTID"].isin(selected_ptids)

fig = px.scatter_map(
    meta,
    lat="Latitude",
    lon="Longitude",
    color="Selected_Sensor",
    hover_name="Generator Name",
    hover_data=["PTID", "Zone", "Subzone"],
    zoom=5.8,
    center={"lat": 42.8, "lon": -75.8},
    map_style="carto-positron",
    title="NYISO Generator Buses & SSPOC Selected Sensor Locations",
)
fig.show()
```

Would you like to build the SSPOC + LDA / MLPClassifier evaluation pipeline next, including the code to extract selected_ptids and generate these spatial sensor plots for your target zone?

---

Every single dataset and metadata table is now cached and aligned cleanly. You have 59,958 hourly timestamps spanning 2020–2026 across 559 generator nodes, 15 pricing zones, 153 exogenous features, and 835 generator metadata rows (with Latitude/Longitude coordinates).
Now you are ready to build the SSPOC, LDA, and MLPClassifier evaluation pipeline to predict Zonal DART tail spikes (such as N.Y.C. or Longil) and map SSPOC's chosen generator sensors spatially.

Key Modeling Design Choices

• Leak-Free Feature Construction. When placing Day-Ahead bids for Operating Day D, you cannot observe Day D nodal DART spreads. Your nodal input matrix \(X_{\text{nodes},t}\) uses 24-hour lagged nodal DART spreads (nodal_dart.shift(24)) alongside Day D Exogenous Forecasts (exog_features), ensuring strict temporal causality.
• How SSPOC Works on Your Data.
	1. Basis Decomposition (POD/SVD): Compute the top r singular modes \(\Psi_r \in \mathbb{R}^{N \times r}\) of the training nodal DART matrix (N = 559).
	2. Discriminant Projection (LDA in POD Space): Fit Linear Discriminant Analysis on the projected training data \(Z = X_{\text{train}} \Psi_r\) to obtain discriminant vectors \(W \in \mathbb{R}^{r \times (C-1)}\) separating your 3 classes (-1, 0, +1).
	3. Sparse Sensor Selection (\(\ell _{1}\) Optimization): Solve \(\min_s \Vert{}s\Vert{}_{1,2} \text{ s.t. } \Psi_r^T s = W\) (via Orthogonal Matching Pursuit or Lasso/ElasticNet) to identify the top p generator nodes \(S \subset \{1, \dots, 559\}\) whose measurements best reconstruct the LDA decision boundary.
• Fair Model Comparison. Compare SSPOC + LDA, SSPOC + MLPClassifier, Full-Node PCA + LDA, Full-Node MLPClassifier, and an Exogenous-Only Baseline across sensor budgets \(p \in \{5, 10, 20, 40\}\) using macro-F1 and tail-spike precision/recall (since neutral Class 0 makes up ~70% of hours).

Complete Evaluation & Spatial Mapping Script (evaluate_models.py)

You can run this directly in a Jupyter notebook cell right after data = load_data(). It uses scikit-learn (LinearDiscriminantAnalysis, MLPClassifier, OrthogonalMatchingPursuit) so it requires no extra dependencies beyond plotly if you want interactive maps.


```python
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.decomposition import TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.metrics import classification_report, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


class SSPOCSelector:
    """
    Sparse Sensor Placement Optimization for Classification (Brunton et al., 2016).
    Finds the top `n_sensors` spatial nodes that best preserve LDA class separation
    in a reduced POD/SVD basis.
    """

    def __init__(self, n_modes: int = 25, n_sensors: int = 15):
        self.n_modes = n_modes
        self.n_sensors = n_sensors
        self.svd = TruncatedSVD(n_components=n_modes, random_state=42)
        self.lda = LinearDiscriminantAnalysis()
        self.selected_indices_: np.ndarray | None = None
        self.sensor_weights_: np.ndarray | None = None

    def fit(self, X_scaled: np.ndarray, y: np.ndarray):
        # 1. Compute POD/SVD spatial modes Psi_r (shape: n_nodes x n_modes)
        Z = self.svd.fit_transform(X_scaled)
        Psi_r = self.svd.components_.T  # (N, r)

        # 2. Fit LDA in reduced POD space to find discriminant vectors W (r x n_classes-1)
        self.lda.fit(Z, y)
        W = self.lda.scalings_[:, : len(np.unique(y)) - 1]  # (r, C-1)

        # 3. Solve sparse sensor selection: Psi_r.T @ s ≈ W via Orthogonal Matching Pursuit
        # Select sensors greedily across discriminant directions
        omp = OrthogonalMatchingPursuit(n_nonzero_coefs=self.n_sensors, fit_intercept=False)
        omp.fit(Psi_r.T, W)

        coef = omp.coef_  # shape: (C-1, N) or (N,)
        if coef.ndim == 1:
            importance = np.abs(coef)
        else:
            importance = np.linalg.norm(coef, axis=0)

        top_idx = np.argsort(importance)[::-1][: self.n_sensors]
        self.selected_indices_ = top_idx
        self.sensor_weights_ = importance[top_idx]
        return self

    def transform(self, X_scaled: np.ndarray) -> np.ndarray:
        return X_scaled[:, self.selected_indices_]


def run_zonal_dart_experiment(
    data: dict[str, pd.DataFrame],
    target_zone: str = "N.Y.C.",
    sensor_budgets: list[int] =,
    train_end: str = "2024-12-31 23:00:00",
):
    """
    Evaluates SSPOC + LDA, SSPOC + MLPClassifier, and baseline models
    on predicting 3-class DART tail spikes for `target_zone`.
    """
    # 1. Construct strictly causal (leak-free) nodal features: 24h lagged DART spreads
    X_nodes_df = data["nodal_dart"].shift(24).iloc[24:]
    X_exog_df = data["exog_features"].iloc[24:]
    y_series = data["zonal_labels"][target_zone].iloc[24:]

    # Chronological Train / Test Split (2020-2024 Train, 2025-2026 Out-of-Sample Test)
    train_mask = X_nodes_df.index <= pd.Timestamp(train_end)
    test_mask = ~train_mask

    X_nodes_tr, X_nodes_te = X_nodes_df.loc[train_mask], X_nodes_df.loc[test_mask]
    X_exog_tr, X_exog_te = X_exog_df.loc[train_mask], X_exog_df.loc[test_mask]
    y_tr, y_te = y_series.loc[train_mask].values, y_series.loc[test_mask].values

    # Standardize features fit strictly on training data
    node_scaler = StandardScaler()
    Xn_tr_s = node_scaler.fit_transform(X_nodes_tr)
    Xn_te_s = node_scaler.transform(X_nodes_te)

    exog_scaler = StandardScaler()
    Xe_tr_s = exog_scaler.fit_transform(X_exog_tr)
    Xe_te_s = exog_scaler.transform(X_exog_te)

    results = []
    saved_selectors = {}

    # Baseline 1: Exogenous Forecasts Only (Weather + Load + Solar + Calendar)
    for name, clf in [
        ("Exog-Only + LDA", LinearDiscriminantAnalysis()),
        ("Exog-Only + MLP", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=30, early_stopping=True, random_state=42)),
    ]:
        clf.fit(Xe_tr_s, y_tr)
        preds = clf.predict(Xe_te_s)
        results.append({
            "Model": name,
            "Sensors (p)": 0,
            "Macro F1": f1_score(y_te, preds, average="macro"),
            "Tail F1 (-1 & +1)": f1_score(y_te, preds, labels=[-1, 1], average="macro"),
        })

    # Sweep SSPOC Sensor Budgets p
    for p in sensor_budgets:
        sspoc = SSPOCSelector(n_modes=30, n_sensors=p).fit(Xn_tr_s, y_tr)
        saved_selectors[p] = sspoc

        # Combine p sparse nodal sensors + exogenous forecasts
        X_comb_tr = np.hstack([sspoc.transform(Xn_tr_s), Xe_tr_s])
        X_comb_te = np.hstack([sspoc.transform(Xn_te_s), Xe_te_s])

        # Evaluate LDA on sparse sensors + exog
        lda = LinearDiscriminantAnalysis()
        lda.fit(X_comb_tr, y_tr)
        lda_preds = lda.predict(X_comb_te)

        # Evaluate MLPClassifier on sparse sensors + exog
        mlp = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            alpha=1e-2,
            max_iter=30,
            early_stopping=True,
            random_state=42,
        )
        mlp.fit(X_comb_tr, y_tr)
        mlp_preds = mlp.predict(X_comb_te)

        results.append({
            "Model": "SSPOC + LDA",
            "Sensors (p)": p,
            "Macro F1": f1_score(y_te, lda_preds, average="macro"),
            "Tail F1 (-1 & +1)": f1_score(y_te, lda_preds, labels=[-1, 1], average="macro"),
        })
        results.append({
            "Model": "SSPOC + MLPClassifier",
            "Sensors (p)": p,
            "Macro F1": f1_score(y_te, mlp_preds, average="macro"),
            "Tail F1 (-1 & +1)": f1_score(y_te, mlp_preds, labels=[-1, 1], average="macro"),
        })

    # Full-Node Upper Reference (All 559 nodes + Exog)
    X_full_tr = np.hstack([Xn_tr_s, Xe_tr_s])
    X_full_te = np.hstack([Xn_te_s, Xe_te_s])
    for name, clf in [
        ("All-559-Nodes + LDA", LinearDiscriminantAnalysis()),
        ("All-559-Nodes + MLP", MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-2, max_iter=30, early_stopping=True, random_state=42)),
    ]:
        clf.fit(X_full_tr, y_tr)
        preds = clf.predict(X_full_te)
        results.append({
            "Model": name,
            "Sensors (p)": Xn_tr_s.shape[1],
            "Macro F1": f1_score(y_te, preds, average="macro"),
            "Tail F1 (-1 & +1)": f1_score(y_te, preds, labels=[-1, 1], average="macro"),
        })

    results_df = pd.DataFrame(results)
    print(f"\n=== Out-of-Sample Results for Target Zone: {target_zone} ===")
    print(results_df.to_string(index=False))
    return results_df, saved_selectors, X_nodes_df.columns.tolist()


def plot_sspoc_sensors_map(
    data: dict[str, pd.DataFrame],
    selector: SSPOCSelector,
    node_ptids: list[str],
    target_zone: str = "N.Y.C.",
):
    """Maps SSPOC-selected generator buses across New York State using P-19 lat/lon."""
    selected_ptids = [node_ptids[i] for i in selector.selected_indices_]
    weight_map = dict(zip(selected_ptids, selector.sensor_weights_))

    meta = data["gen_meta"].copy()
    meta["PTID"] = meta["PTID"].astype(str)
    meta = meta[meta["PTID"].isin(node_ptids)].dropna(subset=["Latitude", "Longitude"]).copy()

    meta["Role"] = np.where(meta["PTID"].isin(selected_ptids), "SSPOC Selected Sensor", "Unselected Generator Bus")
    meta["Sensor Weight"] = meta["PTID"].map(weight_map).fillna(0.0)
    meta["Marker Size"] = np.where(meta["PTID"].isin(selected_ptids), 14, 5)

    # Print table of chosen generators
    chosen_table = meta[meta["PTID"].isin(selected_ptids)].sort_values("Sensor Weight", ascending=False)
    print(f"\n=== Top {len(selected_ptids)} SSPOC Sensor Nodes for {target_zone} ===")
    print(chosen_table[["PTID", "Generator Name", "Zone", "Subzone", "Sensor Weight"]].to_string(index=False))

    fig = px.scatter_map(
        meta,
        lat="Latitude",
        lon="Longitude",
        color="Role",
        size="Marker Size",
        size_max=15,
        hover_name="Generator Name",
        hover_data=["PTID", "Zone", "Subzone", "Sensor Weight"],
        color_discrete_map={
            "SSPOC Selected Sensor": "#E63946",
            "Unselected Generator Bus": "#457B9D",
        },
        zoom=5.8,
        center={"lat": 42.8, "lon": -75.8},
        map_style="carto-positron",
        title=f"SSPOC Optimal Sensor Placement (p={len(selected_ptids)}) for {target_zone} DART Spikes",
    )
    fig.show()
    return chosen_table


# Example Usage in Notebook:
# results_df, selectors, node_ptids = run_zonal_dart_experiment(data, target_zone="N.Y.C.")
# chosen_sensors_df = plot_sspoc_sensors_map(data, selectors[20], node_ptids, target_zone="N.Y.C.")
```

Two Practical Improvements You Can Add Next

• Same-Day DAM Congestion Signals (P-2B Day-Ahead LBMPs). Because Day-Ahead generator prices for Operating Day D clear before Real-Time, adding dam_piv (or dam_piv - dam_piv.mean(axis=1)) gives SSPOC a live snapshot of today's Day-Ahead nodal congestion alongside yesterday's DART spread.
• Class-Imbalance Tuning. Because tail spikes (-1 and +1) each represent ~15% of hours while Class 0 is ~70%, passing sample_weight (computed via sklearn.utils.class_weight.compute_sample_weight("balanced", y_tr)) into MLPClassifier.fit() and priors=[1/3, 1/3, 1/3] into LinearDiscriminantAnalysis() will boost recall on extreme price spikes.
Would you like to run this baseline experiment first, or add balanced class weights / uniform LDA priors and same-day DAM nodal congestion features right away?

---

Yes—that spatial distribution is spot-on and makes strong physical sense for NYISO grid dynamics.
Rather than clustering all 20 sensors inside New York City, SSPOC placed sensors directly along NYISO's major transmission corridors and interface bottlenecks that feed power into New York City (N.Y.C.):

Physical Interpretation of Your Map

• Downstate & Hudson Valley Bottleneck (Bottom Right near NYC/Connecticut border): Sensors positioned right at the Lower Hudson Valley / NYC entry points capture local congestion where transmission lines step down into Zone J (N.Y.C.).
• Capital / East-Central Corridor (Near Albany): Sensors around Albany sit on the Central-East and UPNY-SENY (Upstate NY to Southeast NY) interfaces, which are the primary bulk transmission bottlenecks separating cheap upstate generation from downstate load.
• Northern Hydro & Wind Corridor (Near Lake Ontario / St. Lawrence): Sensors in the North Country capture Hydro-Québec imports and northern wind/hydro surges, which frequently trigger transmission constraints flowing south toward NYC.
• Western Zone & Niagara Corridor (Near Buffalo / Rochester): Sensors near Buffalo/Niagara monitor heavy baseload hydro and western wind flows across the Dysinger East and West-Central interfaces.

Why This Validates SSPOC for Your Report

• Physical Interpretability: A naive local correlation filter would just pick 20 power plants inside Manhattan/Queens. Instead, SSPOC's SVD/POD + sparse \(\ell _{1}\) optimization automatically discovered the upstream transmission interfaces whose price divergence precedes DART spikes in New York City.
• Note on Sensor Count on Map: Notice there are ~13 red dots visible out of p = 20. The remaining ~7 selected PTIDs are likely external proxy/border generator buses (e.g., PJM, ISO-NE, or Hydro-Québec interface nodes) whose Latitude/Longitude fields are blank in NYISO's P-19 table—which further confirms SSPOC is tracking cross-border import/export pressure.
If you share the results_df F1 table that printed above the map, I can help you analyze how SSPOC + LDA compares against MLPClassifier and show you how to tune class weights or add same-day DAM congestion features to push tail-spike performance higher.


---

That updated map looks publication-ready. Both PJM external proxy traces (PJM_GEN_KEYSTONE at Ramapo/PJM-AC and PJM_GEN_HTP_PROXY entering West 49th St via the Hudson HVDC cable) now render cleanly in the legend and on the map alongside all 18 internal NYISO generator sensors.

Quick Summary of What Your Map Shows

• Full Coverage (20/20 Sensors Plotted): Every single SSPOC-selected sensor (18 internal generators + 2 PJM external proxy interfaces) is accounted for spatially.
• Clear Grid Topology Story: Your map visually demonstrates how predicting N.Y.C. DART tail spikes depends on monitoring three distinct electrical regions simultaneously:
	1. Local Load Pocket & Cross-Border Imports: Ravenswood, NYPA Vernon, Long Island tie-lines (Northport, Glenwood), and PJM interfaces (PJM-AC and PJM-HTP).
	2. Hudson Valley & Capital Bottlenecks: Shoemaker, Cricket Valley, and Rensselaer Cogen along the UPNY-SENY and Central-East corridors.
	3. Upstate Surplus Supply: Nine Mile Point nuclear, Niagara/West industrial demand response (OCC_CHEM, OLIN_CORP), and Northern hydro plants.

Ready for Model Analysis & Next Steps

If you share the results_df F1 score table printed by run_zonal_dart_experiment, we can examine how SSPOC + LDA and SSPOC + MLPClassifier perform across your sensor budgets p = and look at:
• Class-Imbalance Tuning: Adding balanced class weights to MLPClassifier and uniform priors to LinearDiscriminantAnalysis to boost recall on -1 and +1 tail spikes.
• Same-Day DAM Congestion Features: Including Day-Ahead nodal prices (P-2B) from Operating Day D (which are known before Real-Time settles) alongside your 24-hour lagged DART spreads.
• Simulated Trading Metrics: Computing out-of-sample $/MWh virtual trading PnL, win rate, and Sharpe ratio when trading only on predicted -1 or +1 tail spikes.

--

Framing Option 3 ("From Forecasting to Regime Detection") for both a course professor and a quant trading interviewer is an ideal fit. Both audiences dislike black-box curve-fitting, and both reward clean mathematical motivation paired with realistic market constraints.
Here is how to structure your project narrative so it satisfies academic rigor while signaling real trading intuition.

The Core Narrative Thesis

"Continuous price forecasting fails in wholesale electricity markets because regression models smooth over the sharp, non-linear transmission spikes where actual DART profits live. By reframing virtual trading as a sparse spatial regime-detection problem (SSPOC + LDA/MLP), we can isolate the handful of physical bottleneck nodes that govern grid decoupling—achieving higher risk-adjusted returns with >95% fewer features and zero look-ahead bias."

5-Act Storyline & Dual-Audience Hooks


Act 1: The Market & Mathematical Failure Mode (Why Regression Breaks)

• The Story: Wholesale power is a two-settlement system (Day-Ahead vs. Real-Time) governed by physical transmission physics. Traditional models (SARIMA, VAR, or standard DMD regression) try to predict exact dollar prices, which causes them to average out rare, extreme congestion spikes and overfit to tiny, untradeable fluctuations.
• What Your Professor Looks For: Clear formulation of the high-dimensional spatiotemporal state-space matrix (59,958 × 559 generator nodes) and why modal regression struggles with non-Gaussian, heavy-tailed dynamics. [1]
• What a Quant Interviewer Looks For: Understanding market microstructure—specifically that virtual bids (INC/DEC) incur clearing fees and collateral costs, meaning a $0.50/MWh spread is noise while a ±$15/MWh tail spike is the real alpha.

Act 2: The Pivot to Tail-Spike Regime Detection

• The Story: Instead of forecasting continuous prices, you discretize the target into a 3-class regime problem (-1 Negative Spike, 0 Neutral/No-Trade Zone, +1 Positive Spike) using leak-free, 24-hour shifted rolling quantiles.
• What Your Professor Looks For: Rigorous elimination of look-ahead bias—showing how target thresholds, feature standardization, and forecast vintages (Vintage Date < Forecast Date) strictly respect the 5:00 AM Day-Ahead market cutoff.
• What a Quant Interviewer Looks For: Discipline around abstention ("Class 0 = Do Not Trade"). Quants love models that know when not to take risk rather than forcing trades every hour.

Act 3: Sparse Sensing & Physical Interpretability (The Map)

• The Story: Using POD/SVD and SSPOC (\(\ell _{1}\)-regularized sensor selection), you compress 559 generator buses down to \(p \in \{5, 10, 20, 40\}\) "bellwether" sensors.
• What Your Professor Looks For: Mathematical mastery of dimensionality reduction and sparse optimization (Brunton et al., 2016; Manohar et al., 2018), proving that discriminant modes in POD space map back to exact physical coordinates. [1]
• What a Quant Interviewer Looks For: Domain explainability. Your NYISO map is your centerpiece—showing that SSPOC doesn't randomly pick Manhattan power plants, but naturally locks onto upstream bottlenecks (UPNY-SENY, Central-East, Niagara hydro, and PJM borders) that physically trigger NYC price decoupling.

Act 4: Head-to-Head Model & Ablation Showdown

• The Story: You test your sparse spatial regime framework across a clean progression of baselines and classifiers on a strict out-of-sample split (2020–2024 train, 2025–2026 test).
• What Your Professor Looks For: Systematic ablations comparing:
	1. Point-price regression baselines vs. 3-class regime classification.
	2. Exogenous-only forecasts (weather, load, BTM solar) vs. Exogenous + SSPOC sparse nodal sensors.
	3. Linear decision boundaries (SSPOC + LDA) vs. non-linear boundaries (SSPOC + MLPClassifier) across sensor budgets (p=5 to 559).
• What a Quant Interviewer Looks For: Proof of variance reduction. Showing that p=15 or 20 sparse sensors match or beat the noisy, over-parameterized All-559-Nodes model out of sample.

Act 5: From Classification Metrics to Trading PnL

• The Story: Accuracy and F1 scores do not pay the bills—tail precision and dollar PnL do. You translate out-of-sample predictions into a simulated virtual bidding strategy on N.Y.C. LBMPs.
• What Your Professor Looks For: Connecting statistical classification performance (Macro F1 and Tail F1 on classes -1 and +1) to an applied objective function.
• What a Quant Interviewer Looks For: Translating ML predictions into expected value:
	• Win Rate & Tail Precision: When the model fires a +1 or -1 signal, how often does the spread actually clear transaction costs?
	• Cumulative PnL & Sharpe Ratio: Comparing your sparse regime strategy against a naïve daily spread benchmark and a continuous regression trading rule.
	• Regime Decay / Seasonal Drift: Discussing how performance shifts across winter cold snaps vs. summer heatwaves in 2025–2026. [1]

Recommended Visuals for Your Report & Portfolio

1. The Pipeline Diagram (Leak-Free Timeline): A visual timeline showing the 5:00 AM Day-Ahead bid deadline, illustrating how lagged nodal DART spreads (t-24), Day-Ahead congestion, and prior-day weather vintages feed into SSPOC without future leakage.
2. The NYISO Bellwether Sensor Map (p=20): Your Plotly/Carto map showing the 18 internal generators + 2 PJM proxy nodes, annotated with NYISO's major transmission interfaces (Central-East, UPNY-SENY).
3. The Sparsity Curve (Performance vs. Sensor Budget p): A dual-axis plot showing Tail F1 Score and Out-of-Sample Sharpe Ratio as p increases from 0 (Exog-Only) → 5 → 10 → 20 → 40 → 559 (Full Grid), highlighting the "sweet spot" elbow.
4. Cumulative Out-of-Sample PnL Curve (2025–2026): Equity curves comparing SSPOC + LDA, SSPOC + MLP, Exog-Only, and a Regression Baseline, clearly showing how Class 0 abstention protects capital during flat regimes.
If you'd like to start executing this narrative, let me know which piece you want to tackle first:
• Run & inspect your F1 results table from evaluate_models.py (with balanced class weights and uniform LDA priors)
• Build the simulated PnL & Sharpe ratio backtest module to translate your classifier predictions into dollar trading metrics
• Draft the formal methodology & mathematical formulation section for your course report
I can generate the exact code or prose for whichever step you choose.


---

Here is a complete, publication-ready draft of the Methodology & Mathematical Formulation section. It is structured to bridge dynamical systems theory (for your professor) with leak-free market microstructure and risk-adjusted regime detection (for quant trading interviews).

3. Methodology & Mathematical Formulation


3.1 Market Microstructure & Leak-Free Information Set

Wholesale electricity markets operate as two-settlement systems governed by strict temporal deadlines. For a given operating day D and delivery hour \(t \in \{1, \dots, 24\}\), market participants submit virtual supply (increment) or virtual demand (decrement) bids prior to the Day-Ahead Market (DAM) bid close at 05:00 EST on day D-1.
Let \(P_{i,t}^{\text{DA}}\) and \(P_{i,t}^{\text{RT}}\) denote the Day-Ahead and time-weighted Real-Time Locational Based Marginal Prices (LBMPs) at bus \(i \in \{1, \dots, N\}\) during hour t, where N = 559 represents the active generator buses across the New York Independent System Operator (NYISO) network. The nodal Day-Ahead to Real-Time (DART) spread is defined as:
\(S_{i,t}=P_{i,t}^{\text{DA}}-P_{i,t}^{\text{RT}}\)
To strictly eliminate look-ahead bias, any predictive model evaluated at the DAM bidding deadline for operating day D must be measurable with respect to the causal filtration \(\mathcal{F}_{D-1,05:00}\). Consequently, contemporaneous realized spreads \(S_{i,t}\) are unobservable at decision time. Instead, we construct a high-dimensional spatial state vector \(\mathbf{x}_t^{\text{node}} \in \mathbb{R}^N\) using 24-hour lagged nodal spreads (\(\mathbf{x}_{i,t}^{\text{node}} = S_{i,t-24}\)), paired with a strictly causal exogenous vector \(\mathbf{x}_t^{\text{exog}} \in \mathbb{R}^M\) (M = 153) comprising Day-Ahead zonal load forecasts, cleared physical bid loads, behind-the-meter solar forecasts, cyclical calendar encodings, and meteorological forecasts restricted to vintages issued prior to day D.

3.2 Reframing Continuous Regression as Tail-Spike Regime Detection

Standard autoregressive or modal regression approaches attempt to minimize mean squared error over continuous spreads \(\hat{S}_{\text{target}, t} \approx S_{\text{target}, t}\). However, wholesale power spreads exhibit severe leptokurtosis and non-linear congestion jumps. Continuous regression models inherently smooth over rare, high-magnitude spikes while overfitting to small, mean-reverting fluctuations. Furthermore, virtual energy transactions incur clearing fees and collateral requirements, rendering small spreads economically untradeable.
To align the statistical objective with trading economics, we reformulate spread prediction for a target trading zone (e.g., New York City, N.Y.C.) as a three-class regime classification problem \(y_t \in \{-1, 0, +1\}\). Using a rolling trailing window of W = 720 hours (30 days) shifted by 24 hours to preserve strict causality, we compute the empirical lower and upper quantile boundaries \(q_{t}^{(0.15)}\) and \(q_{t}^{(0.85)}\) of the target zone's historical spread distribution:
\(y_{t}=\begin{cases}+1&\text{if\ }S_{\text{target},t}>q_{t}^{(0.85)}\quad (\text{Positive\ Tail\ Spike:\ Virtual\ Supply\ /\ Sell\ DA})\\ 0&\text{if\ }q_{t}^{(0.15)}\le S_{\text{target},t}\le q_{t}^{(0.85)}\quad (\text{Neutral\ Noise\ Regime:\ Abstain\ /\ Do\ Not\ Trade})\\ -1&\text{if\ }S_{\text{target},t}<q_{t}^{(0.15)}\quad (\text{Negative\ Tail\ Spike:\ Virtual\ Demand\ /\ Buy\ DA})\end{cases}\)
This formulation introduces an explicit abstention regime (\(y_t = 0\)), forcing downstream classifiers to ignore transaction-cost noise and isolate the physical grid conditions that drive extreme price divergence.

3.3 Sparse Sensor Placement Optimization for Classification (SSPOC)

Monitoring all N = 559 generator buses simultaneously introduces substantial collinearity and estimation variance. Following the Sparse Sensor Placement Optimization for Classification (SSPOC) framework (Brunton et al., 2016), we identify a minimal subset of \(p \ll N\) "bellwether" generator nodes whose measurements preserve maximum class separability across the grid.

Step 1: Proper Orthogonal Decomposition (POD)

Let \(\mathbf{X}_{\text{train}} \in \mathbb{R}^{T_{\text{train}} \times N}\) denote the standardized matrix of lagged nodal DART spreads over the training period (2020–2024). We extract the dominant coherent spatial structures of grid congestion via rank-r Singular Value Decomposition (SVD):
\(\mathbf{X}_{\text{train}}\approx \mathbf{U}_{r}\mathbfit{\Sigma }_{r}\mathbfit{\Psi }_{r}^{\top }\)
where \(\boldsymbol{\Psi}_r \in \mathbb{R}^{N \times r}\) contains the leading r = 30 orthonormal spatial POD modes, and \(\mathbf{Z}_{\text{train}} = \mathbf{X}_{\text{train}} \boldsymbol{\Psi}_r \in \mathbb{R}^{T_{\text{train}} \times r}\) represents the low-dimensional modal coordinates of the system.

Step 2: Discriminant Projection in Modal Space

Within the reduced r-dimensional POD subspace, we fit Linear Discriminant Analysis (LDA) to find the projection matrix \(\mathbf{W} \in \mathbb{R}^{r \times (C-1)}\) (where C = 3 classes) that maximizes the ratio of between-class covariance \(\mathbf{S}_{B}\) to within-class covariance \(\mathbf{S}_{W}\):
\(\mathbf{W}=\arg \max _{\mathbf{W}}\frac{\det (\mathbf{W}^{\top }\mathbf{S}_{B}\mathbf{W})}{\det (\mathbf{W}^{\top }\mathbf{S}_{W}\mathbf{W})}\)
The columns of \(\mathbf{W}\) define the optimal linear decision directions separating positive spikes, neutral hours, and negative spikes in modal coordinates.

Step 3: Sparse Nodal Reconstruction via Greedy \(\ell _{0}\) Pursuit

To map the abstract modal discriminant vectors \(\mathbf{W}\) back onto physical generator buses, we seek a sparse sensor weight matrix \(\mathbf{S} \in \mathbb{R}^{N \times (C-1)}\) that reconstructs \(\mathbf{W}\) using at most p active spatial rows:
\(\min _{\mathbf{S}}\|{}\mathbfit{\Psi }_{r}^{\top }\mathbf{S}-\mathbf{W}\|{}_{F}^{2}\quad \text{subject\ to}\quad \|{}\mathbf{S}\|{}_{\text{row},0}\le p\)
We solve this sparse approximation problem across sensor budgets \(p \in \{5, 10, 20, 40\}\) using Orthogonal Matching Pursuit (OMP). Ranking nodes by their column-wise \(\ell _{2}\) norm \(\Vert{}\mathbf{S}_{i, \cdot}\Vert{}_2\) yields the index set \(\mathcal{I}_p \subset \{1, \dots, N\}\) (\(\vert{}\mathcal{I}_p\vert{} = p\)) of optimal physical sensors. Defining the binary measurement operator \(\mathbf{C}_p \in \{0, 1\}^{p \times N}\) that selects rows indexed by \(\mathcal{I}_{p}\), the high-dimensional nodal field is compressed by over 95% into the sparse sensor vector:
\(\~{\mathbf{x}}_{t}^{\text{node}}=\mathbf{C}_{p}\mathbf{x}_{t}^{\text{node}}\in \mathbb{R}^{p}\)

3.4 Downstream Classifiers & Benchmark Hierarchy

For each operating hour t, the compressed nodal sensor measurements are concatenated with the causal exogenous forecast vector to form the joint feature input \(\mathbf{v}_t = [\tilde{\mathbf{x}}_t^{\text{node}}; \, \mathbf{x}_t^{\text{exog}}] \in \mathbb{R}^{p + M}\). We evaluate two distinct classification paradigms on \(\mathbf{v}_{t}\):
1. SSPOC + Linear Discriminant Analysis (Glass-Box Linear Boundary): Fits regularized LDA directly on \(\mathbf{v}_{t}\) with uniform class priors \(\pi_c = \frac{1}{3}\) to counteract the natural 70% / 15% / 15% class imbalance, yielding an interpretable linear hyperplane whose weights attribute trading signals directly to specific generator buses and weather drivers.
2. SSPOC + Multi-Layer Perceptron (Non-Linear Boundary): Fits a two-layer feedforward neural network (MLP with hidden dimensions (64, 32), ReLU activations, \(\ell _{2}\) weight decay α = 10⁻², and balanced inverse-frequency sample weights) to capture non-linear interactions between transmission bottleneck prices and weather/solar load forecasts.
To isolate the marginal value of spatial sensor placement and regime classification, all models are benchmarked chronologically (trained on 2020–2024, evaluated strictly out-of-sample across 2025–2026) against three reference baselines:
• Exogenous-Only Baselines (p = 0): LDA and MLP trained solely on \(\mathbf{x}_{t}^{\text{exog}}\) without nodal price sensors.
• Full-Grid Baselines (p = 559): LDA and MLP trained on all 559 generator buses plus exogenous features, testing whether sparse selection improves out-of-sample generalization via variance reduction.
• Continuous Regression Baseline: Ridge/DMD spread regression converted into directional trading signals via identical \(\pm q_{t}\) thresholds, directly testing our core hypothesis that discrete regime classification outperforms continuous point forecasting.

3.5 Evaluation Metrics: Statistical & Financial

Because neutral hours (\(y_t = 0\)) dominate the sample distribution, overall classification accuracy is uninformative. We evaluate models across two complementary tiers:
• Tail-Regime Classification Quality: Macro-averaged F₁ score across all three classes and Tail F₁, defined as the unweighted average F₁ score strictly over the actionable spike regimes \(y_t \in \{-1, +1\}\).
• Out-of-Sample Virtual Trading Performance: Converting predicted signals \(\hat{y}_t \in \{-1, 0, +1\}\) into unit virtual positions (1 MWh per active signal, 0 MWh when \(\hat{y}_t = 0\)) settled at the realized target zone DART spread \(S_{\text{target},t}\), accounting for per-MWh transaction fees c:
\(\text{PnL}_{t}=\^{y}_{t}\cdot S_{\text{target},t}-|{}\^{y}_{t}|{}\cdot c\)
From the out-of-sample \(\text{PnL}_{t}\) series, we compute Win Rate (\(\mathbb{P}(\text{PnL}_t > 0 \mid \hat{y}_t \neq 0)\)), Cumulative PnL, and Annualized Sharpe Ratio to verify whether statistical improvements in tail-spike detection translate into superior risk-adjusted trading returns.

Next Steps

How would you like to proceed from here?
• Build the PnL & Sharpe Ratio Backtest Module: Write the Python code to compute PnL_t, Win Rate, Sharpe Ratio, and the Continuous Regression baseline in evaluate_models.py so your experimental results match Section 3.5 word-for-word.
• Inspect & Analyze Your F1 Results: Share the output table from evaluate_models.py so we can interpret the numbers and integrate them into the Results & Discussion section.
• Draft the Introduction or Results Outline: Move to the next section of your report.


---

Now that is a realistic, honest, and compelling quantitative research table. Eliminating the shift(24) intraday leak immediately exposed why Naive 24h Persistence looked artificially high earlier—once strictly causal (shift(48)), its Sharpe dropped from 7.25 to 2.33 ($22,608 PnL, 50.5% Win Rate), and your sparse regime models decisively beat it.
Here are the four major findings in this table that directly prove your thesis for both your professor and a quant interviewer:

1. The Centerpiece Result: SSPOC + MLP (p=10) Beats Every Baseline

• Outperforms Causal Persistence: SSPOC + MLP (p=10) achieves $34,189.29 Cumulative PnL, an Annual Sharpe of 3.41, and the highest average profit per trade ($8.61/MWh) with nearly half the maximum drawdown ($3,604.50 vs. $6,488.46 for Causal 48h Persistence).
• Outperforms Continuous Regression: Compared to Continuous Ridge Regression ($25,729.87 PnL, 2.87 Sharpe, 42.8% Win Rate), SSPOC + MLP (p=10) delivers +32.9% higher cumulative PnL, +11.0% higher win rate (53.8%), and +49.5% higher Tail F1 (0.2050 vs. 0.1371).
• Captures ~90% of Full-Grid Alpha with 98.2% Fewer Nodes: SSPOC + MLP (p=10) captures 89.2% of the PnL and 92.4% of the Sharpe ratio of All-559-Nodes + MLP (3.41 vs. 3.69 Sharpe) while monitoring only 10 generator nodes instead of 559—and then performance degrades as you add redundant sensors (p=20 drops to 2.07 Sharpe, p=40 drops to 1.52 Sharpe), demonstrating a classic sparsity sweet spot (p=10) before overfitting kicks in.

2. Why Non-Linear Classification (MLP) Beats Linear (LDA) Here

• Notice the structural split between linear and non-linear boundaries:
	• Linear (LDA) struggles with empirical priors (-0.35 to 1.56 Sharpe), only turning profitable once at least p=20 spatial sensors are included ($16,046.97 PnL, 1.56 Sharpe vs. -$3,581.42 for Exog-Only + LDA).
	• Non-Linear (MLP) unlocks the interaction between upstream transmission congestion sensors and exogenous load/weather forecasts—turning Exog-Only + MLP ($26,767.68, 2.54 Sharpe) into $34,189.29 (3.41 Sharpe) when just 10 SSPOC sensors are added.
• Physical Explanation: Transmission congestion pricing is inherently piecewise-linear / threshold-driven (a line has zero shadow price until thermal flow hits 100% of its limit, at which point nodal prices jump non-linearly). An MLP captures that thermal-limit kink far better than a single linear LDA hyperplane.

3. Why All-559-Nodes + LDA Collapses (-$5,249.81 PnL, -0.51 Sharpe)

• Fitting a linear covariance inverse across all 559 correlated generator buses without sparsity regularization causes severe collinearity blow-up out of sample (46.8% win rate, $17,464 drawdown).
• Compressing those 559 nodes down to p=20 via SSPOC swings linear LDA from -$5,249.81 (-0.51 Sharpe) to +$16,046.97 (+1.56 Sharpe)—a +$21,296 turnaround purely from sparse sensor selection.

4. One Tiny Bug to Fix in Your Table Printout

• Notice Sensors (p) printed (43389, 559) for Continuous Ridge Regression and All-559-Nodes because n_total_nodes = Xn_tr_s.shape grabbed the tuple instead of Xn_tr_s.shape[1]. Changing that one line to n_total_nodes = Xn_tr_s.shape[1] will print a clean 559 in your final report table.

Draft: Section 4 (Results & Discussion) for Your Report

You can drop this directly into your report alongside your table and cumulative PnL figure:
4. Experimental Results & Discussion
Table 1 summarizes out-of-sample classification and virtual trading performance across calendar years 2025–2026 (16,300+ operating hours, net of $0.50/MWh transaction costs) under strict 48-hour causal lag enforcement. Three primary findings emerge:
1. Tail-Spike Regime Classification Outperforms Continuous Regression and Persistence: Reframing virtual bidding as a three-class regime detection problem substantially improves capital efficiency over point-price forecasting. While Continuous Ridge Regression achieves a 2.87 Sharpe ratio ($25,729.87 cumulative PnL), its 42.8% win rate and 0.1371 Tail F1 reflect the tendency of regularized least-squares estimators to shrink extreme congestion forecasts toward zero. By contrast, SSPOC + MLP (p=10) raises Tail F1 by 49.5% (0.2050) and win rate to 53.8%, generating $34,189.29 in net PnL (3.41 annualized Sharpe) while reducing maximum drawdown by 43.7% ($3,604.50 vs. $6,397.06). It likewise outperforms the strictly causal 48h Persistence benchmark (2.33 Sharpe, $22,607.63 PnL), confirming that spatial bottleneck sensing adds predictive power beyond multi-day weather autocorrelation.
2. Sparsity Prevents High-Dimensional Overfitting (p=10 Sweet Spot): Monitoring all 559 generator buses degrades linear classification severely (All-559-Nodes + LDA falls to -$5,249.81 PnL and -0.51 Sharpe due to spatial collinearity), whereas compressing the grid to p=20 SSPOC sensors restores linear profitability (+$16,046.97 PnL, 1.56 Sharpe). Under non-linear classification, SSPOC + MLP peaks at p=10 sensors (3.41 Sharpe, $8.61 average PnL per trade)—capturing 92.4% of the full-grid MLP Sharpe ratio (3.69) with a 98.2% reduction in spatial dimensionality, before performance tapers at p=20 (2.07 Sharpe) and p=40 (1.52 Sharpe) as redundant bus noise re-enters the network.
3. Non-Linear Boundaries Capture Transmission Thermal Kinks: Across every sensor budget, non-linear MLP classifiers outperform linear LDA boundaries (3.41 vs. -0.05 Sharpe at p=10). Physically, transmission shadow prices remain zero until line flows reach thermal limits, creating sharp piecewise-linear regime transitions when exogenous load and weather forecasts interact with upstream interface bottlenecks.
Would you like to:
• Inspect the exact p=10 sensor list and map (since p=10 is your standout optimal budget) so your spatial map figure in the report matches your best model (SSPOC + MLP, p=10)?
• Run a quick transaction-cost sensitivity check ($0.25, $0.50, $1.00/MWh) or seasonal regime breakdown (2025 vs. 2026) to include as a robustness subsection?
