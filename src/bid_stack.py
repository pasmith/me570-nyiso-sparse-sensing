import gc
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm


def _extract_hourly_supply_slopes(
    gen_bids_dir: Path,
    clearing_prices: pd.Series,
    mw_band: float = 250.0,
) -> pd.Series:
    """
    Streams yearly NYISO P-27 generator bid Parquet files to estimate the local
    linear supply-stack slope Lambda_INC ($/MWh per MW) around the cleared DAM LBMP.
    """
    hourly_slopes: list[pd.Series] = []
    parquet_files = sorted(gen_bids_dir.glob("*.parquet"))

    for pf in tqdm(parquet_files, desc="Streaming P-27 Supply Slopes", unit="yr"):
        schema_names = pq.read_schema(pf).names
        ts_col = next((c for c in ["Date Time", "Time Stamp", "Timestamp"] if c in schema_names), None)
        mw_cols = [f"Dispatch MW{i}" for i in range(1, 13) if f"Dispatch MW{i}" in schema_names]
        pr_cols = [f"Dispatch $/MW{i}" for i in range(1, 13) if f"Dispatch $/MW{i}" in schema_names]
        if not ts_col or not mw_cols or not pr_cols:
            continue

        # Stream in record batches to keep peak RAM under 300 MB
        parquet_file = pq.ParquetFile(pf)
        yr_records: dict[pd.Timestamp, list[float]] = {}

        for batch in parquet_file.iter_batches(batch_size=250_000, columns=[ts_col] + mw_cols + pr_cols):
            df = batch.to_pandas()
            df[ts_col] = pd.to_datetime(df[ts_col], format="mixed", errors="coerce").dt.floor("h")
            df = df.dropna(subset=[ts_col])

            # Stack piecewise steps into long (timestamp, mw, price) arrays
            ts_vals = np.repeat(df[ts_col].to_numpy(), len(mw_cols))
            mw_vals = df[mw_cols].to_numpy(dtype=np.float32).ravel()
            pr_vals = df[pr_cols].to_numpy(dtype=np.float32).ravel()

            valid = np.isfinite(mw_vals) & np.isfinite(pr_vals) & (mw_vals > 0)
            ts_vals, mw_vals, pr_vals = ts_vals[valid], mw_vals[valid], pr_vals[valid]

            long_df = pd.DataFrame({"ts": ts_vals, "mw": mw_vals, "price": pr_vals})
            for ts, grp in long_df.groupby("ts", sort=False):
                p_clear = clearing_prices.get(ts, np.nan)
                if not np.isfinite(p_clear):
                    continue
                # Estimate local price sensitivity ($/MWh per MW) within +/- $10/MWh of clearing LBMP
                near = grp[(grp["price"] >= p_clear - 10.0) & (grp["price"] <= p_clear + 10.0)]
                local_mw = float(near["mw"].sum())
                if local_mw > 10.0:
                    # Linearized inverse supply density: $20 span / local MW depth
                    slope = 20.0 / max(local_mw, mw_band * 0.2)
                else:
                    slope = 20.0 / (mw_band * 0.2)
                yr_records.setdefault(ts, []).append(slope)

            del df, long_df
            gc.collect()

        if yr_records:
            s_yr = pd.Series({k: float(np.median(v)) for k, v in yr_records.items()}).sort_index()
            hourly_slopes.append(s_yr)

    if not hourly_slopes:
        return pd.Series(dtype=float)

    combined = pd.concat(hourly_slopes)
    return combined.groupby(level=0).median().sort_index()


def _extract_hourly_demand_slopes(
    load_bids_dir: Path,
    clearing_prices: pd.Series,
    mw_band: float = 250.0,
) -> pd.Series:
    """
    Streams yearly NYISO P-27 load bid Parquet files to estimate the local
    demand-stack slope Lambda_DEC ($/MWh per MW) around the cleared DAM LBMP.
    """
    hourly_slopes: list[pd.Series] = []
    if not load_bids_dir.exists():
        return pd.Series(dtype=float)

    parquet_files = sorted(load_bids_dir.glob("*.parquet"))
    for pf in tqdm(parquet_files, desc="Streaming P-27 Demand Slopes", unit="yr"):
        schema_names = pq.read_schema(pf).names
        ts_col = next((c for c in ["Date Time", "Time Stamp", "Timestamp"] if c in schema_names), None)
        mw_cols = [f"Price Cap {i} MW" for i in range(1, 4) if f"Price Cap {i} MW" in schema_names]
        pr_cols = [f"Price Cap {i} Dollar" for i in range(1, 4) if f"Price Cap {i} Dollar" in schema_names]
        if not ts_col or not mw_cols or not pr_cols:
            continue

        parquet_file = pq.ParquetFile(pf)
        yr_records: dict[pd.Timestamp, list[float]] = {}

        for batch in parquet_file.iter_batches(batch_size=250_000, columns=[ts_col] + mw_cols + pr_cols):
            df = batch.to_pandas()
            df[ts_col] = pd.to_datetime(df[ts_col], format="mixed", errors="coerce").dt.floor("h")
            df = df.dropna(subset=[ts_col])

            ts_vals = np.repeat(df[ts_col].to_numpy(), len(mw_cols))
            mw_vals = df[mw_cols].to_numpy(dtype=np.float32).ravel()
            pr_vals = df[pr_cols].to_numpy(dtype=np.float32).ravel()

            valid = np.isfinite(mw_vals) & np.isfinite(pr_vals) & (mw_vals > 0)
            ts_vals, mw_vals, pr_vals = ts_vals[valid], mw_vals[valid], pr_vals[valid]

            long_df = pd.DataFrame({"ts": ts_vals, "mw": mw_vals, "price": pr_vals})
            for ts, grp in long_df.groupby("ts", sort=False):
                p_clear = clearing_prices.get(ts, np.nan)
                if not np.isfinite(p_clear):
                    continue
                near = grp[(grp["price"] >= p_clear - 15.0) & (grp["price"] <= p_clear + 15.0)]
                local_mw = float(near["mw"].sum())
                slope = 30.0 / max(local_mw, mw_band * 0.15)
                yr_records.setdefault(ts, []).append(slope)

            del df, long_df
            gc.collect()

        if yr_records:
            s_yr = pd.Series({k: float(np.median(v)) for k, v in yr_records.items()}).sort_index()
            hourly_slopes.append(s_yr)

    if not hourly_slopes:
        return pd.Series(dtype=float)

    combined = pd.concat(hourly_slopes)
    return combined.groupby(level=0).median().sort_index()


def build_or_load_bid_stack_slopes(
    dam_bids_dir: str | Path,
    zonal_dam_df: pd.DataFrame,
    target_index: pd.DatetimeIndex,
    target_zone: str = "N.Y.C.",
    cache_dir: str | Path = "./nyiso_data/merged_parquet",
    zonal_share: float = 0.35,
) -> pd.DataFrame:
    """
    Computes (or loads cached) causal 24h-shifted hourly DAM bid-stack slopes
    Lambda_INC and Lambda_DEC ($/MWh per MW) scaled to zonal liquidity depth.
    """
    cache_path = Path(cache_dir) / f"bid_stack_slopes_{target_zone.replace('.', '')}.parquet"
    if cache_path.exists():
        print(f"[CACHE HIT] Loading {cache_path.name}...")
        return pd.read_parquet(cache_path).reindex(target_index).ffill().bfill()

    bids_root = Path(dam_bids_dir)
    z_sub = zonal_dam_df[zonal_dam_df["Name"] == target_zone].copy()
    z_sub["Time Stamp"] = pd.to_datetime(z_sub["Time Stamp"])
    clearing_prices = (
        z_sub.drop_duplicates("Time Stamp", keep="last")
        .set_index("Time Stamp")["LBMP ($/MWHr)"]
    )

    sup_slope = _extract_hourly_supply_slopes(bids_root / "gen_bids", clearing_prices)
    dem_slope = _extract_hourly_demand_slopes(bids_root / "load_bids", clearing_prices)

    raw_df = pd.DataFrame(index=target_index)
    # Scale NYISO-wide stack slope inversely by zonal load share (e.g., NYC ~ 35% of NYCA)
    raw_df["lambda_inc_raw"] = (sup_slope.reindex(target_index).ffill().bfill() / zonal_share).fillna(0.05)
    raw_df["lambda_dec_raw"] = (dem_slope.reindex(target_index).ffill().bfill() / zonal_share).fillna(0.08)

    # Strict 24h causal shift + 7-day rolling median so Day D sizing uses only Day D-1 settled stacks
    out_df = pd.DataFrame(index=target_index)
    out_df["lambda_inc"] = raw_df["lambda_inc_raw"].shift(24).rolling(168, min_periods=1).median().bfill()
    out_df["lambda_dec"] = raw_df["lambda_dec_raw"].shift(24).rolling(168, min_periods=1).median().bfill()

    out_df.to_parquet(cache_path)
    print(f"[SAVED] {cache_path.name} ({out_df.shape})")
    return out_df


def evaluate_sized_models(
    signals_df: pd.DataFrame,
    realized_spread: pd.Series,
    slopes_df: pd.DataFrame,
    expected_edge_per_mwh: float | pd.DataFrame = 8.0,
    fee_per_mwh: float = 0.50,
    q_max_mw: float = 50.0,
    hours_per_year: int = 8760,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Applies Hubert et al. (2026) closed-form optimal lot sizing:
        q_t* = min(q_max, max(0, |S_hat_t| - c) / (2 * Lambda_t))
    and evaluates post-slippage dollar PnL, average MW size, and Annual Sharpe.
    """
    spread = realized_spread.reindex(signals_df.index).to_numpy(dtype=float)
    lam_inc = slopes_df["lambda_inc"].reindex(signals_df.index).ffill().bfill().to_numpy(dtype=float)
    lam_dec = slopes_df["lambda_dec"].reindex(signals_df.index).ffill().bfill().to_numpy(dtype=float)

    records = []
    pnl_curves = {}

    for col in signals_df.columns:
        y_hat = signals_df[col].to_numpy(dtype=float)
        if isinstance(expected_edge_per_mwh, pd.DataFrame) and col in expected_edge_per_mwh.columns:
            edge = np.abs(expected_edge_per_mwh[col].reindex(signals_df.index).to_numpy(dtype=float))
        else:
            edge = np.full(len(y_hat), float(expected_edge_per_mwh))

        lam = np.where(y_hat > 0, lam_inc, lam_dec)
        net_edge = np.maximum(0.0, edge - fee_per_mwh)
        q_star = np.where(y_hat != 0, np.minimum(q_max_mw, net_edge / (2.0 * np.maximum(lam, 1e-4))), 0.0)

        # Realized dollar PnL net of fees and quadratic DAM price self-impact (Lambda * q^2)
        pnl = (y_hat * q_star * spread) - (q_star * fee_per_mwh) - (lam * (q_star ** 2))
        active = q_star > 0
        n_trades = int(active.sum())

        std_hr = float(np.std(pnl, ddof=1))
        sharpe = float((np.mean(pnl) / std_hr) * np.sqrt(hours_per_year)) if std_hr > 0 else 0.0
        eq = np.cumsum(pnl)
        max_dd = float(np.max(np.maximum.accumulate(eq) - eq)) if len(pnl) > 0 else 0.0

        records.append({
            "Model": col,
            "Trades": n_trades,
            "Avg Lot (MW)": round(float(q_star[active].mean()), 1) if n_trades > 0 else 0.0,
            "Win Rate (%)": round(float((pnl[active] > 0).mean() * 100), 1) if n_trades > 0 else 0.0,
            "Avg PnL/Trade ($)": round(float(pnl[active].mean()), 2) if n_trades > 0 else 0.0,
            "Cum PnL ($)": round(float(np.sum(pnl)), 2),
            "Annual Sharpe": round(sharpe, 2),
            "Max DD ($)": round(max_dd, 2),
        })
        pnl_curves[col] = pd.Series(pnl, index=signals_df.index)

    results_df = pd.DataFrame(records).sort_values(by="Annual Sharpe", ascending=False).reset_index(drop=True)
    return results_df, pd.DataFrame(pnl_curves)