import hashlib
import io
import re
import joblib
import shutil
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import gc
import pyarrow as pa
import pyarrow.parquet as pq

from nyiso_fetcher import (
    CACHE_DIR,
    fetch_all_datasets,
    fetch_single_zip,
    fetch_static_report,
    get_archive_urls,
    memory,
)

PARQUET_DIR = Path("./nyiso_data/merged_parquet")
PARQUET_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = {
    # Core Pricing & Forecasts (Existing)
    "P-2B": "dam_lbmp_gen",
    "P-4B": "rt_int_lbmp_gen",
    "P-2A": "dam_lbmp_zone",
    "P-4A": "rt_int_lbmp_zone",
    "P-7": "iso_load_forecast",
    "P-7A": "weather_forecast",
    "P-59": "zonal_bid_load",
    "P-70B": "btm_solar_forecast",
    # New Feeds for Galarneau-Vincent et al. (2023) & Hubert et al. (2026)
    "P-58C": "rt_int_actual_load",
    "P-54C": "dam_sched_outages",
    "P-54B": "rt_actual_outages",
    "P-511A": "dam_limiting_constraints",
    "P-63": "rt_fuel_mix",
    "P-32": "interface_flows",
    "P-27": "dam_bid_data",  # 3-month lagged masked DAM bid stack
}


# =====================================================================
# CACHE FINGERPRINTING & ACTIVE-MONTH INVALIDATION HELPERS
# =====================================================================


def _get_cache_entry_dir(url: str) -> Path:
    """Returns the joblib cache directory for fetch_single_zip(url) using public joblib.hash."""
    args_hash = joblib.hash(((url,), {}))
    return Path(CACHE_DIR) / "joblib" / "nyiso_fetcher" / "fetch_single_zip" / args_hash


def refresh_active_month(datasets: dict[str, str], active_ym: str | None = None):
    if active_ym is None:
        active_ym = datetime.now().strftime("%Y%m")

    for report_code in datasets:
        try:
            urls = get_archive_urls(report_code)
        except Exception:
            continue

        for url in urls:
            if active_ym in url and fetch_single_zip.check_call_in_cache(url):
                entry_dir = _get_cache_entry_dir(url)
                if entry_dir.exists():
                    shutil.rmtree(entry_dir, ignore_errors=True)


def compute_urls_fingerprint(urls: list[str]) -> str:
    hasher = hashlib.sha256()
    for url in sorted(urls):
        hasher.update(url.encode("utf-8"))
        if fetch_single_zip.check_call_in_cache(url):
            pkl_path = _get_cache_entry_dir(url) / "output.pkl"
            if pkl_path.exists():
                stat = pkl_path.stat()
                hasher.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode("utf-8"))
            else:
                hasher.update(b"CACHED")
        else:
            hasher.update(b"MISSING")
    return hasher.hexdigest()


def combine_fingerprints(*fps: str) -> str:
    hasher = hashlib.sha256()
    for fp in fps:
        hasher.update(fp.encode("utf-8"))
    return hasher.hexdigest()


def is_parquet_fresh(parquet_path: Path, expected_fp: str) -> bool:
    hash_path = parquet_path.with_suffix(".parquet.hash")
    if not parquet_path.exists() or not hash_path.exists():
        return False
    return hash_path.read_text().strip() == expected_fp


def save_parquet_with_hash(
    df: pd.DataFrame, parquet_path: Path, fp: str, keep_index: bool = False
):
    df.columns = df.columns.astype(str)
    df.to_parquet(parquet_path, index=keep_index)
    parquet_path.with_suffix(".parquet.hash").write_text(fp)


# =====================================================================
# PARSING & DATASET MERGING
# =====================================================================


@memory.cache
def parse_single_zip(zip_bytes: bytes) -> pd.DataFrame:
    dfs = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
        for csv_name in sorted(zf.namelist()):
            if csv_name.endswith(".csv"):
                with zf.open(csv_name) as f:
                    dfs.append(pd.read_csv(io.BytesIO(f.read()), low_memory=False))
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def _merge_p27_partitioned(report_code: str, folder_name: str, current_fp: str) -> tuple[Path, str]:
    """Streams P-27 DAM bid zips into partitioned Parquet files grouped by (bid_type, year)."""
    out_dir = PARQUET_DIR / folder_name
    marker_path = PARQUET_DIR / f"{folder_name}.parquet.hash"

    if out_dir.exists() and marker_path.exists() and marker_path.read_text().strip() == current_fp:
        tqdm.write(f"[CACHE HIT] Partitioned dataset ready at {out_dir}/...")
        return out_dir, current_fp

    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    urls = get_archive_urls(report_code)
    writers: dict[tuple[str, str], pq.ParquetWriter] = {}
    schemas: dict[tuple[str, str], pa.Schema] = {}
    total_rows = 0

    try:
        for url in tqdm(urls, desc=f"Streaming {report_code} ({folder_name})", unit="zip"):
            if not fetch_single_zip.check_call_in_cache(url):
                continue

            ym_match = pd.Series([url]).str.extract(r"(\d{4})\d{2}")[0].iloc[0]
            year_key = ym_match if pd.notna(ym_match) else "unknown"

            raw_bytes = fetch_single_zip(url)
            with zipfile.ZipFile(io.BytesIO(raw_bytes), "r") as zf:
                for csv_name in sorted(zf.namelist()):
                    if not csv_name.endswith(".csv"):
                        continue
                    with zf.open(csv_name) as f:
                        df_chunk = pd.read_csv(
                            io.BytesIO(f.read()),
                            low_memory=False,
                            na_values=["", " ", "  ", "NA", "N/A", "null", "NULL"],
                        )

                    # Skip 1-column README / header cover files (e.g., ['January 2020 Monthly Bids'])
                    if df_chunk.empty or len(df_chunk.columns) <= 2:
                        continue

                    # Classify bid table by first column ('Masked Gen ID', 'Masked Sink ID', etc.)
                    first_col = str(df_chunk.columns).lower()
                    if "gen" in first_col:
                        bid_type = "gen_bids"
                    elif "sink" in first_col or "load" in first_col:
                        bid_type = "load_bids"
                    elif "source" in first_col or "virtual" in first_col or "trans" in first_col:
                        bid_type = "virtual_tx_bids"
                    else:
                        bid_type = re.sub(r"[^a-z0-9]+", "_", first_col).strip("_")[:40] or "other_bids"


                    # Downcast float64 -> float32 to cut disk & RAM usage in half
                    for col in df_chunk.select_dtypes(include=["float64"]).columns:
                        df_chunk[col] = df_chunk[col].astype("float32")

                    writer_key = (bid_type, year_key)

                    if writer_key not in writers:
                        # Cast integer columns to float32 on first chunk so future NaNs never fail int64 casting
                        for col in df_chunk.select_dtypes(include=["int64"]).columns:
                            df_chunk[col] = df_chunk[col].astype("float32")
                        table = pa.Table.from_pandas(df_chunk, preserve_index=False)
                        sub_dir = out_dir / bid_type
                        sub_dir.mkdir(parents=True, exist_ok=True)
                        schemas[writer_key] = table.schema
                        writers[writer_key] = pq.ParquetWriter(
                            sub_dir / f"{year_key}.parquet", table.schema, compression="snappy"
                        )
                    else:
                        target_schema = schemas[writer_key]
                        for field in target_schema:
                            if field.name not in df_chunk.columns:
                                df_chunk[field.name] = np.nan if pa.types.is_floating(field.type) else None
                            elif pa.types.is_floating(field.type):
                                df_chunk[field.name] = pd.to_numeric(df_chunk[field.name], errors="coerce").astype("float32")
                            elif pa.types.is_string(field.type) or pa.types.is_large_string(field.type):
                                df_chunk[field.name] = df_chunk[field.name].astype("string")

                        df_chunk = df_chunk[[f.name for f in target_schema]]
                        table = pa.Table.from_pandas(df_chunk, schema=target_schema, preserve_index=False)

                    writers[writer_key].write_table(table)
                    total_rows += len(df_chunk)
                    del df_chunk, table

            gc.collect()
    finally:
        for w in writers.values():
            w.close()

    marker_path.write_text(current_fp)
    tqdm.write(f"[SAVED] {out_dir}/ ({total_rows:,} rows streamed across {len(writers)} partitions)")
    return out_dir, current_fp


def merge_dataset(report_code: str, folder_name: str) -> tuple[pd.DataFrame | Path, str]:
    urls = get_archive_urls(report_code)
    current_fp = compute_urls_fingerprint(urls)

    # Stream massive P-27 bid stack directly to partitioned Parquet files
    if report_code == "P-27":
        return _merge_p27_partitioned(report_code, folder_name, current_fp)

    parquet_path = PARQUET_DIR / f"{folder_name}.parquet"
    if is_parquet_fresh(parquet_path, current_fp):
        tqdm.write(f"[CACHE HIT] Loading {parquet_path.name}...")
        if report_code == "P-54B":
            avail_cols = pq.read_schema(parquet_path).names
            ts_col = "Timestamp" if "Timestamp" in avail_cols else "Time Stamp"
            out_df = pd.read_parquet(parquet_path, columns=[ts_col])
            out_df["Time Stamp"] = pd.to_datetime(out_df[ts_col], errors="coerce").dt.floor("h")
            hourly_df = out_df.groupby("Time Stamp").size().reset_index(name="count")
            del out_df
            gc.collect()
            return hourly_df, current_fp
        return pd.read_parquet(parquet_path), current_fp

    tqdm.write(f"[REBUILDING] {parquet_path.name} (new or updated archives detected)...")
    monthly_dfs = []
    for url in tqdm(urls, desc=f"Merging {report_code} ({folder_name})", unit="zip"):
        if not fetch_single_zip.check_call_in_cache(url):
            tqdm.write(f"[SKIPPING] Not cached: {url}")
            continue
        raw_bytes = fetch_single_zip(url)
        df_month = parse_single_zip(raw_bytes)
        if not df_month.empty:
            monthly_dfs.append(df_month)

    if not monthly_dfs:
        return pd.DataFrame(), current_fp

    full_df = pd.concat(monthly_dfs, ignore_index=True)
    full_df = full_df.drop_duplicates(keep="last").reset_index(drop=True)

    save_parquet_with_hash(full_df, parquet_path, current_fp, keep_index=False)
    tqdm.write(f"[SAVED] {parquet_path.name} ({len(full_df):,} rows)")
    return full_df, current_fp




# =====================================================================
# NODAL & ZONAL DART TARGETS, EXOGENOUS FEATURES & METADATA
# =====================================================================


def _compute_dart_and_labels_from_pivots(
    dam_piv: pd.DataFrame,
    rt_piv: pd.DataFrame,
    window_hours: int = 24 * 30,
    lower_q: float = 0.15,
    upper_q: float = 0.85,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Shared helper to align DAM/RT pivots, compute DART spread, and assign 3-class labels."""
    common_cols = dam_piv.columns.intersection(rt_piv.columns)
    dam_piv = dam_piv[common_cols]
    rt_piv = rt_piv[common_cols]

    valid_mask = (dam_piv.notna().mean() > 0.99) & (rt_piv.notna().mean() > 0.99)
    # Causal forward-fill; restrict backward-fill strictly to initial 48h startup boundary
    dam_piv = dam_piv.loc[:, valid_mask].ffill().bfill(limit=48)
    rt_piv = rt_piv.loc[:, valid_mask].ffill().bfill(limit=48)

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


def build_nodal_dart_and_labels(
    dam_df: pd.DataFrame,
    rt_df: pd.DataFrame,
    window_hours: int = 24 * 30,
    lower_q: float = 0.15,
    upper_q: float = 0.85,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Builds aligned 559-node DART spread matrix and leak-free 3-class tail spike labels."""
    for df in (dam_df, rt_df):
        df["Time Stamp"] = pd.to_datetime(df["Time Stamp"])

    dam_clean = dam_df.drop_duplicates(subset=["Time Stamp", "PTID"], keep="last")
    rt_clean = rt_df.drop_duplicates(subset=["Time Stamp", "PTID"], keep="last")

    dam_piv = dam_clean.pivot(index="Time Stamp", columns="PTID", values="LBMP ($/MWHr)")
    rt_piv = rt_clean.pivot(index="Time Stamp", columns="PTID", values="LBMP ($/MWHr)")
    return _compute_dart_and_labels_from_pivots(dam_piv, rt_piv, window_hours, lower_q, upper_q)


def build_zonal_dart_and_labels(
    dam_zone_df: pd.DataFrame,
    rt_zone_df: pd.DataFrame,
    target_index: pd.DatetimeIndex,
    window_hours: int = 24 * 30,
    lower_q: float = 0.15,
    upper_q: float = 0.85,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Builds aligned Zonal DART spread matrix and 3-class tail labels (e.g., N.Y.C., Longil)."""
    for df in (dam_zone_df, rt_zone_df):
        df["Time Stamp"] = pd.to_datetime(df["Time Stamp"])

    dam_clean = dam_zone_df.drop_duplicates(subset=["Time Stamp", "Name"], keep="last")
    rt_clean = rt_zone_df.drop_duplicates(subset=["Time Stamp", "Name"], keep="last")

    dam_piv = dam_clean.pivot(index="Time Stamp", columns="Name", values="LBMP ($/MWHr)")
    rt_piv = rt_clean.pivot(index="Time Stamp", columns="Name", values="LBMP ($/MWHr)")

    zonal_dart, zonal_labels = _compute_dart_and_labels_from_pivots(
        dam_piv, rt_piv, window_hours, lower_q, upper_q
    )
    zonal_dart = zonal_dart.reindex(target_index).ffill().bfill(limit=48)
    zonal_labels = zonal_labels.reindex(target_index).fillna(0).astype(int)
    return zonal_dart, zonal_labels


def _normalize_zone_name(s: pd.Series) -> pd.Series:
    """Standardizes NYISO zone names across P-7, P-58C, P-59, P-70B, and P-2A."""
    return (
        s.astype(str)
        .str.upper()
        .str.strip()
        .str.replace(".", "", regex=False)
        .str.replace(" ", "_", regex=False)
    )


def build_exogenous_features(
    merged: dict[str, pd.DataFrame],
    target_index: pd.DatetimeIndex,
    gen_meta: pd.DataFrame,
) -> pd.DataFrame:
    feature_blocks = []

    # -----------------------------------------------------------------
    # 1. P-7: ISO Zonal Load Forecast
    # -----------------------------------------------------------------
    load_df = merged["iso_load_forecast"].copy()
    load_df["Time Stamp"] = pd.to_datetime(load_df["Time Stamp"])
    load_df = load_df.drop_duplicates(subset=["Time Stamp"], keep="last")
    zone_cols = [c for c in load_df.columns if c not in ("Time Stamp", "NYISO")]
    load_piv = load_df.set_index("Time Stamp")[zone_cols]
    load_piv.columns = [_normalize_zone_name(pd.Series([c]))[0] for c in load_piv.columns]
    feature_blocks.append(load_piv.add_prefix("load_fcst_"))

    # -----------------------------------------------------------------
    # 2. P-59: Zonal Cleared Bid Load + Block 1 Bidding Asymmetry
    # -----------------------------------------------------------------
    bid_df = merged["zonal_bid_load"].copy()
    bid_df["Time Stamp"] = pd.to_datetime(bid_df["Time Stamp"])
    bid_zone_col = next((c for c in ["Zone Name", "Name", "Zone"] if c in bid_df.columns), None)
    if bid_zone_col is not None:
        bid_df["Zone_Norm"] = _normalize_zone_name(bid_df[bid_zone_col])
        num_candidates = [
            c for c in bid_df.columns
            if c not in ("Time Stamp", bid_zone_col, "Zone_Norm", "PTID", "Time Zone")
            and pd.api.types.is_numeric_dtype(bid_df[c])
        ]
        val_col = num_candidates[0]
        bid_clean = bid_df.drop_duplicates(subset=["Time Stamp", "Zone_Norm"], keep="last")
        bid_piv = bid_clean.pivot(index="Time Stamp", columns="Zone_Norm", values=val_col).astype(float)
        feature_blocks.append(bid_piv.add_prefix("bid_load_"))

        # Block 1: Day-Ahead Bidding Asymmetry (P-59 vs P-7)
        common_zones = load_piv.columns.intersection(bid_piv.columns)
        if len(common_zones) > 0:
            aligned_fcst = load_piv[common_zones].reindex(bid_piv.index).astype(float)
            unsched_mw = aligned_fcst - bid_piv[common_zones]
            bid_ratio = bid_piv[common_zones] / aligned_fcst.replace(0, np.nan)
            feature_blocks.append(unsched_mw.add_prefix("unsched_load_mw_"))
            feature_blocks.append(bid_ratio.add_prefix("bid_to_fcst_ratio_"))

    # -----------------------------------------------------------------
    # 3. P-70B: BTM Solar Forecast + Block 4 Net-Load Ramps
    # -----------------------------------------------------------------
    solar_df = merged["btm_solar_forecast"].copy()
    solar_df["Time Stamp"] = pd.to_datetime(solar_df["Time Stamp"])
    solar_zone_col = next((c for c in ["Zone Name", "Name", "Zone"] if c in solar_df.columns), None)
    if solar_zone_col is not None:
        solar_df["Zone_Norm"] = _normalize_zone_name(solar_df[solar_zone_col])
        num_candidates = [
            c for c in solar_df.columns
            if c not in ("Time Stamp", solar_zone_col, "Zone_Norm", "PTID", "Time Zone")
            and pd.api.types.is_numeric_dtype(solar_df[c])
        ]
        val_col = num_candidates[0]
        solar_clean = solar_df.drop_duplicates(subset=["Time Stamp", "Zone_Norm"], keep="last")
        solar_piv = solar_clean.pivot(index="Time Stamp", columns="Zone_Norm", values=val_col).astype(float)
        feature_blocks.append(solar_piv.add_prefix("solar_fcst_"))

        # Block 4a: Net-Load Forecast & 1h / 3h Ramps
        common_solar_zones = load_piv.columns.intersection(solar_piv.columns)
        if len(common_solar_zones) > 0:
            net_load = (
                load_piv[common_solar_zones].reindex(target_index).astype(float)
                - solar_piv[common_solar_zones].reindex(target_index).fillna(0.0)
            )
            feature_blocks.append(net_load.add_prefix("net_load_fcst_"))
            feature_blocks.append(net_load.diff(1).add_prefix("net_load_ramp1h_"))
            feature_blocks.append(net_load.diff(3).add_prefix("net_load_ramp3h_"))



    # -----------------------------------------------------------------
    # 4. P-7A: Weather Forecast + Block 4b Nonlinear HDD^2 / CDD^2
    # -----------------------------------------------------------------
    wx_df = merged["weather_forecast"].copy()
    ts_candidates = ["Forecast Date", "Forecast Time Stamp", "Time Stamp"]
    ts_col = next((c for c in ts_candidates if c in wx_df.columns), wx_df.columns[0])
    wx_df[ts_col] = pd.to_datetime(wx_df[ts_col])
    station_col = next((c for c in ["Station ID", "Station Name"] if c in wx_df.columns), None)
    
    if station_col is not None:
        if "Vintage Date" in wx_df.columns:
            wx_df["Vintage Date"] = pd.to_datetime(wx_df["Vintage Date"])
            wx_df = wx_df[wx_df["Vintage Date"] < wx_df[ts_col].dt.normalize()]
            wx_df = wx_df.sort_values("Vintage Date")

        wx_clean = wx_df.drop_duplicates(subset=[ts_col, station_col], keep="last")
        num_cols = wx_clean.select_dtypes(include=[np.number]).columns.tolist()
        for col in num_cols:
            piv = wx_clean.pivot(index=ts_col, columns=station_col, values=col)
            feature_blocks.append(piv.add_prefix(f"wx_{col}_"))
            if "temp" in col.lower() or "dry" in col.lower():
                hdd = np.maximum(0.0, 65.0 - piv)
                cdd = np.maximum(0.0, piv - 65.0)
                feature_blocks.append((hdd ** 2).add_prefix(f"wx_hdd2_{col}_"))
                feature_blocks.append((cdd ** 2).add_prefix(f"wx_cdd2_{col}_"))

    # -----------------------------------------------------------------
    # 5. P-58C: Integrated RT Actual Load -> Block 2 Lagged Forecast Errors
    # -----------------------------------------------------------------
    if "rt_int_actual_load" in merged and not merged["rt_int_actual_load"].empty:
        act_df = merged["rt_int_actual_load"].copy()
        act_df["Time Stamp"] = pd.to_datetime(act_df["Time Stamp"])
        zone_col = next((c for c in ["Name", "Zone Name", "Zone"] if c in act_df.columns), None)
        load_col = next(
            (
                c for c in act_df.columns
                if c not in ("Time Stamp", zone_col, "PTID", "Time Zone")
                and pd.api.types.is_numeric_dtype(act_df[c])
            ),
            None,
        )

        if zone_col and load_col:
            act_df["Zone_Norm"] = _normalize_zone_name(act_df[zone_col])
            act_clean = act_df.drop_duplicates(subset=["Time Stamp", "Zone_Norm"], keep="last")
            act_piv = act_clean.pivot(index="Time Stamp", columns="Zone_Norm", values=load_col)
            common_act_zones = load_piv.columns.intersection(act_piv.columns)

            if len(common_act_zones) > 0:
                fcst_err = act_piv[common_act_zones] - load_piv[common_act_zones].reindex(act_piv.index)
                # Shift 48h to respect 5:00 AM D-1 DAM bidding cutoff without leakage
                err_lag48 = fcst_err.reindex(target_index).shift(48)

                feature_blocks.append(err_lag48.add_prefix("load_err_lag48_"))
                feature_blocks.append(err_lag48.rolling(24, min_periods=1).mean().add_prefix("load_err_r24m_"))
                feature_blocks.append(err_lag48.rolling(168, min_periods=24).mean().add_prefix("load_err_r7dm_"))
                feature_blocks.append(err_lag48.abs().rolling(24, min_periods=1).max().add_prefix("load_err_abs_r24max_"))


    # -----------------------------------------------------------------
    # 6. Block 2b: Lagged Zonal DART & MCC Congestion Features (P-2A & P-4A)
    # -----------------------------------------------------------------
    dam_z = merged["dam_lbmp_zone"].copy()
    rt_z = merged["rt_int_lbmp_zone"].copy()
    dam_z["Time Stamp"] = pd.to_datetime(dam_z["Time Stamp"])
    rt_z["Time Stamp"] = pd.to_datetime(rt_z["Time Stamp"])
    dam_z["Zone_Norm"] = _normalize_zone_name(dam_z["Name"])
    rt_z["Zone_Norm"] = _normalize_zone_name(rt_z["Name"])
    dam_z_piv = dam_z.drop_duplicates(["Time Stamp", "Zone_Norm"], keep="last").pivot(
        index="Time Stamp", columns="Zone_Norm", values="LBMP ($/MWHr)"
    )
    rt_z_piv = rt_z.drop_duplicates(["Time Stamp", "Zone_Norm"], keep="last").pivot(
        index="Time Stamp", columns="Zone_Norm", values="LBMP ($/MWHr)"
    )
    z_dart_lag48 = (dam_z_piv - rt_z_piv).reindex(target_index).shift(48)
    feature_blocks.append(z_dart_lag48.rolling(24, min_periods=1).mean().add_prefix("z_dart_r24m_"))
    feature_blocks.append(z_dart_lag48.rolling(168, min_periods=24).mean().add_prefix("z_dart_r7dm_"))
    feature_blocks.append(z_dart_lag48.abs().rolling(24, min_periods=1).max().add_prefix("z_dart_abs_r24max_"))
    mcc_col = next((c for c in dam_z.columns if "Congestion" in c), None)
    if mcc_col:
        dam_mcc_piv = dam_z.drop_duplicates(["Time Stamp", "Zone_Norm"], keep="last").pivot(
            index="Time Stamp", columns="Zone_Norm", values=mcc_col
        )
        rt_mcc_piv = rt_z.drop_duplicates(["Time Stamp", "Zone_Norm"], keep="last").pivot(
            index="Time Stamp", columns="Zone_Norm", values=mcc_col
        )
        mcc_dart_lag48 = (dam_mcc_piv - rt_mcc_piv).reindex(target_index).shift(48)
        feature_blocks.append(mcc_dart_lag48.rolling(24, min_periods=1).mean().add_prefix("z_mcc_dart_r24m_"))
        feature_blocks.append(mcc_dart_lag48.rolling(168, min_periods=24).mean().add_prefix("z_mcc_dart_r7dm_"))

    # -----------------------------------------------------------------
    # 7. Block 3: Nodal-to-Zonal DAM Congestion Dispersion (P-2B + P-19)
    # -----------------------------------------------------------------
    dam_g = merged["dam_lbmp_gen"].copy()
    gen_mcc_col = next((c for c in dam_g.columns if "Congestion" in c), None)
    if gen_mcc_col and "PTID" in gen_meta.columns and "Zone Name" in gen_meta.columns:
        dam_g["Time Stamp"] = pd.to_datetime(dam_g["Time Stamp"])
        dam_g["PTID_str"] = dam_g["PTID"].astype(str)
        ptid_to_zone = (
            gen_meta.dropna(subset=["PTID", "Zone Name"])
            .assign(PTID_str=lambda d: d["PTID"].astype(str), Zone_Norm=lambda d: _normalize_zone_name(d["Zone Name"]))
            .drop_duplicates("PTID_str")
            .set_index("PTID_str")["Zone_Norm"]
        )
        dam_g["Zone_Norm"] = dam_g["PTID_str"].map(ptid_to_zone)
        dam_g_valid = dam_g.dropna(subset=["Zone_Norm"])
        grp = dam_g_valid.groupby(["Time Stamp", "Zone_Norm"])[gen_mcc_col]

        # Shift 48h (not 24h) because Day D-1 DAM prices clear at ~11:00 EST on D-1,
        # which is 6 hours AFTER the 05:00 EST virtual bidding cutoff.
        nodal_std = grp.std().unstack("Zone_Norm").reindex(target_index).shift(48)
        nodal_rng = (grp.max() - grp.min()).unstack("Zone_Norm").reindex(target_index).shift(48)
        feature_blocks.append(nodal_std.add_prefix("nodal_mcc_std_lag48_"))
        feature_blocks.append(nodal_rng.add_prefix("nodal_mcc_rng_lag48_"))

    # -----------------------------------------------------------------
    # 8. P-54C & P-54B: Outages (System-wide hourly active outage counts)
    # -----------------------------------------------------------------
    for key, prefix in [("dam_sched_outages", "dam_outage"), ("rt_actual_outages", "rt_outage")]:
        if key in merged and not merged[key].empty:
            out_df = merged[key].copy()
            if "count" in out_df.columns and "Time Stamp" in out_df.columns:
                hourly_cnt = out_df.set_index("Time Stamp")[["count"]].rename(columns={"count": f"{prefix}_count"})
            else:
                ts_c = next((c for c in ["Time Stamp", "Timestamp", "Start Date", "Outage Start Date"] if c in out_df.columns), None)
                if not ts_c:
                    continue
                out_df[ts_c] = pd.to_datetime(out_df[ts_c], errors="coerce").dt.floor("h")
                hourly_cnt = out_df.groupby(ts_c).size().to_frame(f"{prefix}_count")

            if key == "rt_actual_outages":
                hourly_cnt = hourly_cnt.reindex(target_index).shift(48)
            else:
                hourly_cnt = hourly_cnt.reindex(target_index)
            feature_blocks.append(hourly_cnt)


    # -----------------------------------------------------------------
    # 9. P-511A: Day-Ahead Limiting Constraints
    # -----------------------------------------------------------------
    if "dam_limiting_constraints" in merged and not merged["dam_limiting_constraints"].empty:
        con_df = merged["dam_limiting_constraints"].copy()
        ts_c = next((c for c in ["Time Stamp", "Date Time"] if c in con_df.columns), None)
        if ts_c:
            con_df[ts_c] = pd.to_datetime(con_df[ts_c], errors="coerce").dt.floor("h")
            con_cnt = con_df.groupby(ts_c).size().to_frame("dam_binding_constraints_cnt")
            shadow_col = next((c for c in con_df.columns if "Shadow" in c or "Cost" in c), None)
            if shadow_col:
                con_df[shadow_col] = pd.to_numeric(con_df[shadow_col], errors="coerce").abs()
                con_shadow = con_df.groupby(ts_c)[shadow_col].sum().to_frame("dam_binding_shadow_sum")
                con_feat = pd.concat([con_cnt, con_shadow], axis=1)
            else:
                con_feat = con_cnt
            feature_blocks.append(con_feat.reindex(target_index).fillna(0.0))


    # -----------------------------------------------------------------
    # 10. P-63: Real-Time Fuel Mix (Lagged 48h to avoid leakage)
    # -----------------------------------------------------------------
    if "rt_fuel_mix" in merged and not merged["rt_fuel_mix"].empty:
        fuel_df = merged["rt_fuel_mix"].copy()
        if "Time Stamp" in fuel_df.columns and "Fuel Category" in fuel_df.columns:
            mw_col = next((c for c in ["Gen MW", "MW"] if c in fuel_df.columns), None)
            if mw_col:
                fuel_df["Time Stamp"] = pd.to_datetime(fuel_df["Time Stamp"]).dt.floor("h")
                fuel_hr = fuel_df.groupby(["Time Stamp", "Fuel Category"])[mw_col].mean().unstack("Fuel Category")
                fuel_lag48 = fuel_hr.reindex(target_index).shift(48)
                feature_blocks.append(fuel_lag48.add_prefix("fuel_mw_lag48_"))

    # -----------------------------------------------------------------
    # 11. P-32: External/Internal Interface Flows (Lagged 48h)
    # -----------------------------------------------------------------
    if "interface_flows" in merged and not merged["interface_flows"].empty:
        if_df = merged["interface_flows"].copy()
        if "Time Stamp" in if_df.columns and "Interface Name" in if_df.columns:
            flow_col = next((c for c in if_df.columns if "Flow" in c), None)
            pos_lim_col = next((c for c in if_df.columns if "Positive Limit" in c), None)
            if flow_col and pos_lim_col:
                if_df["Time Stamp"] = pd.to_datetime(if_df["Time Stamp"]).dt.floor("h")
                if_df["headroom_mw"] = pd.to_numeric(if_df[pos_lim_col], errors="coerce") - pd.to_numeric(if_df[flow_col], errors="coerce")
                hr_piv = if_df.groupby(["Time Stamp", "Interface Name"])["headroom_mw"].min().unstack("Interface Name")
                feature_blocks.append(hr_piv.reindex(target_index).shift(48).add_prefix("if_headroom_lag48_"))

    # -----------------------------------------------------------------
    # Combine, Align, and Add Calendar Harmonics
    # -----------------------------------------------------------------
    exog = pd.concat(feature_blocks, axis=1)
    exog = exog.loc[:, ~exog.columns.duplicated()]
    exog = exog.dropna(axis=1, how="all")  # Drops NYISO P-7A trailing whitespace column

    # Strictly causal forward-fill; only backfill the first 48h startup rows.
    # Any remaining NaNs later in the series are left for train-fitted SimpleImputer.
    exog = exog.reindex(target_index).ffill().bfill(limit=48)

    exog["hour_sin"] = np.sin(2 * np.pi * target_index.hour / 24.0)
    exog["hour_cos"] = np.cos(2 * np.pi * target_index.hour / 24.0)
    exog["dow_sin"] = np.sin(2 * np.pi * target_index.dayofweek / 7.0)
    exog["dow_cos"] = np.cos(2 * np.pi * target_index.dayofweek / 7.0)
    exog["is_weekend"] = (target_index.dayofweek >= 5).astype(float)
    return exog



def load_node_metadata() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Loads P-19 (Generator Names/PTIDs) and P-23 (Subzones by Transmission Owner)."""
    gen_meta_path = PARQUET_DIR / "meta_p19_generators.parquet"
    subzone_meta_path = PARQUET_DIR / "meta_p23_subzones.parquet"

    if not gen_meta_path.exists():
        p19_bytes = fetch_static_report("P-19")
        gen_df = pd.read_csv(io.BytesIO(p19_bytes))
        if "PTID" in gen_df.columns:
            gen_df["PTID"] = gen_df["PTID"].astype(str)
        gen_df.to_parquet(gen_meta_path, index=False)
    else:
        gen_df = pd.read_parquet(gen_meta_path)

    
    if not subzone_meta_path.exists():
        p23_bytes = fetch_static_report("P-23")
        subzone_df = pd.read_csv(io.BytesIO(p23_bytes))
        subzone_df.to_parquet(subzone_meta_path, index=False)
    else:
        subzone_df = pd.read_parquet(subzone_meta_path)
    return gen_df, subzone_df

def load_data(force_refresh_active: bool = False):
    if force_refresh_active:
        refresh_active_month(DATASETS, active_ym="202611")
    # 1. Fetch any new or missing archives (existing 8 datasets hit cache immediately)
    failures = fetch_all_datasets(DATASETS)
    total_failures = sum(len(urls) for urls in failures.values())
    if total_failures > 0:
        raise RuntimeError(f"{total_failures} archives failed to download. Re-run script to retry.")

    # 2. Merge datasets
    merged_data: dict[str, pd.DataFrame] = {}
    fingerprints: dict[str, str] = {}
    for report_code, folder_name in DATASETS.items():
        df, fp = merge_dataset(report_code, folder_name)
        merged_data[folder_name] = df
        fingerprints[folder_name] = fp

    # 3. Load Static Node & Subzone Metadata (P-19 & P-23) - needed early for nodal->zonal mapping
    gen_meta, subzone_meta = load_node_metadata()

    # 4. Build or load Nodal DART Spread & 3-Class Labels
    dart_path = PARQUET_DIR / "nodal_dart_spread.parquet"
    labels_path = PARQUET_DIR / "nodal_dart_3class_labels.parquet"
    dart_fp = combine_fingerprints(
        fingerprints["dam_lbmp_gen"],
        fingerprints["rt_int_lbmp_gen"],
        "w=720_q=0.15_0.85",
    )
    if is_parquet_fresh(dart_path, dart_fp) and is_parquet_fresh(labels_path, dart_fp):
        print("[CACHE HIT] Loading nodal_dart_spread.parquet & labels...")
        nodal_dart = pd.read_parquet(dart_path)
        nodal_labels = pd.read_parquet(labels_path)
    else:
        print("[REBUILDING] nodal_dart_spread.parquet & labels...")
        nodal_dart, nodal_labels = build_nodal_dart_and_labels(
            merged_data["dam_lbmp_gen"],
            merged_data["rt_int_lbmp_gen"],
            window_hours=24 * 30,
            lower_q=0.15,
            upper_q=0.85,
        )
        save_parquet_with_hash(nodal_dart, dart_path, dart_fp, keep_index=True)
        save_parquet_with_hash(nodal_labels, labels_path, dart_fp, keep_index=True)

    # 5. Build or load Zonal DART Spread & 3-Class Labels (P-2A & P-4A)
    zonal_dart_path = PARQUET_DIR / "zonal_dart_spread.parquet"
    zonal_labels_path = PARQUET_DIR / "zonal_dart_3class_labels.parquet"
    zonal_fp = combine_fingerprints(
        dart_fp,
        fingerprints["dam_lbmp_zone"],
        fingerprints["rt_int_lbmp_zone"],
        "w=720_q=0.15_0.85",
    )

    if is_parquet_fresh(zonal_dart_path, zonal_fp) and is_parquet_fresh(zonal_labels_path, zonal_fp):
        print("[CACHE HIT] Loading zonal_dart_spread.parquet & labels...")
        zonal_dart = pd.read_parquet(zonal_dart_path)
        zonal_labels = pd.read_parquet(zonal_labels_path)
    else:
        print("[REBUILDING] zonal_dart_spread.parquet & labels...")
        zonal_dart, zonal_labels = build_zonal_dart_and_labels(
            merged_data["dam_lbmp_zone"],
            merged_data["rt_int_lbmp_zone"],
            target_index=nodal_dart.index,
            window_hours=24 * 30,
            lower_q=0.15,
            upper_q=0.85,
        )
        save_parquet_with_hash(zonal_dart, zonal_dart_path, zonal_fp, keep_index=True)
        save_parquet_with_hash(zonal_labels, zonal_labels_path, zonal_fp, keep_index=True)

    # 6. Build or load Exogenous Feature Matrix (rebuilds automatically with new feeds)
    exog_path = PARQUET_DIR / "exogenous_features.parquet"
    exog_fp = combine_fingerprints(
        dart_fp,
        zonal_fp,
        fingerprints["iso_load_forecast"],
        fingerprints["weather_forecast"],
        fingerprints["zonal_bid_load"],
        fingerprints["btm_solar_forecast"],
        fingerprints["rt_int_actual_load"],
        fingerprints["dam_sched_outages"],
        fingerprints["rt_actual_outages"],
        fingerprints["dam_limiting_constraints"],
        fingerprints["rt_fuel_mix"],
        fingerprints["interface_flows"],
        "feat_v4",
    )

    if is_parquet_fresh(exog_path, exog_fp):
        print("[CACHE HIT] Loading exogenous_features.parquet...")
        exog_features = pd.read_parquet(exog_path)
    else:
        print("[REBUILDING] exogenous_features.parquet...")
        exog_features = build_exogenous_features(merged_data, nodal_dart.index, gen_meta)
        save_parquet_with_hash(exog_features, exog_path, exog_fp, keep_index=True)

    bid_dir = merged_data["dam_bid_data"]
    del merged_data
    gc.collect()

    print("\n=== Pipeline Complete ===")
    print(f"Nodal DART Matrix:      {nodal_dart.shape} (Timestamps x Generator Nodes)")
    print(f"Zonal DART Matrix:      {zonal_dart.shape} (Timestamps x Zones)")
    print(f"Zonal 3-Class Labels:   {zonal_labels.shape}")
    print(f"Exogenous Features:     {exog_features.shape}")
    print(f"DAM Bid Stack Dir:      {bid_dir}")
    print(f"P-19 Generator Meta:    {gen_meta.shape}")
    print(f"P-23 Subzone Meta:      {subzone_meta.shape}")
    return {
        "nodal_dart": nodal_dart,
        "nodal_labels": nodal_labels,
        "zonal_dart": zonal_dart,
        "zonal_labels": zonal_labels,
        "exog_features": exog_features,
        "dam_bids_dir": bid_dir,
        "gen_meta": gen_meta,
        "subzone_meta": subzone_meta,
    }
