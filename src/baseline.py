import warnings
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge, RidgeCV
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False


def compute_trading_metrics(
    preds: np.ndarray,
    realized_spread: pd.Series,
    fee_per_mwh: float = 0.50,
    hours_per_year: int = 8760,
) -> dict[str, float]:
    """
    Computes virtual bidding PnL and risk-adjusted metrics for 1 MWh unit positions:
        PnL_t = y_hat_t * S_target,t - |y_hat_t| * c
    """
    y_hat = np.asarray(preds, dtype=float)
    spread = realized_spread.to_numpy(dtype=float)

    active_mask = y_hat != 0
    n_trades = int(active_mask.sum())
    trade_freq = float(active_mask.mean())

    pnl = (y_hat * spread) - (np.abs(y_hat) * fee_per_mwh)
    cum_pnl = float(np.sum(pnl))
    mean_pnl_per_trade = float(pnl[active_mask].mean()) if n_trades > 0 else 0.0
    win_rate = float((pnl[active_mask] > 0).mean()) if n_trades > 0 else 0.0

    std_hourly = float(np.std(pnl, ddof=1))
    sharpe = (
        float((np.mean(pnl) / std_hourly) * np.sqrt(hours_per_year))
        if std_hourly > 0
        else 0.0
    )

    equity_curve = np.cumsum(pnl)
    running_max = np.maximum.accumulate(equity_curve)
    max_drawdown = float(np.max(running_max - equity_curve)) if len(pnl) > 0 else 0.0

    return {
        "Trades": n_trades,
        "Trade Freq (%)": round(trade_freq * 100, 1),
        "Win Rate (%)": round(win_rate * 100, 1),
        "Avg PnL/Trade ($)": round(mean_pnl_per_trade, 2),
        "Cum PnL ($)": round(cum_pnl, 2),
        "Annual Sharpe": round(sharpe, 2),
        "Max DD ($)": round(max_drawdown, 2),
    }


def _select_core_arimax_exog(X_exog_df: pd.DataFrame, target_zone: str) -> pd.DataFrame:
    """Selects a compact, well-conditioned subset of exogenous drivers for hourly ARIMAX."""
    zone_norm = (
        target_zone.upper()
        .strip()
        .replace(".", "")
        .replace(" ", "_")
    )
    prefixes = (
        f"load_fcst_{zone_norm}",
        f"net_load_fcst_{zone_norm}",
        f"unsched_load_mw_{zone_norm}",
        f"load_err_r24m_{zone_norm}",
        f"z_dart_r24m_{zone_norm}",
        "dam_outage_count",
        "dam_binding_constraints_cnt",
        "is_weekend",
    )
    selected = [c for c in X_exog_df.columns if c.startswith(prefixes)]
    if not selected:
        selected = list(X_exog_df.columns[:10])
    return X_exog_df[selected].copy()


def _fit_predict_hourly_arimax(
    zonal_spread: pd.Series,
    X_exog_df: pd.DataFrame,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    target_zone: str,
) -> np.ndarray:
    """
    Fits 24 separate hourly ARIMAX(1,0,1) models (one per hour-of-day h in 0..23)
    using strictly causal >=48h target lags and standardized exogenous features.
    """
    exog_sub = _select_core_arimax_exog(X_exog_df, target_zone)
    ar_df = exog_sub.copy()
    ar_df["ar_lag48"] = zonal_spread.shift(48)
    ar_df["ar_lag72"] = zonal_spread.shift(72)
    ar_df["ar_lag168"] = zonal_spread.shift(168)
    ar_df = ar_df.ffill().bfill().fillna(0.0)

    y_all = zonal_spread.to_numpy(dtype=float)
    hours = zonal_spread.index.hour.to_numpy()
    test_index = zonal_spread.index[test_mask]
    preds_te = np.zeros(int(test_mask.sum()), dtype=float)

    for h in range(24):
        h_mask = hours == h
        tr_h = train_mask & h_mask
        te_h = test_mask & h_mask
        if te_h.sum() == 0:
            continue

        scaler_h = StandardScaler()
        X_tr_h = np.clip(scaler_h.fit_transform(ar_df.loc[tr_h].to_numpy(dtype=float)), -10.0, 10.0)
        X_te_h = np.clip(scaler_h.transform(ar_df.loc[te_h].to_numpy(dtype=float)), -10.0, 10.0)
        y_tr_h = y_all[tr_h]

        te_positions = test_index.get_indexer(zonal_spread.index[te_h])

        if HAS_STATSMODELS and len(y_tr_h) >= 60:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = SARIMAX(
                        endog=y_tr_h,
                        exog=X_tr_h,
                        order=(1, 0, 1),
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    )
                    res = model.fit(disp=False, maxiter=200)
                    fcst = res.apply(endog=np.zeros(len(X_te_h)), exog=X_te_h).fittedvalues
                    # Direct linear projection using fitted ARIMAX exog + state params
                    exog_names = [f"x{i+1}" for i in range(X_tr_h.shape[1])]
                    beta = np.array([res.params.get(name, 0.0) for name in exog_names], dtype=float)
                    ar1 = float(res.params.get("ar.L1", 0.0))
                    # ar_lag48 is column -3 in ar_df (already standardized in X_te_h)
                    y_hat_h = X_te_h @ beta + ar1 * X_te_h[:, -3] * np.std(y_tr_h)
                    preds_te[te_positions] = y_hat_h
                    continue
            except Exception:
                pass

        ridge_h = Ridge(alpha=100.0)
        ridge_h.fit(X_tr_h, y_tr_h)
        preds_te[te_positions] = ridge_h.predict(X_te_h)

    return preds_te


def evaluate_baselines(
    data: dict[str, pd.DataFrame],
    target_zone: str = "N.Y.C.",
    train_end: str = "2024-12-31 23:00:00",
    fee_per_mwh: float = 0.50,
    window_hours: int = 24 * 30,
    lower_q: float = 0.15,
    upper_q: float = 0.85,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluates econometric and reference literature baselines on a strict causal 48h-lagged
    NYISO virtual bidding split:
      1. Causal 48h Persistence
      2. Hourly ARIMAX (24-hr)
      3. Continuous Ridge Regression
      4. Hubert et al. (2026) Multinomial Logit (Exog-Only & Exog + 559 Nodes)
      5. Galarneau-Vincent et al. (2023) GBDT (Exog-Only & Exog + 559 Nodes)
      6. Reference LDA & Balanced MLP (Exog-Only)
    """
    LAG_HOURS = 48

    X_nodes_df = data["nodal_dart"].shift(LAG_HOURS).iloc[LAG_HOURS:]
    X_exog_df = data["exog_features"].iloc[LAG_HOURS:].copy()

    if "zonal_dart" in data:
        zonal_spread = data["zonal_dart"][target_zone].reindex(X_nodes_df.index)
    else:
        zonal_spread = (
            data["zonal_dam"][target_zone] - data["zonal_rt"][target_zone]
        ).reindex(X_nodes_df.index)

    X_exog_df["target_spread_lag48"] = zonal_spread.shift(48)
    X_exog_df["target_spread_lag72"] = zonal_spread.shift(72)

    shifted_target_spread = zonal_spread.shift(LAG_HOURS)
    low_thresh = shifted_target_spread.rolling(
        window=window_hours, min_periods=24 * 7
    ).quantile(lower_q)
    high_thresh = shifted_target_spread.rolling(
        window=window_hours, min_periods=24 * 7
    ).quantile(upper_q)

    valid_mask = (
        low_thresh.notna()
        & high_thresh.notna()
        & zonal_spread.notna()
        & X_nodes_df.notna().all(axis=1)
        & X_exog_df["target_spread_lag72"].notna()
    )
    X_nodes_df = X_nodes_df.loc[valid_mask]
    X_exog_df = X_exog_df.loc[valid_mask]
    zonal_spread = zonal_spread.loc[valid_mask]
    low_thresh = low_thresh.loc[valid_mask]
    high_thresh = high_thresh.loc[valid_mask]

    y_series = pd.Series(0, index=zonal_spread.index, dtype=int)
    y_series[zonal_spread < low_thresh] = -1
    y_series[zonal_spread > high_thresh] = 1

    train_mask = np.asarray(X_nodes_df.index <= pd.Timestamp(train_end), dtype=bool)
    test_mask = ~train_mask

    X_nodes_tr, X_nodes_te = X_nodes_df.loc[train_mask], X_nodes_df.loc[test_mask]
    X_exog_tr, X_exog_te = X_exog_df.loc[train_mask], X_exog_df.loc[test_mask]
    y_tr, y_te = y_series.loc[train_mask].to_numpy(), y_series.loc[test_mask].to_numpy()

    spread_tr = zonal_spread.loc[train_mask].to_numpy()
    spread_te = zonal_spread.loc[test_mask]
    low_te = low_thresh.loc[test_mask].to_numpy()
    high_te = high_thresh.loc[test_mask].to_numpy()

    non_empty_exog_cols = X_exog_tr.columns[X_exog_tr.notna().any()]
    X_exog_tr = X_exog_tr[non_empty_exog_cols]
    X_exog_te = X_exog_te[non_empty_exog_cols]

    exog_imputer = SimpleImputer(strategy="median")
    Xe_tr_imp = exog_imputer.fit_transform(X_exog_tr)
    Xe_te_imp = exog_imputer.transform(X_exog_te)

    node_scaler = StandardScaler()
    Xn_tr_s = np.clip(node_scaler.fit_transform(X_nodes_tr), -10.0, 10.0)
    Xn_te_s = np.clip(node_scaler.transform(X_nodes_te), -10.0, 10.0)

    exog_scaler = StandardScaler()
    Xe_tr_s = np.clip(exog_scaler.fit_transform(Xe_tr_imp), -10.0, 10.0)
    Xe_te_s = np.clip(exog_scaler.transform(Xe_te_imp), -10.0, 10.0)

    X_full_tr = np.hstack([Xn_tr_s, Xe_tr_s])
    X_full_te = np.hstack([Xn_te_s, Xe_te_s])

    records = []
    hourly_pnl_curves = {}

    def record_result(model_name: str, feature_set: str, preds: np.ndarray):
        macro_f1 = f1_score(y_te, preds, average="macro", zero_division=0)
        tail_f1 = f1_score(y_te, preds, labels=[-1, 1], average="macro", zero_division=0)
        fin = compute_trading_metrics(preds, spread_te, fee_per_mwh=fee_per_mwh)

        records.append({
            "Model": model_name,
            "Feature Set": feature_set,
            "Macro F1": round(float(macro_f1), 4),
            "Tail F1": round(float(tail_f1), 4),
            **fin,
        })
        pnl_series = (preds * spread_te.to_numpy()) - (np.abs(preds) * fee_per_mwh)
        hourly_pnl_curves[f"{model_name} [{feature_set}]"] = pd.Series(
            pnl_series, index=spread_te.index
        )

    # 1. Causal 48h Persistence
    naive_preds = y_series.shift(LAG_HOURS).loc[test_mask].fillna(0).to_numpy(dtype=int)
    record_result("Causal 48h Persistence", "Target Lags", naive_preds)

    # 2. Hourly ARIMAX (24-hr)
    arimax_spread_te = _fit_predict_hourly_arimax(
        zonal_spread=zonal_spread,
        X_exog_df=X_exog_df[non_empty_exog_cols],
        train_mask=train_mask,
        test_mask=test_mask,
        target_zone=target_zone,
    )
    arimax_preds = np.zeros_like(y_te)
    arimax_preds[arimax_spread_te > high_te] = 1
    arimax_preds[arimax_spread_te < low_te] = -1
    record_result("Hourly ARIMAX (24-hr)", "Exog + AR(48,72,168)", arimax_preds)

    # 3. Continuous Ridge Regression
    ridge = RidgeCV(alphas=np.logspace(0, 4, 15))
    ridge.fit(X_full_tr, spread_tr)
    pred_spread_te = ridge.predict(X_full_te)

    reg_preds = np.zeros_like(y_te)
    reg_preds[pred_spread_te > high_te] = 1
    reg_preds[pred_spread_te < low_te] = -1
    record_result("Continuous Ridge Regression", "Exog + 559 Nodes", reg_preds)

    # 4. Hubert et al. (2026) Regularized Multinomial Logit
    logit_exog = LogisticRegression(
        C=0.1,
        class_weight="balanced",
        solver="lbfgs",
        max_iter=2000,
        tol=1e-3,
        random_state=42,
    )
    logit_exog.fit(Xe_tr_s, y_tr)
    record_result("Hubert et al. (2026) Multinomial Logit", "Exog-Only", logit_exog.predict(Xe_te_s))

    logit_full = LogisticRegression(
        C=0.1,
        class_weight="balanced",
        solver="lbfgs",
        max_iter=2000,
        tol=1e-3,
        random_state=42,
    )
    logit_full.fit(X_full_tr, y_tr)
    record_result("Hubert et al. (2026) Multinomial Logit", "Exog + 559 Nodes", logit_full.predict(X_full_te))

    # 5. Galarneau-Vincent et al. (2023) Gradient-Boosted Trees (GBDT)
    gbdt_exog = HistGradientBoostingClassifier(
        class_weight="balanced",
        max_depth=5,
        learning_rate=0.05,
        max_iter=200,
        random_state=42,
    )
    gbdt_exog.fit(Xe_tr_s, y_tr)
    record_result("Galarneau-Vincent et al. (2023) GBDT", "Exog-Only", gbdt_exog.predict(Xe_te_s))

    gbdt_full = HistGradientBoostingClassifier(
        class_weight="balanced",
        max_depth=5,
        learning_rate=0.05,
        max_iter=200,
        random_state=42,
    )
    gbdt_full.fit(X_full_tr, y_tr)
    record_result("Galarneau-Vincent et al. (2023) GBDT", "Exog + 559 Nodes", gbdt_full.predict(X_full_te))

    # 6. Reference LDA & Balanced MLP (Exog-Only)
    lda_exog = LinearDiscriminantAnalysis()
    lda_exog.fit(Xe_tr_s, y_tr)
    record_result("Reference LDA", "Exog-Only", lda_exog.predict(Xe_te_s))

    sample_weights_tr = compute_sample_weight("balanced", y_tr)
    mlp_exog = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        alpha=1e-1,
        max_iter=200,
        early_stopping=False,
        random_state=42,
    )
    mlp_exog.fit(Xe_tr_s, y_tr, sample_weight=sample_weights_tr)
    record_result("Reference Balanced MLP", "Exog-Only", mlp_exog.predict(Xe_te_s))

    results_df = pd.DataFrame(records)
    pnl_curves_df = pd.DataFrame(hourly_pnl_curves)
    return results_df, pnl_curves_df
