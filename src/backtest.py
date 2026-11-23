import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import RidgeCV
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from evaluate_models import SSPOCSelector


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

    # Annualized Sharpe Ratio across all hourly periods (0 PnL during abstention hours)
    std_hourly = float(np.std(pnl, ddof=1))
    sharpe = (
        float((np.mean(pnl) / std_hourly) * np.sqrt(hours_per_year))
        if std_hourly > 0
        else 0.0
    )

    # Maximum drawdown on cumulative hourly equity curve
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


from sklearn.impute import SimpleImputer


def evaluate_model_suite(
    data: dict[str, pd.DataFrame],
    target_zone: str = "N.Y.C.",
    sensor_budgets: list[int] = [5, 10, 20, 40],
    train_end: str = "2024-12-31 23:00:00",
    fee_per_mwh: float = 0.50,
    window_hours: int = 24 * 30,
    lower_q: float = 0.15,
    upper_q: float = 0.85,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, SSPOCSelector]]:
    # Strict 48h lag guarantees all 24 hours of Day D-2 are settled prior to Day D-1 05:00 AM bid cutoff
    LAG_HOURS = 48

    X_nodes_df = data["nodal_dart"].shift(LAG_HOURS).iloc[LAG_HOURS:]
    X_exog_df = data["exog_features"].iloc[LAG_HOURS:].copy()
    y_series = data["zonal_labels"][target_zone].iloc[LAG_HOURS:]

    if "zonal_dart" in data:
        zonal_spread = data["zonal_dart"][target_zone].reindex(X_nodes_df.index)
    else:
        zonal_spread = (
            data["zonal_dam"][target_zone] - data["zonal_rt"][target_zone]
        ).reindex(X_nodes_df.index)

    # Add causal target-zone lagged spread features (48h & 72h) so classifiers capture multi-day regime persistence
    X_exog_df["target_spread_lag48"] = zonal_spread.shift(48)
    X_exog_df["target_spread_lag72"] = zonal_spread.shift(72)

    # Leak-free rolling quantile thresholds (shifted by 48h)
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
    y_series = pd.Series(0, index=zonal_spread.index, dtype=int)
    y_series[zonal_spread < low_thresh] = -1
    y_series[zonal_spread > high_thresh] = 1
    y_series = y_series.loc[valid_mask]
    zonal_spread = zonal_spread.loc[valid_mask]
    low_thresh = low_thresh.loc[valid_mask]
    high_thresh = high_thresh.loc[valid_mask]


    train_mask = X_nodes_df.index <= pd.Timestamp(train_end)
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
    Xn_tr_s = node_scaler.fit_transform(X_nodes_tr)
    Xn_te_s = node_scaler.transform(X_nodes_te)

    exog_scaler = StandardScaler()
    Xe_tr_s = exog_scaler.fit_transform(Xe_tr_imp)
    Xe_te_s = exog_scaler.transform(Xe_te_imp)

    n_total_nodes = Xn_tr_s.shape[1]
    records = []
    hourly_pnl_curves = {}
    saved_selectors = {}

    def record_result(model_name: str, p_sensors: int, preds: np.ndarray):
        macro_f1 = f1_score(y_te, preds, average="macro", zero_division=0)
        tail_f1 = f1_score(y_te, preds, labels=[-1, 1], average="macro", zero_division=0)
        fin = compute_trading_metrics(preds, spread_te, fee_per_mwh=fee_per_mwh)

        records.append({
            "Model": model_name,
            "Sensors (p)": p_sensors,
            "Macro F1": round(float(macro_f1), 4),
            "Tail F1": round(float(tail_f1), 4),
            **fin,
        })
        pnl_series = (preds * spread_te.to_numpy()) - (np.abs(preds) * fee_per_mwh)
        hourly_pnl_curves[f"{model_name} (p={p_sensors})"] = pd.Series(
            pnl_series, index=spread_te.index
        )

    # 1. Strictly Causal Persistence Benchmark (48h Lagged Regime)
    naive_preds = y_series.shift(LAG_HOURS).loc[test_mask].fillna(0).to_numpy(dtype=int)
    record_result("Causal 48h Persistence", 0, naive_preds)

    # 2. Continuous Ridge Regression Baseline
    X_full_tr = np.hstack([Xn_tr_s, Xe_tr_s])
    X_full_te = np.hstack([Xn_te_s, Xe_te_s])

    ridge = RidgeCV(alphas=np.logspace(0, 4, 15))
    ridge.fit(X_full_tr, spread_tr)
    pred_spread_te = ridge.predict(X_full_te)

    reg_preds = np.zeros_like(y_te)
    reg_preds[pred_spread_te > high_te] = 1
    reg_preds[pred_spread_te < low_te] = -1
    record_result("Continuous Ridge Regression", n_total_nodes, reg_preds)

    # 3. Exogenous-Only Classifiers (p = 0)
    lda_exog = LinearDiscriminantAnalysis()
    lda_exog.fit(Xe_tr_s, y_tr)
    record_result("Exog-Only + LDA", 0, lda_exog.predict(Xe_te_s))

    mlp_exog = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        alpha=1e-1,
        max_iter=200,
        early_stopping=False,
        random_state=42,
    )
    mlp_exog.fit(Xe_tr_s, y_tr)
    record_result("Exog-Only + MLP", 0, mlp_exog.predict(Xe_te_s))

    # 4. SSPOC Sparse Sensor Sweep
    for p in sensor_budgets:
        sspoc = SSPOCSelector(n_modes=max(30, p + 10), n_sensors=p).fit(Xn_tr_s, y_tr)
        saved_selectors[p] = sspoc

        X_comb_tr = np.hstack([sspoc.transform(Xn_tr_s), Xe_tr_s])
        X_comb_te = np.hstack([sspoc.transform(Xn_te_s), Xe_te_s])

        lda = LinearDiscriminantAnalysis()
        lda.fit(X_comb_tr, y_tr)
        record_result("SSPOC + LDA", p, lda.predict(X_comb_te))

        mlp = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            alpha=1e-1,
            max_iter=200,
            early_stopping=False,
            random_state=42,
        )
        mlp.fit(X_comb_tr, y_tr)
        record_result("SSPOC + MLP", p, mlp.predict(X_comb_te))

    # 5. All-559-Nodes Classifiers (p = 559)
    lda_full = LinearDiscriminantAnalysis()
    lda_full.fit(X_full_tr, y_tr)
    record_result("All-559-Nodes + LDA", n_total_nodes, lda_full.predict(X_full_te))

    mlp_full = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        alpha=1e-1,
        max_iter=200,
        early_stopping=False,
        random_state=42,
    )
    mlp_full.fit(X_full_tr, y_tr)
    record_result("All-559-Nodes + MLP", n_total_nodes, mlp_full.predict(X_full_te))

    results_df = pd.DataFrame(records)
    pnl_curves_df = pd.DataFrame(hourly_pnl_curves)
    return results_df, pnl_curves_df, saved_selectors
