import numpy as np
import pandas as pd
from scipy.linalg import svd
from scipy.optimize import least_squares
from sklearn.metrics import f1_score


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


def _build_delay_embedded_snapshots(
    zonal_df: pd.DataFrame,
    target_zone: str,
    lags: tuple[int, ...] = (0, 24, 48),
) -> np.ndarray:
    """
    Builds a spatial + time-delay state matrix X_state of shape (T, N_features)
    across all 15 NYISO zones plus lagged target-zone coordinates to resolve
    24h/48h diurnal harmonics in the Koopman operator.
    """
    blocks = [zonal_df.to_numpy(dtype=float)]
    target_series = zonal_df[target_zone]
    for lag in lags:
        if lag > 0:
            blocks.append(target_series.shift(lag).bfill().to_numpy(dtype=float)[:, None])
    return np.hstack(blocks)


def _exact_dmd_step(
    X_win: np.ndarray,
    rank: int,
    horizon: int,
    target_col_idx: int = 0,
) -> float:
    """
    Fits rank-r Exact DMD (Mann & Kutz, 2015) on standardized window X_win of shape (W, D)
    and projects forward by `horizon` steps with unit-circle eigenvalue stabilization.
    """
    mu = X_win.mean(axis=0)
    sd = X_win.std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    Z = (X_win - mu) / sd

    # Snapshot matrices: X1 = [z_0, ..., z_{W-2}], X2 = [z_1, ..., z_{W-1}] (shape D x (W-1))
    X1 = Z[:-1].T
    X2 = Z[1:].T

    U, s, Vt = svd(X1, full_matrices=False)
    r = min(rank, int(np.sum(s > 1e-10)))
    if r == 0:
        return float(mu[target_col_idx])

    U_r = U[:, :r]
    s_inv = np.diag(1.0 / s[:r])
    V_r = Vt[:r, :].T

    A_tilde = U_r.T @ X2 @ V_r @ s_inv
    eigvals, W = np.linalg.eig(A_tilde)

    # Stabilize eigenvalues onto/inside the unit circle for multi-step (48h) projection
    mag = np.abs(eigvals)
    eigvals_stab = np.where(mag > 1.0, eigvals / mag, eigvals)

    # Exact DMD modes & initial modal amplitudes from the final observed snapshot z_{-1}
    Phi = X2 @ V_r @ s_inv @ W
    b, _, _, _ = np.linalg.lstsq(Phi, Z[-1], rcond=None)

    z_pred = np.real(Phi @ ((eigvals_stab ** horizon) * b))
    return float(z_pred[target_col_idx] * sd[target_col_idx] + mu[target_col_idx])


def fit_predict_exact_dmd(
    zonal_df: pd.DataFrame,
    target_zone: str,
    test_mask: np.ndarray,
    lag_hours: int = 48,
    window_hours: int = 168,
    rank: int = 6,
    stride: int = 6,
) -> np.ndarray:
    """
    Sliding-window Exact DMD baseline (Mann & Kutz, 2015).
    At each anchor t, uses only snapshots ending at t - lag_hours (strictly causal
    prior to the 5:00 AM D-1 NYISO bidding deadline).
    """
    state_mat = _build_delay_embedded_snapshots(zonal_df, target_zone)
    target_idx = list(zonal_df.columns).index(target_zone)
    test_indices = np.where(test_mask)[0]
    n_test = len(test_indices)
    preds = np.zeros(n_test, dtype=float)

    for i in range(0, n_test, stride):
        t = test_indices[i]
        end_idx = t - lag_hours + 1  # inclusive end of settled snapshots at t - 48h
        start_idx = max(0, end_idx - window_hours)
        if end_idx - start_idx < rank + 4:
            continue

        X_win = state_mat[start_idx:end_idx]
        # Forecast block of `stride` hours using exact multi-step horizons [48, 48+stride-1]
        mu = X_win.mean(axis=0)
        sd = np.where(X_win.std(axis=0) < 1e-8, 1.0, X_win.std(axis=0))
        Z = (X_win - mu) / sd

        X1 = Z[:-1].T
        X2 = Z[1:].T
        U, s, Vt = svd(X1, full_matrices=False)
        r = min(rank, int(np.sum(s > 1e-10)))
        if r == 0:
            block_val = float(mu[target_idx])
            preds[i : min(i + stride, n_test)] = block_val
            continue

        U_r = U[:, :r]
        s_inv = np.diag(1.0 / s[:r])
        V_r = Vt[:r, :].T
        A_tilde = U_r.T @ X2 @ V_r @ s_inv
        eigvals, W = np.linalg.eig(A_tilde)
        mag = np.abs(eigvals)
        eigvals_stab = np.where(mag > 1.0, eigvals / mag, eigvals)

        Phi = X2 @ V_r @ s_inv @ W
        b, _, _, _ = np.linalg.lstsq(Phi, Z[-1], rcond=None)

        for step in range(min(stride, n_test - i)):
            h = lag_hours + step
            z_pred = np.real(Phi @ ((eigvals_stab ** h) * b))
            preds[i + step] = float(z_pred[target_idx] * sd[target_idx] + mu[target_idx])

    return preds


def fit_predict_bop_dmd(
    zonal_df: pd.DataFrame,
    target_zone: str,
    test_mask: np.ndarray,
    lag_hours: int = 48,
    window_hours: int = 168,
    rank: int = 6,
    n_bags: int = 15,
    bag_frac: float = 0.75,
    stride: int = 12,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Bagging, Optimized Dynamic Mode Decomposition — BOP-DMD (Sashidhar & Kutz, 2021).
    Projects snapshots onto rank-r POD basis, optimizes continuous-time eigenvalues
    omega_k via real-stacked variable projection with stability constraints (Re(omega_k) <= 0)
    across B bootstrap bags, and returns ensemble mean forecast and std (UQ).
    """
    rng = np.random.default_rng(random_state)
    state_mat = _build_delay_embedded_snapshots(zonal_df, target_zone)
    target_idx = list(zonal_df.columns).index(target_zone)
    test_indices = np.where(test_mask)[0]
    n_test = len(test_indices)

    pred_mean = np.zeros(n_test, dtype=float)
    pred_std = np.zeros(n_test, dtype=float)

    for i in range(0, n_test, stride):
        t = test_indices[i]
        end_idx = t - lag_hours + 1
        start_idx = max(0, end_idx - window_hours)
        W_len = end_idx - start_idx
        if W_len < rank + 6:
            continue

        X_win = state_mat[start_idx:end_idx]
        mu = X_win.mean(axis=0)
        sd = np.where(X_win.std(axis=0) < 1e-8, 1.0, X_win.std(axis=0))
        Z = (X_win - mu) / sd  # (W_len, D)

        # Rank-r POD subspace of the window
        U, s, Vt = svd(Z.T, full_matrices=False)
        r = min(rank, int(np.sum(s > 1e-10)))
        if r == 0:
            pred_mean[i : min(i + stride, n_test)] = float(mu[target_idx])
            continue

        U_r = U[:, :r]  # (D, r)
        Y_pod = Z @ U_r  # (W_len, r) reduced state trajectory
        t_grid = np.arange(W_len, dtype=float)

        # Initialize continuous-time frequencies omega_0 from Exact DMD on Y_pod
        X1 = Y_pod[:-1].T
        X2 = Y_pod[1:].T
        U1, s1, Vt1 = svd(X1, full_matrices=False)
        s1_inv = np.diag(1.0 / np.maximum(s1, 1e-10))
        A_init = U1.T @ X2 @ Vt1.T @ s1_inv
        lam0 = np.linalg.eigvals(A_init)
        lam0 = np.where(np.abs(lam0) < 1e-6, 1e-6 + 0j, lam0)
        omega0 = np.log(lam0)
        # Enforce strict feasibility inside left half-plane bounds [lb, ub]
        re0 = np.clip(omega0.real, -0.49, -1e-4)
        im0 = np.clip(omega0.imag, -np.pi + 1e-4, np.pi - 1e-4)
        theta0 = np.concatenate([re0, im0])

        lb = np.concatenate([np.full(r, -0.5), np.full(r, -np.pi)])
        ub = np.concatenate([np.zeros(r), np.full(r, np.pi)])

        n_sub = max(r + 3, int(W_len * bag_frac))
        block_len = min(stride, n_test - i)
        bag_forecasts = np.zeros((n_bags, block_len), dtype=float)

        for b_idx in range(n_bags):
            sub_idx = np.sort(rng.choice(W_len, size=n_sub, replace=False))
            t_sub = t_grid[sub_idx]
            Y_sub = Y_pod[sub_idx]  # (n_sub, r)

            def residuals(theta: np.ndarray) -> np.ndarray:
                om = theta[:r] + 1j * theta[r:]
                Phi_c = np.exp(np.outer(t_sub, om))  # (n_sub, r) complex
                # Real-stacked Vandermonde matrix avoids complex casting in scipy TRF
                Phi_re = np.hstack([Phi_c.real, -Phi_c.imag])  # (n_sub, 2r)
                B_re, _, _, _ = np.linalg.lstsq(Phi_re, Y_sub, rcond=None)
                return (Y_sub - Phi_re @ B_re).ravel()

            res = least_squares(residuals, theta0, bounds=(lb, ub), method="trf", max_nfev=12)
            om_opt = res.x[:r] + 1j * res.x[r:]
            Phi_opt_c = np.exp(np.outer(t_sub, om_opt))
            Phi_opt_re = np.hstack([Phi_opt_c.real, -Phi_opt_c.imag])
            B_opt_re, _, _, _ = np.linalg.lstsq(Phi_opt_re, Y_sub, rcond=None)

            for step in range(block_len):
                t_future = float((W_len - 1) + lag_hours + step)
                phi_f_c = np.exp(t_future * om_opt)
                phi_f_re = np.concatenate([phi_f_c.real, -phi_f_c.imag])
                y_pod_f = phi_f_re @ B_opt_re
                z_f = U_r @ y_pod_f
                bag_forecasts[b_idx, step] = float(
                    z_f[target_idx] * sd[target_idx] + mu[target_idx]
                )

        pred_mean[i : i + block_len] = np.median(bag_forecasts, axis=0)
        pred_std[i : i + block_len] = np.std(bag_forecasts, axis=0)

    return pred_mean, pred_std


def evaluate_dmd_baselines(
    data: dict[str, pd.DataFrame],
    target_zone: str = "N.Y.C.",
    train_end: str = "2024-12-31 23:00:00",
    fee_per_mwh: float = 0.50,
    window_hours: int = 24 * 30,
    lower_q: float = 0.15,
    upper_q: float = 0.85,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Self-contained evaluation runner for Exact DMD (Mann & Kutz, 2015) and
    BOP-DMD (Sashidhar & Kutz, 2021) aligned identically to `evaluate_baselines`.
    """
    LAG_HOURS = 48

    X_nodes_df = data["nodal_dart"].shift(LAG_HOURS).iloc[LAG_HOURS:]
    X_exog_df = data["exog_features"].iloc[LAG_HOURS:].copy()
    zonal_all_df = data["zonal_dart"].reindex(X_nodes_df.index).ffill().bfill()
    zonal_spread = zonal_all_df[target_zone]

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

    zonal_all_df = zonal_all_df.loc[valid_mask]
    zonal_spread = zonal_spread.loc[valid_mask]
    low_thresh = low_thresh.loc[valid_mask]
    high_thresh = high_thresh.loc[valid_mask]

    y_series = pd.Series(0, index=zonal_spread.index, dtype=int)
    y_series[zonal_spread < low_thresh] = -1
    y_series[zonal_spread > high_thresh] = 1

    train_mask = np.asarray(zonal_spread.index <= pd.Timestamp(train_end), dtype=bool)
    test_mask = ~train_mask

    y_te = y_series.loc[test_mask].to_numpy()
    spread_te = zonal_spread.loc[test_mask]
    low_te = low_thresh.loc[test_mask].to_numpy()
    high_te = high_thresh.loc[test_mask].to_numpy()

    records: list[dict] = []
    pnl_curves: dict[str, pd.Series] = {}

    def _score_and_store(name: str, feature_label: str, continuous_fcst: np.ndarray):
        preds = np.zeros_like(y_te)
        preds[continuous_fcst > high_te] = 1
        preds[continuous_fcst < low_te] = -1

        macro_f1 = f1_score(y_te, preds, average="macro", zero_division=0)
        tail_f1 = f1_score(y_te, preds, labels=[-1, 1], average="macro", zero_division=0)
        fin = compute_trading_metrics(preds, spread_te, fee_per_mwh=fee_per_mwh)

        records.append({
            "Model": name,
            "Feature Set": feature_label,
            "Macro F1": round(float(macro_f1), 4),
            "Tail F1": round(float(tail_f1), 4),
            **fin,
        })
        pnl_series = (preds * spread_te.to_numpy()) - (np.abs(preds) * fee_per_mwh)
        pnl_curves[f"{name} [{feature_label}]"] = pd.Series(pnl_series, index=spread_te.index)

    # 1. Exact DMD (Mann & Kutz, 2015)
    exact_fcst = fit_predict_exact_dmd(
        zonal_df=zonal_all_df,
        target_zone=target_zone,
        test_mask=test_mask,
        lag_hours=LAG_HOURS,
        window_hours=168,
        rank=6,
        stride=6,
    )
    _score_and_store("Exact DMD (Mann & Kutz, 2015)", "15 Zones + Delay Lags", exact_fcst)

    # 2. BOP-DMD (Sashidhar & Kutz, 2021)
    bop_fcst, _ = fit_predict_bop_dmd(
        zonal_df=zonal_all_df,
        target_zone=target_zone,
        test_mask=test_mask,
        lag_hours=LAG_HOURS,
        window_hours=168,
        rank=6,
        n_bags=15,
        bag_frac=0.75,
        stride=12,
    )
    _score_and_store("BOP-DMD (Sashidhar & Kutz, 2021)", "15 Zones + Bagged VarPro", bop_fcst)

    return pd.DataFrame(records), pd.DataFrame(pnl_curves)
