import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import qr, svd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, MultiTaskElasticNet, RidgeCV
from sklearn.metrics import f1_score, root_mean_squared_error
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from baseline import _apply_probability_threshold, compute_trading_metrics

RESULTS_CACHE_DIR = Path("./nyiso_data/ablation_cache")
RESULTS_CACHE_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================================
# 1. FINGERPRINTING & ATOMIC CACHE HELPERS
# =====================================================================


def _compute_experiment_fingerprint(
    X_nodes_tr: pd.DataFrame,
    X_exog_tr: pd.DataFrame,
    y_tr: np.ndarray,
    config: dict,
) -> str:
    """Computes a deterministic SHA-256 hash of training arrays, PTIDs, and ablation hyperparams."""
    hasher = hashlib.sha256()
    hasher.update(json.dumps(config, sort_keys=True).encode("utf-8"))
    hasher.update(",".join(map(str, X_nodes_tr.columns)).encode("utf-8"))
    hasher.update(",".join(map(str, X_exog_tr.columns)).encode("utf-8"))
    hasher.update(pd.util.hash_pandas_object(X_nodes_tr, index=True).to_numpy().tobytes())
    hasher.update(pd.util.hash_pandas_object(X_exog_tr, index=True).to_numpy().tobytes())
    hasher.update(y_tr.tobytes())
    return hasher.hexdigest()


def _load_cached_ablation(cache_prefix: Path, fp: str) -> dict | None:
    meta_path = cache_prefix.with_suffix(".meta.json")
    metrics_path = cache_prefix.with_suffix(".metrics.parquet")
    recon_path = cache_prefix.with_suffix(".recon.parquet")
    sensors_path = cache_prefix.with_suffix(".sensors.json")
    pnl_path = cache_prefix.with_suffix(".pnl.parquet")

    required = [meta_path, metrics_path, recon_path, sensors_path, pnl_path]
    if not all(p.exists() for p in required):
        return None

    meta = json.loads(meta_path.read_text())
    if meta.get("fingerprint") != fp:
        return None

    print(f"[CACHE HIT] Loading sparse sensor ablation results ({fp[:12]}...)...")
    return {
        "metrics_df": pd.read_parquet(metrics_path),
        "recon_df": pd.read_parquet(recon_path),
        "selected_sensors": json.loads(sensors_path.read_text()),
        "pnl_curves_df": pd.read_parquet(pnl_path),
        "meta": meta,
    }


def _save_cached_ablation(cache_prefix: Path, fp: str, payload: dict) -> None:
    payload["metrics_df"].to_parquet(cache_prefix.with_suffix(".metrics.parquet"), index=False)
    payload["recon_df"].to_parquet(cache_prefix.with_suffix(".recon.parquet"), index=False)
    payload["pnl_curves_df"].to_parquet(cache_prefix.with_suffix(".pnl.parquet"), index=True)
    cache_prefix.with_suffix(".sensors.json").write_text(
        json.dumps(payload["selected_sensors"], indent=2)
    )
    meta = {**payload["meta"], "fingerprint": fp}
    cache_prefix.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[CACHE SAVED] Saved ablation artifacts to {cache_prefix}.*")


# =====================================================================
# 2. SPARSE SENSOR SELECTION ALGORITHMS (QR, SSPOC, MUTUAL INFO)
# =====================================================================


def compute_pod_basis(Xn_tr_s: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes truncated Spatial POD modes Psi_r (N_nodes x r) from standardized training
    snapshots Xn_tr_s (T_train x N_nodes).
    """
    # SVD on X^T (N_nodes x T_train) -> left singular vectors Psi are spatial modes
    Psi, s, _ = svd(Xn_tr_s.T, full_matrices=False)
    r = min(rank, Psi.shape[1])
    return Psi[:, :r], s


def select_sensors_qr(Psi_r: np.ndarray, max_p: int) -> np.ndarray:
    """
    Manohar et al. (2018) Data-Driven Sensor Placement for Whole-Grid Reconstruction.
    Uses column-pivoted QR on Psi_r^T (for p <= r) and oversampled QR on Psi_r @ Psi_r^T
    (for p > r), returning ordered node indices of length `max_p`.
    """
    n_nodes, r = Psi_r.shape
    if max_p <= r:
        _, _, piv = qr(Psi_r.T, pivoting=True)
        return piv[:max_p]

    # First r pivots from exact modal basis, remaining (max_p - r) from oversampled Gram matrix
    _, _, piv_r = qr(Psi_r.T, pivoting=True)
    base_idx = list(piv_r[:r])
    _, _, piv_over = qr(Psi_r @ Psi_r.T, pivoting=True)
    for idx in piv_over:
        if idx not in base_idx:
            base_idx.append(int(idx))
        if len(base_idx) == max_p:
            break
    return np.array(base_idx, dtype=int)


def select_sensors_sspoc(
    Xn_tr_s: np.ndarray,
    y_tr: np.ndarray,
    Psi_r: np.ndarray,
    l1_ratio: float = 0.9,
    alpha_frac: float = 0.15,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Brunton et al. (2016) Sparse Sensor Placement Optimization for Classification (SSPOC).
    Calibrates alpha as a fraction of alpha_max so exactly ~20-50 discriminant nodes
    receive non-zero Group-LASSO weights independent of pod_rank r.
    """
    r = Psi_r.shape[1]
    A_tr = Xn_tr_s @ Psi_r  # (T_train, r)
    lda = LinearDiscriminantAnalysis()
    lda.fit(A_tr, y_tr)

    n_disc = min(len(np.unique(y_tr)) - 1, lda.scalings_.shape[1])
    W = lda.scalings_[:, :n_disc]  # (r, C-1)
    col_norms = np.linalg.norm(W, axis=0, keepdims=True)
    W_norm = W / np.where(col_norms < 1e-12, 1.0, col_norms)

    # Exact MultiTaskElasticNet zero-solution threshold: alpha_max = max_i ||(Psi_r @ W_norm)_i||_2 / (r * l1_ratio)
    grad_norms = np.linalg.norm(Psi_r @ W_norm, axis=1)
    alpha_max = float(np.max(grad_norms)) / max(r * l1_ratio, 1e-8)
    alpha_cal = alpha_frac * alpha_max

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        solver = MultiTaskElasticNet(
            alpha=alpha_cal,
            l1_ratio=l1_ratio,
            fit_intercept=False,
            max_iter=5000,
            tol=1e-4,
            random_state=42,
        )
        solver.fit(Psi_r.T, W_norm)

    S_weights = solver.coef_.T  # (N_nodes, C-1)
    node_scores = np.linalg.norm(S_weights, axis=1)
    n_nonzero = int(np.sum(node_scores > 1e-8))
    print(f"  [SSPOC] alpha_cal={alpha_cal:.5f} (alpha_frac={alpha_frac}) -> {n_nonzero} non-zero nodes")

    fallback_scores = grad_norms / (np.max(grad_norms) + 1e-12)
    composite_rank = node_scores + 1e-6 * fallback_scores
    ranked_indices = np.argsort(composite_rank)[::-1]
    return ranked_indices, node_scores



def select_sensors_mutual_info(
    Xn_tr_s: np.ndarray,
    y_tr: np.ndarray,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Univariate filter baseline: ranks generator nodes by Mutual Information with target class."""
    mi_scores = mutual_info_classif(Xn_tr_s, y_tr, discrete_features=False, random_state=random_state)
    ranked_indices = np.argsort(mi_scores)[::-1]
    return ranked_indices, mi_scores


# =====================================================================
# 3. WHOLE-GRID RECONSTRUCTION EVALUATION (MANOHAR ET AL., 2018)
# =====================================================================


def evaluate_whole_grid_reconstruction(
    Xn_tr_s: np.ndarray,
    Xn_te_s: np.ndarray,
    Psi_r: np.ndarray,
    sensor_idx: np.ndarray,
    node_std: np.ndarray,
) -> dict[str, float]:
    """
    Evaluates out-of-sample whole-grid reconstruction across all N = 559 generator nodes:
        X_hat = (X_te[:, S] @ pinv(Psi_r[S, :].T)) @ Psi_r.T
    """
    S = np.asarray(sensor_idx, dtype=int)
    Psi_S = Psi_r[S, :]  # (p, r)

    # Ridge-stabilized pseudoinverse prevents blow-up when p > r or sensors are collinear
    reg = 1e-4 * np.eye(Psi_S.shape[1])
    recon_op = np.linalg.solve(Psi_S.T @ Psi_S + reg, Psi_S.T)  # (r, p)
    X_te_hat_s = (Xn_te_s[:, S] @ recon_op.T) @ Psi_r.T  # (T_test, N_nodes)

    # Standardized R^2 across all 559 nodes
    ss_res = float(np.sum((Xn_te_s - X_te_hat_s) ** 2))
    ss_tot = float(np.sum((Xn_te_s - Xn_tr_s.mean(axis=0)) ** 2))
    r2_grid = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Physical $/MWh RMSE across all 559 nodes
    err_dollar = (Xn_te_s - X_te_hat_s) * node_std[None, :]
    rmse_dollar = float(np.sqrt(np.mean(err_dollar ** 2)))

    return {
        "Grid R2": round(r2_grid, 4),
        "Grid RMSE ($/MWh)": round(rmse_dollar, 2),
    }


# =====================================================================
# 4. MAIN CACHED ABLATION RUNNER
# =====================================================================


def run_sparse_sensor_ablation(
    data: dict[str, pd.DataFrame],
    target_zone: str = "N.Y.C.",
    val_end: str = "2023-12-31 23:00:00",
    train_end: str = "2024-12-31 23:00:00",
    p_budgets: tuple[int, ...] = (5, 10, 20, 40),
    pod_rank: int = 25,
    sspoc_alpha_frac: float = 0.02,
    sspoc_l1_ratio: float = 0.9,
    fee_per_mwh: float = 0.50,
    window_hours: int = 24 * 30,
    lower_q: float = 0.15,
    upper_q: float = 0.85,
    force_recompute: bool = False,
) -> dict:
    """
    Runs the full head-to-head ablation comparing:
      - QR-Pivoting (Whole-Grid Reconstruction)
      - SSPOC (Target-Specific Discriminant Placement)
      - Top-p Mutual Information (Univariate Filter)
    across sensor budgets p in `p_budgets` and downstream classifiers.
    Caches all artifacts deterministically on disk.
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
    sub_tr_mask = np.asarray(X_nodes_df.index <= pd.Timestamp(val_end), dtype=bool)
    val_mask = train_mask & (~sub_tr_mask)
    X_nodes_tr, X_nodes_te = X_nodes_df.loc[train_mask], X_nodes_df.loc[test_mask]
    X_exog_tr, X_exog_te = X_exog_df.loc[train_mask], X_exog_df.loc[test_mask]
    y_tr, y_te = y_series.loc[train_mask].to_numpy(), y_series.loc[test_mask].to_numpy()
    config = {
        "target_zone": target_zone,
        "val_end": val_end,
        "train_end": train_end,
        "p_budgets": list(p_budgets),
        "pod_rank": pod_rank,
        "alpha_frac": sspoc_alpha_frac,
        "sspoc_l1_ratio": sspoc_l1_ratio,
        "fee_per_mwh": fee_per_mwh,
        "window_hours": window_hours,
        "lower_q": lower_q,
        "upper_q": upper_q,
        "version": "ablation_v2",
    }
    
    fp = _compute_experiment_fingerprint(X_nodes_tr, X_exog_tr, y_tr, config)
    zone_slug = target_zone.lower().replace(".", "").replace(" ", "_")
    cache_prefix = RESULTS_CACHE_DIR / f"sparse_ablation_{zone_slug}"

    if not force_recompute:
        cached = _load_cached_ablation(cache_prefix, fp)
        if cached is not None:
            return cached

    print(f"[COMPUTING] Running Sparse Sensor Ablation for {target_zone} (fp={fp[:12]}...)...")

    # Standardize strictly on training split
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

    # Validation split scalers for tau tuning
    imp_sub = SimpleImputer(strategy="median")
    scl_sub_e = StandardScaler()
    scl_sub_n = StandardScaler()
    Xe_sub_tr = np.clip(scl_sub_e.fit_transform(imp_sub.fit_transform(X_exog_df[non_empty_exog_cols].loc[sub_tr_mask])), -10.0, 10.0)
    Xe_val = np.clip(scl_sub_e.transform(imp_sub.transform(X_exog_df[non_empty_exog_cols].loc[val_mask])), -10.0, 10.0)
    Xn_sub_tr = np.clip(scl_sub_n.fit_transform(X_nodes_df.loc[sub_tr_mask]), -10.0, 10.0)
    Xn_val = np.clip(scl_sub_n.transform(X_nodes_df.loc[val_mask]), -10.0, 10.0)
    y_sub_tr = y_series.loc[sub_tr_mask].to_numpy()
    spread_val = zonal_spread.loc[val_mask]
    spread_te = zonal_spread.loc[test_mask]

    # Compute POD basis & sensor rankings
    max_p = max(p_budgets)
    Psi_r, singular_vals = compute_pod_basis(Xn_tr_s, rank=pod_rank)
    qr_order = select_sensors_qr(Psi_r, max_p=max_p)
    sspoc_order, sspoc_scores = select_sensors_sspoc(
        Xn_tr_s, y_tr, Psi_r, l1_ratio=sspoc_l1_ratio, alpha_frac=sspoc_alpha_frac
    )
    mi_order, mi_scores = select_sensors_mutual_info(Xn_tr_s, y_tr)
    ptids = [str(c) for c in X_nodes_df.columns]
    selected_sensors = {
        "QR-Pivoting": {
            "ranked_indices": qr_order.tolist(),
            "ranked_ptids": [ptids[i] for i in qr_order],
            "scores": [float(1.0 / (rank_i + 1)) for rank_i in range(len(qr_order))],
        },
        "SSPOC": {
            "ranked_indices": sspoc_order[:max_p].tolist(),
            "ranked_ptids": [ptids[i] for i in sspoc_order[:max_p]],
            "scores": [float(sspoc_scores[i]) for i in sspoc_order[:max_p]],
            "all_node_scores": {ptids[i]: float(sspoc_scores[i]) for i in range(len(ptids))},
        },
        "Mutual-Info": {
            "ranked_indices": mi_order[:max_p].tolist(),
            "ranked_ptids": [ptids[i] for i in mi_order[:max_p]],
            "scores": [float(mi_scores[i]) for i in mi_order[:max_p]],
        },
    }

    # Evaluate Whole-Grid Reconstruction & Downstream Classification across p
    recon_records = []
    metric_records = []
    pnl_curves = {}
    schemes = {
        "QR-Pivoting": qr_order,
        "SSPOC": sspoc_order,
        "Mutual-Info": mi_order,
    }

    def _eval_classifier(
        scheme_name: str,
        p_val: int,
        model_name: str,
        preds: np.ndarray,
    ):
        preds_int = np.asarray(preds, dtype=int)
        macro_f1 = f1_score(y_te, preds_int, average="macro", zero_division=0)
        tail_f1 = f1_score(y_te, preds_int, labels=[-1, 1], average="macro", zero_division=0)
        fin = compute_trading_metrics(preds_int, spread_te, fee_per_mwh=fee_per_mwh)
        metric_records.append({
            "Scheme": scheme_name,
            "p": p_val,
            "Model": model_name,
            "Macro F1": round(float(macro_f1), 4),
            "Tail F1": round(float(tail_f1), 4),
            **fin,
        })
        key = f"{scheme_name} (p={p_val}) | {model_name}"
        pnl_curves[key] = pd.Series(
            (preds_int * spread_te.to_numpy()) - (np.abs(preds_int) * fee_per_mwh),
            index=spread_te.index,
        )

    for scheme_name, order in schemes.items():
        for p in p_budgets:
            S = order[:p]

            # 1. Whole-Grid Reconstruction Metrics
            recon = evaluate_whole_grid_reconstruction(
                Xn_tr_s=Xn_tr_s,
                Xn_te_s=Xn_te_s,
                Psi_r=Psi_r,
                sensor_idx=S,
                node_std=node_scaler.scale_,
            )
            recon_records.append({"Scheme": scheme_name, "p": p, **recon})

            # Combine Exogenous Features + Selected p Bellwether Nodes
            X_tr_p = np.hstack([Xe_tr_s, Xn_tr_s[:, S]])
            X_te_p = np.hstack([Xe_te_s, Xn_te_s[:, S]])
            X_sub_p = np.hstack([Xe_sub_tr, Xn_sub_tr[:, S]])
            X_val_p = np.hstack([Xe_val, Xn_val[:, S]])

            # 2a. Multinomial Logit (Balanced)
            logit = LogisticRegression(
                C=0.1, class_weight="balanced", solver="lbfgs", max_iter=2000, tol=1e-3, random_state=42
            )
            logit.fit(X_tr_p, y_tr)
            _eval_classifier(scheme_name, p, "Multinomial Logit", logit.predict(X_te_p))

            # 2b. GBDT (Balanced)
            gbdt = HistGradientBoostingClassifier(
                class_weight="balanced", max_depth=5, learning_rate=0.05, max_iter=200, random_state=42
            )
            gbdt.fit(X_tr_p, y_tr)
            _eval_classifier(scheme_name, p, "GBDT", gbdt.predict(X_te_p))

            # 2c. LDA
            lda = LinearDiscriminantAnalysis()
            lda.fit(X_tr_p, y_tr)
            _eval_classifier(scheme_name, p, "LDA", lda.predict(X_te_p))

            # 2d. Balanced MLP (Regularized + Early Stopping for Cross-Budget Stability)
            sw_tr = compute_sample_weight("balanced", y_tr)
            mlp = MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                alpha=0.5,                    # Stronger L2 weight decay prevents overfitting at p=20, 40
                batch_size=256,
                learning_rate="adaptive",
                learning_rate_init=1e-3,
                max_iter=600,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20,
                tol=1e-4,
                random_state=42,
            )
            mlp.fit(X_tr_p, y_tr, sample_weight=sw_tr)
            _eval_classifier(scheme_name, p, "Balanced MLP", mlp.predict(X_te_p))

            # 2e. Natural-Prior tau-tuned Logit (Chow 1970 reject option)
            tau_grid = np.arange(0.18, 0.42, 0.02)
            m_val = LogisticRegression(C=0.1, solver="lbfgs", max_iter=2000, tol=1e-3, random_state=42)
            m_val.fit(X_sub_p, y_sub_tr)
            val_probs = m_val.predict_proba(X_val_p)
            best_tau, best_sharpe = 0.25, -np.inf
            for tau in tau_grid:
                vp = _apply_probability_threshold(val_probs, m_val.classes_, tau)
                m_res = compute_trading_metrics(vp, spread_val, fee_per_mwh=fee_per_mwh)
                if m_res["Trade Freq (%)"] >= 15.0 and m_res["Annual Sharpe"] > best_sharpe:
                    best_sharpe = m_res["Annual Sharpe"]
                    best_tau = float(tau)

            m_full = LogisticRegression(C=0.1, solver="lbfgs", max_iter=2000, tol=1e-3, random_state=42)
            m_full.fit(X_tr_p, y_tr)
            te_preds = _apply_probability_threshold(m_full.predict_proba(X_te_p), m_full.classes_, best_tau)
            _eval_classifier(scheme_name, p, f"Logit (tau={best_tau:.2f})", te_preds)

    # Energy captured by rank-r POD basis
    pod_energy_ratio = float(np.sum(singular_vals[:pod_rank] ** 2) / np.sum(singular_vals ** 2))
    payload = {
        "metrics_df": pd.DataFrame(metric_records),
        "recon_df": pd.DataFrame(recon_records),
        "selected_sensors": selected_sensors,
        "pnl_curves_df": pd.DataFrame(pnl_curves),
        "meta": {
            **config,
            "n_nodes_total": len(ptids),
            "pod_energy_captured": round(pod_energy_ratio, 4),
        },
    }
    _save_cached_ablation(cache_prefix, fp, payload)
    return payload


def run_walkforward_sensor_stability(
    data: dict[str, pd.DataFrame],
    target_zone: str = "N.Y.C.",
    cutoffs: tuple[str, ...] = (
        "2024-12-31 23:00:00",
        "2025-03-31 23:00:00",
        "2025-06-30 23:00:00",
        "2025-09-30 23:00:00",
        "2025-12-31 23:00:00",
        "2026-03-31 23:00:00",
        "2026-06-30 23:00:00",
        "2026-09-30 23:00:00",
    ),
    p_budgets: tuple[int, ...] = (5, 10, 20),
    pod_rank: int = 25,
    sspoc_alpha_frac: float = 0.15,
    sspoc_l1_ratio: float = 0.9,
    window_hours: int = 24 * 30,
    lower_q: float = 0.15,
    upper_q: float = 0.85,
    force_recompute: bool = False,
) -> dict:
    """
    Expanding-window quarterly stability audit across 2025-2026.
    Re-fits POD, QR-Pivoting, SSPOC, and Mutual-Info at each quarterly cutoff and
    measures step-to-step and baseline-anchored Jaccard similarity of selected PTIDs.
    """
    LAG_HOURS = 48
    X_nodes_df = data["nodal_dart"].shift(LAG_HOURS).iloc[LAG_HOURS:]
    zonal_spread = data["zonal_dart"][target_zone].reindex(X_nodes_df.index)

    shifted_target = zonal_spread.shift(LAG_HOURS)
    low_thresh = shifted_target.rolling(window=window_hours, min_periods=24 * 7).quantile(lower_q)
    high_thresh = shifted_target.rolling(window=window_hours, min_periods=24 * 7).quantile(upper_q)

    valid_mask = low_thresh.notna() & high_thresh.notna() & zonal_spread.notna() & X_nodes_df.notna().all(axis=1)
    X_nodes_df = X_nodes_df.loc[valid_mask]
    zonal_spread = zonal_spread.loc[valid_mask]
    low_thresh = low_thresh.loc[valid_mask]
    high_thresh = high_thresh.loc[valid_mask]

    y_series = pd.Series(0, index=zonal_spread.index, dtype=int)
    y_series[zonal_spread < low_thresh] = -1
    y_series[zonal_spread > high_thresh] = 1

    # Filter cutoffs to those present in the dataset index
    max_ts = X_nodes_df.index.max()
    active_cutoffs = [c for c in cutoffs if pd.Timestamp(c) <= max_ts + pd.Timedelta(days=1)]

    zone_slug = target_zone.lower().replace(".", "").replace(" ", "_")
    cache_path = RESULTS_CACHE_DIR / f"walkforward_stability_{zone_slug}.json"
    config = {
        "target_zone": target_zone,
        "cutoffs": active_cutoffs,
        "p_budgets": list(p_budgets),
        "pod_rank": pod_rank,
        "sspoc_alpha_frac": sspoc_alpha_frac,
        "sspoc_l1_ratio": sspoc_l1_ratio,
        "n_rows": len(X_nodes_df),
        "version": "wf_v1",
    }
    fp = hashlib.sha256(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()

    if not force_recompute and cache_path.exists():
        cached = json.loads(cache_path.read_text())
        if cached.get("fingerprint") == fp:
            print(f"[CACHE HIT] Loaded walk-forward stability results for {target_zone}...")
            return cached

    print(f"[COMPUTING] Running expanding walk-forward stability across {len(active_cutoffs)} quarterly cutoffs...")
    ptids = [str(c) for c in X_nodes_df.columns]
    max_p = max(p_budgets)

    history: dict[str, dict[str, list[str]]] = {"QR-Pivoting": {}, "SSPOC": {}, "Mutual-Info": {}}

    for cutoff in active_cutoffs:
        mask = np.asarray(X_nodes_df.index <= pd.Timestamp(cutoff), dtype=bool)
        Xn_win = X_nodes_df.loc[mask]
        y_win = y_series.loc[mask].to_numpy()

        scaler = StandardScaler()
        Xn_s = np.clip(scaler.fit_transform(Xn_win), -10.0, 10.0)

        Psi_r, _ = compute_pod_basis(Xn_s, rank=pod_rank)
        qr_idx = select_sensors_qr(Psi_r, max_p=max_p)
        sspoc_idx, _ = select_sensors_sspoc(
            Xn_s, y_win, Psi_r, l1_ratio=sspoc_l1_ratio, alpha_frac=sspoc_alpha_frac
        )
        mi_idx, _ = select_sensors_mutual_info(Xn_s, y_win)

        label = cutoff[:10]
        history["QR-Pivoting"][label] = [ptids[i] for i in qr_idx[:max_p]]
        history["SSPOC"][label] = [ptids[i] for i in sspoc_idx[:max_p]]
        history["Mutual-Info"][label] = [ptids[i] for i in mi_idx[:max_p]]

    # Compute step-to-step and baseline-anchored Jaccard persistence
    jaccard_rows = []
    labels = [c[:10] for c in active_cutoffs]
    base_label = labels[0]

    for scheme, by_date in history.items():
        for p in p_budgets:
            base_set = set(by_date[base_label][:p])
            for idx_d in range(1, len(labels)):
                prev_l, curr_l = labels[idx_d - 1], labels[idx_d]
                prev_set = set(by_date[prev_l][:p])
                curr_set = set(by_date[curr_l][:p])

                jaccard_rows.append({
                    "Scheme": scheme,
                    "p": p,
                    "Quarter": curr_l,
                    "Step Jaccard (t vs t-1)": round(len(curr_set & prev_set) / len(curr_set | prev_set), 4),
                    "Anchor Jaccard (t vs 2024-12)": round(len(curr_set & base_set) / len(curr_set | base_set), 4),
                })

    payload = {
        "fingerprint": fp,
        "config": config,
        "history": history,
        "jaccard_records": jaccard_rows,
    }
    cache_path.write_text(json.dumps(payload, indent=2))
    print(f"[CACHE SAVED] Saved walk-forward stability to {cache_path}")
    return payload
