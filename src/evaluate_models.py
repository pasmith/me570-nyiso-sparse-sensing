import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.decomposition import TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.metrics import classification_report, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


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
    sensor_budgets: list[int] = [5, 10, 20, 40],
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

    # Drop exogenous columns that are mostly/entirely NaN in the 2020-2024 training split
    valid_exog_cols = X_exog_tr.columns[X_exog_tr.notna().mean() > 0.5]
    X_exog_tr = X_exog_tr[valid_exog_cols]
    X_exog_te = X_exog_te[valid_exog_cols]

    # Impute any isolated NaNs fit strictly on training data before scaling
    exog_imputer = SimpleImputer(strategy="median")
    Xe_tr_imp = exog_imputer.fit_transform(X_exog_tr)
    Xe_te_imp = exog_imputer.transform(X_exog_te)

    # Standardize features fit strictly on training data
    node_scaler = StandardScaler()
    Xn_tr_s = node_scaler.fit_transform(X_nodes_tr)
    Xn_te_s = node_scaler.transform(X_nodes_te)

    exog_scaler = StandardScaler()
    Xe_tr_s = exog_scaler.fit_transform(Xe_tr_imp)
    Xe_te_s = exog_scaler.transform(Xe_te_imp)

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
        sspoc = SSPOCSelector(n_modes=max(30, p), n_sensors=p).fit(Xn_tr_s, y_tr)
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
    selected_ptids = [str(node_ptids[i]) for i in selector.selected_indices_]
    weight_map = dict(zip(selected_ptids, selector.sensor_weights_))

    meta = data["gen_meta"].copy()
    meta.columns = meta.columns.str.strip()  # Remove any leading/trailing whitespace

    # Auto-detect NYISO P-19 column names (e.g., 'Generator PTID' vs 'PTID')
    ptid_col = next((c for c in meta.columns if "PTID" in c.upper()), meta.columns[0])
    name_col = next((c for c in meta.columns if "NAME" in c.upper()), ptid_col)
    zone_col = next((c for c in meta.columns if c.upper() == "ZONE"), None)
    subzone_col = next((c for c in meta.columns if "SUBZONE" in c.upper()), None)
    lat_col = next((c for c in meta.columns if "LAT" in c.upper()), "Latitude")
    lon_col = next((c for c in meta.columns if "LON" in c.upper()), "Longitude")

    # Clean numeric PTIDs (handles float strings like '23512.0' if NaNs existed in CSV)
    meta["PTID_str"] = (
        pd.to_numeric(meta[ptid_col], errors="coerce")
        .dropna()
        .astype(int)
        .astype(str)
    )
    meta = meta[meta["PTID_str"].isin(node_ptids)].dropna(subset=[lat_col, lon_col]).copy()

    meta["Role"] = np.where(
        meta["PTID_str"].isin(selected_ptids),
        "SSPOC Selected Sensor",
        "Unselected Generator Bus",
    )
    meta["Sensor Weight"] = meta["PTID_str"].map(weight_map).fillna(0.0)
    meta["Marker Size"] = np.where(meta["PTID_str"].isin(selected_ptids), 14, 5)

    # Build display columns dynamically based on what P-19 contains
    display_cols = ["PTID_str", name_col]
    if zone_col:
        display_cols.append(zone_col)
    if subzone_col:
        display_cols.append(subzone_col)
    display_cols.append("Sensor Weight")

    chosen_table = meta[meta["PTID_str"].isin(selected_ptids)].sort_values(
        "Sensor Weight", ascending=False
    )
    print(f"\n=== Top {len(selected_ptids)} SSPOC Sensor Nodes for {target_zone} ===")
    print(chosen_table[display_cols].to_string(index=False))

    hover_cols = [c for c in display_cols if c != name_col]
    fig = px.scatter_map(
        meta,
        lat=lat_col,
        lon=lon_col,
        color="Role",
        size="Marker Size",
        size_max=15,
        hover_name=name_col,
        hover_data=hover_cols,
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

import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd

# Maps NYISO external proxy zones/subzones to:
# (NYISO Border Substation Name, Border Lat, Border Lon, External Endpoint Lat, External Endpoint Lon)
EXTERNAL_PROXY_LINES = {
    "H Q":         ("Massena / Chateauguay (HQ)",      44.985, -74.845, 45.350, -73.850),
    "HQEX":        ("Massena / Chateauguay (HQ)",      44.985, -74.845, 45.350, -73.850),
    "NPX":         ("Pleasant Valley (ISO-NE AC)",     41.745, -73.820, 41.950, -72.800),
    "NPX-AC":      ("Pleasant Valley (ISO-NE AC)",     41.745, -73.820, 41.950, -72.800),
    "NPX-CSC":     ("Shoreham POI (Cross-Sound HVDC)", 40.960, -72.863, 41.275, -72.905),
    "PJM":         ("Ramapo 345kV (PJM-AC)",           41.135, -74.165, 40.660, -75.350),
    "PJM-AC":      ("Ramapo 345kV (PJM-AC)",           41.135, -74.165, 40.660, -75.350),
    "PJM-HTP":     ("W. 49th St POI (Hudson HVDC)",    40.766, -73.997, 40.824, -74.022),
    "PJM-NEPTUNE": ("Newbridge Rd POI (Neptune HVDC)", 40.728, -73.535, 40.460, -74.290),
    "PJM-VFT":     ("Linden VFT POI (ConEd NYC)",      40.635, -74.225, 40.630, -74.350),
    "OH":          ("Niagara / Beck (IESO)",           43.140, -79.040, 43.450, -79.750),
    "IESO":        ("Niagara / Beck (IESO)",           43.140, -79.040, 43.450, -79.750),
}




def plot_sspoc_sensors_with_interconnects(
    data: dict[str, pd.DataFrame],
    selector,
    node_ptids: list[str],
    target_zone: str = "N.Y.C.",
):
    selected_ptids = [str(node_ptids[i]) for i in selector.selected_indices_]
    weight_map = dict(zip(selected_ptids, selector.sensor_weights_))

    meta = data["gen_meta"].copy()
    meta.columns = meta.columns.str.strip()
    ptid_col = next(c for c in meta.columns if "PTID" in c.upper())
    name_col = next(c for c in meta.columns if "NAME" in c.upper())
    zone_col = next(c for c in meta.columns if c.upper() == "ZONE")
    subzone_col = next(c for c in meta.columns if "SUBZONE" in c.upper())

    meta["PTID_str"] = pd.to_numeric(meta[ptid_col], errors="coerce").dropna().astype(int).astype(str)
    meta = meta[meta["PTID_str"].isin(node_ptids)].copy()
    meta["Sensor Weight"] = meta["PTID_str"].map(weight_map).fillna(0.0)

    # 1. Split internal nodes (with lat/lon) vs. external proxy nodes (blank lat/lon)
    internal_meta = meta.dropna(subset=["Latitude", "Longitude"]).copy()
    internal_meta["Role"] = np.where(
        internal_meta["PTID_str"].isin(selected_ptids),
        "SSPOC Sensor (Internal)",
        "Unselected Generator Bus",
    )
    internal_meta["Marker Size"] = np.where(internal_meta["PTID_str"].isin(selected_ptids), 14, 5)

    fig = px.scatter_map(
        internal_meta,
        lat="Latitude",
        lon="Longitude",
        color="Role",
        size="Marker Size",
        size_max=15,
        hover_name=name_col,
        hover_data=["PTID_str", zone_col, subzone_col, "Sensor Weight"],
        color_discrete_map={
            "SSPOC Sensor (Internal)": "#E63946",
            "Unselected Generator Bus": "#457B9D",
        },
        zoom=5.6,
        center={"lat": 42.8, "lon": -75.8},
        map_style="carto-positron",
        title=f"SSPOC Sensor Placement & Cross-Border Interconnects (p={len(selected_ptids)}) for {target_zone}",
    )

    # 2. Overlay dashed cross-border tie-lines for any selected external proxy nodes
    ext_selected = meta[
        meta["PTID_str"].isin(selected_ptids) & (meta["Latitude"].isna() | meta["Longitude"].isna())
    ]

    for _, row in ext_selected.iterrows():
        key = str(row[subzone_col]).strip() if str(row[subzone_col]).strip() in EXTERNAL_PROXY_LINES else str(row[zone_col]).strip()
        if key not in EXTERNAL_PROXY_LINES:
            continue

        poi_name, b_lat, b_lon, ext_lat, ext_lon = EXTERNAL_PROXY_LINES[key]

        # Draw dashed tie-line from NYISO border substation to external control area
        fig.add_trace(
            go.Scattermap(
                lat=[b_lat, ext_lat],
                lon=[b_lon, ext_lon],
                mode="lines+markers",
                line=dict(width=3, color="#F4A261"),
                marker=dict(size=[10, 14], color=["#2A9D8F", "#F4A261"]),
                name=f"External Proxy: {row[name_col]}",
                hovertext=[
                    f"NYISO Border POI: {poi_name}",
                    f"External Proxy Sensor: {row[name_col]} (PTID {row['PTID_str']}, Weight={row['Sensor Weight']:.3f})",
                ],
                hoverinfo="text",
            )
        )

    fig.show()
    return meta[meta["PTID_str"].isin(selected_ptids)].sort_values("Sensor Weight", ascending=False)

