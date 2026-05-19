"""
CHAP training entry point — Ensemble S+X.

Usage
-----
    python train.py <train_data.csv> <model_output_path>

The training CSV must contain:
    time_period   — ISO week string, e.g. "2024-W01"
    location      — district / spatial unit identifier
    disease_cases — non-negative integer (target)
    rainfall      — mm/week
    mean_temperature — °C
    humidity      — relative humidity %

Outputs
-------
A pickle bundle containing, per location:
    SARIMAX(1,0,1) tuned with informed climate-lag features (Exp 03)
    XGBoost calibrated with quantile regression (Exp 05)
plus the lag-bridging tail data needed at predict time.

Environment variables
---------------------
    CHAP_N_SAMPLES — probabilistic samples PER COMPONENT (default 50).
                     The ensemble draws this many from each of SARIMAX and
                     XGBoost, concatenated to 2×n at predict time.
"""

import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd

import model_lib as ml

warnings.filterwarnings("ignore")


def main(train_csv: str, model_path: str):
    n_samples = int(os.environ.get("CHAP_N_SAMPLES", "50"))
    print(f"[train] ensemble_sx  n_samples={n_samples}")

    df = pd.read_csv(train_csv)
    df = _clean_training_data(df)
    _validate(df)

    # ── Engineer features and capture training tail for lag bridging ─────────
    df_feat   = ml.add_engineered_features(df.copy())
    tail_data = (df_feat.groupby("location", sort=False)
                        .tail(ml.N_LAG_ROWS)
                        .reset_index(drop=True))

    xgb_covs = [c for c in ml.DEFAULT_COVARIATES + ml.EXTRA_COVARIATES
                if c in df_feat.columns]

    # ── SARIMAX with per-location informed feature selection ─────────────────
    print("[train] fitting SARIMAX (informed features per location) ...")
    feature_map = ml.compute_location_feature_map(df_feat)
    sarimax_models = {}
    for loc, grp in df_feat.groupby("location", sort=False):
        grp  = grp.sort_values("time_period")
        y    = grp["disease_cases"].astype(float)
        covs = feature_map[loc]
        X    = grp[covs].astype(float)
        X, y = _remove_nan_rows(X, y)          # drop lag-NaN rows after feature engineering
        sarimax_models[loc] = {
            "payload":    ml.fit_sarimax_one(y, X),
            "covariates": covs,
        }
        print(f"[train]   SARIMAX  {loc}  covariates={covs}  n_rows={len(y)}")

    # ── XGBoost with quantile regression ─────────────────────────────────────
    print("[train] fitting XGBoost (quantile regression) ...")
    xgb_models = {}
    for loc, grp in df_feat.groupby("location", sort=False):
        grp = grp.sort_values("time_period")
        y   = grp["disease_cases"].astype(float)
        X   = grp[xgb_covs].astype(float)
        X, y = _remove_nan_rows(X, y)          # drop lag-NaN rows after feature engineering
        xgb_models[loc] = ml.fit_xgb_one(y, X, grp["time_period"].loc[X.index])
        print(f"[train]   XGBoost  {loc}  n_rows={len(y)}")

    # ── Bundle ───────────────────────────────────────────────────────────────
    bundle = {
        "model_type": "ensemble_sx",
        "n_samples":  n_samples,
        "sarimax": {
            "models":                {loc: v["payload"]    for loc, v in sarimax_models.items()},
            "covariates_by_location": {loc: v["covariates"] for loc, v in sarimax_models.items()},
            "tail_data":             tail_data,
        },
        "xgboost": {
            "models":     xgb_models,
            "covariates": xgb_covs,
            "tail_data":  tail_data,
        },
    }

    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"[train] saved -> {model_path}")


def _clean_training_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where the target is missing; forward-fill climate covariates.

    Missing disease_cases rows are removed early so feature engineering and
    tail-data capture work on a clean series. Missing covariate values are
    forward-filled (then back-filled for any leading NAs) within each location.
    NaN rows introduced later by lag feature engineering are handled at fit
    time via _remove_nan_rows().
    """
    n_before = len(df)
    df = df.dropna(subset=["disease_cases"]).copy()
    n_dropped = n_before - len(df)
    if n_dropped:
        print(f"[train] dropped {n_dropped} row(s) with missing disease_cases")

    cov_cols = ["rainfall", "mean_temperature", "humidity"]
    df[cov_cols] = df.groupby("location", sort=False)[cov_cols].ffill().bfill()
    return df


def _remove_nan_rows(
    X: pd.DataFrame, y: pd.Series
) -> tuple[pd.DataFrame, pd.Series]:
    """Drop rows where any column in X is NaN or y is NaN.

    Mirrors Knut's remove_nan_rows() pattern from simple_multistep_model.
    Applied after feature engineering so that lag-NaN rows at the start of
    each district series are cleanly excluded from model fitting.
    """
    mask = ~(X.isna().any(axis=1) | y.isna())
    return X[mask], y[mask]



def _validate(df: pd.DataFrame):
    required = {"time_period", "location", "disease_cases",
                "rainfall", "mean_temperature", "humidity"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"Training CSV missing columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("No training rows remain after dropping missing disease_cases")
    dups = df.duplicated(subset=["location", "time_period"])
    if dups.any():
        raise ValueError(f"Duplicate (location, time_period) rows: {dups.sum()}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage: python train.py <train_data.csv> <model_output_path>")
    main(sys.argv[1], sys.argv[2])
