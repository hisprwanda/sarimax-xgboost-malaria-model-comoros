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
import pandas as pd

import model_lib as ml

warnings.filterwarnings("ignore")


def main(train_csv: str, model_path: str):
    n_samples = int(os.environ.get("CHAP_N_SAMPLES", "50"))
    print(f"[train] ensemble_sx  n_samples={n_samples}")

    df = pd.read_csv(train_csv)
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
        sarimax_models[loc] = {
            "payload":    ml.fit_sarimax_one(y, X),
            "covariates": covs,
        }
        print(f"[train]   SARIMAX  {loc}  covariates={covs}")

    # ── XGBoost with quantile regression ─────────────────────────────────────
    print("[train] fitting XGBoost (quantile regression) ...")
    xgb_models = {}
    for loc, grp in df_feat.groupby("location", sort=False):
        grp = grp.sort_values("time_period")
        y   = grp["disease_cases"].astype(float)
        X   = grp[xgb_covs].astype(float)
        xgb_models[loc] = ml.fit_xgb_one(y, X, grp["time_period"])
        print(f"[train]   XGBoost  {loc}")

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


def _validate(df: pd.DataFrame):
    required = {"time_period", "location", "disease_cases",
                "rainfall", "mean_temperature", "humidity"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"Training CSV missing columns: {sorted(missing)}")
    if df[list(required)].isnull().any().any():
        raise ValueError("Null values found in required columns")
    dups = df.duplicated(subset=["location", "time_period"])
    if dups.any():
        raise ValueError(f"Duplicate (location, time_period) rows: {dups.sum()}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage: python train.py <train_data.csv> <model_output_path>")
    main(sys.argv[1], sys.argv[2])
