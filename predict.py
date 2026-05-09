"""
CHAP prediction entry point — Ensemble S+X.

Usage
-----
    python predict.py <model.pkl> <historic_data.csv> <future_data.csv> <out_file.csv>

Arguments
---------
model.pkl
    Path to the bundle produced by train.py.

historic_data.csv
    Required by the CHAP signature for forward compatibility. This model
    does not need to read it — the lag-bridging tail data was captured at
    train time and stored inside the bundle.

future_data.csv
    The weeks to forecast. Must contain the same columns as the training
    CSV except disease_cases (which is what we are predicting).

out_file.csv
    Where to write predictions in CHAP standard format:
        time_period, location, sample_0, sample_1, ..., sample_(2N-1)
    where N = CHAP_N_SAMPLES (default 50). The Ensemble S+X concatenates
    50 SARIMAX samples + 50 XGBoost samples = 100 samples per row.
"""

import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd

import model_lib as ml

warnings.filterwarnings("ignore")

RNG_SEED = 42


def main(model_path: str, historic_csv: str, future_csv: str, out_csv: str):
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)

    if bundle.get("model_type") != "ensemble_sx":
        raise ValueError(f"Expected ensemble_sx bundle, got: {bundle.get('model_type')!r}")

    n_samples = bundle["n_samples"]
    rng       = np.random.default_rng(RNG_SEED)

    future_raw = pd.read_csv(future_csv)
    _validate_future(future_raw)

    # Bridge lag features at the train/predict boundary
    sx_bundle  = bundle["sarimax"]
    xgb_bundle = bundle["xgboost"]
    tail_data  = sx_bundle["tail_data"]

    base = pd.concat([tail_data, future_raw], ignore_index=True)
    base = ml.add_engineered_features(base)
    future_times = set(future_raw["time_period"])
    future_feat  = base[base["time_period"].isin(future_times)].reset_index(drop=True)

    sarimax_covs_by_loc = sx_bundle["covariates_by_location"]
    xgb_covs            = xgb_bundle["covariates"]

    records = []
    for loc in sorted(future_feat["location"].unique()):
        s_grp = future_feat[future_feat["location"] == loc].sort_values("time_period")
        if s_grp.empty:
            continue
        periods = s_grp["time_period"].values

        # ── SARIMAX half ──────────────────────────────────────────────────────
        if loc not in sx_bundle["models"]:
            print(f"[predict] WARNING: no SARIMAX model for {loc!r}, skipping")
            continue
        s_covs = sarimax_covs_by_loc.get(loc, ml.DEFAULT_COVARIATES)
        for c in s_covs:
            if c not in s_grp.columns:
                s_grp = s_grp.copy(); s_grp[c] = 0.0
        sX     = s_grp[s_covs].astype(float)
        s_samp = ml.predict_sarimax_one(sx_bundle["models"][loc], sX, n_samples, rng)

        # ── XGBoost half ──────────────────────────────────────────────────────
        if loc not in xgb_bundle["models"]:
            print(f"[predict] WARNING: no XGBoost model for {loc!r}, skipping")
            continue
        x_grp = future_feat[future_feat["location"] == loc].sort_values("time_period")
        for c in xgb_covs:
            if c not in x_grp.columns:
                x_grp = x_grp.copy(); x_grp[c] = 0.0
        xX     = x_grp[xgb_covs].astype(float)
        x_samp = ml.predict_xgb_one(xgb_bundle["models"][loc], xX,
                                    x_grp["time_period"], n_samples, rng)

        # ── Concatenate: N SARIMAX + N XGBoost = 2N samples ───────────────────
        combined = np.hstack([s_samp, x_samp])           # (n_periods, 2*n_samples)
        n_total  = combined.shape[1]
        for i, tp in enumerate(periods):
            row = {"time_period": tp, "location": loc}
            for j in range(n_total):
                row[f"sample_{j}"] = combined[i, j]
            records.append(row)

    out_df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    print(f"[predict] wrote {len(out_df)} rows -> {out_csv}")


def _validate_future(df: pd.DataFrame):
    required = {"time_period", "location", "rainfall", "mean_temperature", "humidity"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"Future CSV missing columns: {sorted(missing)}")


if __name__ == "__main__":
    if len(sys.argv) != 5:
        sys.exit("Usage: python predict.py <model.pkl> <historic.csv> <future.csv> <out.csv>")
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
