"""
Ensemble S+X — SARIMAX + XGBoost ensemble model implementation.

Two structurally distinct probabilistic forecasters combined via sample
concatenation (50 + 50 = 100 samples per location/week). They are
orthogonal:

  • SARIMAX(1,0,1) tuned   — linear state-space model with district-specific
    informed climate-lag features (selected via per-district cross-correlation
    at |r| > 0.10). Captures temporal autocorrelation and lagged climate effects.

  • XGBoost calibrated     — gradient-boosted trees on a 13-feature engineered
    set (lags, rolling means, climate interactions). Uses native multi-quantile
    objective (25 quantile levels) for honest probabilistic forecasts.
    Captures non-linear covariate response.

The two are combined by concatenating their probabilistic samples; this
preserves model disagreement as honest uncertainty and avoids the calibration
collapse caused by averaging medians.

This module is stateless library code — both train.py and predict.py import
from it. It is not run directly.
"""

import warnings
import numpy as np
import pandas as pd

# ── Constants ──────────────────────────────────────────────────────────────────

# Climate covariates the model expects in every input CSV
DEFAULT_COVARIATES = ["rainfall", "mean_temperature", "humidity"]

# Number of trailing training rows kept for lag bridging at the train/predict boundary
N_LAG_ROWS = 4

# Engineered feature names added by add_engineered_features()
EXTRA_COVARIATES = [
    "rainfall_lag1", "rainfall_lag2", "rainfall_lag3", "rainfall_lag4",
    "temp_lag1", "temp_lag2",
    "humidity_lag1", "humidity_lag2",
    "rainfall_roll4", "temp_roll4", "humidity_roll4",
    "rain_x_temp",
    "rain_x_humidity",
]

# SARIMAX configuration
DEFAULT_SARIMAX_ORDER = (1, 0, 1)
SARIMAX_FEATURE_THRESHOLD = 0.10   # |r| threshold for per-district feature selection

# XGBoost calibration via quantile regression
XGB_QUANTILE_LEVELS = np.linspace(0.025, 0.975, 25)
XGB_N_ESTIMATORS    = 100
XGB_MAX_DEPTH       = 4
XGB_LEARNING_RATE   = 0.05

# Lag → column name mapping used by compute_district_feature_map()
_LAG_COL = {
    ("rainfall",         1): "rainfall_lag1",
    ("rainfall",         2): "rainfall_lag2",
    ("rainfall",         3): "rainfall_lag3",
    ("rainfall",         4): "rainfall_lag4",
    ("mean_temperature", 1): "temp_lag1",
    ("mean_temperature", 2): "temp_lag2",
    ("humidity",         1): "humidity_lag1",
    ("humidity",         2): "humidity_lag2",
}
_AVAILABLE_LAGS = {
    "rainfall":         [1, 2, 3, 4],
    "mean_temperature": [1, 2],
    "humidity":         [1, 2],
}


# ── Week-period parsing ────────────────────────────────────────────────────────

def isoweek_to_timestamp(s: str) -> pd.Timestamp:
    """Convert a week-period string to the Monday of that ISO week.

    Accepts:
      'YYYY-Www'               — e.g. '2024-W01'
      'YYYYWww'                — e.g. '2024W01'
      'YYYY-MM-DD/YYYY-MM-DD'  — CHAP range; uses the start date
    """
    s = str(s).strip()
    if "/" in s:
        s = s.split("/")[0]
    if "W" in s.upper():
        s = s.upper().replace("W", "-W") if "-W" not in s.upper() else s
        try:
            return pd.to_datetime(s + "-1", format="%G-W%V-%u")
        except Exception:
            pass
    return pd.to_datetime(s)


# ── Feature engineering ────────────────────────────────────────────────────────

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lagged, rolling, and interaction features per location.

    Biological rationale
    --------------------
    rainfall_lag1-4   : mosquito breeding cycle (rain → standing water → larvae → adults) takes 2-4 weeks
    temp_lag1-2       : temperature governs larval development speed with ~1-2 week lag
    humidity_lag1-2   : humidity affects adult mosquito survival and biting rates
    *_roll4           : 4-week rolling mean captures sustained conditions
    rain_x_temp       : warm AND wet = ideal breeding environment
    rain_x_humidity   : wet AND humid = prolonged mosquito survival
    """
    out = df.copy()

    for loc, grp in out.groupby("location", sort=False):
        idx = grp.index
        r = grp["rainfall"]
        t = grp["mean_temperature"]
        h = grp["humidity"]

        for k in range(1, 5):
            out.loc[idx, f"rainfall_lag{k}"] = r.shift(k)
        for k in range(1, 3):
            out.loc[idx, f"temp_lag{k}"] = t.shift(k)
        for k in range(1, 3):
            out.loc[idx, f"humidity_lag{k}"] = h.shift(k)

        out.loc[idx, "rainfall_roll4"] = r.rolling(4, min_periods=1).mean()
        out.loc[idx, "temp_roll4"]     = t.rolling(4, min_periods=1).mean()
        out.loc[idx, "humidity_roll4"] = h.rolling(4, min_periods=1).mean()

        out.loc[idx, "rain_x_temp"]      = r * t
        out.loc[idx, "rain_x_humidity"]  = r * h

        # Fill NaN introduced by shifting the first rows
        for col in EXTRA_COVARIATES:
            out.loc[idx, col] = out.loc[idx, col].bfill().ffill()

    return out


# ── Per-location informed feature selection ───────────────────────────────────

def compute_location_feature_map(
    train_df: pd.DataFrame,
    threshold: float = SARIMAX_FEATURE_THRESHOLD,
) -> dict:
    """Return {location: [covariate_columns]} using per-location cross-correlation.

    For each location and each climate variable, finds the lag 0–8 with the
    highest |Pearson r| against disease_cases. Includes the corresponding lag
    column when |r| > threshold and the lag is available in the engineered set.
    Always includes the 3 base covariates regardless of correlation.

    Call this on the training split only to avoid data leakage.
    """
    from scipy import stats as sp_stats

    MAX_LAG = 8
    locations = sorted(train_df["location"].unique())
    feature_map = {}

    for loc in locations:
        grp = train_df[train_df["location"] == loc].sort_values("time_period")
        y   = grp["disease_cases"].values
        cols = list(DEFAULT_COVARIATES)

        for var in DEFAULT_COVARIATES:
            x = grp[var].values
            best_lag, best_r = 0, 0.0
            for lag in range(MAX_LAG + 1):
                xi = x[:-lag] if lag > 0 else x
                yi = y[lag:]  if lag > 0 else y
                if len(xi) < 10:
                    continue
                r, _ = sp_stats.pearsonr(xi, yi)
                if abs(r) > abs(best_r):
                    best_r, best_lag = r, lag

            if abs(best_r) > threshold and best_lag > 0:
                avail  = _AVAILABLE_LAGS.get(var, [])
                chosen = next((l for l in sorted(avail, reverse=True)
                               if l <= best_lag), None)
                if chosen is None and avail:
                    chosen = min(avail)
                if chosen is not None:
                    col = _LAG_COL.get((var, chosen))
                    if col and col not in cols:
                        cols.append(col)

        feature_map[loc] = cols

    return feature_map


# ── SARIMAX ────────────────────────────────────────────────────────────────────

def fit_sarimax_one(y: pd.Series, X: pd.DataFrame) -> dict:
    """Fit a SARIMAX(1,0,1) model for a single location."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SARIMAX(
            endog=y.values,
            exog=X.values,
            order=DEFAULT_SARIMAX_ORDER,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fit = model.fit(disp=False, maxiter=200)

    return {"order": DEFAULT_SARIMAX_ORDER, "fit": fit}


def predict_sarimax_one(
    payload: dict,
    future_X: pd.DataFrame,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw n_samples probabilistic forecasts from a fitted SARIMAX.

    Returns (n_periods, n_samples) array, clipped at 0.
    """
    fit = payload["fit"]
    n_periods = len(future_X)
    samples   = np.zeros((n_periods, n_samples))

    for i in range(n_samples):
        sim = fit.simulate(
            nsimulations=n_periods,
            exog=future_X.values,
            anchor="end",
            random_state=rng.integers(0, 2**31 - 1),
        )
        samples[:, i] = np.maximum(0, sim)

    return samples


# ── XGBoost ────────────────────────────────────────────────────────────────────

def _xgb_features(X_df: pd.DataFrame, time_index: pd.Index) -> np.ndarray:
    """Build feature matrix for XGBoost: climate covariates + temporal encoding."""
    weeks = [isoweek_to_timestamp(t) for t in time_index]
    week_of_year = np.array([w.isocalendar()[1] for w in weeks], dtype=float)
    month        = np.array([w.month for w in weeks], dtype=float)

    # Cyclical encoding prevents the model treating week 52→1 as a large jump
    sin_w = np.sin(2 * np.pi * week_of_year / 52)
    cos_w = np.cos(2 * np.pi * week_of_year / 52)
    sin_m = np.sin(2 * np.pi * month / 12)
    cos_m = np.cos(2 * np.pi * month / 12)

    temporal = np.column_stack([week_of_year, month, sin_w, cos_w, sin_m, cos_m])
    return np.hstack([X_df.values, temporal])


def fit_xgb_one(y: pd.Series, X: pd.DataFrame, time_index: pd.Index) -> dict:
    """Fit an XGBoost quantile regression model for a single location.

    Uses native multi-quantile objective for honest probabilistic forecasts.
    Replaces residual bootstrap (which collapses to near-zero intervals because
    in-sample residuals are too small).
    """
    import xgboost as xgb

    X_feat = _xgb_features(X, time_index)
    model = xgb.XGBRegressor(
        objective="reg:quantileerror",
        quantile_alpha=XGB_QUANTILE_LEVELS,
        n_estimators=XGB_N_ESTIMATORS,
        max_depth=XGB_MAX_DEPTH,
        learning_rate=XGB_LEARNING_RATE,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        random_state=42,
        verbosity=0,
    )
    model.fit(X_feat, y.values)

    return {"model": model, "quantile_levels": XGB_QUANTILE_LEVELS}


def predict_xgb_one(
    payload: dict,
    future_X: pd.DataFrame,
    future_times: pd.Index,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw n_samples by interpolating across the predicted quantile distribution.

    For each forecast step, draws u ~ U(0,1) and interpolates across the
    predicted quantile function — correctly capturing covariate-conditional
    uncertainty without relying on in-sample residuals.

    Returns (n_periods, n_samples) array, clipped at 0.
    """
    X_feat   = _xgb_features(future_X, future_times)
    q_preds  = payload["model"].predict(X_feat)            # (n_periods, n_quantiles)
    q_levels = payload["quantile_levels"]

    if q_preds.ndim == 1:
        q_preds = q_preds[:, None]
    q_preds = np.sort(q_preds, axis=1)                     # enforce monotonicity

    n_periods = q_preds.shape[0]
    u = rng.uniform(0, 1, size=(n_periods, n_samples))
    samples = np.zeros((n_periods, n_samples))
    for t in range(n_periods):
        samples[t] = np.interp(u[t], q_levels, q_preds[t])

    return np.maximum(0, samples)
