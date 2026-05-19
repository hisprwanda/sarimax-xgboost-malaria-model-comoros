# Ensemble S+X — Climate-Health Probabilistic Forecasting

A CHAP-compatible probabilistic forecasting model that combines two
structurally distinct learners — **SARIMAX** for temporal autocorrelation and
**XGBoost** for non-linear covariate response — into a single ensemble.

Originally developed and benchmarked on weekly malaria case data from the
Ngadjizi region of the **Union of the Comoros**, where it outperformed
SARIMAX, Prophet, and XGBoost as standalone models, as well as a
three-component (S+P+X) ensemble.

---

## What it does

For each spatial unit (district, region, etc.) and each forecast week, the
model produces a probabilistic forecast represented as a sample distribution.
This lets downstream tools compute any statistic — median, mean, prediction
intervals, exceedance probabilities for outbreak alerts, etc.

Two component models are fit independently per location and combined by
**sample concatenation** (50 SARIMAX samples + 50 XGBoost samples = 100
samples per location/week):

| Component | Role | Key technique |
|---|---|---|
| **SARIMAX(1,0,1) tuned** | Captures temporal autocorrelation and lagged climate signal | Per-location informed feature selection at \|r\| > 0.10 |
| **XGBoost calibrated**   | Captures non-linear climate-response interactions | Native multi-quantile objective (25 quantile levels) |

The two are structurally orthogonal — SARIMAX models *temporal* dependence,
XGBoost models *covariate-conditional* response — which is what makes their
combination genuinely additive.

---

## Benchmark performance (Comoros, 7 districts)

Evaluated on weeks 79–104 (26-week test horizon), training on weeks 1–78:

| Metric | Value |
|---|---|
| **CRPS** | **25.91** |
| **RMSE** | **48.84 cases/week** |
| **80% PI coverage** | **76.9 %** |
| **95% PI coverage** | **94.0 %** |

Best CRPS and best RMSE among 11 model configurations evaluated in the
[full research repository][research-repo].

[research-repo]: https://github.com/hisprwanda/sarimax-xgboost-comoros-malaria-prediction-model

---

## CHAP usage

```bash
# Train
python train.py path/to/training_data.csv path/to/model.pkl

# Predict
python predict.py path/to/model.pkl path/to/historic_data.csv \
                  path/to/future_data.csv path/to/predictions.csv
```

CHAP discovers the entry points and the column-name adapter map from the
top-level `MLproject` file. The model is exposed to CHAP as
`ensemble_sx_climate_health`.

### Environment variables

| Variable | Default | Effect |
|---|---|---|
| `CHAP_N_SAMPLES` | `50` | Probabilistic samples per ensemble component. The ensemble emits `2 × CHAP_N_SAMPLES` samples per row. |

---

## Data format

### Training CSV

| Column | Type | Description |
|---|---|---|
| `time_period` | string | ISO week, `YYYY-Www` (e.g. `2024-W01`) |
| `location` | string | District / region identifier |
| `disease_cases` | int | Weekly case count (target) |
| `rainfall` | float | mm / week |
| `mean_temperature` | float | °C |
| `humidity` | float | Relative humidity, % |

Required: at least one full year (52 weeks) per location. Two years strongly
recommended for stable feature selection.

### Future CSV

Same columns as training, **without** `disease_cases`. Climate covariates for
the future period must be supplied (typically from a climate forecast).

### Output predictions CSV

CHAP standard:

```
time_period,location,sample_0,sample_1,...,sample_99
2025-W27,Hamahamet-Mboinkou,87.3,72.1,...,94.8
...
```

100 samples per row when `CHAP_N_SAMPLES=50` (the default).

---

## Robustness to climate forecast error

The model has been stress-tested under realistic future-climate uncertainty:

| Forecast horizon | Noise on inputs | CRPS degradation |
|---|---|---|
| Short-range (1–4 wk) | ±15 % rainfall, ±0.8 °C | +1.1 % |
| Extended-range (1–8 wk) | ±30 % rainfall, ±1.5 °C | +5.5 % |
| Seasonal (1–26 wk) | ±50 % rainfall, ±2.5 °C | +13.1 % |

The XGBoost component is unusually robust to noisy inputs (its lag features
draw on exact training-tail values for the first weeks). For operational
1–4 week health early-warning, the model is fully viable without
high-fidelity climate forecasts.

---

## Files

```
.
├── MLproject              # CHAP V2 manifest (entry points, metadata)
├── pyproject.toml         # Python dependencies (managed by uv)
├── train.py               # CHAP training entry point
├── predict.py             # CHAP prediction entry point
├── model_lib.py           # SARIMAX + XGBoost implementations
├── README.md              # This file
└── example_data/
    ├── training_data.csv  # Sample input — 78 weeks × 7 districts (Comoros)
    └── future_data.csv    # Sample input — 4 weeks of future climate
```

---

## Local quick test

```bash
pip install -e .
python train.py example_data/training_data.csv /tmp/model.pkl
python predict.py /tmp/model.pkl example_data/training_data.csv \
                  example_data/future_data.csv /tmp/predictions.csv
head /tmp/predictions.csv
```

---

## Citation

If you use this model in research, please cite the originating study:

> Comoros Climate-Health Forecasting — Ensemble S+X Champion Model.
> HISP Rwanda, 2026.
> <https://github.com/hisprwanda/sarimax-xgboost-comoros-malaria-prediction-model>

---

## License

MIT.
