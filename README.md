# Ensemble S+X — Climate-Health Probabilistic Forecasting

A CHAP-compatible probabilistic forecasting model that combines two
structurally distinct learners — **SARIMAX** for temporal autocorrelation and
**XGBoost** for non-linear covariate response — into a single ensemble.

Originally developed and benchmarked on weekly malaria case data from the
**Ngazidja (Grande Comore)** island, Union of the Comoros. Validated on
**5 districts** using ground-truth local station climate data, where it
outperformed SARIMAX, Prophet, and XGBoost as standalone models, as well as a
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

## Benchmark performance (Comoros, 5 districts — local station data)

Evaluated on weeks 79–104 (26-week test horizon), trained on weeks 1–78.
Climate data sourced from **ground-truth local meteorological stations**
(not satellite proxy).

| District | CRPS | RMSE | 80% PI coverage | 95% PI coverage |
|---|---|---|---|---|
| Hamahamet-Mboinkou | 22.23 | 38.00 | 69 % | 88 % |
| Hambou | 13.96 | 29.25 | 69 % | 92 % |
| Itsandra-Hamanvou | 21.51 | 39.89 | 69 % | 92 % |
| Mitsamiouli-Mboudé | 13.56 | 25.00 | 96 % | 100 % |
| Moroni-Bambao | 52.99 | 96.26 | 85 % | 92 % |
| **Aggregate (mean)** | **24.85** | **45.68** | **77.7 %** | **93.1 %** |

Overall R² = **0.91**

Best CRPS and best RMSE among all model configurations evaluated in the
[full research repository][research-repo].

[research-repo]: https://github.com/hisprwanda/sarimax-xgboost-comoros-malaria-prediction-model

> **Note on Moroni-Bambao:** This district has substantially higher case
> volumes than the others, resulting in higher absolute error metrics. The
> model's 95% PI coverage (92 %) remains well-calibrated, indicating that
> uncertainty is correctly quantified even when point accuracy is lower.
> Performance is expected to improve with additional training data (2+ years)
> as the XGBoost component learns non-linear climate-response patterns more
> reliably.

---

## CHAP usage

```bash
# Train
python train.py path/to/training_data.csv path/to/model.pkl

# Predict
python predict.py path/to/model.pkl path/to/historic_data.csv \
                  path/to/future_data.csv path/to/predictions.csv
```

CHAP discovers the entry points from the top-level `MLproject` file. The
model is exposed to CHAP as `ensemble_sx_climate_health`.

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
| `disease_cases` | int | Weekly case count (target). Rows with missing values are dropped automatically. |
| `rainfall` | float | mm / week |
| `mean_temperature` | float | °C |
| `humidity` | float | Relative humidity, % |

**Recommended minimum:** 78 weeks (≈18 months) per location. Two full years
strongly recommended for stable XGBoost feature selection.

**Missing value handling:**
- `disease_cases`: rows with missing target are dropped before training.
- Climate covariates: forward-filled within each location, then back-filled for any leading gaps.

### Future CSV

Same columns as training, **without** `disease_cases`. Climate covariates for
the future period must be supplied (typically from a climate forecast or
climatological average). Missing covariate values are forward-filled per
location.

### Output predictions CSV

CHAP standard format:

```
time_period,location,sample_0,sample_1,...,sample_99
2025-W27,Hamahamet-Mboinkou,87.3,72.1,...,94.8
2025-W27,Hambou,18.4,14.9,...,21.3
2025-W27,Itsandra-Hamanvou,54.2,49.0,...,61.7
2025-W27,Mitsamiouli-Mboudé,42.1,38.5,...,47.3
2025-W27,Moroni-Bambao,203.4,187.2,...,221.6
...
```

100 samples per row when `CHAP_N_SAMPLES=50` (default).

---

## Required climate covariates

CHAP validates that the following covariates are present in both training and
future data before invoking the model:

| Covariate | Description |
|---|---|
| `rainfall` | Weekly total rainfall (mm) |
| `mean_temperature` | Weekly mean temperature (°C) |
| `humidity` | Weekly mean relative humidity (%) |

---

## Robustness to climate forecast error

The model has been stress-tested under realistic future-climate uncertainty:

| Forecast horizon | Noise on inputs | CRPS degradation |
|---|---|---|
| Short-range (1–4 wk) | ±15 % rainfall, ±0.8 °C | +1.1 % |
| Extended-range (1–8 wk) | ±30 % rainfall, ±1.5 °C | +5.5 % |
| Seasonal (1–26 wk) | ±50 % rainfall, ±2.5 °C | +13.1 % |

The XGBoost component is robust to noisy inputs because its lag features draw
on exact training-tail values for the first forecast weeks. For operational
1–4 week health early-warning, the model is fully viable without
high-fidelity climate forecasts.

---

## Files

```
.
├── MLproject              # CHAP V2 manifest (entry points, required covariates, metadata)
├── pyproject.toml         # Python dependencies (managed by uv)
├── train.py               # CHAP training entry point
├── predict.py             # CHAP prediction entry point
├── model_lib.py           # SARIMAX + XGBoost implementations and feature engineering
├── README.md              # This file
└── example_data/
    ├── training_data.csv  # 78 weeks × 5 districts (Ngazidja, local station data)
    └── future_data.csv    # 26 weeks × 5 districts (forecast climate covariates)
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
