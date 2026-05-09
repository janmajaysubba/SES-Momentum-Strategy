# SES Momentum Strategy

A systematic daily momentum trading strategy built on **Simple Exponential Smoothing (SES)** of returns, normalized by rolling volatility into a z-score signal. The strategy is evaluated rigorously via **Walk-Forward Optimization (WFO)** across four U.S. equities.

---

## Table of Contents

1. [Overview](#overview)
2. [Strategy Logic](#strategy-logic)
3. [Walk-Forward Optimization](#walk-forward-optimization)
4. [Parameter Space](#parameter-space)
5. [In-Sample Calibration (Hierarchical Ranking)](#in-sample-calibration-hierarchical-ranking)
6. [Project Structure](#project-structure)
7. [Data](#data)
8. [Results](#results)
9. [Requirements](#requirements)
10. [Usage](#usage)

---

## Overview

This project implements a daily momentum strategy based on the hypothesis that the SES-smoothed return forecast, when normalized by recent volatility, captures a meaningful momentum signal. Rather than fitting the strategy once on all available data (which would overfit), parameters are selected and validated using a **rolling walk-forward framework** — the standard methodology for evaluating systematic trading strategies out-of-sample.

**Tickers:** SPY, QQQ, NVDA, MSFT

---

## Strategy Logic

### Step 1 — Daily Returns

Simple daily returns are computed from closing prices:

```
r(t) = P(t) / P(t-1) - 1
```

### Step 2 — Simple Exponential Smoothing (SES)

SES is applied to the return series to produce a smoothed momentum estimate:

```
error(t)       = r(t) - level(t-1)
level(t)       = level(t-1) + alpha * error(t)
```

where `alpha` is the smoothing parameter. A high `alpha` reacts quickly to recent returns (short memory); a low `alpha` weights the longer history more heavily.

### Step 3 — Volatility-Normalized Z-Score

The SES level is normalized by a rolling standard deviation of returns to produce a scale-free signal:

```
sigma(t) = rolling std of r(t) over the past vol_window days
z(t)     = level(t) / sigma(t)
```

This normalization ensures the signal magnitude is comparable across different volatility regimes and assets.

### Step 4 — Position Generation

Positions are determined daily using a symmetric threshold `k`:

```
position(t) = +1   if z(t) >  k   (long)
position(t) = -1   if z(t) < -k   (short)
position(t) =  0   otherwise      (flat)
```

### Step 5 — P&L Calculation

The strategy uses a **next-day execution** convention — a signal observed at the close of day `t` is traded at the close of day `t`, with P&L realized using the next day's return:

```
strategy_ret(t) = position(t) * r(t+1)
```

This reflects a realistic implementation where signals are computed after market close and positions entered at that close (or equivalently, at the next open).

---

## Walk-Forward Optimization

Walk-Forward Optimization (WFO) is the core evaluation methodology. Instead of a single train/test split, the strategy is calibrated and validated on hundreds of rolling windows across the full sample.

### Window Mechanics

For each window:
1. **In-Sample (IS) window** — A fixed-length segment of historical data used to calibrate the optimal `(alpha, k)` pair via grid search.
2. **Out-of-Sample (OOS) window** — The immediately following segment where the IS-selected parameters are applied without re-fitting.

Windows roll forward by `step_length` (defaulting to `oos_length`), producing **non-overlapping OOS segments** that together cover the full backtest period.

### Window Configurations

Four IS/OOS length combinations are tested to assess sensitivity to the calibration horizon:

| Config Name    | IS Length (days) | OOS Length (days) |
|----------------|-----------------|------------------|
| IS30_OOS5      | 30              | 5                |
| IS60_OOS7      | 60              | 7                |
| IS90_OOS10     | 90              | 10               |
| IS120_OOS15    | 120             | 15               |

**SES and z-scores are always computed on the full return history** before slicing into windows. This ensures the smoothing recursion has a proper warm-up and the volatility estimates are not artificially truncated at window boundaries.

---

## Parameter Space

| Parameter    | Values Tested                          | Description                            |
|--------------|----------------------------------------|----------------------------------------|
| `alpha`      | 0.05, 0.10, ..., 0.95 (19 values)     | SES smoothing parameter                |
| `k`          | 0.25, 0.50, ..., 2.00 (8 values)      | Z-score entry/exit threshold           |
| `vol_window` | 14 (fixed)                             | Rolling volatility window (trading days) |

Total grid size per IS window: **19 × 8 = 152 parameter combinations**.

---

## In-Sample Calibration (Hierarchical Ranking)

A simple "pick the best Sharpe ratio" criterion is prone to overfitting noisy short windows. The calibration applies **neighborhood averaging** and a **hierarchical fallback** to select more robust parameters.

### Neighborhood Averaging

For each grid point `(alpha, k)`, performance metrics are averaged over all neighboring grid points within:
- `|alpha' - alpha| <= 0.05`
- `|k' - k| <= 0.25`

This penalizes "spiky" parameter regions — a configuration that only works for one precise `(alpha, k)` pair is treated with suspicion compared to one that performs well across a range of nearby values.

### Hierarchical Selection

Selection proceeds through three tiers, moving to the next only if the current tier yields no valid candidates:

| Tier | Filter | Rank by |
|------|--------|---------|
| **Primary (neighborhood)** | `avg_n_trades >= min_trades` | Maximize `avg_mean_trade_ret` |
| **Fallback A (raw)** | `n_trades >= min_trades` | Maximize `mean_trade_ret` |
| **Fallback B (no filter)** | None | Maximize `mean_trade_ret` |

`min_trades = max(5, round(0.15 * is_length))` — a minimum trade count scaled to the IS window length ensures statistical meaningfulness. The `selection_mode` field in the output logs which tier was used for every window.

If IS calibration finds no valid configuration, the OOS window is **skipped** (no position taken), rather than forcing a trade with undefined parameters.

---

## Project Structure

```
Momentum/
├── ses_momentum_main.py   # Main driver: loads data, runs WFO, saves CSVs
├── wfo.py                 # WFO engine: window generation, IS calibration, OOS validation
├── helper.py              # Core library: data loading, SES, z-score, backtest, trade stats
├── data/
│   ├── SPY.json
│   ├── QQQ.json
│   ├── NVDA.json
│   └── MSFT.json
└── results/
    ├── ses_is_grid_all.csv
    ├── ses_best_params_all.csv
    ├── ses_oos_results_all.csv
    └── ses_oos_daily_all.csv
```

### Module Summary

**`helper.py`**
- `load_ticker_json` / `load_all_tickers` — parses OHLCV JSON files into DataFrames
- `compute_returns` — computes simple daily returns from closing prices
- `ses_on_returns` — applies the SES recursion to a return series
- `compute_z_score` — divides the SES level by rolling volatility
- `generate_positions_from_z` — maps z-score to `{-1, 0, +1}` positions using threshold `k`
- `backtest_from_z` — runs the full backtest for a given `(z, k)` pair
- `compute_trade_stats` — computes trade-level metrics (win rate, mean trade return, drawdown, etc.)

**`wfo.py`**
- `make_windows` — generates all rolling IS/OOS window tuples from a date index
- `precompute_ses_and_z` — computes SES and z-score on the full history for a given `alpha`
- `evaluate_params_in_sample_from_z` — backtests one `(k, window)` combination using precomputed z
- `calibrate_is_window` — runs the full IS grid search and v2 hierarchical selection for one window
- `validate_oos_window` — evaluates IS-selected `(alpha*, k*)` on the OOS segment
- `run_wfo_for_ticker` — orchestrates the full WFO loop for a single ticker and window config

**`ses_momentum_main.py`**
- Top-level driver that loops over all tickers and window configurations, collects all results, and saves four CSV files to `results/`.

---

## Data

Historical price data was sourced from **Godel Terminal** and downloaded in JSON format. Each file is stored in the `data/` directory and contains a list of daily OHLCV records:

```json
[
  {
    "open": 685.3,
    "high": 685.37,
    "low": 681.34,
    "close": 684.39,
    "volume": 60979750,
    "UTCDate": "Thu, 04 Dec 2025 00:00:00 GMT"
  },
  ...
]
```

The loader normalizes column names, parses `UTCDate` into a `DatetimeIndex`, and retains only the `[open, high, low, close, volume]` columns.

---

## Results

### Output Files

All output files are written to the `results/` directory after a full WFO run. Each CSV is tagged with `ticker`, `is_length`, `oos_length`, and `config_name` columns so results from all tickers and configurations can be analyzed from a single file.

#### `ses_is_grid_all.csv`

Raw in-sample grid search results. Contains one row per `(ticker, config, window, alpha, k)` combination — every point evaluated during IS calibration. Includes both the raw per-point trade statistics and the neighborhood-averaged metrics used for selection.

#### `ses_best_params_all.csv`

The selected parameters per IS window. For each `(ticker, config, window)` triplet, records the chosen `(alpha*, k*)`, their IS trade-level performance metrics (labeled with `_is` suffix), neighborhood-averaged metrics, and the `selection_mode` indicating which tier of the hierarchical ranking was used.

#### `ses_oos_results_all.csv`

Out-of-sample performance per window. Contains trade-level metrics (labeled with `_oos` suffix) for each OOS segment, evaluated using the IS-selected parameters. This is the primary file for assessing strategy performance on unseen data.

#### `ses_oos_daily_all.csv`

Daily-resolution OOS backtest records. Each row is a single trading day in an OOS window and contains the underlying return `r(t)`, z-score `z(t)`, position taken, next-day return, and strategy return. Used for constructing equity curves and time-series analysis.

---

### OOS Performance Summary

Aggregated out-of-sample trade statistics across all rolling windows, broken down by ticker and IS/OOS configuration. Mean and SD are per-trade return means and standard deviations. Avg Win % and Avg Loss % are the average return on winning and losing trades respectively.

#### SPY

| Metric         | IS30 / OOS5 | IS60 / OOS7 | IS90 / OOS10 | IS120 / OOS15 |
|----------------|-------------|-------------|--------------|---------------|
| Total Trades   | 300         | 283         | 254          | 202           |
| Winning Trades | 162         | 144         | 137          | 109           |
| Win Rate       | 54.00%      | 50.88%      | 53.94%       | 53.96%        |
| Mean           | 0.03%       | 0.04%       | -0.03%       | 0.00%         |
| SD             | 0.01151      | 0.00883     | 0.01113      | 0.00935       |
| Avg Win %      | 0.67%       | 0.63%       | 0.59%        | 0.59%         |
| Avg Loss %     | -0.75%      | -0.60%      | -0.79%       | -0.69%        |

#### QQQ

| Metric         | IS30 / OOS5 | IS60 / OOS7 | IS90 / OOS10 | IS120 / OOS15 |
|----------------|-------------|-------------|--------------|---------------|
| Total Trades   | 315         | 277         | 280          | 222           |
| Winning Trades | 161         | 143         | 148          | 114           |
| Win Rate       | 51.11%      | 51.62%      | 52.86%       | 51.35%        |
| Mean           | -0.02%      | 0.06%       | -0.04%       | -0.02%        |
| SD             | 0.01777      | 0.01152     | 0.01402      | 0.01190       |
| Avg Win %      | 0.81%       | 0.87%       | 0.82%        | 0.79%         |
| Avg Loss %     | -0.89%      | -0.82%      | -1.01%       | -0.88%        |

#### NVDA

| Metric         | IS30 / OOS5 | IS60 / OOS7 | IS90 / OOS10 | IS120 / OOS15 |
|----------------|-------------|-------------|--------------|---------------|
| Total Trades   | 307         | 289         | 308          | 313           |
| Winning Trades | 162         | 150         | 148          | 161           |
| Win Rate       | 52.77%      | 51.90%      | 48.05%       | 51.44%        |
| Mean           | -0.02%      | 0.16%       | -0.36%       | 0.02%         |
| SD             | 0.03280      | 0.03158     | 0.03617      | 0.03044       |
| Avg Win %      | 2.18%       | 2.42%       | 2.19%        | 2.06%         |
| Avg Loss %     | -2.49%      | -2.28%      | -2.71%       | -2.14%        |

#### MSFT

| Metric         | IS30 / OOS5 | IS60 / OOS7 | IS90 / OOS10 | IS120 / OOS15 |
|----------------|-------------|-------------|--------------|---------------|
| Total Trades   | 303         | 233         | 255          | 259           |
| Winning Trades | 154         | 121         | 134          | 123           |
| Win Rate       | 50.83%      | 51.93%      | 52.55%       | 47.49%        |
| Mean           | -0.06%      | 0.04%       | -0.03%       | -0.15%        |
| SD             | 0.01573      | 0.01496     | 0.01452      | 0.01498       |
| Avg Win %      | 1.06%       | 1.13%       | 0.98%        | 0.95%         |
| Avg Loss %     | -1.23%      | -1.13%      | -1.15%       | -1.14%        |

---

## Requirements

```
python >= 3.9
numpy
pandas
```

Install dependencies:

```bash
pip install numpy pandas
```

---

## Usage

```bash
python ses_momentum_main.py
```

The script will:
1. Load OHLCV data for SPY, QQQ, NVDA, and MSFT from `data/`
2. Run WFO across all four window configurations for each ticker
3. Save the four result CSVs to `results/`

To change tickers, window configurations, or the parameter grid, edit the `CONFIGURATION` block at the top of `ses_momentum_main.py`.
