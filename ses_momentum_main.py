"""
ses_momentum_main.py

Main driver script for SES-based momentum × volatility-normalized z-score
walk-forward optimization (WFO).

- Loads OHLCV JSON files for multiple tickers.
- Computes daily returns.
- Runs WFO for multiple IS/OOS window configurations.
- Logs results to CSV files for later analysis / paper tables.

Assumes the following files in the same directory:
    - helper.py  (with load_all_tickers, compute_returns, etc.)
    - wfo.py     (with make_windows, run_wfo_for_ticker, etc.)

And JSON data files in ./data named:
    SPY.json, QQQ.json, NVDA.json, MSFT.json
"""

from __future__ import annotations

import os
from typing import Dict, Any, List

import numpy as np
import pandas as pd

from helper import load_all_tickers, compute_returns
from wfo import run_wfo_for_ticker


# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------

DATA_DIR = "data"
OUTPUT_DIR = "results"

TICKERS = ["SPY", "QQQ", "NVDA", "MSFT"]

WINDOW_CONFIGS = [
    {"is_length": 30,  "oos_length": 5,  "name": "IS30_OOS5"},
    {"is_length": 60,  "oos_length": 7,  "name": "IS60_OOS7"},
    {"is_length": 90,  "oos_length": 10, "name": "IS90_OOS10"},
    {"is_length": 120, "oos_length": 15, "name": "IS120_OOS15"},
]

# SES & threshold grids
ALPHA_GRID = np.arange(0.05, 1.00, 0.05)  # 0.05, 0.10, ..., 0.95
K_GRID = [0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00]

VOL_WINDOW = 14


# ---------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------

def main() -> None:
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1) Load prices for all tickers
    print("Loading price data...")
    prices: Dict[str, pd.DataFrame] = load_all_tickers(DATA_DIR, TICKERS)

    # Containers for all-ticker, all-config results
    all_is_results: List[pd.DataFrame] = []
    all_best_params: List[pd.DataFrame] = []
    all_oos_results: List[pd.DataFrame] = []
    all_oos_daily: List[pd.DataFrame] = []

    # 2) Loop over tickers
    for ticker in TICKERS:
        print(f"\n=== Processing ticker: {ticker} ===")
        price_df = prices[ticker].copy()

        # Sort by date just to be safe
        price_df = price_df.sort_index()

        # Compute daily returns
        returns_df = compute_returns(price_df, price_col="close")
        full_returns = returns_df["ret"].astype(float)
        full_returns = full_returns.sort_index()

        # 3) Loop over window configurations
        for cfg in WINDOW_CONFIGS:
            is_len = cfg["is_length"]
            oos_len = cfg["oos_length"]
            cfg_name = cfg["name"]

            print(f"  Running WFO config: {cfg_name} "
                  f"(IS={is_len}, OOS={oos_len})")

            # Run walk-forward optimization for this ticker & config
            results: Dict[str, Any] = run_wfo_for_ticker(
                full_returns=full_returns,
                is_length=is_len,
                oos_length=oos_len,
                alpha_grid=ALPHA_GRID,
                k_grid=K_GRID,
                vol_window=VOL_WINDOW,
                step_length=None,  # default: step = oos_length
            )

            # -------------------------------
            # 3a) In-sample grid results
            # -------------------------------
            df_is = results.get("is_results", pd.DataFrame()).copy()
            if not df_is.empty:
                df_is["ticker"] = ticker
                df_is["is_length"] = is_len
                df_is["oos_length"] = oos_len
                df_is["config_name"] = cfg_name
                all_is_results.append(df_is)

            # -------------------------------
            # 3b) Best params per window
            # -------------------------------
            df_best = results.get("best_params", pd.DataFrame()).copy()
            if not df_best.empty:
                df_best["ticker"] = ticker
                df_best["is_length"] = is_len
                df_best["oos_length"] = oos_len
                df_best["config_name"] = cfg_name
                all_best_params.append(df_best)

            # -------------------------------
            # 3c) OOS performance per window
            # -------------------------------
            df_oos = results.get("oos_results", pd.DataFrame()).copy()
            if not df_oos.empty:
                df_oos["ticker"] = ticker
                df_oos["is_length"] = is_len
                df_oos["oos_length"] = oos_len
                df_oos["config_name"] = cfg_name
                all_oos_results.append(df_oos)

            # -------------------------------
            # 3d) Detailed OOS daily series
            # -------------------------------
            df_oos_daily = results.get("oos_detailed", pd.DataFrame()).copy()
            if not df_oos_daily.empty:
                df_oos_daily["ticker"] = ticker
                df_oos_daily["is_length"] = is_len
                df_oos_daily["oos_length"] = oos_len
                df_oos_daily["config_name"] = cfg_name
                all_oos_daily.append(df_oos_daily)

    # -----------------------------------------------------------------
    # 4) Concatenate and save all results to CSV
    # -----------------------------------------------------------------
    print("\nSaving aggregated CSVs to:", OUTPUT_DIR)

    # 4a) Full IS (alpha, k) grid across all windows, tickers, configs
    # Contains raw and neighborhood-averaged trade-level metrics for every
    # (alpha, k) combination evaluated in every IS window
    if all_is_results:
        is_results_all = pd.concat(all_is_results, ignore_index=True)
        is_results_path = os.path.join(OUTPUT_DIR, "ses_is_grid_all.csv")
        is_results_all.to_csv(is_results_path, index=False)
        print("  - Saved IS grid results to", is_results_path)
    else:
        print("  - No IS grid results to save.")

    # 4b) Best params per window
    # Contains selected (alpha*, k*) for each IS window, along with
    # the trade-level metrics of the selected params and selection_mode
    if all_best_params:
        best_params_all = pd.concat(all_best_params, ignore_index=True)

        # Rename IS trade-level metrics to make clear these are IS metrics
        rename_map = {
            "n_trades": "n_trades_is",
            "win_rate": "win_rate_is",
            "mean_trade_ret": "mean_trade_ret_is",
            "avg_win": "avg_win_is",
            "avg_loss": "avg_loss_is",
            "loss_asymmetry_ratio": "loss_asymmetry_ratio_is",
            "max_drawdown": "max_drawdown_is",
        }
        best_params_all = best_params_all.rename(columns=rename_map)

        best_params_path = os.path.join(OUTPUT_DIR, "ses_best_params_all.csv")
        best_params_all.to_csv(best_params_path, index=False)
        print("  - Saved best params to", best_params_path)
    else:
        print("  - No best-params results to save.")

    # 4c) OOS performance per window
    # Contains trade-level metrics for each OOS window evaluated
    # with the params selected from the corresponding IS window
    if all_oos_results:
        oos_results_all = pd.concat(all_oos_results, ignore_index=True)

        # Rename OOS trade-level metrics to make clear these are OOS metrics
        rename_map_oos = {
            "n_trades": "n_trades_oos",
            "win_rate": "win_rate_oos",
            "mean_trade_ret": "mean_trade_ret_oos",
            "avg_win": "avg_win_oos",
            "avg_loss": "avg_loss_oos",
            "loss_asymmetry_ratio": "loss_asymmetry_ratio_oos",
            "max_drawdown": "max_drawdown_oos",
        }
        oos_results_all = oos_results_all.rename(columns=rename_map_oos)

        oos_results_path = os.path.join(OUTPUT_DIR, "ses_oos_results_all.csv")
        oos_results_all.to_csv(oos_results_path, index=False)
        print("  - Saved OOS window results to", oos_results_path)
    else:
        print("  - No OOS window results to save.")

    # 4d) Detailed OOS daily backtest (for equity curves, plots, etc.)
    if all_oos_daily:
        oos_daily_all = pd.concat(all_oos_daily, axis=0)

        # Ensure chronological order by index (date)
        oos_daily_all = oos_daily_all.sort_index()

        # Move index (dates) into a column so CSV is explicit
        if oos_daily_all.index.name is None:
            oos_daily_all.index.name = "date"
        oos_daily_all = oos_daily_all.reset_index()

        oos_daily_path = os.path.join(OUTPUT_DIR, "ses_oos_daily_all.csv")
        oos_daily_all.to_csv(oos_daily_path, index=False)
        print("  - Saved detailed OOS daily series to", oos_daily_path)
    else:
        print("  - No detailed OOS daily results to save.")

    print("\nDone.")


if __name__ == "__main__":
    main()