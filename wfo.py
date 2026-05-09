from __future__ import annotations

from typing import List, Tuple, Dict, Any, Iterable
import pandas as pd
import numpy as np

from helper import ses_on_returns, compute_z_score, backtest_from_z

def make_windows(
    index: pd.DatetimeIndex,
    is_length: int,
    oos_length: int,
    step_length: int | None = None,
) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """
    Generates rolling in-sample / out-of-sample windows over a date index.

    Each window is defined as:
        [IS_start, IS_end] -> [OOS_start, OOS_end]


    Parameters
    ----------
    index : pd.DatetimeIndex
        Sorted DatetimeIndex of the full sample (e.g., from your returns series).
    is_length : int
        Number of observations in the in-sample window (IS).
    oos_length : int
        Number of observations in the out-of-sample window (OOS).
    step_length : int, optional
        How far to roll forward the IS/OOS windows each step (in observations).
        If None, defaults to `oos_length` (non-overlapping OOS segments).

    Returns
    -------
    windows : list of tuples
        Each tuple is:
            (is_start, is_end, oos_start, oos_end)
        where each element is a pd.Timestamp taken from the index.

        You can then slice your data like:
            is_slice  = returns.loc[is_start:is_end]
            oos_slice = returns.loc[oos_start:oos_end]
    """
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("index must be a pd.DatetimeIndex")

    if len(index) == 0:
        return []

    # Ensure the index is sorted and unique
    index = index.sort_values().unique()
    n = len(index)

    if is_length <= 0 or oos_length <= 0:
        raise ValueError("is_length and oos_length must be positive integers")

    if step_length is None:
        step_length = oos_length

    if step_length <= 0:
        raise ValueError("step_length must be a positive integer")

    windows: List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []

    # Start the first IS window at position 0
    is_start_pos = 0

    while True:
        is_end_pos = is_start_pos + is_length - 1
        oos_start_pos = is_end_pos + 1
        oos_end_pos = oos_start_pos + oos_length - 1

        # Stop if we don't have enough data to form a full IS + OOS window
        if oos_end_pos >= n:
            break

        is_start = index[is_start_pos]
        is_end = index[is_end_pos]
        oos_start = index[oos_start_pos]
        oos_end = index[oos_end_pos]

        windows.append((is_start, is_end, oos_start, oos_end))

        # Roll the window forward
        is_start_pos += step_length

        # If the next IS would be out of range, stop
        if is_start_pos + is_length >= n:
            break

    return windows

def precompute_ses_and_z(
    full_returns: pd.Series,
    alpha: float,
    vol_window: int = 14,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Computes SES (level_curr) and volatility-normalized z-score
    for the ENTIRE returns series, for a given alpha and vol_window.


    we use the full history to build SES + z, then later slice into
    in-sample / out-of-sample windows.

    Parameters
    ----------
    full_returns : pd.Series
        Full daily returns r(t), indexed by date (entire backtest period).
    alpha : float
        SES smoothing parameter.
    vol_window : int
        Rolling window length for volatility normalization (e.g., 14).

    Returns
    -------
    ses_df_full : pd.DataFrame
        Output of ses_on_returns(full_returns, alpha) over the full sample.
    z_full : pd.Series
        z(t) over the full sample, same index as full_returns.
    """
    # 1) SES over the full history
    ses_df_full = ses_on_returns(full_returns, alpha)
    level_curr_full = ses_df_full["level_curr"]
    ret_full_aligned = ses_df_full["ret"]  # aligned with SES

    # 2) z-score over the full history
    z_full = compute_z_score(
        level_curr=level_curr_full,
        returns=ret_full_aligned,
        vol_window=vol_window,
    )

    # Ensure index = full_returns.index (reintroduce NaNs if needed)
    z_full = z_full.reindex(full_returns.index)

    return ses_df_full, z_full

def evaluate_params_in_sample_from_z(
    full_returns: pd.Series,
    z_full: pd.Series,
    is_start: pd.Timestamp,
    is_end: pd.Timestamp,
    k: float,
) -> Dict[str, Any]:
    """
    Evaluate ONE threshold k in a GIVEN in-sample window,
    assuming SES + z_full have already been computed on the full series.

    Parameters
    ----------
    full_returns : pd.Series
        Full daily returns for the entire backtest period.
    z_full : pd.Series
        z-score series computed over the full sample (from precompute_ses_and_z).
    is_start, is_end : pd.Timestamp
        Start and end dates of the in-sample window.
    k : float
        Z-score threshold for long/short decisions.

    Returns
    -------
    perf : dict
        Trade-level performance stats for this window + this k.
    """
    # Slice IS window
    is_returns = full_returns.loc[is_start:is_end]
    is_z = z_full.loc[is_start:is_end]

    # Run in-sample backtest
    _, perf = backtest_from_z(
        returns=is_returns,
        z=is_z,
        k=k,
    )

    # Add parameters and window info for tracking
    perf["k"] = k
    perf["is_start"] = is_start
    perf["is_end"] = is_end

    return perf

def calibrate_is_window(
    full_returns: pd.Series,
    is_start: pd.Timestamp,
    is_end: pd.Timestamp,
    alpha_grid: Iterable[float],
    k_grid: Iterable[float],
    vol_window: int = 14,
    alpha_neighborhood: float = 0.05,
    k_neighborhood: float = 0.25,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Calibrates (alpha, k) for ONE in-sample window [is_start, is_end]
    using the v2 hierarchical ranking criterion.

    Steps (per window)
    ------------------
    1) For each alpha in alpha_grid:
           - Precomputes SES + z_full on the ENTIRE return series (vol_window fixed).
    2) For each k in k_grid:
           - Slices IS [is_start, is_end].
           - Backtests and records trade-level performance metrics.
    3) For each grid point (alpha, k), computes NEIGHBORHOOD-AVERAGED metrics
       over all (alpha', k') such that:
           |alpha' - alpha| <= alpha_neighborhood
           |k'     - k|     <= k_neighborhood
    4) Applies the v2 hierarchical ranking:
           Primary selection (neighborhood metrics):
               - filter: avg_n_trades >= min_trades
               - rank:   maximize avg_mean_trade_ret
           Fallback A (raw metrics, same filter):
               - filter: n_trades >= min_trades
               - rank:   maximize mean_trade_ret
           Fallback B (no filters):
               - rank:   maximize mean_trade_ret
       Logs selection_mode for every window.
    5) Returns:
           - best_params: dict with chosen params + metrics + selection_mode
           - df_results:  DataFrame with raw and averaged metrics for all (alpha, k)

    Parameters
    ----------
    full_returns : pd.Series
        Full daily returns for the entire backtest period.
    is_start, is_end : pd.Timestamp
        Start and end dates of the in-sample window.
    alpha_grid : Iterable[float]
        Candidate SES smoothing parameters.
    k_grid : Iterable[float]
        Candidate z-score thresholds.
    vol_window : int
        Rolling window length for volatility normalization (default 14).
    alpha_neighborhood : float
        Neighborhood radius in alpha for robustness averaging (e.g., ±0.05).
    k_neighborhood : float
        Neighborhood radius in k for robustness averaging (e.g., ±0.25).

    Returns
    -------
    best_params : dict
        Chosen (alpha*, k*) with all metrics and selection_mode logged.
    df_results : pd.DataFrame
        Raw and averaged metrics for all (alpha, k) combinations evaluated.
    """

    # --------------------------------------------------
    # Scaled minimum trade count for this IS window
    # min_trades = max(5, round(0.15 * is_length))
    # --------------------------------------------------
    is_length = len(full_returns.loc[is_start:is_end])
    min_trades = max(5, round(0.15 * is_length))

    all_results = []

    # --------------------------------------------------
    # 1-2) Build raw grid results for this window
    # --------------------------------------------------
    for alpha in alpha_grid:
        # Global SES + z for this alpha
        ses_df_full, z_full = precompute_ses_and_z(
            full_returns=full_returns,
            alpha=alpha,
            vol_window=vol_window,
        )

        for k in k_grid:
            # In-sample performance for (alpha, k)
            perf = evaluate_params_in_sample_from_z(
                full_returns=full_returns,
                z_full=z_full,
                is_start=is_start,
                is_end=is_end,
                k=k,
            )

            # Attach parameter + window info
            perf["alpha"] = alpha
            perf["k"] = k
            perf["vol_window"] = vol_window
            perf["is_start"] = is_start
            perf["is_end"] = is_end

            all_results.append(perf)

    df_results = pd.DataFrame(all_results)

    # Edge case: if no rows at all, return None params
    if df_results.empty:
        best_params = {
            "alpha": None,
            "k": None,
            "vol_window": vol_window,
            "is_start": is_start,
            "is_end": is_end,
            "selection_mode": "no_results",
        }
        return best_params, df_results

    # --------------------------------------------------
    # 3) Compute neighborhood-averaged metrics
    # --------------------------------------------------
    # Initialize averaged metric columns
    df_results["avg_n_trades"] = np.nan
    df_results["avg_mean_trade_ret"] = np.nan
    df_results["avg_max_drawdown"] = np.nan
    df_results["avg_win_rate"] = np.nan                 # diagnostic only
    df_results["avg_loss_asymmetry_ratio"] = np.nan     # diagnostic only

    # Pre-extract arrays for speed
    alphas = df_results["alpha"].values
    ks = df_results["k"].values

    for i in range(len(df_results)):
        alpha_i = alphas[i]
        k_i = ks[i]

        # Neighborhood: |alpha' - alpha_i| <= alpha_neighborhood, same for k
        mask = (
            (np.abs(alphas - alpha_i) <= alpha_neighborhood) &
            (np.abs(ks - k_i) <= k_neighborhood)
        )
        neigh = df_results.loc[mask]

        # Compute averages over neighbors (skip NaNs)
        df_results.loc[i, "avg_n_trades"] = neigh["n_trades"].mean(skipna=True)
        df_results.loc[i, "avg_mean_trade_ret"] = neigh["mean_trade_ret"].mean(skipna=True)
        df_results.loc[i, "avg_max_drawdown"] = neigh["max_drawdown"].mean(skipna=True)
        df_results.loc[i, "avg_win_rate"] = neigh["win_rate"].mean(skipna=True)
        df_results.loc[i, "avg_loss_asymmetry_ratio"] = neigh["loss_asymmetry_ratio"].mean(skipna=True)

    # --------------------------------------------------
    # 4) V2 Hierarchical ranking
    # --------------------------------------------------

    # --- Primary selection: neighborhood metrics ---
    # Filter: avg_n_trades >= min_trades
    # Rank:   maximize avg_mean_trade_ret
    df_primary = df_results[
        df_results["avg_n_trades"] >= min_trades
    ].copy()
    df_primary = df_primary[df_primary["avg_mean_trade_ret"].notna()]

    if not df_primary.empty:
        best_row = df_primary.loc[df_primary["avg_mean_trade_ret"].idxmax()]
        selection_mode = "neighborhood"

    else:
        # --- Fallback A: raw metrics, same filter ---
        # Filter: n_trades >= min_trades
        # Rank:   maximize mean_trade_ret
        df_fallback_a = df_results[
            df_results["n_trades"] >= min_trades
        ].copy()
        df_fallback_a = df_fallback_a[df_fallback_a["mean_trade_ret"].notna()]

        if not df_fallback_a.empty:
            best_row = df_fallback_a.loc[df_fallback_a["mean_trade_ret"].idxmax()]
            selection_mode = "raw"

        else:
            # --- Fallback B: no filters ---
            # Rank: maximize mean_trade_ret
            df_fallback_b = df_results[df_results["mean_trade_ret"].notna()].copy()

            if not df_fallback_b.empty:
                best_row = df_fallback_b.loc[df_fallback_b["mean_trade_ret"].idxmax()]
                selection_mode = "deep_fallback"

            else:
                # Edge case: all mean_trade_ret are NaN (no trades at all)
                best_params = {
                    "alpha": None,
                    "k": None,
                    "vol_window": vol_window,
                    "is_start": is_start,
                    "is_end": is_end,
                    "selection_mode": "no_valid_trades",
                }
                return best_params, df_results

    # --------------------------------------------------
    # 5) Build best_params dict and return
    # --------------------------------------------------
    best_params = best_row.to_dict()

    # Ensure key fields explicitly set
    best_params["alpha"] = best_row["alpha"]
    best_params["k"] = best_row["k"]
    best_params["vol_window"] = vol_window
    best_params["is_start"] = is_start
    best_params["is_end"] = is_end
    best_params["selection_mode"] = selection_mode

    return best_params, df_results

def validate_oos_window(
    full_returns: pd.Series,
    alpha: float,
    k: float,
    oos_start: pd.Timestamp,
    oos_end: pd.Timestamp,
    vol_window: int = 14,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Evaluates OUT-OF-SAMPLE performance for ONE window, given
    fixed hyperparameters (alpha, k) selected from in-sample calibration.

    Parameters
    ----------
    full_returns : pd.Series
        Full daily returns for the entire backtest period.
    alpha : float
        Chosen SES smoothing parameter (alpha*).
    k : float
        Chosen z-score threshold (k*).
    oos_start, oos_end : pd.Timestamp
        Start and end dates of the OOS window.
    vol_window : int
        Rolling window length for volatility normalization (default 14).

    Returns
    -------
    df_bt_oos : pd.DataFrame
        Detailed OOS backtest with columns:
            - 'ret'          : r(t), underlying daily returns
            - 'z'            : z(t), z-score signal
            - 'position'     : position_t in {-1, 0, +1}
            - 'next_ret'     : r(t+1), next-day returns
            - 'strategy_ret' : position_t * r(t+1)
    perf_oos : dict
        Trade-level performance statistics over the OOS window, including:
            - 'alpha', 'k', 'vol_window'
            - 'oos_start', 'oos_end'
            - n_trades, win_rate, mean_trade_ret, avg_win, avg_loss,
              loss_asymmetry_ratio, max_drawdown
    """

    # 1) Precompute SES + z over FULL history for this alpha
    ses_df_full, z_full = precompute_ses_and_z(
        full_returns=full_returns,
        alpha=alpha,
        vol_window=vol_window,
    )

    # 2) Slice OOS window
    oos_returns = full_returns.loc[oos_start:oos_end]
    oos_z = z_full.loc[oos_start:oos_end]

    # 3) Run OOS backtest
    df_bt_oos, perf_oos = backtest_from_z(
        returns=oos_returns,
        z=oos_z,
        k=k,
    )

    # 4) Attach metadata for logging
    perf_oos["alpha"] = alpha
    perf_oos["k"] = k
    perf_oos["vol_window"] = vol_window
    perf_oos["oos_start"] = oos_start
    perf_oos["oos_end"] = oos_end

    return df_bt_oos, perf_oos

def run_wfo_for_ticker(
    full_returns: pd.Series,
    is_length: int,
    oos_length: int,
    alpha_grid: Iterable[float],
    k_grid: Iterable[float],
    vol_window: int = 14,
    step_length: int | None = None,
    alpha_neighborhood: float = 0.05,
    k_neighborhood: float = 0.25,
) -> Dict[str, Any]:
    """
    Runs walk-forward optimization (WFO) for a single ticker.

    For each rolling window:
        1) Use the in-sample segment [is_start, is_end] to calibrate (alpha, k)
           using the v2 hierarchical ranking criterion.
        2) Use the out-of-sample segment [oos_start, oos_end] to validate the
           chosen (alpha*, k*).

    Parameters
    ----------
    full_returns : pd.Series
        Full daily returns r(t) for the ticker, indexed by date.
    is_length : int
        Number of observations in each in-sample (IS) window.
    oos_length : int
        Number of observations in each out-of-sample (OOS) window.
    alpha_grid : iterable of float
        Candidate SES smoothing parameters.
    k_grid : iterable of float
        Candidate z-score thresholds.
    vol_window : int
        Rolling window length for volatility normalization (default 14).
    step_length : int, optional
        Step between consecutive windows (in observations).
        If None, defaults to oos_length (non-overlapping OOS windows).
    alpha_neighborhood : float
        Neighborhood radius in alpha for robustness averaging (e.g., ±0.05).
    k_neighborhood : float
        Neighborhood radius in k for robustness averaging (e.g., ±0.25).

    Returns
    -------
    results : dict with keys
        - 'windows'        : list of (is_start, is_end, oos_start, oos_end)
        - 'is_results'     : pd.DataFrame with ALL (alpha, k) grid results across windows
        - 'best_params'    : pd.DataFrame with chosen (alpha*, k*) per window
        - 'oos_results'    : pd.DataFrame with OOS trade-level performance per window
        - 'oos_strategy_ret' : pd.Series of concatenated OOS strategy returns
        - 'oos_detailed'   : pd.DataFrame with full daily OOS backtest rows
    """

    full_returns = full_returns.sort_index()

    # 1) Build rolling IS/OOS windows
    windows = make_windows(
        index=full_returns.index,
        is_length=is_length,
        oos_length=oos_length,
        step_length=step_length,
    )

    if not windows:
        return {
            "windows": [],
            "is_results": pd.DataFrame(),
            "best_params": pd.DataFrame(),
            "oos_results": pd.DataFrame(),
            "oos_strategy_ret": pd.Series(dtype=float),
        }

    all_is_results: List[pd.DataFrame] = []
    best_params_list: List[Dict[str, Any]] = []
    oos_perf_list: List[Dict[str, Any]] = []
    oos_bt_list: List[pd.DataFrame] = []

    # 2) Loop over each window
    for window_id, (is_start, is_end, oos_start, oos_end) in enumerate(windows, start=1):

        # 2a) In-sample calibration for this window
        best_params, df_is = calibrate_is_window(
            full_returns=full_returns,
            is_start=is_start,
            is_end=is_end,
            alpha_grid=alpha_grid,
            k_grid=k_grid,
            vol_window=vol_window,
            alpha_neighborhood=alpha_neighborhood,
            k_neighborhood=k_neighborhood,
        )

        # Tag with window_id
        df_is["window_id"] = window_id
        best_params["window_id"] = window_id

        all_is_results.append(df_is)
        best_params_list.append(best_params)

        alpha_star = best_params["alpha"]
        k_star = best_params["k"]

        # If alpha_star or k_star is None (no valid calibration), skip OOS window
        # This is intentional: we do not trade when IS calibration finds no valid setup
        if alpha_star is None or k_star is None:
            continue

        # 2b) Out-of-sample validation for this window
        df_bt_oos, perf_oos = validate_oos_window(
            full_returns=full_returns,
            alpha=alpha_star,
            k=k_star,
            oos_start=oos_start,
            oos_end=oos_end,
            vol_window=vol_window,
        )

        perf_oos["window_id"] = window_id
        perf_oos["selection_mode"] = best_params["selection_mode"]  # carry forward from IS
        oos_perf_list.append(perf_oos)

        # Tag the detailed OOS backtest with window_id as well (optional but helpful)
        df_bt_oos = df_bt_oos.copy()
        df_bt_oos["window_id"] = window_id
        df_bt_oos["selection_mode"] = best_params["selection_mode"]  # carry forward from IS
        oos_bt_list.append(df_bt_oos)

    # 3) Combine all IS results across windows
    if all_is_results:
        is_results = pd.concat(all_is_results, ignore_index=True)
    else:
        is_results = pd.DataFrame()

    # 4) Combine best params per window
    if best_params_list:
        best_params_df = pd.DataFrame(best_params_list)
    else:
        best_params_df = pd.DataFrame()

    # 5) Combine OOS performance per window
    if oos_perf_list:
        oos_results = pd.DataFrame(oos_perf_list)
    else:
        oos_results = pd.DataFrame()

    # 6) Build concatenated OOS strategy return series
    if oos_bt_list:
        # Concatenate all detailed OOS backtests, keep chronological order
        df_oos_all = pd.concat(oos_bt_list, axis=0)
        df_oos_all = df_oos_all.sort_index()

        # Extract the strategy returns
        oos_strategy_ret = df_oos_all["strategy_ret"].dropna()
    else:
        df_oos_all = pd.DataFrame()
        oos_strategy_ret = pd.Series(dtype=float)

    results = {
        "windows": windows,
        "is_results": is_results,
        "best_params": best_params_df,
        "oos_results": oos_results,
        "oos_strategy_ret": oos_strategy_ret,
        # Full detailed OOS backtest for equity curves and analysis
        "oos_detailed": df_oos_all,
    }

    return results