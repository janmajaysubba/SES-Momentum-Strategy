from __future__ import annotations

import os
import json
from typing import Dict, Iterable, Tuple, Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# 1. DATA LOADING
# ---------------------------------------------------------------------

def load_ticker_json(path: str) -> pd.DataFrame:
    """
    Loads a single ticker's JSON file into a clean price DataFrame.

    Expected JSON structure (list of dicts), e.g.:
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

    Returns
    -------
    df : pd.DataFrame
        Index: DatetimeIndex ("date")
        Columns: ["open", "high", "low", "close", "volume"]
        Sorted by date ascending.
    """
    with open(path, "r") as f:
        raw = json.load(f)

    if not raw:
        raise ValueError(f"No data found in JSON file: {path}")

    df = pd.DataFrame(raw)

    # Normalize column names (lowercase)
    df.columns = [c.lower() for c in df.columns]

    # Convert date column
    if "utcdate" not in df.columns:
        raise KeyError(f"'UTCDate' (or 'utcdate') column not found in {path}")

    df["date"] = pd.to_datetime(df["utcdate"])
    df = df.set_index("date").sort_index()

    # Keep only standard OHLCV columns that we care about
    keep_cols = ["open", "high", "low", "close", "volume"]
    missing = [c for c in keep_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing expected columns {missing} in {path}")

    df = df[keep_cols].astype(float)

    return df


def load_all_tickers(data_dir: str, tickers: Iterable[str]) -> Dict[str, pd.DataFrame]:
    """
    Loads multiple tickers from JSON files into a dict of DataFrames.

    Parameters
    ----------
    data_dir : str
        Directory containing JSON files, e.g. "data".
    tickers : Iterable[str]
        List of symbols, e.g. ["SPY", "QQQ", "NVDA"].

    Returns
    -------
    prices : dict
        {symbol: price_df}, where price_df is the same format as load_ticker_json.
    """
    prices: Dict[str, pd.DataFrame] = {}

    for symbol in tickers:
        fname = f"{symbol}.json"
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found for {symbol}: {path}")

        df = load_ticker_json(path)
        prices[symbol] = df

    return prices


# ---------------------------------------------------------------------
# 2. FEATURE CONSTRUCTION: RETURNS, SES LEVEL, Z-SCORE
# ---------------------------------------------------------------------

def compute_returns(price_df: pd.DataFrame, price_col: str = "close") -> pd.DataFrame:
    """
    Computes daily simple returns from a price DataFrame.

    Parameters
    ----------
    price_df : pd.DataFrame
        Must contain a column given by price_col (default "close").
    price_col : str
        Which column to use as the price series.

    Returns
    -------
    df_ret : pd.DataFrame
        Index: same DatetimeIndex as input, but the first row is dropped.
        Columns:
            - price : original price (aligned with returns)
            - ret   : daily simple return (scale-free)
    """
    if price_col not in price_df.columns:
        raise KeyError(f"Column '{price_col}' not found in price_df")

    price = price_df[price_col].astype(float)

    # Simple returns
    ret = (price / price.shift(1)) - 1

    df_ret = pd.DataFrame({
        "price": price,
        "ret": ret,
    }).dropna()

    return df_ret


def ses_on_returns(returns: pd.Series, alpha: float) -> pd.DataFrame:
    """
    Applies Simple Exponential Smoothing (SES) to daily returns.

    SES recursion:
        error_t = r_t - level_prev
        level_curr = level_prev + alpha * error_t

    Parameters
    ----------
    returns : pd.Series
        Daily simple returns r_t.
    alpha : float
        Smoothing parameter, 0 < alpha <= 1.

    Returns
    -------
    df_ses : pd.DataFrame with:
        - ret         : original returns
        - level_prev  : previous day's SES estimate
        - level_curr  : updated SES estimate after seeing r_t
        - error       : r_t - level_prev
    """
    if not (0 < alpha <= 1):
        raise ValueError("alpha must be in (0, 1]")

    r = returns.dropna().astype(float)
    n = len(r)

    level_prev_arr = np.zeros(n)
    level_curr_arr = np.zeros(n)
    error_arr = np.zeros(n)

    # Initialize: set first level equal to the first return
    level_prev = r.iloc[0]
    level_curr = level_prev

    level_prev_arr[0] = np.nan
    level_curr_arr[0] = level_curr
    error_arr[0] = np.nan

    # SES recursion
    for i in range(1, n):
        error = r.iloc[i] - level_prev
        level_curr = level_prev + alpha * error

        level_prev_arr[i] = level_prev
        level_curr_arr[i] = level_curr
        error_arr[i] = error

        # move forward
        level_prev = level_curr

    df_ses = pd.DataFrame({
        "ret": r.values,
        "level_prev": level_prev_arr,
        "level_curr": level_curr_arr,
        "error": error_arr,
    }, index=r.index)

    return df_ses


def compute_z_score(level_curr: pd.Series, returns: pd.Series, vol_window: int) -> pd.Series:
    """
    Converts the SES level (forecast of expected return) into a z-score.
    
    Formula for the volatility-normalized z-score of the SES signal:

        z(t) = l(t) / sigma(t)

    where:
        - l(t)     = SES smoothed return estimate at time t
        - sigma(t) = rolling standard deviation of returns over the past
                     `vol_window` days, including the current day's return.

    Parameters
    ----------
    level_curr : pd.Series
        SES updated level l(t) for each date.
    returns : pd.Series
        Daily simple returns r(t), same index as level_curr.
    vol_window : int
        Rolling window length used to compute volatility.

    Returns
    -------
    z : pd.Series
        The z-score series z(t), indexed the same as the inputs.
    """
    r = returns.astype(float).copy()
    lc = level_curr.astype(float).copy()

    # Rolling volatility including current-day return
    sigma = r.rolling(vol_window).std(ddof=0)

    z = lc / sigma
    z = z.replace([np.inf, -np.inf], np.nan)
    z.name = "z"
    return z


# ---------------------------------------------------------------------
# 3. STRATEGY RULE: POSITIONS FROM Z
# ---------------------------------------------------------------------

def generate_positions_from_z(z: pd.Series, k: float) -> pd.Series:
    """
    Map a z-score signal into trading positions.

    Trading rule:
        - Go long  (+1) if z(t) >  k
        - Go short (-1) if z(t) < -k
        - Stay flat (0) otherwise

    Parameters
    ----------
    z : pd.Series
        Z-score series z(t), indexed by date.
    k : float
        Threshold parameter. Typical values might be 0.5, 1.0, 1.5, etc.

    Returns
    -------
    pos : pd.Series
        Position series with values in {-1, 0, +1}, same index as z.
    """
    z_clean = z.astype(float).copy()

    pos = pd.Series(0.0, index=z_clean.index)
    pos[z_clean > k] = 1.0
    pos[z_clean < -k] = -1.0
    pos.name = "position"

    return pos


# ---------------------------------------------------------------------
# 4. TRADE-LEVEL STATISTICS
# ---------------------------------------------------------------------

def compute_trade_stats(df_bt: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes trade-level performance metrics from a backtest DataFrame.
    Only considers active trades: rows where position != 0
    and strategy_ret is not NaN.

    Parameters
    ----------
    df_bt : pd.DataFrame
        Output of backtest_from_z(), must contain columns:
            - 'position'     : position_t in {-1, 0, +1}
            - 'strategy_ret' : daily strategy returns

    Returns
    -------
    stats : dict
        - n_trades             : number of active trades
        - win_rate             : fraction of active trades where strategy_ret > 0
        - mean_trade_ret       : mean strategy_ret on active trades
        - avg_win              : mean strategy_ret on winning trades
        - avg_loss             : mean strategy_ret on losing trades
        - loss_asymmetry_ratio : abs(avg_loss) / avg_win
        - max_drawdown         : computed on all days including flat
    """
    # Filter to active trades only
    active = df_bt[
        (df_bt["position"] != 0) & (df_bt["strategy_ret"].notna())
    ]["strategy_ret"]

    n_trades = len(active)

    if n_trades == 0:
        return {
            "n_trades": 0,
            "win_rate": np.nan,
            "mean_trade_ret": np.nan,
            "avg_win": np.nan,
            "avg_loss": np.nan,
            "loss_asymmetry_ratio": np.nan,
            "max_drawdown": np.nan,
        }

    win_rate = (active > 0).mean()
    mean_trade_ret = active.mean()

    wins = active[active > 0]
    losses = active[active < 0]

    avg_win = wins.mean() if len(wins) > 0 else np.nan
    avg_loss = losses.mean() if len(losses) > 0 else np.nan

    # Loss asymmetry ratio edge cases
    if len(wins) == 0:
        loss_asymmetry_ratio = np.inf
    elif len(losses) == 0:
        loss_asymmetry_ratio = 0.0
    else:
        loss_asymmetry_ratio = abs(avg_loss) / avg_win

    # Max drawdown computed on all days including flat
    all_rets = df_bt["strategy_ret"].dropna().astype(float)
    if len(all_rets) > 0:
        equity = (1.0 + all_rets).cumprod()
        peak = equity.cummax()
        drawdown = equity / peak - 1.0
        max_drawdown = -drawdown.min()
    else:
        max_drawdown = np.nan

    return {
        "n_trades": n_trades,
        "win_rate": win_rate,
        "mean_trade_ret": mean_trade_ret,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "loss_asymmetry_ratio": loss_asymmetry_ratio,
        "max_drawdown": max_drawdown,
    }


# ---------------------------------------------------------------------
# 5. BACKTEST FOR A SINGLE SIGNAL
# ---------------------------------------------------------------------

def backtest_from_z(
    returns: pd.Series,
    z: pd.Series,
    k: float,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Backtests a simple daily strategy driven by a z-score signal and threshold k.

    Trading logic (timing convention)
    ---------------------------------
    - We assume z(t) is computed at the end of day t (using information up to t).
    - A position based on z(t) is entered at the close of day t.
    - P/L is realized using the next day's return r(t+1).

    So, for each date t:

        position_t = f(z(t), k)
        strategy_ret_t = position_t * r(t+1)

    Implementation Steps
    --------------------
    1) Aligns z with returns.
    2) Generates raw positions from z using generate_positions_from_z(z, k).
    3) Computes next-day returns: r_next = returns.shift(-1).
    4) Computes daily strategy returns:
           strategy_ret_t = position_t * r_next_t
    5) Builds a detailed backtest DataFrame.
    6) Computes trade-level statistics via compute_trade_stats().

    Parameters
    ----------
    returns : pd.Series
        Daily simple returns r(t) of the underlying, indexed by date.
    z : pd.Series
        Z-score signal z(t), indexed by date (will be aligned to returns).
    k : float
        Threshold used to generate positions.

    Returns
    -------
    df_bt : pd.DataFrame
        Detailed backtest results with columns:
            - 'ret'          : r(t), underlying daily returns
            - 'z'            : z(t), z-score signal
            - 'position'     : position_t in {-1, 0, +1}
            - 'next_ret'     : r(t+1), next-day returns
            - 'strategy_ret' : position_t * r(t+1)
    perf : dict
        Trade-level statistics from compute_trade_stats(), plus:
            - 'k' : threshold used
    """
    # 1) Clean and align inputs
    r = returns.astype(float).sort_index()
    z_aligned = z.astype(float).reindex(r.index)

    # 2) Generate positions from z(t)
    raw_pos = generate_positions_from_z(z_aligned, k)  # position_t

    # 3) Next-day returns r(t+1)
    r_next = r.shift(-1)

    # 4) Strategy daily returns: position_t * r(t+1)
    strategy_ret = raw_pos * r_next
    strategy_ret.name = "strategy_ret"

    # 5) Build detailed backtest DataFrame
    df_bt = pd.DataFrame({
        "ret": r,               # r(t)
        "z": z_aligned,         # z(t)
        "position": raw_pos,    # position_t
        "next_ret": r_next,     # r(t+1)
        "strategy_ret": strategy_ret,
    })

    # 6) Compute trade-level statistics
    perf = compute_trade_stats(df_bt)
    perf["k"] = k

    return df_bt, perf