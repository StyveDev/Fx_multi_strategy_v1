"""
Technical indicators used by the mean-reversion and trend-following
strategies, plus the shared risk-sizing inputs (ATR, relative volume).
"""

import numpy as np
import pandas as pd

from logging_utils.logger import get_logger

log = get_logger(__name__)


def compute_daily_volatility(prices: pd.Series) -> pd.Series:
    return prices.pct_change()


def compute_sma(prices: pd.Series, window: int) -> pd.Series:
    return prices.rolling(window).mean()


def compute_ema(prices: pd.Series, span: int) -> pd.Series:
    return prices.ewm(span=span, adjust=False).mean()


def compute_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def compute_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Average True Range — intrabar volatility from High/Low/Close, with a
    close-to-close fallback if High/Low aren't available."""
    if not {"high", "low"}.issubset(df.columns):
        log.warning("compute_atr: no High/Low — falling back to close-to-close range as a proxy.")
        return df["close"].diff().abs().rolling(window).mean()

    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(window).mean()


def compute_adx(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    Wilder's ADX — measures trend STRENGTH (not direction). Used to gate the
    trend-following strategy: only trade the crossover signal when ADX says
    a real trend is actually present, not during a flat/ranging market.
    Falls back to a rolling-slope-based proxy if High/Low aren't available
    (ADX proper needs directional movement, which requires intrabar range).
    """
    if not {"high", "low"}.issubset(df.columns):
        log.warning("compute_adx: no High/Low — falling back to a slope-based trend-strength proxy.")
        slope = df["close"].diff(window).abs() / df["close"].shift(window)
        return (slope * 100).rolling(window).mean()

    high, low, close = df["high"], df["low"], df["close"]
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / window, min_periods=window, adjust=False).mean() / atr.replace(0, np.nan)
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / window, min_periods=window, adjust=False).mean() / atr.replace(0, np.nan)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    return adx  # genuine NaN during warmup — callers must handle explicitly, not silently treat as 0


def compute_zscore(prices: pd.Series, window: int) -> pd.Series:
    """Rolling z-score — how many std-devs price is from its own rolling mean. Core mean-reversion signal."""
    mean = prices.rolling(window).mean()
    std = prices.rolling(window).std()
    return (prices - mean) / std.replace(0, np.nan)


def compute_relative_volume(volume: pd.Series, window: int = 20) -> pd.Series:
    avg_vol = volume.rolling(window).mean()
    return volume / avg_vol.replace(0, np.nan)


def compute_all_indicators(
    df: pd.DataFrame,
    zscore_window: int = 20,
    ema_fast: int = 12,
    ema_slow: int = 26,
    adx_window: int = 14,
    atr_window: int = 14,
    rsi_window: int = 14,
    volume_window: int = 20,
) -> pd.DataFrame:
    """
    df must have ['date', 'close'] (aliased 'price'); OHLCV extras used where present.
    """
    out = df.copy()
    out["volatility"] = compute_daily_volatility(out["price"])
    out["zscore"] = compute_zscore(out["price"], zscore_window)
    out["ema_fast"] = compute_ema(out["price"], ema_fast)
    out["ema_slow"] = compute_ema(out["price"], ema_slow)
    out["rsi"] = compute_rsi(out["price"], rsi_window)
    out["adx"] = compute_adx(out, adx_window)
    out["atr"] = compute_atr(out, atr_window)
    out["atr_pct"] = out["atr"] / out["price"] * 100

    if "volume" in out.columns:
        out["relative_volume"] = compute_relative_volume(out["volume"], volume_window)
    else:
        out["relative_volume"] = np.nan

    log.info(
        f"Computed indicators over {len(out)} rows "
        f"(zscore_window={zscore_window}, ema={ema_fast}/{ema_slow}, adx_window={adx_window}, "
        f"has_volume={'volume' in out.columns})"
    )
    return out
