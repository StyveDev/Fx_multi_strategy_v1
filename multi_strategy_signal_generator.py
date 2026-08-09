"""
Multi-strategy signal generator — the new equivalent of the old
signal_generator.py, but combining TWO strategies via a regime detector
instead of one forecasting model + RSI.

DESIGN (explicit, so it's clear what's mechanical vs. a judgment call):
  1. Compute mean-reversion signal and trend-following signal independently.
  2. Fit/apply the HMM regime detector -> each day is "trending" or "ranging".
  3. PRIMARY strategy = trend-following on "trending" days, mean-reversion on
     "ranging" days. SECONDARY = whichever wasn't picked. This is the
     regime detector's whole job: decide which strategy's call to trust more
     right now.
  4. Final trade direction = PRIMARY strategy's direction for that day.
  5. r1 = rolling back-cast accuracy of the PRIMARY strategy; r2 = same for
     SECONDARY. "Consistent" = both strategies agree on direction (secondary
     giving no signal at all is treated as consistent — no conflict exists,
     same convention as the original system).
  6. shrink_ratio = shrink_ratio_from_r1r2(r1, r2, consistent) ** 3 — UNCHANGED
     equations, per requirement.
"""

import numpy as np
import pandas as pd

from data_layer.indicators import compute_all_indicators
from strategy.mean_reversion import mean_reversion_signal
from strategy.trend_following import trend_following_signal
from strategy.regime_detector import RegimeDetector
from strategy.backcast import strategy_direction_accuracy
from strategy.shrink_ratio import compute_shrink_ratio
from logging_utils.logger import get_logger

log = get_logger(__name__)


def generate_multi_strategy_signals(
    price_df: pd.DataFrame,
    mr_window: int = 20,
    mr_entry: float = 2.0,
    mr_exit: float = 0.5,
    tf_ema_fast: int = 12,
    tf_ema_slow: int = 26,
    tf_adx_window: int = 14,
    tf_adx_threshold: float = 25.0,
    backcast_window: int = 30,
    hmm_n_states: int = 2,
    regime_detector: RegimeDetector = None,
) -> pd.DataFrame:
    """
    price_df: ['date', 'price'] at minimum, OHLCV extras used where available.

    regime_detector: pass an ALREADY-FITTED RegimeDetector to reuse it (e.g.
    fit once on training data during optimization/live trading, then reuse
    for prediction on new data without refitting). Pass None to fit a fresh
    one on this data (used for plain backtesting over one continuous history).

    Returns price_df enriched with both strategies' raw signals, regime,
    r1/r2, shrink_ratio, final direction, and a `reason` string per row.
    """
    df = compute_all_indicators(price_df, zscore_window=mr_window, ema_fast=tf_ema_fast,
                                 ema_slow=tf_ema_slow, adx_window=tf_adx_window)

    df = mean_reversion_signal(df, window=mr_window, entry_threshold=mr_entry, exit_threshold=mr_exit)
    df = trend_following_signal(df, ema_fast=tf_ema_fast, ema_slow=tf_ema_slow,
                                 adx_window=tf_adx_window, adx_threshold=tf_adx_threshold)

    if regime_detector is None:
        regime_detector = RegimeDetector(n_states=hmm_n_states)
        df = regime_detector.fit_predict(df)
    else:
        df = regime_detector.predict(df)

    is_trending = df["regime"] == "trending"
    is_unknown = df["regime"] == "unknown"

    primary_direction = np.where(is_trending, df["tf_direction"], df["mr_direction"])
    secondary_direction = np.where(is_trending, df["mr_direction"], df["tf_direction"])
    primary_direction = np.where(is_unknown, 0.0, primary_direction)  # no reliable regime read -> no trade
    primary_label = np.where(is_trending, "trend-following", "mean-reversion")
    secondary_label = np.where(is_trending, "mean-reversion", "trend-following")

    df["primary_direction"] = primary_direction
    df["secondary_direction"] = secondary_direction
    df["primary_strategy"] = primary_label
    df["secondary_strategy"] = secondary_label

    r1_tf = strategy_direction_accuracy(df["price"], df["tf_direction"], backcast_window)
    r1_mr = strategy_direction_accuracy(df["price"], df["mr_direction"], backcast_window)
    df["r1"] = np.where(is_trending, r1_tf, r1_mr)
    df["r2"] = np.where(is_trending, r1_mr, r1_tf)

    consistent = (df["secondary_direction"] == 0) | (df["primary_direction"] == df["secondary_direction"])
    df["consistent"] = consistent

    r0 = compute_shrink_ratio(pd.Series(df["r1"], index=df.index), pd.Series(df["r2"], index=df.index), consistent)
    df["r0"] = r0
    df["shrink_ratio"] = r0 ** 3

    df["direction"] = np.where(df["shrink_ratio"] > 0, primary_direction, 0)

    def _reason(row):
        if row["primary_direction"] == 0:
            return f"{row['primary_strategy']} (primary, regime={row['regime']}) has no clear signal"
        d = "up" if row["primary_direction"] > 0 else "down"
        agree = "agrees with" if row["consistent"] else "conflicts with"
        return (
            f"regime={row['regime']}; {row['primary_strategy']} (primary) says {d}; "
            f"{row['secondary_strategy']} (secondary) {agree} primary; shrink={row['shrink_ratio']:.3f}"
        )

    df["reason"] = df.apply(_reason, axis=1)

    log.info(
        f"Multi-strategy signals generated over {len(df)} rows — "
        f"{(df['regime']=='trending').mean()*100:.1f}% trending days, "
        f"avg shrink ratio={df['shrink_ratio'].mean():.3f}, "
        f"non-zero direction days={(df['direction'] != 0).sum()}"
    )
    return df, regime_detector
