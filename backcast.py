"""
Back-casting accuracy — generalized from the original Holt-Winters/RSI
system to work for ANY strategy's directional signal, since this system now
runs mean-reversion and trend-following instead.

For each day, judge whether a strategy's signal for that day had a clear
direction, and whether that direction matched the actual next-period price
move:
    - no clear signal that day (direction == 0)  -> NA (excluded from the "1" count)
    - signal direction matched actual direction    -> 1
    - signal direction contradicted actual         -> 0

Rolling accuracy over the trailing `window` days = (# of 1s) / window —
same fixed-window convention as the original system, so the shrink-ratio
math downstream behaves identically regardless of which two strategies feed
it (this is the part the user asked to keep unchanged).
"""

import numpy as np
import pandas as pd

from logging_utils.logger import get_logger

log = get_logger(__name__)


def _rolling_accuracy(correct_flags: pd.Series, window: int) -> pd.Series:
    is_one = (correct_flags == 1).astype(float)
    return is_one.rolling(window, min_periods=1).sum() / window


def strategy_direction_accuracy(prices: pd.Series, direction: pd.Series, window: int) -> pd.Series:
    """
    correct[t] = did the signal decided at t-1 (using data available through
    t-1) correctly call the realized price move from t-1 to t?

    Both `prior_direction[t]` and `actual_move[t]` are fully known by end of
    day t — so the rolling accuracy through day t (used to size day t's OWN
    trade, which executes at price[t]) never depends on price[t+1]. This
    mirrors exactly how the original Holt-Winters system avoided look-ahead
    (it compared a forecast made at t-1 against the realized price at t).

    An earlier version of this function used prices.shift(-1) (comparing
    today's signal to TOMORROW's move) — that made today's shrink ratio
    depend on tomorrow's close, which is look-ahead bias. Fixed here.
    """
    prior_direction = direction.shift(1)
    actual_move = np.sign(prices - prices.shift(1))

    correct = pd.Series(np.nan, index=prices.index)
    has_signal = (prior_direction != 0) & (~prior_direction.isna()) & (~actual_move.isna())
    correct[has_signal] = (prior_direction[has_signal] == actual_move[has_signal]).astype(float)

    acc = _rolling_accuracy(correct, window)
    log.debug(f"Direction accuracy computed over {len(prices)} rows, mean={acc.mean():.3f}")
    return acc
