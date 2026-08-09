"""
Trend-following strategy — dual EMA crossover, gated by ADX.

CHOICE RATIONALE: picked the classic fast/slow EMA crossover over
Donchian-channel breakout or MACD-based trend entries because it pairs
cleanly with an ADX trend-strength filter (both are standard, well-studied
building blocks) and gives a simple, always-defined directional call (fast
above/below slow) rather than requiring a breakout event to occur — useful
here since the regime detector (not this module) is what decides how much
weight to give a trend call vs. a mean-reversion call, so this strategy
should just answer "which way is the trend pointing right now" cleanly.

ADX filter matters because a naive EMA crossover fires constantly in a flat/
choppy market (whipsaws) — ADX > threshold requires a real trend to actually
be present before the crossover direction is trusted at all.

Signal logic:
    fast_ema > slow_ema AND adx > threshold  -> direction = +1 (uptrend)
    fast_ema < slow_ema AND adx > threshold  -> direction = -1 (downtrend)
    adx <= threshold (no real trend)         -> direction = 0 (flat market, sit out)
"""

import numpy as np
import pandas as pd

from data_layer.indicators import compute_ema, compute_adx
from logging_utils.logger import get_logger

log = get_logger(__name__)


def trend_following_signal(
    df: pd.DataFrame,
    ema_fast: int = 12,
    ema_slow: int = 26,
    adx_window: int = 14,
    adx_threshold: float = 25.0,
) -> pd.DataFrame:
    """
    Returns df with added columns: tf_ema_fast, tf_ema_slow, tf_adx,
    tf_direction, tf_confidence (ADX normalized to a 0..1-ish scale via
    tanh, for logging only — actual sizing still runs through shrink_ratio.py).
    """
    out = df.copy()
    out["tf_ema_fast"] = compute_ema(out["price"], ema_fast)
    out["tf_ema_slow"] = compute_ema(out["price"], ema_slow)
    out["tf_adx"] = compute_adx(out, adx_window)

    crossover_direction = np.sign(out["tf_ema_fast"] - out["tf_ema_slow"])
    trend_present = out["tf_adx"] > adx_threshold

    out["tf_direction"] = np.where(trend_present, crossover_direction, 0.0)
    out["tf_confidence"] = np.tanh(out["tf_adx"] / (adx_threshold * 2))

    log.debug(
        f"Trend-following signal computed (ema={ema_fast}/{ema_slow}, adx_window={adx_window}, "
        f"adx_threshold={adx_threshold}) — {(out['tf_direction'] != 0).sum()} trending days out of {len(out)}"
    )
    return out
