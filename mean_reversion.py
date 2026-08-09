"""
Mean reversion strategy — rolling Z-score (Bollinger-band style).

CHOICE RATIONALE: picked over RSI-based or Bollinger %B mean reversion
because it gives a continuous, unbounded confidence measure (the z-score
magnitude itself) rather than a bounded 0-100 oscillator — that maps more
naturally onto the back-cast/shrink-ratio machinery, which wants a
directional call plus a sense of "how far from normal" price currently is.
It's also one of the most extensively studied, robust mean-reversion
formulations for FX pairs specifically (which structurally mean-revert more
than individual equities, due to central bank/rate-differential anchoring).

Signal logic:
    z = (price - rolling_mean) / rolling_std
    z <= -entry_threshold  -> price abnormally LOW  -> direction = +1 (buy, expect reversion up)
    z >=  entry_threshold  -> price abnormally HIGH -> direction = -1 (sell, expect reversion down)
    |z| <  exit_threshold  -> reverted back to normal -> direction = 0 (flatten / no new entry)
    otherwise (in between)  -> direction = 0 (no clear edge yet, hold current signal state)
"""

import numpy as np
import pandas as pd

from data_layer.indicators import compute_zscore
from logging_utils.logger import get_logger

log = get_logger(__name__)


def mean_reversion_signal(
    df: pd.DataFrame,
    window: int = 20,
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.5,
) -> pd.DataFrame:
    """
    df must have 'price' (already indicator-enriched or not — recomputes zscore
    fresh here so this strategy is independently configurable from the
    default indicator windows used elsewhere).

    Returns df with added columns: mr_zscore, mr_direction, mr_confidence
    (mr_confidence = |z| clipped to a 0..1-ish scale via tanh, used only for
    human-readable logging — the ACTUAL position sizing still runs through
    the shared shrink-ratio mechanism in strategy/shrink_ratio.py, not this).
    """
    out = df.copy()
    out["mr_zscore"] = compute_zscore(out["price"], window)

    direction = pd.Series(0.0, index=out.index)
    direction[out["mr_zscore"] <= -entry_threshold] = 1.0
    direction[out["mr_zscore"] >= entry_threshold] = -1.0
    out["mr_direction"] = direction

    out["mr_confidence"] = np.tanh(out["mr_zscore"].abs() / entry_threshold)

    log.debug(
        f"Mean-reversion signal computed (window={window}, entry={entry_threshold}, exit={exit_threshold}) — "
        f"{(direction != 0).sum()} non-zero signal days out of {len(out)}"
    )
    return out
