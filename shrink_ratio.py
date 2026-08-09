"""
Trading shrink ratio — KEPT UNCHANGED from the original Holt-Winters/RSI
system, per explicit requirement. Only the strategies feeding r1/r2 have
changed (now: regime-selected primary strategy vs. secondary/confirming
strategy, instead of Holt-Winters vs. RSI) — the confidence-scaling math
itself is identical.

When the PRIMARY strategy's direction is OPPOSITE of the SECONDARY strategy's
implied direction:
    r0 = (r1 - r1*r2) / (r1 + r2 - 2*r1*r2)                         ... (2)

When they're CONSISTENT (agree):
    r0 = (r1*r2) / (2*r1*r2 - r1 - r2 + 1)                          ... (3)

Then r0 is cubed to get the final shrink ratio applied to trade size, "to
increase the penalty of incorrect prediction" — same as before.
"""

import numpy as np
import pandas as pd

from logging_utils.logger import get_logger

log = get_logger(__name__)


def compute_shrink_ratio(r1: pd.Series, r2: pd.Series, consistent: pd.Series) -> pd.Series:
    """
    r1: rolling back-cast accuracy of the PRIMARY strategy (regime-selected leader), 0..1
    r2: rolling back-cast accuracy of the SECONDARY strategy (confirming), 0..1
    consistent: True where primary and secondary agree on direction.
    Returns r0 (uncubed) — cube it yourself where you need the final shrink ratio.
    """
    r1 = r1.clip(1e-6, 1 - 1e-6)
    r2 = r2.clip(1e-6, 1 - 1e-6)

    opposite_denom = r1 + r2 - 2 * r1 * r2
    r0_opposite = (r1 - r1 * r2) / opposite_denom.replace(0, np.nan)

    consistent_denom = 2 * r1 * r2 - r1 - r2 + 1
    r0_consistent = (r1 * r2) / consistent_denom.replace(0, np.nan)

    r0 = pd.Series(np.where(consistent, r0_consistent, r0_opposite), index=r1.index)
    r0 = r0.clip(0, 1).fillna(0)

    log.debug(f"Shrink ratio r0 computed, mean={r0.mean():.3f}")
    return r0
