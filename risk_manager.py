"""
Risk management — reuses the drawdown circuit breaker, ATR-based volatility
scaling, and liquidity filter from the original system unchanged, and adds
FX-specific margin/lot-size modeling — this is the part that actually
answers "can this survive on small capital":

A broker won't let you trade an arbitrary dollar notional — it trades in
LOTS (standard lot = 100,000 units of base currency; micro = 0.01 lot =
1,000 units, the usual minimum), and requires MARGIN = notional / leverage
to be available as free cash. With a small account:
  - the minimum tradeable lot might already be a large % of your account
    (e.g. $100 account, 1:100 leverage, EURUSD @ 1.10: 0.01 lot needs
    1,000*1.10/100 = $11 margin — fine; but 1:30 leverage needs $36.7, still
    fine; the SIZING from the shrink ratio might still ask for less than one
    lot, which gets rejected below rather than silently trading zero)
  - rounding to the lot step can make the ACTUAL executed size very different
    from the "ideal" shrink-ratio-scaled size the strategy asked for

Both effects are modeled explicitly below and logged/counted so you can see
exactly why a small-capital run underperforms or gets stuck — not just that
it does.

SIMPLIFICATION (documented): margin is computed as notional/leverage in
account-currency terms directly (no separate base/quote currency conversion
step) — accurate when the account currency matches the quote currency of the
pair (e.g. USD account trading EURUSD or USDJPY, both quoted with USD on one
side), which covers the two pairs this system is scoped to. A cross-currency
account (e.g. EUR account trading USDJPY) would need an extra conversion this
doesn't do.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from logging_utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class RiskLimits:
    max_portfolio_drawdown_pct: float = 0.30
    cooldown_days_after_breach: int = 5
    min_trade_notional: float = 1.0
    max_allocation_pct: float = 0.8
    vol_scaling_enabled: bool = True
    vol_lookback: int = 20
    high_vol_annualized_pct: float = 80.0
    vol_shrink_multiplier: float = 0.5
    liquidity_filter_enabled: bool = True
    min_relative_volume: float = 0.3
    # FX margin/lot modeling:
    leverage: float = 100.0
    lot_size: float = 100_000.0   # units of base currency per standard lot
    min_lot: float = 0.1
    lot_step: float = 0.01
    margin_buffer_pct: float = 0.10  # keep this much of equity as a safety buffer, never fully margin out


class RiskManager:
    def __init__(self, limits: RiskLimits):
        self.limits = limits
        self.peak_equity = None
        self.cooldown_until = None  # a real timestamp, not a bar counter — see check_drawdown for why
        self.breach_log = []
        self.rejected_lot_too_small = 0
        self.rejected_insufficient_margin = 0

    def _annualized_vol_from_returns(self, returns: pd.Series) -> float:
        if len(returns) < 2 or returns.std() == 0 or returns.isna().all():
            return 0.0
        return float(returns.std() * np.sqrt(252) * 100)

    def _in_cooldown(self, as_of_date) -> bool:
        return self.cooldown_until is not None and pd.Timestamp(as_of_date) < self.cooldown_until

    def check_drawdown(self, current_equity: float, as_of_date) -> bool:
        if self.peak_equity is None or current_equity > self.peak_equity:
            self.peak_equity = current_equity
        drawdown_pct = (current_equity - self.peak_equity) / self.peak_equity if self.peak_equity else 0.0

        if drawdown_pct <= -self.limits.max_portfolio_drawdown_pct and not self._in_cooldown(as_of_date):
            log.warning(
                f"[{as_of_date}] RISK BREACH: drawdown {drawdown_pct:.2%} exceeds "
                f"-{self.limits.max_portfolio_drawdown_pct:.0%} — flattening, cooldown {self.limits.cooldown_days_after_breach}d."
            )
            self.breach_log.append({"date": as_of_date, "drawdown_pct": drawdown_pct})
            # A real calendar-time cutoff — NOT "N bars from now". With bar-count
            # cooldowns, a 5-day cooldown on 30-min bars would only last 2.5 hours
            # (5 bars), not 5 actual days. This works correctly at any timeframe.
            self.cooldown_until = pd.Timestamp(as_of_date) + pd.Timedelta(days=self.limits.cooldown_days_after_breach)
            self.peak_equity = current_equity  # reset high-water mark so a flat cash position can eventually resume trading
            return True
        return False

    def adjust_size(self, raw_shrink_ratio, recent_returns, as_of_date, atr_pct=None, relative_volume=None) -> float:
        if self._in_cooldown(as_of_date):
            return 0.0

        size = raw_shrink_ratio

        if self.limits.vol_scaling_enabled:
            if atr_pct is not None and not np.isnan(atr_pct):
                ann_vol = atr_pct * np.sqrt(252)
            else:
                ann_vol = self._annualized_vol_from_returns(recent_returns.tail(self.limits.vol_lookback))
            if ann_vol > self.limits.high_vol_annualized_pct:
                size *= self.limits.vol_shrink_multiplier

        if self.limits.liquidity_filter_enabled and relative_volume is not None and not np.isnan(relative_volume):
            if relative_volume < self.limits.min_relative_volume:
                return 0.0

        return max(min(size, self.limits.max_allocation_pct), 0.0)

    def size_to_lots(self, notional: float, price: float, account_currency_is_quote: bool = True) -> tuple:
        """
        Converts a desired notional (account-currency exposure) into a
        broker-tradeable lot size, respecting min_lot and lot_step, and
        checks required margin.

        account_currency_is_quote matters because EURUSD and USDJPY convert
        opposite ways for a USD account:
          - EURUSD: quote currency = USD = account currency. `notional` (USD)
            converts to base-currency (EUR) units by DIVIDING by price.
          - USDJPY: base currency = USD = account currency. `notional` (USD)
            IS ALREADY the base-currency (USD) unit count — no division by
            price at all (price is JPY-per-USD, irrelevant to sizing the
            USD-denominated leg). Getting this backwards silently shrinks
            every USDJPY trade by ~100x (price divides when it shouldn't).

        Returns (lots, actual_units, actual_notional, margin_required, rejection_reason).
        actual_notional is always in ACCOUNT currency (what cash/margin actually moves by).
        """
        if account_currency_is_quote:
            units = notional / price if price else 0.0
        else:
            units = notional  # already in account-currency == base-currency units

        raw_lots = units / self.limits.lot_size
        lots = np.floor(raw_lots / self.limits.lot_step) * self.limits.lot_step
        lots = round(lots, 2)

        if lots < self.limits.min_lot:
            self.rejected_lot_too_small += 1
            return 0.0, 0.0, 0.0, 0.0, (
                f"desired size rounds to {lots:.4f} lots, below the broker minimum of "
                f"{self.limits.min_lot} lots — account too small for this trade's risk-scaled size"
            )

        actual_units = lots * self.limits.lot_size
        actual_notional = actual_units * price if account_currency_is_quote else actual_units
        margin_required = actual_notional / self.limits.leverage

        return lots, actual_units, actual_notional, margin_required, None

    def check_margin(self, margin_required: float, free_cash: float, equity: float) -> str:
        """Returns None if margin is available, or a rejection reason string."""
        usable_cash = free_cash - (equity * self.limits.margin_buffer_pct)
        if margin_required > usable_cash:
            self.rejected_insufficient_margin += 1
            return (
                f"margin required ({margin_required:.2f}) exceeds usable free cash "
                f"({usable_cash:.2f} after {self.limits.margin_buffer_pct:.0%} safety buffer) — "
                "account cannot support this position at current leverage"
            )
        return None
