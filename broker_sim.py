"""
Execution — simulated broker for backtesting. Applies transaction cost as a
% of notional (spread/commission approximation) — swap for real spreads per
pair if you have them (EURUSD/USDJPY typically much tighter than crypto/gold,
adjust config accordingly).
"""

from dataclasses import dataclass

from logging_utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class Fill:
    date: object
    asset: str
    side: str
    price: float
    notional: float
    cost: float
    units: float


class BrokerSim:
    def __init__(self, transaction_cost_pct: dict):
        self.transaction_cost_pct = transaction_cost_pct

    def execute(self, date, asset: str, side: str, price: float, notional: float) -> Fill:
        cost_pct = self.transaction_cost_pct.get(asset, 0.0)
        cost = notional * cost_pct
        units = notional / price if price else 0.0
        log.info(f"[{date}] FILL {side} {asset}: notional={notional:.2f} @ {price:.5f} (units={units:.2f}, cost={cost:.2f})")
        return Fill(date=date, asset=asset, side=side, price=price, notional=notional, cost=cost, units=units)
