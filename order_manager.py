"""
Order manager — records every order regardless of fill outcome. Same
pattern as the original system.
"""

from dataclasses import dataclass

import pandas as pd

from logging_utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class Order:
    date: object
    asset: str
    side: str
    target_notional: float
    status: str = "PENDING"
    fill_price: float = None
    fill_notional: float = None
    cost: float = None
    lots: float = None
    reason: str = ""


class OrderManager:
    def __init__(self):
        self.orders: list[Order] = []

    def create_order(self, date, asset: str, side: str, target_notional: float, reason: str = "", lots: float = None) -> Order:
        order = Order(date=date, asset=asset, side=side, target_notional=target_notional, reason=reason, lots=lots)
        self.orders.append(order)
        log.info(f"[{date}] ORDER CREATED: {side} {asset} target_notional={target_notional:.2f} lots={lots} ({reason})")
        return order

    def mark_filled(self, order: Order, fill_price: float, fill_notional: float, cost: float):
        order.status = "FILLED"
        order.fill_price = fill_price
        order.fill_notional = fill_notional
        order.cost = cost

    def mark_rejected(self, order: Order, reason: str):
        order.status = "REJECTED"
        order.reason = f"{order.reason} | rejected: {reason}".strip(" |")
        log.warning(f"[{order.date}] ORDER REJECTED: {order.side} {order.asset} — {reason}")

    def to_dataframe(self) -> pd.DataFrame:
        if not self.orders:
            return pd.DataFrame(columns=["date", "asset", "side", "target_notional", "status",
                                          "fill_price", "fill_notional", "cost", "lots", "reason"])
        return pd.DataFrame([o.__dict__ for o in self.orders])
