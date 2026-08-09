"""
Portfolio — tracks cash + per-asset unit holdings, marks to market each day.
Same pattern as the original system's portfolio module.
"""

from logging_utils.logger import get_logger

log = get_logger(__name__)


class Portfolio:
    def __init__(self, initial_cash: float, assets: list, account_currency_is_quote: dict = None):
        """
        account_currency_is_quote: {asset_name: bool} — see risk_manager.size_to_lots
        for why this matters (EURUSD vs USDJPY convert opposite ways for a USD
        account). Defaults to True for every asset if not given (safe for
        quote-based pairs; USDJPY-style pairs MUST pass False or position
        values will be wrong by a factor of ~price).
        """
        self.cash = initial_cash
        self.units = {a: 0.0 for a in assets}
        self.history = []
        self.account_currency_is_quote = account_currency_is_quote or {a: True for a in assets}

    def _value(self, asset: str, units: float, price: float) -> float:
        return units * price if self.account_currency_is_quote.get(asset, True) else units

    def buy(self, asset: str, units: float, notional: float, price: float, cost: float):
        """units and notional must already be consistent (from risk_manager.size_to_lots) —
        this method no longer re-derives units from notional/price, since that
        conversion direction differs by asset (see size_to_lots)."""
        spend = notional + cost
        if spend > self.cash:
            scale = max(self.cash - cost, 0.0) / notional if notional else 0.0
            units *= scale
            notional *= scale
            spend = notional + cost
        self.cash -= spend
        self.units[asset] += units
        return units

    def sell(self, asset: str, units: float, notional: float, price: float, cost: float):
        units = min(units, self.units[asset])
        notional = self._value(asset, units, price)
        self.cash += notional - cost
        self.units[asset] -= units
        return units

    def flatten_all(self, prices: dict, broker) -> list:
        fills = []
        for asset, u in list(self.units.items()):
            if u > 0:
                notional = self._value(asset, u, prices[asset])
                cost = notional * broker.transaction_cost_pct.get(asset, 0.0)
                self.sell(asset, u, notional, prices[asset], cost)
                fills.append((asset, notional, cost))
        return fills

    def mark_to_market(self, date, prices: dict) -> float:
        asset_values = {a: self._value(a, self.units[a], prices.get(a, 0.0)) for a in self.units}
        equity = self.cash + sum(asset_values.values())
        row = {"date": date, "cash": self.cash, "equity": equity}
        for a, v in asset_values.items():
            row[f"{a}_units"] = self.units[a]
            row[f"{a}_value"] = v
        self.history.append(row)
        return equity
