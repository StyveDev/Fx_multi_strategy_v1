"""
Backtest engine — orchestrates data -> multi-strategy signals -> risk ->
execution -> orders -> portfolio, for exactly two pairs: EURUSD and USDJPY.

IMPORTANT DESIGN NOTE — avoiding look-ahead bias in the regime detector:
Fitting the HMM on the full history and then using it to classify the SAME
history's early days would be look-ahead (the model's parameters would have
been shaped by data that hadn't happened yet at that point in time). To avoid
this, each asset's history is split:
    [regime_fit_window]  ->  used ONLY to fit the HMM, never traded
    [trading_window]     ->  the (already-fitted) HMM just classifies these
                              days going forward; actual trading only
                              happens here
This mirrors exactly how it would work live: you fit once on history you
already have, then only ever call .predict() on new days as they arrive.
"""

import os
import numpy as np
import pandas as pd

from data_layer.loader import load_price_series
from data_layer.mt5_connector import MT5Connector
from strategy.multi_strategy_signal_generator import generate_multi_strategy_signals
from strategy.regime_detector import RegimeDetector
from risk_management.risk_manager import RiskManager, RiskLimits
from execution.broker_sim import BrokerSim
from orders.order_manager import OrderManager
from portfolio.portfolio import Portfolio
from logging_utils.logger import get_logger

log = get_logger(__name__)

ALLOWED_SYMBOLS = {"EURUSD", "USDJPY"}
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(BASE_DIR, path)


def _validate_assets(assets_cfg: list):
    names = {a["name"].upper() for a in assets_cfg}
    if not names.issubset(ALLOWED_SYMBOLS):
        raise ValueError(
            f"This system is strictly scoped to {sorted(ALLOWED_SYMBOLS)} — found {sorted(names)}. "
            "Remove any other symbols from config.yaml's assets list."
        )


def _load_asset_prices(asset_cfg: dict, data_source: str, mt5_connector: MT5Connector = None, mt5_cfg: dict = None) -> pd.DataFrame:
    if data_source == "mt5":
        if mt5_connector is None:
            raise RuntimeError("data_source='mt5' but no active MT5 connection was passed in.")
        return mt5_connector.get_price_history(
            symbol=asset_cfg["mt5_symbol"], date_from=mt5_cfg["date_from"],
            date_to=mt5_cfg["date_to"], timeframe=mt5_cfg.get("timeframe", "D1"),
        )
    col_overrides = {f: asset_cfg[f"{f}_col"] for f in ["date", "open", "high", "low", "close", "volume"] if asset_cfg.get(f"{f}_col")}
    return load_price_series(_resolve_path(asset_cfg["price_file"]), col_overrides=col_overrides or None)


def _generate_signals_for_asset(raw: pd.DataFrame, strat_cfg: dict, regime_fit_fraction: float):
    """Returns (signals_df, regime_detector, trade_start_index) — signals_df covers
    the FULL history (needed since indicators use rolling lookback), but the
    caller should only start creating orders from trade_start_index onward."""
    fit_end = max(int(len(raw) * regime_fit_fraction), strat_cfg["hmm_n_states"] * 15)
    fit_end = min(fit_end, len(raw) - 1)

    # Pass 1: fit the regime detector using ONLY the fit window's data.
    fit_slice = raw.iloc[:fit_end].reset_index(drop=True)
    _, fitted_detector = generate_multi_strategy_signals(
        fit_slice,
        mr_window=strat_cfg["mean_reversion"]["window"], mr_entry=strat_cfg["mean_reversion"]["entry_threshold"],
        mr_exit=strat_cfg["mean_reversion"]["exit_threshold"],
        tf_ema_fast=strat_cfg["trend_following"]["ema_fast"], tf_ema_slow=strat_cfg["trend_following"]["ema_slow"],
        tf_adx_window=strat_cfg["trend_following"]["adx_window"], tf_adx_threshold=strat_cfg["trend_following"]["adx_threshold"],
        backcast_window=strat_cfg["backcast_window"], hmm_n_states=strat_cfg["hmm_n_states"],
    )

    # Pass 2: classify the FULL history using the already-fitted detector (predict-only, no refit).
    full_signals, _ = generate_multi_strategy_signals(
        raw,
        mr_window=strat_cfg["mean_reversion"]["window"], mr_entry=strat_cfg["mean_reversion"]["entry_threshold"],
        mr_exit=strat_cfg["mean_reversion"]["exit_threshold"],
        tf_ema_fast=strat_cfg["trend_following"]["ema_fast"], tf_ema_slow=strat_cfg["trend_following"]["ema_slow"],
        tf_adx_window=strat_cfg["trend_following"]["adx_window"], tf_adx_threshold=strat_cfg["trend_following"]["adx_threshold"],
        backcast_window=strat_cfg["backcast_window"], hmm_n_states=strat_cfg["hmm_n_states"],
        regime_detector=fitted_detector,
    )
    return full_signals, fitted_detector, fit_end


def _size_order(direction, shrink_ratio, min_trade_notional, cash, position_value):
    if direction > 0:
        notional, side = shrink_ratio * cash, "BUY"
    elif direction < 0:
        notional, side = shrink_ratio * position_value, "SELL"
    else:
        return None, 0.0
    if notional < min_trade_notional:
        return None, 0.0
    return side, notional


def run_backtest(config: dict, data_source: str = "csv") -> dict:
    assets_cfg = config["assets"]
    _validate_assets(assets_cfg)
    asset_names = [a["name"] for a in assets_cfg]
    tx_costs = {a["name"]: a["transaction_cost_pct"] for a in assets_cfg}
    strat_cfg = config["strategy"]
    risk_cfg = config["risk"]

    log.info(f"Starting multi-strategy backtest — assets={asset_names}, initial_cash={config['initial_cash']}, data_source={data_source}")

    mt5_connector = None
    if data_source == "mt5":
        from data_layer.mt5_connector import MT5Credentials
        creds = MT5Credentials.from_config(config["mt5"])
        mt5_connector = MT5Connector(creds)
        mt5_connector.connect()

    try:
        signals = {}
        trade_start = {}
        for a in assets_cfg:
            raw = _load_asset_prices(a, data_source, mt5_connector, config.get("mt5"))
            sig_df, _, fit_end = _generate_signals_for_asset(raw, strat_cfg, strat_cfg["regime_fit_fraction"])
            sig_df = sig_df.set_index("date")
            signals[a["name"]] = sig_df
            trade_start[a["name"]] = sig_df.index[fit_end] if fit_end < len(sig_df) else sig_df.index[-1]
            log.info(f"{a['name']}: regime detector fit on first {fit_end} bars; trading starts {trade_start[a['name']]}")

        all_dates = sorted(set().union(*[set(df.index) for df in signals.values()]))
        price_ffill = {name: df["price"].reindex(all_dates).ffill() for name, df in signals.items()}

        currency_flags = {a["name"]: a.get("account_currency_is_quote", True) for a in assets_cfg}
        portfolio = Portfolio(initial_cash=config["initial_cash"], assets=asset_names, account_currency_is_quote=currency_flags)
        broker = BrokerSim(transaction_cost_pct=tx_costs)
        order_mgr = OrderManager()

        limits = RiskLimits(
            max_portfolio_drawdown_pct=risk_cfg["max_portfolio_drawdown_pct"],
            cooldown_days_after_breach=risk_cfg["cooldown_days_after_breach"],
            min_trade_notional=risk_cfg["min_trade_notional"],
            max_allocation_pct=assets_cfg[0].get("max_allocation_pct", 0.8),
            vol_scaling_enabled=risk_cfg["volatility_scaling"]["enabled"],
            vol_lookback=risk_cfg["volatility_scaling"]["lookback"],
            high_vol_annualized_pct=risk_cfg["volatility_scaling"]["high_vol_annualized_pct"],
            vol_shrink_multiplier=risk_cfg["volatility_scaling"]["shrink_multiplier"],
            liquidity_filter_enabled=risk_cfg.get("liquidity_filter", {}).get("enabled", True),
            min_relative_volume=risk_cfg.get("liquidity_filter", {}).get("min_relative_volume", 0.3),
            leverage=risk_cfg["margin"]["leverage"],
            lot_size=risk_cfg["margin"]["lot_size"],
            min_lot=risk_cfg["margin"]["min_lot"],
            lot_step=risk_cfg["margin"]["lot_step"],
            margin_buffer_pct=risk_cfg["margin"]["margin_buffer_pct"],
        )
        risk_mgr = RiskManager(limits)

        for date in all_dates:
            prices_today = {name: price_ffill[name].loc[date] for name in asset_names}
            current_equity = portfolio.mark_to_market(date, prices_today)
            portfolio.history.pop()

            breached = risk_mgr.check_drawdown(current_equity, date)
            if breached:
                fills = portfolio.flatten_all(prices_today, broker)
                for asset, notional, cost in fills:
                    order = order_mgr.create_order(date, asset, "SELL", notional, reason="risk circuit breaker")
                    order_mgr.mark_filled(order, prices_today[asset], notional, cost)

            for name in asset_names:
                if date not in signals[name].index or date < trade_start[name]:
                    continue

                row = signals[name].loc[date]
                direction = row["direction"]
                shrink_ratio = row["shrink_ratio"]
                is_quote = currency_flags[name]

                asset_cfg = next(a for a in assets_cfg if a["name"] == name)
                per_asset_limits = RiskLimits(**{**limits.__dict__, "max_allocation_pct": asset_cfg.get("max_allocation_pct", 0.8)})
                risk_mgr.limits = per_asset_limits

                recent_vol = signals[name]["volatility"].loc[:date].tail(limits.vol_lookback)
                size_fraction = risk_mgr.adjust_size(
                    shrink_ratio, recent_vol, date,
                    atr_pct=row.get("atr_pct"), relative_volume=row.get("relative_volume"),
                )

                position_value = portfolio._value(name, portfolio.units[name], prices_today[name])
                side, notional = _size_order(direction, size_fraction, limits.min_trade_notional, portfolio.cash, position_value)
                if side is None:
                    continue

                lots, actual_units, actual_notional, margin_required, lot_rejection = risk_mgr.size_to_lots(
                    notional, prices_today[name], account_currency_is_quote=is_quote
                )
                if lot_rejection:
                    order = order_mgr.create_order(date, name, side, notional, reason=row["reason"])
                    order_mgr.mark_rejected(order, lot_rejection)
                    continue

                if side == "BUY":
                    margin_rejection = risk_mgr.check_margin(margin_required, portfolio.cash, current_equity)
                    if margin_rejection:
                        order = order_mgr.create_order(date, name, side, actual_notional, reason=row["reason"], lots=lots)
                        order_mgr.mark_rejected(order, margin_rejection)
                        continue

                order = order_mgr.create_order(date, name, side, actual_notional, reason=row["reason"], lots=lots)
                cost = actual_notional * tx_costs[name]

                if side == "BUY":
                    if actual_notional + cost > portfolio.cash:
                        order_mgr.mark_rejected(order, "insufficient cash")
                        continue
                    portfolio.buy(name, actual_units, actual_notional, prices_today[name], cost)
                else:
                    if portfolio.units[name] <= 0:
                        order_mgr.mark_rejected(order, "no position to sell")
                        continue
                    portfolio.sell(name, actual_units, actual_notional, prices_today[name], cost)

                order_mgr.mark_filled(order, prices_today[name], actual_notional, cost)

            portfolio.mark_to_market(date, prices_today)

        equity_df = pd.DataFrame(portfolio.history)
        equity_df["peak"] = equity_df["equity"].cummax()
        equity_df["drawdown"] = (equity_df["equity"] - equity_df["peak"]) / equity_df["peak"]
        equity_df["pnl"] = equity_df["equity"].diff().fillna(equity_df["equity"] - config["initial_cash"])

        orders_df = order_mgr.to_dataframe()
        risk_breaches_df = pd.DataFrame(risk_mgr.breach_log, columns=["date", "drawdown_pct"])

        survived = equity_df["equity"].min() > 0 and not (equity_df["equity"] <= 0.01 * config["initial_cash"]).any()

        log.info(
            f"Backtest complete — {len(equity_df)} bars, final equity={equity_df['equity'].iloc[-1]:.2f}, "
            f"{len(orders_df)} orders ({(orders_df['status']=='FILLED').sum() if len(orders_df) else 0} filled, "
            f"{risk_mgr.rejected_lot_too_small} lot-too-small, {risk_mgr.rejected_insufficient_margin} insufficient-margin), "
            f"{len(risk_breaches_df)} risk breaches, survived={survived}"
        )

        return {
            "equity_df": equity_df,
            "orders_df": orders_df,
            "signals": {name: df.reset_index() for name, df in signals.items()},
            "risk_breaches_df": risk_breaches_df,
            "survived": survived,
            "rejected_lot_too_small": risk_mgr.rejected_lot_too_small,
            "rejected_insufficient_margin": risk_mgr.rejected_insufficient_margin,
        }
    finally:
        if mt5_connector is not None:
            mt5_connector.disconnect()


def save_results(results: dict, results_dir: str):
    os.makedirs(results_dir, exist_ok=True)
    results["equity_df"].to_csv(os.path.join(results_dir, "equity_curve.csv"), index=False)
    results["orders_df"].to_csv(os.path.join(results_dir, "orders.csv"), index=False)
    results["risk_breaches_df"].to_csv(os.path.join(results_dir, "risk_breaches.csv"), index=False)
    for name, df in results["signals"].items():
        df.to_csv(os.path.join(results_dir, f"signals_{name}.csv"), index=False)
    log.info(f"Results written to {results_dir}/")
