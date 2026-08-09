"""
MT5 connection — the ONLY file that imports MetaTrader5, so everything else
stays importable/testable on machines without MT5 installed.

Two connection modes:
  - EXPLICIT LOGIN (login/password/server given): logs into a specific account,
    useful for backtesting against a specific account's history, or running
    live on an account that isn't the one currently open in the terminal.
  - ATTACH (no credentials given): just calls mt5.initialize() and uses
    whatever account is ALREADY logged into the running terminal — this is
    what you want for live trading where you've manually logged into the
    terminal yourself and just want the script to use that session.
"""

import os
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from logging_utils.logger import get_logger

log = get_logger(__name__)

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

_TIMEFRAME_MAP = {}
if mt5 is not None:
    _TIMEFRAME_MAP = {
        "M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30, "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1, "W1": mt5.TIMEFRAME_W1,
    }


@dataclass
class MT5Credentials:
    login: int = 0
    password: str = ""
    server: str = ""
    terminal_path: str = ""
    attach_only: bool = False  # True = don't log in, just attach to whatever's already open

    @classmethod
    def from_config(cls, mt5_cfg: dict) -> "MT5Credentials":
        password = os.environ.get(mt5_cfg.get("password_env", ""), "") or mt5_cfg.get("password", "")
        login = mt5_cfg.get("login", 0)
        attach_only = bool(mt5_cfg.get("attach_only", False)) or not login
        if not password and login and not attach_only:
            log.warning(
                f"config specifies password_env='{mt5_cfg.get('password_env')}' but that environment "
                "variable is not set, and no login was clearly intended without one — falling back to "
                "attach-only mode (using whatever account is already logged into the terminal)."
            )
            attach_only = True
        return cls(
            login=login, password=password, server=mt5_cfg.get("server", ""),
            terminal_path=mt5_cfg.get("terminal_path", ""), attach_only=attach_only,
        )


class MT5Connector:
    def __init__(self, creds: MT5Credentials):
        self.creds = creds
        self.connected = False

    def connect(self) -> dict:
        if mt5 is None:
            raise RuntimeError(
                "MetaTrader5 package not installed. Run `pip install MetaTrader5` "
                "on a Windows machine with the MT5 terminal installed."
            )

        init_kwargs = {}
        if self.creds.terminal_path:
            init_kwargs["path"] = self.creds.terminal_path

        if not mt5.initialize(**init_kwargs):
            raise RuntimeError(f"mt5.initialize() failed: {mt5.last_error()}")

        if not self.creds.attach_only:
            if not mt5.login(self.creds.login, password=self.creds.password, server=self.creds.server):
                err = mt5.last_error()
                mt5.shutdown()
                raise RuntimeError(f"mt5.login() failed for account {self.creds.login}: {err}")
            log.info(f"Logged into MT5 account {self.creds.login} on server '{self.creds.server}'")
        else:
            log.info("Attached to the MT5 terminal's already-logged-in account (no separate login performed).")

        account_info = mt5.account_info()
        if account_info is None:
            mt5.shutdown()
            raise RuntimeError(
                "Connected to the terminal but could not read account info — "
                "make sure a user is actually logged in (check the terminal window)."
            )

        self.connected = True
        info = account_info._asdict()
        log.info(
            f"MT5 account {info.get('login')}: balance={info.get('balance')} {info.get('currency')}, "
            f"leverage=1:{info.get('leverage')}, server={info.get('server')}"
        )
        return info

    def get_account_info(self) -> dict:
        if not self.connected:
            raise RuntimeError("Not connected — call connect() first.")
        return mt5.account_info()._asdict()

    def get_price_history(self, symbol: str, date_from: str, date_to: str, timeframe: str = "D1") -> pd.DataFrame:
        if not self.connected:
            raise RuntimeError("Not connected — call connect() first.")
        tf = _TIMEFRAME_MAP.get(timeframe.upper())
        if tf is None:
            raise ValueError(f"Unknown timeframe '{timeframe}'. Valid: {list(_TIMEFRAME_MAP)}")

        d_from = datetime.strptime(date_from, "%Y-%m-%d")
        d_to = datetime.strptime(date_to, "%Y-%m-%d")

        if not mt5.symbol_select(symbol, True):
            raise RuntimeError(f"Could not select symbol '{symbol}' — check it exists in Market Watch.")

        rates = mt5.copy_rates_range(symbol, tf, d_from, d_to)
        if rates is None or len(rates) == 0:
            raise RuntimeError(f"No price history returned for {symbol} {date_from}..{date_to} — check symbol/dates.")

        df = pd.DataFrame(rates)
        df["date"] = pd.to_datetime(df["time"], unit="s")
        df = df.rename(columns={"tick_volume": "volume"})
        df["close"] = df["close"]
        log.info(f"Pulled {len(df)} {timeframe} bars for {symbol} from MT5 ({date_from}..{date_to})")
        return df[["date", "open", "high", "low", "close", "volume"]].assign(price=df["close"])

    def get_current_tick(self, symbol: str) -> dict:
        """Live bid/ask — used by the live trading loop for real-time fills."""
        if not self.connected:
            raise RuntimeError("Not connected — call connect() first.")
        if not mt5.symbol_select(symbol, True):
            raise RuntimeError(f"Could not select symbol '{symbol}'.")
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            raise RuntimeError(f"No live tick available for {symbol} — market may be closed.")
        return tick._asdict()

    def get_symbol_info(self, symbol: str) -> dict:
        if not self.connected:
            raise RuntimeError("Not connected — call connect() first.")
        info = mt5.symbol_info(symbol)
        if info is None:
            raise RuntimeError(f"No symbol info for '{symbol}'.")
        return info._asdict()

    def disconnect(self):
        if mt5 is not None and self.connected:
            mt5.shutdown()
            log.info("MT5 session closed.")
        self.connected = False


def quick_connection_test(config: dict) -> dict:
    creds = MT5Credentials.from_config(config["mt5"])
    conn = MT5Connector(creds)
    try:
        return conn.connect()
    finally:
        conn.disconnect()


if __name__ == "__main__":
    import yaml
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "config.yaml")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    print("Attempting MT5 connection using config/config.yaml ...")
    try:
        account_info = quick_connection_test(cfg)
    except RuntimeError as e:
        print(f"\n Connection failed: {e}\n")
        raise SystemExit(1)
    print("Connected successfully:")
    for k, v in account_info.items():
        print(f"  {k}: {v}")
