"""
Data loading — the only module that touches raw price files.

Loads full OHLCV where available (Open, High, Low, Close, Volume) — not just
a single "price" column. High/Low feed ATR (a realistic volatility measure
for risk sizing); Volume feeds a liquidity filter. Only Date + Close are
strictly required — everything else degrades gracefully if missing.

Column names don't have to be literally "date"/"close" — common real-world
variants are auto-detected (see _ALIASES), covering raw MT5 exports too
(Time, Open, High, Low, Close, Tick Volume, Volume, Spread).
"""

import pandas as pd

from logging_utils.logger import get_logger

log = get_logger(__name__)

_ALIASES = {
    "date": ["date", "time", "datetime", "timestamp", "date/time"],
    "open": ["open", "open price"],
    "high": ["high", "high price"],
    "low": ["low", "low price"],
    "close": ["close", "price", "close price", "closing price", "adj close", "adjusted close", "last"],
    "volume": ["volume", "vol", "tick volume", "tickvol", "real volume"],
}
_REQUIRED = ["date", "close"]
_OPTIONAL = ["open", "high", "low", "volume"]


def _detect_encoding(path: str) -> str:
    """
    Sniffs the first few bytes for a byte-order-mark to detect encoding
    before trying to parse. Files exported from Excel/MT5 on Windows using
    "Unicode text" (rather than "CSV UTF-8") save as UTF-16 with a BOM —
    reading that as UTF-8 fails immediately on the BOM byte itself
    ('utf-8' codec can't decode byte 0xff ...), which is confusing without
    this check since the error looks like a data problem, not an encoding one.
    """
    with open(path, "rb") as f:
        head = f.read(4)

    if head.startswith(b"\xff\xfe"):
        return "utf-16-le"
    if head.startswith(b"\xfe\xff"):
        return "utf-16-be"
    if head.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    return "utf-8"


def _looks_like_data_row(values) -> bool:
    """
    True if a row of values looks like actual price DATA rather than column
    labels — i.e. the file has no header row at all. Common with raw MT5
    "Export Bars" output, which writes straight to data with no label line.
    Heuristic: first cell parses as a date/datetime, and at least the next
    couple of cells parse as plain numbers (real column names never do both).
    """
    if len(values) < 2:
        return False
    first_is_date = pd.notna(pd.to_datetime(str(values[0]), errors="coerce"))
    if not first_is_date:
        return False
    numeric_count = sum(1 for v in values[1:] if pd.notna(pd.to_numeric(str(v), errors="coerce")))
    return numeric_count >= min(2, len(values) - 1)


# Standard column order for the common headerless MT5 "Export Bars" formats,
# keyed by column COUNT since that's all we have to go on with no header.
_HEADERLESS_LAYOUTS = {
    5: ["date", "open", "high", "low", "close"],
    6: ["date", "open", "high", "low", "close", "volume"],
    7: ["date", "open", "high", "low", "close", "volume", "spread"],
    8: ["date", "time", "open", "high", "low", "close", "volume", "spread"],  # separate date+time columns variant
}


def _read_csv_robust(path: str) -> pd.DataFrame:
    """Tries the BOM-detected encoding first, then falls back through the
    other common ones rather than failing outright on the first miss —
    covers files with no BOM at all that still aren't plain UTF-8
    (e.g. Windows-1252 from older Excel exports). Also detects and repairs
    headerless files (see _looks_like_data_row)."""
    detected = _detect_encoding(path)
    tried = []
    df = None
    used_encoding = None
    for enc in [detected, "utf-8-sig", "utf-16", "cp1252", "latin-1"]:
        if enc in tried:
            continue
        tried.append(enc)
        try:
            df = pd.read_csv(path, encoding=enc)
            used_encoding = enc
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
    if df is None:
        raise ValueError(
            f"{path}: could not read with any of the attempted encodings {tried}. "
            "Try re-saving the file as 'CSV UTF-8' from Excel, or tell me the actual encoding if you know it."
        )
    if used_encoding != "utf-8":
        log.info(f"{path}: read using '{used_encoding}' encoding (not plain UTF-8 — detected via {'BOM' if used_encoding == detected else 'fallback'}).")

    if _looks_like_data_row(list(df.columns)):
        n_cols = len(df.columns)
        layout = _HEADERLESS_LAYOUTS.get(n_cols)
        if layout is None:
            raise ValueError(
                f"{path}: file appears to have no header row (first row looks like data: {list(df.columns)}), "
                f"but {n_cols} columns doesn't match any known layout (expected 5, 6, 7, or 8 columns for the "
                "standard headerless MT5 export formats). Add a header row to the file, or tell me the exact "
                "column order and I'll add that layout."
            )
        log.info(
            f"{path}: no header row detected (first row was data: {list(df.columns)[:3]}...) — "
            f"re-reading as headerless with {n_cols}-column layout: {layout}"
        )
        df = pd.read_csv(path, encoding=used_encoding, header=None, names=layout)
        if "time" in df.columns:  # 8-col variant: merge separate date+time columns into one
            df["date"] = df["date"].astype(str) + " " + df["time"].astype(str)
            df = df.drop(columns=["time"])

    return df


def _find_column(columns, aliases):
    lower_map = {str(c).strip().lower(): c for c in columns}
    for alias in aliases:
        if alias in lower_map:
            return lower_map[alias]
    return None


def load_price_series(path: str, col_overrides: dict = None) -> pd.DataFrame:
    """
    Returns ['date', 'close', 'price'] always ('price' aliases 'close'),
    plus ['open', 'high', 'low', 'volume'] wherever found.
    """
    col_overrides = col_overrides or {}
    df = _read_csv_robust(path)
    original_columns = list(df.columns)

    resolved = {}
    for field in ["date", "open", "high", "low", "close", "volume"]:
        resolved[field] = col_overrides.get(field) or _find_column(original_columns, _ALIASES[field])

    missing_required = [f for f in _REQUIRED if resolved[f] is None]
    if missing_required:
        raise ValueError(
            f"Could not find required column(s) {missing_required} in {path}.\n"
            f"  Columns found in the file: {original_columns}\n"
            f"  Recognized headers (case-insensitive): { {f: _ALIASES[f] for f in missing_required} }\n"
            "  Fix: rename the column(s) in your CSV, or pass col_overrides={'date': 'YourExactHeader', ...}."
        )

    found_optional = [f for f in _OPTIONAL if resolved[f] is not None]
    missing_optional = [f for f in _OPTIONAL if resolved[f] is None]
    log.info(
        f"{path}: using '{resolved['date']}' as date, '{resolved['close']}' as close. "
        f"OHLCV extras found: {found_optional or 'none'}"
        + (f" — missing: {missing_optional}" if missing_optional else "")
    )

    rename_map = {resolved[f]: f for f in ["date", "open", "high", "low", "close", "volume"] if resolved[f] is not None}
    df = df.rename(columns=rename_map)
    keep_cols = [f for f in ["date", "open", "high", "low", "close", "volume"] if f in df.columns]
    df = df[keep_cols].copy()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for f in ["open", "high", "low", "close", "volume"]:
        if f in df.columns:
            df[f] = pd.to_numeric(df[f], errors="coerce")

    n_before = len(df)
    df = df.dropna(subset=["date", "close"])
    df = df.drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)
    n_after = len(df)

    dropped_pct = (n_before - n_after) / n_before * 100 if n_before else 0
    log.info(f"Loaded {path}: {n_after} valid rows ({dropped_pct:.2f}% dropped as missing/invalid)")
    if n_after == 0:
        raise ValueError(f"{path}: all rows dropped as invalid after parsing date/close — check formats.")

    if {"high", "low"}.issubset(df.columns):
        bad_hl = (df["high"] < df["low"]).sum()
        if bad_hl > 0:
            log.warning(f"{path}: {bad_hl} rows have High < Low — check for a column mixup.")

    df["price"] = df["close"]
    return df
