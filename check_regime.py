"""
Quick diagnostic: checks whether the HMM regime detector found a genuine
split between "trending" and "ranging" days, or collapsed into the
degenerate near-single-state behavior that was found and fixed during
development (see README's "Bugs I found and fixed" section).

Run AFTER a backtest (needs results/signals_<ASSET>.csv to exist):
    python check_regime.py
"""

import os
import pandas as pd

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def check_asset(asset: str):
    path = os.path.join(RESULTS_DIR, f"signals_{asset}.csv")
    if not os.path.exists(path):
        print(f"{asset}: no results file found at {path} — run a backtest first.")
        return

    df = pd.read_csv(path)
    counts = df["regime"].value_counts()
    pct = (counts / len(df) * 100).round(1)

    print(f"\n=== {asset} ===")
    for regime, n in counts.items():
        print(f"  {regime:10s}: {n:6d} bars ({pct[regime]}%)")

    trending_pct = pct.get("trending", 0)
    ranging_pct = pct.get("ranging", 0)
    unknown_pct = pct.get("unknown", 0)

    if unknown_pct > 5:
        print(f"  ⚠️  {unknown_pct}% 'unknown' (indicator warmup) — normal only if this is a small fraction of a short dataset.")

    minority = min(trending_pct, ranging_pct) if trending_pct and ranging_pct else 0
    if minority < 5:
        print(
            f"  ⚠️  DEGENERATE SPLIT: one regime has under 5% of bars — this looks like the "
            f"bug found during development (HMM dominated by feature scale or warmup artifacts), "
            f"not a genuine regime split. Worth investigating before trusting results."
        )
    else:
        print(f"  ✅ Looks like a genuine split — both regimes have meaningful representation.")


if __name__ == "__main__":
    for asset in ["EURUSD", "USDJPY"]:
        check_asset(asset)
