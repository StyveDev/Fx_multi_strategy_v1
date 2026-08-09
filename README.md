# FX Multi-Strategy Portfolio — V1 (Backtest Core)

Strictly scoped to **EURUSD and USDJPY**. Same framework/architecture as the
original Holt-Winters/RSI system, new strategies, and the trading shrink
ratio technique kept exactly as-is (just generalized to work with any two
signals instead of only Holt-Winters + RSI).

## What's in V1

```
├── config/config.yaml          ← everything tunable: pairs, strategy params, risk, margin/lots
├── data_layer/
│   ├── loader.py                 ← CSV OHLCV loader (auto-detects header variants)
│   ├── indicators.py              ← EMA, RSI, ATR, ADX, rolling z-score, relative volume
│   └── mt5_connector.py            ← optional: pull backtest history from MT5 instead of CSV
├── strategy/
│   ├── mean_reversion.py            ← rolling z-score reversion (see rationale in-file)
│   ├── trend_following.py            ← dual EMA crossover, gated by ADX (see rationale in-file)
│   ├── regime_detector.py             ← Gaussian HMM, 2 states, labels by avg ADX
│   ├── backcast.py                     ← rolling directional accuracy — generalized, same math
│   ├── shrink_ratio.py                  ← the ORIGINAL equations (2)/(3), unchanged
│   └── multi_strategy_signal_generator.py  ← ties regime + both strategies + shrink ratio together
├── risk_management/risk_manager.py        ← drawdown breaker, vol scaling, liquidity filter, margin/lot sizing
├── execution/broker_sim.py                 ← simulated fills
├── orders/order_manager.py                  ← order lifecycle log
├── portfolio/portfolio.py                    ← margin-aware FX portfolio (not cash-switching)
├── backtest/engine.py                         ← orchestrates all of the above
├── sample_data/{eurusd,usdjpy}.csv              ← synthetic OHLCV for testing without real data
└── run_backtest.py                                ← entry point
```

## Quick start

```bash
pip install -r requirements.txt
python run_backtest.py --source csv
```

## Testing small-capital survival

```bash
python run_backtest.py --initial-cash 100      # tiny account
python run_backtest.py --initial-cash 20000    # comfortably capitalized
```

The engine tracks and reports:
- `survived`: did equity ever collapse toward zero?
- how many orders got rejected for being **below the broker's minimum lot** (account too small to act on the risk-scaled size) vs **insufficient margin** (account can't cover the position at your configured leverage)

In testing, this system's default $1,000 sits right at the edge of what standard 0.01-lot EURUSD/USDJPY sizing can support — below roughly $5,000 in this configuration, most signals get correctly rejected rather than forced through at an unsafely large relative size. That's the survival mechanism working, not a bug — see `risk_management/risk_manager.py`'s module docstring for the exact lot/margin math.

## Framework decisions carried over vs. changed

**Unchanged (per your requirement):**
- The shrink ratio equations themselves (`strategy/shrink_ratio.py`) — identical math, cubed at the end, same "consistent vs. opposite" branching.
- The back-cast accuracy convention (rolling window, count-of-correct/window) — generalized in `strategy/backcast.py` to work for any strategy's direction signal, not just Holt-Winters.
- Overall module layout: data → strategy → risk → execution → orders → portfolio → backtest engine.

**Necessarily changed:**
- **Portfolio mechanics**: the original system was cash↔asset switching (buy gold with cash). FX is leveraged position trading, so `portfolio/portfolio.py` now tracks lots/margin/floating P&L instead of simple cash-for-units swaps. This is a real structural difference, not a stylistic one.
- **What r1/r2 measure**: r1 = the currently-active strategy's own rolling back-cast accuracy; r2 = the other (secondary) strategy's accuracy. "Consistent" = both strategies agree on direction. The regime detector's job is deciding *which* strategy is primary each day — trend-following on "trending" days, mean-reversion on "ranging" days.

## Bugs I found and fixed while building this (verified, not just described)

1. **Look-ahead bias** in the original back-cast accuracy function — it compared today's signal to *tomorrow's* price move, meaning today's position size secretly depended on information from the future. Fixed to compare yesterday's signal against today's realized move (matching the original Holt-Winters system's causal structure exactly).
2. **Currency conversion never wired in** — `Portfolio` was constructed without `account_currency_is_quote`, so USDJPY positions would have been valued ~100x wrong (JPY-per-USD price applied where it shouldn't be). Fixed and verified: a $15,000 USDJPY position now correctly values at $15,000, not $2.2M.
3. **Crashing signature mismatch** — the engine's calls to `portfolio.buy()`/`sell()` didn't match the method's actual parameters, and would have raised `TypeError` the moment any order actually filled. This had gone unnoticed because a prior test run's capital was too small for any order to reach that code path. Fixed and verified with real fills across multiple capital levels.
4. **Degenerate regime detection** — the HMM was finding a fake "regime" driven entirely by the ADX indicator's own warmup-period zeros, not real market structure (98%+ of days landed in one state). Root cause was two-fold: warmup `NaN`s were being silently coerced to `0` before reaching the HMM, and the two input features (`return` ~1e-4 scale, `ADX` ~10-70 scale) were fed in unstandardized, letting the scale difference dominate the fit. Fixed both — warmup rows are now excluded and marked `"unknown"` rather than misclassified, and features are z-scored before fitting. Regime split is now a genuine ~70/30 rather than ~98/2.

## Coming in later versions (already drafted, held back for now)

- **V2 — Parameter optimizer**: grid search over strategy parameters with a train/test split to catch overfitting, ranking by Sharpe/Calmar/etc.
- **V3 — Live MT5 trading**: connects to your already-logged-in MT5 terminal, reuses this exact signal-generation code (so live behavior can't silently diverge from what you backtested), with a dry-run safety default before any real orders get sent.


