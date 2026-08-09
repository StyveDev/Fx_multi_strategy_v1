"""
Regime detector — Gaussian HMM over [return, ADX] to classify each day into
one of two hidden states: "trending" or "ranging".

WHY THESE TWO FEATURES: return magnitude captures realized volatility/
momentum, ADX captures trend persistence specifically (not just movement —
a choppy, high-volatility range has big returns but low ADX). Using both
lets the HMM separate "trending" from "volatile-but-directionless" instead
of conflating them.

HMM states are unordered/unlabeled by construction — hmmlearn just returns
state indices 0..n-1 with no inherent meaning. After fitting, this module
labels whichever state has the HIGHER average ADX as "trending" and the
other as "ranging" — a necessary post-hoc step, not something the HMM
itself knows.

This is fit ONCE on the available history (in-sample) and then used to
classify every day, INCLUDING a rolling refit option for walk-forward-style
usage during optimization/live trading — see `fit_predict` vs `RegimeDetector`
class below for the two ways to use this.
"""

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

from logging_utils.logger import get_logger

log = get_logger(__name__)


class RegimeDetector:
    """
    Stateful wrapper so the SAME fitted HMM can be reused: fit once on
    training/history data, then call `predict` repeatedly on new data
    (e.g. day by day in live trading, or on held-out data during
    optimization) without refitting from scratch every time.
    """

    def __init__(self, n_states: int = 2, random_state: int = 42, n_iter: int = 100):
        self.n_states = n_states
        self.model = GaussianHMM(n_components=n_states, covariance_type="diag",
                                  n_iter=n_iter, random_state=random_state)
        self.trending_state_ = None
        self.fitted = False
        self.feature_mean_ = None
        self.feature_std_ = None

    def _features(self, df: pd.DataFrame) -> np.ndarray:
        ret = df["price"].pct_change()
        adx = df["adx"] if "adx" in df.columns else pd.Series(np.nan, index=df.index)
        return np.column_stack([ret.values, adx.values])  # may contain NaN — callers handle explicitly below

    def _standardize(self, feats: np.ndarray) -> np.ndarray:
        """
        Z-score each feature dimension using FIT-time mean/std (never
        recomputed at predict time — same principle as reusing a fitted
        sklearn scaler). Without this, 'return' (~1e-4 scale) and 'adx'
        (~10-70 scale) differ by ~5-6 orders of magnitude, and a Gaussian
        HMM's EM fit can end up numerically dominated by the larger-scale
        feature rather than finding genuine structure in both.
        """
        return (feats - self.feature_mean_) / self.feature_std_

    def fit(self, df: pd.DataFrame):
        feats = self._features(df)
        valid_mask = ~np.isnan(feats).any(axis=1)
        valid_feats = feats[valid_mask]

        if len(valid_feats) < self.n_states * 10:
            raise ValueError(
                f"Not enough valid (non-warmup) data to fit a {self.n_states}-state HMM reliably "
                f"({len(valid_feats)} usable rows out of {len(feats)}) — need at least ~{self.n_states * 10}."
            )

        self.feature_mean_ = valid_feats.mean(axis=0)
        self.feature_std_ = valid_feats.std(axis=0)
        self.feature_std_[self.feature_std_ == 0] = 1.0  # guard against a constant feature

        scaled_feats = self._standardize(valid_feats)
        self.model.fit(scaled_feats)
        states_valid = self.model.predict(scaled_feats)

        adx_valid = df["adx"].values[valid_mask]
        adx_by_state = {s: adx_valid[states_valid == s].mean() for s in range(self.n_states)}
        state_counts = {s: int((states_valid == s).sum()) for s in range(self.n_states)}
        self.trending_state_ = max(adx_by_state, key=adx_by_state.get)
        self.fitted = True

        log.info(
            f"HMM regime detector fit on {len(valid_feats)} valid rows (of {len(df)} total, "
            f"{len(df) - len(valid_feats)} excluded as warmup/NaN) — "
            f"trending_state={self.trending_state_}, avg ADX by state={adx_by_state}, state counts={state_counts}"
        )
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Returns df with added 'regime' ('trending'/'ranging'/'unknown') and 'regime_prob_trending' columns.
        'unknown' means indicators hadn't warmed up yet for this row — never coerced into a real regime label."""
        if not self.fitted:
            raise RuntimeError("Call fit() before predict().")
        feats = self._features(df)
        valid_mask = ~np.isnan(feats).any(axis=1)

        states = np.full(len(df), -1, dtype=int)
        probs_trending = np.full(len(df), np.nan)
        if valid_mask.any():
            scaled_feats = self._standardize(feats[valid_mask])
            states[valid_mask] = self.model.predict(scaled_feats)
            probs = self.model.predict_proba(scaled_feats)
            probs_trending[valid_mask] = probs[:, self.trending_state_]

        out = df.copy()
        out["regime"] = np.where(
            states == -1, "unknown", np.where(states == self.trending_state_, "trending", "ranging")
        )
        out["regime_prob_trending"] = probs_trending
        return out

    def fit_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit(df)
        return self.predict(df)
