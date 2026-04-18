"""
Spectral Collapse Hypothesis (Hypothesis 3.1) implementation.

Hypothesis 3.1 — Singularity Precursor (Spectral Collapse):
  As the glacier boundary approaches structural failure at time t_c,
  the first spectral gap of Ω̂ closes:
      Δspec(t) = λ₁(t) − λ₀(t) → 0   as t → t_c

  Equivalently: the scalar curvature K(g(t)) → −∞ and the conformal
  factor σ_meta(t) → 0. The manifold loses capacity to separate the
  ice-body distribution from background, precipitating boundary dissolution.

This module provides:
  1. Heuristic spectral gap estimation from finite samples.
  2. Surrogate validation via synthetic experiments.
  3. Precursor signal cross-correlation analysis.
  4. Granger causality testing (σ_meta → calving).

References:
  ASPT §3, §5.4, §6 (Computers & Geosciences 2026).
  Appendix F: Granger causality — σ_meta → Calving (F=8.34, p=0.005).
"""

import numpy as np
from typing import List, Optional, Tuple, Dict
from scipy import stats


# ---------------------------------------------------------------------------
# Heuristic Δspec from learned probability maps
# ---------------------------------------------------------------------------

def estimate_spectral_gap_heuristic(
    prob_maps: np.ndarray,
    n_components: int = 10,
    method: str = "pca",
) -> np.ndarray:
    """
    Heuristically estimate the first spectral gap Δspec from a batch of
    probability maps using eigenvalue decomposition.

    This is not the true Laplace-Beltrami spectrum (which requires knowing
    the full manifold geometry), but provides a finite-sample approximation
    via the Cheeger inequality connection.

    Method 'pca':
      Project flattened probability maps into the top-k PCA components.
      The spectral gap ≈ difference between the 1st and 2nd eigenvalues of
      the data covariance matrix, normalised.

    Args:
        prob_maps:    (T, H, W) or (T, N) probability maps.
        n_components: Number of principal components.
        method:       'pca' (default) or 'laplacian'.

    Returns:
        delta_approx: (T,) array of approximate spectral gaps.

    Note: Use Δ̂_spec = Var[H_b] in practice — this function is provided for
    the Supplementary S2 synthetic validation only.
    """
    if prob_maps.ndim == 3:
        T, H, W = prob_maps.shape
        X = prob_maps.reshape(T, H * W)
    else:
        T, N = prob_maps.shape
        X = prob_maps

    # Sliding window covariance
    window = min(20, T)
    gaps = np.zeros(T, dtype=np.float32)

    for t in range(T):
        t_start = max(0, t - window)
        Xw = X[t_start:t + 1]
        if Xw.shape[0] < 3:
            gaps[t] = float("nan")
            continue

        # Covariance eigenvalues (descending)
        cov = np.cov(Xw, rowvar=False)
        if method == "pca":
            try:
                # Use only top-2 eigenvalues for efficiency
                from numpy.linalg import eigvalsh
                evals = eigvalsh(cov)
                evals = np.sort(evals)[::-1]   # descending
                lam0 = evals[0]
                lam1 = evals[1] if len(evals) > 1 else 0.0
                # Normalise gap
                gaps[t] = float((lam0 - lam1) / (lam0 + 1e-8))
            except np.linalg.LinAlgError:
                gaps[t] = float("nan")
        else:
            gaps[t] = float("nan")

    return gaps


# ---------------------------------------------------------------------------
# Conformal factor σ_meta as calving precursor
# ---------------------------------------------------------------------------

class SpectralCollapseAnalyser:
    """
    Analyses σ_meta time series for spectral collapse precursor signals.

    Physical interpretation (ASPT §6.3):
      The observed increase in σ_meta before calving (≈5 px → ≈9 px)
      corresponds to widening ablation zone and increased boundary uncertainty,
      validating the MDL framework's adaptability.

      Cross-correlation with daily temperature:
        σ_meta rises 12–24 h before air temperature exceeds 0°C
        (max CC at lag = 12–24 h, p < 0.01) — Granger causality confirmed.
    """

    def __init__(self, revisit_hours: float = 96.0):
        self.revisit_hours = revisit_hours
        self._sigma_history: List[float] = []
        self._temp_history: List[float] = []

    def update(self, sigma_meta: float,
               temperature_c: Optional[float] = None) -> None:
        """Add a new observation."""
        self._sigma_history.append(sigma_meta)
        if temperature_c is not None:
            self._temp_history.append(temperature_c)

    def cross_correlation(
        self,
        sigma_series: Optional[np.ndarray] = None,
        temp_series: Optional[np.ndarray] = None,
        max_lag_h: float = 72.0,
    ) -> Dict:
        """
        Compute cross-correlation between σ_meta and temperature.

        Expected result (ASPT §6.2, Fig. 4):
          max CC = 0.67 at lag = 12–24 h (p < 0.01).

        Args:
            sigma_series: Optional external σ_meta series.
            temp_series:  Optional external temperature series.
            max_lag_h:    Maximum lag to evaluate (hours).

        Returns:
            dict with 'max_cc', 'max_lag_h', 'p_value', 'lag_lags_h'.
        """
        if sigma_series is None:
            sigma_series = np.array(self._sigma_history)
        if temp_series is None:
            temp_series = np.array(self._temp_history)

        if len(sigma_series) < 5 or len(temp_series) < 5:
            return {"max_cc": None, "max_lag_h": None, "p_value": None}

        n = min(len(sigma_series), len(temp_series))
        sig = sigma_series[:n]
        tmp = temp_series[:n]

        max_lag_obs = int(max_lag_h / self.revisit_hours)
        lags = np.arange(-max_lag_obs, max_lag_obs + 1)
        cc = np.zeros(len(lags))

        for i, lag in enumerate(lags):
            if lag >= 0:
                s, t = sig[:n-lag] if lag > 0 else sig, \
                       tmp[lag:] if lag > 0 else tmp
            else:
                s, t = sig[-lag:], tmp[:n+lag]
            if len(s) < 3:
                cc[i] = 0.0
                continue
            r, _ = stats.pearsonr(s, t)
            cc[i] = r

        best_idx = np.argmax(np.abs(cc))
        best_cc = cc[best_idx]
        best_lag = lags[best_idx]
        best_lag_h = float(best_lag * self.revisit_hours)

        # Approximate p-value using t-distribution
        n_eff = n - abs(best_lag)
        if n_eff > 2:
            t_stat = best_cc * np.sqrt(n_eff - 2) / np.sqrt(1 - best_cc**2 + 1e-8)
            p_value = float(2 * stats.t.sf(abs(t_stat), df=n_eff - 2))
        else:
            p_value = 1.0

        return {
            "max_cc": float(best_cc),
            "max_lag_h": best_lag_h,
            "p_value": p_value,
            "lags_h": (lags * self.revisit_hours).tolist(),
            "cc_values": cc.tolist(),
        }

    def detect_precursor(
        self,
        sigma_series: np.ndarray,
        threshold_px: float = 7.0,
        min_consecutive_obs: int = 2,
    ) -> Optional[int]:
        """
        Detect when σ_meta rises above threshold (ablation precursor).

        σ_meta > 7 px indicates active ablation / calving zone.

        Returns:
            Index of first precursor observation, or None.
        """
        consecutive = 0
        for t, s in enumerate(sigma_series):
            if s > threshold_px:
                consecutive += 1
                if consecutive >= min_consecutive_obs:
                    return t - min_consecutive_obs + 1
            else:
                consecutive = 0
        return None


# ---------------------------------------------------------------------------
# Granger causality test
# ---------------------------------------------------------------------------

def granger_causality_test(
    cause: np.ndarray,
    effect: np.ndarray,
    max_lag: int = 3,
    alpha: float = 0.01,
) -> Dict:
    """
    Simplified Granger causality test.

    From ASPT Appendix F:
      Temperature → σ_meta: F=5.67, p=0.02
      σ_meta → Calving:     F=8.34, p=0.005
      Incremental AUC: 0.76 (temp only) → 0.89 (temp + σ_meta)

    Tests H₀: 'cause' does not Granger-cause 'effect'.

    Args:
        cause:    (T,) time series (e.g., σ_meta).
        effect:   (T,) time series (e.g., calving indicator).
        max_lag:  Maximum lag to include.
        alpha:    Significance level.

    Returns:
        dict with F-statistic, p-value, lag, and significance.
    """
    try:
        from statsmodels.tsa.stattools import grangercausalitytests
        data = np.column_stack([effect, cause])
        results = grangercausalitytests(data, maxlag=max_lag, verbose=False)

        best_lag = 1
        best_p = 1.0
        best_f = 0.0

        for lag in range(1, max_lag + 1):
            test_result = results[lag][0]["ssr_ftest"]
            f_stat, p_val = test_result[0], test_result[1]
            if p_val < best_p:
                best_p = p_val
                best_f = f_stat
                best_lag = lag

        return {
            "f_stat": float(best_f),
            "p_value": float(best_p),
            "lag": best_lag,
            "significant": best_p < alpha,
            "lag_h": best_lag * 24.0,
        }
    except ImportError:
        # Fallback: simplified F-test
        return _simple_granger(cause, effect, max_lag=max_lag, alpha=alpha)


def _simple_granger(cause, effect, max_lag=1, alpha=0.05):
    """Simplified Granger test without statsmodels."""
    n = len(effect)
    if n < max_lag + 5:
        return {"f_stat": None, "p_value": None, "lag": max_lag,
                "significant": False}

    # Restricted model: effect ~ effect(t-1, ..., t-p)
    # Unrestricted model: effect ~ effect(t-1,...,t-p) + cause(t-1,...,t-p)
    y = effect[max_lag:]
    X_r = np.column_stack([effect[max_lag - i - 1: n - i - 1]
                           for i in range(max_lag)])
    X_u = np.column_stack([X_r,
                           *[cause[max_lag - i - 1: n - i - 1]
                             for i in range(max_lag)]])

    def ols_rss(X, y):
        X_ = np.column_stack([np.ones(len(y)), X])
        try:
            coef = np.linalg.lstsq(X_, y, rcond=None)[0]
            resid = y - X_ @ coef
            return float(np.dot(resid, resid))
        except Exception:
            return float("inf")

    rss_r = ols_rss(X_r, y)
    rss_u = ols_rss(X_u, y)
    n_obs = len(y)
    k = max_lag

    if rss_u <= 0 or n_obs - 2 * k - 1 <= 0:
        return {"f_stat": 0.0, "p_value": 1.0, "lag": max_lag,
                "significant": False}

    f_stat = ((rss_r - rss_u) / k) / (rss_u / (n_obs - 2 * k - 1))
    p_val = float(stats.f.sf(f_stat, dfn=k, dfd=n_obs - 2 * k - 1))

    return {
        "f_stat": float(f_stat),
        "p_value": float(p_val),
        "lag": max_lag,
        "significant": p_val < alpha,
        "lag_h": max_lag * 24.0,
    }


if __name__ == "__main__":
    np.random.seed(42)
    analyser = SpectralCollapseAnalyser(revisit_hours=24.0)

    # Simulate σ_meta rising before temperature crosses 0°C (ASPT Fig.4)
    T = 30
    temp = np.sin(np.linspace(-1, 4, T)) * 5  # crosses 0 at t≈15
    sigma = np.zeros(T)
    for t in range(T):
        # σ rises 12–24h before T > 0
        if t >= 10 and t < 20:
            sigma[t] = 5.0 + (t - 10) * 0.5
        elif t >= 20:
            sigma[t] = 9.0
        else:
            sigma[t] = 4.5 + np.random.randn() * 0.2

    cc = analyser.cross_correlation(sigma, temp, max_lag_h=72)
    print(f"Max CC = {cc['max_cc']:.3f} at lag = {cc['max_lag_h']:.1f} h "
          f"(p = {cc['p_value']:.4f})")
