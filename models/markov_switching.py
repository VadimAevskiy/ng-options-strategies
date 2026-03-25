"""
Markov Regime-Switching Model
==============================
Hamilton (1989) regime-switching model for returns with:
  - Switching mean and variance
  - Filtered and smoothed regime probabilities
  - Integration with GARCH (RS-GARCH) when possible

Two regimes: Low-volatility (bull) and High-volatility (bear/crisis)
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class MarkovSwitchingModel:
    """
    Two-regime Markov-switching model for daily returns.

    Model:
      r_t | S_t=j ~ N(mu_j, sigma_j^2), j = 0, 1
      P(S_t=j | S_{t-1}=i) = p_ij

    Regime 0: Low volatility (normal market)
    Regime 1: High volatility (stress/crisis)
    """

    def __init__(self, n_regimes: int = 2):
        self.n_regimes = n_regimes
        self.params = None
        self.filtered_probs = None
        self.smoothed_probs = None
        self.returns = None
        self.loglikelihood = None

    def _unpack_params(self, params):
        """Unpack parameter vector into model parameters."""
        mu0, mu1 = params[0], params[1]
        log_sig0, log_sig1 = params[2], params[3]
        sigma0 = np.exp(log_sig0)
        sigma1 = np.exp(log_sig1)
        # Transition probabilities (logistic transform for [0,1] constraint)
        p00 = 1.0 / (1.0 + np.exp(-params[4]))
        p11 = 1.0 / (1.0 + np.exp(-params[5]))
        return mu0, mu1, sigma0, sigma1, p00, p11

    def _hamilton_filter(self, returns, params):
        """
        Hamilton (1989) filter: vectorised forward pass.
        Returns: filtered_probs (T, 2), log-likelihood
        """
        mu0, mu1, sigma0, sigma1, p00, p11 = self._unpack_params(params)
        T = len(returns)

        P = np.array([[p00, 1 - p00],
                      [1 - p11, p11]])

        denom = 2.0 - p00 - p11
        if abs(denom) < 1e-10:
            pi_stat = np.array([0.5, 0.5])
        else:
            pi_stat = np.array([(1 - p11) / denom, (1 - p00) / denom])

        # Precompute ALL likelihoods vectorised (no per-step scipy.norm calls)
        inv_s0 = 1.0 / sigma0; inv_s1 = 1.0 / sigma1
        c0 = inv_s0 * 0.3989422804014327  # 1/sqrt(2*pi)/sigma
        c1 = inv_s1 * 0.3989422804014327
        z0 = (returns - mu0) * inv_s0
        z1 = (returns - mu1) * inv_s1
        eta_all = np.column_stack([
            c0 * np.exp(-0.5 * z0 * z0),
            c1 * np.exp(-0.5 * z1 * z1)
        ])  # shape (T, 2)

        filtered = np.zeros((T, 2))
        log_lik = 0.0
        xi_pred = pi_stat.copy()
        Pt = P.T

        for t in range(T):
            xi_joint = xi_pred * eta_all[t]
            f_t = xi_joint[0] + xi_joint[1]
            if f_t < 1e-300:
                f_t = 1e-300
                filtered[t] = xi_pred
            else:
                filtered[t, 0] = xi_joint[0] / f_t
                filtered[t, 1] = xi_joint[1] / f_t
            log_lik += np.log(f_t)
            xi_pred[0] = Pt[0, 0] * filtered[t, 0] + Pt[0, 1] * filtered[t, 1]
            xi_pred[1] = Pt[1, 0] * filtered[t, 0] + Pt[1, 1] * filtered[t, 1]

        return filtered, log_lik

    def _kim_smoother(self, filtered, params):
        """
        Kim (1994) smoother: backward pass for smoothed probabilities.
        """
        mu0, mu1, sigma0, sigma1, p00, p11 = self._unpack_params(params)
        T = filtered.shape[0]

        P = np.array([[p00, 1 - p00],
                      [1 - p11, p11]])

        smoothed = np.zeros((T, 2))
        smoothed[-1] = filtered[-1]

        for t in range(T - 2, -1, -1):
            xi_pred = P.T @ filtered[t]
            xi_pred = np.maximum(xi_pred, 1e-15)
            for j in range(2):
                smoothed[t, j] = filtered[t, j] * np.sum(
                    P[j, :] * smoothed[t + 1, :] / xi_pred
                )
            # Normalise
            s = smoothed[t].sum()
            if s > 0:
                smoothed[t] /= s

        return smoothed

    def fit(self, returns: "np.ndarray", verbose: bool = True) -> "MarkovSwitchingModel":
        """
        Fit the Markov-switching model via MLE.

        Parameters
        ----------
        returns : array-like
            Log returns series.
        """
        self.returns = np.asarray(returns, dtype=np.float64)
        r = self.returns[~np.isnan(self.returns)]

        # Initial guesses
        mu_all = np.mean(r)
        sig_all = np.std(r)
        x0 = np.array([
            mu_all,              # mu0 (low vol regime)
            mu_all,              # mu1 (high vol regime)
            np.log(sig_all * 0.7),  # log(sigma0)
            np.log(sig_all * 1.5),  # log(sigma1)
            2.0,                 # logistic(p00) -> ~0.88
            2.0,                 # logistic(p11) -> ~0.88
        ])

        def neg_loglik(params):
            try:
                _, ll = self._hamilton_filter(r, params)
                if np.isnan(ll) or np.isinf(ll):
                    return 1e15
                return -ll
            except Exception:
                return 1e15

        # Optimise — Powell converges faster than Nelder-Mead for this problem
        result = minimize(neg_loglik, x0, method="Powell",
                         options={"maxiter": 500, "ftol": 1e-8})

        # Quick refinement
        try:
            result2 = minimize(neg_loglik, result.x, method="L-BFGS-B",
                             options={"maxiter": 200})
            if result2.fun < result.fun:
                result = result2
        except Exception:
            pass

        self.params = result.x
        self.filtered_probs, self.loglikelihood = self._hamilton_filter(r, self.params)
        self.smoothed_probs = self._kim_smoother(self.filtered_probs, self.params)

        mu0, mu1, sigma0, sigma1, p00, p11 = self._unpack_params(self.params)

        # Ensure regime 0 is the low-vol regime
        if sigma0 > sigma1:
            # Swap regimes
            self.params[[0, 1]] = self.params[[1, 0]]
            self.params[[2, 3]] = self.params[[3, 2]]
            # Swap transition probs
            self.params[4], self.params[5] = (
                np.log(1.0 / (1.0 / (1.0 + np.exp(-self.params[5]))) - 1e-10 + 1e-10),
                np.log(1.0 / (1.0 / (1.0 + np.exp(-self.params[4]))) - 1e-10 + 1e-10),
            )
            self.filtered_probs = self.filtered_probs[:, ::-1]
            self.smoothed_probs = self.smoothed_probs[:, ::-1]
            mu0, mu1, sigma0, sigma1, p00, p11 = self._unpack_params(self.params)

        if verbose:
            print(f"\nMarkov Switching Model Estimation:")
            print(f"  Regime 0 (low-vol):  mu={mu0:.6f}, sigma={sigma0:.6f} (annualised={sigma0*np.sqrt(252):.2%})")
            print(f"  Regime 1 (high-vol): mu={mu1:.6f}, sigma={sigma1:.6f} (annualised={sigma1*np.sqrt(252):.2%})")
            print(f"  P(stay low)  = p00 = {p00:.4f}, E[duration] = {1/(1-p00):.1f} days")
            print(f"  P(stay high) = p11 = {p11:.4f}, E[duration] = {1/(1-p11):.1f} days")
            print(f"  Log-likelihood: {self.loglikelihood:.2f}")
            n = len(r)
            k = 6
            print(f"  AIC: {-2*self.loglikelihood + 2*k:.2f}")
            print(f"  BIC: {-2*self.loglikelihood + k*np.log(n):.2f}")

        return self

    def get_regime_params(self) -> dict:
        """Return regime parameters as dict."""
        mu0, mu1, sigma0, sigma1, p00, p11 = self._unpack_params(self.params)
        return dict(
            mu_low=mu0, mu_high=mu1,
            sigma_low=sigma0, sigma_high=sigma1,
            sigma_low_ann=sigma0 * np.sqrt(252),
            sigma_high_ann=sigma1 * np.sqrt(252),
            p00=p00, p11=p11,
            expected_duration_low=1 / (1 - p00),
            expected_duration_high=1 / (1 - p11),
        )

    def current_regime_vol(self):
        """
        Probability-weighted conditional volatility at most recent observation.
        sigma_t = P(regime=0) * sigma_0 + P(regime=1) * sigma_1
        """
        mu0, mu1, sigma0, sigma1, p00, p11 = self._unpack_params(self.params)
        prob_high = self.filtered_probs[-1, 1]
        return (1 - prob_high) * sigma0 + prob_high * sigma1

    def regime_conditional_vol_series(self):
        """
        Time series of regime-weighted conditional volatility.
        """
        mu0, mu1, sigma0, sigma1, _, _ = self._unpack_params(self.params)
        p_high = self.smoothed_probs[:, 1]
        return (1 - p_high) * sigma0 + p_high * sigma1

    def predict_regime(self, n_ahead=5):
        """
        Predict regime probabilities n_ahead steps forward.
        """
        mu0, mu1, sigma0, sigma1, p00, p11 = self._unpack_params(self.params)
        P = np.array([[p00, 1 - p00],
                      [1 - p11, p11]])

        current = self.filtered_probs[-1]
        predictions = np.zeros((n_ahead, 2))
        for h in range(n_ahead):
            current = P.T @ current
            predictions[h] = current
        return predictions

    def get_info_criteria(self) -> dict:
        """Return AIC and BIC."""
        n = len(self.returns[~np.isnan(self.returns)])
        k = 6
        aic = -2 * self.loglikelihood + 2 * k
        bic = -2 * self.loglikelihood + k * np.log(n)
        return dict(aic=aic, bic=bic, loglikelihood=self.loglikelihood, n_params=k)


class RSGARCHModel:
    """
    Regime-Switching GARCH: GARCH dynamics within each regime.
    Simplified implementation: run separate GARCH filters for each regime,
    weight by regime probabilities.
    """

    def __init__(self):
        self.ms_model = MarkovSwitchingModel()
        self.garch_models = {}

    def fit(self, returns):
        """Fit MS model first, then GARCH within each regime."""
        from utils.statistics import fit_garch11

        r = np.asarray(returns, dtype=np.float64)
        r = r[~np.isnan(r)]

        # Fit Markov switching
        self.ms_model.fit(r, verbose=True)

        # Get smoothed probabilities
        probs = self.ms_model.smoothed_probs

        # Fit GARCH on returns weighted by regime probability
        for regime in range(2):
            weights = probs[:, regime]
            # Use observations where regime probability > 0.5
            regime_mask = weights > 0.5
            if regime_mask.sum() > 50:
                regime_returns = r[regime_mask]
                self.garch_models[regime] = fit_garch11(regime_returns)
                print(f"  Regime {regime} GARCH: alpha={self.garch_models[regime]['alpha']:.4f}, "
                      f"beta={self.garch_models[regime]['beta']:.4f}")
            else:
                print(f"  Regime {regime}: insufficient data ({regime_mask.sum()} obs), using sample vol")
                self.garch_models[regime] = {
                    "conditional_vol": np.full(regime_mask.sum(), r[regime_mask].std() if regime_mask.sum() > 0 else r.std())
                }

        return self

    def conditional_vol(self, returns):
        """
        Compute conditional volatility as regime-probability-weighted GARCH vols.
        """
        from utils.numba_kernels import garch11_filter

        r = np.asarray(returns, dtype=np.float64)
        n = len(r)

        # Get regime-specific GARCH vols
        vol_series = np.zeros(n)
        for regime in range(2):
            m = self.garch_models.get(regime, {})
            if "omega" in m:
                sigma2 = garch11_filter(r, m["omega"], m["alpha"], m["beta"])
                vol_r = np.sqrt(sigma2)
            else:
                vol_r = np.full(n, np.std(r))

            probs = self.ms_model.smoothed_probs
            # Handle size mismatch
            min_len = min(len(vol_r), len(probs))
            vol_series[:min_len] += probs[:min_len, regime] * vol_r[:min_len]

        return vol_series
