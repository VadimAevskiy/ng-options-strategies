"""
Black-Scholes model
====================
Analytical pricing, full Greek chain (up to 3rd order), and IV.
"""
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.numba_kernels import (bs_price_scalar, bs_greeks_scalar,
                                  bs_greeks_vectorised, implied_vol_scalar,
                                  implied_vol_vectorised)


class BlackScholes:
    """
    Black-Scholes option pricing model with full Greek chain.

    Greeks computed:
        1st order: delta, vega, theta, rho
        2nd order: gamma, vanna (d(delta)/d(sigma)), charm (d(delta)/d(t)),
                   volga/vomma (d(vega)/d(sigma))
        3rd order: speed (d(gamma)/d(S)), colour (d(gamma)/d(t)),
                   ultima (d(vomma)/d(sigma))
    """

    def __init__(self, r: float = 0.02):
        self.r = r

    def price(self, S: float, K: float, T: float, sigma: float,
              is_call: bool = True) -> float:
        """Scalar or array pricing."""
        if np.isscalar(S):
            return bs_price_scalar(float(S), float(K), float(T),
                                   self.r, float(sigma), is_call)
        S_arr = np.asarray(S, dtype=np.float64)
        K_arr = np.asarray(K, dtype=np.float64)
        T_arr = np.asarray(T, dtype=np.float64)
        sigma_arr = np.asarray(sigma, dtype=np.float64)
        is_call_arr = np.asarray(is_call, dtype=np.bool_)
        return bs_greeks_vectorised(S_arr, K_arr, T_arr, self.r,
                                    sigma_arr, is_call_arr)[:, 0]

    def greeks(self, S: float, K: float, T: float, sigma: float,
              is_call: bool = True) -> dict:
        """Full Greek chain for a single option."""
        S, K, T, sigma = float(S), float(K), float(T), float(sigma)
        if T <= 0 or sigma <= 0:
            return self._expired_greeks(S, K, is_call)

        sqrt_T = np.sqrt(T)
        d1 = (np.log(S / K) + (self.r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T
        Nd1 = norm.cdf(d1)
        Nd2 = norm.cdf(d2)
        nd1 = norm.pdf(d1)
        disc = np.exp(-self.r * T)

        # 1st order
        if is_call:
            price = S * Nd1 - K * disc * Nd2
            delta = Nd1
            theta = -(S * nd1 * sigma) / (2 * sqrt_T) - self.r * K * disc * Nd2
            rho = K * T * disc * Nd2
        else:
            price = K * disc * (1 - Nd2) - S * (1 - Nd1)
            delta = Nd1 - 1
            theta = -(S * nd1 * sigma) / (2 * sqrt_T) + self.r * K * disc * (1 - Nd2)
            rho = -K * T * disc * (1 - Nd2)

        gamma = nd1 / (S * sigma * sqrt_T)
        vega = S * nd1 * sqrt_T

        # 2nd order
        vanna = -nd1 * d2 / sigma          # d(delta)/d(sigma)
        charm_val = -nd1 * (2 * self.r * T - d2 * sigma * sqrt_T) / (2 * T * sigma * sqrt_T)
        volga = vega * d1 * d2 / sigma      # d(vega)/d(sigma) = vomma

        # 3rd order
        speed = -gamma / S * (d1 / (sigma * sqrt_T) + 1)
        colour = -nd1 / (2 * S * T * sigma * sqrt_T) * (
            2 * self.r * T - d2 * sigma * sqrt_T + (1 + d1 * d2) / (sigma * sqrt_T)
        ) if T > 0 else 0.0
        ultima = -vega / (sigma ** 2) * (d1 * d2 * (1 - d1 * d2) + d1 ** 2 + d2 ** 2)

        return dict(
            price=price, delta=delta, gamma=gamma, vega=vega,
            theta=theta, rho=rho,
            vanna=vanna, charm=charm_val, volga=volga,
            speed=speed, colour=colour, ultima=ultima,
            d1=d1, d2=d2,
        )

    def greeks_sv(self, S: float, K: float, T: float, sigma: float,
                  is_call: bool, rho_sv: float, vol_of_vol: float) -> dict:
        """
        Smile-adjusted Greeks using Bartlett's (2006) delta correction.

        BS delta assumes flat vol across spot levels. In reality, vol and spot
        are correlated. Bartlett's correction:
            Delta_SV = Delta_BS + Vega_BS * rho_sv * vol_of_vol / S

        For nat gas: rho_sv > 0 (vol rises when prices spike), so the SV delta
        differs from BS delta, reducing unnecessary hedge rebalancing.

        Parameters
        ----------
        rho_sv : float
            Correlation between spot returns and vol changes.
        vol_of_vol : float
            Annualised vol-of-vol (std of daily vol changes * sqrt(252)).
        """
        g = self.greeks(S, K, T, sigma, is_call)
        sv_correction = g["vega"] * rho_sv * vol_of_vol / S if S > 0 else 0
        g["delta_bs"] = g["delta"]
        g["delta"] = g["delta"] + sv_correction
        g["sv_correction"] = sv_correction
        return g

    def implied_vol(self, market_price, S, K, T, is_call=True):
        """Implied volatility via bisection (numba-accelerated)."""
        if np.isscalar(S):
            return implied_vol_scalar(float(market_price), float(S), float(K),
                                      float(T), self.r, is_call)
        return implied_vol_vectorised(
            np.asarray(market_price, dtype=np.float64),
            np.asarray(S, dtype=np.float64),
            np.asarray(K, dtype=np.float64),
            np.asarray(T, dtype=np.float64),
            self.r,
            np.asarray(is_call, dtype=np.bool_),
        )

    def _expired_greeks(self, S, K, is_call):
        intrinsic = max(S - K, 0) if is_call else max(K - S, 0)
        d = 1.0 if (is_call and S > K) else (-1.0 if (not is_call and S < K) else 0.0)
        return dict(price=intrinsic, delta=d, gamma=0, vega=0,
                    theta=0, rho=0, vanna=0, charm=0, volga=0,
                    speed=0, colour=0, ultima=0, d1=np.nan, d2=np.nan)

    def payoff_diagram(self, S_range, K, T, sigma, is_call=True, premium=None):
        """Generate payoff/PnL diagram data."""
        prices = np.array([bs_price_scalar(s, K, T, self.r, sigma, is_call) for s in S_range])
        intrinsic = np.maximum(S_range - K, 0) if is_call else np.maximum(K - S_range, 0)
        if premium is None:
            premium = bs_price_scalar(S_range[len(S_range) // 2], K, T, self.r, sigma, is_call)
        pnl = intrinsic - premium
        return dict(spot=S_range, price=prices, intrinsic=intrinsic, pnl=pnl)
