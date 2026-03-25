"""
Numba-accelerated computational kernels
========================================
JIT-compiled inner loops for BS pricing, IV solver, GARCH filter, and realised vol.
Falls back to plain NumPy if Numba is unavailable.
"""
import numpy as np

try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # Provide pass-through decorators so code still runs (slower)
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator
    prange = range


# ═══════════════════════════════════════════════════════════════════════════
#  Black-Scholes kernel
# ═══════════════════════════════════════════════════════════════════════════
@njit(cache=True)
def _norm_cdf(x):
    """Abramowitz & Stegun approximation to cumulative normal."""
    a1, a2, a3, a4, a5 = (0.254829592, -0.284496736,
                           1.421413741, -1.453152027, 1.061405429)
    p = 0.3275911
    sign = 1.0 if x >= 0 else -1.0
    x_abs = abs(x)
    t = 1.0 / (1.0 + p * x_abs)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x_abs * x_abs / 2.0)
    return 0.5 * (1.0 + sign * y)


@njit(cache=True)
def _norm_pdf(x):
    return np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)


@njit(cache=True)
def bs_price_scalar(S, K, T, r, sigma, is_call):
    """Black-Scholes price for a single option (scalar inputs)."""
    if T <= 0.0 or sigma <= 0.0:
        if is_call:
            return max(S - K, 0.0)
        else:
            return max(K - S, 0.0)
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    if is_call:
        return S * _norm_cdf(d1) - K * np.exp(-r * T) * _norm_cdf(d2)
    else:
        return K * np.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)


@njit(cache=True)
def bs_greeks_scalar(S, K, T, r, sigma, is_call):
    """
    Returns (price, delta, gamma, vega, theta, rho) for a single option.
    Vega and theta are per-unit (not per-cent or per-day).
    """
    if T <= 1e-10 or sigma <= 1e-10:
        intrinsic = max(S - K, 0.0) if is_call else max(K - S, 0.0)
        delta = 1.0 if (is_call and S > K) else (-1.0 if (not is_call and S < K) else 0.0)
        return intrinsic, delta, 0.0, 0.0, 0.0, 0.0

    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    nd1 = _norm_cdf(d1)
    nd2 = _norm_cdf(d2)
    pdf_d1 = _norm_pdf(d1)
    disc = np.exp(-r * T)

    if is_call:
        price = S * nd1 - K * disc * nd2
        delta = nd1
        theta = (-(S * pdf_d1 * sigma) / (2.0 * sqrt_T)
                 - r * K * disc * nd2)
    else:
        price = K * disc * (1.0 - nd2) - S * (1.0 - nd1)
        delta = nd1 - 1.0
        theta = (-(S * pdf_d1 * sigma) / (2.0 * sqrt_T)
                 + r * K * disc * (1.0 - nd2))

    gamma = pdf_d1 / (S * sigma * sqrt_T)
    vega = S * pdf_d1 * sqrt_T
    rho = K * T * disc * nd2 if is_call else -K * T * disc * (1.0 - nd2)

    return price, delta, gamma, vega, theta, rho


@njit(parallel=True, cache=True)
def bs_greeks_vectorised(S_arr, K_arr, T_arr, r, sigma_arr, is_call_arr):
    """
    Vectorised Greeks computation across arrays.
    Returns (N, 6) array: [price, delta, gamma, vega, theta, rho].
    """
    n = len(S_arr)
    result = np.empty((n, 6), dtype=np.float64)
    for i in prange(n):
        p, d, g, v, th, rh = bs_greeks_scalar(
            S_arr[i], K_arr[i], T_arr[i], r, sigma_arr[i], is_call_arr[i]
        )
        result[i, 0] = p
        result[i, 1] = d
        result[i, 2] = g
        result[i, 3] = v
        result[i, 4] = th
        result[i, 5] = rh
    return result


# ═══════════════════════════════════════════════════════════════════════════
#  Implied Volatility (Brent's method, fully numba-compiled)
# ═══════════════════════════════════════════════════════════════════════════
@njit(cache=True)
def implied_vol_scalar(price_mkt, S, K, T, r, is_call,
                       tol=1e-8, max_iter=100):
    """Brent's method for implied volatility on a single option."""
    if T <= 0 or price_mkt <= 0:
        return np.nan

    # Bracket: [sigma_lo, sigma_hi]
    sigma_lo, sigma_hi = 0.001, 5.0
    f_lo = bs_price_scalar(S, K, T, r, sigma_lo, is_call) - price_mkt
    f_hi = bs_price_scalar(S, K, T, r, sigma_hi, is_call) - price_mkt

    if f_lo * f_hi > 0:
        return np.nan  # No root in bracket

    # Simple bisection (reliable in numba)
    for _ in range(max_iter):
        sigma_mid = 0.5 * (sigma_lo + sigma_hi)
        f_mid = bs_price_scalar(S, K, T, r, sigma_mid, is_call) - price_mkt
        if abs(f_mid) < tol or (sigma_hi - sigma_lo) < tol:
            return sigma_mid
        if f_mid * f_lo < 0:
            sigma_hi = sigma_mid
            f_hi = f_mid
        else:
            sigma_lo = sigma_mid
            f_lo = f_mid
    return 0.5 * (sigma_lo + sigma_hi)


@njit(parallel=True, cache=True)
def implied_vol_vectorised(prices, S_arr, K_arr, T_arr, r, is_call_arr):
    """Vectorised implied vol across arrays. Uses numba prange when available."""
    n = len(prices)
    result = np.empty(n, dtype=np.float64)
    for i in prange(n):
        result[i] = implied_vol_scalar(
            prices[i], S_arr[i], K_arr[i], T_arr[i], r, is_call_arr[i]
        )
    return result


def implied_vol_numpy(prices, S_arr, K_arr, T_arr, r, is_call_arr, max_iter=60):
    """
    Pure-numpy vectorised bisection IV solver. No loops over options.
    Processes ALL options simultaneously at each bisection step.
    ~10x faster than scalar loop without numba.
    """
    n = len(prices)
    sig_lo = np.full(n, 0.001)
    sig_hi = np.full(n, 5.0)
    valid = (T_arr > 0) & (prices > 0) & (S_arr > 0) & (K_arr > 0)
    result = np.full(n, np.nan)

    idx = np.where(valid)[0]
    if len(idx) == 0:
        return result

    p = prices[idx]; S = S_arr[idx]; K = K_arr[idx]; T = T_arr[idx]
    ic = is_call_arr[idx]
    lo = sig_lo[idx]; hi = sig_hi[idx]

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        # Vectorised BS price
        d1 = (np.log(S / K) + (r + 0.5 * mid * mid) * T) / (mid * np.sqrt(T) + 1e-15)
        d2 = d1 - mid * np.sqrt(T)
        from scipy.stats import norm as ndist
        Nd1 = ndist.cdf(d1); Nd2 = ndist.cdf(d2)
        call_price = S * Nd1 - K * np.exp(-r * T) * Nd2
        put_price = call_price - S + K * np.exp(-r * T)
        bs_p = np.where(ic, call_price, put_price)
        f_mid = bs_p - p

        mask_lo = f_mid < 0
        lo = np.where(mask_lo, mid, lo)
        hi = np.where(~mask_lo, mid, hi)

        if np.all((hi - lo) < 1e-6):
            break

    result[idx] = 0.5 * (lo + hi)
    result[idx[np.abs(result[idx]) < 1e-6]] = np.nan
    return result


# Auto-select best implementation
if not HAS_NUMBA:
    # Override with numpy version when numba is absent
    _original_iv_vec = implied_vol_vectorised
    implied_vol_vectorised = implied_vol_numpy


# ═══════════════════════════════════════════════════════════════════════════
#  GARCH(1,1) estimation (simple MLE via scipy later, kernel here)
# ═══════════════════════════════════════════════════════════════════════════
@njit(cache=True)
def garch11_filter(returns, omega, alpha, beta):
    """
    Compute GARCH(1,1) conditional variance series.
    sigma2_t = omega + alpha * r_{t-1}^2 + beta * sigma2_{t-1}
    Returns sigma2 array of same length as returns.
    """
    n = len(returns)
    sigma2 = np.empty(n, dtype=np.float64)
    sigma2[0] = omega / (1.0 - alpha - beta)  # Unconditional variance
    for t in range(1, n):
        sigma2[t] = omega + alpha * returns[t - 1] ** 2 + beta * sigma2[t - 1]
    return sigma2


@njit(cache=True)
def garch11_loglik(returns, omega, alpha, beta):
    """Negative log-likelihood for GARCH(1,1) with normal innovations."""
    sigma2 = garch11_filter(returns, omega, alpha, beta)
    n = len(returns)
    ll = 0.0
    for t in range(n):
        if sigma2[t] <= 0:
            return 1e15  # Penalty
        ll += -0.5 * (np.log(2.0 * np.pi) + np.log(sigma2[t])
                       + returns[t] ** 2 / sigma2[t])
    return -ll  # Return negative for minimisation


@njit(cache=True)
def gjr_garch_filter(returns, omega, alpha, gamma, beta):
    """
    GJR-GARCH(1,1) filter (leverage effect).
    sigma2_t = omega + (alpha + gamma * I_{r<0}) * r_{t-1}^2 + beta * sigma2_{t-1}
    """
    n = len(returns)
    sigma2 = np.empty(n, dtype=np.float64)
    sigma2[0] = omega / (1.0 - alpha - 0.5 * gamma - beta)
    for t in range(1, n):
        indicator = 1.0 if returns[t - 1] < 0 else 0.0
        sigma2[t] = (omega
                     + (alpha + gamma * indicator) * returns[t - 1] ** 2
                     + beta * sigma2[t - 1])
    return sigma2


@njit(cache=True)
def egarch_filter(returns, omega, alpha, gamma, beta):
    """
    EGARCH(1,1) filter.
    log(sigma2_t) = omega + alpha * |z_{t-1}| + gamma * z_{t-1} + beta * log(sigma2_{t-1})
    where z_t = r_t / sigma_t
    """
    n = len(returns)
    log_sigma2 = np.empty(n, dtype=np.float64)
    log_sigma2[0] = omega / (1.0 - beta)
    for t in range(1, n):
        sig_prev = np.sqrt(np.exp(log_sigma2[t - 1]))
        z = returns[t - 1] / sig_prev if sig_prev > 1e-12 else 0.0
        log_sigma2[t] = omega + alpha * (abs(z) - np.sqrt(2.0 / np.pi)) + gamma * z + beta * log_sigma2[t - 1]
    sigma2 = np.empty(n, dtype=np.float64)
    for t in range(n):
        sigma2[t] = np.exp(log_sigma2[t])
    return sigma2


# ═══════════════════════════════════════════════════════════════════════════
#  Realised volatility estimators
# ═══════════════════════════════════════════════════════════════════════════
@njit(cache=True)
def realised_vol_close_to_close(log_returns, window):
    """Rolling close-to-close realised volatility (annualised, 252 days)."""
    n = len(log_returns)
    rv = np.full(n, np.nan)
    for i in range(window - 1, n):
        s = 0.0
        for j in range(i - window + 1, i + 1):
            s += log_returns[j] ** 2
        rv[i] = np.sqrt(s / window * 252.0)
    return rv


@njit(cache=True)
def parkinson_vol(high, low, window):
    """
    Parkinson (1980) high-low volatility estimator.
    sigma^2 = 1/(4*ln(2)) * (ln(H/L))^2, annualised.
    """
    n = len(high)
    rv = np.full(n, np.nan)
    factor = 1.0 / (4.0 * np.log(2.0))
    for i in range(window - 1, n):
        s = 0.0
        for j in range(i - window + 1, i + 1):
            if low[j] > 0 and high[j] > 0:
                hl = np.log(high[j] / low[j])
                s += factor * hl * hl
        rv[i] = np.sqrt(s / window * 252.0)
    return rv
