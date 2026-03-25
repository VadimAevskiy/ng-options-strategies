"""
Statistical utilities
=====================
Custom implementations of stationarity and distribution tests,
with fallback when statsmodels / arch are unavailable.
"""
import numpy as np
from scipy import stats
from scipy.optimize import minimize
import warnings


# ═══════════════════════════════════════════════════════════════════════════
#  ADF test (via statsmodels or manual)
# ═══════════════════════════════════════════════════════════════════════════
def adf_test(series, maxlag=None, regression="c"):
    """Augmented Dickey-Fuller test. Returns dict with statistic, p-value, etc."""
    try:
        from statsmodels.tsa.stattools import adfuller
        res = adfuller(series.dropna(), maxlag=maxlag, regression=regression, autolag="AIC")
        return dict(statistic=res[0], pvalue=res[1], usedlag=res[2],
                    nobs=res[3], critical_values=res[4])
    except ImportError:
        return _manual_adf(np.asarray(series.dropna(), dtype=np.float64))


def _manual_adf(y):
    """Simplified ADF: ΔY_t = α + β*Y_{t-1} + ε_t  (no lags, constant only)."""
    dy = np.diff(y)
    y_lag = y[:-1]
    X = np.column_stack([np.ones(len(y_lag)), y_lag])
    beta_hat = np.linalg.lstsq(X, dy, rcond=None)[0]
    resid = dy - X @ beta_hat
    se = np.sqrt(np.sum(resid ** 2) / (len(dy) - 2) / np.sum((y_lag - y_lag.mean()) ** 2))
    t_stat = beta_hat[1] / se
    # Rough p-value mapping (MacKinnon critical values for n>250, constant)
    cv = {1: -3.43, 5: -2.86, 10: -2.57}
    if t_stat < cv[1]:
        p = 0.001
    elif t_stat < cv[5]:
        p = 0.03
    elif t_stat < cv[10]:
        p = 0.08
    else:
        p = 0.5
    return dict(statistic=t_stat, pvalue=p, usedlag=0,
                nobs=len(dy), critical_values={"1%": cv[1], "5%": cv[5], "10%": cv[10]})


# ═══════════════════════════════════════════════════════════════════════════
#  KPSS test
# ═══════════════════════════════════════════════════════════════════════════
def kpss_test(series, regression="c", nlags="auto"):
    """
    Kwiatkowski-Phillips-Schmidt-Shin stationarity test.
    H0: series is stationary.
    """
    try:
        from statsmodels.tsa.stattools import kpss as _kpss
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stat, p, lags, cv = _kpss(series.dropna(), regression=regression, nlags=nlags)
        return dict(statistic=stat, pvalue=p, usedlag=lags, critical_values=cv)
    except ImportError:
        return _manual_kpss(np.asarray(series.dropna(), dtype=np.float64))


def _manual_kpss(y):
    """Simplified KPSS with Bartlett kernel."""
    n = len(y)
    ybar = y.mean()
    e = y - ybar
    S = np.cumsum(e)
    # Long-run variance (Bartlett kernel)
    nlags = int(np.sqrt(n))
    gamma0 = np.sum(e ** 2) / n
    lrv = gamma0
    for j in range(1, nlags + 1):
        w = 1.0 - j / (nlags + 1)
        gj = np.sum(e[j:] * e[:-j]) / n
        lrv += 2.0 * w * gj
    stat = np.sum(S ** 2) / (n ** 2 * lrv)
    cv = {"10%": 0.347, "5%": 0.463, "2.5%": 0.574, "1%": 0.739}
    if stat > cv["1%"]:
        p = 0.005
    elif stat > cv["5%"]:
        p = 0.03
    elif stat > cv["10%"]:
        p = 0.08
    else:
        p = 0.15
    return dict(statistic=stat, pvalue=p, usedlag=nlags, critical_values=cv)


# ═══════════════════════════════════════════════════════════════════════════
#  Phillips-Perron test
# ═══════════════════════════════════════════════════════════════════════════
def pp_test(series, regression="c"):
    """Phillips-Perron unit-root test via statsmodels or manual fallback."""
    try:
        from statsmodels.tsa.stattools import adfuller
        # PP test: use ADF with 0 lags and Newey-West correction
        res = adfuller(series.dropna(), maxlag=0, regression=regression)
        return dict(statistic=res[0], pvalue=res[1], nobs=res[3],
                    critical_values=res[4], note="Approximated via ADF(0)")
    except ImportError:
        return _manual_adf(np.asarray(series.dropna(), dtype=np.float64))


# ═══════════════════════════════════════════════════════════════════════════
#  Distribution tests
# ═══════════════════════════════════════════════════════════════════════════
def jarque_bera_test(series):
    """Jarque-Bera normality test."""
    x = np.asarray(series.dropna(), dtype=np.float64)
    n = len(x)
    skew = stats.skew(x)
    kurt = stats.kurtosis(x)  # Excess kurtosis
    jb = n / 6.0 * (skew ** 2 + kurt ** 2 / 4.0)
    p = 1.0 - stats.chi2.cdf(jb, 2)
    return dict(statistic=jb, pvalue=p, skewness=skew, kurtosis=kurt)


def ks_test(series, dist="norm"):
    """Kolmogorov-Smirnov test against a fitted distribution."""
    x = np.asarray(series.dropna(), dtype=np.float64)
    if dist == "norm":
        params = stats.norm.fit(x)
        stat, p = stats.kstest(x, "norm", args=params)
    elif dist == "t":
        params = stats.t.fit(x)
        stat, p = stats.kstest(x, "t", args=params)
    else:
        stat, p = stats.kstest(x, dist)
    return dict(statistic=stat, pvalue=p, distribution=dist)


def anderson_darling_test(series):
    """Anderson-Darling normality test."""
    x = np.asarray(series.dropna(), dtype=np.float64)
    res = stats.anderson(x, dist="norm")
    return dict(statistic=res.statistic,
                critical_values=dict(zip(res.significance_level, res.critical_values)))


# ═══════════════════════════════════════════════════════════════════════════
#  Ljung-Box test (for autocorrelation / ARCH effects)
# ═══════════════════════════════════════════════════════════════════════════
def ljung_box_test(series, lags=20):
    """Ljung-Box Q test for serial correlation."""
    try:
        from statsmodels.stats.diagnostic import acorr_ljungbox
        res = acorr_ljungbox(series.dropna(), lags=lags, return_df=True)
        return res
    except ImportError:
        return _manual_ljung_box(np.asarray(series.dropna(), dtype=np.float64), lags)


def _manual_ljung_box(x, lags):
    """Manual Ljung-Box."""
    n = len(x)
    xbar = x.mean()
    gamma0 = np.sum((x - xbar) ** 2) / n
    Q = 0.0
    results = {}
    for k in range(1, lags + 1):
        rk = np.sum((x[k:] - xbar) * (x[:-k] - xbar)) / (n * gamma0)
        Q += rk ** 2 / (n - k)
        Q_stat = n * (n + 2) * Q
        p = 1.0 - stats.chi2.cdf(Q_stat, k)
        results[k] = dict(lb_stat=Q_stat, lb_pvalue=p)
    return results


# ═══════════════════════════════════════════════════════════════════════════
#  Granger causality
# ═══════════════════════════════════════════════════════════════════════════
def granger_causality_test(y, x, maxlag=5):
    """
    Granger causality: does x Granger-cause y?
    Uses F-test on restricted vs unrestricted regression.
    """
    try:
        from statsmodels.tsa.stattools import grangercausalitytests
        import pandas as pd
        data = pd.DataFrame({"y": y, "x": x}).dropna()
        results = grangercausalitytests(data[["y", "x"]], maxlag=maxlag, verbose=False)
        summary = {}
        for lag, res in results.items():
            f_stat = res[0]["ssr_ftest"][0]
            p_val = res[0]["ssr_ftest"][1]
            summary[lag] = dict(f_statistic=f_stat, pvalue=p_val)
        return summary
    except ImportError:
        return _manual_granger(np.asarray(y, dtype=np.float64),
                               np.asarray(x, dtype=np.float64), maxlag)


def _manual_granger(y, x, maxlag):
    """Manual Granger causality via OLS F-test."""
    n = len(y)
    results = {}
    for lag in range(1, maxlag + 1):
        # Restricted model: y_t ~ y_{t-1}, ..., y_{t-lag}
        Y = y[lag:]
        X_r = np.column_stack([y[lag - j - 1:n - j - 1] for j in range(lag)])
        X_r = np.column_stack([np.ones(len(Y)), X_r])
        beta_r = np.linalg.lstsq(X_r, Y, rcond=None)[0]
        ssr_r = np.sum((Y - X_r @ beta_r) ** 2)

        # Unrestricted: add lags of x
        X_u = np.column_stack([X_r] + [x[lag - j - 1:n - j - 1] for j in range(lag)])
        beta_u = np.linalg.lstsq(X_u, Y, rcond=None)[0]
        ssr_u = np.sum((Y - X_u @ beta_u) ** 2)

        df1 = lag
        df2 = len(Y) - 2 * lag - 1
        f_stat = ((ssr_r - ssr_u) / df1) / (ssr_u / df2) if df2 > 0 else 0
        p_val = 1.0 - stats.f.cdf(f_stat, df1, df2) if df2 > 0 else 1.0
        results[lag] = dict(f_statistic=f_stat, pvalue=p_val)
    return results


# ═══════════════════════════════════════════════════════════════════════════
#  GARCH estimation (scipy-based)
# ═══════════════════════════════════════════════════════════════════════════
def fit_garch11(returns):
    """
    Fit GARCH(1,1) by MLE. Returns dict with params and diagnostics.
    Uses arch library if available, otherwise scipy.optimize.
    """
    try:
        from arch import arch_model
        am = arch_model(returns * 100, vol="GARCH", p=1, q=1, dist="normal", mean="Zero")
        res = am.fit(disp="off")
        return dict(
            omega=res.params["omega"] / 1e4,
            alpha=res.params["alpha[1]"],
            beta=res.params["beta[1]"],
            loglikelihood=res.loglikelihood,
            aic=res.aic,
            bic=res.bic,
            conditional_vol=res.conditional_volatility / 100.0,
            model_type="GARCH(1,1)",
            source="arch"
        )
    except ImportError:
        return _fit_garch11_scipy(np.asarray(returns, dtype=np.float64))


def _fit_garch11_scipy(returns):
    """GARCH(1,1) via scipy minimisation of negative log-likelihood."""
    from utils.numba_kernels import garch11_loglik, garch11_filter

    var_sample = np.var(returns)
    x0 = np.array([var_sample * 0.05, 0.08, 0.88])  # omega, alpha, beta

    def objective(params):
        omega, alpha, beta = params
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1.0:
            return 1e15
        return garch11_loglik(returns, omega, alpha, beta)

    bounds = [(1e-8, 1.0), (1e-8, 0.999), (1e-8, 0.999)]
    constraints = [{"type": "ineq", "fun": lambda p: 0.9999 - p[1] - p[2]}]

    res = minimize(objective, x0, method="SLSQP", bounds=bounds,
                   constraints=constraints, options={"maxiter": 500})

    omega, alpha, beta = res.x
    sigma2 = garch11_filter(returns, omega, alpha, beta)
    n = len(returns)
    ll = -res.fun
    k = 3
    aic = -2 * ll + 2 * k
    bic = -2 * ll + k * np.log(n)

    return dict(
        omega=omega, alpha=alpha, beta=beta,
        loglikelihood=ll, aic=aic, bic=bic,
        conditional_vol=np.sqrt(sigma2),
        model_type="GARCH(1,1)",
        source="scipy"
    )


def fit_gjr_garch(returns):
    """Fit GJR-GARCH(1,1) via arch or scipy."""
    try:
        from arch import arch_model
        am = arch_model(returns * 100, vol="GARCH", p=1, o=1, q=1, dist="normal", mean="Zero")
        res = am.fit(disp="off")
        return dict(
            omega=res.params.get("omega", 0) / 1e4,
            alpha=res.params.get("alpha[1]", 0),
            gamma=res.params.get("gamma[1]", 0),
            beta=res.params.get("beta[1]", 0),
            loglikelihood=res.loglikelihood,
            aic=res.aic, bic=res.bic,
            conditional_vol=res.conditional_volatility / 100.0,
            model_type="GJR-GARCH(1,1)",
            source="arch"
        )
    except ImportError:
        return _fit_gjr_scipy(np.asarray(returns, dtype=np.float64))


def _fit_gjr_scipy(returns):
    """GJR-GARCH(1,1) via scipy."""
    from utils.numba_kernels import gjr_garch_filter

    var_sample = np.var(returns)
    x0 = np.array([var_sample * 0.05, 0.05, 0.05, 0.88])

    def neg_ll(params):
        omega, alpha, gamma, beta = params
        if omega <= 0 or alpha < 0 or gamma < 0 or beta < 0:
            return 1e15
        if alpha + 0.5 * gamma + beta >= 1.0:
            return 1e15
        sigma2 = gjr_garch_filter(returns, omega, alpha, gamma, beta)
        ll = -0.5 * np.sum(np.log(2 * np.pi) + np.log(sigma2 + 1e-20) + returns ** 2 / (sigma2 + 1e-20))
        return -ll

    bounds = [(1e-8, 1.0)] * 4
    res = minimize(neg_ll, x0, method="SLSQP", bounds=bounds, options={"maxiter": 500})
    omega, alpha, gamma, beta = res.x
    sigma2 = gjr_garch_filter(returns, omega, alpha, gamma, beta)
    n = len(returns)
    ll = -res.fun
    k = 4
    return dict(
        omega=omega, alpha=alpha, gamma=gamma, beta=beta,
        loglikelihood=ll,
        aic=-2 * ll + 2 * k,
        bic=-2 * ll + k * np.log(n),
        conditional_vol=np.sqrt(sigma2),
        model_type="GJR-GARCH(1,1)",
        source="scipy"
    )


def fit_egarch(returns):
    """Fit EGARCH(1,1)."""
    try:
        from arch import arch_model
        am = arch_model(returns * 100, vol="EGARCH", p=1, q=1, dist="normal", mean="Zero")
        res = am.fit(disp="off")
        return dict(
            omega=res.params.get("omega", 0),
            alpha=res.params.get("alpha[1]", 0),
            gamma=res.params.get("gamma[1]", 0),
            beta=res.params.get("beta[1]", 0),
            loglikelihood=res.loglikelihood,
            aic=res.aic, bic=res.bic,
            conditional_vol=res.conditional_volatility / 100.0,
            model_type="EGARCH(1,1)",
            source="arch"
        )
    except ImportError:
        return _fit_egarch_scipy(np.asarray(returns, dtype=np.float64))


def _fit_egarch_scipy(returns):
    """EGARCH(1,1) via scipy."""
    from utils.numba_kernels import egarch_filter

    x0 = np.array([-0.1, 0.1, -0.05, 0.95])

    def neg_ll(params):
        omega, alpha, gamma, beta = params
        if abs(beta) >= 1.0:
            return 1e15
        sigma2 = egarch_filter(returns, omega, alpha, gamma, beta)
        if np.any(sigma2 <= 0):
            return 1e15
        ll = -0.5 * np.sum(np.log(2 * np.pi) + np.log(sigma2) + returns ** 2 / sigma2)
        return -ll

    bounds = [(-10, 10), (-2, 2), (-2, 2), (-0.9999, 0.9999)]
    res = minimize(neg_ll, x0, method="SLSQP", bounds=bounds, options={"maxiter": 500})
    omega, alpha, gamma, beta = res.x
    sigma2 = egarch_filter(returns, omega, alpha, gamma, beta)
    n = len(returns)
    ll = -res.fun
    k = 4
    return dict(
        omega=omega, alpha=alpha, gamma=gamma, beta=beta,
        loglikelihood=ll,
        aic=-2 * ll + 2 * k,
        bic=-2 * ll + k * np.log(n),
        conditional_vol=np.sqrt(sigma2),
        model_type="EGARCH(1,1)",
        source="scipy"
    )
