"""
GARCH volatility models
========================
Multiple GARCH specifications for conditional volatility estimation
and forecasting. Each model produces its own vol forecast that feeds
into option pricing and Greeks.

Models:
  - GARCH(1,1)
  - GJR-GARCH(1,1) (asymmetric / leverage)
  - EGARCH(1,1) (exponential)
  - Component GARCH (short-run + long-run)
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.numba_kernels import garch11_filter, gjr_garch_filter, egarch_filter
from utils.statistics import fit_garch11, fit_gjr_garch, fit_egarch


class GARCHSuite:
    """
    Unified interface for fitting multiple GARCH specifications,
    comparing them via information criteria, and generating forecasts.
    """

    def __init__(self, returns: "np.ndarray"):
        """
        Parameters
        ----------
        returns : array-like
            Log returns series (not %).
        """
        self.returns = np.asarray(returns, dtype=np.float64)
        self.returns = self.returns[~np.isnan(self.returns)]
        self.models = {}
        self.best_model = None

    def fit_all(self) -> "GARCHSuite":
        """Fit GARCH(1,1), GJR-GARCH(1,1), and EGARCH(1,1)."""
        print("Fitting GARCH(1,1)...")
        self.models["GARCH"] = fit_garch11(self.returns)

        print("Fitting GJR-GARCH(1,1)...")
        self.models["GJR-GARCH"] = fit_gjr_garch(self.returns)

        print("Fitting EGARCH(1,1)...")
        self.models["EGARCH"] = fit_egarch(self.returns)

        # Fit Component GARCH manually
        print("Fitting Component-GARCH...")
        self.models["CGARCH"] = self._fit_component_garch()

        # Select best model by BIC
        bics = {name: m.get("bic", np.inf) for name, m in self.models.items()}
        self.best_model = min(bics, key=bics.get)
        print(f"\nBest model by BIC: {self.best_model}")

        return self.models

    def _fit_component_garch(self):
        """
        Component GARCH (Engle & Lee, 1999):
        q_t = omega + rho * q_{t-1} + phi * (r_{t-1}^2 - sigma2_{t-1})
        sigma2_t = q_t + alpha * (r_{t-1}^2 - q_{t-1}) + beta * (sigma2_{t-1} - q_{t-1})
        """
        r = self.returns
        var_sample = np.var(r)

        def component_filter(params):
            omega, rho, phi, alpha, beta = params
            n = len(r)
            q = np.empty(n)
            sigma2 = np.empty(n)
            q[0] = var_sample
            sigma2[0] = var_sample
            for t in range(1, n):
                q[t] = omega + rho * q[t - 1] + phi * (r[t - 1] ** 2 - sigma2[t - 1])
                sigma2[t] = q[t] + alpha * (r[t - 1] ** 2 - q[t - 1]) + beta * (sigma2[t - 1] - q[t - 1])
                sigma2[t] = max(sigma2[t], 1e-12)
            return sigma2

        def neg_ll(params):
            omega, rho, phi, alpha, beta = params
            if omega < 0 or alpha < 0 or beta < 0 or rho < 0 or rho > 1:
                return 1e15
            if alpha + beta >= 1.0:
                return 1e15
            try:
                sigma2 = component_filter(params)
                if np.any(sigma2 <= 0) or np.any(np.isnan(sigma2)):
                    return 1e15
                ll = -0.5 * np.sum(np.log(2 * np.pi) + np.log(sigma2) + r ** 2 / sigma2)
                return -ll
            except Exception:
                return 1e15

        x0 = [var_sample * 0.01, 0.95, 0.04, 0.05, 0.90]
        bounds = [(1e-10, 0.1), (0.5, 0.9999), (0.001, 0.5), (0.001, 0.5), (0.5, 0.9999)]

        res = minimize(neg_ll, x0, method="SLSQP", bounds=bounds, options={"maxiter": 500})
        sigma2 = component_filter(res.x)
        n = len(r)
        ll = -res.fun
        k = 5
        return dict(
            omega=res.x[0], rho=res.x[1], phi=res.x[2],
            alpha=res.x[3], beta=res.x[4],
            loglikelihood=ll,
            aic=-2 * ll + 2 * k,
            bic=-2 * ll + k * np.log(n),
            conditional_vol=np.sqrt(sigma2),
            model_type="Component-GARCH",
            source="scipy"
        )

    def comparison_table(self):
        """Return DataFrame comparing all fitted models."""
        rows = []
        for name, m in self.models.items():
            row = {"Model": name}
            for key in ["omega", "alpha", "beta", "gamma", "rho", "phi"]:
                if key in m:
                    row[key] = m[key]
            row["LogLik"] = m.get("loglikelihood", np.nan)
            row["AIC"] = m.get("aic", np.nan)
            row["BIC"] = m.get("bic", np.nan)
            rows.append(row)
        return pd.DataFrame(rows).set_index("Model")

    def forecast(self, model_name=None, horizon=20):
        """
        Multi-step variance forecast.
        For GARCH(1,1): E[sigma2_{t+h}] = omega/(1-a-b) + (a+b)^h * (sigma2_t - omega/(1-a-b))
        """
        if model_name is None:
            model_name = self.best_model
        m = self.models[model_name]
        cond_vol = m["conditional_vol"]
        last_var = cond_vol[-1] ** 2
        last_ret2 = self.returns[-1] ** 2

        forecasts = np.empty(horizon)

        if model_name in ("GARCH", "GJR-GARCH"):
            omega = m["omega"]
            alpha = m["alpha"]
            beta = m["beta"]
            gamma = m.get("gamma", 0)
            uncond_var = omega / (1 - alpha - 0.5 * gamma - beta)

            var_t = omega + alpha * last_ret2 + beta * last_var
            if gamma > 0 and self.returns[-1] < 0:
                var_t += gamma * last_ret2

            persistence = alpha + 0.5 * gamma + beta
            for h in range(horizon):
                forecasts[h] = uncond_var + persistence ** (h + 1) * (var_t - uncond_var)
        else:
            # Simple persistence-based forecast
            for h in range(horizon):
                forecasts[h] = last_var  # Flat forecast as approximation

        return np.sqrt(forecasts)  # Return vol, not variance

    def get_conditional_vol(self, model_name=None):
        """Get conditional volatility series for a given model."""
        if model_name is None:
            model_name = self.best_model
        return self.models[model_name]["conditional_vol"]
