"""
State-Space Markov-Switching Models with Kim Filter
=====================================================
Implementation of Kim (1994) filtering and smoothing for state-space
models with Markov-switching parameters.

Follows the R kimfilter package logic (Aevskiy reference code):
  State eq:   a_t = D(s_t) + F(s_t) a_{t-1} + eta_t,  eta ~ N(0, Q(s_t))
  Obs eq:     y_t = A(s_t) + H(s_t) a_t   + eps_t,  eps ~ N(0, R(s_t))
  Regimes:    P(s_t = j | s_{t-1} = i) = P_ij

The Kim filter combines:
  1. Hamilton (1989) forward filter for regime probabilities
  2. Kalman (1960) filter for state estimation within each regime
  3. Kim (1994) collapsing step to keep computation tractable (S^2 -> S)

Models:
  A. MarkovSwitchingAR   -- MS-AR(p) with switching mean/variance
  B. MarkovSwitchingUC   -- Unobserved Components: trend + cycle with switching vol
  C. MarkovSwitchingDCF  -- Dynamic Common Factor (Aevskiy / Kim-Yoo style)
"""
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm as norm_dist
import warnings

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =========================================================================
#  Kim Filter Core
# =========================================================================

class KimFilter:
    """
    Kim (1994) approximate filter for state-space models with
    Markov-switching parameters.

    Per time step t, for each regime pair (i, j):
      1. Kalman prediction -> a_{t|t-1}^{ij}, P_{t|t-1}^{ij}
      2. Innovation        -> v_t^{ij}, S_t^{ij}
      3. Kalman update     -> a_{t|t}^{ij}, P_{t|t}^{ij}
      4. Regime lik        -> f(y_t | s_t=j, s_{t-1}=i)
    Hamilton step:
      5. Joint prob        -> P(s_t=j, s_{t-1}=i | Y_t)
      6. Marginal prob     -> P(s_t=j | Y_t)
    Kim collapsing:
      7. Collapse S^2 Gaussians to S Gaussians for tractability
    """

    def __init__(self, n_states, n_obs, n_state_vars):
        self.S = n_states
        self.p = n_obs
        self.m = n_state_vars

    def filter(self, y, F, H, Q, R, D, A, Pm, B0, P0, **kwargs):
        S, m, p = self.S, self.m, self.p
        T = y.shape[0]

        F = self._ensure_3d(F, m, m)
        H = self._ensure_3d(H, p, m)
        Q = self._ensure_3d(Q, m, m)
        R = self._ensure_3d(R, p, p)
        D = self._ensure_3d_vec(D, m)
        A = self._ensure_3d_vec(A, p)

        B_filt = np.zeros((m, T, S))
        P_filt = np.zeros((m, m, T, S))
        Pr_filt = np.zeros((T, S))
        lnl = 0.0
        v_store = np.zeros((p, T))
        K_store = np.zeros((m, p, T))

        pr_prev = self._ergodic_probs(Pm)

        B_prev = np.zeros((m, S))
        P_prev = np.zeros((m, m, S))
        for j in range(S):
            B_prev[:, j] = B0[:, 0, j] if B0.ndim == 3 else B0[:, 0]
            P_prev[:, :, j] = P0[:, :, j] if P0.ndim == 3 else P0

        for t in range(T):
            yt = y[t, :].reshape(-1, 1)
            obs_mask = ~np.isnan(yt.ravel())
            if not obs_mask.any():
                for j in range(S):
                    B_filt[:, t, j] = D[:, 0, j] + F[:, :, j] @ B_prev[:, j]
                    P_filt[:, :, t, j] = F[:, :, j] @ P_prev[:, :, j] @ F[:, :, j].T + Q[:, :, j]
                Pr_filt[t, :] = Pm.T @ pr_prev
                B_prev = B_filt[:, t, :]
                P_prev = P_filt[:, :, t, :]
                pr_prev = Pr_filt[t, :]
                continue

            yt_obs = yt[obs_mask]
            p_obs = obs_mask.sum()

            joint_pr = np.zeros((S, S))
            B_ij = np.zeros((m, S, S))
            P_ij = np.zeros((m, m, S, S))

            for i in range(S):
                for j in range(S):
                    B_pred = D[:, 0, j] + F[:, :, j] @ B_prev[:, i]
                    P_pred = F[:, :, j] @ P_prev[:, :, i] @ F[:, :, j].T + Q[:, :, j]

                    H_j = H[:, :, j][obs_mask, :]
                    A_j = A[:, 0, j][obs_mask].reshape(-1, 1)
                    R_j = R[:, :, j][np.ix_(obs_mask, obs_mask)]

                    v = yt_obs - A_j - H_j @ B_pred.reshape(-1, 1)
                    S_inn = H_j @ P_pred @ H_j.T + R_j
                    S_inn = 0.5 * (S_inn + S_inn.T) + np.eye(p_obs) * 1e-10

                    try:
                        S_inv = np.linalg.solve(S_inn, np.eye(p_obs))
                    except np.linalg.LinAlgError:
                        S_inv = np.linalg.pinv(S_inn)

                    K = P_pred @ H_j.T @ S_inv
                    B_upd = B_pred.reshape(-1, 1) + K @ v
                    P_upd = (np.eye(m) - K @ H_j) @ P_pred

                    B_ij[:, i, j] = B_upd.ravel()
                    P_ij[:, :, i, j] = 0.5 * (P_upd + P_upd.T)

                    sign, logdet = np.linalg.slogdet(S_inn)
                    if sign <= 0:
                        f_val = 1e-300
                    else:
                        ll_val = -0.5 * (p_obs * np.log(2 * np.pi) + logdet
                                         + (v.T @ S_inv @ v).item())
                        f_val = np.exp(max(ll_val, -500))

                    joint_pr[i, j] = f_val * Pm[i, j] * pr_prev[i]

            f_t = max(joint_pr.sum(), 1e-300)
            lnl += np.log(f_t)
            joint_pr /= f_t

            pr_new = joint_pr.sum(axis=0)
            pr_new = np.maximum(pr_new, 1e-15)
            pr_new /= pr_new.sum()
            Pr_filt[t, :] = pr_new

            # Kim collapsing
            for j in range(S):
                if pr_new[j] < 1e-15:
                    B_filt[:, t, j] = B_ij[:, 0, j]
                    P_filt[:, :, t, j] = P_ij[:, :, 0, j]
                    continue
                w = joint_pr[:, j] / pr_new[j]
                w = np.maximum(w, 0)
                ws = w.sum()
                if ws > 0:
                    w /= ws
                B_col = np.zeros(m)
                for ii in range(S):
                    B_col += w[ii] * B_ij[:, ii, j]
                P_col = np.zeros((m, m))
                for ii in range(S):
                    diff = (B_ij[:, ii, j] - B_col).reshape(-1, 1)
                    P_col += w[ii] * (P_ij[:, :, ii, j] + diff @ diff.T)
                B_filt[:, t, j] = B_col
                P_filt[:, :, t, j] = 0.5 * (P_col + P_col.T)

            for j in range(S):
                H_j = H[:, :, j]
                A_j = A[:, 0, j].reshape(-1, 1)
                v_j = yt - A_j - H_j @ B_filt[:, t, j].reshape(-1, 1)
                v_store[:, t] += pr_new[j] * v_j.ravel()

            B_prev = B_filt[:, t, :]
            P_prev = P_filt[:, :, t, :]
            pr_prev = pr_new

        B_col_out = np.zeros((m, T))
        for t in range(T):
            for j in range(S):
                B_col_out[:, t] += Pr_filt[t, j] * B_filt[:, t, j]

        return dict(B_tt=B_filt, P_tt=P_filt, Pr_tt=Pr_filt,
                    lnl=lnl, B_col=B_col_out, v_t=v_store, K_t=K_store)

    def smoother(self, filter_output, F, Pm):
        S, m = self.S, self.m
        F = self._ensure_3d(F, m, m)
        Pr_tt = filter_output["Pr_tt"]
        B_tt = filter_output["B_tt"]
        T = Pr_tt.shape[0]

        Pr_smooth = np.zeros((T, S))
        Pr_smooth[-1, :] = Pr_tt[-1, :]

        for t in range(T - 2, -1, -1):
            pr_pred = Pm.T @ Pr_tt[t, :]
            pr_pred = np.maximum(pr_pred, 1e-15)
            for j in range(S):
                s = 0.0
                for k in range(S):
                    s += Pm[j, k] * Pr_smooth[t + 1, k] / pr_pred[k]
                Pr_smooth[t, j] = Pr_tt[t, j] * s
            ps = Pr_smooth[t, :].sum()
            if ps > 0:
                Pr_smooth[t, :] /= ps

        B_smooth = np.zeros((m, T))
        for t in range(T):
            for j in range(S):
                B_smooth[:, t] += Pr_smooth[t, j] * B_tt[:, t, j]

        return dict(Pr_smooth=Pr_smooth, B_smooth=B_smooth)

    def _ensure_3d(self, arr, d1, d2):
        arr = np.asarray(arr, dtype=np.float64)
        if arr.ndim == 2:
            return np.stack([arr] * self.S, axis=2)
        return arr

    def _ensure_3d_vec(self, arr, d):
        arr = np.asarray(arr, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.ndim == 2:
            return np.stack([arr] * self.S, axis=2)
        return arr

    def _ergodic_probs(self, Pm):
        S = Pm.shape[0]
        A = np.vstack([Pm.T - np.eye(S), np.ones((1, S))])
        b = np.zeros(S + 1); b[-1] = 1.0
        try:
            pi_s = np.linalg.lstsq(A, b, rcond=None)[0]
            pi_s = np.maximum(pi_s, 1e-10)
            pi_s /= pi_s.sum()
        except Exception:
            pi_s = np.ones(S) / S
        return pi_s


# =========================================================================
#  Transition probability builder utility
# =========================================================================

def _build_transition_matrix(logits, S):
    """Build transition matrix from unconstrained logits."""
    Pm = np.zeros((S, S))
    k = 0
    for i in range(S):
        for j in range(S - 1):
            Pm[i, j] = 1.0 / (1.0 + np.exp(-logits[k])); k += 1
        total = Pm[i, :S-1].sum()
        if total >= 1.0:
            Pm[i, :S-1] *= 0.98 / total
        Pm[i, S-1] = 1.0 - Pm[i, :S-1].sum()
    return np.maximum(Pm, 1e-10)


# =========================================================================
#  Model A: MS-AR (switching mean & variance)
# =========================================================================

class MarkovSwitchingAR:
    """
    MS-AR(p): r_t = mu(s_t) + phi_1 r_{t-1} + ... + phi_p r_{t-p} + sigma(s_t) eps_t
    Cast into state-space form and estimated via Kim filter MLE.
    """
    def __init__(self, n_regimes=2, ar_order=1):
        self.S = n_regimes
        self.ar_order = max(ar_order, 1)
        self.params = None; self.filter_output = None
        self.smooth_output = None; self.ssm = None

    def _build_ssm(self, params):
        p, S = self.ar_order, self.S; m = p
        idx = 0
        phi = params[idx:idx+p]; idx += p
        mu = params[idx:idx+S]; idx += S
        sigma = np.exp(params[idx:idx+S]); idx += S
        Pm = _build_transition_matrix(params[idx:idx+S*(S-1)], S)

        F_mat = np.zeros((m, m))
        F_mat[0, :] = phi
        if m > 1: F_mat[1:, :-1] = np.eye(m - 1)
        F = np.stack([F_mat] * S, axis=2)

        H = np.zeros((1, m)); H[0, 0] = 1.0
        H = np.stack([H] * S, axis=2)

        D = np.zeros((m, 1, S))
        for j in range(S): D[0, 0, j] = mu[j]

        Q = np.zeros((m, m, S))
        for j in range(S): Q[0, 0, j] = sigma[j]**2

        R = np.zeros((1, 1, S))
        for j in range(S): R[0, 0, j] = 1e-10
        A = np.zeros((1, 1, S))

        B0 = np.zeros((m, 1, S)); P0 = np.zeros((m, m, S))
        for j in range(S):
            phi_sum = phi.sum()
            if abs(phi_sum) < 1.0:
                B0[0, 0, j] = mu[j] / (1.0 - phi_sum)
            P0[:, :, j] = np.eye(m) * sigma[j]**2 / max(1.0 - phi_sum**2, 0.01)

        return dict(F=F, H=H, Q=Q, R=R, D=D, A=A, Pm=Pm, B0=B0, P0=P0,
                    mu=mu, sigma=sigma, phi=phi)

    def _fast_filter_scalar(self, y, phi, mu, sigma, Pm):
        """Ultra-fast Kim filter for AR(1) with 2 regimes: all scalar ops."""
        T = len(y); S = 2
        Pr = np.zeros((T, 2)); lnl = 0.0
        
        # Ergodic probs
        d = 2.0 - Pm[0,0] - Pm[1,1]
        pr = np.array([(1-Pm[1,1])/d, (1-Pm[0,0])/d]) if abs(d)>1e-10 else np.array([0.5,0.5])
        
        # Init per-regime state
        b = np.zeros(2); P = np.zeros(2)
        for j in range(2):
            ps = phi[0] if len(phi)>0 else 0
            b[j] = mu[j]/(1-ps) if abs(ps)<1 else 0
            P[j] = sigma[j]**2 / max(1-ps**2, 0.01)
        
        phi0 = phi[0] if len(phi) > 0 else 0.0
        sig2 = sigma**2
        log2pi = np.log(2*np.pi)
        
        for t in range(T):
            yt = y[t]
            if np.isnan(yt):
                for j in range(2):
                    b[j] = mu[j] + phi0*b[j]
                    P[j] = phi0*phi0*P[j] + sig2[j]
                pr = Pm.T @ pr; Pr[t,:] = pr
                continue
            
            joint = np.zeros((2,2))
            b_ij = np.zeros((2,2)); P_ij = np.zeros((2,2))
            
            for i in range(2):
                bp_i = b[i]; Pp_i = P[i]
                for j in range(2):
                    bp = mu[j] + phi0 * bp_i
                    Pp = phi0*phi0*Pp_i + sig2[j]
                    v = yt - bp
                    Si = Pp + 1e-10  # R=~0 for state-space AR
                    K = Pp / Si
                    b_ij[i,j] = bp + K*v
                    P_ij[i,j] = (1-K)*Pp
                    ll_v = -0.5*(log2pi + np.log(Si) + v*v/Si)
                    joint[i,j] = np.exp(max(ll_v, -500)) * Pm[i,j] * pr[i]
            
            ft = max(joint.sum(), 1e-300)
            lnl += np.log(ft)
            joint /= ft
            pn = np.maximum(joint.sum(axis=0), 1e-15); pn /= pn.sum()
            Pr[t,:] = pn
            
            # Collapse
            for j in range(2):
                if pn[j] < 1e-15:
                    b[j] = b_ij[0,j]; P[j] = P_ij[0,j]; continue
                w = joint[:,j]/pn[j]; w = np.maximum(w,0)
                ws = w.sum()
                if ws>0: w /= ws
                bc = w[0]*b_ij[0,j] + w[1]*b_ij[1,j]
                pc = w[0]*(P_ij[0,j]+(b_ij[0,j]-bc)**2) + w[1]*(P_ij[1,j]+(b_ij[1,j]-bc)**2)
                b[j] = bc; P[j] = pc
        
        return Pr, lnl

    def _fast_filter_2s_1d(self, y, ssm):
        """Specialized fast Kim filter for S=2, p=1 (avoids generic overhead)."""
        T = len(y); m = self.ar_order
        F0,F1 = ssm['F'][:,:,0], ssm['F'][:,:,1]
        Q0,Q1 = ssm['Q'][:,:,0], ssm['Q'][:,:,1]
        H0,H1 = ssm['H'][:,:,0], ssm['H'][:,:,1]
        D0,D1 = ssm['D'][:,0,0], ssm['D'][:,0,1]
        Pm = ssm['Pm']
        
        B_prev = np.zeros((m, 2))
        P_prev = np.zeros((m, m, 2))
        B_prev[:,0] = ssm['B0'][:,0,0]; B_prev[:,1] = ssm['B0'][:,0,1]
        P_prev[:,:,0] = ssm['P0'][:,:,0]; P_prev[:,:,1] = ssm['P0'][:,:,1]
        
        denom = 2.0 - Pm[0,0] - Pm[1,1]
        pr = np.array([(1-Pm[1,1]), (1-Pm[0,0])]) / denom if abs(denom)>1e-10 else np.array([0.5,0.5])
        
        Pr = np.zeros((T, 2)); B_col = np.zeros((m, T))
        lnl = 0.0; I_m = np.eye(m)
        Fs = [F0,F1]; Qs = [Q0,Q1]; Hs = [H0,H1]; Ds = [D0,D1]
        
        for t in range(T):
            yt = y[t]
            if np.isnan(yt):
                for j in range(2):
                    B_prev[:,j] = Ds[j] + Fs[j] @ B_prev[:,j]
                    P_prev[:,:,j] = Fs[j] @ P_prev[:,:,j] @ Fs[j].T + Qs[j]
                pr = Pm.T @ pr; Pr[t,:] = pr
                B_col[:,t] = pr[0]*B_prev[:,0] + pr[1]*B_prev[:,1]
                continue
            
            joint = np.zeros((2,2))
            Bij = np.zeros((m,2,2)); Pij = np.zeros((m,m,2,2))
            for i in range(2):
                for j in range(2):
                    bp = Ds[j] + Fs[j] @ B_prev[:,i]
                    Pp = Fs[j] @ P_prev[:,:,i] @ Fs[j].T + Qs[j]
                    v = yt - (Hs[j] @ bp)[0]
                    Si = (Hs[j] @ Pp @ Hs[j].T)[0,0] + 1e-10
                    K = (Pp @ Hs[j].T) / Si
                    Bij[:,i,j] = bp + K.ravel()*v
                    Pu = (I_m - K@Hs[j])@Pp
                    Pij[:,:,i,j] = 0.5*(Pu + Pu.T)
                    ll_v = -0.5*(np.log(2*np.pi) + np.log(Si) + v*v/Si)
                    joint[i,j] = np.exp(max(ll_v, -500)) * Pm[i,j] * pr[i]
            
            ft = max(joint.sum(), 1e-300); lnl += np.log(ft); joint /= ft
            pn = np.maximum(joint.sum(axis=0), 1e-15); pn /= pn.sum()
            Pr[t,:] = pn
            
            Bn = np.zeros((m,2)); Pn = np.zeros((m,m,2))
            for j in range(2):
                if pn[j]<1e-15: Bn[:,j]=Bij[:,0,j]; Pn[:,:,j]=Pij[:,:,0,j]; continue
                w = joint[:,j]/pn[j]; w = np.maximum(w,0); ws=w.sum()
                if ws>0: w /= ws
                bc = w[0]*Bij[:,0,j]+w[1]*Bij[:,1,j]
                pc = np.zeros((m,m))
                for ii in range(2):
                    d = (Bij[:,ii,j]-bc).reshape(-1,1)
                    pc += w[ii]*(Pij[:,:,ii,j]+d@d.T)
                Bn[:,j]=bc; Pn[:,:,j]=0.5*(pc+pc.T)
            B_prev=Bn; P_prev=Pn; pr=pn
            B_col[:,t] = pn[0]*Bn[:,0]+pn[1]*Bn[:,1]
        return Pr, B_col, lnl

    def fit(self, y, verbose=True):
        y = np.asarray(y, dtype=np.float64).ravel()
        y = y[~np.isnan(y)]; T = len(y)
        y_mat = y.reshape(-1, 1)
        p, S = self.ar_order, self.S; m = p
        kf = KimFilter(S, 1, m)

        # Smart init from simple statistics
        sig_y = np.std(y); mu_y = np.mean(y)
        x0 = [0.3, -0.1][:p]  # AR coefficients
        # Spread mu across regimes
        mu_init = [mu_y * (1 - 0.5*j) for j in range(S)]
        x0 += mu_init
        # Spread sigma across regimes (log-scale)
        sig_init = [np.log(sig_y * (0.5 + j * 1.0 / max(S-1, 1))) for j in range(S)]
        x0 += sig_init
        # Transition matrix logits
        x0 += [2.5] * (S * (S - 1))
        x0 = np.array(x0, dtype=np.float64)

        # Use fast specialized filter for S=2, p=1
        use_scalar = (S == 2 and p == 1)  # Ultra-fast scalar path
        use_fast = (S == 2 and p > 1 and p <= 4)  # Matrix path

        def neg_ll(par):
            try:
                if abs(par[:p].sum()) >= 0.999: return 1e12
                if use_scalar:
                    phi_v = par[:1]; mu_v = par[1:3]; sig_v = np.exp(par[3:5])
                    Pm_v = _build_transition_matrix(par[5:], 2)
                    _, ll = self._fast_filter_scalar(y, phi_v, mu_v, sig_v, Pm_v)
                elif use_fast:
                    ssm = self._build_ssm(par)
                    _, _, ll = self._fast_filter_2s_1d(y, ssm)
                else:
                    ssm = self._build_ssm(par)
                    res = kf.filter(y_mat, **ssm)
                    ll = res["lnl"]
                return -ll if np.isfinite(ll) else 1e12
            except Exception: return 1e12

        if verbose: print(f"Fitting MS-AR({p}) with {S} regimes (Kim filter)...")
        best = minimize(neg_ll, x0, method="Powell", options={"maxiter": 400, "ftol": 1e-5})
        # Skip refinement for speed — Powell is already good enough

        self.params = best.x
        ssm = self._build_ssm(self.params)
        self.ssm = ssm
        self.filter_output = kf.filter(y_mat, **ssm)
        self.smooth_output = kf.smoother(self.filter_output, ssm["F"], ssm["Pm"])

        if verbose:
            print(f"  AR coefficients: {ssm['phi']}")
            for j in range(S):
                print(f"  Regime {j}: mu={ssm['mu'][j]:.6f}, sigma={ssm['sigma'][j]:.6f} "
                      f"(ann={ssm['sigma'][j]*np.sqrt(252):.2%})")
                print(f"    P(stay)={ssm['Pm'][j,j]:.4f}, E[dur]={1/(1-ssm['Pm'][j,j]):.1f} days")
            ll = self.filter_output['lnl']; k = len(self.params); n = T
            print(f"  LogLik: {ll:.2f}, AIC: {-2*ll+2*k:.2f}, BIC: {-2*ll+k*np.log(n):.2f}")
        return self

    def filtered_probs(self): return self.filter_output["Pr_tt"]
    def smoothed_probs(self): return self.smooth_output["Pr_smooth"]
    def filtered_state(self): return self.filter_output["B_col"][0, :]
    def smoothed_state(self): return self.smooth_output["B_smooth"][0, :]

    def conditional_vol(self):
        probs = self.smoothed_probs(); sigma = self.ssm["sigma"]
        return sum(probs[:, j] * sigma[j] for j in range(self.S))

    def get_info_criteria(self):
        n = self.filter_output["Pr_tt"].shape[0]; k = len(self.params)
        ll = self.filter_output["lnl"]
        return dict(loglikelihood=ll, aic=-2*ll+2*k, bic=-2*ll+k*np.log(n), n_params=k)


# =========================================================================
#  Model B: MS-UC (trend + cycle, switching vol)
# =========================================================================

class MarkovSwitchingUC:
    """
    Unobserved-Components with MS volatility.
    y_t = tau_t + c_t + eps_t
      tau_t = tau_{t-1} + delta(s_t) + eta_t
      c_t   = phi1 c_{t-1} + phi2 c_{t-2} + kappa_t
    State: [tau, c, c_lag]. Switching: delta, sigma_eta, sigma_kappa.
    """
    def __init__(self, n_regimes=2):
        self.S = n_regimes
        self.params = None; self.filter_output = None; self.smooth_output = None; self.ssm = None

    def _build_ssm(self, params):
        S = self.S; m = 3; idx = 0
        phi1 = params[idx]; idx += 1
        phi2 = params[idx]; idx += 1
        delta = params[idx:idx+S]; idx += S
        sig_eta = np.exp(params[idx:idx+S]); idx += S
        sig_kappa = np.exp(params[idx:idx+S]); idx += S
        sig_eps = np.exp(params[idx]); idx += 1
        Pm = _build_transition_matrix(params[idx:idx+S*(S-1)], S)

        F_b = np.array([[1.,0.,0.],[0.,phi1,phi2],[0.,1.,0.]])
        F = np.stack([F_b]*S, axis=2)
        H_b = np.array([[1.,1.,0.]]); H = np.stack([H_b]*S, axis=2)

        D = np.zeros((m,1,S))
        for j in range(S): D[0,0,j] = delta[j]

        Q = np.zeros((m,m,S))
        for j in range(S):
            Q[0,0,j] = sig_eta[j]**2; Q[1,1,j] = sig_kappa[j]**2

        R = np.zeros((1,1,S))
        for j in range(S): R[0,0,j] = sig_eps**2
        A = np.zeros((1,1,S))

        B0 = np.zeros((m,1,S)); P0 = np.zeros((m,m,S))
        for j in range(S): P0[:,:,j] = np.eye(m)*10.0

        return dict(F=F, H=H, Q=Q, R=R, D=D, A=A, Pm=Pm, B0=B0, P0=P0,
                    delta=delta, sig_eta=sig_eta, sig_kappa=sig_kappa,
                    sig_eps=sig_eps, phi1=phi1, phi2=phi2)

    def fit(self, y, verbose=True):
        y = np.asarray(y, dtype=np.float64).ravel(); y = y[~np.isnan(y)]
        T = len(y); y_mat = y.reshape(-1,1); S = self.S
        kf = KimFilter(S, 1, 3)
        sy = np.std(y); my = np.mean(y)
        x0 = [0.3, -0.1]
        x0 += [my*0.5, my*-0.5][:S]
        x0 += [np.log(sy*0.5)]*S + [np.log(sy*0.3)]*S + [np.log(sy*0.2)]
        x0 += [2.0]*(S*(S-1))
        x0 = np.array(x0, dtype=np.float64)

        def neg_ll(par):
            try:
                ssm = self._build_ssm(par)
                res = kf.filter(y_mat, **ssm)
                ll = res["lnl"]
                return -ll if np.isfinite(ll) else 1e12
            except Exception: return 1e12

        if verbose: print(f"Fitting MS-UC model with {S} regimes (Kim filter)...")
        r = minimize(neg_ll, x0, method="Nelder-Mead", options={"maxiter": 8000})
        try:
            r2 = minimize(neg_ll, r.x, method="L-BFGS-B", options={"maxiter": 2000})
            if r2.fun < r.fun: r = r2
        except Exception: pass

        self.params = r.x; ssm = self._build_ssm(self.params); self.ssm = ssm
        self.filter_output = kf.filter(y_mat, **ssm)
        self.smooth_output = kf.smoother(self.filter_output, ssm["F"], ssm["Pm"])

        if verbose:
            print(f"  Cycle AR: phi1={ssm['phi1']:.4f}, phi2={ssm['phi2']:.4f}")
            for j in range(S):
                print(f"  Regime {j}: drift={ssm['delta'][j]:.6f}, "
                      f"sig_trend={ssm['sig_eta'][j]:.6f} (ann={ssm['sig_eta'][j]*np.sqrt(252):.2%}), "
                      f"sig_cycle={ssm['sig_kappa'][j]:.6f}")
                print(f"    P(stay)={ssm['Pm'][j,j]:.4f}, E[dur]={1/(1-ssm['Pm'][j,j]):.1f}")
            print(f"  sig_obs={ssm['sig_eps']:.6f}")
            print(f"  LogLik: {self.filter_output['lnl']:.2f}")
        return self

    def trend(self): return self.smooth_output["B_smooth"][0, :]
    def cycle(self): return self.smooth_output["B_smooth"][1, :]
    def filtered_probs(self): return self.filter_output["Pr_tt"]
    def smoothed_probs(self): return self.smooth_output["Pr_smooth"]

    def conditional_vol(self):
        probs = self.smoothed_probs()
        return sum(probs[:, j] * np.sqrt(self.ssm["sig_eta"][j]**2 + self.ssm["sig_kappa"][j]**2)
                   for j in range(self.S))


# =========================================================================
#  Model C: MS-DCF (Dynamic Common Factor — mirrors R kimfilter reference)
# =========================================================================

class MarkovSwitchingDCF:
    """
    Markov-Switching Dynamic Common Factor (Aevskiy / Kim-Yoo).
    For N series y_{i,t}:
      y_{i,t} = gamma_i * C_t + e_{i,t}
      C_t = mu(s_t) + phi_1 C_{t-1} + ... + eta_t,  eta ~ N(0,1)
      e_{i,t} = psi_{i,1} e_{i,t-1} + ... + sigma_i * eps_{i,t}
    Switching: mu(s_t) factor mean depends on regime.
    """
    def __init__(self, n_regimes=2, factor_ar=2, idio_ar=2):
        self.S = n_regimes; self.factor_ar = factor_ar; self.idio_ar = idio_ar
        self.params = None; self.filter_output = None; self.smooth_output = None; self.ssm = None

    def fit(self, Y, verbose=True):
        import pandas as pd
        if isinstance(Y, pd.DataFrame):
            self.var_names = Y.columns.tolist(); Y = Y.values
        else: self.var_names = [f"Var{i}" for i in range(Y.shape[1])]
        Y = np.asarray(Y, dtype=np.float64)
        valid = ~np.all(np.isnan(Y), axis=1); Y = Y[valid]
        T, N = Y.shape
        fa, ia, S = self.factor_ar, self.idio_ar, self.S
        m = fa + N * ia
        kf = KimFilter(S, N, m)
        Y_mean = np.nanmean(Y, axis=0); Y_dm = Y - Y_mean

        def _build(params):
            idx = 0
            phi = params[idx:idx+fa]; idx += fa
            mu = params[idx:idx+S]; idx += S
            gamma = params[idx:idx+N]; idx += N
            psi = params[idx:idx+N*ia].reshape(N, ia); idx += N*ia
            sig = np.exp(params[idx:idx+N]); idx += N
            Pm = _build_transition_matrix(params[idx:idx+S*(S-1)], S)

            F_mat = np.zeros((m, m))
            F_mat[0, :fa] = phi
            if fa > 1: F_mat[1:fa, :fa-1] = np.eye(fa - 1)
            for i in range(N):
                off = fa + i*ia
                F_mat[off, off:off+ia] = psi[i]
                if ia > 1:
                    for lag in range(1, ia):
                        F_mat[off+lag, off+lag-1] = 1.0
            F = np.stack([F_mat]*S, axis=2)

            H_mat = np.zeros((N, m))
            for i in range(N):
                H_mat[i, 0] = gamma[i]
                H_mat[i, fa + i*ia] = 1.0
            H = np.stack([H_mat]*S, axis=2)

            Q_mat = np.zeros((m, m))
            Q_mat[0, 0] = 1.0
            for i in range(N): Q_mat[fa+i*ia, fa+i*ia] = sig[i]**2
            Q = np.stack([Q_mat]*S, axis=2)

            R = np.stack([np.eye(N)*1e-10]*S, axis=2)
            D = np.zeros((m,1,S))
            for j in range(S): D[0,0,j] = mu[j]
            A = np.zeros((N,1,S))

            B0 = np.zeros((m,1,S)); P0 = np.zeros((m,m,S))
            for j in range(S):
                ps = phi.sum()
                if abs(ps) < 1: B0[0,0,j] = mu[j]/(1-ps)
                P0[:,:,j] = np.eye(m)*1.0
            return dict(F=F,H=H,Q=Q,R=R,D=D,A=A,Pm=Pm,B0=B0,P0=P0,
                        phi=phi,mu=mu,gamma=gamma,psi=psi,sig=sig)

        x0 = list([0.5, -0.1][:fa])
        x0 += [0.2, -0.5][:S]
        x0 += [0.01]*N
        x0 += [0.1, -0.05]*N
        x0 += [np.log(np.nanstd(Y_dm[:, i])*0.5) for i in range(N)]
        x0 += [2.5]*(S*(S-1))
        x0 = np.array(x0, dtype=np.float64)

        def neg_ll(par):
            try:
                ssm = _build(par)
                if abs(par[:fa].sum()) >= 0.999: return 1e12
                res = kf.filter(Y_dm, **ssm)
                return -res["lnl"] if np.isfinite(res["lnl"]) else 1e12
            except Exception: return 1e12

        if verbose:
            print(f"Fitting MS-DCF: {N} series, {S} regimes, factor AR({fa}), idio AR({ia})...")
        r = minimize(neg_ll, x0, method="Nelder-Mead", options={"maxiter": 10000})
        try:
            r2 = minimize(neg_ll, r.x, method="L-BFGS-B", options={"maxiter": 2000})
            if r2.fun < r.fun: r = r2
        except Exception: pass

        self.params = r.x; ssm = _build(self.params); self.ssm = ssm
        self._build_fn = _build; self.Y = Y; self.Y_dm = Y_dm; self.Y_mean = Y_mean; self.N = N
        self.filter_output = kf.filter(Y_dm, **ssm)
        self.smooth_output = kf.smoother(self.filter_output, ssm["F"], ssm["Pm"])

        if verbose:
            print(f"  Factor AR: {ssm['phi']}")
            print(f"  Factor loadings: {ssm['gamma']}")
            for j in range(S):
                print(f"  Regime {j}: mu_factor={ssm['mu'][j]:.6f}, "
                      f"P(stay)={ssm['Pm'][j,j]:.4f}, E[dur]={1/(1-ssm['Pm'][j,j]):.1f}")
            print(f"  Idio sigma: {ssm['sig']}")
            print(f"  LogLik: {self.filter_output['lnl']:.2f}")
        return self

    def common_factor(self): return self.smooth_output["B_smooth"][0, :]
    def filtered_probs(self): return self.filter_output["Pr_tt"]
    def smoothed_probs(self): return self.smooth_output["Pr_smooth"]

    def variance_decomposition(self):
        factor = self.common_factor(); gamma = self.ssm["gamma"]
        result = {}
        for i, name in enumerate(self.var_names):
            fc = gamma[i]*factor; tv = np.nanvar(self.Y_dm[:len(factor), i])
            result[name] = np.nanvar(fc)/tv if tv > 0 else 0
        return result
