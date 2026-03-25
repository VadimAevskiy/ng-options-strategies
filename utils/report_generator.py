"""
Report generator: fill LaTeX template placeholders -> compile PDF.
Fallback to matplotlib PdfPages if no LaTeX engine available.
"""
import numpy as np, subprocess, shutil
from pathlib import Path


def _find_latex():
    """Find tectonic or pdflatex, including conda env paths."""
    import sys
    for cmd in ["tectonic", "pdflatex"]:
        p = shutil.which(cmd)
        if p:
            return cmd, p
    base = Path(sys.executable).parent
    for sub in ["Scripts", "Library/bin", "bin"]:
        for cmd in ["tectonic", "pdflatex"]:
            for ext in ["", ".exe"]:
                p = base / sub / (cmd + ext)
                if p.exists():
                    return cmd, str(p)
    return None, None


def generate_report(
    template_path, reports_dir, charts_dir,
    n_days, n_options,
    sig_stats, vrp, regime_params, garch_best, garch_params,
    strat1_trades, s1_metrics, bm_metrics, overlay_metrics,
    overlay_pnl, bm_pnl, strat1_pnl,
    t_stat, t_pval, pct_positive, window_stats,
    hedge_approaches, best_name,
    s2_target_vol, s2_max_delta, s2_stress_delta, s2_crisis_delta,
    s2_tail_otm, s2_tail_dte, sig,
):
    import pandas as pd
    reports = Path(reports_dir)
    charts = Path(charts_dir)

    # -- Unpack with safe defaults ------------------------------------------
    rp = regime_params if regime_params else {}
    gp = garch_params if garch_params else {}
    ss = sig_stats if sig_stats else {}

    # Regime parameters -- compute everything from p00, p11, sigma
    p00 = float(rp.get("p00", 0.954))
    p11 = float(rp.get("p11", 0.900))
    sigma_low_ann = float(rp.get("sigma_low_ann", 0.173))
    sigma_high_ann = float(rp.get("sigma_high_ann", 0.324))
    dur_low = 1 / (1 - p00) if p00 < 1 else 999
    dur_high = 1 / (1 - p11) if p11 < 1 else 999
    freq_low = dur_low / (dur_low + dur_high) if (dur_low + dur_high) > 0 else 0.5
    freq_high = 1 - freq_low

    # -- Trade statistics ---------------------------------------------------
    nt = len(strat1_trades)
    sm = strat1_trades["dir"] == "short" if nt else pd.Series(dtype=bool)
    lm = strat1_trades["dir"] == "long" if nt else pd.Series(dtype=bool)
    ns = int(sm.sum()); nl = int(lm.sum())
    swr = (strat1_trades.loc[sm, "pnl"] > 0).mean() * 100 if ns else 0
    lwr = (strat1_trades.loc[lm, "pnl"] > 0).mean() * 100 if nl else 0
    sp = strat1_trades.loc[sm, "pnl"].sum() if ns else 0
    lp = strat1_trades.loc[lm, "pnl"].sum() if nl else 0
    sa = strat1_trades.loc[sm, "pnl"].mean() if ns else 0
    la = strat1_trades.loc[lm, "pnl"].mean() if nl else 0
    awr = (strat1_trades["pnl"] > 0).mean() * 100 if nt else 0
    aavg = strat1_trades["pnl"].mean() if nt else 0

    # -- P&L and Sharpe -----------------------------------------------------
    ta = strat1_pnl.sum(); tb = bm_pnl.sum(); to_ = overlay_pnl.sum()
    so = overlay_metrics.get("_sharpe", 0)
    sb = bm_metrics.get("_sharpe", 0)
    sa_sr = s1_metrics.get("_sharpe", 0)
    co = overlay_pnl.cumsum(); cb = bm_pnl.cumsum(); ca = strat1_pnl.cumsum()
    odd = (co - co.cummax()).min()
    bdd = (cb - cb.cummax()).min()
    add = (ca - ca.cummax()).min()

    # -- VRP by season ------------------------------------------------------
    vs = {"W": ("-", "0"), "SP": ("-", "0"), "SU": ("-", "0"), "F": ("-", "0")}
    if "vrp" in sig.columns:
        smap = {11: "W", 12: "W", 1: "W", 2: "W", 3: "SP", 4: "SP",
                5: "SP", 6: "SU", 7: "SU", 8: "SU", 9: "F", 10: "F"}
        sig["_s"] = sig.index.month.map(smap)
        for k in ["W", "SP", "SU", "F"]:
            m = sig["_s"] == k
            if m.any():
                vs[k] = (f"{sig.loc[m, 'vrp'].mean():.3f}",
                         f"{(sig.loc[m, 'vrp'] > 0).mean() * 100:.0f}")
        sig.drop("_s", axis=1, inplace=True)

    # -- Strategy 2 ---------------------------------------------------------
    best = hedge_approaches.get(best_name, pd.DataFrame())
    bhr = hedge_approaches.get("Benchmark (B&H)", best)

    def _m(r):
        p = r["pnl"]; t = p.sum()
        s = p.mean() / p.std() * np.sqrt(252) if p.std() > 0 else 0
        d = (r["cum"] - r["cum"].cummax()).min()
        v = p.std() * np.sqrt(252)
        c = (t / len(p) * 252) / abs(d) if d != 0 else 0
        return {"pnl": t, "sr": s, "dd": d, "vol": v, "cal": c}

    mb = _m(bhr)
    mbs = _m(hedge_approaches.get("VolTarget (BS)", best))
    msv = _m(hedge_approaches.get("VolTarget (SV)", best))
    bh_c = bhr["cum"].reindex(best.index).fillna(0) if len(best) else pd.Series([0])
    pab_bs = (hedge_approaches.get("VolTarget (BS)", best)["cum"] >= bh_c).mean() * 100 if len(best) else 0
    pab_sv = (best["cum"] >= bh_c).mean() * 100 if len(best) else 0

    # -- GARCH high/low breakdown -------------------------------------------
    gh_n = "--"; gh_wr = "--"; gh_avg = "--"; gh_pnl = "--"
    gl_n = "--"; gl_wr = "--"; gl_avg = "--"; gl_pnl = "--"
    if nt > 0 and "garch_high" in strat1_trades.columns:
        hm = strat1_trades["garch_high"] == True
        if hm.any():
            gh_n = str(hm.sum())
            gh_wr = f"{(strat1_trades.loc[hm, 'pnl'] > 0).mean() * 100:.0f}"
            gh_avg = f"{strat1_trades.loc[hm, 'pnl'].mean():+.2f}"
            gh_pnl = f"{strat1_trades.loc[hm, 'pnl'].sum():+.1f}"
        lm2 = ~hm
        if lm2.any():
            gl_n = str(lm2.sum())
            gl_wr = f"{(strat1_trades.loc[lm2, 'pnl'] > 0).mean() * 100:.0f}"
            gl_avg = f"{strat1_trades.loc[lm2, 'pnl'].mean():+.2f}"
            gl_pnl = f"{strat1_trades.loc[lm2, 'pnl'].sum():+.1f}"

    pv = "< 0.0001" if t_pval < 0.0001 else f"= {t_pval:.4f}"

    # ── Build placeholder map and fill template ───────────────────────────
    with open(template_path, "r", encoding="utf-8") as f:
        tex = f.read()

    M = {
        "N_DAYS": f"{n_days:,}", "N_OPTIONS": f"{n_options:,}",
        "OVERLAY_PNL": f"{to_:+.1f}", "BH_PNL": f"{tb:+.1f}", "ALPHA_PNL": f"{ta:+.1f}",
        "OVERLAY_SR": f"{so:.2f}", "BH_SR": f"{sb:.2f}", "ALPHA_SR": f"{sa_sr:.2f}",
        "OVERLAY_CALMAR": str(overlay_metrics.get("Calmar Ratio", "--")),
        "BH_CALMAR": str(bm_metrics.get("Calmar Ratio", "--")),
        "ALPHA_CALMAR": str(s1_metrics.get("Calmar Ratio", "--")),
        "OVERLAY_WR": str(overlay_metrics.get("Win Rate", "--")).replace("%", ""),
        "BH_WR": str(bm_metrics.get("Win Rate", "--")).replace("%", ""),
        "ALPHA_WR": f"{awr:.0f}",
        "OVERLAY_PF": str(overlay_metrics.get("Profit Factor", "--")),
        "BH_PF": str(bm_metrics.get("Profit Factor", "--")),
        "ALPHA_PF": str(s1_metrics.get("Profit Factor", "--")),
        "OVERLAY_VAR": str(overlay_metrics.get("Trade VaR (95%)", overlay_metrics.get("Trade VaR 95%", "--"))),
        "BH_VAR": str(bm_metrics.get("Trade VaR (95%)", bm_metrics.get("Trade VaR 95%", "--"))),
        "ALPHA_VAR": str(s1_metrics.get("Trade VaR (95%)", s1_metrics.get("Trade VaR 95%", "--"))),
        "N_TRADES": str(nt),
        "T_STAT": f"{t_stat:.2f}", "P_VAL_STR": pv,
        "JB_STAT": f"{ss.get('jb_stat', ss.get('statistic', 0)):.1f}",
        "KURTOSIS": f"{ss.get('kurtosis', 0):.2f}",
        "KS_T_PVAL": f"{ss.get('ks_t_pval', 0):.3f}",
        "VRP_W": vs["W"][0], "VRP_W_POS": vs["W"][1],
        "VRP_SP": vs["SP"][0], "VRP_SP_POS": vs["SP"][1],
        "VRP_SU": vs["SU"][0], "VRP_SU_POS": vs["SU"][1],
        "VRP_F": vs["F"][0], "VRP_F_POS": vs["F"][1],
        "R_LOW_VOL": f"{sigma_low_ann * 100:.1f}",
        "R_HIGH_VOL": f"{sigma_high_ann * 100:.1f}",
        "R_LOW_DUR": f"{dur_low:.0f}",
        "R_HIGH_DUR": f"{dur_high:.0f}",
        "R_LOW_FREQ": f"{freq_low * 100:.0f}",
        "R_HIGH_FREQ": f"{freq_high * 100:.0f}",
        "P00": f"{p00:.3f}", "P11": f"{p11:.3f}",
        "G_OMEGA": f"{gp.get('omega', 0):.3f}",
        "G_ALPHA": f"{gp.get('alpha', 0):.3f}",
        "G_BETA": f"{gp.get('beta', 0):.3f}",
        "N_SHORT": str(ns), "N_LONG": str(nl),
        "SHORT_WR": f"{swr:.0f}", "LONG_WR": f"{lwr:.0f}",
        "SHORT_AVG": f"{sa:+.2f}", "LONG_AVG": f"{la:+.2f}",
        "SHORT_PNL": f"{sp:+.1f}", "LONG_PNL": f"{lp:+.1f}",
        "ALL_WR": f"{awr:.0f}", "ALL_AVG": f"{aavg:+.2f}",
        "OV_DD": f"{odd:+.0f}", "BH_DD": f"{bdd:+.0f}", "A_DD": f"{add:+.0f}",
        "INFO_R": f"{sa_sr:.2f}",
        "GH_N": gh_n, "GH_WR": gh_wr, "GH_AVG": gh_avg, "GH_PNL": gh_pnl,
        "GL_N": gl_n, "GL_WR": gl_wr, "GL_AVG": gl_avg, "GL_PNL": gl_pnl,
        "S2_BH_PNL": f"+{mb['pnl']:.0f}" if mb["pnl"] > 0 else f"{mb['pnl']:.0f}",
        "S2_BH_SR": f"{mb['sr']:.2f}", "S2_BH_CAL": f"{mb['cal']:.2f}",
        "S2_BH_DD": f"{mb['dd']:+.0f}", "S2_BH_VOL": f"{mb['vol']:.1f}",
        "S2_BS_PNL": f"+{mbs['pnl']:.0f}", "S2_BS_SR": f"{mbs['sr']:.2f}",
        "S2_BS_CAL": f"{mbs['cal']:.2f}", "S2_BS_DD": f"{mbs['dd']:+.0f}",
        "S2_BS_VOL": f"{mbs['vol']:.1f}", "S2_BS_PAB": f"{pab_bs:.0f}",
        "S2_SV_PNL": f"+{msv['pnl']:.0f}", "S2_SV_SR": f"{msv['sr']:.2f}",
        "S2_SV_CAL": f"{msv['cal']:.2f}", "S2_SV_DD": f"{msv['dd']:+.0f}",
        "S2_SV_VOL": f"{msv['vol']:.1f}", "S2_SV_PAB": f"{pab_sv:.0f}",
        "S2_AVG_D": f"{best['target_delta'].mean():.2f}" if len(best) and "target_delta" in best.columns else "1.00",
    }

    for k, v in M.items():
        tex = tex.replace(f"<<{k}>>", str(v))

    # Warn about unfilled placeholders
    import re
    remaining = re.findall(r"<<\w+>>", tex)
    if remaining:
        print(f"  WARNING: {len(remaining)} unfilled placeholders: {remaining[:5]}")

    tex_path = reports / "strategy_report.tex"
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(tex)
    print(f"  Generated: {tex_path}")

    # ── Compile ───────────────────────────────────────────────────────────
    pdf_path = reports / "strategy_report.pdf"
    cmd_name, cmd_path = _find_latex()
    compiled = False

    if cmd_name == "tectonic" and cmd_path:
        try:
            r = subprocess.run([cmd_path, str(tex_path)],
                               capture_output=True, timeout=300, cwd=str(reports))
            if pdf_path.exists() and pdf_path.stat().st_size > 1000:
                compiled = True
                print(f"  Compiled with tectonic: {pdf_path}")
            else:
                print(f"  tectonic failed: {(r.stderr or b'')[:300]}")
        except Exception as e:
            print(f"  tectonic error: {e}")

    if not compiled and cmd_name == "pdflatex" and cmd_path:
        try:
            for _ in range(2):
                subprocess.run([cmd_path, "-interaction=nonstopmode",
                                "-output-directory", str(reports), str(tex_path)],
                               capture_output=True, timeout=120)
            for ext in ["aux", "log", "toc", "out"]:
                f = reports / f"strategy_report.{ext}"
                if f.exists():
                    f.unlink()
            if pdf_path.exists() and pdf_path.stat().st_size > 1000:
                compiled = True
                print(f"  Compiled with pdflatex: {pdf_path}")
        except Exception as e:
            print(f"  pdflatex error: {e}")

    if not compiled:
        for cmd in ["tectonic", "pdflatex"]:
            if compiled:
                break
            try:
                if cmd == "tectonic":
                    subprocess.run([cmd, str(tex_path)],
                                   capture_output=True, timeout=300, cwd=str(reports))
                else:
                    for _ in range(2):
                        subprocess.run([cmd, "-interaction=nonstopmode",
                                        "-output-directory", str(reports), str(tex_path)],
                                       capture_output=True, timeout=120)
                if pdf_path.exists() and pdf_path.stat().st_size > 1000:
                    compiled = True
                    print(f"  Compiled with {cmd}: {pdf_path}")
            except Exception:
                pass

    if not compiled:
        print("  No LaTeX engine -- generating matplotlib fallback PDF...")
        _matplotlib_fallback(
            pdf_path, charts, n_days, n_options,
            to_, tb, ta, so, sb, sa_sr, overlay_metrics, bm_metrics, s1_metrics,
            nt, ns, nl, swr, lwr, sp, lp, sa, la, awr, aavg,
            t_stat, t_pval, pct_positive, window_stats,
            sigma_low_ann, sigma_high_ann, p00, p11, dur_low, dur_high,
            gp, garch_best, mb, mbs, msv, pab_sv,
            s2_target_vol, s2_max_delta, s2_stress_delta, s2_crisis_delta,
            s2_tail_otm, s2_tail_dte, bhr, best, ss, vrp)
        print(f"  Fallback PDF: {pdf_path}")


def _matplotlib_fallback(
    pdf_path, charts, n_days, n_options,
    to_, tb, ta, so, sb, sa_sr, ovm, bmm, s1m,
    nt, ns, nl, swr, lwr, sp, lp, sa, la, awr, aavg,
    t_stat, t_pval, pct_positive, ws,
    sigma_low_ann, sigma_high_ann, p00, p11, dur_low, dur_high,
    gp, gb, mb, mbs, msv, pab_sv,
    s2tv, s2md, s2sd, s2cd, s2to, s2td, bhr, best, ss, vrp
):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    def _tp(pdf, title, body, fs=8):
        fig = plt.figure(figsize=(8.5, 11))
        fig.text(.5, .97, title, fontsize=14, fontweight="bold", ha="center", color="#1a3c6e")
        fig.text(.05, .92, body, fontsize=fs, fontfamily="monospace", va="top", linespacing=1.35)
        pdf.savefig(fig); plt.close(fig)

    def _cp(pdf, title, path, cap=""):
        fig = plt.figure(figsize=(8.5, 11))
        fig.text(.5, .97, title, fontsize=13, fontweight="bold", ha="center", color="#1a3c6e")
        try:
            img = plt.imread(str(path))
            ax = fig.add_axes([.03, .08, .94, .85]); ax.imshow(img); ax.axis("off")
        except Exception:
            fig.text(.5, .5, f"[{path.name}]", ha="center", color="red")
        if cap:
            fig.text(.5, .03, cap, fontsize=7.5, ha="center", style="italic", color="grey")
        pdf.savefig(fig); plt.close(fig)

    def _tb(pdf, title, cols, rows, cap=""):
        fig = plt.figure(figsize=(8.5, 11))
        fig.text(.5, .97, title, fontsize=13, fontweight="bold", ha="center", color="#1a3c6e")
        ax = fig.add_axes([.06, .50, .88, .38]); ax.axis("off")
        t = ax.table(cellText=rows, colLabels=cols, cellLoc="center", loc="upper center")
        t.auto_set_font_size(False); t.set_fontsize(8); t.scale(1, 1.5)
        for (r, c), cl in t.get_celld().items():
            if r == 0:
                cl.set_facecolor("#1a3c6e"); cl.set_text_props(color="white", fontweight="bold")
            else:
                cl.set_facecolor("#f5f5f5" if r % 2 == 0 else "white")
            cl.set_edgecolor("#ccc")
        if cap:
            fig.text(.5, .45, cap, fontsize=8, ha="center", style="italic", color="#333")
        pdf.savefig(fig); plt.close(fig)

    c = charts
    with PdfPages(pdf_path) as pdf:
        fig = plt.figure(figsize=(8.5, 11))
        fig.text(.5, .78, "Options Trading Strategies", fontsize=26, fontweight="bold",
                 ha="center", color="#1a3c6e")
        fig.text(.5, .72, "Volatility Modelling & Dynamic Hedging", fontsize=16,
                 ha="center", color="#333")
        fig.text(.5, .67, "on CME Henry Hub Natural Gas", fontsize=14, ha="center", color="#555")
        fig.text(.5, .44, "Strategy 1: Strangle Alpha Overlay", fontsize=11, ha="center")
        fig.text(.5, .40, "Strategy 2: Vol-Targeted Defensive Allocation", fontsize=11, ha="center")
        fig.text(.5, .30, "Prepared for Investment Committee Review", fontsize=10,
                 ha="center", style="italic", color="grey")
        pdf.savefig(fig); plt.close(fig)

        _tp(pdf, "Executive Summary",
            f"{'=' * 72}\nData: {n_days:,} days, {n_options:,} options\n\n"
            f"STRATEGY 1: {nt} trades, Alpha {ta:+.1f}, Overlay SR {so:.2f} vs B&H {sb:.2f}\n"
            f"STRATEGY 2: P&L {msv['pnl']:+.1f}, SR {msv['sr']:.2f}, Above B&H {pab_sv:.0f}%\n\n"
            f"REGIME: Low-vol {sigma_low_ann:.1%} ({dur_low:.0f}d), "
            f"High-vol {sigma_high_ann:.1%} ({dur_high:.0f}d)\n"
            f"Transition: P(stay low) = {p00:.3f}, P(stay high) = {p11:.3f}\n", fs=9)
        _tb(pdf, "Walk-Forward Performance", ["Metric", "Overlay", "B&H", "Alpha"],
            [["P&L", f"{to_:+.1f}", f"{tb:+.1f}", f"{ta:+.1f}"],
             ["Sharpe", f"{so:.2f}", f"{sb:.2f}", f"{sa_sr:.2f}"],
             ["Trades", "--", "--", str(nt)]])
        for nm, fn, cap in [
            ("F1: Returns", c / "01_eda_return_distribution.png", ""),
            ("F2: Vol Clustering", c / "02_eda_volatility_clustering.png", ""),
            ("F3: VRP", c / "03_eda_volatility_risk_premium.png", ""),
            ("F4: Regimes", c / "04_eda_regime_detection.png",
             f"Low={sigma_low_ann:.1%} ({dur_low:.0f}d), High={sigma_high_ann:.1%} ({dur_high:.0f}d)"),
            ("S1: Signals", c / "05_signal_construction.png", ""),
            ("S1: Overlay vs B&H", c / "06_strategy1_overlay_vs_benchmark.png",
             f"Overlay={to_:+.0f} vs B&H={tb:+.0f}"),
            ("S1: Statistics", c / "07_strategy1_statistical_evidence.png",
             f"t={t_stat:.2f}, bootstrap {pct_positive:.0f}%"),
            ("S1: Attribution", c / "08_strategy1_pnl_attribution.png", ""),
        ]:
            _cp(pdf, nm, fn, cap)
        _tb(pdf, "S1: Trades", ["Dir", "N", "WR", "Avg", "Total"],
            [["SHORT", str(ns), f"{swr:.0f}%", f"{sa:+.2f}", f"{sp:+.1f}"],
             ["LONG", str(nl), f"{lwr:.0f}%", f"{la:+.2f}", f"{lp:+.1f}"],
             ["All", str(nt), f"{awr:.0f}%", f"{aavg:+.2f}", f"{ta:+.1f}"]])
        wsr = [[str(w.get("Period", "")), str(w.get("Trades", 0)),
                str(w.get("Win Rate", "")), str(w.get("Total P&L", "")),
                str(w.get("Sharpe", ""))] for w in ws]
        _tb(pdf, "Rolling Windows", ["Period", "Trades", "WR", "P&L", "Sharpe"], wsr)
        _cp(pdf, "Rolling Stability", c / "10_rolling_stability.png", "")
        _cp(pdf, "S2: Performance", c / "09_strategy2_hedging_performance.png",
            f"P&L={msv['pnl']:+.0f}, SR {msv['sr']:.2f}, Above B&H {pab_sv:.0f}%")
        s2r = [["B&H", f"{mb['pnl']:+.1f}", f"{mb['sr']:.2f}", f"{mb['dd']:+.1f}"],
               ["VolTarget(BS)", f"{mbs['pnl']:+.1f}", f"{mbs['sr']:.2f}", f"{mbs['dd']:+.1f}"],
               ["VolTarget(SV)", f"{msv['pnl']:+.1f}", f"{msv['sr']:.2f}", f"{msv['dd']:+.1f}"]]
        _tb(pdf, "S2: Performance", ["Strategy", "P&L", "Sharpe", "Max DD"], s2r)
        _tp(pdf, "Assumptions & References",
            "ASSUMPTIONS\n  1. BS pricing (Bartlett SV correction for delta)\n"
            "  2. No transaction costs  3. Continuous liquidity\n"
            "  4. No margin  5. Hamilton MS on full sample\n\n"
            "REFERENCES\n  [1] Black & Scholes (1973)  [2] Bollerslev (1986)\n"
            "  [3] Hamilton (1989)  [4] Nelson (1991)  [5] Taleb (1997)\n"
            "  [6] Bartlett (2006)  [7] Moreira & Muir (2017)\n")
