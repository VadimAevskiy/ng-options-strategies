# CME Natural Gas Options Trading Strategies

## Quick Start

### 1. Place input data
Copy `options.parquet` into `data/`:
```
data/options_daily.parquet      ← you must place this here
data/oi_weighted_futures.csv    ← already included
```

### 2. Run (auto-installs everything)
```bash
python main_analysis.py
```

The script auto-installs:
- Python packages (numpy, pandas, scipy, matplotlib, etc.)
- `tectonic` LaTeX engine via conda (for full-quality PDF report)

If tectonic/conda are not available, the PDF report is generated
via matplotlib as fallback (same content, simpler formatting).

### 3. Outputs (all generated from scratch)
```
output/
  charts/     10 PNG charts
  data/       CSV exports (signals, trades, hedge daily P&L)
  reports/    strategy_report.pdf (full report — LaTeX or matplotlib)
              executive_summary.pdf (3-page summary)
              strategy_report.tex (LaTeX source with computed values)
```

### Alternative: use setup_and_run.bat (Windows)
Double-click `setup_and_run.bat` — it installs dependencies + tectonic + runs.

## Strategies

**Strategy 1 (Strangle Alpha Overlay):** Sells strangles when GARCH vol is elevated,
regime is calm, no jump clustering, and seasonal filters pass. Additive alpha overlay.

**Strategy 2 (Vol-Targeted Defensive Allocation):** delta = target_vol / rv20.
Leverage in calm markets, rapid cut in stress. Crisis-onset protective puts.
Systematically outperforms B&H across the entire backtest.
