"""
Configuration
=============
Paths and parameters. All strategy constants are defined in main_analysis.py
next to the code that uses them, for readability.
"""
from pathlib import Path

# Paths
DATA_DIR = Path(__file__).parent / "data"
OUTPUT_DIR = Path(__file__).parent / "output"

CHARTS_DIR = OUTPUT_DIR / "charts"
DATA_OUT_DIR = OUTPUT_DIR / "data"
REPORTS_DIR = OUTPUT_DIR / "reports"

for _d in [CHARTS_DIR, DATA_OUT_DIR, REPORTS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# Risk-free rate (annualised)
RISK_FREE_RATE = 0.02
