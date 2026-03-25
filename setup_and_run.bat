@echo off
echo ================================================================
echo   Options Trading Project - Setup and Run
echo ================================================================
echo.

REM Navigate to the folder where this .bat file lives
cd /d "%~dp0"
echo Working directory: %cd%
echo.

echo [1/4] Installing Python dependencies...
call pip install numpy pandas scipy matplotlib seaborn scikit-learn pyarrow statsmodels numba arch -q 2>nul

echo [2/4] Installing tectonic (LaTeX engine for PDF report)...
call conda install -c conda-forge tectonic -y -q 2>nul
if errorlevel 1 (
    echo   tectonic not installed via conda - report will use matplotlib fallback
)

echo [3/4] Checking data files...
if not exist "data\options_daily.parquet" (
    echo   WARNING: data\options_daily.parquet not found!
    echo   Copy your options.parquet to data\options_daily.parquet
    echo   Or the script will use synthetic data as fallback.
)

echo [4/4] Running analysis...
echo.
python main_analysis.py

echo.
echo ================================================================
echo   Done! Check output/ for results.
echo ================================================================
pause
