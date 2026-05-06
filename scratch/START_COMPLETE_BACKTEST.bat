@echo off
echo ============================================================
echo COMPLETE FRESH BACKTEST - FULL 6-YEAR SCAN
echo ============================================================
echo.
echo This will:
echo   1. Scan ALL 3,527 symbols from 2019-2024
echo   2. Apply ML risk engine to each setup found
echo   3. Compare V5 Relaxed vs V5 Institutional
echo   4. Generate comprehensive report
echo.
echo Estimated time: 6-8 hours
echo.
echo ============================================================
echo.

REM Start monitor in new window
echo Starting monitor in new window...
start "Backtest Monitor" cmd /k "cd /d c:\quant_trading && .\venv\Scripts\activate && python monitor_fresh_backtest.py"

REM Wait a moment
timeout /t 3 /nobreak >nul

echo.
echo ============================================================
echo Starting backtest in 5 seconds...
echo Press Ctrl+C to cancel
echo ============================================================
timeout /t 5 /nobreak >nul

REM Run backtest
cd /d c:\quant_trading
.\venv\Scripts\activate
python run_complete_fresh_backtest.py

echo.
echo ============================================================
echo BACKTEST COMPLETE
echo ============================================================
echo.
echo View results:
echo   reports/complete_fresh_backtest/report.json
echo.
pause
