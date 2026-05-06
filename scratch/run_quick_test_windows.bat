@echo off
echo ============================================
echo RL Quick Test (1-2 hours) - Windows Version
echo ============================================
echo.
echo This requires Python with ray, polars, torch installed.
echo If you get "module not found" errors, use run_quick_test.bat instead (WSL version).
echo.
pause

cd /d C:\quant_trading

echo.
echo Starting quick test in Windows Python...
echo.

python src\scripts\train_wfo_quick_test.py

echo.
echo ============================================
echo Test complete! Check results above.
echo ============================================
pause
