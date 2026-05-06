@echo off
echo ================================================
echo FULL 3,571 SYMBOL BACKTEST WITH ML RISK ENGINE
echo ================================================
echo.
echo This will run the complete fresh backtest using
echo ALL 3,571 symbols from the extended universe.
echo.
echo Estimated runtime: 8-12 hours
echo Output will be saved to: reports/full_3571_backtest/
echo.
echo Press any key to start or Ctrl+C to cancel...
pause > nul

.\venv\Scripts\activate && python run_full_3571_backtest.py

echo.
echo Backtest complete!
echo Results saved to: reports/full_3571_backtest/
pause
