@echo off
echo ================================================
echo CACHED-ONLY PARALLEL BACKTEST
echo ================================================
echo.
echo This will process only symbols with cached tick
echo data using all CPU cores in parallel.
echo.
echo Cached symbols: ~719
echo Estimated time: 2-3 hours
echo.
echo Press any key to start...
pause > nul

.\venv\Scripts\activate && python run_cached_only_parallel.py

echo.
echo Backtest complete!
echo Results in: reports\cached_parallel_backtest\
pause
