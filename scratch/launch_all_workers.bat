@echo off
echo ================================================
echo LAUNCHING PARALLEL BACKTEST WORKERS
echo ================================================
echo.
echo This will start 8 parallel workers processing
echo all 3,571 symbols simultaneously.
echo.
echo Each worker will run in its own window.
echo.
echo Press any key to start or Ctrl+C to cancel...
pause > nul

set NUM_WORKERS=8
set OUTPUT_DIR=reports\parallel_backtest

if not exist %OUTPUT_DIR% mkdir %OUTPUT_DIR%

echo.
echo Starting %NUM_WORKERS% workers...
echo.

for /L %%i in (0,1,7) do (
    echo Starting worker %%i of %NUM_WORKERS%...
    start "Worker %%i" cmd /k ".\venv\Scripts\activate && python run_batch_worker.py %%i %NUM_WORKERS%"
    timeout /t 2 /nobreak > nul
)

echo.
echo All workers started!
echo.
echo To monitor progress:
echo   - Check each worker window
echo   - View stats: type reports\parallel_backtest\worker_0\stats.json
echo.
echo To combine results when done:
echo   python combine_worker_results.py
echo.
pause
