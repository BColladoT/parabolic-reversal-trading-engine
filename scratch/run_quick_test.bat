@echo off
echo ============================================
echo RL Quick Test (1-2 hours)
echo ============================================
echo.
echo This will run a shortened training to verify:
echo 1. Data provider loads correctly
echo 2. Environment runs without errors
echo 3. Agent learns (non-zero PnL)
echo.
echo Parameters:
echo   - 1 fold (vs 4 in full)
echo   - 6 months train, 1 month test
echo   - 20,000 total timesteps (vs 70,000 in full)
echo   - ~20-30 minutes per phase
echo.
pause

echo.
echo Starting quick test in WSL...
echo.

:: Run the test in WSL with proper command structure
wsl bash -lc "cd /mnt/c/quant_trading && source venv_wsl/bin/activate && cd src/scripts && python train_wfo_quick_test.py"

echo.
echo ============================================
echo Test complete! Check results above.
echo ============================================
pause
