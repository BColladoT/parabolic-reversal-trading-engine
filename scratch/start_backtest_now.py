
import sys
sys.path.insert(0, '.')
from run_complete_fresh_backtest import FreshBacktestEngine
from datetime import datetime

# Monkey-patch input to auto-confirm
original_input = input
def auto_input(prompt):
    print(f'{prompt} yes')
    return 'yes'
__builtins__.input = auto_input

# Run
start_date = datetime(2019, 1, 1)
end_date = datetime(2024, 12, 31)
engine = FreshBacktestEngine(start_date, end_date)
engine.run_complete_backtest()
