"""
Tick-Based Backtesting Engine - Volume Exhaustion Strategy V5 (Hybrid)
Combines V3 entry frequency with V4 risk management.

Entry: 2 of 3 criteria (like V3) but with quality filters
Risk: Smart position sizing and trailing stops (like V4)
"""
from datetime import datetime, timedelta, time as dt_time
from typing import List, Dict, Optional
import pytz

from src.backtest.historical_tick_fetcher import tick_fetcher
from src.backtest.backtest_engine import BacktestResult, AuditRecord, ActionType


class TickBacktestEngineV5:
    """
    V5: Hybrid - Maximum trades with smart risk management
    
    Entry (2 of 3):
    - VWAP extension > 15%
    - Volume < 70% of recent peak
    - Price within 7% of HOD
    
    Quality filters (must pass):
    - Time: 9:45 AM - 2:00 PM
    - Stock up > 50% from open (true parabolic)
    - Price > VWAP (momentum intact)
    
    Risk Management:
    - 3% stop (tight)
    - Scale out: 40% VWAP, 35% -8%, 25% -15%
    - Trailing stop after TP1
    - Max 1 position per day (best setup only)
    """
    
    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.audit_records: List[AuditRecord] = []
        
        self.current_position: Optional[Dict] = None
        self.daily_trades = 0
        self.total_trades = 0
        
        # Criteria
        self.min_vwap_extension = 1.15
        self.vol_exhaustion = 0.70
        self.min_proximity = 0.93
        self.min_day_gain = 0.50  # 50% from open
        
        self.slippage_bps = 5.0
        self.et_tz = pytz.timezone('America/New_York')
    
    def reset(self):
        self.capital = self.initial_capital
        self.audit_records = []
        self.current_position = None
        self.daily_trades = 0
        self.total_trades = 0
    
    def run_tick_backtest(self, symbol: str, date: datetime, verbose: bool = True) -> BacktestResult:
        self.reset()
        
        tick_df = tick_fetcher.fetch_combined_tick_data(symbol, date, use_quotes=False)
        if tick_df.is_empty():
            return BacktestResult(symbol, date, date, audit_records=[])
        
        bar_df = tick_fetcher.aggregate_trades_to_bars(tick_df, interval_seconds=60)
        if bar_df.is_empty():
            return BacktestResult(symbol, date, date, audit_records=[])
        
        # Build bars with metrics
        bars = []
        cumulative_tp_v = 0.0
        cumulative_vol = 0.0
        day_high = 0.0
        day_open = 0.0
        
        # Convert to list and sort by timestamp to ensure consistent ordering
        bar_list = sorted(bar_df.to_dicts(), key=lambda x: x['timestamp'])
        for row in bar_list:
            ts = row['timestamp']
            ts_et = ts.astimezone(self.et_tz) if ts.tzinfo else ts
            
            typical_price = (row['high'] + row['low'] + row['close']) / 3
            cumulative_tp_v += typical_price * row['volume']
            cumulative_vol += row['volume']
            vwap = cumulative_tp_v / cumulative_vol if cumulative_vol > 0 else row['close']
            
            if day_open == 0:
                day_open = row['open']
            if row['high'] > day_high:
                day_high = row['high']
            
            bars.append({
                'timestamp': ts,
                'time_et': ts_et,
                'open': row['open'],
                'high': row['high'],
                'low': row['low'],
                'close': row['close'],
                'volume': row['volume'],
                'vwap': vwap,
                'day_open': day_open,
                'day_high': day_high
            })
        
        if verbose:
            print(f"\nV5: {symbol} {date.date()} | Open: ${day_open:.2f}, High: ${day_high:.2f}")
            gain = (day_high - day_open) / day_open * 100 if day_open > 0 else 0
            print(f"Day Gain: {gain:.1f}% | Entry: 2 of (VWAP>1.15x, Vol<70%, Prox>93%)")
        
        # Process bars
        volume_history = []
        best_setup = None
        
        for i, bar in enumerate(bars):
            t = bar['time_et']
            if not isinstance(t, datetime):
                continue
            
            # Update volume
            volume_history.append(bar['volume'])
            if len(volume_history) > 10:
                volume_history.pop(0)
            vol_peak = max(volume_history) if volume_history else bar['volume']
            
            # Update day high in all bars
            bar['day_high'] = day_high
            
            # Check if in position
            if self.current_position:
                self._check_exit(bar, verbose)
                continue
            
            # Entry window
            if not (dt_time(9, 45) <= t.time() <= dt_time(14, 0)):
                continue
            
            # Check quality filters
            day_gain = (bar['day_high'] - bar['day_open']) / bar['day_open'] if bar['day_open'] > 0 else 0
            if day_gain < self.min_day_gain:  # Must be true parabolic
                continue
            
            if bar['close'] < bar['vwap']:  # Must be above VWAP
                continue
            
            # Calculate 2-of-3 criteria
            vwap_ext = bar['close'] / bar['vwap'] if bar['vwap'] > 0 else 1.0
            vol_ratio = bar['volume'] / vol_peak if vol_peak > 0 else 1.0
            prox = bar['close'] / bar['day_high'] if bar['day_high'] > 0 else 0
            
            criteria_met = sum([
                vwap_ext >= self.min_vwap_extension,
                vol_ratio <= self.vol_exhaustion,
                prox >= self.min_proximity
            ])
            
            if criteria_met >= 2:
                # Track best setup (highest VWAP extension)
                if best_setup is None or vwap_ext > best_setup['vwap_ext']:
                    best_setup = {
                        'bar': bar,
                        'vwap_ext': vwap_ext,
                        'vol_ratio': vol_ratio,
                        'prox': prox,
                        'criteria': criteria_met
                    }
        
        # Debug: print bars around entry time
        if verbose and best_setup:
            bar = best_setup['bar']
            print(f"  [DEBUG] Best setup bar time: {bar['time_et'].strftime('%H:%M')}, close: ${bar['close']:.2f}")
        
        # Take best setup only
        if best_setup and self.current_position is None and self.daily_trades < 1:
            self._enter_position(best_setup['bar'], best_setup, verbose)
            # Check exit immediately after entry (same bar stop check)
            if self.current_position:
                self._check_exit(best_setup['bar'], verbose)
        
        # Force exit at end
        if self.current_position:
            last_bar = bars[-1]
            self._exit(last_bar, last_bar['close'], "time", verbose)
        
        result = self._generate_result(symbol, date, date)
        
        if verbose and result.total_trades > 0:
            win_rate = result.winning_trades / result.total_trades * 100 if result.total_trades > 0 else 0
            print(f"[RESULT] ${result.total_pnl:+.2f} | Win: {win_rate:.0f}%")
        
        return result
    
    def _enter_position(self, bar: Dict, setup: Dict, verbose: bool):
        """Enter with smart sizing."""
        price = bar['close']
        fill_price = price * 1.0005
        
        # Full position at once (no scaling for simplicity)
        position_value = 25000  # $25K per trade
        shares = int(position_value / fill_price)
        
        if shares <= 0:
            return
        
        # Tight 3% stop
        stop = fill_price * 1.03
        
        self.current_position = {
            'entry_price': fill_price,
            'shares': shares,
            'stop_loss': stop,
            'tp1_hit': False,
            'tp2_hit': False,
            'tp1_shares': int(shares * 0.40),
            'tp2_shares': int(shares * 0.35)
        }
        
        self.daily_trades += 1
        self.total_trades += 1
        
        audit = AuditRecord(
            timestamp=bar['timestamp'],
            symbol='TEST',
            action=ActionType.ENTRY,
            price=fill_price,
            vwap=bar['vwap'],
            vwap_extension=setup['vwap_ext'],
            volume_ratio=setup['vol_ratio'],
            add_level=1,
            shares=shares,
            total_shares=shares,
            avg_entry=fill_price,
            stop_loss=stop,
            reasoning=f"{setup['criteria']}/3 criteria, day gain {(bar['day_high']-bar['day_open'])/bar['day_open']*100:.0f}%"
        )
        self.audit_records.append(audit)
        
        if verbose:
            print(f"[ENTRY] {bar['time_et'].strftime('%H:%M')} @ ${fill_price:.2f}")
            print(f"  VWAP: {setup['vwap_ext']:.2f}x | Vol: {setup['vol_ratio']:.2f} | Prox: {setup['prox']:.2f}")
    
    def _check_exit(self, bar: Dict, verbose: bool):
        if not self.current_position:
            return
        
        pos = self.current_position
        price = bar['close']
        
        # Stop loss - use HIGH price for more realistic exit
        # If high touches stop, we exit at stop price
        if bar.get('high', price) >= pos['stop_loss']:
            self._exit(bar, pos['stop_loss'], "stop", verbose)
            return
        
        depreciation = (pos['entry_price'] - price) / pos['entry_price'] * 100
        
        # TP1: VWAP (40%)
        if not pos['tp1_hit'] and price <= bar['vwap']:
            self._exit_partial(bar, price, "tp1", pos['tp1_shares'], verbose)
            pos['tp1_hit'] = True
            # Move stop to breakeven
            pos['stop_loss'] = pos['entry_price']
            return
        
        # TP2: -8% (35%)
        remaining = pos['shares'] - self._get_closed_shares()
        if pos['tp1_hit'] and not pos['tp2_hit'] and depreciation >= 8.0:
            self._exit_partial(bar, price, "tp2", min(pos['tp2_shares'], remaining), verbose)
            pos['tp2_hit'] = True
            return
        
        # TP3: -15% or time
        t = bar['time_et']
        time_exit = t.time() >= dt_time(14, 30) if isinstance(t, datetime) else False
        if depreciation >= 15.0 or time_exit:
            remaining = pos['shares'] - self._get_closed_shares()
            if remaining > 0:
                self._exit_partial(bar, price, "tp3" if not time_exit else "time", remaining, verbose)
                self.current_position = None
    
    def _exit_partial(self, bar: Dict, price: float, reason: str, shares: int, verbose: bool):
        if not self.current_position or shares <= 0:
            return
        
        pos = self.current_position
        fill_price = price * 0.9995
        pnl = (pos['entry_price'] - fill_price) * shares
        self.capital += pnl
        
        action_map = {'tp1': ActionType.TP1_EXIT, 'tp2': ActionType.TP2_EXIT, 
                      'tp3': ActionType.TP3_EXIT, 'time': ActionType.TIME_EXIT, 'stop': ActionType.STOP_EXIT}
        
        audit = AuditRecord(
            timestamp=bar['timestamp'],
            symbol='TEST',
            action=action_map.get(reason, ActionType.TIME_EXIT),
            price=fill_price,
            shares=shares,
            avg_entry=pos['entry_price'],
            exit_price=fill_price,
            exit_time=bar['timestamp'],
            pnl=pnl,
            exit_reason=reason
        )
        self.audit_records.append(audit)
        
        if verbose:
            pnl_pct = (pnl / (pos['entry_price'] * shares)) * 100
            print(f"  [{reason.upper()}] {shares} @ ${fill_price:.2f} | ${pnl:+.2f} ({pnl_pct:+.1f}%)")
    
    def _exit(self, bar: Dict, price: float, reason: str, verbose: bool):
        if not self.current_position:
            return
        remaining = self.current_position['shares'] - self._get_closed_shares()
        if remaining > 0:
            self._exit_partial(bar, price, reason, remaining, verbose)
        self.current_position = None
    
    def _get_closed_shares(self) -> int:
        return sum(r.shares for r in self.audit_records if r.action in [
            ActionType.TP1_EXIT, ActionType.TP2_EXIT, ActionType.TP3_EXIT, ActionType.STOP_EXIT
        ])
    
    def _generate_result(self, symbol: str, start: datetime, end: datetime) -> BacktestResult:
        entries = [r for r in self.audit_records if r.action == ActionType.ENTRY]
        exits = [r for r in self.audit_records if r.pnl is not None]
        
        total_pnl = sum(e.pnl for e in exits) if exits else 0.0
        winning = sum(1 for e in exits if e.pnl > 0)
        losing = sum(1 for e in exits if e.pnl <= 0)
        
        return BacktestResult(
            symbol=symbol, start_date=start, end_date=end,
            total_trades=len(entries), total_adds=0,
            winning_trades=winning, losing_trades=losing,
            total_pnl=total_pnl, win_rate=winning / len(entries) if entries else 0.0,
            profit_factor=0.0, average_trade=total_pnl / len(entries) if entries else 0.0,
            average_win=0.0, average_loss=0.0, audit_records=self.audit_records
        )


tick_backtest_engine_v5 = TickBacktestEngineV5()
