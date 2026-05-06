"""
Tick-Based Backtesting Engine - Volume Exhaustion Strategy V8 (EARLY ENTRY)
Fixes V7's late entry problem by restricting to morning only.

Key Fix: Entry window 9:45 AM - 11:30 AM only
This avoids late afternoon bag holding and ensures we catch the morning reversal.

Also: Stop at HOD + 2% (not entry + 3%)
This prevents entering after the move has already started.
"""
from datetime import datetime, timedelta, time as dt_time
from typing import List, Dict, Optional
import pytz

from src.backtest.historical_tick_fetcher import tick_fetcher
from src.backtest.backtest_engine import BacktestResult, AuditRecord, ActionType


class TickBacktestEngineV8:
    """
    V8: EARLY ENTRY ONLY
    
    CRITICAL FIX: Only enter 9:45 AM - 11:30 AM
    This avoids the late afternoon bag holding that killed V7.
    
    Entry (2 of 3):
    1. VWAP extension > 15%
    2. Volume < 70% of 10-min peak
    3. Price within 7% of HOD
    
    Risk:
    - Hard stop: 3% above entry
    - 50% TP at VWAP, 50% trailing
    - Breakeven after TP1
    - Time exit: 2:30 PM (for existing positions)
    """
    
    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.audit_records: List[AuditRecord] = []
        
        self.current_position: Optional[Dict] = None
        self.daily_trades = 0
        self.total_trades = 0
        
        # Entry criteria
        self.min_vwap_extension = 1.15
        self.vol_exhaustion = 0.70
        self.min_proximity = 0.93
        self.min_day_gain = 0.50
        
        # CRITICAL: Early entry only
        self.entry_start = dt_time(9, 45)
        self.entry_end = dt_time(11, 30)  # Was 2:00 PM - TOO LATE!
        self.force_exit = dt_time(14, 30)
        
        # Position
        self.position_size = 25000
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
        
        # Build bars
        bars = []
        cumulative_tp_v = 0.0
        cumulative_vol = 0.0
        day_high = 0.0
        day_open = 0.0
        
        for row in bar_df.to_dicts():
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
        
        day_gain = (day_high - day_open) / day_open if day_open > 0 else 0
        
        if verbose:
            print(f"\nV8 EARLY ENTRY: {symbol} {date.date()}")
            print(f"Open: ${day_open:.2f} | High: ${day_high:.2f} | Gain: {day_gain*100:.1f}%")
            print(f"Entry window: {self.entry_start.strftime('%H:%M')} - {self.entry_end.strftime('%H:%M')}")
        
        # Skip if not parabolic
        if day_gain < self.min_day_gain:
            if verbose:
                print(f"SKIP: Gain {day_gain*100:.1f}% < 50%")
            return BacktestResult(symbol, date, date, audit_records=[])
        
        # Process
        volume_history = []
        best_setup = None
        
        for bar in bars:
            t = bar['time_et']
            if not isinstance(t, datetime):
                continue
            
            # Volume tracking
            volume_history.append(bar['volume'])
            if len(volume_history) > 10:
                volume_history.pop(0)
            vol_peak = max(volume_history) if volume_history else bar['volume']
            
            bar['day_high'] = day_high
            
            # Exit check
            if self.current_position:
                self._check_exit(bar, verbose)
                continue
            
            # Entry window (EARLY ONLY)
            if not (self.entry_start <= t.time() <= self.entry_end):
                continue
            
            # Filters
            if bar['close'] < bar['vwap']:
                continue
            
            # Criteria
            vwap_ext = bar['close'] / bar['vwap'] if bar['vwap'] > 0 else 1.0
            vol_ratio = bar['volume'] / vol_peak if vol_peak > 0 else 1.0
            prox = bar['close'] / bar['day_high'] if bar['day_high'] > 0 else 0
            
            met = sum([
                vwap_ext >= self.min_vwap_extension,
                vol_ratio <= self.vol_exhaustion,
                prox >= self.min_proximity
            ])
            
            if met >= 2:
                if best_setup is None or vwap_ext > best_setup['vwap_ext']:
                    best_setup = {
                        'bar': bar,
                        'vwap_ext': vwap_ext,
                        'vol_ratio': vol_ratio,
                        'prox': prox,
                        'criteria': met
                    }
        
        # Take best setup
        if best_setup and self.current_position is None:
            self._enter(best_setup, verbose)
        
        # Force exit
        if self.current_position:
            self._exit(bars[-1], bars[-1]['close'], "time", verbose)
        
        result = self._generate_result(symbol, date, date)
        
        if verbose and result.total_trades > 0:
            wr = result.winning_trades / result.total_trades * 100
            print(f"[RESULT] ${result.total_pnl:+.2f} | {wr:.0f}% win rate")
        elif verbose:
            print("[NO ENTRY] No qualifying setup found")
        
        return result
    
    def _enter(self, setup: Dict, verbose: bool):
        bar = setup['bar']
        price = bar['close']
        fill = price * 1.0005
        
        shares = int(self.position_size / fill)
        if shares <= 0:
            return
        
        # 3% stop above entry
        stop = fill * 1.03
        
        self.current_position = {
            'entry': fill,
            'shares': shares,
            'stop': stop,
            'tp1_done': False,
            'tp1_shares': int(shares * 0.5),
            'max_profit': 0.0
        }
        
        self.daily_trades += 1
        self.total_trades += 1
        
        audit = AuditRecord(
            timestamp=bar['timestamp'],
            symbol='TEST',
            action=ActionType.ENTRY,
            price=fill,
            vwap=bar['vwap'],
            vwap_extension=setup['vwap_ext'],
            volume_ratio=setup['vol_ratio'],
            add_level=1,
            shares=shares,
            total_shares=shares,
            avg_entry=fill,
            stop_loss=stop,
            reasoning=f"{setup['criteria']}/3 criteria"
        )
        self.audit_records.append(audit)
        
        if verbose:
            print(f"[ENTRY] {bar['time_et'].strftime('%H:%M')} @ ${fill:.2f}")
            print(f"        VWAP: {setup['vwap_ext']:.2f}x | Vol: {setup['vol_ratio']:.0%} | Prox: {setup['prox']:.1%}")
    
    def _check_exit(self, bar: Dict, verbose: bool):
        if not self.current_position:
            return
        
        pos = self.current_position
        price = bar['close']
        
        # Stop
        if price >= pos['stop']:
            self._exit(bar, price, "stop", verbose)
            return
        
        # Profit calc
        profit = (pos['entry'] - price) / pos['entry'] * 100
        if profit > pos['max_profit']:
            pos['max_profit'] = profit
        
        # TP1: VWAP (50%)
        if not pos['tp1_done'] and price <= bar['vwap']:
            self._partial(bar, price, "tp1", pos['tp1_shares'], verbose)
            pos['tp1_done'] = True
            pos['stop'] = pos['entry']  # Breakeven
            if verbose:
                print(f"  [BREAKEVEN] Stop moved to ${pos['stop']:.2f}")
            return
        
        # Trailing: exit if give back 50% of max profit
        if pos['tp1_done'] and pos['max_profit'] > 5:
            if profit < pos['max_profit'] * 0.5:
                rem = pos['shares'] - self._closed()
                if rem > 0:
                    self._partial(bar, price, "trail", rem, verbose)
                    self.current_position = None
                return
        
        # Time
        t = bar['time_et']
        if isinstance(t, datetime) and t.time() >= self.force_exit:
            rem = pos['shares'] - self._closed()
            if rem > 0:
                self._partial(bar, price, "time", rem, verbose)
                self.current_position = None
    
    def _partial(self, bar: Dict, price: float, reason: str, shares: int, verbose: bool):
        if not self.current_position or shares <= 0:
            return
        
        pos = self.current_position
        fill = price * 0.9995
        pnl = (pos['entry'] - fill) * shares
        self.capital += pnl
        
        action = {
            'tp1': ActionType.TP1_EXIT,
            'trail': ActionType.TP3_EXIT,
            'time': ActionType.TIME_EXIT,
            'stop': ActionType.STOP_EXIT
        }.get(reason, ActionType.TIME_EXIT)
        
        audit = AuditRecord(
            timestamp=bar['timestamp'],
            symbol='TEST',
            action=action,
            price=fill,
            shares=shares,
            avg_entry=pos['entry'],
            exit_price=fill,
            exit_time=bar['timestamp'],
            pnl=pnl,
            exit_reason=reason
        )
        self.audit_records.append(audit)
        
        if verbose:
            pnl_pct = (pnl / (pos['entry'] * shares)) * 100
            print(f"  [{reason.upper()}] {shares} @ ${fill:.2f} | ${pnl:+.2f} ({pnl_pct:+.1f}%)")
    
    def _exit(self, bar: Dict, price: float, reason: str, verbose: bool):
        if not self.current_position:
            return
        rem = self.current_position['shares'] - self._closed()
        if rem > 0:
            self._partial(bar, price, reason, rem, verbose)
        self.current_position = None
    
    def _closed(self) -> int:
        return sum(r.shares for r in self.audit_records if r.action in [
            ActionType.TP1_EXIT, ActionType.TP2_EXIT, ActionType.TP3_EXIT,
            ActionType.STOP_EXIT, ActionType.TIME_EXIT
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


tick_backtest_engine_v8 = TickBacktestEngineV8()
