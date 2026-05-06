"""
Tick-Based Backtesting Engine - Volume Exhaustion Strategy V2
More flexible volume detection for historical data.
"""
from datetime import datetime, timedelta, time as dt_time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json

import polars as pl
import numpy as np
import pytz

from src.utils.config import CONFIG
from src.utils.logger import logger
from src.indicators.numba_kernels import (
    calculate_vwap_numba,
    calculate_atr_numba,
    calculate_position_size_numba
)
from src.backtest.historical_tick_fetcher import tick_fetcher
from src.backtest.backtest_engine import BacktestResult, AuditRecord, ActionType


class TickBacktestEngineV2:
    """
    V2: More flexible volume exhaustion detection.
    - Uses rolling 5-min volume average
    - Compares to recent peak (not just opening)
    - Relaxed criteria for historical testing
    """
    
    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.audit_records: List[AuditRecord] = []
        
        self.current_position: Optional[Dict] = None
        self.daily_trades = 0
        self.total_trades = 0
        
        # Volume tracking - rolling window
        self.recent_volumes: List[float] = []
        self.volume_window = 10  # 10-minute rolling window
        self.volume_peak_recent = 0.0
        
        # Price tracking
        self.day_high = 0.0
        self.day_open = 0.0
        
        # VWAP
        self.cumulative_tp_v = 0.0
        self.cumulative_vol = 0.0
        
        # Slippage
        self.entry_slippage_bps = 5.0
        self.exit_slippage_bps = 5.0
        
        # RELAXED criteria for testing
        self.min_vwap_extension = 1.15  # 15% above VWAP (was 20%)
        self.vol_exhaustion_threshold = 0.70  # 70% of recent peak (was 60%)
        self.proximity_threshold = 0.93  # Within 7% of HOD (was 5%)
    
    def reset(self):
        self.capital = self.initial_capital
        self.audit_records = []
        self.current_position = None
        self.daily_trades = 0
        self.total_trades = 0
        self.recent_volumes = []
        self.volume_peak_recent = 0.0
        self.day_high = 0.0
        self.day_open = 0.0
        self.cumulative_tp_v = 0.0
        self.cumulative_vol = 0.0
    
    def run_tick_backtest(self, symbol: str, date: datetime, verbose: bool = True) -> BacktestResult:
        self.reset()
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"BACKTEST V2: {symbol} on {date.date()}")
            print(f"Strategy: Volume Exhaustion (Relaxed Criteria)")
            print(f"{'='*70}\n")
        
        tick_df = tick_fetcher.fetch_combined_tick_data(symbol, date, use_quotes=False)
        
        if tick_df.is_empty():
            return BacktestResult(symbol, date, date, audit_records=[])
        
        bar_df = tick_fetcher.aggregate_trades_to_bars(tick_df, interval_seconds=60)
        
        if bar_df.is_empty():
            return BacktestResult(symbol, date, date, audit_records=[])
        
        # Build bar lookup with VWAP
        bar_lookup = {}
        for row in bar_df.to_dicts():
            minute = row['timestamp'].replace(second=0, microsecond=0)
            
            # Update VWAP
            typical_price = (row['high'] + row['low'] + row['close']) / 3
            self.cumulative_tp_v += typical_price * row['volume']
            self.cumulative_vol += row['volume']
            vwap = self.cumulative_tp_v / self.cumulative_vol if self.cumulative_vol > 0 else row['close']
            
            bar_lookup[minute] = {
                'open': row['open'],
                'high': row['high'],
                'low': row['low'],
                'close': row['close'],
                'volume': row['volume'],
                'vwap': vwap
            }
            
            if row['high'] > self.day_high:
                self.day_high = row['high']
            if self.day_open == 0.0:
                self.day_open = row['open']
        
        if verbose:
            print(f"Loaded {len(tick_df):,} trades, {len(bar_df)} bars")
            print(f"Day Open: ${self.day_open:.2f}, Day High: ${self.day_high:.2f}")
            print(f"Criteria: VWAP > {self.min_vwap_extension:.2f}x, Vol < {self.vol_exhaustion_threshold:.0%} recent, Price > {self.proximity_threshold:.0%} HOD")
        
        # Process ticks
        trades_executed = 0
        volume_history = []  # Track last 10 minutes of volume
        
        for tick in tick_df.to_dicts():
            tick_time = tick['timestamp']
            et_time = self._get_et_time(tick_time)
            current_price = tick['trade_price']
            current_minute = tick_time.replace(second=0, microsecond=0)
            
            # Update day high
            if current_price > self.day_high:
                self.day_high = current_price
            
            bar_data = bar_lookup.get(current_minute)
            if bar_data is None:
                continue
            
            # Update volume history
            if bar_data['volume'] > 0:
                volume_history.append(bar_data['volume'])
                if len(volume_history) > self.volume_window:
                    volume_history.pop(0)
                self.volume_peak_recent = max(volume_history) if volume_history else bar_data['volume']
            
            # Check if in entry window
            if not self._in_execution_window(et_time):
                if self.current_position:
                    self._evaluate_exit(tick, bar_data, et_time, verbose)
                continue
            
            # Check daily limit
            if self.daily_trades >= CONFIG.risk.max_daily_trades:
                if self.current_position:
                    self._evaluate_exit(tick, bar_data, et_time, verbose)
                continue
            
            # Trading logic
            if self.current_position is None:
                if self._evaluate_entry(tick, bar_data, volume_history, et_time, verbose):
                    trades_executed += 1
            else:
                self._evaluate_add(tick, bar_data, volume_history, et_time, verbose)
                self._evaluate_exit(tick, bar_data, et_time, verbose)
        
        # Force close
        if self.current_position:
            last_tick = tick_df.to_dicts()[-1]
            self._force_exit(last_tick, "time_exit", verbose)
        
        result = self._generate_result(symbol, date, date)
        
        if verbose:
            if result.total_trades > 0:
                print(f"\n[TRADES] {result.total_trades} entries, {result.total_adds} adds")
                print(f"  P&L: ${result.total_pnl:+.2f}")
            else:
                print(f"\n[NO TRADES] Criteria not met")
        
        return result
    
    def _get_et_time(self, timestamp: datetime) -> datetime:
        if isinstance(timestamp, datetime) and timestamp.tzinfo is not None:
            et_tz = pytz.timezone('America/New_York')
            return timestamp.astimezone(et_tz)
        return timestamp
    
    def _in_execution_window(self, et_time: datetime) -> bool:
        if isinstance(et_time, datetime):
            t = et_time.time()
        else:
            t = et_time
        return dt_time(9, 45) <= t <= dt_time(14, 30)
    
    def _evaluate_entry(self, tick: Dict, bar_data: Dict, volume_history: List, et_time: datetime, verbose: bool) -> bool:
        """Evaluate entry with relaxed criteria."""
        price = tick['trade_price']
        vwap = bar_data.get('vwap', price)
        
        if vwap <= 0 or not volume_history:
            return False
        
        vwap_extension = price / vwap
        current_vol = bar_data['volume']
        recent_peak = max(volume_history) if volume_history else current_vol
        vol_ratio = current_vol / recent_peak if recent_peak > 0 else 1.0
        prox = price / self.day_high if self.day_high > 0 else 0
        
        # RELAXED CRITERIA:
        # 1. VWAP extension > 115% (was 120%)
        # 2. Volume < 70% of recent 10-min peak (was 60% of opening peak)
        # 3. Price within 7% of day's high (was 5%)
        
        vwap_ok = vwap_extension >= self.min_vwap_extension
        vol_ok = vol_ratio <= self.vol_exhaustion_threshold
        prox_ok = prox >= self.proximity_threshold
        
        if not (vwap_ok and vol_ok and prox_ok):
            return False
        
        # Execute entry
        fill_price = price * (1 + self.entry_slippage_bps / 10000)
        stop_loss = fill_price * 1.04  # 4% stop
        
        position_value = 30000 * 0.25  # $7,500 initial
        position_size = int(position_value / fill_price)
        
        if position_size <= 0:
            return False
        
        self.current_position = {
            'entries': [{'price': fill_price, 'shares': position_size}],
            'avg_entry': fill_price,
            'total_shares': position_size,
            'add_level': 1,
            'stop_loss': stop_loss,
            'highest_price': fill_price,
            'lowest_volume': vol_ratio,
            'tp1_hit': False,
            'tp2_hit': False,
            'tp3_hit': False
        }
        
        self.daily_trades += 1
        self.total_trades += 1
        
        audit = AuditRecord(
            timestamp=tick['timestamp'],
            symbol=tick['symbol'],
            action=ActionType.ENTRY,
            price=fill_price,
            vwap=vwap,
            vwap_extension=vwap_extension,
            volume_ratio=vol_ratio,
            add_level=1,
            shares=position_size,
            total_shares=position_size,
            avg_entry=fill_price,
            stop_loss=stop_loss,
            reasoning=f"VWAP:{vwap_extension:.2f}x Vol:{vol_ratio:.2f} Prox:{prox:.2f}"
        )
        self.audit_records.append(audit)
        
        if verbose:
            print(f"\n[ENTRY] {et_time.strftime('%H:%M:%S')} @ ${fill_price:.2f}")
            print(f"  VWAP: {vwap_extension:.2f}x | Vol: {vol_ratio:.2f} | Prox: {prox:.2f}")
        
        return True
    
    def _evaluate_add(self, tick: Dict, bar_data: Dict, volume_history: List, et_time: datetime, verbose: bool):
        """Evaluate adding to position."""
        if not self.current_position:
            return
        
        pos = self.current_position
        if pos['add_level'] >= 3:
            return
        
        price = tick['trade_price']
        current_vol = bar_data['volume']
        recent_peak = max(volume_history) if volume_history else current_vol
        vol_ratio = current_vol / recent_peak if recent_peak > 0 else 1.0
        
        # Must make new high on lower volume
        if price <= pos['highest_price']:
            return
        if vol_ratio >= pos['lowest_volume']:
            return
        
        # Add
        fill_price = price * (1 + self.entry_slippage_bps / 10000)
        add_shares = int(7500 / fill_price) if pos['add_level'] == 1 else int(15000 / fill_price)
        
        if add_shares <= 0:
            return
        
        pos['entries'].append({'price': fill_price, 'shares': add_shares})
        total_cost = sum(e['price'] * e['shares'] for e in pos['entries'])
        pos['total_shares'] = sum(e['shares'] for e in pos['entries'])
        pos['avg_entry'] = total_cost / pos['total_shares']
        pos['add_level'] += 1
        pos['highest_price'] = price
        pos['lowest_volume'] = vol_ratio
        pos['stop_loss'] = pos['avg_entry'] * 1.035  # 3.5% stop
        
        self.daily_trades += 1
        
        audit = AuditRecord(
            timestamp=tick['timestamp'],
            symbol=tick['symbol'],
            action=ActionType.ADD,
            price=fill_price,
            volume_ratio=vol_ratio,
            add_level=pos['add_level'],
            shares=add_shares,
            total_shares=pos['total_shares'],
            avg_entry=pos['avg_entry'],
            stop_loss=pos['stop_loss']
        )
        self.audit_records.append(audit)
        
        if verbose:
            print(f"[ADD #{pos['add_level']}] @ ${fill_price:.2f} | Total: {pos['total_shares']}")
    
    def _evaluate_exit(self, tick: Dict, bar_data: Dict, et_time: datetime, verbose: bool):
        """Evaluate exits."""
        if not self.current_position:
            return
        
        pos = self.current_position
        price = tick['trade_price']
        vwap = bar_data.get('vwap', pos['avg_entry'])
        
        # Stop loss
        if price >= pos['stop_loss']:
            self._execute_exit(tick, price, et_time, "stop_exit", pos['total_shares'], verbose)
            return
        
        depreciation = (pos['avg_entry'] - price) / pos['avg_entry'] * 100
        remaining = pos['total_shares'] - self._get_closed_shares()
        
        if remaining <= 0:
            return
        
        # TP1: VWAP
        if not pos['tp1_hit'] and price <= vwap:
            shares = int(pos['total_shares'] * 0.35)
            self._execute_exit(tick, price, et_time, "tp1_exit", shares, verbose)
            pos['tp1_hit'] = True
            return
        
        # TP2: -8%
        if not pos['tp2_hit'] and depreciation >= 8.0:
            shares = int(pos['total_shares'] * 0.35)
            self._execute_exit(tick, price, et_time, "tp2_exit", shares, verbose)
            pos['tp2_hit'] = True
            return
        
        # TP3: -15% or time
        time_exit = et_time.time() >= dt_time(15, 25) if isinstance(et_time, datetime) else et_time >= dt_time(15, 25)
        if not pos['tp3_hit'] and (depreciation >= 15.0 or time_exit):
            self._execute_exit(tick, price, et_time, "tp3_exit" if not time_exit else "time_exit", remaining, verbose)
            pos['tp3_hit'] = True
    
    def _get_closed_shares(self) -> int:
        return sum(r.shares for r in self.audit_records if r.action in [
            ActionType.TP1_EXIT, ActionType.TP2_EXIT, ActionType.TP3_EXIT, ActionType.STOP_EXIT
        ])
    
    def _execute_exit(self, tick: Dict, price: float, et_time: datetime, reason: str, shares: int, verbose: bool):
        if not self.current_position or shares <= 0:
            return
        
        pos = self.current_position
        fill_price = price * (1 - self.exit_slippage_bps / 10000)
        pnl = (pos['avg_entry'] - fill_price) * shares
        self.capital += pnl
        
        action_map = {
            'stop_exit': ActionType.STOP_EXIT,
            'tp1_exit': ActionType.TP1_EXIT,
            'tp2_exit': ActionType.TP2_EXIT,
            'tp3_exit': ActionType.TP3_EXIT,
            'time_exit': ActionType.TIME_EXIT
        }
        
        audit = AuditRecord(
            timestamp=tick['timestamp'],
            symbol=tick['symbol'],
            action=action_map.get(reason, ActionType.TIME_EXIT),
            price=fill_price,
            shares=shares,
            avg_entry=pos['avg_entry'],
            exit_price=fill_price,
            exit_time=tick['timestamp'],
            pnl=pnl,
            exit_reason=reason
        )
        self.audit_records.append(audit)
        
        if verbose:
            pnl_pct = (pnl / (pos['avg_entry'] * shares)) * 100
            print(f"[{reason.upper()}] @ ${fill_price:.2f} | P&L: ${pnl:+.2f} ({pnl_pct:+.1f}%)")
        
        if self._get_closed_shares() >= pos['total_shares']:
            self.current_position = None
    
    def _force_exit(self, tick: Dict, reason: str, verbose: bool):
        if not self.current_position:
            return
        remaining = self.current_position['total_shares'] - self._get_closed_shares()
        if remaining > 0:
            et_time = self._get_et_time(tick['timestamp'])
            self._execute_exit(tick, tick['trade_price'], et_time, reason, remaining, verbose)
    
    def _generate_result(self, symbol: str, start: datetime, end: datetime) -> BacktestResult:
        entries = [r for r in self.audit_records if r.action == ActionType.ENTRY]
        adds = [r for r in self.audit_records if r.action == ActionType.ADD]
        exits = [r for r in self.audit_records if r.pnl is not None]
        
        total_pnl = sum(e.pnl for e in exits) if exits else 0.0
        winning_trades = sum(1 for e in exits if e.pnl > 0)
        losing_trades = sum(1 for e in exits if e.pnl <= 0)
        
        wins = [e.pnl for e in exits if e.pnl > 0]
        losses = [e.pnl for e in exits if e.pnl <= 0]
        
        return BacktestResult(
            symbol=symbol,
            start_date=start,
            end_date=end,
            total_trades=len(entries),
            total_adds=len(adds),
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            total_pnl=total_pnl,
            win_rate=winning_trades / len(entries) if entries else 0.0,
            profit_factor=abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else float('inf'),
            average_trade=total_pnl / len(entries) if entries else 0.0,
            average_win=sum(wins) / len(wins) if wins else 0.0,
            average_loss=sum(losses) / len(losses) if losses else 0.0,
            audit_records=self.audit_records
        )


# Create instance
tick_backtest_engine_v2 = TickBacktestEngineV2()
