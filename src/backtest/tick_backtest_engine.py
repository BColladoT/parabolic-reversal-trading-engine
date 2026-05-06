"""
Tick-Based Backtesting Engine - Progressive Exhaustion Scale-In Strategy
Uses actual historical trade data for ultra-accurate simulation.
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


class TickBacktestEngine:
    """
    High-precision backtesting using actual historical tick (trade) data.
    Implements Progressive Exhaustion Scale-In strategy.
    """
    
    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.audit_records: List[AuditRecord] = []
        
        # Position tracking
        self.current_position: Optional[Dict] = None
        self.daily_trades = 0
        self.total_trades = 0
        
        # Volume tracking for the day
        self.minute_volumes: Dict[datetime, float] = {}
        self.volume_peak = 0.0
        self.peak_established = False
        self.peak_establishment_minutes = 30  # First 30 min
        
        # Price tracking
        self.day_high = 0.0
        self.day_open = 0.0
        
        # VWAP calculation
        self.cumulative_tp_v = 0.0  # Cumulative typical price * volume
        self.cumulative_vol = 0.0   # Cumulative volume
        
        # Slippage
        self.entry_slippage_bps = 5.0
        self.exit_slippage_bps = 5.0
    
    def reset(self):
        """Reset engine state."""
        self.capital = self.initial_capital
        self.audit_records = []
        self.current_position = None
        self.daily_trades = 0
        self.total_trades = 0
        self.minute_volumes = {}
        self.volume_peak = 0.0
        self.peak_established = False
        self.day_high = 0.0
        self.day_open = 0.0
        self.cumulative_tp_v = 0.0
        self.cumulative_vol = 0.0
    
    def run_tick_backtest(
        self,
        symbol: str,
        date: datetime,
        verbose: bool = True
    ) -> BacktestResult:
        """
        Run tick-level backtest for a single day with new strategy.
        """
        self.reset()
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"TICK-LEVEL BACKTEST: {symbol} on {date.date()}")
            print(f"Strategy: Progressive Exhaustion Scale-In")
            print(f"{'='*70}\n")
        
        # Fetch tick data
        tick_df = tick_fetcher.fetch_combined_tick_data(symbol, date, use_quotes=False)
        
        if tick_df.is_empty():
            logger.warning(f"No tick data for {symbol} on {date.date()}")
            return BacktestResult(symbol, date, date, audit_records=[])
        
        # Pre-aggregate to bars for volume tracking
        bar_df = tick_fetcher.aggregate_trades_to_bars(tick_df, interval_seconds=60)
        
        if bar_df.is_empty():
            return BacktestResult(symbol, date, date, audit_records=[])
        
        # Build volume lookup by minute
        for row in bar_df.to_dicts():
            minute = row['timestamp'].replace(second=0, microsecond=0)
            self.minute_volumes[minute] = {
                'volume': row['volume'],
                'high': row['high'],
                'low': row['low'],
                'close': row['close'],
                'vwap': 0.0  # Will calculate
            }
            
            # Track day high
            if row['high'] > self.day_high:
                self.day_high = row['high']
            
            # Track open
            if self.day_open == 0.0:
                self.day_open = row['open']
            
            # Update VWAP calculation
            typical_price = (row['high'] + row['low'] + row['close']) / 3
            self.cumulative_tp_v += typical_price * row['volume']
            self.cumulative_vol += row['volume']
            
            # Calculate VWAP up to this minute
            if self.cumulative_vol > 0:
                self.minute_volumes[minute]['vwap'] = self.cumulative_tp_v / self.cumulative_vol
        
        # Calculate volume peak from first N minutes
        market_open = None
        for ts in sorted(self.minute_volumes.keys()):
            if market_open is None:
                market_open = ts
            elapsed_minutes = (ts - market_open).total_seconds() / 60
            
            if elapsed_minutes <= self.peak_establishment_minutes:
                vol = self.minute_volumes[ts]['volume']
                if vol > self.volume_peak:
                    self.volume_peak = vol
        
        if verbose:
            print(f"Loaded {len(tick_df):,} trades")
            print(f"Day Open: ${self.day_open:.2f}, Day High: ${self.day_high:.2f}")
            print(f"Volume Peak (first 30 min): {self.volume_peak:,.0f}")
            print(f"Entry Window: 9:45 AM - 2:30 PM ET")
            print(f"Entry Criteria: VWAP > 1.20x, Vol < 60% peak, Price > 95% HOD\n")
        
        # Skip if no significant volume peak
        if self.volume_peak == 0:
            if verbose:
                print("No volume peak established - skipping")
            return BacktestResult(symbol, date, date, audit_records=[])
        
        # Simulate tick-by-tick
        last_minute = None
        trades_executed = 0
        
        for tick in tick_df.to_dicts():
            tick_time = tick['timestamp']
            current_price = tick['trade_price']
            
            # Get ET time
            et_time = self._get_et_time(tick_time)
            
            # Update day high
            if current_price > self.day_high:
                self.day_high = current_price
            
            # Get current minute bar data
            current_minute = tick_time.replace(second=0, microsecond=0)
            bar_data = self.minute_volumes.get(current_minute)
            
            if bar_data is None:
                continue
            
            # Check if in entry window (9:45 AM - 2:30 PM ET)
            if not self._in_execution_window(et_time):
                # Still check for exits if in position
                if self.current_position is not None:
                    self._evaluate_tick_exit(tick, bar_data, et_time, verbose)
                continue
            
            # Check daily trade limit
            if self.daily_trades >= CONFIG.risk.max_daily_trades:
                if self.current_position is not None:
                    self._evaluate_tick_exit(tick, bar_data, et_time, verbose)
                continue
            
            # Process this tick
            if self.current_position is None:
                # Look for entry
                entry = self._evaluate_tick_entry(tick, bar_data, et_time, verbose)
                if entry:
                    trades_executed += 1
            else:
                # Look for adds and exits
                self._evaluate_tick_add(tick, bar_data, et_time, verbose)
                self._evaluate_tick_exit(tick, bar_data, et_time, verbose)
            
            last_minute = current_minute
        
        # Force close at end of session
        if self.current_position:
            last_tick = tick_df.to_dicts()[-1]
            self._force_tick_exit(last_tick, "time_exit", verbose)
        
        # Generate results
        result = self._generate_result(symbol, date, date)
        
        if verbose:
            if result.total_trades > 0:
                print(f"\n{result.generate_report()}")
            else:
                print(f"\nNo trades executed on this date.")
                print(f"Possible reasons:")
                print(f"  - Volume never exhausted below 60% of peak")
                print(f"  - Price never extended > 120% of VWAP")
                print(f"  - Price never stayed within 5% of day's high")
        
        return result
    
    def _get_et_time(self, timestamp: datetime) -> datetime:
        """Convert timestamp to ET."""
        if isinstance(timestamp, datetime) and timestamp.tzinfo is not None:
            et_tz = pytz.timezone('America/New_York')
            return timestamp.astimezone(et_tz)
        return timestamp
    
    def _in_execution_window(self, et_time: datetime) -> bool:
        """Check if in 9:45 AM - 2:30 PM ET entry window."""
        if isinstance(et_time, datetime):
            t = et_time.time()
        else:
            t = et_time
        return dt_time(9, 45) <= t <= dt_time(14, 30)
    
    def _get_volume_ratio(self, current_minute: datetime) -> float:
        """Get current volume vs peak."""
        bar_data = self.minute_volumes.get(current_minute)
        if bar_data is None or self.volume_peak == 0:
            return 1.0
        return bar_data['volume'] / self.volume_peak
    
    def _evaluate_tick_entry(
        self,
        tick: Dict,
        bar_data: Dict,
        et_time: datetime,
        verbose: bool
    ) -> bool:
        """Evaluate initial entry (Add #1) based on volume exhaustion."""
        price = tick['trade_price']
        vwap = bar_data.get('vwap', price)
        
        if vwap <= 0:
            return False
        
        vwap_extension = price / vwap
        volume_ratio = self._get_volume_ratio(tick['timestamp'].replace(second=0, microsecond=0))
        price_proximity = price / self.day_high if self.day_high > 0 else 0
        
        # ENTRY CRITERIA:
        # 1. VWAP extension > 120%
        # 2. Volume < 60% of peak
        # 3. Price within 5% of day's high
        
        criteria_met = {
            'vwap_extension': vwap_extension >= CONFIG.signals.vwap_extension_threshold,
            'volume_exhausted': volume_ratio < CONFIG.volume_exhaustion.entry_threshold,
            'near_high': price_proximity >= CONFIG.volume_exhaustion.price_proximity_to_high
        }
        
        if not all(criteria_met.values()):
            return False
        
        # Calculate fill with slippage
        fill_price = price * (1 + self.entry_slippage_bps / 10000)
        
        # Stop loss: 4% above entry
        stop_loss = fill_price * (1 + CONFIG.risk.initial_stop_percent / 100)
        
        # Position size: 25% of max
        position_value = CONFIG.scaling.max_position_value * (CONFIG.scaling.initial_size_percent / 100)
        position_size = int(position_value / fill_price)
        
        if position_size <= 0:
            return False
        
        risk_amount = position_size * (stop_loss - fill_price)
        
        # Create position
        self.current_position = {
            'entries': [{'price': fill_price, 'shares': position_size, 'add_level': 1}],
            'avg_entry': fill_price,
            'total_shares': position_size,
            'add_level': 1,
            'max_add_level': 1,
            'stop_loss': stop_loss,
            'profit_target_vwap': vwap,
            'highest_price': fill_price,
            'lowest_volume': volume_ratio,
            'vwap_entry': vwap,
            'tp1_hit': False,
            'tp2_hit': False,
            'tp3_hit': False,
            'entry_time': et_time
        }
        
        self.daily_trades += 1
        self.total_trades += 1
        
        # Audit record
        audit = AuditRecord(
            timestamp=tick['timestamp'],
            symbol=tick['symbol'],
            action=ActionType.ENTRY,
            price=fill_price,
            vwap=vwap,
            vwap_extension=vwap_extension,
            volume_ratio=volume_ratio,
            add_level=1,
            shares=position_size,
            total_shares=position_size,
            avg_entry=fill_price,
            stop_loss=stop_loss,
            reasoning=f"Entry: VWAP {vwap_extension:.2f}x, Vol {volume_ratio:.2f}, Prox {price_proximity:.2f}"
        )
        self.audit_records.append(audit)
        
        if verbose:
            print(f"\n[ENTRY] {et_time.strftime('%H:%M:%S')} {tick['symbol']} @ ${fill_price:.2f}")
            print(f"  Size: {position_size} shares (${position_size * fill_price:,.0f})")
            print(f"  VWAP: {vwap_extension:.2f}x | Vol Ratio: {volume_ratio:.2f} | Prox to HOD: {price_proximity:.2f}")
            print(f"  Stop: ${stop_loss:.2f} | VWAP Target: ${vwap:.2f}")
        
        return True
    
    def _evaluate_tick_add(
        self,
        tick: Dict,
        bar_data: Dict,
        et_time: datetime,
        verbose: bool
    ):
        """Evaluate adding to position (Add #2, Add #3)."""
        if not self.current_position:
            return
        
        pos = self.current_position
        if pos['add_level'] >= CONFIG.scaling.max_adds:
            return
        
        price = tick['trade_price']
        volume_ratio = self._get_volume_ratio(tick['timestamp'].replace(second=0, microsecond=0))
        
        # Must make new high on lower volume
        if price <= pos['highest_price']:
            return
        
        if volume_ratio >= pos['lowest_volume']:
            return
        
        # Check volume thresholds
        if pos['add_level'] == 1:
            if volume_ratio > CONFIG.volume_exhaustion.add2_threshold:
                return
            size_pct = CONFIG.scaling.add2_size_percent / 100
            add_num = 2
        else:
            if volume_ratio > CONFIG.volume_exhaustion.add3_threshold:
                return
            size_pct = CONFIG.scaling.add3_size_percent / 100
            add_num = 3
        
        # Calculate fill
        fill_price = price * (1 + self.entry_slippage_bps / 10000)
        position_value = CONFIG.scaling.max_position_value * size_pct
        add_shares = int(position_value / fill_price)
        
        if add_shares <= 0:
            return
        
        # Update position
        pos['entries'].append({'price': fill_price, 'shares': add_shares, 'add_level': add_num})
        
        # Recalculate average
        total_cost = sum(e['price'] * e['shares'] for e in pos['entries'])
        pos['total_shares'] = sum(e['shares'] for e in pos['entries'])
        pos['avg_entry'] = total_cost / pos['total_shares']
        pos['add_level'] = add_num
        pos['max_add_level'] = max(pos['max_add_level'], add_num)
        pos['highest_price'] = price
        pos['lowest_volume'] = volume_ratio
        
        # Update stop to 3.5%
        pos['stop_loss'] = pos['avg_entry'] * (1 + CONFIG.risk.average_stop_percent / 100)
        
        self.daily_trades += 1
        
        # Audit
        audit = AuditRecord(
            timestamp=tick['timestamp'],
            symbol=tick['symbol'],
            action=ActionType.ADD,
            price=fill_price,
            volume_ratio=volume_ratio,
            add_level=add_num,
            shares=add_shares,
            total_shares=pos['total_shares'],
            avg_entry=pos['avg_entry'],
            stop_loss=pos['stop_loss'],
            reasoning=f"Add #{add_num}: New high ${price:.2f} on vol {volume_ratio:.2f}"
        )
        self.audit_records.append(audit)
        
        if verbose:
            print(f"\n[ADD #{add_num}] {et_time.strftime('%H:%M:%S')} @ ${fill_price:.2f}")
            print(f"  Added: {add_shares} | Total: {pos['total_shares']} | Avg: ${pos['avg_entry']:.2f}")
    
    def _evaluate_tick_exit(self, tick: Dict, bar_data: Dict, et_time: datetime, verbose: bool):
        """Evaluate exit with layered profit taking."""
        if not self.current_position:
            return
        
        pos = self.current_position
        price = tick['trade_price']
        vwap = bar_data.get('vwap', pos['vwap_entry'])
        
        # Check stop loss first
        if price >= pos['stop_loss']:
            self._execute_exit(tick, price, et_time, "stop_exit", pos['total_shares'], verbose)
            return
        
        # Calculate depreciation from average entry
        depreciation = (pos['avg_entry'] - price) / pos['avg_entry'] * 100
        
        # Get remaining shares
        remaining = pos['total_shares'] - self._get_closed_shares()
        if remaining <= 0:
            return
        
        # TP1: VWAP (35% of position)
        if not pos['tp1_hit'] and price <= vwap:
            shares = int(pos['total_shares'] * CONFIG.exits.tp1_percent / 100)
            self._execute_exit(tick, price, et_time, "tp1_exit", shares, verbose)
            pos['tp1_hit'] = True
            return
        
        # TP2: -8% (35% of position)
        if not pos['tp2_hit'] and depreciation >= CONFIG.exits.tp2_percent_drop:
            shares = int(pos['total_shares'] * CONFIG.exits.tp2_percent / 100)
            self._execute_exit(tick, price, et_time, "tp2_exit", shares, verbose)
            pos['tp2_hit'] = True
            return
        
        # TP3: -15% or time exit (remaining)
        time_exit = et_time.time() >= dt_time(15, 25) if isinstance(et_time, datetime) else et_time >= dt_time(15, 25)
        
        if not pos['tp3_hit'] and (depreciation >= CONFIG.exits.tp3_percent_drop or time_exit):
            self._execute_exit(tick, price, et_time, "tp3_exit" if not time_exit else "time_exit", remaining, verbose)
            pos['tp3_hit'] = True
    
    def _get_closed_shares(self) -> int:
        """Get shares already closed."""
        closed = sum(
            r.shares for r in self.audit_records
            if r.action in [ActionType.TP1_EXIT, ActionType.TP2_EXIT, ActionType.TP3_EXIT, ActionType.STOP_EXIT]
        )
        return closed
    
    def _execute_exit(self, tick: Dict, price: float, et_time: datetime, reason: str, shares: int, verbose: bool):
        """Execute partial or full exit."""
        if not self.current_position or shares <= 0:
            return
        
        pos = self.current_position
        
        # Apply slippage
        fill_price = price * (1 - self.exit_slippage_bps / 10000)
        
        # Calculate P&L
        pnl = (pos['avg_entry'] - fill_price) * shares
        self.capital += pnl
        
        # Map action type
        action_map = {
            'stop_exit': ActionType.STOP_EXIT,
            'tp1_exit': ActionType.TP1_EXIT,
            'tp2_exit': ActionType.TP2_EXIT,
            'tp3_exit': ActionType.TP3_EXIT,
            'time_exit': ActionType.TIME_EXIT
        }
        
        # Audit
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
            print(f"\n[{reason.upper()}] {et_time.strftime('%H:%M:%S')} @ ${fill_price:.2f}")
            print(f"  Shares: {shares} | P&L: ${pnl:+.2f} ({pnl_pct:+.1f}%)")
        
        # Check if fully closed
        if self._get_closed_shares() >= pos['total_shares']:
            self.current_position = None
    
    def _force_tick_exit(self, tick: Dict, reason: str, verbose: bool):
        """Force exit at end of session."""
        if not self.current_position:
            return
        
        remaining = self.current_position['total_shares'] - self._get_closed_shares()
        if remaining > 0:
            et_time = self._get_et_time(tick['timestamp'])
            self._execute_exit(tick, tick['trade_price'], et_time, reason, remaining, verbose)
    
    def _generate_result(self, symbol: str, start: datetime, end: datetime) -> BacktestResult:
        """Generate backtest result."""
        entries = [r for r in self.audit_records if r.action == ActionType.ENTRY]
        adds = [r for r in self.audit_records if r.action == ActionType.ADD]
        exits = [r for r in self.audit_records if r.pnl is not None]
        
        total_pnl = sum(e.pnl for e in exits) if exits else 0.0
        winning_trades = sum(1 for e in exits if e.pnl > 0)
        losing_trades = sum(1 for e in exits if e.pnl <= 0)
        
        wins = [e.pnl for e in exits if e.pnl > 0]
        losses = [e.pnl for e in exits if e.pnl <= 0]
        
        tp1_exits = sum(1 for r in self.audit_records if r.action == ActionType.TP1_EXIT)
        tp2_exits = sum(1 for r in self.audit_records if r.action == ActionType.TP2_EXIT)
        tp3_exits = sum(1 for r in self.audit_records if r.action == ActionType.TP3_EXIT)
        stop_exits = sum(1 for r in self.audit_records if r.action == ActionType.STOP_EXIT)
        
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
            tp1_exits=tp1_exits,
            tp2_exits=tp2_exits,
            tp3_exits=tp3_exits,
            stop_exits=stop_exits,
            audit_records=self.audit_records
        )


# Singleton
tick_backtest_engine = TickBacktestEngine()
