"""
Backtesting Engine - Progressive Exhaustion Scale-In Strategy
Simulates trades on historical data with full audit trails.
"""
import json
import os
from datetime import datetime, timedelta, time as dt_time
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
from collections import defaultdict

import polars as pl
import numpy as np
import pandas as pd
import pytz

from src.utils.config import CONFIG
from src.utils.logger import logger
from src.indicators.numba_kernels import (
    calculate_vwap_numba,
    calculate_atr_numba,
    detect_momentum_divergence_numba,
    detect_absorption_numba,
    calculate_position_size_numba
)
from src.backtest.data_fetcher import DataFetcher


class ActionType(Enum):
    ENTRY = "entry"
    ADD = "add"                     # Scale-in add
    TP1_EXIT = "tp1_exit"           # 35% at VWAP
    TP2_EXIT = "tp2_exit"           # 35% at -8%
    TP3_EXIT = "tp3_exit"           # 30% at -15%
    STOP_EXIT = "stop_exit"
    TIME_EXIT = "time_exit"
    SKIP = "skip"


@dataclass
class AuditRecord:
    """Detailed audit record for every decision."""
    timestamp: datetime
    symbol: str
    action: ActionType
    price: float
    
    # Market conditions
    vwap: float = 0.0
    vwap_extension: float = 0.0
    atr: float = 0.0
    volume: int = 0
    volume_ratio: float = 0.0       # Current vs peak
    day_high: float = 0.0
    
    # Signal factors
    volume_exhaustion: bool = False
    near_high: bool = False
    momentum_divergence: bool = False
    
    # Decision reasoning
    reasoning: str = ""
    confidence_score: float = 0.0
    confirming_factors: int = 0
    add_level: int = 0              # 1=initial, 2=add2, 3=add3
    
    # Position details
    shares: int = 0                 # Shares for this action
    total_shares: int = 0           # Total position after this action
    avg_entry: float = 0.0
    stop_loss: float = 0.0
    
    # Outcome (filled after exit)
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    pnl: Optional[float] = None
    exit_reason: Optional[str] = None
    cumulative_pnl: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'action': self.action.value,
            'price': self.price,
            'vwap': self.vwap,
            'vwap_extension': self.vwap_extension,
            'atr': self.atr,
            'volume': self.volume,
            'volume_ratio': self.volume_ratio,
            'day_high': self.day_high,
            'volume_exhaustion': self.volume_exhaustion,
            'near_high': self.near_high,
            'momentum_divergence': self.momentum_divergence,
            'reasoning': self.reasoning,
            'confidence_score': self.confidence_score,
            'confirming_factors': self.confirming_factors,
            'add_level': self.add_level,
            'shares': self.shares,
            'total_shares': self.total_shares,
            'avg_entry': self.avg_entry,
            'stop_loss': self.stop_loss,
            'exit_price': self.exit_price,
            'exit_time': self.exit_time.isoformat() if self.exit_time else None,
            'pnl': self.pnl,
            'exit_reason': self.exit_reason,
            'cumulative_pnl': self.cumulative_pnl
        }


@dataclass
class BacktestResult:
    """Complete backtest results."""
    symbol: str
    start_date: datetime
    end_date: datetime
    total_trades: int = 0
    total_adds: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    average_trade: float = 0.0
    average_win: float = 0.0
    average_loss: float = 0.0
    
    # Exit breakdown
    tp1_exits: int = 0
    tp2_exits: int = 0
    tp3_exits: int = 0
    stop_exits: int = 0
    time_exits: int = 0
    
    # Audit trail
    audit_records: List[AuditRecord] = field(default_factory=list)
    
    def generate_report(self) -> str:
        """Generate human-readable report."""
        lines = [
            "=" * 70,
            f"BACKTEST REPORT: {self.symbol}",
            f"Period: {self.start_date.date()} to {self.end_date.date()}",
            f"Strategy: Progressive Exhaustion Scale-In",
            "=" * 70,
            "",
            "PERFORMANCE METRICS:",
            f"  Total Trades:        {self.total_trades}",
            f"  Total Adds:          {self.total_adds}",
            f"  Winning Trades:      {self.winning_trades}",
            f"  Losing Trades:       {self.losing_trades}",
            f"  Win Rate:            {self.win_rate:.1%}",
            f"  Total P&L:           ${self.total_pnl:,.2f}",
            f"  Average Trade:       ${self.average_trade:,.2f}",
            f"  Average Win:         ${self.average_win:,.2f}",
            f"  Average Loss:        ${self.average_loss:,.2f}",
            f"  Profit Factor:       {self.profit_factor:.2f}",
            f"  Max Drawdown:        ${self.max_drawdown:,.2f}",
            "",
            "EXIT BREAKDOWN:",
            f"  TP1 (VWAP):          {self.tp1_exits}",
            f"  TP2 (-8%):           {self.tp2_exits}",
            f"  TP3 (-15%/Time):     {self.tp3_exits}",
            f"  Stop Loss:           {self.stop_exits}",
            "",
            "AUDIT SUMMARY:",
        ]
        
        for record in self.audit_records:
            if record.action == ActionType.ENTRY:
                lines.append(f"\n  [{record.timestamp.strftime('%H:%M')}] ENTRY {record.symbol} @ ${record.price:.2f}")
                lines.append(f"    Shares: {record.shares} | VWAP Ext: {record.vwap_extension:.2f}x")
                lines.append(f"    Confidence: {record.confidence_score:.1%} | VolRatio: {record.volume_ratio:.2f}")
            elif record.action == ActionType.ADD:
                lines.append(f"  [{record.timestamp.strftime('%H:%M')}] ADD #{record.add_level} @ ${record.price:.2f}")
                lines.append(f"    Shares: {record.shares} | Total: {record.total_shares} | Avg: ${record.avg_entry:.2f}")
            elif record.exit_reason:
                lines.append(f"  [{record.timestamp.strftime('%H:%M')}] {record.exit_reason.upper()} @ ${record.exit_price:.2f}")
                lines.append(f"    P&L: ${record.pnl:+.2f} | Cumulative: ${record.cumulative_pnl:+.2f}")
        
        lines.extend([
            "",
            "=" * 70
        ])
        
        return "\n".join(lines)


class BacktestEngine:
    """
    Backtesting engine for Progressive Exhaustion Scale-In strategy.
    """
    
    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.data_fetcher = DataFetcher()
        self.audit_records: List[AuditRecord] = []
        
        # Position tracking
        self.current_position: Optional[Dict] = None
        self.daily_trades = 0
        self.total_trades = 0
        self.daily_pnl = 0.0
        
        # Volume tracking
        self.volume_peak = 0.0
        self.volume_history: List[float] = []
        
    def reset(self):
        """Reset engine state."""
        self.capital = self.initial_capital
        self.audit_records = []
        self.current_position = None
        self.daily_trades = 0
        self.total_trades = 0
        self.daily_pnl = 0.0
        self.volume_peak = 0.0
        self.volume_history = []
    
    def run_backtest(
        self,
        symbol: str,
        date: datetime,
        verbose: bool = True
    ) -> BacktestResult:
        """
        Run backtest for a single symbol on a specific date.
        """
        self.reset()
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"BACKTEST: {symbol} on {date.date()}")
            print(f"Strategy: Progressive Exhaustion Scale-In")
            print(f"{'='*70}\n")
        
        # Get intraday data
        df = self.data_fetcher.get_intraday_for_date(symbol, date)
        
        if df.is_empty():
            logger.warning(f"No data for {symbol} on {date.date()}")
            return BacktestResult(symbol, date, date, audit_records=[])
        
        if verbose:
            print(f"Loaded {len(df)} minute bars")
        
        # Calculate indicators
        df = self._calculate_indicators(df)
        
        # Simulate trading session
        for i, row in enumerate(df.to_dicts()):
            timestamp = row['timestamp']
            
            # Update volume tracking
            self._update_volume_metrics(row['volume'])
            
            # Check if in entry window (9:45 AM - 2:30 PM)
            if not self._in_entry_window(timestamp) and self.current_position is None:
                continue
            
            # Check daily trade limit
            if self.daily_trades >= CONFIG.risk.max_daily_trades:
                if verbose and i == len(df) - 1:
                    print(f"Daily trade limit reached ({self.daily_trades})")
                continue
            
            # If no position, look for entry
            if self.current_position is None:
                self._evaluate_entry(row, verbose)
            
            # If in position, check for adds and exits
            else:
                self._evaluate_add(row, verbose)
                self._evaluate_exits(row, verbose)
        
        # Close any open position at end of day
        if self.current_position:
            last_row = df.to_dicts()[-1]
            self._force_exit(last_row, "time_exit", verbose)
        
        # Generate result
        result = self._generate_result(symbol, date, date)
        
        if verbose:
            print(result.generate_report())
        
        return result
    
    def _calculate_indicators(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate VWAP, ATR, and other indicators."""
        highs = df['high'].to_numpy()
        lows = df['low'].to_numpy()
        closes = df['close'].to_numpy()
        volumes = df['volume'].to_numpy()
        
        # Calculate VWAP (session anchored)
        vwap_values = calculate_vwap_numba(highs, lows, closes, volumes)
        
        # Calculate ATR
        atr_values = calculate_atr_numba(highs, lows, closes, period=14)
        
        # Add to DataFrame
        df = df.with_columns([
            pl.Series('vwap', vwap_values),
            pl.Series('atr', atr_values),
        ])
        
        # Calculate VWAP extension
        df = df.with_columns([
            (pl.col('close') / pl.col('vwap')).alias('vwap_extension'),
        ])
        
        # Calculate rolling volume (5-min)
        df = df.with_columns([
            pl.col('volume').rolling_mean(window_size=5).alias('volume_5min'),
        ])
        
        # Track day high
        df = df.with_columns([
            pl.col('high').cum_max().alias('day_high'),
        ])
        
        return df
    
    def _update_volume_metrics(self, volume: int):
        """Update volume peak tracking."""
        self.volume_history.append(volume)
        
        # 5-min rolling volume
        if len(self.volume_history) >= 5:
            vol_5min = sum(self.volume_history[-5:])
            if vol_5min > self.volume_peak:
                self.volume_peak = vol_5min
    
    def _get_volume_ratio(self) -> float:
        """Get current volume vs peak ratio."""
        if self.volume_peak == 0 or len(self.volume_history) < 5:
            return 1.0
        
        current_5min = sum(self.volume_history[-5:])
        return current_5min / self.volume_peak
    
    def _in_entry_window(self, timestamp: datetime) -> bool:
        """Check if timestamp is in entry window (9:45 AM - 2:30 PM ET)."""
        if isinstance(timestamp, datetime) and timestamp.tzinfo is not None:
            et_tz = pytz.timezone('America/New_York')
            timestamp = timestamp.astimezone(et_tz)
        
        t = timestamp.time()
        exec_start = dt_time(9, 45)
        exec_end = dt_time(14, 30)  # 2:30 PM
        return exec_start <= t <= exec_end
    
    def _evaluate_entry(self, row: Dict, verbose: bool):
        """Evaluate entry conditions."""
        price = row['close']
        vwap = row['vwap']
        vwap_extension = row['vwap_extension']
        atr = row['atr']
        day_high = row['day_high']
        volume_ratio = self._get_volume_ratio()
        
        # Check VWAP extension (> 120%)
        if vwap_extension < CONFIG.signals.vwap_extension_threshold:
            return
        
        # Check volume exhaustion (< 60% of peak)
        volume_exhausted = volume_ratio < CONFIG.volume_exhaustion.entry_threshold
        if not volume_exhausted:
            return
        
        # Check price proximity to high (within 5%)
        price_proximity = price / day_high if day_high > 0 else 0
        near_high = price_proximity >= CONFIG.volume_exhaustion.price_proximity_to_high
        
        # Calculate confidence
        confidence = 0.0
        if 1.20 <= vwap_extension <= 1.50:
            confidence += 0.40
        elif vwap_extension > 1.50:
            confidence += 0.30
        
        if volume_ratio < 0.40:
            confidence += 0.35
        elif volume_ratio < 0.50:
            confidence += 0.30
        elif volume_ratio < 0.60:
            confidence += 0.25
        
        if near_high:
            confidence += 0.25
        
        confidence = min(1.0, confidence)
        
        # Calculate position size (25% of max for initial)
        stop_loss = price * (1 + CONFIG.risk.initial_stop_percent / 100)
        risk_per_share = stop_loss - price
        
        if risk_per_share <= 0:
            return
        
        max_risk = self.capital * (CONFIG.risk.max_portfolio_risk_percent / 100) * 0.5
        max_shares_risk = int(max_risk / risk_per_share)
        
        position_value_limit = CONFIG.scaling.max_position_value * 0.25
        max_shares_value = int(position_value_limit / price)
        
        shares = min(max_shares_risk, max_shares_value, CONFIG.scaling.max_shares_per_position)
        
        if shares <= 0:
            return
        
        # Create position
        self.current_position = {
            'entries': [{'price': price, 'shares': shares, 'add_level': 1}],
            'avg_entry': price,
            'total_shares': shares,
            'add_level': 1,
            'stop_loss': stop_loss,
            'highest_price': price,
            'lowest_volume': volume_ratio,
            'vwap_entry': vwap,
            'tp1_hit': False,
            'tp2_hit': False,
            'tp3_hit': False,
            'entry_time': row['timestamp']
        }
        
        self.daily_trades += 1
        self.total_trades += 1
        
        # Create audit record
        audit = AuditRecord(
            timestamp=row['timestamp'],
            symbol=row['symbol'],
            action=ActionType.ENTRY,
            price=price,
            vwap=vwap,
            vwap_extension=vwap_extension,
            atr=atr,
            volume=row['volume'],
            volume_ratio=volume_ratio,
            day_high=day_high,
            volume_exhaustion=True,
            near_high=near_high,
            confidence_score=confidence,
            add_level=1,
            shares=shares,
            total_shares=shares,
            avg_entry=price,
            stop_loss=stop_loss
        )
        self.audit_records.append(audit)
        
        if verbose:
            print(f"\n[ENTRY] {row['timestamp'].strftime('%H:%M')} {row['symbol']} @ ${price:.2f}")
            print(f"  Shares: {shares} | VWAP Ext: {vwap_extension:.2f}x | VolRatio: {volume_ratio:.2f}")
            print(f"  Stop: ${stop_loss:.2f} | Confidence: {confidence:.1%}")
    
    def _evaluate_add(self, row: Dict, verbose: bool):
        """Evaluate adding to position (scale-in)."""
        if not self.current_position:
            return
        
        pos = self.current_position
        if pos['add_level'] >= 3:
            return
        
        price = row['close']
        volume_ratio = self._get_volume_ratio()
        
        # Must make new high on lower volume
        if price <= pos['highest_price']:
            return
        
        if volume_ratio >= pos['lowest_volume']:
            return
        
        # Check volume threshold for add level
        if pos['add_level'] == 1 and volume_ratio > CONFIG.volume_exhaustion.add2_threshold:
            return
        
        if pos['add_level'] == 2 and volume_ratio > CONFIG.volume_exhaustion.add3_threshold:
            return
        
        # Calculate add size
        if pos['add_level'] == 1:
            size_pct = CONFIG.scaling.add2_size_percent / 100  # 25%
        else:
            size_pct = CONFIG.scaling.add3_size_percent / 100  # 50%
        
        target_value = CONFIG.scaling.max_position_value * size_pct
        shares = int(target_value / price)
        
        # Update position
        pos['entries'].append({'price': price, 'shares': shares, 'add_level': pos['add_level'] + 1})
        
        # Recalculate average
        total_cost = sum(e['price'] * e['shares'] for e in pos['entries'])
        pos['total_shares'] = sum(e['shares'] for e in pos['entries'])
        pos['avg_entry'] = total_cost / pos['total_shares']
        pos['add_level'] += 1
        pos['highest_price'] = price
        pos['lowest_volume'] = volume_ratio
        
        # Update stop to 3.5%
        pos['stop_loss'] = pos['avg_entry'] * (1 + CONFIG.risk.average_stop_percent / 100)
        
        self.daily_trades += 1
        
        # Audit record
        audit = AuditRecord(
            timestamp=row['timestamp'],
            symbol=row['symbol'],
            action=ActionType.ADD,
            price=price,
            volume_ratio=volume_ratio,
            add_level=pos['add_level'],
            shares=shares,
            total_shares=pos['total_shares'],
            avg_entry=pos['avg_entry'],
            stop_loss=pos['stop_loss']
        )
        self.audit_records.append(audit)
        
        if verbose:
            print(f"\n[ADD #{pos['add_level']}] {row['timestamp'].strftime('%H:%M')} @ ${price:.2f}")
            print(f"  Shares added: {shares} | Total: {pos['total_shares']} | Avg: ${pos['avg_entry']:.2f}")
    
    def _evaluate_exits(self, row: Dict, verbose: bool):
        """Evaluate exit conditions (TP1, TP2, TP3, Stop)."""
        if not self.current_position:
            return
        
        pos = self.current_position
        price = row['close']
        vwap = row['vwap']
        
        # Check stop loss first
        if price >= pos['stop_loss']:
            self._execute_exit(row, price, "stop_exit", pos['total_shares'], verbose)
            return
        
        # Calculate depreciation from average entry
        depreciation = (pos['avg_entry'] - price) / pos['avg_entry'] * 100
        
        # TP1: VWAP (35% of position)
        if not pos['tp1_hit'] and price <= vwap:
            shares = int(pos['total_shares'] * CONFIG.exits.tp1_percent / 100)
            self._execute_exit(row, price, "tp1_exit", shares, verbose)
            pos['tp1_hit'] = True
            return  # One exit per bar
        
        # TP2: -8% (35% of position)
        if not pos['tp2_hit'] and depreciation >= CONFIG.exits.tp2_percent_drop:
            shares = int(pos['total_shares'] * CONFIG.exits.tp2_percent / 100)
            self._execute_exit(row, price, "tp2_exit", shares, verbose)
            pos['tp2_hit'] = True
            return
        
        # TP3: -15% (remaining)
        remaining = pos['total_shares'] - self._get_closed_shares()
        if not pos['tp3_hit'] and depreciation >= CONFIG.exits.tp3_percent_drop and remaining > 0:
            self._execute_exit(row, price, "tp3_exit", remaining, verbose)
            pos['tp3_hit'] = True
            return
        
        # Time exit (3:25 PM)
        if row['timestamp'].time() >= dt_time(15, 25):
            remaining = pos['total_shares'] - self._get_closed_shares()
            if remaining > 0:
                self._execute_exit(row, price, "time_exit", remaining, verbose)
    
    def _get_closed_shares(self) -> int:
        """Get total shares already closed."""
        closed = sum(
            r.shares for r in self.audit_records
            if r.action in [ActionType.TP1_EXIT, ActionType.TP2_EXIT, ActionType.TP3_EXIT, ActionType.STOP_EXIT]
        )
        return closed
    
    def _execute_exit(self, row: Dict, exit_price: float, reason: str, 
                      shares: int, verbose: bool):
        """Execute partial or full exit."""
        if not self.current_position:
            return
        
        pos = self.current_position
        
        # Calculate P&L
        pnl = (pos['avg_entry'] - exit_price) * shares
        self.capital += pnl
        self.daily_pnl += pnl
        
        # Map reason to action type
        action_map = {
            'stop_exit': ActionType.STOP_EXIT,
            'tp1_exit': ActionType.TP1_EXIT,
            'tp2_exit': ActionType.TP2_EXIT,
            'tp3_exit': ActionType.TP3_EXIT,
            'time_exit': ActionType.TIME_EXIT
        }
        
        action = action_map.get(reason, ActionType.TIME_EXIT)
        
        # Audit record
        audit = AuditRecord(
            timestamp=row['timestamp'],
            symbol=row['symbol'],
            action=action,
            price=exit_price,
            shares=shares,
            avg_entry=pos['avg_entry'],
            exit_price=exit_price,
            exit_time=row['timestamp'],
            pnl=pnl,
            exit_reason=reason,
            cumulative_pnl=self.daily_pnl
        )
        self.audit_records.append(audit)
        
        if verbose:
            pnl_pct = (pnl / (pos['avg_entry'] * shares)) * 100
            print(f"\n[{reason.upper()}] {row['timestamp'].strftime('%H:%M')} @ ${exit_price:.2f}")
            print(f"  Shares: {shares} | P&L: ${pnl:+.2f} ({pnl_pct:+.1f}%)")
        
        # Check if fully closed
        total_closed = self._get_closed_shares()
        if total_closed >= pos['total_shares']:
            self.current_position = None
    
    def _force_exit(self, row: Dict, reason: str, verbose: bool):
        """Force exit at end of day."""
        if not self.current_position:
            return
        
        remaining = self.current_position['total_shares'] - self._get_closed_shares()
        if remaining > 0:
            self._execute_exit(row, row['close'], reason, remaining, verbose)
    
    def _generate_result(self, symbol: str, start: datetime, end: datetime) -> BacktestResult:
        """Generate backtest result with statistics."""
        entries = [r for r in self.audit_records if r.action == ActionType.ENTRY]
        adds = [r for r in self.audit_records if r.action == ActionType.ADD]
        exits = [r for r in self.audit_records if r.pnl is not None]
        
        total_pnl = sum(e.pnl for e in exits) if exits else 0.0
        winning_trades = sum(1 for e in exits if e.pnl > 0)
        losing_trades = sum(1 for e in exits if e.pnl <= 0)
        
        wins = [e.pnl for e in exits if e.pnl > 0]
        losses = [e.pnl for e in exits if e.pnl <= 0]
        
        # Count exit types
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
backtest_engine = BacktestEngine()
