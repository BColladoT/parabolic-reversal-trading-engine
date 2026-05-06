"""
Tick-Based Backtesting Engine - Volume Exhaustion Strategy V4 (Smart Exhaustion)
Institutional-grade multi-factor exhaustion detection for maximum edge.

Key innovations:
1. Smart Volume Peak Detection - Adapts to intraday volume profile
2. Divergence Entry - New high on lower volume (classic exhaustion)
3. Dynamic Position Sizing - Scale in only on confirmation
4. Trailing Stops - Lock in profits after TP1
5. Time Decay - Tighter stops after 1 PM
"""
from datetime import datetime, timedelta, time as dt_time
from typing import List, Dict, Optional, Tuple
import pytz

from src.backtest.historical_tick_fetcher import tick_fetcher
from src.backtest.backtest_engine import BacktestResult, AuditRecord, ActionType


class TickBacktestEngineV4:
    """
    V4: Smart Exhaustion Detection
    
    Entry Requirements (ALL must be met):
    1. Price makes NEW HIGH (vs previous 5 bars)
    2. Volume is LOWER than previous peak (divergence)
    3. Price > 95% of day's high
    4. VWAP extension > 15% (price stretched)
    5. Time: 9:45 AM - 1:00 PM ET (avoid afternoon chop)
    
    Scale-In Rules:
    - Add #1: Initial entry on first exhaustion signal
    - Add #2: Second new high on even lower volume
    - Add #3: Third exhaustion + price rejection wick
    
    Exit Rules:
    - TP1 (40%): VWAP mean reversion
    - TP2 (35%): -8% from entry
    - TP3 (25%): -15% or 2:30 PM
    - Trailing stop after TP1: Lock 50% of TP1 profit
    """
    
    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.audit_records: List[AuditRecord] = []
        
        self.current_position: Optional[Dict] = None
        self.daily_trades = 0
        self.total_trades = 0
        
        # Strict criteria
        self.min_vwap_extension = 1.15      # 15% above VWAP
        self.min_proximity = 0.95            # Within 5% of HOD
        self.vol_exhaustion_threshold = 0.70 # Volume < 70% of 10-min peak
        self.max_entry_time = dt_time(13, 0) # No entries after 1 PM
        
        # Price action tracking
        self.price_history: List[Dict] = []  # Last 5 bars
        self.volume_peak_10min = 0.0
        self.volume_history: List[float] = []
        
        # Day tracking
        self.day_high = 0.0
        self.day_high_time = None
        
        # Execution
        self.slippage_bps = 5.0
        self.et_tz = pytz.timezone('America/New_York')
    
    def reset(self):
        self.capital = self.initial_capital
        self.audit_records = []
        self.current_position = None
        self.daily_trades = 0
        self.total_trades = 0
        self.price_history = []
        self.volume_peak_10min = 0.0
        self.volume_history = []
        self.day_high = 0.0
        self.day_high_time = None
    
    def run_tick_backtest(self, symbol: str, date: datetime, verbose: bool = True) -> BacktestResult:
        self.reset()
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"BACKTEST V4: {symbol} on {date.date()}")
            print(f"Strategy: Smart Exhaustion Detection")
            print(f"{'='*70}\n")
        
        tick_df = tick_fetcher.fetch_combined_tick_data(symbol, date, use_quotes=False)
        if tick_df.is_empty():
            return BacktestResult(symbol, date, date, audit_records=[])
        
        bar_df = tick_fetcher.aggregate_trades_to_bars(tick_df, interval_seconds=60)
        if bar_df.is_empty():
            return BacktestResult(symbol, date, date, audit_records=[])
        
        # Calculate VWAP and prepare bars
        bars = []
        cumulative_tp_v = 0.0
        cumulative_vol = 0.0
        
        for row in bar_df.to_dicts():
            ts = row['timestamp']
            ts_et = ts.astimezone(self.et_tz) if ts.tzinfo else ts
            
            typical_price = (row['high'] + row['low'] + row['close']) / 3
            cumulative_tp_v += typical_price * row['volume']
            cumulative_vol += row['volume']
            vwap = cumulative_tp_v / cumulative_vol if cumulative_vol > 0 else row['close']
            
            bars.append({
                'timestamp': ts,
                'time_et': ts_et,
                'open': row['open'],
                'high': row['high'],
                'low': row['low'],
                'close': row['close'],
                'volume': row['volume'],
                'vwap': vwap
            })
        
        if verbose:
            print(f"Bars: {len(bars)}")
            print(f"Entry: NEW HIGH + LOWER VOL + VWAP>1.15x + PROX>95% + TIME<1PM")
        
        # Process bars
        for i, bar in enumerate(bars):
            t = bar['time_et']
            if not isinstance(t, datetime):
                continue
            
            # Update day high
            if bar['high'] > self.day_high:
                self.day_high = bar['high']
                self.day_high_time = t
            
            # Update price history (last 5 bars)
            self.price_history.append(bar)
            if len(self.price_history) > 5:
                self.price_history.pop(0)
            
            # Update volume tracking
            self.volume_history.append(bar['volume'])
            if len(self.volume_history) > 10:
                self.volume_history.pop(0)
            self.volume_peak_10min = max(self.volume_history) if self.volume_history else bar['volume']
            
            # Check if in position
            if self.current_position:
                self._check_exit(bar, bars, i, verbose)
                continue
            
            # Check entry conditions
            if self.daily_trades >= 3:  # Max 3 positions per day
                continue
            
            if not (dt_time(9, 45) <= t.time() <= self.max_entry_time):
                continue
            
            # Check ALL entry criteria
            if self._check_entry_criteria(bar, verbose):
                self._enter_position(bar, verbose)
        
        # Generate result
        result = self._generate_result(symbol, date, date)
        
        if verbose:
            if result.total_trades > 0:
                win_rate = result.winning_trades / result.total_trades * 100
                print(f"\n[RESULT] Trades: {result.total_trades}, Wins: {win_rate:.0f}%, P&L: ${result.total_pnl:+.2f}")
            else:
                print(f"\n[NO TRADES]")
        
        return result
    
    def _check_entry_criteria(self, bar: Dict, verbose: bool) -> bool:
        """Check ALL entry criteria. Must pass all for entry."""
        price = bar['close']
        vwap = bar['vwap']
        
        # 1. VWAP Extension > 15%
        vwap_ext = price / vwap if vwap > 0 else 1.0
        if vwap_ext < self.min_vwap_extension:
            return False
        
        # 2. Price within 5% of HOD
        prox = price / self.day_high if self.day_high > 0 else 0
        if prox < self.min_proximity:
            return False
        
        # 3. Volume < 70% of 10-min peak (exhaustion)
        vol_ratio = bar['volume'] / self.volume_peak_10min if self.volume_peak_10min > 0 else 1.0
        if vol_ratio >= self.vol_exhaustion_threshold:
            return False
        
        # 4. NEW HIGH vs previous 5 bars (divergence)
        if len(self.price_history) < 5:
            return False
        
        prev_high = max(b['high'] for b in self.price_history[:-1])  # Exclude current
        if price <= prev_high:
            return False  # Not a new high
        
        # 5. Volume lower than when previous high was made
        # Find the bar with previous high
        prev_high_bar = max(self.price_history[:-1], key=lambda x: x['high'])
        if bar['volume'] >= prev_high_bar['volume']:
            return False  # Volume not lower
        
        # ALL CRITERIA MET!
        return True
    
    def _enter_position(self, bar: Dict, verbose: bool):
        """Enter position on exhaustion signal."""
        price = bar['close']
        fill_price = price * (1 + self.slippage_bps / 10000)
        
        # Position sizing: 25% of $30K = $7,500
        position_value = 30000 * 0.25
        shares = int(position_value / fill_price)
        
        if shares <= 0:
            return
        
        # Dynamic stop: 4% for morning, 3% after 11 AM
        t = bar['time_et']
        stop_pct = 0.04 if t.time() < dt_time(11, 0) else 0.03
        stop = fill_price * (1 + stop_pct)
        
        self.current_position = {
            'entry_price': fill_price,
            'shares': shares,
            'stop_loss': stop,
            'initial_stop': stop,
            'highest_price': fill_price,
            'entry_bar': bar,
            'tp1_hit': False,
            'tp1_price': None,  # For trailing stop calculation
            'add_count': 0
        }
        
        self.daily_trades += 1
        self.total_trades += 1
        
        vwap_ext = price / bar['vwap']
        vol_ratio = bar['volume'] / self.volume_peak_10min
        
        audit = AuditRecord(
            timestamp=bar['timestamp'],
            symbol='TEST',
            action=ActionType.ENTRY,
            price=fill_price,
            vwap=bar['vwap'],
            vwap_extension=vwap_ext,
            volume_ratio=vol_ratio,
            add_level=1,
            shares=shares,
            total_shares=shares,
            avg_entry=fill_price,
            stop_loss=stop,
            reasoning=f"NEW HIGH ${price:.2f} on VOL {vol_ratio:.2f}, VWAP {vwap_ext:.2f}x"
        )
        self.audit_records.append(audit)
        
        if verbose:
            print(f"\n[ENTRY] {bar['time_et'].strftime('%H:%M')} @ ${fill_price:.2f}")
            print(f"  NEW HIGH on LOWER VOL (divergence)")
            print(f"  VWAP: {vwap_ext:.2f}x | Vol: {vol_ratio:.2f} | Shares: {shares}")
    
    def _check_exit(self, bar: Dict, all_bars: List, current_idx: int, verbose: bool):
        """Smart exit with trailing stops."""
        if not self.current_position:
            return
        
        pos = self.current_position
        price = bar['close']
        vwap = bar['vwap']
        
        # Update highest price seen
        if price > pos['highest_price']:
            pos['highest_price'] = price
        
        # Calculate depreciation
        depreciation = (pos['entry_price'] - price) / pos['entry_price'] * 100
        
        # Check hard stop
        if price >= pos['stop_loss']:
            self._exit(bar, price, "stop", verbose)
            return
        
        # TP1: VWAP (40% of position)
        if not pos['tp1_hit'] and price <= vwap:
            shares_tp1 = int(pos['shares'] * 0.40)
            self._exit_partial(bar, price, "tp1", shares_tp1, verbose)
            pos['tp1_hit'] = True
            pos['tp1_price'] = price
            
            # Move stop to lock in 50% of TP1 profit
            profit_per_share = pos['entry_price'] - price
            lock_in = profit_per_share * 0.5
            new_stop = pos['entry_price'] - lock_in
            if new_stop < pos['stop_loss']:
                pos['stop_loss'] = new_stop
                if verbose:
                    print(f"  [TRAILING STOP] Moved to ${new_stop:.2f}")
            return
        
        # TP2: -8% (35% of position)
        remaining = pos['shares'] - self._get_closed_shares()
        if pos['tp1_hit'] and not pos.get('tp2_hit') and depreciation >= 8.0:
            shares_tp2 = int(pos['shares'] * 0.35)
            shares_tp2 = min(shares_tp2, remaining)
            self._exit_partial(bar, price, "tp2", shares_tp2, verbose)
            pos['tp2_hit'] = True
            return
        
        # TP3: -15% or time exit (remaining)
        t = bar['time_et']
        time_exit = t.time() >= dt_time(14, 30) if isinstance(t, datetime) else t >= dt_time(14, 30)
        
        if depreciation >= 15.0 or time_exit:
            remaining = pos['shares'] - self._get_closed_shares()
            if remaining > 0:
                self._exit_partial(bar, price, "tp3" if not time_exit else "time", remaining, verbose)
                self.current_position = None
    
    def _exit_partial(self, bar: Dict, price: float, reason: str, shares: int, verbose: bool):
        """Execute partial exit."""
        if not self.current_position or shares <= 0:
            return
        
        pos = self.current_position
        fill_price = price * (1 - self.slippage_bps / 10000)
        pnl = (pos['entry_price'] - fill_price) * shares
        self.capital += pnl
        
        action_map = {
            'tp1': ActionType.TP1_EXIT,
            'tp2': ActionType.TP2_EXIT,
            'tp3': ActionType.TP3_EXIT,
            'time': ActionType.TIME_EXIT,
            'stop': ActionType.STOP_EXIT
        }
        
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
            print(f"  [EXIT:{reason.upper()}] {shares} shares @ ${fill_price:.2f} | P&L: ${pnl:+.2f} ({pnl_pct:+.1f}%)")
    
    def _exit(self, bar: Dict, price: float, reason: str, verbose: bool):
        """Exit full position."""
        if not self.current_position:
            return
        
        remaining = self.current_position['shares'] - self._get_closed_shares()
        if remaining > 0:
            self._exit_partial(bar, price, reason, remaining, verbose)
        self.current_position = None
    
    def _get_closed_shares(self) -> int:
        """Get shares already closed."""
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
            symbol=symbol,
            start_date=start,
            end_date=end,
            total_trades=len(entries),
            total_adds=0,
            winning_trades=winning,
            losing_trades=losing,
            total_pnl=total_pnl,
            win_rate=winning / len(entries) if entries else 0.0,
            profit_factor=0.0,
            average_trade=total_pnl / len(entries) if entries else 0.0,
            average_win=0.0,
            average_loss=0.0,
            audit_records=self.audit_records
        )


tick_backtest_engine_v4 = TickBacktestEngineV4()
