"""
Tick-Based Backtesting Engine - Volume Exhaustion Strategy V3
Enter on 2 of 3 criteria to catch the exhaustion transition.
"""
from datetime import datetime, timedelta, time as dt_time
from typing import List, Dict, Optional
import pytz

from src.backtest.historical_tick_fetcher import tick_fetcher
from src.backtest.backtest_engine import BacktestResult, AuditRecord, ActionType


class TickBacktestEngineV3:
    """
    V3: Enter when 2 of 3 criteria are met.
    This catches the transition from parabolic to exhausted.
    """
    
    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.audit_records: List[AuditRecord] = []
        
        self.current_position: Optional[Dict] = None
        self.daily_trades = 0
        self.total_trades = 0
        
        # RELAXED CRITERIA for V3
        self.min_vwap_extension = 1.12  # 12% above VWAP
        self.vol_exhaustion_threshold = 0.75  # 75% of recent peak  
        self.proximity_threshold = 0.90  # Within 10% of HOD
        self.min_criteria = 2  # Need 2 of 3
        
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
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"BACKTEST V3: {symbol} on {date.date()}")
            print(f"Strategy: 2-of-3 Criteria Entry")
            print(f"{'='*70}\n")
        
        tick_df = tick_fetcher.fetch_combined_tick_data(symbol, date, use_quotes=False)
        if tick_df.is_empty():
            return BacktestResult(symbol, date, date, audit_records=[])
        
        bar_df = tick_fetcher.aggregate_trades_to_bars(tick_df, interval_seconds=60)
        if bar_df.is_empty():
            return BacktestResult(symbol, date, date, audit_records=[])
        
        # Pre-calculate all bars with metrics
        bars = []
        cumulative_tp_v = 0.0
        cumulative_vol = 0.0
        day_high = 0.0
        
        for row in bar_df.to_dicts():
            ts = row['timestamp']
            ts_et = ts.astimezone(self.et_tz) if ts.tzinfo else ts
            
            typical_price = (row['high'] + row['low'] + row['close']) / 3
            cumulative_tp_v += typical_price * row['volume']
            cumulative_vol += row['volume']
            vwap = cumulative_tp_v / cumulative_vol if cumulative_vol > 0 else row['close']
            
            if row['high'] > day_high:
                day_high = row['high']
            
            bars.append({
                'timestamp': ts,
                'time_et': ts_et,
                'close': row['close'],
                'volume': row['volume'],
                'vwap': vwap,
                'day_high': day_high  # Snapshot at this bar
            })
        
        # Update day_high in all bars
        for b in bars:
            b['day_high'] = day_high
        
        if verbose:
            print(f"Bars: {len(bars)}, Day High: ${day_high:.2f}")
            print(f"Entry: 2 of (VWAP>{self.min_vwap_extension:.2f}x, Vol<{self.vol_exhaustion_threshold:.0%}, Prox>{self.proximity_threshold:.0%})")
        
        # Process each bar
        volume_window = []
        entries_found = 0
        
        for i, bar in enumerate(bars):
            t = bar['time_et']
            if not isinstance(t, datetime):
                continue
            
            # Update volume window
            volume_window.append(bar['volume'])
            if len(volume_window) > 10:
                volume_window.pop(0)
            
            recent_peak = max(volume_window) if volume_window else bar['volume']
            
            # Calculate criteria
            vwap_ext = bar['close'] / bar['vwap'] if bar['vwap'] > 0 else 1.0
            vol_ratio = bar['volume'] / recent_peak if recent_peak > 0 else 1.0
            prox = bar['close'] / bar['day_high'] if bar['day_high'] > 0 else 0
            
            # Check entry window
            if not (dt_time(9, 45) <= t.time() <= dt_time(14, 30)):
                continue
            
            # Check if in position
            if self.current_position:
                self._check_exit(bar, verbose)
                continue
            
            # Check daily limit
            if self.daily_trades >= 9:
                continue
            
            # Count criteria met
            criteria_count = sum([
                vwap_ext >= self.min_vwap_extension,
                vol_ratio <= self.vol_exhaustion_threshold,
                prox >= self.proximity_threshold
            ])
            
            if criteria_count >= self.min_criteria and self.current_position is None:
                # ENTRY!
                entries_found += 1
                self._enter_position(bar, vwap_ext, vol_ratio, prox, criteria_count, verbose)
        
        # Generate result
        result = self._generate_result(symbol, date, date)
        
        if verbose:
            if result.total_trades > 0:
                print(f"\n[ENTRY FOUND] Trades: {result.total_trades}")
                for audit in result.audit_records[:3]:
                    print(f"  {audit.timestamp}: {audit.action.value} @ ${audit.price:.2f}")
            else:
                print(f"\n[NO ENTRY] No bar met 2+ criteria")
        
        return result
    
    def _enter_position(self, bar: Dict, vwap_ext: float, vol_ratio: float, prox: float, count: int, verbose: bool):
        """Enter position."""
        price = bar['close']
        fill_price = price * 1.0005  # Slippage
        
        position_value = 30000 * 0.25
        shares = int(position_value / fill_price)
        
        if shares <= 0:
            return
        
        stop = fill_price * 1.04
        
        self.current_position = {
            'entry_price': fill_price,
            'shares': shares,
            'stop_loss': stop,
            'tp1_hit': False
        }
        
        self.daily_trades += 1
        self.total_trades += 1
        
        audit = AuditRecord(
            timestamp=bar['timestamp'],
            symbol='TEST',  # Would be actual symbol
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
            reasoning=f"{count}/3 criteria: VWAP{vwap_ext:.2f} Vol{vol_ratio:.2f} Prox{prox:.2f}"
        )
        self.audit_records.append(audit)
        
        if verbose:
            print(f"\n[ENTRY] {bar['time_et'].strftime('%H:%M')} @ ${fill_price:.2f}")
            print(f"  Criteria met: {count}/3")
            print(f"  VWAP: {vwap_ext:.2f}x | Vol: {vol_ratio:.2f} | Prox: {prox:.2f}")
    
    def _check_exit(self, bar: Dict, verbose: bool):
        """Exit logic with VWAP target or stop."""
        if not self.current_position:
            return
        
        pos = self.current_position
        price = bar['close']
        
        # Stop loss
        if price >= pos['stop_loss']:
            self._exit(bar, price, "stop", verbose)
            self.current_position = None
            return
        
        # VWAP target
        if not pos['tp1_hit'] and price <= bar['vwap']:
            self._exit(bar, price, "vwap", verbose)
            pos['tp1_hit'] = True
            self.current_position = None
    
    def _exit(self, bar: Dict, price: float, reason: str, verbose: bool):
        """Execute exit with P&L."""
        if not self.current_position:
            return
        
        pos = self.current_position
        fill_price = price * 0.9995
        pnl = (pos['entry_price'] - fill_price) * pos['shares']
        self.capital += pnl
        
        action_type = ActionType.TP1_EXIT if reason == "vwap" else ActionType.STOP_EXIT
        
        audit = AuditRecord(
            timestamp=bar['timestamp'],
            symbol='TEST',
            action=action_type,
            price=fill_price,
            shares=pos['shares'],
            avg_entry=pos['entry_price'],
            exit_price=fill_price,
            exit_time=bar['timestamp'],
            pnl=pnl,
            exit_reason=reason
        )
        self.audit_records.append(audit)
        
        if verbose:
            pnl_pct = (pnl / (pos['entry_price'] * pos['shares'])) * 100
            print(f"  [EXIT:{reason.upper()}] @ ${fill_price:.2f} | P&L: ${pnl:+.2f} ({pnl_pct:+.1f}%)")
    
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


tick_backtest_engine_v3 = TickBacktestEngineV3()
