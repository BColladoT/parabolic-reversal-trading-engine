"""
Tick-Based Backtesting Engine - Volume Exhaustion Strategy V6 (High Frequency)
V6: Relaxed criteria for more trades while maintaining win rate.

Key changes from V5:
- 1-of-3 criteria (was 2-of-3) - More entry opportunities
- Earlier entry window: 9:35 AM (was 9:45) - Catch early exhaustion
- VWAP pullback allowed - Entry below VWAP if other criteria strong
- Multiple trades per day (up to 3) - Capture multiple setups
- Relaxed proximity: 85% (was 93%) - Earlier entries
- Relaxed volume: 80% (was 70%) - More volume tolerance
"""
from datetime import datetime, timedelta, time as dt_time
from typing import List, Dict, Optional
import pytz

from src.backtest.historical_tick_fetcher import tick_fetcher
from src.backtest.backtest_engine import BacktestResult, AuditRecord, ActionType


class TickBacktestEngineV6:
    """
    V6: High-frequency parabolic trading
    
    Entry (1 of 3):
    - VWAP extension > 12% (relaxed from 15%)
    - Volume < 80% of peak (relaxed from 70%)
    - Price within 15% of HOD (relaxed from 7%)
    
    Quality filters:
    - Time: 9:35 AM - 2:30 PM (extended)
    - Stock up > 40% from open (relaxed from 50%)
    - VWAP pullback allowed if criteria strong
    
    Risk: Same 3% stop, VWAP target
    """
    
    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.audit_records: List[AuditRecord] = []
        
        self.current_position: Optional[Dict] = None
        self.daily_trades = 0
        self.total_trades = 0
        
        # RELAXED Criteria for more entries
        self.min_vwap_extension = 1.12  # 12% (was 15%)
        self.vol_exhaustion = 0.80      # 80% (was 70%)
        self.min_proximity = 0.85       # 15% from high (was 7%)
        self.min_day_gain = 0.40        # 40% gain (was 50%)
        
        self.slippage_bps = 5.0
        self.et_tz = pytz.timezone('America/New_York')
        self.max_trades_per_day = 3     # Multiple trades (was 1)
    
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
            print(f"\nV6: {symbol} {date.date()} | Open: ${day_open:.2f}, High: ${day_high:.2f}")
            gain = (day_high - day_open) / day_open * 100 if day_open > 0 else 0
            print(f"Day Gain: {gain:.1f}% | Entry: 1 of (VWAP>1.12x, Vol<80%, Prox>85%)")
        
        # Process bars - allow multiple entries
        volume_history = []
        cooldown_bars = 0  # Bars to wait between trades
        
        for i, bar in enumerate(bars):
            t = bar['time_et']
            if not isinstance(t, datetime):
                continue
            
            # Handle cooldown between trades
            if cooldown_bars > 0:
                cooldown_bars -= 1
                continue
            
            # Update volume
            volume_history.append(bar['volume'])
            if len(volume_history) > 10:
                volume_history.pop(0)
            vol_peak = max(volume_history) if volume_history else bar['volume']
            
            # Update day high
            bar['day_high'] = day_high
            
            # Check if in position
            if self.current_position:
                exit_result = self._check_exit(bar, verbose)
                if exit_result:
                    cooldown_bars = 5  # 5-bar cooldown after exit
                continue
            
            # Skip if max trades reached
            if self.daily_trades >= self.max_trades_per_day:
                continue
            
            # Entry window - EARLIER start (9:35 AM)
            if not (dt_time(9, 35) <= t.time() <= dt_time(14, 30)):
                continue
            
            # Relaxed quality filters
            day_gain = (bar['day_high'] - bar['day_open']) / bar['day_open'] if bar['day_open'] > 0 else 0
            if day_gain < self.min_day_gain:  # 40% minimum
                continue
            
            # RELAXED: Allow entry below VWAP if other criteria strong
            above_vwap = bar['close'] >= bar['vwap']
            
            # Calculate 1-of-3 criteria (RELAXED)
            vwap_ext = bar['close'] / bar['vwap'] if bar['vwap'] > 0 else 1.0
            vol_ratio = bar['volume'] / vol_peak if vol_peak > 0 else 1.0
            prox = bar['close'] / bar['day_high'] if bar['day_high'] > 0 else 0
            
            criteria_met = sum([
                vwap_ext >= self.min_vwap_extension,
                vol_ratio <= self.vol_exhaustion,
                prox >= self.min_proximity
            ])
            
            # 1-of-3 with bonus for being above VWAP, OR 2-of-3 regardless
            should_enter = (criteria_met >= 1 and above_vwap) or criteria_met >= 2
            
            if should_enter:
                setup = {
                    'bar': bar,
                    'vwap_ext': vwap_ext,
                    'vol_ratio': vol_ratio,
                    'prox': prox,
                    'criteria': criteria_met
                }
                
                if verbose:
                    print(f"  [SETUP] {t.strftime('%H:%M')} - Criteria: {criteria_met}/3, Above VWAP: {above_vwap}")
                
                self._enter_position(bar, setup, verbose)
                # Check immediate exit (same bar)
                if self.current_position:
                    self._check_exit(bar, verbose)
        
        # Force exit at end
        if self.current_position:
            last_bar = bars[-1]
            self._exit(last_bar, last_bar['close'], "time", verbose)
        
        result = self._generate_result(symbol, date, date)
        
        if verbose and result.total_trades > 0:
            win_rate = result.winning_trades / result.total_trades * 100 if result.total_trades > 0 else 0
            print(f"[RESULT] Trades: {result.total_trades}, P&L: ${result.total_pnl:+.2f} | Win: {win_rate:.0f}%")
        
        return result
    
    def _enter_position(self, bar: Dict, setup: Dict, verbose: bool):
        """Enter with smart sizing."""
        price = bar['close']
        fill_price = price * 1.0005
        
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
            'entry_time': bar['time_et']
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
            reasoning=f"V6: {setup['criteria']}/3 criteria, gain {(bar['day_high']-bar['day_open'])/bar['day_open']*100:.0f}%"
        )
        self.audit_records.append(audit)
        
        if verbose:
            print(f"[ENTRY] {bar['time_et'].strftime('%H:%M')} @ ${fill_price:.2f} (Trade #{self.daily_trades})")
    
    def _check_exit(self, bar: Dict, verbose: bool) -> bool:
        """Check exits. Returns True if position closed."""
        if not self.current_position:
            return False
        
        pos = self.current_position
        price = bar['close']
        
        # Stop loss
        if bar.get('high', price) >= pos['stop_loss']:
            self._exit(bar, pos['stop_loss'], "stop", verbose)
            return True
        
        # VWAP target (full position)
        if price <= bar['vwap']:
            self._exit(bar, price, "vwap", verbose)
            return True
        
        # Time exit at 3:30 PM
        t = bar['time_et']
        if isinstance(t, datetime) and t.time() >= dt_time(15, 30):
            self._exit(bar, price, "time", verbose)
            return True
        
        return False
    
    def _exit(self, bar: Dict, price: float, reason: str, verbose: bool):
        """Exit full position."""
        if not self.current_position:
            return
        
        pos = self.current_position
        fill_price = price * 0.9995
        pnl = (pos['entry_price'] - fill_price) * pos['shares']
        self.capital += pnl
        
        action_map = {
            'vwap': ActionType.TP1_EXIT,
            'time': ActionType.TIME_EXIT,
            'stop': ActionType.STOP_EXIT
        }
        
        audit = AuditRecord(
            timestamp=bar['timestamp'],
            symbol='TEST',
            action=action_map.get(reason, ActionType.TIME_EXIT),
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
            print(f"  [EXIT] ({reason}) @ ${fill_price:.2f} | ${pnl:+.2f} ({pnl_pct:+.1f}%)")
        
        self.current_position = None
    
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


tick_backtest_engine_v6 = TickBacktestEngineV6()
