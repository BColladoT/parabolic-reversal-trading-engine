"""
Risk Management & Position Sizing Module - Progressive Scale-In Strategy
Volatility-based position sizing with progressive entry and layered exits.
"""
import json
import os
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import Enum
import numpy as np

from src.utils.config import CONFIG
from src.utils.logger import logger
from src.data.alpaca_client import AlpacaClient
from src.indicators.numba_kernels import calculate_position_size_numba


class PositionStatus(Enum):
    PENDING = "pending"
    BUILDING = "building"           # Actively scaling in
    FULL_SIZE = "full_size"         # All adds complete
    SCALING_OUT = "scaling_out"     # Taking profits
    CLOSING = "closing"
    CLOSED = "closed"


@dataclass
class Position:
    """Track position state with scale-in and scale-out support."""
    symbol: str
    side: str = "short"
    
    # Entry tracking (supports multiple entries)
    entries: List[Dict] = field(default_factory=list)  # {price, shares, time}
    entry_price_avg: float = 0.0
    total_shares: int = 0
    
    # Current state
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    
    # Scale-in state
    add_level: int = 0               # 0=initial, 1=after first, up to 3
    max_adds: int = 3
    
    # Exit tracking
    tp1_hit: bool = False
    tp2_hit: bool = False
    tp3_hit: bool = False
    shares_tp1_closed: int = 0
    shares_tp2_closed: int = 0
    shares_tp3_closed: int = 0
    
    # Risk levels
    stop_loss: float = 0.0           # Dynamic based on avg entry
    initial_stop: float = 0.0        # Stop for initial entry
    profit_target_vwap: float = 0.0
    
    # Tracking
    vwap_entry: float = 0.0
    parabolic_apex: float = 0.0      # Day's high at entry
    highest_price_since_entry: float = 0.0
    entry_time: datetime = field(default_factory=datetime.now)
    status: PositionStatus = PositionStatus.PENDING
    max_exposure: float = 0.0

    # Feature snapshot captured at signal-emission time. Persisted to the trade journal
    # on close so the edge estimator can compute conditional win-rates downstream.
    entry_features: Dict[str, float] = field(default_factory=dict)

    def add_entry(self, price: float, shares: int, add_level: int):
        """Add an entry to this position (scale-in)."""
        self.entries.append({
            'price': price,
            'shares': shares,
            'time': datetime.now(),
            'add_level': add_level
        })
        
        # Recalculate average entry
        total_cost = sum(e['price'] * e['shares'] for e in self.entries)
        self.total_shares = sum(e['shares'] for e in self.entries)
        self.entry_price_avg = total_cost / self.total_shares if self.total_shares > 0 else 0
        
        self.add_level = add_level
        
        # Update status
        if add_level >= self.max_adds:
            self.status = PositionStatus.FULL_SIZE
        else:
            self.status = PositionStatus.BUILDING
        
        # Update stop loss based on add level
        self._update_stop_loss()
    
    def _update_stop_loss(self):
        """Update stop loss based on position state."""
        if self.add_level == 1:
            # Initial entry: 4% stop
            self.stop_loss = self.entry_price_avg * (1 + CONFIG.risk.initial_stop_percent / 100)
        else:
            # Full or partial position: 3.5% stop
            self.stop_loss = self.entry_price_avg * (1 + CONFIG.risk.average_stop_percent / 100)
    
    def update_price(self, price: float):
        """Update current price and P&L."""
        self.current_price = price
        if self.total_shares > 0:
            # Short position: profit when price drops
            self.unrealized_pnl = (self.entry_price_avg - price) * self.total_shares
        
        # Track highest price for risk management
        if price > self.highest_price_since_entry:
            self.highest_price_since_entry = price
    
    def get_remaining_shares(self) -> int:
        """Get shares still open."""
        closed = self.shares_tp1_closed + self.shares_tp2_closed + self.shares_tp3_closed
        return self.total_shares - closed
    
    def close_partial(self, percent: float, exit_price: float) -> Tuple[int, float]:
        """
        Close a percentage of position.
        Returns (shares_closed, pnl).
        """
        remaining = self.get_remaining_shares()
        shares_to_close = int(remaining * percent / 100)
        
        if shares_to_close <= 0:
            return 0, 0.0
        
        pnl = (self.entry_price_avg - exit_price) * shares_to_close
        return shares_to_close, pnl


class RiskManager:
    """
    Central risk management controller for progressive scale-in strategy.
    """
    
    def __init__(self, alpaca_client: AlpacaClient):
        self.client = alpaca_client
        self.positions: Dict[str, Position] = {}
        self.daily_stats = {
            'trades_today': 0,
            'profits_today': 0.0,
            'losses_today': 0.0,
            'last_reset': datetime.now(),
            'daily_loss_limit_hit': False
        }
        self.account_equity = 0.0
        self.daily_pnl = 0.0

        # Restore persisted daily state if available (same ET day only)
        self._restore_daily_state()

    def _state_path(self) -> Path:
        return Path(os.environ.get("DAILY_STATE_PATH", "data/state/daily_state.json"))

    def _persist_daily_state(self) -> None:
        """Persist daily PnL + loss-limit-hit flag. Safe to call frequently."""
        try:
            p = self._state_path()
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps({
                "date": date.today().isoformat(),
                "daily_pnl": self.daily_pnl,
                "daily_loss_limit_hit": self.daily_stats['daily_loss_limit_hit'],
            }))
        except Exception as e:  # never let persistence break the engine
            logger.warning("Failed to persist daily state", error=str(e))

    def _restore_daily_state(self) -> None:
        """Restore daily PnL + loss-limit flag if state file is from today."""
        try:
            p = self._state_path()
            if not p.exists():
                return
            data = json.loads(p.read_text())
            if data.get("date") == date.today().isoformat():
                self.daily_pnl = float(data.get("daily_pnl", 0.0))
                self.daily_stats['daily_loss_limit_hit'] = bool(
                    data.get("daily_loss_limit_hit", False)
                )
                logger.info(
                    "Daily state restored",
                    daily_pnl=self.daily_pnl,
                    loss_limit_hit=self.daily_stats['daily_loss_limit_hit'],
                )
        except Exception as e:
            logger.warning("Failed to restore daily state", error=str(e))
        
    def update_account(self) -> float:
        """Update account equity from broker."""
        account = self.client.get_account()
        self.account_equity = account.get('equity', 0.0)
        return self.account_equity
    
    def check_daily_loss_limit(self) -> bool:
        """Check if daily loss limit has been reached."""
        if self.daily_stats['daily_loss_limit_hit']:
            return True
        
        daily_loss_limit = self.account_equity * (CONFIG.risk.daily_loss_limit_percent / 100)
        
        if self.daily_pnl <= -daily_loss_limit:
            self.daily_stats['daily_loss_limit_hit'] = True
            logger.critical(
                f"DAILY LOSS LIMIT HIT",
                daily_pnl=self.daily_pnl,
                limit=-daily_loss_limit
            )
            self._persist_daily_state()
            return True

        return False
    
    def check_margin_requirements(self, position_value: float, stock_price: float) -> bool:
        """Check if account meets margin requirements for short position."""
        if self.account_equity < CONFIG.risk.min_account_equity:
            logger.warning(
                f"Insufficient equity for margin trading",
                equity=self.account_equity,
                required=CONFIG.risk.min_account_equity
            )
            return False
        
        # For no-leverage trading, we need sufficient buying power
        buying_power = self.client.get_account().get('buying_power', 0)
        
        # Alpaca uses 102.5% - 104% collar for short orders
        collar_factor = 1.04
        total_required = position_value * collar_factor
        
        if total_required > buying_power:
            logger.warning(
                f"Insufficient buying power",
                required=total_required,
                available=buying_power
            )
            return False
        
        return True
    
    def calculate_position_size(self, symbol: str, entry_price: float,
                                atr: float, vwap: float, day_high: float,
                                add_level: int = 1) -> Dict:
        """
        Calculate position size for initial entry or scale-in.
        
        Scale-In Sizing:
        - Add 1 (Initial): 25% of max position
        - Add 2: 25% of max position  
        - Add 3: 50% of max position
        """
        # Update account
        self.update_account()
        
        # Check daily loss limit
        if self.check_daily_loss_limit():
            return {'shares': 0, 'valid': False, 'reason': 'daily_loss_limit'}
        
        # Check daily trade limit
        if self.daily_stats['trades_today'] >= CONFIG.risk.max_daily_trades:
            logger.warning("Daily trade limit reached")
            return {'shares': 0, 'valid': False, 'reason': 'daily_limit'}
        
        # Check max positions
        open_positions = len([p for p in self.positions.values() 
                             if p.status not in [PositionStatus.CLOSED, PositionStatus.CLOSING]])
        
        # For new positions (not adds), check position limit
        if add_level == 1 and symbol not in self.positions:
            if open_positions >= CONFIG.risk.max_positions:
                logger.warning("Max positions reached")
                return {'shares': 0, 'valid': False, 'reason': 'max_positions'}
        
        # Determine stop loss
        if add_level == 1:
            stop_distance = entry_price * (CONFIG.risk.initial_stop_percent / 100)
        else:
            stop_distance = entry_price * (CONFIG.risk.average_stop_percent / 100)
        
        stop_loss = entry_price + stop_distance
        
        # Hard stop at day high (parabolic apex)
        hard_stop = day_high * 1.01  # 1% buffer above high
        stop_loss = max(stop_loss, hard_stop)
        
        risk_per_share = stop_loss - entry_price
        if risk_per_share <= 0:
            return {'shares': 0, 'valid': False, 'reason': 'invalid_stop'}
        
        # Calculate max position value based on risk
        # Risk = 1% of account for FULL position
        max_risk = self.account_equity * (CONFIG.risk.max_portfolio_risk_percent / 100)
        
        # For partial positions, scale the risk proportionally
        if add_level == 1:
            risk_allocation = max_risk * 0.5  # First add uses partial risk budget
        elif add_level == 2:
            risk_allocation = max_risk * 0.3
        else:
            risk_allocation = max_risk * 0.2
        
        # Calculate shares
        max_shares_by_risk = int(risk_allocation / risk_per_share)
        
        # Apply position size limits from config
        size_percent = CONFIG.scaling.initial_size_percent
        if add_level == 2:
            size_percent = CONFIG.scaling.add2_size_percent
        elif add_level >= 3:
            size_percent = CONFIG.scaling.add3_size_percent
        
        max_position_value = CONFIG.scaling.max_position_value * (size_percent / 100)
        max_shares_by_value = int(max_position_value / entry_price)
        
        # Take the more conservative of risk-based or value-based
        shares = min(max_shares_by_risk, max_shares_by_value)
        
        # Hard share limit
        shares = min(shares, CONFIG.scaling.max_shares_per_position)
        
        if shares <= 0:
            return {'shares': 0, 'valid': False, 'reason': 'zero_shares'}
        
        position_value = shares * entry_price
        
        # Check margin requirements
        if not self.check_margin_requirements(position_value, entry_price):
            return {'shares': 0, 'valid': False, 'reason': 'margin'}
        
        return {
            'shares': shares,
            'valid': True,
            'stop_loss': stop_loss,
            'profit_target': vwap,  # TP1 target
            'risk_per_share': risk_per_share,
            'total_risk': risk_per_share * shares,
            'position_value': position_value,
            'add_level': add_level
        }
    
    def open_position(self, symbol: str, entry_price: float, qty: int,
                      stop_loss: float, vwap: float, day_high: float,
                      add_level: int = 1,
                      entry_features: Optional[Dict[str, float]] = None) -> Position:
        """Open or add to a position."""

        if symbol not in self.positions:
            # New position
            position = Position(
                symbol=symbol,
                stop_loss=stop_loss,
                profit_target_vwap=vwap,
                vwap_entry=vwap,
                parabolic_apex=day_high,
                status=PositionStatus.BUILDING,
                max_exposure=qty * entry_price,
                entry_features=dict(entry_features or {}),
            )
            self.positions[symbol] = position
        else:
            position = self.positions[symbol]
        
        # Add the entry
        position.add_entry(entry_price, qty, add_level)
        
        self.daily_stats['trades_today'] += 1
        
        logger.info(
            f"Position entry/add",
            symbol=symbol,
            add_level=add_level,
            qty=qty,
            entry_price=entry_price,
            avg_price=position.entry_price_avg,
            total_shares=position.total_shares,
            stop=position.stop_loss
        )
        
        return position
    
    def update_positions(self, price_data: Dict[str, float]):
        """Update all positions with latest prices."""
        for symbol, position in self.positions.items():
            if position.status not in [PositionStatus.CLOSED, PositionStatus.CLOSING]:
                if symbol in price_data:
                    position.update_price(price_data[symbol])
    
    def check_exit_signals(self, symbol: str) -> List[Tuple[str, float, int]]:
        """
        Check for exit signals for a position.
        Returns list of (exit_type, exit_price, shares_to_close) tuples.
        """
        exits = []
        
        if symbol not in self.positions:
            return exits
        
        position = self.positions[symbol]
        
        if position.status in [PositionStatus.CLOSED, PositionStatus.CLOSING]:
            return exits
        
        current_price = position.current_price
        remaining_shares = position.get_remaining_shares()
        
        if remaining_shares <= 0:
            return exits
        
        # Check stop loss first (highest priority)
        if current_price >= position.stop_loss:
            exits.append(('stop_loss', position.stop_loss, remaining_shares))
            return exits
        
        # TP1: VWAP target (35% of position)
        if not position.tp1_hit and current_price <= position.profit_target_vwap:
            shares = int(position.total_shares * CONFIG.exits.tp1_percent / 100)
            exits.append(('tp1_vwap', current_price, shares))
        
        # TP2: -8% from entry (35% of position)
        depreciation = (position.entry_price_avg - current_price) / position.entry_price_avg * 100
        if not position.tp2_hit and depreciation >= CONFIG.exits.tp2_percent_drop:
            shares = int(position.total_shares * CONFIG.exits.tp2_percent / 100)
            exits.append(('tp2_momentum', current_price, shares))
        
        # TP3: -15% from entry (remaining 30%)
        if not position.tp3_hit and depreciation >= CONFIG.exits.tp3_percent_drop:
            remaining = position.get_remaining_shares()
            exits.append(('tp3_final', current_price, remaining))
        
        return exits
    
    def execute_partial_exit(self, symbol: str, exit_price: float, 
                             shares: int, reason: str) -> float:
        """Execute partial position exit."""
        if symbol not in self.positions:
            return 0.0
        
        position = self.positions[symbol]
        
        # Calculate P&L for this exit
        pnl = (position.entry_price_avg - exit_price) * shares
        
        # Update position state
        if reason == 'tp1_vwap':
            position.tp1_hit = True
            position.shares_tp1_closed += shares
        elif reason == 'tp2_momentum':
            position.tp2_hit = True
            position.shares_tp2_closed += shares
        elif reason == 'tp3_final':
            position.tp3_hit = True
            position.shares_tp3_closed += shares
        
        position.realized_pnl += pnl
        self.daily_pnl += pnl
        
        # Update status
        if position.get_remaining_shares() <= 0:
            position.status = PositionStatus.CLOSED
        else:
            position.status = PositionStatus.SCALING_OUT
        
        # Update daily stats
        if pnl > 0:
            self.daily_stats['profits_today'] += pnl
        else:
            self.daily_stats['losses_today'] += abs(pnl)
        
        logger.info(
            f"Partial exit executed",
            symbol=symbol,
            shares=shares,
            exit_price=exit_price,
            pnl=f"${pnl:.2f}",
            reason=reason,
            remaining=position.get_remaining_shares()
        )

        self._persist_daily_state()
        return pnl

    def close_position(self, symbol: str, exit_price: float, reason: str) -> float:
        """Close entire remaining position."""
        if symbol not in self.positions:
            return 0.0
        
        position = self.positions[symbol]
        remaining_shares = position.get_remaining_shares()
        
        if remaining_shares <= 0:
            return 0.0
        
        pnl = (position.entry_price_avg - exit_price) * remaining_shares
        position.realized_pnl += pnl
        position.status = PositionStatus.CLOSED
        
        self.daily_pnl += pnl
        
        if pnl > 0:
            self.daily_stats['profits_today'] += pnl
        else:
            self.daily_stats['losses_today'] += abs(pnl)
        
        logger.info(
            f"Position fully closed",
            symbol=symbol,
            exit_price=exit_price,
            total_pnl=f"${position.realized_pnl:.2f}",
            reason=reason
        )

        self._persist_daily_state()
        return pnl
    
    def check_time_based_exits(self, current_time: datetime, market_close: datetime) -> List[str]:
        """Check for time-based position closures."""
        symbols_to_close = []
        
        time_to_close = market_close - current_time
        flatten_threshold = timedelta(minutes=CONFIG.compliance.flat_before_close_minutes)
        
        if time_to_close <= flatten_threshold:
            for symbol, position in self.positions.items():
                if position.status not in [PositionStatus.CLOSED, PositionStatus.CLOSING]:
                    symbols_to_close.append(symbol)
                    logger.info(
                        f"Time-based exit triggered",
                        symbol=symbol,
                        minutes_to_close=time_to_close.seconds // 60
                    )
        
        return symbols_to_close
    
    def get_position_summary(self) -> Dict:
        """Get summary of all positions."""
        open_positions = [
            p for p in self.positions.values()
            if p.status not in [PositionStatus.CLOSED, PositionStatus.CLOSING]
        ]
        
        total_exposure = sum(p.max_exposure for p in open_positions)
        total_unrealized = sum(p.unrealized_pnl for p in open_positions)
        total_realized = sum(p.realized_pnl for p in self.positions.values())
        
        return {
            'open_count': len(open_positions),
            'total_exposure': total_exposure,
            'unrealized_pnl': total_unrealized,
            'realized_pnl_today': total_realized,
            'daily_pnl': self.daily_pnl,
            'trades_today': self.daily_stats['trades_today'],
            'loss_limit_hit': self.daily_stats['daily_loss_limit_hit'],
            'positions': [
                {
                    'symbol': p.symbol,
                    'status': p.status.value,
                    'add_level': p.add_level,
                    'total_shares': p.total_shares,
                    'remaining_shares': p.get_remaining_shares(),
                    'avg_entry': p.entry_price_avg,
                    'current': p.current_price,
                    'unrealized': p.unrealized_pnl,
                    'realized': p.realized_pnl,
                    'stop': p.stop_loss,
                    'tp1_hit': p.tp1_hit,
                    'tp2_hit': p.tp2_hit,
                    'tp3_hit': p.tp3_hit
                }
                for p in open_positions
            ]
        }
    
    def reset_daily_stats(self):
        """Reset daily statistics (call at market open)."""
        self.daily_stats = {
            'trades_today': 0,
            'profits_today': 0.0,
            'losses_today': 0.0,
            'last_reset': datetime.now(),
            'daily_loss_limit_hit': False
        }
        self.daily_pnl = 0.0
        
        # Clear closed positions
        self.positions = {
            k: v for k, v in self.positions.items()
            if v.status != PositionStatus.CLOSED
        }

        # Persist fresh state so a restart later today starts clean too
        self._persist_daily_state()

        logger.info("Daily stats reset")
    
    def emergency_flatten_all(self) -> bool:
        """Emergency position closure."""
        logger.critical("EMERGENCY FLATTEN INITIATED")
        
        try:
            result = self.client.close_all_positions()
            
            for position in self.positions.values():
                position.status = PositionStatus.CLOSED
            
            return result.get('success', False)
        except Exception as e:
            logger.critical(f"Emergency flatten failed: {e}")
            return False
