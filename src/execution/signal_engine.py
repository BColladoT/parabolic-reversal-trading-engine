"""
Signal Generation Engine - Progressive Exhaustion Scale-In Strategy
Detects volume exhaustion during intraday parabolic moves for short entry.
"""
from typing import Optional, Dict, List, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import numpy as np

from src.utils.config import CONFIG
from src.utils.logger import logger
from src.data.polars_engine import StreamingBuffer, PolarsSignalEngine
from src.indicators.numba_kernels import (
    detect_momentum_divergence_numba,
    detect_absorption_numba
)
from src.screening.screener import ScreenedAsset


class SignalType(Enum):
    ENTRY_SHORT = "entry_short"
    ADD_POSITION = "add_position"  # Scale-in signal
    EXIT_COVER = "exit_cover"
    STOP_LOSS = "stop_loss"
    TP1_VWAP = "tp1_vwap"          # Take profit 1 at VWAP
    TP2_MOMENTUM = "tp2_momentum"  # Take profit 2 at -8%
    TP3_FINAL = "tp3_final"        # Take profit 3 at -15%
    TIME_EXIT = "time_exit"


@dataclass
class VolumeProfile:
    """Track volume metrics throughout the day."""
    peak_volume: float = 0.0           # Highest 5-min volume seen
    peak_volume_time: Optional[datetime] = None
    current_volume_5min: float = 0.0
    volume_ratio: float = 1.0          # Current / Peak
    volume_trend: str = "neutral"      # increasing, decreasing, neutral
    
    def update(self, volume_5min: float, timestamp: datetime):
        """Update volume profile with new 5-min volume."""
        self.current_volume_5min = volume_5min
        
        # Track peak
        if volume_5min > self.peak_volume:
            self.peak_volume = volume_5min
            self.peak_volume_time = timestamp
        
        # Calculate ratio
        if self.peak_volume > 0:
            self.volume_ratio = volume_5min / self.peak_volume


@dataclass
class TradeSignal:
    """Generated trading signal."""
    symbol: str
    signal_type: SignalType
    timestamp: datetime
    price: float
    confidence: float
    vwap: float
    atr: float
    volume_ratio: float              # Current volume vs peak
    volume_exhaustion: bool
    is_add_signal: bool = False      # True if scaling into existing position
    add_level: int = 0               # 1=initial, 2=add2, 3=add3
    notes: str = ""


@dataclass
class PositionState:
    """Track position building state for scale-in strategy."""
    symbol: str
    entry_price_avg: float = 0.0
    total_shares: int = 0
    add_level: int = 0               # 0=none, 1=initial, 2=add2, 3=add3
    last_add_time: Optional[datetime] = None
    entry_vwap: float = 0.0
    day_high_at_entry: float = 0.0
    highest_price_seen: float = 0.0
    lowest_volume_seen: float = float('inf')
    
    def can_add(self, current_time: datetime, current_price: float, 
                current_volume_ratio: float) -> bool:
        """Check if we can add to this position."""
        if self.add_level >= 3:
            return False
        
        # Cooldown period between adds
        if self.last_add_time:
            cooldown = timedelta(minutes=CONFIG.signals.min_minutes_between_adds)
            if current_time - self.last_add_time < cooldown:
                return False
        
        # Must make new high on lower volume
        if current_price <= self.highest_price_seen:
            return False
        
        if current_volume_ratio >= self.lowest_volume_seen:
            return False
        
        return True


class ParabolicSignalEngine:
    """
    Volume Exhaustion Signal Engine with Progressive Scale-In.
    Monitors intraday parabolic moves and fades volume exhaustion.
    """
    
    def __init__(self, data_engine: PolarsSignalEngine):
        self.data_engine = data_engine
        self.tick_history: Dict[str, List[Dict]] = {}
        self.signal_callbacks: List[Callable[[TradeSignal], None]] = []
        
        # Volume tracking per symbol
        self.volume_profiles: Dict[str, VolumeProfile] = {}
        
        # Position building state
        self.position_states: Dict[str, PositionState] = {}
        
        # Price tracking for new high detection
        self.day_highs: Dict[str, float] = {}
        self.day_lows: Dict[str, float] = {}
        
    def register_callback(self, callback: Callable[[TradeSignal], None]):
        """Register signal callback."""
        self.signal_callbacks.append(callback)
    
    def _emit_signal(self, signal: TradeSignal):
        """Emit signal to all callbacks."""
        for callback in self.signal_callbacks:
            try:
                callback(signal)
            except Exception as e:
                logger.error(f"Signal callback error: {e}")
    
    def update_volume_profile(self, symbol: str, volume_5min: float, 
                               timestamp: datetime) -> VolumeProfile:
        """Update and return volume profile for symbol."""
        if symbol not in self.volume_profiles:
            self.volume_profiles[symbol] = VolumeProfile()
        
        profile = self.volume_profiles[symbol]
        profile.update(volume_5min, timestamp)
        
        return profile
    
    def update_price_extremes(self, symbol: str, price: float):
        """Track day's high and low for each symbol."""
        if symbol not in self.day_highs or price > self.day_highs[symbol]:
            self.day_highs[symbol] = price
        
        if symbol not in self.day_lows or price < self.day_lows[symbol]:
            self.day_lows[symbol] = price
        
        # Update position state if exists
        if symbol in self.position_states:
            state = self.position_states[symbol]
            if price > state.highest_price_seen:
                state.highest_price_seen = price
    
    def generate_entry_signal(self, asset: ScreenedAsset, 
                               volume_profile: VolumeProfile) -> Optional[TradeSignal]:
        """
        Generate initial short entry signal.
        
        Entry Criteria:
        1. Volume has dropped 40%+ from peak (ratio < 0.60)
        2. Price within 5% of day's high
        3. VWAP extension > 20%
        4. At least 2 confirming factors
        """
        symbol = asset.symbol
        metrics = self.data_engine.get_signal_data(symbol)
        
        if not metrics:
            return None
        
        current_price = metrics.get('last_price', asset.current_price)
        vwap = metrics.get('vwap', current_price)
        atr = metrics.get('atr', current_price * 0.02)
        
        # Check VWAP extension (> 120% = 20% extension)
        vwap_extension = current_price / vwap if vwap > 0 else 1.0
        if vwap_extension < CONFIG.signals.vwap_extension_threshold:
            return None
        
        # Check volume exhaustion
        volume_ratio = volume_profile.volume_ratio
        volume_exhausted = volume_ratio < CONFIG.volume_exhaustion.entry_threshold
        
        # Check price proximity to high (must be within 5%)
        day_high = self.day_highs.get(symbol, asset.day_high)
        price_proximity = current_price / day_high if day_high > 0 else 0
        near_high = price_proximity >= CONFIG.volume_exhaustion.price_proximity_to_high
        
        # Check momentum divergence
        momentum_divergence = self._check_momentum_divergence(symbol)
        
        # Check absorption
        absorption = self._detect_absorption(symbol, current_price)
        
        # Count confirming factors
        confirming_factors = sum([
            volume_exhausted,
            near_high,
            momentum_divergence,
            absorption,
            vwap_extension > 1.30  # Extreme extension > 30%
        ])
        
        if confirming_factors < CONFIG.signals.min_exhaustion_factors:
            return None
        
        # Calculate confidence
        confidence = self._calculate_confidence(
            vwap_extension=vwap_extension,
            volume_ratio=volume_ratio,
            volume_exhausted=volume_exhausted,
            near_high=near_high,
            momentum_divergence=momentum_divergence,
            absorption=absorption
        )
        
        # Create position state for this symbol
        self.position_states[symbol] = PositionState(
            symbol=symbol,
            day_high_at_entry=day_high,
            highest_price_seen=current_price,
            lowest_volume_seen=volume_ratio,
            entry_vwap=vwap
        )
        
        signal = TradeSignal(
            symbol=symbol,
            signal_type=SignalType.ENTRY_SHORT,
            timestamp=datetime.now(),
            price=current_price,
            confidence=confidence,
            vwap=vwap,
            atr=atr,
            volume_ratio=volume_ratio,
            volume_exhaustion=volume_exhausted,
            is_add_signal=False,
            add_level=1,
            notes=f"Initial entry. VWAP:{vwap_extension:.2f}x, VolRatio:{volume_ratio:.2f}, Factors:{confirming_factors}"
        )
        
        # Update position state
        self.position_states[symbol].add_level = 1
        self.position_states[symbol].last_add_time = datetime.now()
        
        logger.info(
            f"Entry signal generated",
            symbol=symbol,
            price=current_price,
            confidence=f"{confidence:.2f}",
            vwap_extension=f"{vwap_extension:.2f}",
            volume_ratio=f"{volume_ratio:.2f}"
        )
        
        self._emit_signal(signal)
        return signal
    
    def generate_add_signal(self, symbol: str, current_price: float,
                            volume_profile: VolumeProfile) -> Optional[TradeSignal]:
        """
        Generate scale-in (add) signal for existing position.
        
        Add Criteria:
        1. Must make NEW HIGH on LOWER VOLUME
        2. Volume ratio must be below previous add level
        3. Cooldown period must have passed
        """
        if symbol not in self.position_states:
            return None
        
        state = self.position_states[symbol]
        current_time = datetime.now()
        volume_ratio = volume_profile.volume_ratio
        
        # Check if we can add
        if not state.can_add(current_time, current_price, volume_ratio):
            return None
        
        # Check volume thresholds for each add level
        if state.add_level == 1 and volume_ratio > CONFIG.volume_exhaustion.add2_threshold:
            return None
        
        if state.add_level == 2 and volume_ratio > CONFIG.volume_exhaustion.add3_threshold:
            return None
        
        metrics = self.data_engine.get_signal_data(symbol)
        vwap = metrics.get('vwap', current_price)
        atr = metrics.get('atr', current_price * 0.02)
        
        # Determine add level
        new_add_level = state.add_level + 1
        
        signal = TradeSignal(
            symbol=symbol,
            signal_type=SignalType.ADD_POSITION,
            timestamp=current_time,
            price=current_price,
            confidence=0.85,  # High confidence for adds
            vwap=vwap,
            atr=atr,
            volume_ratio=volume_ratio,
            volume_exhaustion=True,
            is_add_signal=True,
            add_level=new_add_level,
            notes=f"Add #{new_add_level}. New high:${current_price:.2f} on vol:{volume_ratio:.2f}"
        )
        
        # Update position state
        state.add_level = new_add_level
        state.last_add_time = current_time
        state.lowest_volume_seen = volume_ratio
        
        logger.info(
            f"Add signal generated",
            symbol=symbol,
            add_level=new_add_level,
            price=current_price,
            volume_ratio=f"{volume_ratio:.2f}"
        )
        
        self._emit_signal(signal)
        return signal
    
    def generate_exit_signals(self, symbol: str, position: 'Position') -> List[TradeSignal]:
        """
        Generate exit signals for open position.
        Multiple exit levels: Stop, TP1 (VWAP), TP2 (-8%), TP3 (-15%)
        """
        signals = []
        
        metrics = self.data_engine.get_signal_data(symbol)
        if not metrics:
            return signals
        
        current_price = metrics.get('last_price', position.current_price)
        vwap = metrics.get('vwap', position.vwap_entry)
        
        # Calculate depreciation from average entry
        if position.entry_price_avg > 0:
            depreciation = (position.entry_price_avg - current_price) / position.entry_price_avg * 100
        else:
            depreciation = 0
        
        # Check stop loss
        if current_price >= position.stop_loss:
            signals.append(TradeSignal(
                symbol=symbol,
                signal_type=SignalType.STOP_LOSS,
                timestamp=datetime.now(),
                price=current_price,
                confidence=1.0,
                vwap=vwap,
                atr=0.0,
                volume_ratio=0.0,
                volume_exhaustion=False,
                notes=f"Stop loss hit: {position.stop_loss:.2f}"
            ))
            return signals  # Stop is priority
        
        # TP1: VWAP target (35% of position)
        if not position.tp1_hit and current_price <= vwap:
            signals.append(TradeSignal(
                symbol=symbol,
                signal_type=SignalType.TP1_VWAP,
                timestamp=datetime.now(),
                price=current_price,
                confidence=0.9,
                vwap=vwap,
                atr=0.0,
                volume_ratio=0.0,
                volume_exhaustion=False,
                notes=f"TP1: VWAP reached ${vwap:.2f}"
            ))
        
        # TP2: -8% from entry (35% of position)
        if not position.tp2_hit and depreciation >= CONFIG.exits.tp2_percent_drop:
            signals.append(TradeSignal(
                symbol=symbol,
                signal_type=SignalType.TP2_MOMENTUM,
                timestamp=datetime.now(),
                price=current_price,
                confidence=0.85,
                vwap=vwap,
                atr=0.0,
                volume_ratio=0.0,
                volume_exhaustion=False,
                notes=f"TP2: -{depreciation:.1f}% from entry"
            ))
        
        # TP3: -15% from entry or time exit (30% of position)
        if not position.tp3_hit and depreciation >= CONFIG.exits.tp3_percent_drop:
            signals.append(TradeSignal(
                symbol=symbol,
                signal_type=SignalType.TP3_FINAL,
                timestamp=datetime.now(),
                price=current_price,
                confidence=0.8,
                vwap=vwap,
                atr=0.0,
                volume_ratio=0.0,
                volume_exhaustion=False,
                notes=f"TP3: -{depreciation:.1f}% from entry"
            ))
        
        return signals
    
    def _check_momentum_divergence(self, symbol: str) -> bool:
        """Check for price-volume momentum divergence."""
        buffer = self.data_engine.buffers.get(symbol)
        if not buffer or len(buffer.bar_history) < 6:
            return False
        
        bar_df = buffer.get_bar_df(n_bars=20)
        if len(bar_df) < 6:
            return False
        
        prices = bar_df['close'].to_numpy()
        volumes = bar_df['volume'].to_numpy()
        
        return detect_momentum_divergence_numba(
            prices, volumes, CONFIG.signals.momentum_divergence_periods
        )
    
    def _detect_absorption(self, symbol: str, current_price: float) -> bool:
        """Detect absorption pattern (high volume, price stalled)."""
        if symbol not in self.tick_history:
            return False
        
        ticks = self.tick_history[symbol]
        if len(ticks) < CONFIG.signals.absorption_lookback_ticks:
            return False
        
        recent_ticks = ticks[-CONFIG.signals.absorption_lookback_ticks:]
        prices = np.array([t['price'] for t in recent_ticks])
        volumes = np.array([float(t['volume']) for t in recent_ticks])
        
        return detect_absorption_numba(
            prices=prices,
            volumes=volumes,
            lookback=CONFIG.signals.absorption_lookback_ticks,
            volume_threshold=2.0,
            price_change_threshold=0.005
        )
    
    def _calculate_confidence(self, vwap_extension: float, volume_ratio: float,
                              volume_exhausted: bool, near_high: bool,
                              momentum_divergence: bool, absorption: bool) -> float:
        """Calculate signal confidence score (0.0 - 1.0)."""
        score = 0.0
        
        # VWAP extension (optimal: 120-150%)
        if 1.20 <= vwap_extension <= 1.50:
            score += 0.30
        elif 1.50 < vwap_extension <= 2.0:
            score += 0.25
        elif vwap_extension > 2.0:
            score += 0.15
        
        # Volume exhaustion (lower ratio = better)
        if volume_ratio < 0.40:
            score += 0.30
        elif volume_ratio < 0.50:
            score += 0.25
        elif volume_ratio < 0.60:
            score += 0.20
        
        # Near day's high
        if near_high:
            score += 0.15
        
        # Momentum divergence
        if momentum_divergence:
            score += 0.15
        
        # Absorption
        if absorption:
            score += 0.10
        
        return min(1.0, score)
    
    def clear_symbol_state(self, symbol: str):
        """Clear tracking state for symbol (after position closed)."""
        if symbol in self.position_states:
            del self.position_states[symbol]
        if symbol in self.volume_profiles:
            del self.volume_profiles[symbol]


# Import for type hints
from src.risk.position_manager import Position
