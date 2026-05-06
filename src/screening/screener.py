"""
Asset Screener Module - Intraday Parabolic Setup Detection
Identifies stocks with intraday parabolic moves for volume exhaustion fading.
"""
from typing import List, Dict, Optional, Set
from datetime import datetime, timedelta
from dataclasses import dataclass
import numpy as np

from src.utils.config import CONFIG
from src.utils.logger import logger
from src.data.alpaca_client import AlpacaClient


@dataclass
class ScreenedAsset:
    """Qualified asset for intraday parabolic reversal strategy."""
    symbol: str
    current_price: float
    day_high: float
    day_low: float
    day_open: float
    day_volume: int
    percent_gain: float           # From day's open
    percent_from_high: float      # Distance from HOD
    shortable: bool
    easy_to_borrow: bool
    float_shares: Optional[float] = None
    setup_quality: float = 0.0    # 0-100 score
    
    # Intraday tracking
    volume_peak: float = 0.0      # Peak volume seen so far
    time_qualified: Optional[datetime] = None
    
    def is_valid_setup(self) -> bool:
        """Check if asset meets all criteria for monitoring."""
        return (
            self.shortable and
            self.easy_to_borrow and
            self.percent_gain >= CONFIG.screening.min_percent_gain and
            self.percent_gain <= CONFIG.screening.max_percent_gain and
            self.current_price >= CONFIG.screening.min_price and
            self.current_price <= CONFIG.screening.max_price and
            self.day_volume >= CONFIG.screening.min_volume
        )


class ParabolicScreener:
    """
    Real-time screener for intraday parabolic setups.
    Monitors stocks with 60%+ intraday gains for volume exhaustion entry.
    """
    
    def __init__(self, alpaca_client: AlpacaClient):
        self.client = alpaca_client
        self.watched_symbols: Set[str] = set()
        self.screened_assets: Dict[str, ScreenedAsset] = {}
        self.blacklist: Dict[str, datetime] = {}  # Optional compliance
        
        # Intraday tracking
        self.price_history: Dict[str, List[Dict]] = {}
        self.volume_history: Dict[str, List[Dict]] = {}
        
    def load_blacklist(self):
        """Load blacklisted symbols from persistent storage."""
        pass
    
    def add_to_blacklist(self, symbol: str, reason: str = "loss_realized"):
        """Add symbol to blacklist (optional compliance)."""
        if not CONFIG.compliance.spain_homogeneous_loss_rule:
            return
        
        self.blacklist[symbol] = datetime.now()
        logger.info(f"Added to blacklist", symbol=symbol, reason=reason)
    
    def is_blacklisted(self, symbol: str) -> bool:
        """Check if symbol is blacklisted."""
        if symbol not in self.blacklist:
            return False
        
        blacklist_date = self.blacklist[symbol]
        days_elapsed = (datetime.now() - blacklist_date).days
        
        if days_elapsed >= CONFIG.compliance.blacklist_duration_days:
            del self.blacklist[symbol]
            return False
        
        return True
    
    def get_top_gainers(self, min_gain_percent: float = 50.0) -> List[Dict]:
        """
        Get stocks with significant intraday gains.
        Uses predefined universe of micro/small-cap stocks.
        """
        try:
            # Get all active assets
            assets = self.client.trading_client.get_all_assets()
            
            # Filter for US equities in our target universe
            us_equities = [
                a for a in assets
                if (a.exchange == 'NYSE' or a.exchange == 'NASDAQ')
                and a.tradable
                and a.shortable
            ]
            
            return [{'symbol': a.symbol} for a in us_equities[:200]]
            
        except Exception as e:
            logger.error(f"Failed to get assets: {e}")
            return []
    
    def screen_symbol(self, symbol: str, market_data: Dict) -> Optional[ScreenedAsset]:
        """
        Screen a single symbol for intraday parabolic setup.
        
        Key Difference from v1: NO multi-day requirement.
        We only care about TODAY's move.
        """
        # Check blacklist
        if self.is_blacklisted(symbol):
            return None
        
        # Check shortability
        asset_info = self.client.check_asset_shortable(symbol)
        if not asset_info.get('shortable') or not asset_info.get('easy_to_borrow'):
            return None
        
        try:
            # Get current quote
            quote = market_data.get('quote', {})
            current_price = quote.get('ask_price', 0)
            
            if current_price == 0:
                return None
            
            # Get daily bar data
            daily = market_data.get('daily', {})
            day_open = daily.get('open', current_price)
            day_high = daily.get('high', current_price)
            day_low = daily.get('low', current_price)
            day_volume = daily.get('volume', 0)
            
            # Calculate intraday gain (key metric)
            percent_gain = ((current_price - day_open) / day_open) * 100
            percent_from_high = ((day_high - current_price) / day_high) * 100
            
            # Check volume
            if day_volume < CONFIG.screening.min_volume:
                return None
            
            # Check price range
            if not (CONFIG.screening.min_price <= current_price <= CONFIG.screening.max_price):
                return None
            
            # Check gain threshold (60%+ for intraday parabolic)
            if not (CONFIG.screening.min_percent_gain <= percent_gain <= CONFIG.screening.max_percent_gain):
                return None
            
            # Calculate setup quality score
            quality_score = self._calculate_quality_score(
                percent_gain=percent_gain,
                percent_from_high=percent_from_high,
                volume=day_volume,
                price_range=(day_high - day_low) / day_low if day_low > 0 else 0
            )
            
            # Track if this is a new qualification
            time_qualified = datetime.now()
            if symbol in self.screened_assets:
                time_qualified = self.screened_assets[symbol].time_qualified or time_qualified
            
            asset = ScreenedAsset(
                symbol=symbol,
                current_price=current_price,
                day_high=day_high,
                day_low=day_low,
                day_open=day_open,
                day_volume=day_volume,
                percent_gain=percent_gain,
                percent_from_high=percent_from_high,
                shortable=asset_info['shortable'],
                easy_to_borrow=asset_info['easy_to_borrow'],
                setup_quality=quality_score,
                time_qualified=time_qualified
            )
            
            if asset.is_valid_setup():
                is_new = symbol not in self.screened_assets
                self.screened_assets[symbol] = asset
                
                if is_new:
                    logger.info(
                        f"New parabolic setup detected",
                        symbol=symbol,
                        gain=f"{percent_gain:.1f}%",
                        price=f"${current_price:.2f}",
                        volume=day_volume,
                        quality=quality_score
                    )
                
                return asset
            
        except Exception as e:
            logger.error(f"Error screening {symbol}: {e}")
        
        return None
    
    def _calculate_quality_score(self, percent_gain: float, percent_from_high: float,
                                  volume: int, price_range: float) -> float:
        """
        Calculate setup quality score (0-100).
        Higher score = better parabolic reversal candidate.
        """
        score = 0.0
        
        # Gain magnitude (optimal: 60-150% for intraday)
        if 60 <= percent_gain <= 100:
            score += 35
        elif 100 < percent_gain <= 200:
            score += 30
        else:
            score += 20
        
        # Distance from high (some pullback on volume exhaustion is ideal)
        if 2 <= percent_from_high <= 8:
            score += 25  # Perfect - some exhaustion showing
        elif percent_from_high < 2:
            score += 20  # Still pushing, might extend more
        elif 8 < percent_from_high <= 15:
            score += 15  # Already pulled back, might have missed entry
        else:
            score += 10  # Too far from high
        
        # Volume (higher is better for liquidity)
        if volume > 5000000:
            score += 25
        elif volume > 2000000:
            score += 20
        elif volume > 1000000:
            score += 15
        else:
            score += 10
        
        # Price range (wider range = more volatile = better fade potential)
        if price_range > 0.80:  # >80% intraday range
            score += 15
        elif price_range > 0.50:
            score += 10
        else:
            score += 5
        
        return min(100, score)
    
    def update_volume_peak(self, symbol: str, volume_5min: float):
        """Update peak volume for tracked symbol."""
        if symbol in self.screened_assets:
            asset = self.screened_assets[symbol]
            if volume_5min > asset.volume_peak:
                asset.volume_peak = volume_5min
    
    def get_volume_exhaustion_candidates(self) -> List[ScreenedAsset]:
        """
        Get symbols showing volume exhaustion for entry.
        """
        candidates = []
        
        for symbol, asset in self.screened_assets.items():
            if not asset.is_valid_setup():
                continue
            
            # Check if we've tracked enough volume
            if asset.volume_peak == 0:
                continue
            
            # This will be further analyzed by signal engine
            candidates.append(asset)
        
        # Sort by setup quality
        candidates.sort(key=lambda x: x.setup_quality, reverse=True)
        return candidates
    
    def update_screened_assets(self, market_data_map: Dict[str, Dict]):
        """Update all screened assets with latest data."""
        for symbol, data in market_data_map.items():
            if symbol in self.screened_assets:
                asset = self.screened_assets[symbol]
                
                # Update current price
                quote = data.get('quote', {})
                asset.current_price = quote.get('ask_price', asset.current_price)
                
                # Update percent from high
                asset.percent_from_high = ((asset.day_high - asset.current_price) / asset.day_high) * 100
                
                # Check if still valid
                if asset.current_price < asset.day_open * 1.20:  # Dropped below 20% gain
                    logger.info(f"Asset no longer parabolic", symbol=symbol, 
                               gain=f"{((asset.current_price - asset.day_open) / asset.day_open * 100):.1f}%")
    
    def get_qualified_symbols(self) -> List[str]:
        """Get list of currently qualified symbols."""
        return [
            s for s, a in self.screened_assets.items()
            if a.is_valid_setup()
        ]
    
    def get_best_setup(self) -> Optional[ScreenedAsset]:
        """Get highest quality setup currently available."""
        qualified = [
            a for a in self.screened_assets.values()
            if a.is_valid_setup()
        ]
        
        if not qualified:
            return None
        
        return max(qualified, key=lambda x: x.setup_quality)
    
    def clear_screened(self):
        """Clear screened assets (end of day)."""
        count = len(self.screened_assets)
        self.screened_assets.clear()
        self.price_history.clear()
        self.volume_history.clear()
        logger.info(f"Screened assets cleared", count=count)
    
    def track_price(self, symbol: str, price: float, timestamp: datetime):
        """Track price history for analysis."""
        if symbol not in self.price_history:
            self.price_history[symbol] = []
        
        self.price_history[symbol].append({
            'price': price,
            'timestamp': timestamp
        })
        
        # Keep only recent history
        cutoff = datetime.now() - timedelta(hours=8)
        self.price_history[symbol] = [
            p for p in self.price_history[symbol]
            if p['timestamp'] > cutoff
        ]
