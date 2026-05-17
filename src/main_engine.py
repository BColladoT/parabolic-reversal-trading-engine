"""
Parabolic Reversal Trading Engine - Main Orchestrator
Autonomous, self-healing algorithmic trading system for fading blow-off tops.
"""
import os
import sys
import time
import signal
import asyncio
import threading
from datetime import datetime, timedelta
from typing import Optional, Set, Dict
import pytz

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import CONFIG
from src.utils.logger import logger
from src.data.alpaca_client import AlpacaClient
from src.data.polars_engine import PolarsSignalEngine, TickData
from src.screening.screener import ParabolicScreener, ScreenedAsset
from src.risk.position_manager import RiskManager, Position
from src.execution.signal_engine import ParabolicSignalEngine, TradeSignal, SignalType
from src.utils.alerting import send_alert


class TradingEngine:
    """
    Main trading engine orchestrating all components.
    Self-healing, fault-tolerant design.
    """
    
    def __init__(self):
        logger.info("Initializing Parabolic Reversal Trading Engine...")
        
        # Timezone setup
        self.local_tz = pytz.timezone(CONFIG.timezone.local_tz)
        self.market_tz = pytz.timezone(CONFIG.timezone.market_tz)
        
        # Components
        self.alpaca = AlpacaClient()
        self.data_engine = PolarsSignalEngine()
        self.screener = ParabolicScreener(self.alpaca)
        self.risk_manager = RiskManager(self.alpaca)
        self.signal_engine = ParabolicSignalEngine(self.data_engine)
        
        # State
        self.running = False
        self.market_open = False
        self.subscribed_symbols: Set[str] = set()
        self.watch_symbols: Set[str] = set()
        
        # Self-healing
        self.error_count = 0
        self.max_errors = 10
        self.last_health_check = time.time()
        self._last_error_time = 0.0
        self._error_decay_seconds = 3600  # decay error_count after 1h clean

        # Daily reset tracking (ET trading day)
        self._last_reset_date = None

        # Stale-feed alert debounce (avoid spamming Slack when feed is down)
        self._last_stale_alert_ts = 0.0
        self._stale_alert_min_interval = 300  # 5 minutes
        
        # Setup callbacks
        self._setup_callbacks()
        
        # Signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("Trading engine initialized")
    
    def _setup_callbacks(self):
        """Setup data and signal callbacks."""
        self.alpaca.set_tick_callback(self._on_tick)
        self.signal_engine.register_callback(self._on_signal)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.shutdown()
        sys.exit(0)
    
    def _on_tick(self, tick: TickData):
        """Process incoming tick data."""
        try:
            # Update data engine
            self.data_engine.process_tick(tick)
            
            # Update signal engine
            self.signal_engine.update_tick(
                tick.symbol, tick.price, tick.size, tick.timestamp
            )
            
            # Update risk manager
            self.risk_manager.update_positions({tick.symbol: tick.price})
            
            # Check for exit signals on existing positions
            self._check_exits(tick.symbol, tick.price)
            
        except Exception as e:
            logger.error(f"Tick processing error: {e}")
            self._handle_error(e)
    
    def _on_signal(self, signal: TradeSignal):
        """Process trading signal."""
        try:
            if signal.signal_type == SignalType.ENTRY_SHORT:
                self._execute_entry(signal)
            elif signal.signal_type in [SignalType.EXIT_COVER, SignalType.STOP_LOSS, SignalType.PROFIT_TAKE]:
                self._execute_exit(signal)
        except Exception as e:
            logger.error(f"Signal processing error: {e}")
            self._handle_error(e)
    
    def _execute_entry(self, signal: TradeSignal):
        """Execute short entry."""
        symbol = signal.symbol

        # Hard circuit breaker - refuse all new entries once daily loss limit hit
        if self.risk_manager.check_daily_loss_limit():
            logger.critical(
                "Entry refused: daily loss limit tripped",
                symbol=symbol,
                daily_pnl=self.risk_manager.daily_pnl,
            )
            send_alert(
                "Daily loss limit tripped",
                f"Entry refused for {symbol}; daily_pnl={self.risk_manager.daily_pnl:.2f}",
                level="critical",
            )
            return

        # Check if already in position
        if symbol in self.risk_manager.positions:
            logger.info(f"Already in position for {symbol}, skipping entry")
            return

        # Get metrics for position sizing
        metrics = self.data_engine.get_signal_data(symbol)
        
        # Calculate position size
        sizing = self.risk_manager.calculate_position_size(
            symbol=symbol,
            entry_price=signal.price,
            atr=signal.atr,
            vwap=signal.vwap,
            parabolic_apex=self.screener.screened_assets.get(symbol, ScreenedAsset(
                symbol=symbol, current_price=signal.price, day_high=signal.price * 1.1,
                day_low=signal.price * 0.9, day_open=signal.price * 0.5,
                day_volume=1000000, percent_gain=100, percent_from_high=5,
                days_up=2, shortable=True, easy_to_borrow=True
            )).day_high,
            features=getattr(signal, "features", None),
        )
        
        if not sizing['valid']:
            logger.warning(f"Position sizing invalid for {symbol}: {sizing.get('reason')}")
            return
        
        # Calculate limit price (slight offset for better fill)
        limit_price = signal.price + (signal.price * 0.001)  # 0.1% above
        
        # Submit order
        result = self.alpaca.submit_short_order(
            symbol=symbol,
            qty=sizing['shares'],
            limit_price=limit_price,
            time_in_force=CONFIG.execution.time_in_force
        )
        
        if result.get('success'):
            # Record position
            self.risk_manager.open_position(
                symbol=symbol,
                entry_price=signal.price,
                qty=sizing['shares'],
                stop_loss=sizing['stop_loss'],
                profit_target=sizing['profit_target'],
                vwap=signal.vwap,
                parabolic_apex=sizing['stop_loss'],  # Use stop as apex proxy
                entry_features=getattr(signal, "features", None),  # tolerate older signals without this field
            )
            
            logger.info(
                f"Entry executed",
                symbol=symbol,
                shares=sizing['shares'],
                price=signal.price,
                stop=sizing['stop_loss'],
                target=sizing['profit_target']
            )
        else:
            logger.error(f"Entry failed for {symbol}: {result.get('error')}")
    
    def _execute_exit(self, signal: TradeSignal):
        """Execute position exit."""
        symbol = signal.symbol
        
        if symbol not in self.risk_manager.positions:
            return
        
        position = self.risk_manager.positions[symbol]
        
        # Calculate limit price for cover
        limit_price = signal.price - (signal.price * 0.001)  # 0.1% below for cover
        
        # Submit cover order
        result = self.alpaca.submit_cover_order(
            symbol=symbol,
            qty=position.qty,
            limit_price=limit_price
        )
        
        if result.get('success'):
            # Record close
            pnl = self.risk_manager.close_position(
                symbol=symbol,
                exit_price=signal.price,
                reason=signal.notes
            )
            
            # Check for loss (Spain tax compliance)
            if pnl < 0 and CONFIG.compliance.spain_homogeneous_loss_rule:
                self.screener.add_to_blacklist(symbol, "loss_realized")
            
            logger.info(f"Exit executed", symbol=symbol, pnl=f"${pnl:.2f}")
        else:
            # Try market order if limit fails
            result = self.alpaca.submit_cover_order(symbol=symbol, qty=position.qty)
            if result.get('success'):
                self.risk_manager.close_position(symbol, signal.price, "market_cover")
    
    def _check_exits(self, symbol: str, current_price: float):
        """Check for exit conditions."""
        if symbol not in self.risk_manager.positions:
            return
        
        position = self.risk_manager.positions[symbol]
        
        signal = self.signal_engine.generate_exit_signals(
            symbol=symbol,
            entry_price=position.entry_price,
            current_price=current_price,
            stop_loss=position.stop_loss,
            profit_target=position.profit_target,
            vwap=position.vwap_entry
        )
        
        if signal:
            self._on_signal(signal)
    
    def _handle_error(self, error: Exception):
        """Handle errors with self-healing logic."""
        self.error_count += 1
        self._last_error_time = time.time()
        logger.error(f"Error {self.error_count}/{self.max_errors}: {error}")

        if self.error_count >= self.max_errors:
            logger.critical("Max errors reached, initiating emergency shutdown")
            self.emergency_shutdown()

    def _maybe_decay_errors(self) -> None:
        """Reset error_count after a clean window (default: 1 hour)."""
        if self.error_count == 0:
            return
        decay_seconds = getattr(self, "_error_decay_seconds", 3600)
        last_error = getattr(self, "_last_error_time", 0.0)
        if (time.time() - last_error) >= decay_seconds:
            logger.info(
                "Error count decayed after clean window",
                previous=self.error_count,
            )
            self.error_count = 0
    
    def _reconcile_on_startup(self) -> None:
        """Refuse to start if the broker has positions the engine doesn't know about.

        Auto-adoption would be unsafe without per-position metadata (stop, target,
        entry VWAP). Operator must manually flatten or restore engine state instead.
        """
        broker_positions = self.alpaca.get_positions() or []
        if not broker_positions:
            return

        def _sym(p):
            # Tolerate both dict and SDK object shapes
            return p['symbol'] if isinstance(p, dict) else getattr(p, 'symbol', None)

        unknown = [
            _sym(p) for p in broker_positions
            if _sym(p) not in self.risk_manager.positions
        ]
        if unknown:
            logger.critical("Startup blocked: broker has positions", symbols=unknown)
            send_alert(
                "Startup blocked: unreconciled broker positions",
                f"Refusing to start. Broker holds positions not in engine state: {unknown}. "
                f"Operator must flatten or restore state before restart.",
                level="critical",
            )
            raise RuntimeError(
                f"broker has positions not in engine state: {unknown}"
            )

    def _maybe_reset_daily(self, now_et) -> None:
        """Reset daily risk stats once per ET trading day."""
        today = now_et.date()
        if self._last_reset_date != today:
            self.risk_manager.reset_daily_stats()
            self.error_count = 0  # also reset error decay
            self._last_reset_date = today
            logger.info("Daily reset complete", date=str(today))

    def _is_market_open(self) -> bool:
        """Check if market is currently open."""
        try:
            clock = self.alpaca.trading_client.get_clock()
            return clock.is_open
        except (ConnectionError, TimeoutError, OSError) as e:
            logger.warning(
                "Alpaca clock unavailable, falling back to time-based check",
                error=str(e),
            )
            # Fallback to time-based check
            now_et = datetime.now(self.market_tz)
            market_open = now_et.replace(hour=9, minute=30, second=0)
            market_close = now_et.replace(hour=16, minute=0, second=0)
            return market_open <= now_et <= market_close
    
    def _in_execution_window(self) -> bool:
        """Check if within optimal execution window."""
        now_et = datetime.now(self.market_tz)
        
        # Parse execution window times
        exec_start = datetime.strptime(CONFIG.timezone.execution_window_start, "%H:%M").time()
        exec_end = datetime.strptime(CONFIG.timezone.execution_window_end, "%H:%M").time()
        
        return exec_start <= now_et.time() <= exec_end
    
    def _should_flatten(self) -> bool:
        """Check if should flatten positions before close."""
        now_et = datetime.now(self.market_tz)
        flatten_time = datetime.strptime(CONFIG.timezone.flatten_time, "%H:%M").time()
        
        return now_et.time() >= flatten_time
    
    def _scan_for_setups(self):
        """Scan for new trading setups."""
        if not self._in_execution_window():
            return
        
        # In production, this would query a real scanner
        # For now, use predefined watchlist or mock scanning
        # TODO: Integrate with external scanner API
        
        # Mock: Assume we have some candidates
        # In reality, this would screen from market data
        pass
    
    def run(self):
        """Main trading loop."""
        logger.info("Starting trading engine main loop...")

        # Hard safety gate: refuse to start if broker has positions we don't track.
        # Raises RuntimeError on mismatch - operator must intervene.
        self._reconcile_on_startup()

        self.running = True
        
        # Start WebSocket
        self.alpaca.start_websocket(self.subscribed_symbols)
        
        # Main loop
        last_scan = 0
        last_health = 0
        
        try:
            while self.running:
                now = time.time()

                # Daily reset on new ET trading day
                now_et = datetime.now(self.market_tz)
                self._maybe_reset_daily(now_et)

                # Decay error count after a clean window
                self._maybe_decay_errors()

                # Market hours check
                self.market_open = self._is_market_open()
                
                if not self.market_open:
                    logger.debug("Market closed, waiting...")
                    time.sleep(60)
                    continue
                
                # Health check every 30 seconds
                if now - last_health > 30:
                    self._health_check()
                    last_health = now
                
                # Scan for setups every 5 seconds during execution window
                if now - last_scan > 5 and self._in_execution_window():
                    self._scan_for_setups()
                    last_scan = now
                
                # Check flatten time
                if self._should_flatten():
                    self._flatten_all()
                
                # Sleep to prevent CPU spin
                time.sleep(0.1)
                
        except Exception as e:
            logger.critical(f"Main loop error: {e}")
            self.emergency_shutdown()
    
    def _health_check(self):
        """Perform system health check."""
        # Stale-feed watchdog (debounced so we don't spam alerts every cycle)
        try:
            if self.alpaca.is_feed_stale():
                now = time.time()
                if (now - self._last_stale_alert_ts) >= self._stale_alert_min_interval:
                    logger.critical("WebSocket feed appears stale")
                    send_alert(
                        "WebSocket feed stale",
                        "No tick messages received within freshness window. "
                        "Reconnect will be attempted.",
                        level="critical",
                    )
                    self._last_stale_alert_ts = now
        except AttributeError:
            # AlpacaClient predates is_feed_stale — skip silently
            pass

        # Check WebSocket connection
        if not self.alpaca.is_connected():
            logger.warning("WebSocket disconnected, attempting reconnect...")
            self.alpaca.stop_websocket()
            self.alpaca.start_websocket(self.subscribed_symbols)
        
        # Check account status
        account = self.alpaca.get_account()
        if account.get('equity', 0) < CONFIG.risk.min_account_equity:
            logger.critical("Account equity below minimum, halting trading")
            self.running = False
        
        # Log position summary
        summary = self.risk_manager.get_position_summary()
        if summary['open_count'] > 0:
            logger.info(
                "Position summary",
                open=summary['open_count'],
                unrealized=f"${summary['unrealized_pnl']:.2f}",
                daily_pnl=f"${summary['daily_pnl']:.2f}"
            )
    
    def _flatten_all(self):
        """Close all positions before market close."""
        logger.info("Flattening all positions before market close...")
        
        # Get open positions
        open_positions = [
            s for s, p in self.risk_manager.positions.items()
            if p.status.value == 'open'
        ]
        
        for symbol in open_positions:
            # Get current price
            metrics = self.data_engine.get_signal_data(symbol)
            current_price = metrics.get('last_price', 0)
            
            if current_price > 0:
                signal = TradeSignal(
                    symbol=symbol,
                    signal_type=SignalType.EXIT_COVER,
                    timestamp=datetime.now(),
                    price=current_price,
                    confidence=1.0,
                    vwap=0,
                    atr=0,
                    volume_exhaustion=False,
                    absorption_detected=False,
                    notes="flatten_before_close"
                )
                self._on_signal(signal)
        
        # Use Alpaca's close all as backup
        self.alpaca.close_all_positions()
    
    def emergency_shutdown(self):
        """Emergency shutdown with position closure."""
        logger.critical("EMERGENCY SHUTDOWN INITIATED")
        send_alert(
            "Engine emergency shutdown initiated",
            f"TradingEngine.emergency_shutdown fired. error_count={getattr(self, 'error_count', 'n/a')}. "
            f"Attempting to flatten all positions and stop the WebSocket.",
            level="critical",
        )
        
        # Try to close all positions
        try:
            self.risk_manager.emergency_flatten_all()
        except Exception as e:
            logger.critical(f"Emergency flatten failed: {e}")
        
        # Stop WebSocket
        try:
            self.alpaca.stop_websocket()
        except (ConnectionError, TimeoutError, OSError, RuntimeError) as e:
            logger.warning("WebSocket stop failed during shutdown", error=str(e))
        
        self.running = False
        logger.critical("Emergency shutdown complete")
    
    def shutdown(self):
        """Graceful shutdown."""
        logger.info("Initiating graceful shutdown...")
        
        # Flatten positions
        self._flatten_all()
        
        # Stop WebSocket
        self.alpaca.stop_websocket()
        
        self.running = False
        logger.info("Shutdown complete")


def main():
    """Entry point."""
    engine = TradingEngine()
    
    # Validate credentials
    if not CONFIG.broker.api_key or not CONFIG.broker.secret_key:
        logger.critical("Alpaca API credentials not found in environment variables!")
        logger.critical("Please set ALPACA_API_KEY and ALPACA_SECRET_KEY")
        sys.exit(1)
    
    # Test connection
    try:
        account = engine.alpaca.get_account()
        logger.info(
            f"Connected to Alpaca",
            equity=f"${account.get('equity', 0):.2f}",
            buying_power=f"${account.get('buying_power', 0):.2f}"
        )
    except Exception as e:
        logger.critical(f"Failed to connect to Alpaca: {e}")
        sys.exit(1)
    
    # Start trading
    engine.run()


if __name__ == "__main__":
    main()
