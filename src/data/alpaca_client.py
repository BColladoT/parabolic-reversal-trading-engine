"""
Alpaca Markets API Client
WebSocket streaming and REST API integration for real-time data and order execution.
"""
import asyncio
import json
import os
import random
from typing import Callable, Dict, Set, Optional
from datetime import datetime
import threading
import time

import websockets
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderType
from alpaca.data.live.stock import StockDataStream
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockSnapshotRequest

from src.utils.config import CONFIG
from src.utils.logger import logger
from src.data.polars_engine import TickData


def compute_backoff(attempt: int, base: float = 1.0, cap: float = 60.0) -> float:
    """Exponential backoff with full jitter: ``random.uniform(0, min(base * 2**attempt, cap))``.

    Used by :meth:`AlpacaClient._ws_handler` to space reconnect attempts under
    "thundering herd" conditions (e.g. broker outage recovery). The full-jitter
    strategy is from the AWS architecture blog post on retries.
    """
    expo = min(base * (2 ** max(0, int(attempt))), cap)
    return random.uniform(0, expo)


class AlpacaClient:
    """
    Unified Alpaca client for both WebSocket streaming and REST API.
    Handles authentication, reconnection, and data parsing.
    """
    
    def __init__(self):
        self.api_key = CONFIG.broker.api_key
        self.secret_key = CONFIG.broker.secret_key
        self.paper = CONFIG.broker.paper_trading
        
        # REST client
        self.trading_client = TradingClient(
            api_key=self.api_key,
            secret_key=self.secret_key,
            paper=self.paper
        )

        # Historical-data REST client (for the dynamic scanner — full market
        # snapshots in batches; bypasses the WS IEX 30-symbol cap).
        self.data_client = StockHistoricalDataClient(
            api_key=self.api_key,
            secret_key=self.secret_key,
        )
        
        # WebSocket
        self.ws = None
        # Use IEX feed (free) instead of SIP (requires paid subscription)
        self.ws_url = "wss://stream.data.alpaca.markets/v2/iex"
        self.ws_thread: Optional[threading.Thread] = None
        self.ws_running = False
        # Reference to the event loop owned by ws_thread. Used so the main
        # thread can schedule coroutines (close, subscribe) on the correct
        # loop via asyncio.run_coroutine_threadsafe. None when no thread.
        self._ws_loop: Optional[asyncio.AbstractEventLoop] = None
        self.subscribed_symbols: Set[str] = set()
        
        # Callbacks
        self.tick_callback: Optional[Callable[[TickData], None]] = None
        self.bar_callback: Optional[Callable[[dict], None]] = None
        self.quote_callback: Optional[Callable[[dict], None]] = None
        
        # Connection state
        self.last_message_time = time.time()
        self.reconnect_delay = CONFIG.performance.reconnect_delay_seconds
        self.max_reconnect_delay = 60
        self._reconnect_attempt = 0

    # ==================== Reliability Primitives ====================

    def is_feed_stale(self, max_age_s: int = 30) -> bool:
        """True if no WebSocket message has been received within ``max_age_s`` seconds."""
        return (time.time() - self.last_message_time) > max_age_s

    def poll_fill(
        self,
        order_id: str,
        timeout_s: float = 10.0,
        interval_s: float = 0.5,
    ) -> dict:
        """Poll Alpaca for order status until it reaches a terminal state or ``timeout_s`` elapses.

        Returns a dict with ``status``, ``filled_qty``, ``filled_avg_price``.
        On exception or timeout, returns the last observed state (or a
        ``status="unknown"`` placeholder if no observation was made).
        """
        deadline = time.time() + timeout_s
        last: Optional[Dict] = None
        terminal = ("filled", "canceled", "rejected", "expired")
        while time.time() < deadline:
            try:
                order = self.trading_client.get_order_by_id(order_id)
                raw_status = order.status
                if hasattr(raw_status, "name"):
                    status_str = raw_status.name.lower()
                else:
                    status_str = str(raw_status).split(".")[-1].lower()
                last = {
                    "status": status_str,
                    "filled_qty": int(float(order.filled_qty or 0)),
                    "filled_avg_price": float(order.filled_avg_price or 0.0),
                }
                if last["status"] in terminal:
                    return last
            except Exception as e:
                logger.warning("poll_fill error", order_id=order_id, error=str(e))
            time.sleep(interval_s)
        return last or {"status": "unknown", "filled_qty": 0, "filled_avg_price": 0.0}

    # ==================== REST API Methods ====================
    
    def get_account(self) -> dict:
        """Get account information."""
        try:
            account = self.trading_client.get_account()
            return {
                'id': account.id,
                'equity': float(account.equity),
                'cash': float(account.cash),
                'buying_power': float(account.buying_power),
                'daytrading_buying_power': float(account.daytrading_buying_power) if account.daytrading_buying_power else 0,
                'portfolio_value': float(account.portfolio_value),
                'status': account.status
            }
        except Exception as e:
            logger.error(f"Failed to get account: {e}")
            return {}
    
    def get_snapshots(self, symbols, batch_size: int = 500) -> Dict[str, dict]:
        """Fetch multi-symbol snapshots from Alpaca historical data API.

        The snapshot endpoint returns the latest trade, latest quote, latest
        minute bar, latest daily bar, and previous daily bar for each symbol
        in one call. Used by the DynamicScanner to screen the full
        micro-cap universe (~3,500 symbols) without hitting the WS streaming
        cap (~30 symbols on IEX free tier).

        Returns a dict ``{symbol: {latest_trade, latest_quote, daily_bar,
        prev_daily_bar, minute_bar}}`` with primitives extracted from the
        Alpaca SDK objects. Failures are logged but don't raise — partial
        results are returned for whatever batches succeeded.

        Args:
            symbols: iterable of symbol strings.
            batch_size: max symbols per request (Alpaca recommends <= 1000).
        """
        symbols = list(symbols)
        out: Dict[str, dict] = {}
        if not symbols:
            return out
        for i in range(0, len(symbols), batch_size):
            chunk = symbols[i:i + batch_size]
            try:
                req = StockSnapshotRequest(symbol_or_symbols=chunk)
                snaps = self.data_client.get_stock_snapshot(req)
                for sym, snap in snaps.items():
                    daily = getattr(snap, "daily_bar", None)
                    prev_daily = getattr(snap, "previous_daily_bar", None)
                    latest_trade = getattr(snap, "latest_trade", None)
                    latest_quote = getattr(snap, "latest_quote", None)
                    minute_bar = getattr(snap, "minute_bar", None)
                    out[sym] = {
                        "daily_bar": {
                            "open": getattr(daily, "open", None),
                            "high": getattr(daily, "high", None),
                            "low": getattr(daily, "low", None),
                            "close": getattr(daily, "close", None),
                            "volume": getattr(daily, "volume", None),
                        } if daily else None,
                        "previous_daily_bar": {
                            "close": getattr(prev_daily, "close", None),
                        } if prev_daily else None,
                        "latest_trade_price": getattr(latest_trade, "price", None),
                        "latest_quote_ask": getattr(latest_quote, "ask_price", None),
                        "latest_quote_bid": getattr(latest_quote, "bid_price", None),
                        "minute_bar_close": getattr(minute_bar, "close", None),
                    }
            except Exception as e:
                logger.error(f"get_snapshots batch {i}-{i+len(chunk)} failed: {e}")
        return out

    def check_asset_shortable(self, symbol: str) -> Dict:
        """
        Check if an asset is shortable and easy to borrow.
        Critical for parabolic reversal strategy.
        """
        try:
            asset = self.trading_client.get_asset(symbol)
            return {
                'symbol': symbol,
                'shortable': asset.shortable,
                'easy_to_borrow': asset.easy_to_borrow,
                'marginable': asset.marginable,
                'tradable': asset.tradable,
                'status': asset.status
            }
        except Exception as e:
            logger.error(f"Failed to check asset {symbol}: {e}")
            return {
                'symbol': symbol,
                'shortable': False,
                'easy_to_borrow': False,
                'error': str(e)
            }
    
    def get_positions(self) -> list:
        """Get current positions."""
        try:
            positions = self.trading_client.get_all_positions()
            return [
                {
                    'symbol': p.symbol,
                    'qty': int(p.qty),
                    'market_value': float(p.market_value),
                    'avg_entry_price': float(p.avg_entry_price),
                    'current_price': float(p.current_price),
                    'unrealized_pl': float(p.unrealized_pl),
                    'side': 'short' if int(p.qty) < 0 else 'long'
                }
                for p in positions
            ]
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []
    
    def submit_short_order(
        self,
        symbol: str,
        qty: int,
        limit_price: Optional[float] = None,
        time_in_force: str = "ioc",
        client_order_id: Optional[str] = None,
    ) -> Dict:
        """
        Submit a short sell order.
        Uses limit orders by default to control slippage.

        ``client_order_id`` provides idempotency — Alpaca rejects duplicate
        submissions sharing the same client_order_id. If not supplied, we
        autogenerate ``f"{symbol}-{int(time.time()*1000)}"`` so that an
        accidental retry within the same millisecond is still deduped.
        """
        if client_order_id is None:
            client_order_id = f"{symbol}-{int(time.time() * 1000)}"
        try:
            if limit_price:
                order_data = LimitOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=OrderSide.SELL,
                    limit_price=limit_price,
                    time_in_force=TimeInForce.IOC if time_in_force == "ioc" else TimeInForce.DAY,
                    client_order_id=client_order_id,
                )
            else:
                order_data = MarketOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.IOC,
                    client_order_id=client_order_id,
                )

            order = self.trading_client.submit_order(order_data=order_data)

            logger.info(
                f"Short order submitted",
                symbol=symbol,
                qty=qty,
                limit_price=limit_price,
                order_id=order.id,
                client_order_id=client_order_id,
            )

            return {
                'success': True,
                'order_id': order.id,
                'status': order.status,
                'symbol': symbol,
                'qty': qty,
                'client_order_id': client_order_id,
            }

        except Exception as e:
            logger.error(f"Failed to submit short order for {symbol}: {e}")
            return {
                'success': False,
                'error': str(e),
                'symbol': symbol,
                'client_order_id': client_order_id,
            }
    
    def submit_cover_order(
        self,
        symbol: str,
        qty: int,
        limit_price: Optional[float] = None,
        client_order_id: Optional[str] = None,
    ) -> Dict:
        """Submit a buy-to-cover order.

        ``client_order_id`` provides idempotency; autogenerated when missing.
        """
        if client_order_id is None:
            client_order_id = f"{symbol}-cover-{int(time.time() * 1000)}"
        try:
            if limit_price:
                order_data = LimitOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=OrderSide.BUY,
                    limit_price=limit_price,
                    time_in_force=TimeInForce.IOC,
                    client_order_id=client_order_id,
                )
            else:
                order_data = MarketOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.IOC,
                    client_order_id=client_order_id,
                )

            order = self.trading_client.submit_order(order_data=order_data)

            logger.info(
                f"Cover order submitted",
                symbol=symbol,
                qty=qty,
                order_id=order.id,
                client_order_id=client_order_id,
            )

            return {
                'success': True,
                'order_id': order.id,
                'status': order.status,
                'client_order_id': client_order_id,
            }

        except Exception as e:
            logger.error(f"Failed to submit cover order for {symbol}: {e}")
            return {
                'success': False,
                'error': str(e),
                'client_order_id': client_order_id,
            }
    
    def close_all_positions(self) -> Dict:
        """Close all positions (flatten before market close)."""
        try:
            result = self.trading_client.close_all_positions()
            logger.info("All positions closed")
            return {'success': True, 'result': result}
        except Exception as e:
            logger.error(f"Failed to close all positions: {e}")
            return {'success': False, 'error': str(e)}
    
    # ==================== WebSocket Methods ====================
    
    async def _ws_connect(self):
        """Establish WebSocket connection."""
        try:
            # Connect without headers - Alpaca uses message-based auth
            self.ws = await websockets.connect(self.ws_url)
            
            # First message is always "connected"
            connected_msg = await self.ws.recv()
            logger.debug(f"WebSocket connected: {connected_msg}")
            
            # Authenticate (10 second window)
            auth_msg = {
                "action": "auth",
                "key": self.api_key,
                "secret": self.secret_key
            }
            await self.ws.send(json.dumps(auth_msg))
            
            # Wait for auth response (should be "authenticated")
            auth_response = await self.ws.recv()
            logger.info(f"WebSocket auth response: {auth_response}")
            
            # Verify authentication succeeded
            resp_data = json.loads(auth_response)
            if resp_data and len(resp_data) > 0:
                if resp_data[0].get("msg") == "authenticated":
                    logger.info("WebSocket authenticated successfully")
                    # Reset reconnect attempt counter on successful auth.
                    self._reconnect_attempt = 0
                    return True
                else:
                    logger.error(f"WebSocket auth failed: {resp_data}")
                    return False

            return True
            
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            return False
    
    async def _ws_unsubscribe(self, symbols: Set[str]):
        """Unsubscribe from streaming data for the given symbols.

        Alpaca v2 streaming protocol supports symmetric unsubscribe via the
        same channel layout as subscribe. Used by update_subscriptions() to
        rotate the active symbol set as the DynamicScanner finds new top
        candidates and drops stale ones — keeps the IEX free-tier slot
        budget (~30 channel-subscriptions) usable.
        """
        if not symbols or not self.ws:
            return
        symbol_list = list(symbols)
        unsubscribe_msg = {
            "action": "unsubscribe",
            "trades": symbol_list,
            "quotes": symbol_list,
            "bars": symbol_list,
        }
        try:
            await self.ws.send(json.dumps(unsubscribe_msg))
            logger.info(f"Unsubscribed from: {symbol_list}")
        except Exception as e:
            logger.error(f"Unsubscribe error: {e}")

    def update_subscriptions(self, target: Set[str]):
        """Diff current subscriptions vs ``target`` and send adds/removes.

        Called by the engine's dynamic-scanner tick to rotate streaming
        symbols. Thread-safe (uses run_coroutine_threadsafe via the
        WS-thread event loop). No-op if the WS is not connected yet.
        """
        target = {s.upper() for s in target}
        to_add = target - self.subscribed_symbols
        to_remove = self.subscribed_symbols - target
        if not to_add and not to_remove:
            return
        self.subscribed_symbols = set(target)
        if self.ws is None or not self._is_ws_open():
            # Not connected yet; the next reconnect cycle will pick up the
            # updated subscribed_symbols set automatically (see _ws_handler).
            logger.info(f"WS not open; subscriptions deferred (+{len(to_add)} / -{len(to_remove)})")
            return
        if to_remove:
            self._schedule_on_ws_loop(self._ws_unsubscribe(to_remove))
        if to_add:
            self._schedule_on_ws_loop(self._ws_subscribe(to_add))

    async def _ws_subscribe(self, symbols: Set[str]):
        """Subscribe to real-time data streams.

        Alpaca v2 streaming API expects symbol lists per stream. The previous
        ``"bars": ["1Min"]`` was a syntax error (Alpaca returned
        ``{'T': 'error', 'code': 400, 'msg': 'invalid syntax'}``) because the
        bars channel takes SYMBOLS, not timeframe strings. IEX stream emits
        1-minute bars by default — no timeframe parameter exists. Subscribe
        ``bars`` to the same symbol list as trades/quotes.
        """
        if not symbols:
            return

        symbol_list = list(symbols)

        # Subscribe to trades, quotes, and bars — all keyed by symbol list.
        subscribe_msg = {
            "action": "subscribe",
            "trades": symbol_list,
            "quotes": symbol_list,
            "bars": symbol_list,
        }

        try:
            await self.ws.send(json.dumps(subscribe_msg))
            logger.info(f"Subscribed to: {symbol_list}")
        except Exception as e:
            logger.error(f"Subscribe error: {e}")
    
    async def _ws_handler(self):
        """Main WebSocket message handler."""
        while self.ws_running:
            try:
                if self.ws is None or not self._is_ws_open():
                    logger.info("Reconnecting WebSocket...")
                    success = await self._ws_connect()
                    if success and self.subscribed_symbols:
                        await self._ws_subscribe(self.subscribed_symbols)
                    # Exponential backoff with full jitter on reconnect path.
                    self._reconnect_attempt = getattr(self, "_reconnect_attempt", 0) + 1
                    delay = compute_backoff(
                        self._reconnect_attempt,
                        base=self.reconnect_delay,
                        cap=self.max_reconnect_delay,
                    )
                    logger.info(
                        "WebSocket backoff",
                        attempt=self._reconnect_attempt,
                        delay=delay,
                    )
                    await asyncio.sleep(delay)
                    continue

                message = await self.ws.recv()
                self.last_message_time = time.time()

                # Parse message
                data = json.loads(message)

                if isinstance(data, list):
                    for item in data:
                        await self._process_message(item)
                else:
                    await self._process_message(data)

            except websockets.exceptions.ConnectionClosed:
                logger.warning("WebSocket connection closed, reconnecting...")
                self.ws = None
                self._reconnect_attempt = getattr(self, "_reconnect_attempt", 0) + 1
                delay = compute_backoff(
                    self._reconnect_attempt,
                    base=self.reconnect_delay,
                    cap=self.max_reconnect_delay,
                )
                await asyncio.sleep(delay)
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                self._reconnect_attempt = getattr(self, "_reconnect_attempt", 0) + 1
                delay = compute_backoff(
                    self._reconnect_attempt,
                    base=self.reconnect_delay,
                    cap=self.max_reconnect_delay,
                )
                await asyncio.sleep(delay)
    
    async def _process_message(self, data: dict):
        """Process incoming WebSocket message."""
        msg_type = data.get('T')
        
        if msg_type == 't':  # Trade
            tick = TickData(
                timestamp=datetime.now(),
                symbol=data.get('S', ''),
                price=float(data.get('p', 0)),
                size=int(data.get('s', 0)),
                side='A' if data.get('t') == 'A' else 'B',  # A=Ask, B=Bid
                exchange=data.get('x', '')
            )
            
            if self.tick_callback:
                self.tick_callback(tick)
                
        elif msg_type == 'q':  # Quote
            if self.quote_callback:
                self.quote_callback(data)
                
        elif msg_type == 'b':  # Bar
            if self.bar_callback:
                self.bar_callback(data)
                
        elif msg_type == 'success':
            logger.info(f"WebSocket success: {data}")
        elif msg_type == 'error':
            logger.error(f"WebSocket error: {data}")
    
    def start_websocket(self, symbols: Set[str]):
        """Start WebSocket connection in background thread.

        Idempotent: if a thread is already running, this is a no-op. Callers
        (e.g., the engine watchdog) MUST NOT pair this with stop_websocket()
        for transient connection drops — the internal _ws_handler loop already
        reconnects with exponential backoff. Only restart externally if the
        thread itself has died (check ws_thread.is_alive()).
        """
        if self.ws_thread is not None and self.ws_thread.is_alive():
            # Refresh subscribed set in case caller passed new symbols
            self.subscribed_symbols = symbols
            return
        self.subscribed_symbols = symbols
        self.ws_running = True

        def run_async_loop():
            loop = asyncio.new_event_loop()
            self._ws_loop = loop
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self._ws_handler())
            finally:
                self._ws_loop = None
                try:
                    loop.close()
                except Exception:
                    pass

        self.ws_thread = threading.Thread(target=run_async_loop, daemon=True)
        self.ws_thread.start()
        logger.info("WebSocket thread started")

    def _schedule_on_ws_loop(self, coro) -> None:
        """Run ``coro`` on the WebSocket's own event loop from the main thread.

        asyncio.create_task() called from a non-asyncio thread raises
        RuntimeError. asyncio.run_coroutine_threadsafe() is the supported API
        for cross-thread scheduling. Silently no-ops if the WS loop isn't
        running (e.g., during shutdown teardown).
        """
        loop = self._ws_loop
        if loop is None or loop.is_closed():
            return
        try:
            asyncio.run_coroutine_threadsafe(coro, loop)
        except RuntimeError:
            pass

    def stop_websocket(self):
        """Stop WebSocket connection cleanly from any thread."""
        self.ws_running = False  # signals _ws_handler loop to exit
        if self.ws is not None:
            self._schedule_on_ws_loop(self.ws.close())
        if self.ws_thread is not None:
            self.ws_thread.join(timeout=10)
        self.ws_thread = None
        self.ws = None
        self._ws_loop = None
        logger.info("WebSocket stopped")

    def subscribe_symbols(self, symbols: Set[str]):
        """Add symbols to subscription."""
        new_symbols = symbols - self.subscribed_symbols
        if new_symbols:
            self.subscribed_symbols.update(new_symbols)
            if self.ws and self._is_ws_open():
                self._schedule_on_ws_loop(self._ws_subscribe(new_symbols))

    def unsubscribe_symbols(self, symbols: Set[str]):
        """Remove symbols from subscription."""
        self.subscribed_symbols -= symbols
        # Alpaca doesn't support unsubscribe, need to reconnect
        if self.ws:
            self._schedule_on_ws_loop(self.ws.close())
    
    def set_tick_callback(self, callback: Callable[[TickData], None]):
        """Set callback for tick data."""
        self.tick_callback = callback
    
    def set_bar_callback(self, callback: Callable[[dict], None]):
        """Set callback for bar data."""
        self.bar_callback = callback
    
    def _is_ws_open(self) -> bool:
        """Check if WebSocket connection is open (compatible with websockets 12+)."""
        if self.ws is None:
            return False
        # websockets 12+ uses state attribute
        if hasattr(self.ws, 'state'):
            from websockets.protocol import State
            return self.ws.state == State.OPEN
        # Fallback for older versions
        if hasattr(self.ws, 'closed'):
            return not self.ws.closed
        return True
    
    def is_connected(self) -> bool:
        """Check if WebSocket is connected."""
        return self._is_ws_open()
