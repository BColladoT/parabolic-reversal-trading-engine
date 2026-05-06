"""
Alpaca Markets API Client
WebSocket streaming and REST API integration for real-time data and order execution.
"""
import asyncio
import json
import os
from typing import Callable, Dict, Set, Optional
from datetime import datetime
import threading
import time

import websockets
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderType
from alpaca.data.live.stock import StockDataStream

from src.utils.config import CONFIG
from src.utils.logger import logger
from src.data.polars_engine import TickData


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
        
        # WebSocket
        self.ws = None
        # Use IEX feed (free) instead of SIP (requires paid subscription)
        self.ws_url = "wss://stream.data.alpaca.markets/v2/iex"
        self.ws_thread: Optional[threading.Thread] = None
        self.ws_running = False
        self.subscribed_symbols: Set[str] = set()
        
        # Callbacks
        self.tick_callback: Optional[Callable[[TickData], None]] = None
        self.bar_callback: Optional[Callable[[dict], None]] = None
        self.quote_callback: Optional[Callable[[dict], None]] = None
        
        # Connection state
        self.last_message_time = time.time()
        self.reconnect_delay = CONFIG.performance.reconnect_delay_seconds
        self.max_reconnect_delay = 60
        
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
        time_in_force: str = "ioc"
    ) -> Dict:
        """
        Submit a short sell order.
        Uses limit orders by default to control slippage.
        """
        try:
            if limit_price:
                order_data = LimitOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=OrderSide.SELL,
                    limit_price=limit_price,
                    time_in_force=TimeInForce.IOC if time_in_force == "ioc" else TimeInForce.DAY
                )
            else:
                order_data = MarketOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.IOC
                )
            
            order = self.trading_client.submit_order(order_data=order_data)
            
            logger.info(
                f"Short order submitted",
                symbol=symbol,
                qty=qty,
                limit_price=limit_price,
                order_id=order.id
            )
            
            return {
                'success': True,
                'order_id': order.id,
                'status': order.status,
                'symbol': symbol,
                'qty': qty
            }
            
        except Exception as e:
            logger.error(f"Failed to submit short order for {symbol}: {e}")
            return {
                'success': False,
                'error': str(e),
                'symbol': symbol
            }
    
    def submit_cover_order(
        self,
        symbol: str,
        qty: int,
        limit_price: Optional[float] = None
    ) -> Dict:
        """Submit a buy-to-cover order."""
        try:
            if limit_price:
                order_data = LimitOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=OrderSide.BUY,
                    limit_price=limit_price,
                    time_in_force=TimeInForce.IOC
                )
            else:
                order_data = MarketOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.IOC
                )
            
            order = self.trading_client.submit_order(order_data=order_data)
            
            logger.info(
                f"Cover order submitted",
                symbol=symbol,
                qty=qty,
                order_id=order.id
            )
            
            return {
                'success': True,
                'order_id': order.id,
                'status': order.status
            }
            
        except Exception as e:
            logger.error(f"Failed to submit cover order for {symbol}: {e}")
            return {
                'success': False,
                'error': str(e)
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
                    return True
                else:
                    logger.error(f"WebSocket auth failed: {resp_data}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            return False
    
    async def _ws_subscribe(self, symbols: Set[str]):
        """Subscribe to real-time data streams."""
        if not symbols:
            return
        
        symbol_list = list(symbols)
        
        # Subscribe to trades, quotes, and bars
        subscribe_msg = {
            "action": "subscribe",
            "trades": symbol_list,
            "quotes": symbol_list,
            "bars": ["1Min"]  # 1-minute bars
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
                    await asyncio.sleep(self.reconnect_delay)
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
                await asyncio.sleep(self.reconnect_delay)
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                await asyncio.sleep(self.reconnect_delay)
    
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
        """Start WebSocket connection in background thread."""
        self.subscribed_symbols = symbols
        self.ws_running = True
        
        def run_async_loop():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._ws_handler())
        
        self.ws_thread = threading.Thread(target=run_async_loop, daemon=True)
        self.ws_thread.start()
        logger.info("WebSocket thread started")
    
    def stop_websocket(self):
        """Stop WebSocket connection."""
        self.ws_running = False
        if self.ws:
            asyncio.create_task(self.ws.close())
        if self.ws_thread:
            self.ws_thread.join(timeout=5)
        logger.info("WebSocket stopped")
    
    def subscribe_symbols(self, symbols: Set[str]):
        """Add symbols to subscription."""
        new_symbols = symbols - self.subscribed_symbols
        if new_symbols:
            self.subscribed_symbols.update(new_symbols)
            if self.ws and self._is_ws_open():
                asyncio.create_task(self._ws_subscribe(new_symbols))
    
    def unsubscribe_symbols(self, symbols: Set[str]):
        """Remove symbols from subscription."""
        self.subscribed_symbols -= symbols
        # Alpaca doesn't support unsubscribe, need to reconnect
        if self.ws:
            asyncio.create_task(self.ws.close())
    
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
