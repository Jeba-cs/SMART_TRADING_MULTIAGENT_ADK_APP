import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import aiohttp
import json
import os
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

logger = logging.getLogger(__name__)


class AlpacaService:
    """
    Alpaca service for trading and market data
    """

    def __init__(self, config):
        self.config = config
        self.api_key = config.ALPACA_API_KEY
        self.secret_key = config.ALPACA_SECRET_KEY
        self.base_url = config.ALPACA_BASE_URL

        try:
            # Initialize trading client
            self.trading_client = TradingClient(
                api_key=self.api_key,
                secret_key=self.secret_key,
                paper=True  # Use paper trading
            )

            # Initialize data client
            self.data_client = StockHistoricalDataClient(
                api_key=self.api_key,
                secret_key=self.secret_key
            )

            logger.info("Alpaca service initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Alpaca service: {str(e)}")
            self.trading_client = None
            self.data_client = None

    async def get_account_info(self) -> Dict[str, Any]:
        """Get account information"""
        try:
            if not self.trading_client:
                return {"error": "Trading client not initialized"}

            account = self.trading_client.get_account()

            return {
                "account_id": account.id,
                "equity": float(account.equity),
                "cash": float(account.cash),
                "buying_power": float(account.buying_power),
                "portfolio_value": float(account.portfolio_value),
                "day_trade_buying_power": float(account.day_trade_buying_power),
                "initial_margin": float(account.initial_margin),
                "maintenance_margin": float(account.maintenance_margin),
                "sma": float(account.sma),
                "status": account.status.value,
                "trading_blocked": account.trading_blocked,
                "transfers_blocked": account.transfers_blocked,
                "account_blocked": account.account_blocked,
                "created_at": account.created_at.isoformat() if account.created_at else None
            }

        except Exception as e:
            logger.error(f"Failed to get account info: {str(e)}")
            return {"error": str(e)}

    async def get_positions(self) -> List[Dict[str, Any]]:
        """Get current positions"""
        try:
            if not self.trading_client:
                return []

            positions = self.trading_client.get_all_positions()

            position_list = []
            for position in positions:
                position_data = {
                    "symbol": position.symbol,
                    "qty": float(position.qty),
                    "side": position.side.value,
                    "market_value": float(position.market_value),
                    "cost_basis": float(position.cost_basis),
                    "unrealized_pl": float(position.unrealized_pl),
                    "unrealized_plpc": float(position.unrealized_plpc),
                    "current_price": float(position.current_price),
                    "lastday_price": float(position.lastday_price),
                    "change_today": float(position.change_today)
                }
                position_list.append(position_data)

            return position_list

        except Exception as e:
            logger.error(f"Failed to get positions: {str(e)}")
            return []

    async def get_orders(self, status: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get orders"""
        try:
            if not self.trading_client:
                return []

            from alpaca.trading.requests import GetOrdersRequest
            from alpaca.trading.enums import QueryOrderStatus

            # Convert status string to enum
            status_enum = None
            if status:
                status_mapping = {
                    'open': QueryOrderStatus.OPEN,
                    'closed': QueryOrderStatus.CLOSED,
                    'all': QueryOrderStatus.ALL
                }
                status_enum = status_mapping.get(status.lower(), QueryOrderStatus.ALL)

            request = GetOrdersRequest(
                status=status_enum,
                limit=limit
            )

            orders = self.trading_client.get_orders(filter=request)

            order_list = []
            for order in orders:
                order_data = {
                    "id": order.id,
                    "symbol": order.symbol,
                    "qty": float(order.qty),
                    "filled_qty": float(order.filled_qty),
                    "side": order.side.value,
                    "order_type": order.order_type.value,
                    "time_in_force": order.time_in_force.value,
                    "status": order.status.value,
                    "created_at": order.created_at.isoformat() if order.created_at else None,
                    "updated_at": order.updated_at.isoformat() if order.updated_at else None,
                    "submitted_at": order.submitted_at.isoformat() if order.submitted_at else None,
                    "filled_at": order.filled_at.isoformat() if order.filled_at else None,
                    "expired_at": order.expired_at.isoformat() if order.expired_at else None,
                    "canceled_at": order.canceled_at.isoformat() if order.canceled_at else None,
                    "limit_price": float(order.limit_price) if order.limit_price else None,
                    "stop_price": float(order.stop_price) if order.stop_price else None,
                    "filled_avg_price": float(order.filled_avg_price) if order.filled_avg_price else None
                }
                order_list.append(order_data)

            return order_list

        except Exception as e:
            logger.error(f"Failed to get orders: {str(e)}")
            return []

    async def place_market_order(self, symbol: str, qty: float, side: str) -> Dict[str, Any]:
        """Place market order"""
        try:
            if not self.trading_client:
                return {"error": "Trading client not initialized"}

            # Convert side string to enum
            order_side = OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL

            market_order_data = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=order_side,
                time_in_force=TimeInForce.DAY
            )

            order = self.trading_client.submit_order(order_data=market_order_data)

            return {
                "order_id": order.id,
                "symbol": order.symbol,
                "qty": float(order.qty),
                "side": order.side.value,
                "status": order.status.value,
                "created_at": order.created_at.isoformat() if order.created_at else None,
                "success": True
            }

        except Exception as e:
            logger.error(f"Failed to place market order for {symbol}: {str(e)}")
            return {"error": str(e), "success": False}

    async def place_limit_order(self, symbol: str, qty: float, side: str, limit_price: float) -> Dict[str, Any]:
        """Place limit order"""
        try:
            if not self.trading_client:
                return {"error": "Trading client not initialized"}

            # Convert side string to enum
            order_side = OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL

            limit_order_data = LimitOrderRequest(
                symbol=symbol,
                qty=qty,
                side=order_side,
                time_in_force=TimeInForce.DAY,
                limit_price=limit_price
            )

            order = self.trading_client.submit_order(order_data=limit_order_data)

            return {
                "order_id": order.id,
                "symbol": order.symbol,
                "qty": float(order.qty),
                "side": order.side.value,
                "limit_price": float(order.limit_price),
                "status": order.status.value,
                "created_at": order.created_at.isoformat() if order.created_at else None,
                "success": True
            }

        except Exception as e:
            logger.error(f"Failed to place limit order for {symbol}: {str(e)}")
            return {"error": str(e), "success": False}

    async def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel order"""
        try:
            if not self.trading_client:
                return {"error": "Trading client not initialized"}

            self.trading_client.cancel_order_by_id(order_id)

            return {"order_id": order_id, "success": True, "message": "Order cancelled"}

        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {str(e)}")
            return {"error": str(e), "success": False}

    async def get_historical_bars(self, symbols: List[str], timeframe: str = "1Day",
                                  start: datetime = None, end: datetime = None) -> Dict[str, Any]:
        """Get historical price bars"""
        try:
            if not self.data_client:
                return {"error": "Data client not initialized"}

            # Set default time range if not provided
            if not end:
                end = datetime.now()
            if not start:
                start = end - timedelta(days=30)

            # Convert timeframe string to TimeFrame enum
            timeframe_mapping = {
                "1Min": TimeFrame.Minute,
                "5Min": TimeFrame(5, "Min"),
                "15Min": TimeFrame(15, "Min"),
                "30Min": TimeFrame(30, "Min"),
                "1Hour": TimeFrame.Hour,
                "1Day": TimeFrame.Day,
                "1Week": TimeFrame.Week,
                "1Month": TimeFrame.Month
            }

            tf = timeframe_mapping.get(timeframe, TimeFrame.Day)

            request_params = StockBarsRequest(
                symbol_or_symbols=symbols,
                timeframe=tf,
                start=start,
                end=end
            )

            bars = self.data_client.get_stock_bars(request_params)

            # Convert to dictionary format
            result = {}
            for symbol in symbols:
                if symbol in bars.data:
                    symbol_bars = []
                    for bar in bars.data[symbol]:
                        bar_data = {
                            "timestamp": bar.timestamp.isoformat(),
                            "open": float(bar.open),
                            "high": float(bar.high),
                            "low": float(bar.low),
                            "close": float(bar.close),
                            "volume": int(bar.volume),
                            "trade_count": int(bar.trade_count) if hasattr(bar, 'trade_count') else None,
                            "vwap": float(bar.vwap) if hasattr(bar, 'vwap') else None
                        }
                        symbol_bars.append(bar_data)

                    result[symbol] = {
                        "bars": symbol_bars,
                        "count": len(symbol_bars)
                    }
                else:
                    result[symbol] = {"error": "No data available"}

            return {
                "data": result,
                "timeframe": timeframe,
                "start": start.isoformat(),
                "end": end.isoformat(),
                "symbols": symbols
            }

        except Exception as e:
            logger.error(f"Failed to get historical bars: {str(e)}")
            return {"error": str(e)}

    async def get_latest_quotes(self, symbols: List[str]) -> Dict[str, Any]:
        """Get latest quotes"""
        try:
            if not self.data_client:
                return {"error": "Data client not initialized"}

            request_params = StockLatestQuoteRequest(symbol_or_symbols=symbols)
            quotes = self.data_client.get_stock_latest_quote(request_params)

            result = {}
            for symbol in symbols:
                if symbol in quotes:
                    quote = quotes[symbol]
                    result[symbol] = {
                        "timestamp": quote.timestamp.isoformat(),
                        "bid_price": float(quote.bid_price),
                        "bid_size": int(quote.bid_size),
                        "ask_price": float(quote.ask_price),
                        "ask_size": int(quote.ask_size),
                        "spread": float(quote.ask_price - quote.bid_price),
                        "spread_pct": float(
                            (quote.ask_price - quote.bid_price) / quote.bid_price * 100) if quote.bid_price > 0 else 0
                    }
                else:
                    result[symbol] = {"error": "No quote available"}

            return {"data": result, "symbols": symbols}

        except Exception as e:
            logger.error(f"Failed to get latest quotes: {str(e)}")
            return {"error": str(e)}

    async def get_portfolio_history(self, period: str = "1M", timeframe: str = "1D") -> Dict[str, Any]:
        """Get portfolio history"""
        try:
            if not self.trading_client:
                return {"error": "Trading client not initialized"}

            from alpaca.trading.requests import GetPortfolioHistoryRequest

            request = GetPortfolioHistoryRequest(
                period=period,
                timeframe=timeframe
            )

            history = self.trading_client.get_portfolio_history(filter=request)

            # Convert to list format
            history_data = []
            if history.timestamp and history.equity:
                for i, timestamp in enumerate(history.timestamp):
                    if i < len(history.equity):
                        history_data.append({
                            "timestamp": timestamp.isoformat(),
                            "equity": float(history.equity[i]),
                            "profit_loss": float(history.profit_loss[i]) if history.profit_loss and i < len(
                                history.profit_loss) else None,
                            "profit_loss_pct": float(history.profit_loss_pct[i]) if history.profit_loss_pct and i < len(
                                history.profit_loss_pct) else None
                        })

            return {
                "history": history_data,
                "period": period,
                "timeframe": timeframe,
                "base_value": float(history.base_value) if history.base_value else None
            }

        except Exception as e:
            logger.error(f"Failed to get portfolio history: {str(e)}")
            return {"error": str(e)}

    async def get_market_calendar(self, start: datetime = None, end: datetime = None) -> List[Dict[str, Any]]:
        """Get market calendar"""
        try:
            if not self.trading_client:
                return []

            # Set default date range if not provided
            if not start:
                start = datetime.now()
            if not end:
                end = start + timedelta(days=30)

            calendar = self.trading_client.get_calendar(start=start.date(), end=end.date())

            calendar_data = []
            for day in calendar:
                calendar_data.append({
                    "date": day.date.isoformat(),
                    "open": day.open.isoformat() if day.open else None,
                    "close": day.close.isoformat() if day.close else None,
                    "session_open": day.session_open.isoformat() if hasattr(day,
                                                                            'session_open') and day.session_open else None,
                    "session_close": day.session_close.isoformat() if hasattr(day,
                                                                              'session_close') and day.session_close else None
                })

            return calendar_data

        except Exception as e:
            logger.error(f"Failed to get market calendar: {str(e)}")
            return []

    async def get_clock(self) -> Dict[str, Any]:
        """Get market clock"""
        try:
            if not self.trading_client:
                return {"error": "Trading client not initialized"}

            clock = self.trading_client.get_clock()

            return {
                "timestamp": clock.timestamp.isoformat(),
                "is_open": clock.is_open,
                "next_open": clock.next_open.isoformat() if clock.next_open else None,
                "next_close": clock.next_close.isoformat() if clock.next_close else None
            }

        except Exception as e:
            logger.error(f"Failed to get market clock: {str(e)}")
            return {"error": str(e)}

    async def execute_trades(self, trade_plan: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute multiple trades from trade plan"""
        try:
            if not self.trading_client:
                return [{"error": "Trading client not initialized"}]

            execution_results = []

            for trade in trade_plan:
                symbol = trade.get('symbol')
                qty = trade.get('qty', trade.get('quantity'))
                side = trade.get('side')
                order_type = trade.get('order_type', trade.get('type', 'market'))
                limit_price = trade.get('limit_price')

                if not all([symbol, qty, side]):
                    execution_results.append({
                        "symbol": symbol,
                        "error": "Missing required trade parameters",
                        "success": False
                    })
                    continue

                try:
                    if order_type.lower() == 'market':
                        result = await self.place_market_order(symbol, qty, side)
                    elif order_type.lower() == 'limit' and limit_price:
                        result = await self.place_limit_order(symbol, qty, side, limit_price)
                    else:
                        result = {
                            "symbol": symbol,
                            "error": f"Unsupported order type: {order_type}",
                            "success": False
                        }

                    execution_results.append(result)

                    # Add small delay between orders
                    await asyncio.sleep(0.1)

                except Exception as trade_error:
                    execution_results.append({
                        "symbol": symbol,
                        "error": str(trade_error),
                        "success": False
                    })

            return execution_results

        except Exception as e:
            logger.error(f"Failed to execute trades: {str(e)}")
            return [{"error": str(e)}]

    async def health_check(self) -> bool:
        """Perform health check on Alpaca service"""
        try:
            if not self.trading_client:
                return False

            # Try to get account info
            account_info = await self.get_account_info()
            return not account_info.get('error')

        except Exception as e:
            logger.error(f"Alpaca health check failed: {str(e)}")
            return False

    async def close(self):
        """Close Alpaca service connections"""
        try:
            # Alpaca clients don't need explicit closing
            logger.info("Alpaca service closed")
        except Exception as e:
            logger.error(f"Error closing Alpaca service: {str(e)}")
