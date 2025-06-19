import asyncio
import logging
import sys
from datetime import datetime, timezone
from typing import Dict, Any, List
import yfinance as yf
import aiohttp

# Python version compatibility check
if sys.version_info < (3, 7):
    raise RuntimeError("This application requires Python 3.7 or higher")

# Import with proper error handling for concurrent.futures
try:
    from concurrent.futures import ThreadPoolExecutor
except ImportError:
    # Fallback for older Python versions
    class ThreadPoolExecutor:
        def __init__(self, max_workers=None):
            self.max_workers = max_workers

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

        def submit(self, fn, *args, **kwargs):
            return fn(*args, **kwargs)

# Import with proper error handling
try:
    from google.adk.agents import Agent
    from google.adk.tools import Tool
except ImportError:
    # Fallback implementations
    class Tool:
        def __init__(self, name=None, description=None):
            self.name = name
            self.description = description

        async def call(self, *args, **kwargs):
            raise NotImplementedError("Subclasses must implement call method")


    class Agent:
        def __init__(self, model=None, name=None, description=None, instructions=None, tools=None):
            self.model = model
            self.name = name
            self.description = description
            self.instructions = instructions
            self.tools = tools or []

logger = logging.getLogger(__name__)


class MarketDataTool(Tool):
    """Tool for fetching comprehensive market data from multiple sources"""

    def __init__(self):
        # Use explicit parent class reference for Python 2.7 compatibility
        Tool.__init__(
            self,
            name="market_data_fetcher",
            description="Fetch real-time and historical market data for stocks, ETFs, indices, and crypto"
        )
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def call(self, symbols: List[str], period: str = "1d", interval: str = "1m",
                   include_options: bool = False, include_fundamentals: bool = True) -> Dict[str, Any]:
        """
        Fetch comprehensive market data for given symbols
        Args:
            symbols: List of stock symbols to fetch
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            include_options: Whether to include options data
            include_fundamentals: Whether to include fundamental data
        Returns:
            Dictionary containing comprehensive market data
        """
        try:
            logger.info(f"Fetching market data for symbols: {symbols}")
            market_data = {}

            # Use ThreadPoolExecutor for CPU-bound yfinance operations
            with ThreadPoolExecutor(max_workers=5) as executor:
                tasks = []
                for symbol in symbols:
                    task = asyncio.get_event_loop().run_in_executor(
                        executor, self._fetch_symbol_data, symbol, period, interval,
                        include_options, include_fundamentals
                    )
                    tasks.append((symbol, task))

                # Wait for all tasks to complete
                for symbol, task in tasks:
                    try:
                        symbol_data = await task
                        market_data[symbol] = symbol_data
                    except Exception as e:
                        logger.error(f"Error fetching data for {symbol}: {str(e)}")
                        market_data[symbol] = {"error": str(e), "status": "failed"}

            # Add market summary and indices
            market_data["market_summary"] = await self._fetch_market_summary()
            market_data["sector_performance"] = await self._fetch_sector_performance()

            logger.info("Market data fetching completed successfully")
            return {
                "status": "success",
                "data": market_data,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "data_count": len(market_data)
            }

        except Exception as e:
            error_msg = f"Error in market data fetching: {str(e)}"
            logger.error(error_msg)
            return {
                "status": "error",
                "error": error_msg,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

    def _fetch_symbol_data(self, symbol: str, period: str, interval: str,
                           include_options: bool, include_fundamentals: bool) -> Dict[str, Any]:
        """Fetch data for a single symbol (runs in thread executor)"""
        try:
            ticker = yf.Ticker(symbol)

            # Fetch historical data
            hist = ticker.history(period=period, interval=interval)

            # Basic price data
            symbol_data = {
                "price_data": hist.to_dict('records') if not hist.empty else [],
                "current_price": float(hist['Close'].iloc[-1]) if not hist.empty else None,
                "volume": int(hist['Volume'].iloc[-1]) if not hist.empty else None,
                "high_52w": None,
                "low_52w": None,
                "market_cap": None,
                "status": "success"
            }

            # Add fundamental data if requested
            if include_fundamentals:
                try:
                    info = ticker.info
                    symbol_data.update({
                        "company_info": {
                            "longName": info.get("longName", symbol),
                            "sector": info.get("sector"),
                            "industry": info.get("industry"),
                            "country": info.get("country"),
                            "website": info.get("website"),
                            "business_summary": info.get("longBusinessSummary", "")[:500]  # Truncate
                        },
                        "financial_metrics": {
                            "market_cap": info.get("marketCap"),
                            "enterprise_value": info.get("enterpriseValue"),
                            "pe_ratio": info.get("trailingPE"),
                            "forward_pe": info.get("forwardPE"),
                            "peg_ratio": info.get("pegRatio"),
                            "price_to_book": info.get("priceToBook"),
                            "price_to_sales": info.get("priceToSalesTrailing12Months"),
                            "dividend_yield": info.get("dividendYield"),
                            "dividend_rate": info.get("dividendRate"),
                            "beta": info.get("beta"),
                            "52_week_high": info.get("fiftyTwoWeekHigh"),
                            "52_week_low": info.get("fiftyTwoWeekLow"),
                            "50_day_average": info.get("fiftyDayAverage"),
                            "200_day_average": info.get("twoHundredDayAverage")
                        },
                        "trading_metrics": {
                            "volume": info.get("volume"),
                            "avg_volume": info.get("averageVolume"),
                            "avg_volume_10d": info.get("averageVolume10days"),
                            "bid": info.get("bid"),
                            "ask": info.get("ask"),
                            "bid_size": info.get("bidSize"),
                            "ask_size": info.get("askSize")
                        }
                    })

                    # Update 52-week high/low
                    symbol_data["high_52w"] = info.get("fiftyTwoWeekHigh")
                    symbol_data["low_52w"] = info.get("fiftyTwoWeekLow")
                    symbol_data["market_cap"] = info.get("marketCap")
                except Exception as e:
                    logger.warning(f"Could not fetch fundamental data for {symbol}: {str(e)}")
                    symbol_data["fundamental_error"] = str(e)

            # Add options data if requested
            if include_options:
                try:
                    options_dates = ticker.options
                    if options_dates:
                        # Get options for nearest expiry
                        nearest_expiry = options_dates[0]
                        option_chain = ticker.option_chain(nearest_expiry)

                        # Convert to simple dict structure for type compatibility
                        calls_list = option_chain.calls.to_dict('records')[:10]  # Limit to 10
                        puts_list = option_chain.puts.to_dict('records')[:10]  # Limit to 10

                        symbol_data["options"] = {
                            "expiry_dates": list(options_dates),
                            "nearest_expiry": nearest_expiry,
                            "calls": calls_list,
                            "puts": puts_list
                        }
                except Exception as e:
                    logger.warning(f"Could not fetch options data for {symbol}: {str(e)}")
                    symbol_data["options_error"] = str(e)

            return symbol_data

        except Exception as e:
            logger.error(f"Error fetching data for symbol {symbol}: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }

    async def _fetch_market_summary(self) -> Dict[str, Any]:
        """Fetch overall market summary and indices"""
        try:
            indices = ['^GSPC', '^DJI', '^IXIC', '^RUT', '^VIX']  # S&P 500, Dow, NASDAQ, Russell 2000, VIX
            summary = {}

            for index in indices:
                try:
                    ticker = yf.Ticker(index)
                    hist = ticker.history(period="1d")

                    if not hist.empty:
                        current = float(hist['Close'].iloc[-1])
                        prev_close = float(hist['Open'].iloc[0])
                        change = current - prev_close
                        change_pct = (change / prev_close) * 100

                        summary[index] = {
                            "current": current,
                            "change": change,
                            "change_percent": change_pct,
                            "volume": int(hist['Volume'].iloc[-1]) if 'Volume' in hist.columns else None
                        }
                except Exception as e:
                    logger.warning(f"Could not fetch data for index {index}: {str(e)}")
                    summary[index] = {"error": str(e)}

            return summary

        except Exception as e:
            logger.error(f"Error fetching market summary: {str(e)}")
            return {"error": str(e)}

    async def _fetch_sector_performance(self) -> Dict[str, Any]:
        """Fetch sector performance data"""
        try:
            # Sector ETFs to track performance
            sector_etfs = {
                'Technology': 'XLK',
                'Healthcare': 'XLV',
                'Financial': 'XLF',
                'Consumer Discretionary': 'XLY',
                'Communication Services': 'XLC',
                'Industrial': 'XLI',
                'Consumer Staples': 'XLP',
                'Energy': 'XLE',
                'Utilities': 'XLU',
                'Real Estate': 'XLRE',
                'Materials': 'XLB'
            }

            sector_performance = {}

            for sector, etf in sector_etfs.items():
                try:
                    ticker = yf.Ticker(etf)
                    hist = ticker.history(period="1d")

                    if not hist.empty:
                        current = float(hist['Close'].iloc[-1])
                        prev_close = float(hist['Open'].iloc[0])
                        change_pct = ((current - prev_close) / prev_close) * 100

                        sector_performance[sector] = {
                            "etf_symbol": etf,
                            "current_price": current,
                            "change_percent": change_pct,
                            "volume": int(hist['Volume'].iloc[-1])
                        }
                except Exception as e:
                    logger.warning(f"Could not fetch sector data for {sector}: {str(e)}")

            return sector_performance

        except Exception as e:
            logger.error(f"Error fetching sector performance: {str(e)}")
            return {"error": str(e)}


class MarketDataAgent(Agent):
    """
    Specialized agent for collecting comprehensive market data.
    Part of the parallel data collection workflow.
    """

    def __init__(self, gcp_services, config):
        # Use explicit parent class reference for Python 2.7 compatibility
        Agent.__init__(
            self,
            model="gemini-2.0-flash-exp",
            name="market_data_collector",
            description="Expert market data collector specializing in multi-source data aggregation",
            instructions="""You are a specialized market data collection agent with expertise in:
1. Real-time and historical price data collection
2. Multi-asset class data (stocks, ETFs, indices, options)
3. Fundamental and technical data aggregation
4. Market summary and sector analysis
5. Data quality validation and error handling

Your primary responsibilities:
- Collect comprehensive market data from multiple sources
- Ensure data accuracy and completeness
- Handle API rate limits and errors gracefully
- Provide structured, normalized data output
- Monitor data freshness and market hours

Always prioritize data quality and provide detailed metadata about your data sources.""",
            tools=[MarketDataTool()]
        )

        self.gcp_services = gcp_services
        self.config = config
        logger.info("MarketDataAgent initialized")

    async def collect_market_data(self, symbols: List[str], **kwargs) -> Dict[str, Any]:
        """
        Main method to collect comprehensive market data
        Args:
            symbols: List of symbols to collect data for
            **kwargs: Additional parameters for data collection
        Returns:
            Comprehensive market data dictionary
        """
        try:
            logger.info(f"Starting market data collection for {len(symbols)} symbols")

            # Use the market data tool
            async with MarketDataTool() as tool:
                market_data = await tool.call(
                    symbols=symbols,
                    period=kwargs.get('period', '1d'),
                    interval=kwargs.get('interval', '1m'),
                    include_options=kwargs.get('include_options', False),
                    include_fundamentals=kwargs.get('include_fundamentals', True)
                )

            # Store data in Firestore for real-time access
            if hasattr(self.gcp_services, 'store_market_data'):
                await self.gcp_services.store_market_data(market_data)

            # Store in BigQuery for analytics
            if hasattr(self.gcp_services, 'store_bigquery_market_data'):
                await self.gcp_services.store_bigquery_market_data(market_data)

            # Add collection metadata
            market_data["collection_metadata"] = {
                "agent": self.name,
                "collection_time": datetime.now(timezone.utc).isoformat(),
                "symbols_requested": symbols,
                "symbols_collected": len([k for k, v in market_data.get('data', {}).items()
                                          if isinstance(v, dict) and v.get('status') != 'error']),
                "data_quality_score": self._calculate_data_quality_score(market_data)
            }

            logger.info(
                f"Market data collection completed. Quality score: {market_data['collection_metadata']['data_quality_score']}")
            return market_data

        except Exception as e:
            error_msg = f"Error in market data collection: {str(e)}"
            logger.error(error_msg)
            # Return error response with partial data if available
            return {
                "status": "error",
                "error": error_msg,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "agent": self.name
            }

    @staticmethod
    def _calculate_data_quality_score(market_data: Dict[str, Any]) -> float:
        """Calculate data quality score based on completeness and accuracy"""
        if market_data.get('status') != 'success':
            return 0.0

        data = market_data.get('data', {})
        if not data:
            return 0.0

        total_symbols = len([k for k in data.keys() if not k.startswith('market_') and not k.startswith('sector_')])
        successful_symbols = len([k for k, v in data.items()
                                  if isinstance(v, dict) and v.get('status') != 'error'
                                  and not k.startswith('market_') and not k.startswith('sector_')])

        if total_symbols == 0:
            return 0.0

        base_score = (successful_symbols / total_symbols) * 100

        # Bonus for additional data availability
        bonus = 0
        if 'market_summary' in data and (
                not isinstance(data['market_summary'], dict) or not data['market_summary'].get('error')):
            bonus += 5

        if 'sector_performance' in data and (
                not isinstance(data['sector_performance'], dict) or not data['sector_performance'].get('error')):
            bonus += 5

        return min(100.0, base_score + bonus)

    async def get_real_time_quotes(self, symbols: List[str]) -> Dict[str, Any]:
        """Get real-time quotes for given symbols"""
        return await self.collect_market_data(
            symbols=symbols,
            period="1d",
            interval="1m",
            include_options=False,
            include_fundamentals=False
        )

    async def get_historical_data(self, symbols: List[str], period: str = "1y") -> Dict[str, Any]:
        """Get historical data for given symbols"""
        return await self.collect_market_data(
            symbols=symbols,
            period=period,
            interval="1d",
            include_options=False,
            include_fundamentals=True
        )

    async def get_options_data(self, symbols: List[str]) -> Dict[str, Any]:
        """Get options data for given symbols"""
        return await self.collect_market_data(
            symbols=symbols,
            period="1d",
            interval="1d",
            include_options=True,
            include_fundamentals=False
        )
