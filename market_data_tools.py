import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import aiohttp
import json
import re
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class MarketDataTools:
    """Advanced market data processing utilities"""

    @staticmethod
    async def fetch_real_time_data(symbol: str, api_url: str, api_key: str) -> Optional[Dict[str, Any]]:
        """Fetch real-time market data with retry logic"""
        headers = {"Authorization": f"Bearer {api_key}"}
        params = {"symbol": symbol}

        retries = 3
        backoff_factor = 0.5

        for attempt in range(retries):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(api_url, params=params, headers=headers, timeout=10) as resp:
                        if resp.status == 200:
                            return await resp.json()
                        logger.warning(f"API returned {resp.status} for {symbol}")
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                logger.warning(f"Attempt {attempt + 1}/{retries} failed: {str(e)}")
                await asyncio.sleep(backoff_factor * (2 ** attempt))
        return None

    @staticmethod
    def normalize_symbol(symbol: str) -> str:
        """Clean and standardize symbol formatting"""
        return re.sub(r"\s+", "", symbol.upper().replace(".", "-"))

    @staticmethod
    def calculate_returns(prices: List[float]) -> List[float]:
        """Calculate daily returns from price series"""
        if len(prices) < 2:
            return []
        returns = [0.0]  # First day has no return
        for i in range(1, len(prices)):
            returns.append((prices[i] - prices[i - 1]) / prices[i - 1])
        return returns

    @staticmethod
    def calculate_volatility(prices: List[float], window: int = 20) -> List[float]:
        """Calculate rolling volatility"""
        if len(prices) < window:
            return [0.0] * len(prices)

        returns = MarketDataTools.calculate_returns(prices)
        volatilities = []

        for i in range(len(returns)):
            if i < window:
                volatilities.append(0.0)
            else:
                period_returns = returns[i - window:i]
                volatilities.append(np.std(period_returns) * np.sqrt(252))

        return volatilities

    @staticmethod
    def detect_gaps(prices: List[Dict]) -> List[Dict]:
        """Detect price gaps in historical data"""
        gaps = []
        for i in range(1, len(prices)):
            prev_close = prices[i - 1]['close']
            current_open = prices[i]['open']
            gap_percent = (current_open - prev_close) / prev_close * 100

            if abs(gap_percent) > 2.0:  # Significant gap threshold
                gaps.append({
                    "date": prices[i]['date'],
                    "gap_percent": gap_percent,
                    "prev_close": prev_close,
                    "current_open": current_open
                })
        return gaps

    @staticmethod
    def calculate_volume_profile(ohlc_data: List[Dict], price_bins: int = 20) -> Dict:
        """Calculate volume profile for given OHLC data"""
        if not ohlc_data:
            return {}

        lows = [d['low'] for d in ohlc_data]
        highs = [d['high'] for d in ohlc_data]
        min_price = min(lows)
        max_price = max(highs)

        bin_size = (max_price - min_price) / price_bins
        volume_profile = {i: 0 for i in range(price_bins)}

        for bar in ohlc_data:
            low, high, volume = bar['low'], bar['high'], bar['volume']
            price_range = np.arange(
                start=low,
                stop=high + bin_size,
                step=bin_size
            )

            for price_level in price_range:
                bin_index = min(int((price_level - min_price) / bin_size), price_bins - 1)
                volume_profile[bin_index] += volume / len(price_range)

        return {
            "min_price": min_price,
            "max_price": max_price,
            "bin_size": bin_size,
            "volume_profile": volume_profile
        }

    @staticmethod
    def calculate_vwap(ohlc_data: List[Dict]) -> List[float]:
        """Calculate Volume Weighted Average Price"""
        vwap_values = []
        cumulative_dollar = 0.0
        cumulative_volume = 0.0

        for bar in ohlc_data:
            typical_price = (bar['high'] + bar['low'] + bar['close']) / 3
            dollar_volume = typical_price * bar['volume']

            cumulative_dollar += dollar_volume
            cumulative_volume += bar['volume']

            if cumulative_volume > 0:
                vwap_values.append(cumulative_dollar / cumulative_volume)
            else:
                vwap_values.append(typical_price)

        return vwap_values

    @staticmethod
    def detect_anomalies(prices: List[float], window: int = 20, threshold: float = 2.5) -> List[int]:
        """Detect price anomalies using standard deviation"""
        anomalies = []
        if len(prices) < window:
            return anomalies

        for i in range(window, len(prices)):
            window_prices = prices[i - window:i]
            mean = np.mean(window_prices)
            std = np.std(window_prices)

            if std == 0:
                continue

            z_score = (prices[i] - mean) / std
            if abs(z_score) > threshold:
                anomalies.append(i)

        return anomalies
