import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TradingTools:
    """Advanced trading utilities and calculators"""

    @staticmethod
    def calculate_position_size(
            account_value: float,
            risk_per_trade_pct: float,
            entry_price: float,
            stop_loss_price: float
    ) -> Tuple[int, Dict[str, float]]:
        """
        Calculate optimal position size with risk management

        Returns:
            (shares, details)
        """
        risk_amount = account_value * (risk_per_trade_pct / 100)
        risk_per_share = abs(entry_price - stop_loss_price)

        if risk_per_share <= 0:
            return 0, {"error": "Invalid stop loss price"}

        raw_shares = risk_amount / risk_per_share
        shares = int(np.floor(raw_shares))

        return shares, {
            "risk_amount": risk_amount,
            "risk_per_share": risk_per_share,
            "position_value": shares * entry_price,
            "position_risk": shares * risk_per_share
        }

    @staticmethod
    def calculate_pyramiding_levels(
            entry_price: float,
            stop_loss: float,
            take_profit: float,
            levels: int = 3
    ) -> Dict[str, List[float]]:
        """Calculate pyramiding entry levels"""
        price_range = take_profit - entry_price
        risk_range = entry_price - stop_loss

        entries = [entry_price]
        exits = [take_profit]
        stops = [stop_loss]

        for i in range(1, levels):
            entry_level = entry_price + (price_range * i / levels)
            stop_level = stop_loss + (risk_range * i / levels)
            entries.append(entry_level)
            stops.append(stop_level)
            exits.append(take_profit)

        return {
            "entries": entries,
            "stops": stops,
            "exits": exits
        }

    @staticmethod
    def calculate_risk_reward_ratio(
            entry_price: float,
            stop_loss: float,
            take_profit: float
    ) -> Dict[str, float]:
        """Calculate risk/reward metrics"""
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)

        return {
            "risk": risk,
            "reward": reward,
            "ratio": reward / risk if risk > 0 else 0,
            "breakeven": entry_price + (entry_price - stop_loss)
        }

    @staticmethod
    def generate_trade_plan(
            symbol: str,
            direction: str,  # 'long' or 'short'
            entry_price: float,
            stop_loss: float,
            take_profit: float,
            position_size: int,
            time_frame: str = '1d'
    ) -> Dict[str, Any]:
        """Generate comprehensive trade plan"""
        if direction not in ['long', 'short']:
            raise ValueError("Direction must be 'long' or 'short'")

        is_long = direction == 'long'

        # Validate prices
        if is_long:
            if not (entry_price < take_profit and stop_loss < entry_price):
                raise ValueError("Invalid prices for long position")
        else:
            if not (entry_price > take_profit and stop_loss > entry_price):
                raise ValueError("Invalid prices for short position")

        rr_ratio = TradingTools.calculate_risk_reward_ratio(
            entry_price, stop_loss, take_profit
        )

        return {
            "symbol": symbol,
            "direction": direction,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "position_size": position_size,
            "risk_reward": rr_ratio,
            "time_frame": time_frame,
            "risk_per_share": abs(entry_price - stop_loss),
            "reward_per_share": abs(take_profit - entry_price),
            "position_risk": position_size * abs(entry_price - stop_loss),
            "position_reward": position_size * abs(take_profit - entry_price)
        }

    @staticmethod
    def calculate_slippage(
            order_size: int,
            avg_daily_volume: int,
            volatility: float
    ) -> float:
        """Estimate slippage cost"""
        participation_rate = order_size / avg_daily_volume
        slippage = 0.1 * participation_rate + 0.05 * volatility
        return max(0.01, min(0.5, slippage))  # 1-50% range

    @staticmethod
    def optimize_order_execution(
            order_size: int,
            historical_volumes: List[int],
            volatility: float
    ) -> Dict[str, Any]:
        """Optimize order execution schedule"""
        # Calculate volume profile
        avg_volume = np.mean(historical_volumes)
        volume_std = np.std(historical_volumes)

        # Time slices (market hours)
        time_slices = [
            ("09:30-10:00", 0.15),
            ("10:00-12:00", 0.25),
            ("12:00-14:00", 0.20),
            ("14:00-15:30", 0.25),
            ("15:30-16:00", 0.15)
        ]

        # Calculate participation rates
        participation_rates = []
        remaining = order_size

        for _, weight in time_slices:
            slice_size = int(order_size * weight)
            slice_size = min(slice_size, remaining)
            participation_rate = slice_size / (avg_volume * weight)
            participation_rates.append(min(0.2, participation_rate))  # Cap at 20%
            remaining -= slice_size

        # Calculate expected slippage
        slippage = TradingTools.calculate_slippage(order_size, avg_volume, volatility)

        return {
            "order_size": order_size,
            "avg_daily_volume": avg_volume,
            "time_slices": time_slices,
            "participation_rates": participation_rates,
            "estimated_slippage": slippage,
            "estimated_cost": order_size * slippage
        }

    @staticmethod
    def calculate_hedge_ratio(
            asset_prices: List[float],
            hedge_prices: List[float]
    ) -> float:
        """Calculate optimal hedge ratio"""
        if len(asset_prices) != len(hedge_prices) or len(asset_prices) < 10:
            return 0.0

        returns_asset = np.diff(asset_prices) / asset_prices[:-1]
        returns_hedge = np.diff(hedge_prices) / hedge_prices[:-1]

        covariance = np.cov(returns_asset, returns_hedge)[0, 1]
        variance_hedge = np.var(returns_hedge)

        return covariance / variance_hedge if variance_hedge > 0 else 0
