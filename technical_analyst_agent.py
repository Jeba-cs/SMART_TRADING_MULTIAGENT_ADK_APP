import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
import pandas_ta as ta

# Fallback for Tool and Agent if ADK not available
try:
    from google.adk.agents import Agent
    from google.adk.tools import Tool
except ImportError:
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


class AdvancedTechnicalAnalysisTool(Tool):
    """Advanced technical analysis tool with comprehensive indicators and pattern recognition"""

    def __init__(self):
        super().__init__(
            name="advanced_technical_analyzer",
            description="Perform comprehensive technical analysis with advanced indicators and pattern recognition"
        )

    async def call(self, market_data: Dict[str, Any], analysis_type: str = "comprehensive") -> Dict[str, Any]:
        """
        Perform advanced technical analysis on market data
        Args:
            market_data: Market data from data collection agents
            analysis_type: Type of analysis (comprehensive, quick, patterns_only)
        Returns:
            Comprehensive technical analysis results
        """
        try:
            logger.info("Starting advanced technical analysis")
            if market_data.get('status') != 'success':
                return {
                    "status": "error",
                    "error": "Invalid market data provided",
                    "timestamp": datetime.utcnow().isoformat()
                }

            data = market_data.get('data', {})
            analysis_results = {}

            # Analyze each symbol
            for symbol, symbol_data in data.items():
                if symbol.startswith('market_') or symbol.startswith('sector_'):
                    continue
                if isinstance(symbol_data, dict) and symbol_data.get('price_data'):
                    logger.info(f"Analyzing {symbol}")
                    symbol_analysis = await self._analyze_symbol(symbol, symbol_data, analysis_type)
                    analysis_results[symbol] = symbol_analysis

            # Market-wide technical analysis
            market_analysis = await self._analyze_market_conditions(data)

            # Sector rotation analysis
            sector_analysis = await self._analyze_sector_rotation(data)

            return {
                "status": "success",
                "timestamp": datetime.utcnow().isoformat(),
                "analysis_type": analysis_type,
                "symbol_analysis": analysis_results,
                "market_analysis": market_analysis,
                "sector_analysis": sector_analysis,
                "summary": self._generate_analysis_summary(analysis_results, market_analysis)
            }

        except Exception as e:
            error_msg = f"Error in technical analysis: {str(e)}"
            logger.error(error_msg)
            return {
                "status": "error",
                "error": error_msg,
                "timestamp": datetime.utcnow().isoformat()
            }

    async def _analyze_symbol(self, symbol: str, symbol_data: Dict[str, Any], analysis_type: str) -> Dict[str, Any]:
        """Perform comprehensive technical analysis for a single symbol"""
        try:
            price_data = symbol_data.get('price_data', [])
            if not price_data:
                return {"error": "No price data available"}

            # Convert to pandas DataFrame
            df = pd.DataFrame(price_data)
            if df.empty or len(df) < 20:  # Need minimum data points
                return {"error": "Insufficient price data"}

            # Ensure datetime index
            if 'Datetime' in df.columns:
                df['Datetime'] = pd.to_datetime(df['Datetime'])
                df.set_index('Datetime', inplace=True)

            # Extract OHLCV data
            close = df['Close'].values.astype(float)
            high = df['High'].values.astype(float)
            low = df['Low'].values.astype(float)
            open_prices = df['Open'].values.astype(float)
            volume = df['Volume'].values.astype(float) if 'Volume' in df.columns else np.ones(len(close))

            analysis_result = {
                "symbol": symbol,
                "current_price": float(close[-1]),
                "price_change": float(close[-1] - close[-2]) if len(close) > 1 else 0,
                "price_change_pct": ((close[-1] - close[-2]) / close[-2] * 100) if len(close) > 1 and close[
                    -2] != 0 else 0
            }

            # Trend Analysis
            trend_analysis = self._analyze_trends(close, high, low)
            analysis_result["trend_analysis"] = trend_analysis

            # Momentum Analysis
            momentum_analysis = self._analyze_momentum(close, high, low, volume)
            analysis_result["momentum_analysis"] = momentum_analysis

            # Volatility Analysis
            volatility_analysis = self._analyze_volatility(close, high, low)
            analysis_result["volatility_analysis"] = volatility_analysis

            # Support and Resistance
            support_resistance = self._identify_support_resistance(close, high, low)
            analysis_result["support_resistance"] = support_resistance

            # Volume Analysis
            volume_analysis = self._analyze_volume(close, volume)
            analysis_result["volume_analysis"] = volume_analysis

            if analysis_type == "comprehensive":
                # Pattern Recognition
                patterns = self._identify_patterns(open_prices, high, low, close)
                analysis_result["patterns"] = patterns

                # Advanced Indicators
                advanced_indicators = self._calculate_advanced_indicators(open_prices, high, low, close, volume)
                analysis_result["advanced_indicators"] = advanced_indicators

                # Market Structure Analysis
                market_structure = self._analyze_market_structure(close, high, low)
                analysis_result["market_structure"] = market_structure

                # Generate trading signals
                signals = self._generate_trading_signals(analysis_result)
                analysis_result["signals"] = signals

                # Overall rating
                overall_rating = self._calculate_overall_rating(analysis_result)
                analysis_result["overall_rating"] = overall_rating

            return analysis_result

        except Exception as e:
            return {"error": f"Error analyzing {symbol}: {str(e)}"}

    def _analyze_trends(self, close: np.ndarray, high: np.ndarray, low: np.ndarray) -> Dict[str, Any]:
        """Comprehensive trend analysis"""
        try:
            # Moving averages using pandas_ta
            close_series = pd.S极eries(close)
            sma_20 = ta.sma(close_series, length=20).values
            sma_50 = ta.sma(close_series, length=50).values
            sma_200 = ta.sma(close_series, length=200).values
            ema_12 = ta.ema(close_series, length=12).values
            ema_26 = ta.ema(close_series, length=26).values

            # Current values
            current_price = close[-1]
            current_sma_20 = sma_20[-1] if not np.isnan(sma_20[-1]) else None
            current_sma_50 = sma_50[-1] if not np.isnan(sma_50[-1]) else None
            current_sma_200 = sma_200[-1] if not np.isnan(sma_200[-1]) else None

            # Trend determination
            short_term_trend = self._determine_trend(current_price, current_sma_20, current_sma_50)
            long_term_trend = self._determine_trend(current_price, current_sma_50, current_sma_200)

            # Trend strength
            trend_strength = self._calculate_trend_strength(close, sma_20, sma_50)

            # ADX for trend strength
            high_series = pd.Series(high)
            low_series = pd.Series(low)
            adx_df = ta.adx(high_series, low_series, close_series, length=14)
            adx = adx_df['ADX_14'].values if 'ADX_14' in adx_df else np.array([np.nan])
            current_adx = adx[-1] if not np.isnan(adx[-1]) else None

            return {
                "short_term_trend": short_term_trend,
                "long_term_trend": long_term_trend,
                "tre极nd_strength": trend_strength,
                "adx": current_adx,
                "adx_interpretation": self._interpret_adx(current_adx),
                "moving_averages": {
                    "sma_20": current_sma_20,
                    "sma_50": current_sma_50,
                    "sma_200": current_sma_200,
                    "ema_12": ema_12[-1] if not np.isnan(ema_12[-1]) else None,
                    "ema_26": ema_26[-1] if not np.isnan(ema_26[-1]) else None
                },
                "golden_cross": current_sma_50 > current_sma_200 if (current_sma_50 and current_sma_200) else None,
                "death_cross": current_sma_50 < current_sma_200 if (current_sma_50 and current_sma_200) else None
            }

        except Exception as e:
            return {"error": f"Error in trend analysis: {str(e)}"}

    def _analyze_momentum(self, close: np.ndarray, high: np.ndarray, low: np.ndarray, volume: np.ndarray) -> Dict[
        str, Any]:
        """Comprehensive momentum analysis"""
        try:
            # RSI
            close_series = pd.Series(close)
            rsi = ta.rsi(close_series, length=14).values
            current_rsi = rsi[-1] if not np.isnan(rsi[-1]) else None

            # MACD
            macd_df = ta.macd(close_series, fast=12, slow=26, signal=9)
            current_macd = macd_df['MACD_12_26_9'].iloc[-1] if not macd_df.empty else None
            current_macd_signal = macd_df['MACDs_12_26_9'].iloc[-1] if not macd_df.empty else None
            current_macd_hist = macd_df['MACDh_12_26_9'].iloc[-1] if not macd_df.empty else None

            # Stochastic
            high_series = pd.Series(high)
            low_series = pd.Series(low)
            stoch_df = ta.stoch(high_series, low_series, close_series, k=14, d=3)
            current_stoch_k = stoch_df['STOCHk_14_3'].iloc[-1] if not stoch_df.empty else None
            current_stoch_d = stoch_df['STOCHd_14_3'].iloc[-1] if not stoch_df.empty else None

            # Williams %R
            willr = ta.willr(high_series, low_series, close_series, length=14).values
            current_willr = willr[-1] if not np.isnan(willr[-1]) else None

            # CCI (Commodity Channel Index)
            cci = ta.cci(high_series, low_series, close_series, length=20).values
            current_cci = cci[-1] if not np.isnan(cci[-1]) else None

            # Rate of Change
            roc = ta.roc(close_series, length=10).values
            current_roc = roc[-1] if not np.isnan(roc[-1]) else None

            return {
                "rsi": {
                    "value": current_rsi,
                    "interpretation": self._interpret_rsi(current_rsi),
                    "overbought": current_rsi > 70 if current_rsi else None,
                    "oversold": current_rsi < 30 if current_rsi else None
                },
                "macd": {
                    "macd": current_macd,
                    "signal": current_macd_signal,
                    "histogram": current_macd_hist,
                    "bullish_crossover": (current_macd > current_macd_signal) if (
                                current_macd and current_macd_signal) else None,
                    "interpretation": self._interpret_macd(current_macd, current_macd_signal, current_macd_hist)
                },
                "stochastic": {
                    "k": current_stoch_k,
                    "d": current_stoch_d,
                    "overbought": current_stoch_k > 80 if current_stoch_k else None,
                    "oversold": current_stoch_k < 20 if current_stoch_k else None
                },
                "williams_r": {
                    "value": current_willr,
                    "overbought": current_willr > -20 if current_willr else None,
                    "oversold": current_willr < -80 if current_willr else None
                },
                "cci": {
                    "value": current_cci,
                    "overbought": current_cci > 100 if current_cci else None,
                    "oversold": current_cci < -100 if current_cci else None
                },
                "rate_of_change": current_roc,
                "momentum_score": self._calculate_momentum_score(current_rsi, current_macd, current_macd_signal,
                                                                 current_stoch_k)
            }

        except Exception as e:
            return {"error": f"Error in momentum analysis: {str(e)}"}

    def _analyze_volatility(self, close: np.ndarray, high: np.ndarray, low: np.ndarray) -> Dict[str, Any]:
        """Comprehensive volatility analysis"""
        try:
            close_series = pd.Series(close)
            high_series = pd.Series(high)
            low_series = pd.Series(low)

            # Bollinger Bands
            bb_df = ta.bbands(close_series, length=20, std=2)
            current_bb_upper = bb_df['BBU_20_2.0'].iloc[-1] if not bb_df.empty else None
            current_bb_middle = bb_df['BBM_20_2.0'].iloc[-1] if not bb_df.empty else None
            current_bb_lower = bb_df['BBL_20_2.0'].iloc[-1] if not bb_df.empty else None

            # ATR (Average True Range)
            atr = ta.atr(high_series, low_series, close_series, length=14).values
            current_atr = atr[-1] if not np.isnan(atr[-1]) else None

            # Historical volatility
            returns = np.diff(np.log(close))
            historical_vol = np.std(returns) * np.sqrt(252) * 100  # Annualized volatility

            # Bollinger Band position
            current_price = close[-1]
            if current_bb_upper and current_bb_lower:
                bb_position = (current_price - current_bb_lower) / (current_bb_upper - current_bb_lower)
            else:
                bb_position = None

            return {
                "bollinger_bands": {
                    "upper": current_bb_upper,
                    "middle": current_bb_middle,
                    "lower": current_bb_lower,
                    "position": bb_position,
                    "squeeze": self._detect_bb_squeeze(bb_df['BBU_20_2.0'].values, bb_df['BBL_20_2.0'].values, atr)
                },
                "atr": {
                    "value": current_atr,
                    "interpretation": self._interpret_atr(current_atr, current_price)
                },
                "historical_volatility": {
                    "annualized_pct": historical_vol,
                    "interpretation": self._interpret_volatility(historical_vol)
                },
                "volatility_regime": self._determine_volatility_regime(historical_vol),
                "volatility_trend": self._analyze_volatility_trend(atr)
            }

        except Exception as e:
            return {"error": f"Error in volatility analysis: {str(e)}"}

    def _identify_support_resistance(self, close: np.ndarray, high: np.ndarray, low: np.ndarray) -> Dict[str, Any]:
        """Identify key support and resistance levels"""
        try:
            current_price = close[-1]

            # Pivot points (traditional)
            pivot_points = self._calculate_pivot_points(high[-1], low[-1], close[-1])

            # Support and resistance from recent highs/lows
            recent_data_points = min(50, len(close))
            recent_highs = high[-recent_data_points:]
            recent_lows = low[-recent_data_points:]

            # Find significant levels
            resistance_levels = self._find_resistance_levels(recent_highs, current_price)
            support_levels = self._find_support_levels(recent_lows, current_price)

            # Fibonacci retracements (if we have a clear trend)
            fib_levels = self._calculate_fibonacci_levels(high, low, close)

            return {
                "current_price": current_price,
                "pivot_points": pivot_points,
                "resistance_levels": resistance_levels,
                "support_levels": support_levels,
                "fibonacci_levels": fib_levels,
                "nearest_support": min(support_levels,
                                       key=lambda x: abs(x - current_price)) if support_levels else None,
                "nearest_resistance": min(resistance_levels,
                                          key=lambda x: abs(x - current_price)) if resistance_levels else None
            }

        except Exception as e:
            return {"error": f"Error identifying support/resistance: {str(e)}"}

    def _analyze_volume(self, close: np.ndarray, volume: np.ndarray) -> Dict[str, Any]:
        """Comprehensive volume analysis"""
        try:
            close_series = pd.Series(close)
            volume_series = pd.Series(volume)

            # Volume moving averages
            vol_sma_20 = ta.sma(volume_series, length=20).values
            current_vol_sma = vol_sma_20[-1] if not np.isnan(vol_sma_20[-1]) else None
            current_volume = volume[-1]

            # On-Balance Volume
            obv = ta.obv(close_series, volume_series).values
            current_obv = obv[-1] if not np.isnan(obv[-1]) else None

            # Accumulation/Distribution Line
            high_series = pd.Series(np.ones_like(close))  # Simplified for AD calculation
            low_series = pd.Series(np.ones_like(close))  # Simplified for AD calculation
            ad_line = ta.ad(high_series, low_series, close_series, volume_series).values
            current_ad = ad_line[-1] if not np.isnan(ad_line[-1]) else None

            # Volume Rate of Change
            vol_roc = ta.roc(volume_series, length=10).values
            current_vol_roc = vol_roc[-1] if not np.isnan(vol_roc[-1]) else None

            # Volume analysis
            volume_trend = self._analyze_volume_trend(obv)
            volume_confirmation = self._check_volume_confirmation(close, volume)

            return {
                "current_volume": current_volume,
                "volume_sma_20": current_vol_sma,
                "volume_vs_average": (current_volume / current_vol_sma) if current_vol_sma else None,
                "obv": {
                    "value": current_obv,
                    "trend": volume_trend
                },
                "accumulation_distribution": current_ad,
                "volume_roc": current_vol_roc,
                "volume_confirmation": volume_confirmation,
                "volume_interpretation": self._interpret_volume(current_volume, current_vol_sma, volume_trend)
            }

        except Exception as e:
            return {"error": f"Error in volume analysis: {str(e)}"}

    def _identify_patterns(self, open_prices: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Dict[
        str, Any]:
        """Identify chart patterns using pandas_ta pattern recognition"""
        try:
            patterns = {}

            # Create DataFrame for pattern detection
            df = pd.DataFrame({
                'open': open_prices,
                'high': high,
                'low': low,
                'close': close
            })

            # Candlestick patterns - using pandas_ta pattern functions
            candlestick_patterns = {}

            # Map of pattern functions to names
            pattern_functions = {
                'doji': ta.cdl_doji,
                'hammer': ta.cdl_hammer,
                'hanging_man': ta.cdl_hanging_man,
                'shooting_star': ta.cdl_shooting_star,
                'engulfing': ta.cdl_engulfing,
                'harami': ta.cdl_harami,
                'morning_star': ta.cdl_morning_star,
                'evening_star': ta.cdl_evening_star
            }

            # Check for patterns
            for pattern_name, pattern_func in pattern_functions.items():
                try:
                    pattern_result = pattern_func(df['open'], df['high'], df['low'], df['close'])
                    if pattern_result is not None:
                        candlestick_patterns[pattern_name] = pattern_result
                except:
                    # Skip if pattern function not available
                    pass

            # Check for recent patterns (last 5 periods)
            recent_patterns = []
            for pattern_name, pattern_data in candlestick_patterns.items():
                if len(pattern_data) > 0:
                    recent_signals = pattern_data[-5:]  # Last 5 periods
                    if np.any(recent_signals != 0):
                        latest_signal = recent_signals[np.nonzero(recent_signals)][-1] if np.any(
                            recent_signals != 0) else 0
                        if latest_signal != 0:
                            recent_patterns.append({
                                'pattern': pattern_name,
                                'signal': 'bullish' if latest_signal > 0 else 'bearish',
                                'strength': abs(latest_signal),
                                'periods_ago': len(recent_signals) - np.where(recent_signals == latest_signal)[0][
                                    -1] - 1
                            })

            patterns['candlestick_patterns'] = recent_patterns

            # Price patterns (simplified detection)
            price_patterns = self._detect_price_patterns(close, high, low)
            patterns['price_patterns'] = price_patterns

            return patterns

        except Exception as e:
            return {"error": f"Error in pattern recognition: {str(e)}"}

    def _calculate_advanced_indicators(self, open_prices: np.ndarray, high: np.ndarray, low: np.ndarray,
                                       close: np.ndarray, volume: np.ndarray) -> Dict[str, Any]:
        """Calculate advanced technical indicators"""
        try:
            indicators = {}

            # Convert to pandas Series for pandas_ta
            open_series = pd.Series(open_prices)
            high_series = pd.Series(high)
            low_series = pd.Series(low)
            close_series = pd.Series(close)
            volume_series = pd.Series(volume)

            # Ichimoku Cloud
            try:
                # Using pandas_ta ichimoku
                ichimoku = ta.ichimoku(high=high_series, low=low_series, close=close_series)
                tenkan = ichimoku[0]['ITS_9'].iloc[-1] if not ichimoku[0].empty and 'ITS_9' in ichimoku[0] else None
                kijun = ichimoku[0]['IKS_26'].iloc[-1] if not ichimoku[0].empty and 'IKS_26' in ichimoku[0] else None

                indicators['ichimoku'] = {
                    'tenkan_sen': tenkan,
                    'kijun_sen': kijun
                }
            except Exception as e:
                # Fallback to simplified calculation if pandas_ta ichimoku fails
                period9_high = high_series.rolling(window=9).max()
                period9_low = high_series.rolling(window=9).min()
                tenkan_sen = (period9_high + period9_low) / 2

                period26_high = high_series.rolling(window=26).max()
                period26_low = high_series.rolling(window=26).min()
                kijun_sen = (period26_high + period26_low) / 2

                indicators['ichimoku'] = {
                    'tenkan_sen': tenkan_sen.iloc[-1] if not pd.isna(tenkan_sen.iloc[-1]) else None,
                    'kijun_sen': kijun_sen.iloc[-1] if not pd.isna(kijun_sen.iloc[-1]) else None
                }

            # Parabolic SAR
            try:
                sar = ta.psar(high_series, low_series)
                current_sar = sar['PSARl_0.02_0.2'].iloc[-1] if 'PSARl_0.02_0.2' in sar else None
                if pd.isna(current_sar):
                    current_sar = sar['PSARs_0.02_0.2'].iloc[-1] if 'PSARs_0.02_0.2' in sar else None

                indicators['parabolic_sar'] = {
                    'value': current_sar,
                    'signal': 'bullish' if close[-1] > current_sar else 'bearish' if current_sar else 'neutral'
                }
            except Exception as e:
                indicators['parabolic_sar'] = {"error": f"Could not calculate SAR: {str(e)}"}

            # Money Flow Index
            try:
                mfi = ta.mfi(high_series, low_series, close_series, volume_series, length=14).values
                current_mfi = mfi[-1] if not np.isnan(mfi[-1]) else None

                indicators['money_flow_index'] = {
                    'value': current_mfi,
                    'overbought': current_mfi > 80 if current_mfi else None,
                    'oversold': current_mfi < 20 if current_mfi else None
                }
            except Exception as e:
                indicators['money_flow_index'] = {"error": f"Could not calculate MFI: {str(e)}"}

            return indicators

        except Exception as e:
            return {"error": f"Error calculating advanced indicators: {str(e)}"}

    def _analyze_market_structure(self, close: np.ndarray, high: np.ndarray, low: np.ndarray) -> Dict[str, Any]:
        """Analyze market structure (higher highs, higher lows, etc.)"""
        try:
            # Find swing highs and lows
            swing_highs = []
            swing_lows = []

            # Simple swing point detection (can be enhanced)
            for i in range(2, len(close) - 2):
                # Swing high: higher than 2 periods before and after
                if high[i] > high[i - 1] and high[i] > high[i - 2] and high[i] > high[i + 1] and high[i] > high[i + 2]:
                    swing_highs.append((i, high[i]))

                # Swing low: lower than 2 periods before and after
                if low[i] < low[i - 1] and low[i] < low[i - 2] and low[i] < low[i + 1] and low[i] < low[i + 2]:
                    swing_lows.append((i, low[i]))

            # Determine market structure
            market_structure = "sideways"
            if len(swing_highs) >= 2 and len(swing_lows) >= 2:
                recent_highs = swing_highs[-2:]
                recent_lows = swing_lows[-2:]

                higher_highs = recent_highs[1][1] > recent_highs[0][1]
                higher_lows = recent_lows[1][1] > recent_lows[0][1]
                lower_highs = recent_highs[1][1] < recent_highs[0][1]
                lower_lows = recent_lows[1][1] < recent_lows[0][1]

                if higher_highs and higher_lows:
                    market_structure = "uptrend"
                elif lower_highs and lower_lows:
                    market_structure = "downtrend"
                elif higher_highs and lower_lows:
                    market_structure = "expanding_range"
                elif lower_highs and higher_lows:
                    market_structure = "contracting_range"

            return {
                "market_structure": market_structure,
                "swing_highs_count": len(swing_highs),
                "swing_lows_count": len(swing_lows),
                "recent_swing_high": swing_highs[-1][1] if swing_highs else None,
                "recent_swing_low": swing_lows[-1][1] if swing_lows else None
            }

        except Exception as e:
            return {"error": f"Error analyzing market structure: {str(e)}"}

    def _generate_trading_signals(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trading signals based on technical analysis"""
        try:
            signals = {
                "overall_signal": "neutral",
                "signal_strength": 0,
                "buy_signals": [],
                "sell_signals": [],
                "neutral_signals": []
            }

            signal_score = 0

            # Trend signals
            trend_analysis = analysis_result.get("trend_analysis", {})
            if trend_analysis.get("short_term_trend") == "bullish":
                signals["buy_signals"].append("Short-term uptrend")
                signal_score += 1
            elif trend_analysis.get("short_term_trend") == "bearish":
                signals["sell_signals"].append("Short-term downtrend")
                signal_score -= 1

            # Momentum signals
            momentum_analysis = analysis_result.get("momentum_analysis", {})
            rsi_data = momentum_analysis.get("rsi", {})
            if rsi_data.get("oversold"):
                signals["buy_signals"].append("RSI oversold")
                signal_score += 1
            elif rsi_data.get("overbought"):
                signals["sell_signals"].append("RSI overbought")
                signal_score -= 1

            macd_data = momentum_analysis.get("macd", {})
            if macd_data.get("bullish_crossover"):
                signals["buy_signals"].append("MACD bullish crossover")
                signal_score += 1
            elif macd_data.get("bullish_crossover") == False:
                signals["sell_signals"].append("MACD bearish crossover")
                signal_score -= 1

            # Volume confirmation
            volume_analysis = analysis_result.get("volume_analysis", {})
            if volume_analysis.get("volume_confirmation") == "bullish":
                signals["buy_signals"].append("Volume confirms upward movement")
                signal_score += 0.5
            elif volume_analysis.get("volume_confirmation") == "bearish":
                signals["sell_signals"].append("Volume confirms downward movement")
                signal_score -= 0.5

            # Support/Resistance signals
            support_resistance = analysis_result.get("support_resistance", {})
            current_price = support_resistance.get("current_price")
            nearest_support = support_resistance.get("nearest_support")
            nearest_resistance = support_resistance.get("nearest_resistance")

            if current_price and nearest_support and current_price <= nearest_support * 1.02:
                signals["buy_signals"].append("Near support level")
                signal_score += 0.5

            if current_price and nearest_resistance and current_price >= nearest_resistance * 0.98:
                signals["sell_signals"].append("Near resistance level")
                signal_score -= 0.5

            # Determine overall signal
            if signal_score >= 2:
                signals["overall_signal"] = "strong_buy"
                signals["signal_strength"] = min(10, int(signal_score * 2))
            elif signal_score >= 1:
                signals["overall_signal"] = "buy"
                signals["signal_strength"] = min(10, int(signal_score * 3))
            elif signal_score <= -2:
                signals["overall_signal"] = "strong_sell"
                signals["signal_strength"] = min(10, int(abs(signal_score) * 2))
            elif signal_score <= -1:
                signals["overall_signal"] = "sell"
                signals["signal_strength"] = min(10, int(abs(signal_score) * 3))
            else:
                signals["overall_signal"] = "neutral"
                signals["signal_strength"] = 0

            return signals

        except Exception as e:
            return {"error": f"Error generating trading signals: {str(e)}"}

    def _calculate_overall_rating(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall technical rating"""
        try:
            scores = {
                "trend": 0,
                "momentum": 0,
                "volatility": 0,
                "volume": 0,
                "support_resistance": 0
            }

            # Trend score
            trend_analysis = analysis_result.get("trend_analysis", {})
            if trend_analysis.get("short_term_trend") == "bullish":
                scores["trend"] += 2
            elif trend_analysis.get("short_term_trend") == "bearish":
                scores["trend"] -= 2

            if trend_analysis.get("long_term_trend") == "bullish":
                scores["trend"] += 1
            elif trend_analysis.get("long_term_trend") == "bearish":
                scores["trend"] -= 1

            # Momentum score
            momentum_analysis = analysis_result.get("momentum_analysis", {})
            momentum_score = momentum_analysis.get("momentum_score", 0)
            scores["momentum"] = momentum_score

            # Volume score
            volume_analysis = analysis_result.get("volume_analysis", {})
            volume_confirmation = volume_analysis.get("volume_confirmation")
            if volume_confirmation == "bullish":
                scores["volume"] = 1
            elif volume_confirmation == "bearish":
                scores["volume"] = -1

            # Calculate overall score
            total_score = sum(scores.values())
            max_possible_score = 10  # Theoretical maximum
            normalized_score = max(-10, min(10, total_score))

            # Rating categories
            if normalized_score >= 6:
                rating = "strong_buy"
            elif normalized_score >= 3:
                rating = "buy"
            elif normalized_score >= 1:
                rating = "weak_buy"
            elif normalized_score <= -6:
                rating = "strong_sell"
            elif normalized_score <= -3:
                rating = "sell"
            elif normalized_score <= -1:
                rating = "weak_sell"
            else:
                rating = "neutral"

            return {
                "overall_score": normalized_score,
                "rating": rating,
                "component_scores": scores,
                "confidence_level": min(100, abs(normalized_score) * 10)
            }

        except Exception as e:
            return {"error": f"Error calculating overall rating: {str(e)}"}

    # Helper methods for various calculations and interpretations
    def _determine_trend(self, price: float, ma1: float, ma2: float) -> str:
        """Determine trend based on price and moving averages"""
        if not (price and ma1 and ma2):
            return "neutral"

        if price > ma1 > ma2:
            return "bullish"
        elif price < ma1 < ma2:
            return "bearish"
        else:
            return "neutral"

    def _calculate_trend_strength(self, close: np.ndarray, sma_20: np.ndarray, sma_50: np.ndarray) -> str:
        """Calculate trend strength"""
        try:
            if len(close) < 20:
                return "insufficient_data"

            # Calculate average distance between price and moving averages
            recent_close = close[-10:]
            recent_sma20 = sma_20[-10:]
            recent_sma50 = sma_50[-10:]

            # Remove NaN values
            valid_indices = ~(np.isnan(recent_close) | np.isnan(recent_sma20) | np.isnan(recent_sma50))
            if not np.any(valid_indices):
                return "insufficient_data"

            recent_close = recent_close[valid_indices]
            recent_sma20 = recent_sma20[valid_indices]
            recent_sma50 = recent_sma50[valid_indices]

            # Calculate average percentage distance
            avg_distance_20 = np.mean(np.abs((recent_close - recent_sma20) / recent_sma20) * 100)
            avg_distance_50 = np.mean(np.abs((recent_close - recent_sma50) / recent_sma50) * 100)
            avg_distance = (avg_distance_20 + avg_distance_50) / 2

            if avg_distance > 5:
                return "strong"
            elif avg_distance > 2:
                return "moderate"
            else:
                return "weak"

        except Exception:
            return "error"

    def _interpret_adx(self, adx: float) -> str:
        """Interpret ADX values"""
        if not adx:
            return "unknown"

        if adx > 50:
            return "very_strong_trend"
        elif adx > 25:
            return "strong_trend"
        elif adx > 20:
            return "trending"
        else:
            return "weak_trend_or_ranging"

    def _interpret_rsi(self, rsi: float) -> str:
        """Interpret RSI values"""
        if not rsi:
            return "unknown"

        if rsi > 80:
            return "extremely_overbought"
        elif rsi > 70:
            return "overbought"
        elif rsi > 50:
            return "bullish"
        elif rsi > 30:
            return "bearish"
        elif rsi > 20:
            return "oversold"
        else:
            return "extremely_oversold"

    def _interpret_macd(self, macd: float, signal: float, histogram: float) -> str:
        """Interpret MACD signals"""
        if not all([macd, signal, histogram]):
            return "unknown"

        if macd > signal and histogram > 0:
            return "bullish_momentum"
        elif macd < signal and histogram < 0:
            return "bearish_momentum"
        elif macd > signal and histogram < 0:
            return "bullish_weakening"
        elif macd < signal and histogram > 0:
            return "bearish_weakening"
        else:
            return "neutral"

    def _detect_bb_squeeze(self, bb_upper: np.ndarray, bb_lower: np.ndarray, atr: np.ndarray) -> bool:
        """Detect Bollinger Band squeeze"""
        try:
            if len(bb_upper) < 20 or len(atr) < 20:
                return False

            # Current band width
            current_width = bb_upper[-1] - bb_lower[-1]

            # Average band width over last 20 periods
            recent_widths = bb_upper[-20:] - bb_lower[-20:]
            avg_width = np.mean(recent_widths[~np.isnan(recent_widths)])

            # Squeeze if current width is significantly below average
            return current_width < avg_width * 0.8

        except Exception:
            return False

    def _interpret_atr(self, atr: float, price: float) -> str:
        """Interpret ATR relative to price"""
        if not (atr and price):
            return "unknown"

        atr_pct = (atr / price) * 100

        if atr_pct > 5:
            return "very_high_volatility"
        elif atr_pct > 3:
            return "high_volatility"
        elif atr_pct > 1.5:
            return "moderate_volatility"
        else:
            return "low_volatility"

    def _interpret_volatility(self, vol_pct: float) -> str:
        """Interpret historical volatility percentage"""
        if vol_pct > 40:
            return "extremely_high"
        elif vol_pct > 25:
            return "high"
        elif vol_pct > 15:
            return "moderate"
        elif vol_pct > 10:
            return "low"
        else:
            return "very_low"

    def _determine_volatility_regime(self, vol_pct: float) -> str:
        """Determine volatility regime"""
        if vol_pct > 30:
            return "crisis"
        elif vol_pct > 20:
            return "high_vol"
        elif vol_pct > 15:
            return "normal"
        else:
            return "low_vol"

    def _analyze_volatility_trend(self, atr: np.ndarray) -> str:
        """Analyze if volatility is increasing or decreasing"""
        try:
            if len(atr) < 10:
                return "insufficient_data"

            recent_atr = atr[-5:]
            earlier_atr = atr[-10:-5]

            # Remove NaN values
            recent_atr = recent_atr[~np.isnan(recent_atr)]
            earlier_atr = earlier_atr[~np.isnan(earlier_atr)]

            if len(recent_atr) == 0 or len(earlier_atr) == 0:
                return "insufficient_data"

            recent_avg = np.mean(recent_atr)
            earlier_avg = np.mean(earlier_atr)

            change_pct = ((recent_avg - earlier_avg) / earlier_avg) * 100

            if change_pct > 10:
                return "increasing"
            elif change_pct < -10:
                return "decreasing"
            else:
                return "stable"

        except Exception:
            return "error"

    def _calculate_pivot_points(self, high: float, low: float, close: float) -> Dict[str, float]:
        """Calculate traditional pivot points"""
        pivot = (high + low + close) / 3

        return {
            "pivot": pivot,
            "r1": 2 * pivot - low,
            "r2": pivot + (high - low),
            "r3": high + 2 * (pivot - low),
            "s1": 2 * pivot - high,
            "s2": pivot - (high - low),
            "s3": low - 2 * (high - pivot)
        }

    def _find_resistance_levels(self, highs: np.ndarray, current_price: float) -> List[float]:
        """Find significant resistance levels"""
        # Simplified resistance finding - can be enhanced with more sophisticated algorithms
        unique_highs = np.unique(np.round(highs, 2))
        resistance_levels = [high for high in unique_highs if high > current_price]

        # Sort by proximity to current price
        resistance_levels.sort(key=lambda x: abs(x - current_price))

        return resistance_levels[:5]  # Return top 5 levels

    def _find_support_levels(self, lows: np.ndarray, current_price: float) -> List[float]:
        """Find significant support levels"""
        # Simplified support finding
        unique_lows = np.unique(np.round(lows, 2))
        support_levels = [low for low in unique_lows if low < current_price]

        # Sort by proximity to current price
        support_levels.sort(key=lambda x: abs(x - current_price))

        return support_levels[:5]  # Return top 5 levels

    def _calculate_fibonacci_levels(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Dict[str, Any]:
        """Calculate Fibonacci retracement levels"""
        try:
            # Find recent swing high and low
            recent_periods = min(50, len(high))
            recent_high = np.max(high[-recent_periods:])
            recent_low = np.min(low[-recent_periods:])

            # Determine if we're in uptrend or downtrend
            trend_direction = "up" if close[-1] > close[-recent_periods] else "down"

            # Calculate Fibonacci levels
            diff = recent_high - recent_low

            if trend_direction == "up":
                # Retracement levels for uptrend
                levels = {
                    "0%": recent_high,
                    "23.6%": recent_high - (diff * 0.236),
                    "38.2%": recent_high - (diff * 0.382),
                    "50%": recent_high - (diff * 0.5),
                    "61.8%": recent_high - (diff * 0.618),
                    "78.6%": recent_high - (diff * 0.786),
                    "100%": recent_low
                }
            else:
                # Retracement levels for downtrend
                levels = {
                    "0%": recent_low,
                    "23.6%": recent_low + (diff * 0.236),
                    "38.2%": recent_low + (diff * 0.382),
                    "50%": recent_low + (diff * 0.5),
                    "61.8%": recent_low + (diff * 0.618),
                    "78.6%": recent_low + (diff * 0.786),
                    "100%": recent_high
                }

            return {
                "trend_direction": trend_direction,
                "swing_high": recent_high,
                "swing_low": recent_low,
                "levels": levels
            }

        except Exception as e:
            return {"error": f"Error calculating Fibonacci levels: {str(e)}"}

    def _analyze_volume_trend(self, obv: np.ndarray) -> str:
        """Analyze OBV trend"""
        try:
            if len(obv) < 10:
                return "insufficient_data"

            recent_obv = obv[-5:]
            earlier_obv = obv[-10:-5]

            # Remove NaN values
            recent_obv = recent_obv[~np.isnan(recent_obv)]
            earlier_obv = earlier_obv[~np.isnan(earlier_obv)]

            if len(recent_obv) == 0 or len(earlier_obv) == 0:
                return "insufficient_data"

            recent_avg = np.mean(recent_obv)
            earlier_avg = np.mean(earlier_obv)

            change_pct = ((recent_avg - earlier_avg) / abs(earlier_avg)) * 100 if earlier_avg != 0 else 0

            if change_pct > 5:
                return "accumulation"
            elif change_pct < -5:
                return "distribution"
            else:
                return "neutral"

        except Exception:
            return "error"

    def _check_volume_confirmation(self, close: np.ndarray, volume: np.ndarray) -> str:
        """Check if volume confirms price movement"""
        try:
            if len(close) < 5 or len(volume) < 5:
                return "insufficient_data"

            # Recent price change
            recent_price_change = close[-1] - close[-5]

            # Recent volume vs average
            recent_volume = np.mean(volume[-3:])
            avg_volume = np.mean(volume[-20:]) if len(volume) >= 20 else np.mean(volume)
            volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1

            # Volume confirmation logic
            if recent_price_change > 0 and volume_ratio > 1.2:
                return "bullish"
            elif recent_price_change < 0 and volume_ratio > 1.2:
                return "bearish"
            else:
                return "neutral"

        except Exception:
            return "error"

    def _interpret_volume(self, current_volume: float, avg_volume: float, volume_trend: str) -> str:
        """Interpret volume conditions"""
        if not (current_volume and avg_volume):
            return "unknown"

        volume_ratio = current_volume / avg_volume
        interpretation = []

        if volume_ratio > 2:
            interpretation.append("very_high_volume")
        elif volume_ratio > 1.5:
            interpretation.append("high_volume")
        elif volume_ratio < 0.5:
            interpretation.append("low_volume")
        else:
            interpretation.append("normal_volume")

        if volume_trend == "accumulation":
            interpretation.append("accumulation_phase")
        elif volume_trend == "distribution":
            interpretation.append("distribution_phase")

        return "_".join(interpretation)

    def _detect_price_patterns(self, close: np.ndarray, high: np.ndarray, low: np.ndarray) -> List[Dict[str, Any]]:
        """Detect basic price patterns"""
        patterns = []

        try:
            if len(close) < 20:
                return patterns

            # Simple triangle pattern detection
            if self._detect_triangle_pattern(close, high, low):
                patterns.append({
                    "pattern": "triangle",
                    "type": "continuation",
                    "timeframe": "intermediate"
                })

            # Simple head and shoulders detection (very basic)
            if self._detect_head_shoulders_pattern(high):
                patterns.append({
                    "pattern": "head_and_shoulders",
                    "type": "reversal",
                    "timeframe": "intermediate"
                })

            # Double top/bottom detection (basic)
            double_pattern = self._detect_double_pattern(high, low)
            if double_pattern:
                patterns.append(double_pattern)

            return patterns

        except Exception:
            return []

    def _detect_triangle_pattern(self, close: np.ndarray, high: np.ndarray, low: np.ndarray) -> bool:
        """Basic triangle pattern detection"""
        try:
            if len(close) < 20:
                return False

            recent_data = close[-20:]
            recent_highs = high[-20:]
            recent_lows = low[-20:]

            # Check if highs are decreasing and lows are increasing (converging)
            high_trend = np.polyfit(range(len(recent_highs)), recent_highs, 1)[0]
            low_trend = np.polyfit(range(len(recent_lows)), recent_lows, 1)[0]

            # Converging if high trend is negative and low trend is positive
            return high_trend < -0.01 and low_trend > 0.01

        except Exception:
            return False

    def _detect_head_shoulders_pattern(self, high: np.ndarray) -> bool:
        """Very basic head and shoulders detection"""
        try:
            if len(high) < 15:
                return False

            # Find three peaks in recent data
            recent_highs = high[-15:]

            # This is a very simplified detection - in practice, would need more sophisticated logic
            max_indices = []
            for i in range(2, len(recent_highs) - 2):
                if (recent_highs[i] > recent_highs[i - 1] and recent_highs[i] > recent_highs[i - 2] and
                        recent_highs[i] > recent_highs[i + 1] and recent_highs[i] > recent_highs[i + 2]):
                    max_indices.append(i)

            # Need at least 3 peaks for head and shoulders
            if len(max_indices) >= 3:
                # Check if middle peak is highest (head)
                peaks = [recent_highs[i] for i in max_indices[-3:]]
                return peaks[1] > peaks[0] and peaks[1] > peaks[2]

            return False

        except Exception:
            return False

    def _detect_double_pattern(self, high: np.ndarray, low: np.ndarray) -> Optional[Dict[str, Any]]:
        """Basic double top/bottom detection"""
        try:
            if len(high) < 20:
                return None

            recent_highs = high[-20:]
            recent_lows = low[-20:]

            # Find peaks and troughs
            peaks = []
            troughs = []

            for i in range(2, len(recent_highs) - 2):
                if (recent_highs[i] > recent_highs[i - 1] and recent_highs[i] > recent_highs[i - 2] and
                        recent_highs[i] > recent_highs[i + 1] and recent_highs[i] > recent_highs[i + 2]):
                    peaks.append(recent_highs[i])

                if (recent_lows[i] < recent_lows[i - 1] and recent_lows[i] < recent_lows[i - 2] and
                        recent_lows[i] < recent_lows[i + 1] and recent_lows[i] < recent_lows[i + 2]):
                    troughs.append(recent_lows[i])

            # Check for double top
            if len(peaks) >= 2:
                last_two_peaks = peaks[-2:]
                if abs(last_two_peaks[0] - last_two_peaks[1]) / last_two_peaks[0] < 0.05:  # Within 5%
                    return {
                        "pattern": "double_top",
                        "type": "reversal",
                        "timeframe": "intermediate"
                    }

            # Check for double bottom
            if len(troughs) >= 2:
                last_two_troughs = troughs[-2:]
                if abs(last_two_troughs[0] - last_two_troughs[1]) / last_two_troughs[0] < 0.05:  # Within 5%
                    return {
                        "pattern": "double_bottom",
                        "type": "reversal",
                        "timeframe": "intermediate"
                    }

            return None

        except Exception:
            return None

    def _calculate_momentum_score(self, rsi: float, macd: float, macd_signal: float, stoch_k: float) -> float:
        """Calculate overall momentum score"""
        try:
            score = 0
            count = 0

            # RSI contribution
            if rsi is not None:
                if rsi > 70:
                    score -= 2  # Overbought
                elif rsi > 50:
                    score += 1  # Bullish
                elif rsi < 30:
                    score += 2  # Oversold (contrarian bullish)
                else:
                    score -= 1  # Bearish
                count += 1

            # MACD contribution
            if macd is not None and macd_signal is not None:
                if macd > macd_signal:
                    score += 1
                else:
                    score -= 1
                count += 1

            # Stochastic contribution
            if stoch_k is not None:
                if stoch_k > 80:
                    score -= 1  # Overbought
                elif stoch_k > 50:
                    score += 0.5  # Bullish
                elif stoch_k < 20:
                    score += 1  # Oversold
                else:
                    score -= 0.5  # Bearish
                count += 1

            # Normalize score
            if count > 0:
                normalized_score = score / count
                return max(-3, min(3, normalized_score))  # Cap between -3 and 3
            else:
                return 0

        except Exception:
            return 0

    async def _analyze_market_conditions(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze overall market conditions"""
        try:
            market_summary = data.get('market_summary', {})
            market_analysis = {
                "market_trend": "neutral",
                "market_volatility": "moderate",
                "risk_sentiment": "neutral"
            }

            # Analyze major indices
            indices_analysis = {}
            major_indices = ['^GSPC', '^DJI', '^IXIC', '^RUT', '^VIX']

            for index in major_indices:
                if index in market_summary and not market_summary[index].get('error'):
                    index_data = market_summary[index]
                    change_pct = index_data.get('change_percent', 0)
                    indices_analysis[index] = {
                        "change_percent": change_pct,
                        "trend": "bullish" if change_pct > 0.5 else "bearish" if change_pct < -0.5 else "neutral"
                    }

            # Determine overall market trend
            if indices_analysis:
                positive_indices = sum(1 for data in indices_analysis.values() if data['trend'] == 'bullish')
                negative_indices = sum(1 for data in indices_analysis.values() if data['trend'] == 'bearish')

                if positive_indices > negative_indices:
                    market_analysis["market_trend"] = "bullish"
                elif negative_indices > positive_indices:
                    market_analysis["market_trend"] = "bearish"

            # VIX analysis for volatility
            if '^VIX' in indices_analysis:
                vix_change = indices_analysis['^VIX']['change_percent']
                if vix_change > 10:
                    market_analysis["market_volatility"] = "high"
                    market_analysis["risk_sentiment"] = "risk_off"
                elif vix_change < -10:
                    market_analysis["market_volatility"] = "low"
                    market_analysis["risk_sentiment"] = "risk_on"

            market_analysis["indices_analysis"] = indices_analysis

            return market_analysis

        except Exception as e:
            return {"error": f"Error analyzing market conditions: {str(e)}"}

    async def _analyze_sector_rotation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze sector rotation patterns"""
        try:
            sector_performance = data.get('sector_performance', {})
            if not sector_performance or sector_performance.get('error'):
                return {"error": "No sector performance data available"}

            # Rank sectors by performance
            sector_rankings = []
            for sector, sector_data in sector_performance.items():
                if isinstance(sector_data, dict) and 'change_percent' in sector_data:
                    sector_rankings.append({
                        'sector': sector,
                        'performance': sector_data['change_percent'],
                        'volume': sector_data.get('volume', 0)
                    })

            # Sort by performance
            sector_rankings.sort(key=lambda x: x['performance'], reverse=True)

            # Identify market regime based on sector leadership
            market_regime = self._identify_market_regime(sector_rankings)

            return {
                "sector_rankings": sector_rankings,
                "top_performing_sectors": sector_rankings[:3],
                "worst_performing_sectors": sector_rankings[-3:],
                "market_regime": market_regime,
                "rotation_signal": self._detect_rotation_signal(sector_rankings)
            }

        except Exception as e:
            return {"error": f"Error analyzing sector rotation: {str(e)}"}

    def _identify_market_regime(self, sector_rankings: List[Dict]) -> str:
        """Identify market regime based on sector leadership"""
        try:
            if not sector_rankings:
                return "unknown"

            top_sectors = [s['sector'] for s in sector_rankings[:3]]

            # Growth regime indicators
            growth_sectors = ['Technology', 'Communication Services', 'Consumer Discretionary']
            growth_leadership = any(sector in growth_sectors for sector in top_sectors)

            # Value regime indicators
            value_sectors = ['Financials', 'Energy', 'Materials']
            value_leadership = any(sector in value_sectors for sector in top_sectors)

            # Defensive regime indicators
            defensive_sectors = ['Utilities', 'Consumer Staples', 'Healthcare']
            defensive_leadership = any(sector in defensive_sectors for sector in top_sectors)

            if growth_leadership:
                return "growth_regime"
            elif value_leadership:
                return "value_regime"
            elif defensive_leadership:
                return "defensive_regime"
            else:
                return "mixed_regime"

        except Exception as e:
            return "unknown"

    def _detect_rotation_signal(self, sector_rankings: List[Dict]) -> str:
        """Detect sector rotation signals"""
        try:
            if len(sector_rankings) < 5:
                return "insufficient_data"

            # Simple rotation detection based on performance spread
            top_performance = sector_rankings[0]['performance']
            bottom_performance = sector_rankings[-1]['performance']
            performance_spread = top_performance - bottom_performance

            if performance_spread > 5:
                return "strong_rotation"
            elif performance_spread > 2:
                return "moderate_rotation"
            else:
                return "low_rotation"

        except Exception:
            return "unknown"

    def _generate_analysis_summary(self, analysis_results: Dict[str, Any], market_analysis: Dict[str, Any]) -> str:
        """Generate summary of technical analysis"""
        try:
            # Count bullish and bearish symbols
            bullish_count = 0
            bearish_count = 0

            for symbol, analysis in analysis_results.items():
                if analysis.get('signals', {}).get('overall_signal') == 'bullish':
                    bullish_count += 1
                elif analysis.get('signals', {}).get('overall_signal') == 'bearish':
                    bearish_count += 1

            # Market trend summary
            market_trend = market_analysis.get('market_trend', 'neutral')
            market_volatility = market_analysis.get('market_volatility', 'moderate')

            return (
                f"Technical Analysis Summary: "
                f"Bullish Symbols: {bullish_count}, "
                f"Bearish Symbols: {bearish_count}, "
                f"Market Trend: {market_trend}, "
                f"Market Volatility: {market_volatility}"
            )

        except Exception as e:
            return f"Error generating summary: {str(e)}"


class TechnicalAnalystAgent(Agent):
    """Agent for performing comprehensive technical analysis"""

    def __init__(self, gcp_services, config):
        super().__init__(
            model="gemini-2.0-flash-exp",
            name="technical_analyst_agent",
            description="Expert technical analyst with comprehensive market analysis capabilities",
            instructions="""You are a senior technical analyst with expertise in:
1. Price action analysis and pattern recognition
2. Technical indicator interpretation
3. Market structure analysis
4. Support/resistance identification
5. Trading signal generation

Your primary responsibilities:
- Analyze price charts and identify key patterns
- Interpret technical indicators for trend/momentum
- Identify key support/resistance levels
- Generate trading signals with confidence levels
- Provide risk-reward assessments

Focus on providing analysis that considers:
- Multiple time frame analysis
- Confluence of technical factors
- Volume confirmation
- Market context and conditions""",
            tools=[AdvancedTechnicalAnalysisTool()]
        )
        self.gcp_services = gcp_services
        self.config = config
        logger.info("TechnicalAnalystAgent initialized")

    async def analyze_market(self, market_data: Dict[str, Any], analysis_type: str = "comprehensive") -> Dict[str, Any]:
        """Main method for technical analysis"""
        try:
            logger.info(f"Starting technical analysis ({analysis_type})")

            # Call the technical analysis tool
            analysis_results = await self.tools[0].call(
                market_data=market_data,
                analysis_type=analysis_type
            )

            # Store results if needed
            if self.gcp_services:
                await self.gcp_services.store_technical_analysis(analysis_results)

            return analysis_results

        except Exception as e:
            error_msg = f"Error in technical analysis: {str(e)}"
            logger.error(error_msg)
            return {
                "status": "error",
                "error": error_msg,
                "timestamp": datetime.utcnow().isoformat()
            }
