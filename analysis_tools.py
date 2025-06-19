import numpy as np
import pandas as pd
import pandas_ta as ta
from typing import List, Dict, Any, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class AnalysisTools:
    """Advanced financial analysis toolkit using pandas-ta"""

    @staticmethod
    def calculate_technical_indicators(ohlc: Dict[str, List[float]]) -> Dict[str, List[Optional[float]]]:
        """
        Calculate comprehensive set of technical indicators

        Args:
            ohlc: Dict with keys 'open', 'high', 'low', 'close', 'volume'

        Returns:
            Dict of indicator names to values
        """
        results = {}

        try:
            df = pd.DataFrame(ohlc)

            # Ensure columns exist
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in required_cols:
                if col not in df.columns:
                    logger.warning(f"Missing {col} in OHLC data")
                    df[col] = pd.Series([None] * len(df))

            close = df['close']
            high = df['high']
            low = df['low']
            volume = df['volume']

            # Trend indicators
            results['sma_20'] = ta.sma(close, length=20).tolist()
            results['sma_50'] = ta.sma(close, length=50).tolist()
            results['sma_200'] = ta.sma(close, length=200).tolist()
            results['ema_12'] = ta.ema(close, length=12).tolist()
            results['ema_26'] = ta.ema(close, length=26).tolist()

            # Momentum indicators
            results['rsi'] = ta.rsi(close, length=14).tolist()
            stoch = ta.stoch(high, low, close)
            results['stoch_k'] = stoch[f"STOCHk_14_3"].tolist() if f"STOCHk_14_3" in stoch else []
            results['stoch_d'] = stoch[f"STOCHd_14_3"].tolist() if f"STOCHd_14_3" in stoch else []
            macd = ta.macd(close)
            results['macd'] = macd["MACD_12_26_9"].tolist() if "MACD_12_26_9" in macd else []
            results['macd_signal'] = macd["MACDs_12_26_9"].tolist() if "MACDs_12_26_9" in macd else []
            results['macd_hist'] = macd["MACDh_12_26_9"].tolist() if "MACDh_12_26_9" in macd else []

            # Volatility indicators
            results['atr'] = ta.atr(high, low, close, length=14).tolist()
            bbands = ta.bbands(close, length=20, std=2)
            results['bb_upper'] = bbands[f"BBU_20_2.0"].tolist() if f"BBU_20_2.0" in bbands else []
            results['bb_middle'] = bbands[f"BBM_20_2.0"].tolist() if f"BBM_20_2.0" in bbands else []
            results['bb_lower'] = bbands[f"BBL_20_2.0"].tolist() if f"BBL_20_2.0" in bbands else []

            # Volume indicators
            results['obv'] = ta.obv(close, volume).tolist()
            results['adi'] = ta.ad(high, low, close, volume).tolist()

        except Exception as e:
            logger.error(f"Error calculating indicators with pandas_ta: {str(e)}")
            return AnalysisTools._calculate_manual_indicators(ohlc)

        return results

    @staticmethod
    def _calculate_manual_indicators(ohlc: Dict[str, List[float]]) -> Dict[str, List[Optional[float]]]:
        """Manual calculation of technical indicators as fallback"""
        results = {}
        close = np.array(ohlc['close'])

        # Simple Moving Averages
        results['sma_20'] = AnalysisTools._calculate_sma(close, 20)
        results['sma_50'] = AnalysisTools._calculate_sma(close, 50)
        results['sma_200'] = AnalysisTools._calculate_sma(close, 200)

        # Exponential Moving Averages
        results['ema_12'] = AnalysisTools._calculate_ema(close, 12)
        results['ema_26'] = AnalysisTools._calculate_ema(close, 26)

        # RSI
        results['rsi'] = AnalysisTools._calculate_rsi(close, 14)

        # Placeholder for other indicators
        results['stoch_k'] = [None] * len(close)
        results['stoch_d'] = [None] * len(close)
        results['macd'] = [None] * len(close)
        results['macd_signal'] = [None] * len(close)
        results['macd_hist'] = [None] * len(close)
        results['bb_upper'] = [None] * len(close)
        results['bb_middle'] = [None] * len(close)
        results['bb_lower'] = [None] * len(close)
        results['obv'] = [None] * len(close)
        results['adi'] = [None] * len(close)

        return results

    @staticmethod
    def _calculate_sma(prices: np.ndarray, period: int) -> List[Optional[float]]:
        """Calculate Simple Moving Average"""
        result = [None] * len(prices)
        for i in range(period - 1, len(prices)):
            result[i] = np.mean(prices[i - period + 1:i + 1])
        return result

    @staticmethod
    def _calculate_ema(prices: np.ndarray, period: int) -> List[Optional[float]]:
        """Calculate Exponential Moving Average"""
        result = [None] * len(prices)
        if len(prices) == 0:
            return result

        multiplier = 2.0 / (period + 1)
        result[0] = float(prices[0])

        for i in range(1, len(prices)):
            if result[i - 1] is not None:
                result[i] = (prices[i] * multiplier) + (result[i - 1] * (1 - multiplier))
            else:
                result[i] = float(prices[i])

        return result

    @staticmethod
    def _calculate_rsi(prices: np.ndarray, period: int) -> List[Optional[float]]:
        """Calculate RSI"""
        result = [None] * len(prices)
        if len(prices) < period + 1:
            return result

        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])

        for i in range(period, len(prices)):
            if avg_loss != 0:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
                result[i] = rsi
            else:
                result[i] = 100

            # Update averages
            if i < len(deltas):
                gain = max(deltas[i], 0)
                loss = max(-deltas[i], 0)
                avg_gain = ((avg_gain * (period - 1)) + gain) / period
                avg_loss = ((avg_loss * (period - 1)) + loss) / period

        return result

    @staticmethod
    def calculate_fundamental_metrics(
            financials: Dict[str, Any],
            prices: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate fundamental valuation metrics"""
        metrics = {}

        revenue = financials.get('revenue', 0)
        net_income = financials.get('net_income', 0)
        shares_outstanding = financials.get('shares_outstanding', 1)
        current_price = prices.get('current', 0)

        total_assets = financials.get('total_assets', 1)
        total_liabilities = financials.get('total_liabilities', 0)
        total_equity = total_assets - total_liabilities

        metrics['pe_ratio'] = current_price / (net_income / shares_outstanding) if net_income > 0 else 0
        metrics['ps_ratio'] = (current_price * shares_outstanding) / revenue if revenue > 0 else 0
        metrics['pb_ratio'] = (current_price * shares_outstanding) / total_equity if total_equity > 0 else 0

        metrics['gross_margin'] = financials.get('gross_profit', 0) / revenue if revenue > 0 else 0
        metrics['net_margin'] = net_income / revenue if revenue > 0 else 0
        metrics['roe'] = net_income / total_equity if total_equity > 0 else 0

        current_assets = financials.get('current_assets', 0)
        current_liabilities = financials.get('current_liabilities', 0)
        metrics['current_ratio'] = current_assets / current_liabilities if current_liabilities > 0 else 0

        metrics['debt_to_equity'] = total_liabilities / total_equity if total_equity > 0 else 0

        return metrics

    @staticmethod
    def calculate_risk_metrics(
            returns: List[float],
            benchmark_returns: List[float]
    ) -> Dict[str, float]:
        """Calculate portfolio risk metrics"""
        if len(returns) < 2 or len(returns) != len(benchmark_returns):
            return {}

        np_returns = np.array(returns)
        np_bench = np.array(benchmark_returns)

        metrics = {}
        metrics['volatility'] = np.std(np_returns) * np.sqrt(252)
        metrics['sharpe_ratio'] = (np.mean(np_returns) * 252) / metrics['volatility'] if metrics[
                                                                                             'volatility'] > 0 else 0

        cum_returns = np.cumprod(1 + np_returns)
        peak = np.maximum.accumulate(cum_returns)
        drawdown = (peak - cum_returns) / peak
        metrics['max_drawdown'] = np.max(drawdown)

        covariance = np.cov(np_returns, np_bench)[0, 1]
        benchmark_variance = np.var(np_bench)
        metrics['beta'] = covariance / benchmark_variance if benchmark_variance > 0 else 0

        metrics['alpha'] = np.mean(np_returns) - metrics['beta'] * np.mean(np_bench)

        metrics['var_95'] = np.percentile(np_returns, 5) * np.sqrt(252)

        return metrics

    @staticmethod
    def detect_market_regime(
            prices: List[float],
            volatility_thresholds: Tuple[float, float] = (0.15, 0.30)
    ) -> List[str]:
        """Detect market regime based on volatility"""
        if len(prices) < 20:
            return []

        returns = np.diff(prices) / prices[:-1]
        regimes = []

        for i in range(20, len(returns)):
            window = returns[i - 20:i]
            volatility = np.std(window) * np.sqrt(252)

            if volatility < volatility_thresholds[0]:
                regimes.append('low_volatility')
            elif volatility < volatility_thresholds[1]:
                regimes.append('moderate_volatility')
            else:
                regimes.append('high_volatility')

        return regimes

    @staticmethod
    def calculate_correlation_matrix(assets: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
        """Calculate correlation matrix between assets"""
        symbols = list(assets.keys())
        returns = {}

        for symbol, prices in assets.items():
            if len(prices) > 1:
                returns[symbol] = np.diff(prices) / prices[:-1]

        corr_matrix = {}
        for sym1 in symbols:
            if sym1 not in returns:
                continue

            corr_matrix[sym1] = {}
            for sym2 in symbols:
                if sym2 not in returns or sym1 == sym2:
                    corr_matrix[sym1][sym2] = 1.0
                    continue

                min_len = min(len(returns[sym1]), len(returns[sym2]))
                if min_len > 1:
                    corr = np.corrcoef(
                        returns[sym1][:min_len],
                        returns[sym2][:min_len]
                    )[0, 1]
                    corr_matrix[sym1][sym2] = corr if not np.isnan(corr) else 0.0
                else:
                    corr_matrix[sym1][sym2] = 0.0

        return corr_matrix

    @staticmethod
    def calculate_efficient_frontier(
            expected_returns: List[float],
            cov_matrix: np.ndarray,
            risk_free_rate: float = 0.03
    ) -> Dict[str, Any]:
        """Calculate Markowitz efficient frontier"""
        try:
            import cvxpy as cp
        except ImportError:
            return {"error": "cvxpy not available for portfolio optimization"}

        n_assets = len(expected_returns)

        target_returns = np.linspace(min(expected_returns), max(expected_returns), 20)
        frontier = []

        for ret in target_returns:
            weights = cp.Variable(n_assets)
            portfolio_return = cp.matmul(weights, expected_returns)
            portfolio_risk = cp.quad_form(weights, cov_matrix)

            constraints = [
                cp.sum(weights) == 1,
                weights >= 0,
                portfolio_return >= ret
            ]

            problem = cp.Problem(cp.Minimize(portfolio_risk), constraints)
            problem.solve()

            if problem.status == cp.OPTIMAL:
                frontier.append({
                    "return": ret,
                    "risk": np.sqrt(portfolio_risk.value),
                    "weights": weights.value.tolist()
                })

        weights = cp.Variable(n_assets)
        portfolio_return = cp.matmul(weights, expected_returns)
        portfolio_risk = cp.quad_form(weights, cov_matrix)
        sharpe_ratio = (portfolio_return - risk_free_rate) / cp.sqrt(portfolio_risk)

        constraints = [cp.sum(weights) == 1, weights >= 0]
        problem = cp.Problem(cp.Maximize(sharpe_ratio), constraints)
        problem.solve()

        max_sharpe = {
            "return": portfolio_return.value,
            "risk": np.sqrt(portfolio_risk.value),
            "weights": weights.value.tolist(),
            "sharpe_ratio": sharpe_ratio.value
        }

        return {
            "efficient_frontier": frontier,
            "max_sharpe_portfolio": max_sharpe
        }
