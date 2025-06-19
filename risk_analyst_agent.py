import logging
import sys
import numpy as np
from datetime import datetime, timezone
from typing import Dict, Any, List

# Python version compatibility check
if sys.version_info < (3, 7):
    raise RuntimeError("This application requires Python 3.7 or higher")

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


class RiskAnalysisTool(Tool):
    """Tool for comprehensive risk analysis"""

    def __init__(self):
        super().__init__(
            name="risk_analyzer",
            description="Perform comprehensive risk analysis including market risk, credit risk, liquidity risk, and portfolio risk metrics"
        )

    async def call(self, market_data: Dict[str, Any], portfolio_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Perform comprehensive risk analysis
        Args:
            market_data: Market data from data collection agents
            portfolio_data: Optional portfolio composition data
        Returns:
            Comprehensive risk analysis results
        """
        try:
            logger.info("Starting comprehensive risk analysis")
            if market_data.get('status') != 'success':
                return {
                    "status": "error",
                    "error": "Invalid market data provided",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }

            data = market_data.get('data', {})
            risk_results = {}

            # Analyze risk for each symbol
            for symbol, symbol_data in data.items():
                if symbol.startswith('market_') or symbol.startswith('sector_'):
                    continue
                if isinstance(symbol_data, dict) and not symbol_data.get('error'):
                    logger.info(f"Analyzing risk for {symbol}")
                    symbol_risk = await self._analyze_symbol_risk(symbol, symbol_data)
                    risk_results[symbol] = symbol_risk

            # Market-wide risk analysis
            market_risk = await self._analyze_market_risk(data)

            # Sector risk analysis
            sector_risk = await self._analyze_sector_risk(risk_results)

            # Portfolio risk analysis (if portfolio data provided)
            portfolio_risk = None
            if portfolio_data:
                portfolio_risk = await self._analyze_portfolio_risk(risk_results, portfolio_data)

            # Systemic risk indicators
            systemic_risk = await self._analyze_systemic_risk(data, market_data)

            return {
                "status": "success",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "symbol_risk_analysis": risk_results,
                "market_risk_analysis": market_risk,
                "sector_risk_analysis": sector_risk,
                "portfolio_risk_analysis": portfolio_risk,
                "systemic_risk_analysis": systemic_risk,
                "risk_summary": self._generate_risk_summary(risk_results, market_risk)
            }

        except Exception as e:
            error_msg = f"Error in risk analysis: {str(e)}"
            logger.error(error_msg)
            return {
                "status": "error",
                "error": error_msg,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

    async def _analyze_symbol_risk(self, symbol: str, symbol_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive risk analysis for a single symbol"""
        try:
            price_data = symbol_data.get('price_data', [])
            financial_metrics = symbol_data.get('financial_metrics', {})
            current_price = symbol_data.get('current_price', 0)

            risk_analysis = {
                "symbol": symbol,
                "current_price": current_price
            }

            # Market Risk Analysis
            market_risk = self._calculate_market_risk(price_data, financial_metrics)
            risk_analysis["market_risk"] = market_risk

            # Volatility Risk
            volatility_risk = self._calculate_volatility_risk(price_data)
            risk_analysis["volatility_risk"] = volatility_risk

            # Liquidity Risk
            liquidity_risk = self._calculate_liquidity_risk(symbol_data)
            risk_analysis["liquidity_risk"] = liquidity_risk

            # Credit Risk (based on available fundamental data)
            credit_risk = self._calculate_credit_risk(financial_metrics)
            risk_analysis["credit_risk"] = credit_risk

            # Concentration Risk
            concentration_risk = self._calculate_concentration_risk(financial_metrics)
            risk_analysis["concentration_risk"] = concentration_risk

            # Value at Risk (VaR)
            var_analysis = self._calculate_var(price_data)
            risk_analysis["value_at_risk"] = var_analysis

            # Overall risk score
            overall_risk = self._calculate_overall_risk_score(risk_analysis)
            risk_analysis["overall_risk_score"] = overall_risk

            return risk_analysis

        except Exception as e:
            return {"error": f"Error analyzing risk for {symbol}: {str(e)}"}

    def _calculate_market_risk(self, price_data: List[Dict[str, Any]], financial_metrics: Dict[str, Any]) -> Dict[
        str, Any]:
        """Calculate market risk metrics"""
        try:
            import pandas as pd

            beta = financial_metrics.get('beta')
            market_risk = {
                "beta": beta,
                "beta_interpretation": self._interpret_beta(beta)
            }

            # Calculate correlation with market if we have price data
            if price_data and len(price_data) > 20:
                df = pd.DataFrame(price_data)
                returns = df['Close'].pct_change().dropna()

                # Market correlation (simplified - using price volatility as proxy)
                volatility = returns.std() * np.sqrt(252)  # Annualized volatility
                market_risk["annualized_volatility"] = volatility
                market_risk["volatility_level"] = self._categorize_volatility(volatility)

                # Drawdown analysis
                cumulative_returns = (1 + returns).cumprod()
                rolling_max = cumulative_returns.expanding().max()
                drawdown = (cumulative_returns - rolling_max) / rolling_max
                max_drawdown = drawdown.min()
                market_risk["max_drawdown"] = max_drawdown
                market_risk["drawdown_risk"] = self._categorize_drawdown_risk(max_drawdown)

            return market_risk

        except Exception as e:
            return {"error": f"Error calculating market risk: {str(e)}"}

    def _calculate_volatility_risk(self, price_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate volatility-based risk metrics"""
        try:
            import pandas as pd

            if not price_data or len(price_data) < 10:
                return {"error": "Insufficient price data for volatility analysis"}

            df = pd.DataFrame(price_data)
            closes = df['Close']
            returns = closes.pct_change().dropna()

            # Basic volatility metrics
            daily_vol = returns.std()
            annualized_vol = daily_vol * np.sqrt(252)

            # Rolling volatility
            rolling_vol = returns.rolling(window=10).std()
            vol_of_vol = rolling_vol.std()  # Volatility of volatility

            # Volatility clustering
            vol_clustering = self._detect_volatility_clustering(returns)

            volatility_risk = {
                "daily_volatility": daily_vol,
                "annualized_volatility": annualized_vol,
                "volatility_of_volatility": vol_of_vol,
                "volatility_clustering": vol_clustering,
                "volatility_regime": self._determine_volatility_regime(annualized_vol),
                "volatility_trend": self._analyze_volatility_trend(rolling_vol)
            }

            return volatility_risk

        except Exception as e:
            return {"error": f"Error calculating volatility risk: {str(e)}"}

    def _calculate_liquidity_risk(self, symbol_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate liquidity risk metrics"""
        try:
            trading_metrics = symbol_data.get('trading_metrics', {})
            price_data = symbol_data.get('price_data', [])

            volume = trading_metrics.get('volume', 0)
            avg_volume = trading_metrics.get('avg_volume', 0)
            market_cap = symbol_data.get('market_cap', 0)

            liquidity_risk = {
                "current_volume": volume,
                "average_volume": avg_volume,
                "market_cap": market_cap
            }

            # Volume analysis
            if avg_volume and volume:
                volume_ratio = volume / avg_volume
                liquidity_risk["volume_ratio"] = volume_ratio
                liquidity_risk["volume_assessment"] = self._assess_volume_liquidity(volume_ratio)

            # Market cap liquidity assessment
            if market_cap:
                liquidity_risk["market_cap_liquidity"] = self._assess_market_cap_liquidity(market_cap)

            # Bid-ask spread analysis (if available)
            bid = trading_metrics.get('bid')
            ask = trading_metrics.get('ask')
            if bid and ask and bid > 0:
                spread = (ask - bid) / bid
                liquidity_risk["bid_ask_spread"] = spread
                liquidity_risk["spread_assessment"] = self._assess_spread_liquidity(spread)

            # Price impact estimation
            if price_data and len(price_data) > 5:
                price_impact = self._estimate_price_impact(price_data)
                liquidity_risk["estimated_price_impact"] = price_impact

            # Overall liquidity score
            liquidity_risk["liquidity_score"] = self._calculate_liquidity_score(liquidity_risk)

            return liquidity_risk

        except Exception as e:
            return {"error": f"Error calculating liquidity risk: {str(e)}"}

    def _calculate_credit_risk(self, financial_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate credit risk indicators"""
        try:
            # Basic credit risk assessment based on available metrics
            market_cap = financial_metrics.get('market_cap', 0)
            beta = financial_metrics.get('beta')
            pe_ratio = financial_metrics.get('pe_ratio')

            credit_risk = {
                "market_cap": market_cap,
                "beta": beta,
                "pe_ratio": pe_ratio
            }

            # Size-based credit assessment
            if market_cap:
                credit_risk["size_risk"] = self._assess_size_credit_risk(market_cap)

            # Volatility-based credit proxy
            if beta:
                credit_risk["volatility_credit_risk"] = self._assess_volatility_credit_risk(beta)

            # Valuation-based risk
            if pe_ratio:
                credit_risk["valuation_risk"] = self._assess_valuation_credit_risk(pe_ratio)

            # Overall credit risk score (simplified)
            credit_risk["credit_risk_score"] = self._calculate_credit_risk_score(credit_risk)

            return credit_risk

        except Exception as e:
            return {"error": f"Error calculating credit risk: {str(e)}"}

    def _calculate_concentration_risk(self, financial_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate concentration risk"""
        try:
            # Basic concentration risk based on company size and sector
            market_cap = financial_metrics.get('market_cap', 0)

            concentration_risk = {
                "market_cap": market_cap
            }

            # Single-name concentration risk
            if market_cap:
                if market_cap > 100_000_000_000:  # > $100B
                    concentration_risk["single_name_risk"] = "Low (Mega Cap)"
                elif market_cap > 10_000_000_000:  # > $10B
                    concentration_risk["single_name_risk"] = "Moderate (Large Cap)"
                elif market_cap > 2_000_000_000:  # > $2B
                    concentration_risk["single_name_risk"] = "High (Mid Cap)"
                else:
                    concentration_risk["single_name_risk"] = "Very High (Small Cap)"

            # Sector concentration (would need portfolio context)
            concentration_risk["sector_concentration_note"] = "Requires portfolio-level analysis"

            return concentration_risk

        except Exception as e:
            return {"error": f"Error calculating concentration risk: {str(e)}"}

    def _calculate_var(self, price_data: List[Dict[str, Any]], confidence_level: float = 0.05) -> Dict[str, Any]:
        """Calculate Value at Risk (VaR)"""
        try:
            import pandas as pd

            if not price_data or len(price_data) < 30:
                return {"error": "Insufficient data for VaR calculation"}

            df = pd.DataFrame(price_data)
            returns = df['Close'].pct_change().dropna()

            # Historical VaR
            var_1d = np.percentile(returns, confidence_level * 100)
            var_1w = var_1d * np.sqrt(5)  # 5-day VaR
            var_1m = var_1d * np.sqrt(21)  # 21-day VaR

            # Parametric VaR (assuming normal distribution)
            mean_return = returns.mean()
            std_return = returns.std()
            z_score = np.percentile(np.random.normal(0, 1, 10000), confidence_level * 100)
            parametric_var_1d = mean_return + z_score * std_return

            # Expected Shortfall (Conditional VaR)
            tail_returns = returns[returns <= var_1d]
            expected_shortfall = tail_returns.mean() if len(tail_returns) > 0 else var_1d

            var_analysis = {
                "confidence_level": confidence_level,
                "historical_var_1d": var_1d,
                "historical_var_1w": var_1w,
                "historical_var_1m": var_1m,
                "parametric_var_1d": parametric_var_1d,
                "expected_shortfall_1d": expected_shortfall,
                "var_interpretation": self._interpret_var(var_1d)
            }

            return var_analysis

        except Exception as e:
            return {"error": f"Error calculating VaR: {str(e)}"}

    def _calculate_overall_risk_score(self, risk_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall risk score"""
        try:
            score = 50  # Neutral baseline
            factors = []

            # Market risk contribution
            market_risk = risk_analysis.get('market_risk', {})
            beta = market_risk.get('beta')
            if beta:
                if beta > 1.5:
                    score += 15
                    factors.append("High Beta (+15)")
                elif beta > 1.2:
                    score += 8
                    factors.append("Moderate High Beta (+8)")
                elif beta < 0.8:
                    score -= 5
                    factors.append("Low Beta (-5)")

            # Volatility risk contribution
            volatility_risk = risk_analysis.get('volatility_risk', {})
            annualized_vol = volatility_risk.get('annualized_volatility')
            if annualized_vol:
                if annualized_vol > 0.4:  # > 40%
                    score += 20
                    factors.append("Very High Volatility (+20)")
                elif annualized_vol > 0.25:  # > 25%
                    score += 12
                    factors.append("High Volatility (+12)")
                elif annualized_vol < 0.15:  # < 15%
                    score -= 8
                    factors.append("Low Volatility (-8)")

            # Liquidity risk contribution
            liquidity_risk = risk_analysis.get('liquidity_risk', {})
            liquidity_score = liquidity_risk.get('liquidity_score', 50)
            liquidity_contribution = (100 - liquidity_score) / 10  # Convert to risk score
            score += liquidity_contribution
            factors.append(f"Liquidity Risk ({liquidity_contribution:+.1f})")

            # Credit risk contribution
            credit_risk = risk_analysis.get('credit_risk', {})
            credit_score = credit_risk.get('credit_risk_score', 50)
            credit_contribution = (credit_score - 50) / 5  # Scale contribution
            score += credit_contribution
            factors.append(f"Credit Risk ({credit_contribution:+.1f})")

            # VaR contribution
            var_analysis = risk_analysis.get('value_at_risk', {})
            var_1d = var_analysis.get('historical_var_1d')
            if var_1d:
                var_contribution = abs(var_1d) * 200  # Scale VaR to score
                score += min(15, var_contribution)
                factors.append(f"VaR Risk (+{min(15, var_contribution):.1f})")

            # Normalize score
            final_score = max(0, min(100, score))

            # Risk categories
            if final_score >= 80:
                risk_level = "very_high"
            elif final_score >= 65:
                risk_level = "high"
            elif final_score >= 50:
                risk_level = "moderate"
            elif final_score >= 35:
                risk_level = "low"
            else:
                risk_level = "very_low"

            return {
                "overall_risk_score": final_score,
                "risk_level": risk_level,
                "contributing_factors": factors,
                "risk_assessment": self._assess_risk_level(final_score)
            }

        except Exception as e:
            return {"error": f"Error calculating overall risk score: {str(e)}"}

    # Helper methods for risk analysis
    def _interpret_beta(self, beta) -> str:
        """Interpret beta values"""
        if not beta:
            return "Unknown"
        if beta < 0:
            return "Negative correlation with market"
        elif beta < 0.5:
            return "Very low market sensitivity"
        elif beta < 0.8:
            return "Low market sensitivity"
        elif beta <= 1.2:
            return "Market-like sensitivity"
        elif beta <= 1.5:
            return "High market sensitivity"
        else:
            return "Very high market sensitivity"

    def _categorize_volatility(self, volatility: float) -> str:
        """Categorize volatility levels"""
        if volatility > 0.5:
            return "Extremely High"
        elif volatility > 0.3:
            return "Very High"
        elif volatility > 0.2:
            return "High"
        elif volatility > 0.15:
            return "Moderate"
        else:
            return "Low"

    def _categorize_drawdown_risk(self, max_drawdown: float) -> str:
        """Categorize drawdown risk"""
        abs_drawdown = abs(max_drawdown)
        if abs_drawdown > 0.5:
            return "Extreme"
        elif abs_drawdown > 0.3:
            return "High"
        elif abs_drawdown > 0.15:
            return "Moderate"
        else:
            return "Low"

    def _detect_volatility_clustering(self, returns) -> Dict[str, Any]:
        """Detect volatility clustering patterns"""
        try:
            # Simple volatility clustering detection
            abs_returns = abs(returns)
            autocorr_1 = abs_returns.autocorr(lag=1)
            autocorr_5 = abs_returns.autocorr(lag=5)

            clustering_detected = autocorr_1 > 0.1 or autocorr_5 > 0.05

            return {
                "clustering_detected": clustering_detected,
                "autocorr_lag1": autocorr_1,
                "autocorr_lag5": autocorr_5,
                "interpretation": "Volatility clustering present" if clustering_detected else "No significant clustering"
            }

        except Exception:
            return {"error": "Could not detect volatility clustering"}

    def _determine_volatility_regime(self, annualized_vol: float) -> str:
        """Determine volatility regime"""
        if annualized_vol > 0.4:
            return "crisis"
        elif annualized_vol > 0.25:
            return "high_vol"
        elif annualized_vol > 0.15:
            return "normal"
        else:
            return "low_vol"

    def _analyze_volatility_trend(self, rolling_vol) -> str:
        """Analyze volatility trend"""
        try:
            if len(rolling_vol.dropna()) < 5:
                return "insufficient_data"

            recent_vol = rolling_vol.dropna().tail(5).mean()
            earlier_vol = rolling_vol.dropna().head(5).mean()

            change_pct = (recent_vol - earlier_vol) / earlier_vol * 100

            if change_pct > 20:
                return "increasing"
            elif change_pct < -20:
                return "decreasing"
            else:
                return "stable"

        except Exception:
            return "error"

    def _assess_volume_liquidity(self, volume_ratio: float) -> str:
        """Assess liquidity based on volume ratio"""
        if volume_ratio > 2:
            return "High volume - Good liquidity"
        elif volume_ratio > 1.5:
            return "Above average volume - Good liquidity"
        elif volume_ratio > 0.5:
            return "Normal volume - Fair liquidity"
        else:
            return "Low volume - Poor liquidity"

    def _assess_market_cap_liquidity(self, market_cap: float) -> str:
        """Assess liquidity based on market cap"""
        if market_cap > 10_000_000_000:
            return "Large cap - Excellent liquidity"
        elif market_cap > 2_000_000_000:
            return "Mid cap - Good liquidity"
        elif market_cap > 300_000_000:
            return "Small cap - Fair liquidity"
        else:
            return "Micro cap - Poor liquidity"

    def _assess_spread_liquidity(self, spread: float) -> str:
        """Assess liquidity based on bid-ask spread"""
        if spread < 0.001:  # < 0.1%
            return "Tight spread - Excellent liquidity"
        elif spread < 0.005:  # < 0.5%
            return "Narrow spread - Good liquidity"
        elif spread < 0.02:  # < 2%
            return "Wide spread - Fair liquidity"
        else:
            return "Very wide spread - Poor liquidity"

    def _estimate_price_impact(self, price_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Estimate price impact of trades"""
        try:
            import pandas as pd

            df = pd.DataFrame(price_data)

            # Simple price impact estimation based on price and volume volatility
            price_volatility = df['Close'].pct_change().std()
            volume_data = df.get('Volume', pd.Series([1] * len(df)))
            volume_volatility = volume_data.pct_change().std()

            # Estimated impact for different trade sizes (as % of average volume)
            impact_1pct = price_volatility * 0.1  # 1% of avg volume
            impact_5pct = price_volatility * 0.3  # 5% of avg volume
            impact_10pct = price_volatility * 0.6  # 10% of avg volume

            return {
                "impact_1pct_volume": impact_1pct,
                "impact_5pct_volume": impact_5pct,
                "impact_10pct_volume": impact_10pct,
                "interpretation": self._interpret_price_impact(impact_5pct)
            }

        except Exception:
            return {"error": "Could not estimate price impact"}

    def _calculate_liquidity_score(self, liquidity_risk: Dict[str, Any]) -> float:
        """Calculate overall liquidity score"""
        score = 50  # Neutral baseline

        # Market cap contribution
        market_cap = liquidity_risk.get('market_cap', 0)
        if market_cap > 10_000_000_000:
            score += 25
        elif market_cap > 2_000_000_000:
            score += 15
        elif market_cap > 300_000_000:
            score += 5
        else:
            score -= 15

        # Volume contribution
        volume_ratio = liquidity_risk.get('volume_ratio', 1)
        if volume_ratio > 2:
            score += 15
        elif volume_ratio > 1.5:
            score += 10
        elif volume_ratio < 0.5:
            score -= 10

        # Spread contribution
        spread = liquidity_risk.get('bid_ask_spread')
        if spread:
            if spread < 0.005:
                score += 10
            elif spread > 0.02:
                score -= 15

        return max(0, min(100, score))

    def _assess_size_credit_risk(self, market_cap: float) -> str:
        """Assess credit risk based on company size"""
        if market_cap > 50_000_000_000:
            return "Very Low (Mega Cap)"
        elif market_cap > 10_000_000_000:
            return "Low (Large Cap)"
        elif market_cap > 2_000_000_000:
            return "Moderate (Mid Cap)"
        else:
            return "High (Small Cap)"

    def _assess_volatility_credit_risk(self, beta: float) -> str:
        """Assess credit risk based on volatility"""
        if beta > 2.0:
            return "Very High Volatility Risk"
        elif beta > 1.5:
            return "High Volatility Risk"
        elif beta > 1.2:
            return "Moderate Volatility Risk"
        else:
            return "Low Volatility Risk"

    def _assess_valuation_credit_risk(self, pe_ratio: float) -> str:
        """Assess credit risk based on valuation"""
        if pe_ratio < 0:
            return "Negative Earnings - High Risk"
        elif pe_ratio > 50:
            return "Very High Valuation - High Risk"
        elif pe_ratio > 25:
            return "High Valuation - Moderate Risk"
        else:
            return "Reasonable Valuation - Low Risk"

    def _calculate_credit_risk_score(self, credit_risk: Dict[str, Any]) -> float:
        """Calculate simplified credit risk score"""
        score = 50  # Neutral baseline

        # Size contribution (lower is better for credit risk)
        market_cap = credit_risk.get('market_cap', 0)
        if market_cap > 50_000_000_000:
            score -= 20  # Very low credit risk
        elif market_cap > 10_000_000_000:
            score -= 10  # Low credit risk
        elif market_cap < 300_000_000:
            score += 20  # High credit risk

        # Beta contribution
        beta = credit_risk.get('beta')
        if beta and beta > 1.5:
            score += 15  # High volatility = higher credit risk
        elif beta and beta < 0.8:
            score -= 5  # Low volatility = lower credit risk

        return max(0, min(100, score))

    def _interpret_var(self, var_1d: float) -> str:
        """Interpret VaR values"""
        abs_var = abs(var_1d)
        if abs_var > 0.1:  # > 10% daily VaR
            return "Extreme Risk"
        elif abs_var > 0.05:  # > 5% daily VaR
            return "Very High Risk"
        elif abs_var > 0.03:  # > 3% daily VaR
            return "High Risk"
        elif abs_var > 0.02:  # > 2% daily VaR
            return "Moderate Risk"
        else:
            return "Low Risk"

    def _interpret_price_impact(self, impact: float) -> str:
        """Interpret price impact estimates"""
        if impact > 0.05:  # > 5%
            return "Very High Impact - Illiquid"
        elif impact > 0.02:  # > 2%
            return "High Impact - Low Liquidity"
        elif impact > 0.01:  # > 1%
            return "Moderate Impact - Fair Liquidity"
        else:
            return "Low Impact - Good Liquidity"

    def _assess_risk_level(self, score: float) -> str:
        """Assess overall risk level"""
        if score >= 80:
            return "Very High Risk - Significant caution advised"
        elif score >= 65:
            return "High Risk - Careful monitoring required"
        elif score >= 50:
            return "Moderate Risk - Standard risk management"
        elif score >= 35:
            return "Low Risk - Conservative investment"
        else:
            return "Very Low Risk - Capital preservation focus"

    async def _analyze_market_risk(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze overall market risk conditions"""
        try:
            market_summary = data.get('market_summary', {})
            market_risk = {
                "vix_level": None,
                "market_correlation": None,
                "systemic_risk_indicators": []
            }

            # VIX analysis
            if '^VIX' in market_summary:
                vix_data = market_summary['^VIX']
                if not vix_data.get('error'):
                    vix_current = vix_data.get('current', 0)
                    market_risk["vix_level"] = vix_current
                    market_risk["vix_interpretation"] = self._interpret_vix(vix_current)

            # Market indices correlation
            indices = ['^GSPC', '^DJI', '^IXIC']
            index_changes = []
            for index in indices:
                if index in market_summary and not market_summary[index].get('error'):
                    change_pct = market_summary[index].get('change_percent', 0)
                    index_changes.append(change_pct)

            if index_changes:
                avg_market_change = np.mean(index_changes)
                market_volatility = np.std(index_changes)
                market_risk["average_market_change"] = avg_market_change
                market_risk["index_volatility"] = market_volatility
                market_risk[
                    "market_trend"] = "positive" if avg_market_change > 0.5 else "negative" if avg_market_change < -0.5 else "neutral"

            return market_risk

        except Exception as e:
            return {"error": f"Error analyzing market risk: {str(e)}"}

    async def _analyze_sector_risk(self, symbol_risks: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze risk by sector"""
        try:
            # This would require sector information from the symbols
            # Simplified implementation
            total_symbols = len([r for r in symbol_risks.values() if not r.get('error')])
            high_risk_symbols = len([
                r for r in symbol_risks.values()
                if not r.get('error') and r.get('overall_risk_score', {}).get('overall_risk_score', 50) > 70
            ])

            sector_risk = {
                "total_symbols_analyzed": total_symbols,
                "high_risk_symbols": high_risk_symbols,
                "high_risk_percentage": (high_risk_symbols / total_symbols * 100) if total_symbols > 0 else 0,
                "note": "Detailed sector risk analysis requires sector classification data"
            }

            return sector_risk

        except Exception as e:
            return {"error": f"Error analyzing sector risk: {str(e)}"}

    async def _analyze_portfolio_risk(self, symbol_risks: Dict[str, Any], portfolio_data: Dict[str, Any]) -> Dict[
        str, Any]:
        """Analyze portfolio-level risk"""
        try:
            # Portfolio risk analysis would require position sizes and correlations
            # This is a simplified implementation
            portfolio_symbols = portfolio_data.get('holdings', [])
            portfolio_weights = portfolio_data.get('weights', {})

            weighted_risk_score = 0
            total_weight = 0

            for symbol in portfolio_symbols:
                if symbol in symbol_risks and symbol in portfolio_weights:
                    symbol_risk = symbol_risks[symbol]
                    if not symbol_risk.get('error'):
                        risk_score = symbol_risk.get('overall_risk_score', {}).get('overall_risk_score', 50)
                        weight = portfolio_weights[symbol]
                        weighted_risk_score += risk_score * weight
                        total_weight += weight

            portfolio_risk_score = weighted_risk_score / total_weight if total_weight > 0 else 50

            portfolio_risk = {
                "portfolio_risk_score": portfolio_risk_score,
                "portfolio_risk_level": self._assess_risk_level(portfolio_risk_score),
                "concentration_risk": self._assess_portfolio_concentration(portfolio_weights),
                "diversification_score": self._calculate_diversification_score(portfolio_weights)
            }

            return portfolio_risk

        except Exception as e:
            return {"error": f"Error analyzing portfolio risk: {str(e)}"}

    async def _analyze_systemic_risk(self, data: Dict[str, Any], market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze systemic risk indicators"""
        try:
            systemic_indicators = []

            # VIX analysis
            market_summary = data.get('market_summary', {})
            if '^VIX' in market_summary:
                vix_data = market_summary['^VIX']
                if not vix_data.get('error'):
                    vix_level = vix_data.get('current', 0)
                    if vix_level > 30:
                        systemic_indicators.append("High VIX indicating market stress")
                    elif vix_level > 20:
                        systemic_indicators.append("Elevated VIX indicating uncertainty")

            # Market correlation analysis
            # (Would require more sophisticated correlation analysis in practice)

            # Economic indicators integration
            # (Would integrate with economic data from EconomicIndicatorsAgent)

            systemic_risk = {
                "systemic_risk_indicators": systemic_indicators,
                "systemic_risk_level": "elevated" if len(systemic_indicators) > 2 else "moderate" if len(
                    systemic_indicators) > 0 else "low",
                "note": "Comprehensive systemic risk analysis requires additional economic and correlation data"
            }

            return systemic_risk

        except Exception as e:
            return {"error": f"Error analyzing systemic risk: {str(e)}"}

    def _generate_risk_summary(self, symbol_risks: Dict[str, Any], market_risk: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive risk summary"""
        try:
            total_symbols = len(symbol_risks)
            successful_analyses = len([r for r in symbol_risks.values() if not r.get('error')])

            # Risk distribution
            risk_distribution = {"very_low": 0, "low": 0, "moderate": 0, "high": 0, "very_high": 0}
            for symbol_risk in symbol_risks.values():
                if not symbol_risk.get('error'):
                    risk_level = symbol_risk.get('overall_risk_score', {}).get('risk_level', 'moderate')
                    if risk_level in risk_distribution:
                        risk_distribution[risk_level] += 1

            # Average risk score
            risk_scores = [
                r.get('overall_risk_score', {}).get('overall_risk_score', 50)
                for r in symbol_risks.values() if not r.get('error')
            ]
            avg_risk_score = np.mean(risk_scores) if risk_scores else 50

            return {
                "total_symbols_analyzed": total_symbols,
                "successful_risk_analyses": successful_analyses,
                "average_risk_score": avg_risk_score,
                "risk_distribution": risk_distribution,
                "market_risk_level": market_risk.get('vix_interpretation', 'Unknown'),
                "analysis_timestamp": datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            return {"error": f"Error generating risk summary: {str(e)}"}

    def _interpret_vix(self, vix_level: float) -> str:
        """Interpret VIX levels"""
        if vix_level > 40:
            return "Extreme Fear - Crisis Level"
        elif vix_level > 30:
            return "High Fear - Significant Stress"
        elif vix_level > 20:
            return "Elevated Fear - Market Uncertainty"
        elif vix_level > 15:
            return "Normal Volatility"
        else:
            return "Low Volatility - Complacency Risk"

    def _assess_portfolio_concentration(self, weights: Dict[str, float]) -> str:
        """Assess portfolio concentration risk"""
        if not weights:
            return "Unknown"

        max_weight = max(weights.values())
        if max_weight > 0.3:
            return "High Concentration Risk"
        elif max_weight > 0.2:
            return "Moderate Concentration Risk"
        else:
            return "Low Concentration Risk"

    def _calculate_diversification_score(self, weights: Dict[str, float]) -> float:
        """Calculate diversification score"""
        if not weights:
            return 0

        # Herfindahl-Hirschman Index (HHI) approach
        hhi = sum(w ** 2 for w in weights.values())

        # Convert to diversification score (0-100)
        # Perfect diversification (equal weights) = 100
        # Complete concentration = 0
        num_holdings = len(weights)
        min_hhi = 1 / num_holdings if num_holdings > 0 else 1
        diversification_score = (1 - (hhi - min_hhi) / (1 - min_hhi)) * 100

        return max(0, min(100, diversification_score))


class RiskAnalystAgent(Agent):
    """Specialized agent for comprehensive risk analysis"""

    def __init__(self, gcp_services, config):
        super().__init__(
            model="gemini-2.0-flash-exp",
            name="risk_analyst_expert",
            description="Expert risk analyst specializing in market risk, credit risk, and portfolio risk management",
            instructions="""You are a senior risk analyst with expertise in:
1. Market risk assessment and VaR calculations
2. Credit risk analysis and default probability estimation
3. Liquidity risk evaluation and market impact analysis
4. Portfolio risk optimization and concentration analysis
5. Systemic risk identification and stress testing
6. Risk-adjusted performance measurement

Your primary responsibilities:
- Perform comprehensive risk analysis across multiple risk dimensions
- Calculate Value-at-Risk and stress test scenarios
- Assess liquidity and market impact for different position sizes
- Identify concentration risks and correlation breakdowns
- Provide risk-adjusted position sizing recommendations
- Monitor systemic risk indicators and market stress signals

Focus on providing analysis that considers:
- Multi-dimensional risk factors (market, credit, liquidity, operational)
- Tail risk events and extreme scenarios
- Risk correlation and diversification benefits
- Dynamic risk management and hedging strategies
- Regulatory capital requirements and compliance""",
            tools=[RiskAnalysisTool()]
        )

        self.gcp_services = gcp_services
        self.config = config
        logger.info("RiskAnalystAgent initialized")

    async def analyze_portfolio_risk(self, market_data: Dict[str, Any], portfolio_data: Dict[str, Any] = None) -> Dict[
        str, Any]:
        """Main method for comprehensive risk analysis"""
        try:
            logger.info("Starting comprehensive risk analysis")

            risk_analysis = await self.tools[0].call(
                market_data=market_data,
                portfolio_data=portfolio_data
            )

            # Store results
            if self.gcp_services:
                if hasattr(self.gcp_services, 'store_risk_analysis'):
                    await self.gcp_services.store_risk_analysis(risk_analysis)
                if hasattr(self.gcp_services, 'store_bigquery_risk_data'):
                    await self.gcp_services.store_bigquery_risk_data(risk_analysis)

            # Add metadata
            risk_analysis["analysis_metadata"] = {
                "agent": self.name,
                "analysis_time": datetime.now(timezone.utc).isoformat(),
                "portfolio_included": portfolio_data is not None,
                "quality_score": self._calculate_risk_quality_score(risk_analysis)
            }

            logger.info("Risk analysis completed successfully")
            return risk_analysis

        except Exception as e:
            error_msg = f"Error in risk analysis: {str(e)}"
            logger.error(error_msg)
            return {
                "status": "error",
                "error": error_msg,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "agent": self.name
            }

    def _calculate_risk_quality_score(self, risk_analysis: Dict[str, Any]) -> float:
        """Calculate quality score for risk analysis"""
        if risk_analysis.get('status') != 'success':
            return 0.0

        symbol_risks = risk_analysis.get('symbol_risk_analysis', {})
        total_symbols = len(symbol_risks)
        if total_symbols == 0:
            return 0.0

        # Base score from successful analysis
        successful_analysis = len([s for s in symbol_risks.values() if not s.get('error')])
        base_score = (successful_analysis / total_symbols) * 40

        # VaR calculation score
        var_analyses = len([
            s for s in symbol_risks.values()
            if not s.get('error') and s.get('value_at_risk') and not s['value_at_risk'].get('error')
        ])
        var_score = (var_analyses / total_symbols * 25) if total_symbols > 0 else 0

        # Market risk analysis score
        market_risk_score = 15 if risk_analysis.get('market_risk_analysis') and not risk_analysis[
            'market_risk_analysis'].get('error') else 0

        # Portfolio analysis bonus
        portfolio_bonus = 10 if risk_analysis.get('portfolio_risk_analysis') else 0

        # Systemic risk analysis score
        systemic_score = 10 if risk_analysis.get('systemic_risk_analysis') else 0

        total_score = base_score + var_score + market_risk_score + portfolio_bonus + systemic_score
        return round(min(100.0, total_score), 2)
