import logging
import numpy as np
from datetime import datetime

# Import with proper error handling
try:
    from google.adk.tools import Tool
except ImportError:
    # Fallback implementation
    class Tool:
        def __init__(self, name=None, description=None):
            self.name = name
            self.description = description

        async def call(self, *args, **kwargs):
            raise NotImplementedError("Subclasses must implement call method")

logger = logging.getLogger(__name__)


class MarketRegimeTool(Tool):
    """Tool for market regime identification and analysis"""

    def __init__(self):
        super().__init__(
            name="market_regime_analyzer",
            description="Identify and analyze market regimes including trend, volatility, and correlation regimes"
        )

    async def call(self, market_data: dict[str, any], economic_data: dict[str, any] = None) -> dict[str, any]:
        """
        Identify current market regime and analyze regime transitions
        Args:
            market_data: Market data from data collection agents
            economic_data: Optional economic indicators data
        Returns:
            Market regime analysis results
        """
        try:
            logger.info("Starting market regime analysis")
            if market_data.get('status') != 'success':
                return {
                    "status": "error",
                    "error": "Invalid market data provided",
                    "timestamp": datetime.utcnow().isoformat()
                }

            data = market_data.get('data', {})

            # Market indices analysis
            market_regime = await self._analyze_market_indices_regime(data)
            # Volatility regime analysis
            volatility_regime = await self._analyze_volatility_regime(data)
            # Correlation regime analysis
            correlation_regime = await self._analyze_correlation_regime(data)
            # Sector rotation regime
            sector_regime = await self._analyze_sector_rotation_regime(data)
            # Economic regime integration
            economic_regime = None
            if economic_data:
                economic_regime = await self._analyze_economic_regime(economic_data)

            # Overall regime classification
            overall_regime = await self._classify_overall_regime(
                market_regime, volatility_regime, correlation_regime, sector_regime, economic_regime
            )

            # Regime transition signals
            transition_signals = await self._detect_regime_transitions(
                market_regime, volatility_regime, correlation_regime
            )

            return {
                "status": "success",
                "timestamp": datetime.utcnow().isoformat(),
                "market_trend_regime": market_regime,
                "volatility_regime": volatility_regime,
                "correlation_regime": correlation_regime,
                "sector_rotation_regime": sector_regime,
                "economic_regime": economic_regime,
                "overall_market_regime": overall_regime,
                "regime_transition_signals": transition_signals,
                "regime_summary": self._generate_regime_summary(overall_regime, transition_signals)
            }

        except Exception as e:
            error_msg = f"Error in market regime analysis: {str(e)}"
            logger.error(error_msg)
            return {
                "status": "error",
                "error": error_msg,
                "timestamp": datetime.utcnow().isoformat()
            }

    async def _analyze_market_indices_regime(self, data: dict[str, any]) -> dict[str, any]:
        """Analyze market trend regime based on major indices"""
        try:
            market_summary = data.get('market_summary', {})
            # Major indices to analyze
            indices = {
                '^GSPC': 'S&P 500',
                '^DJI': 'Dow Jones',
                '^IXIC': 'NASDAQ',
                '^RUT': 'Russell 2000'
            }

            index_trends = {}
            trend_signals = []
            for index_symbol, index_name in indices.items():
                if index_symbol in market_summary and not market_summary[index_symbol].get('error'):
                    index_data = market_summary[index_symbol]
                    change_pct = index_data.get('change_percent', 0)
                    # Categorize trend
                    if change_pct > 1:
                        trend = "strong_bullish"
                        trend_signals.append(1)
                    elif change_pct > 0.5:
                        trend = "bullish"
                        trend_signals.append(0.5)
                    elif change_pct > -0.5:
                        trend = "neutral"
                        trend_signals.append(0)
                    elif change_pct > -1:
                        trend = "bearish"
                        trend_signals.append(-0.5)
                    else:
                        trend = "strong_bearish"
                        trend_signals.append(-1)
                    index_trends[index_name] = {
                        "change_percent": change_pct,
                        "trend": trend,
                        "symbol": index_symbol
                    }

            # Overall market trend regime
            if trend_signals:
                avg_trend_signal = np.mean(trend_signals)
                if avg_trend_signal > 0.3:
                    market_trend_regime = "bull_market"
                elif avg_trend_signal > 0:
                    market_trend_regime = "mild_bull_market"
                elif avg_trend_signal > -0.3:
                    market_trend_regime = "neutral_market"
                elif avg_trend_signal > -0.6:
                    market_trend_regime = "mild_bear_market"
                else:
                    market_trend_regime = "bear_market"
            else:
                market_trend_regime = "unknown"

            # Trend strength assessment
            trend_strength = abs(avg_trend_signal) if trend_signals else 0
            return {
                "regime": market_trend_regime,
                "trend_strength": trend_strength,
                "average_trend_signal": avg_trend_signal if trend_signals else 0,
                "index_trends": index_trends,
                "regime_confidence": self._calculate_trend_confidence(trend_signals)
            }
        except Exception as e:
            return {"error": f"Error analyzing market trend regime: {str(e)}"}

    async def _analyze_volatility_regime(self, data: dict[str, any]) -> dict[str, any]:
        """Analyze volatility regime based on VIX and market volatility"""
        try:
            market_summary = data.get('market_summary', {})
            volatility_regime = {
                "vix_level": None,
                "vix_regime": None,
                "market_volatility": None
            }

            # VIX analysis
            if '^VIX' in market_summary and not market_summary['^VIX'].get('error'):
                vix_data = market_summary['^VIX']
                vix_level = vix_data.get('current', 0)
                volatility_regime["vix_level"] = vix_level
                # VIX regime classification
                if vix_level > 40:
                    vix_regime = "extreme_fear"
                elif vix_level > 30:
                    vix_regime = "high_volatility"
                elif vix_level > 20:
                    vix_regime = "elevated_volatility"
                elif vix_level > 15:
                    vix_regime = "normal_volatility"
                else:
                    vix_regime = "low_volatility"
                volatility_regime["vix_regime"] = vix_regime

            # Calculate realized volatility from index movements
            index_changes = []
            for index in ['^GSPC', '^DJI', '^IXIC']:
                if index in market_summary and not market_summary[index].get('error'):
                    change_pct = market_summary[index].get('change_percent', 0)
                    index_changes.append(change_pct)

            if index_changes:
                realized_volatility = np.std(index_changes)
                volatility_regime["realized_volatility"] = realized_volatility
                # Realized volatility regime
                if realized_volatility > 2:
                    realized_regime = "high_volatility"
                elif realized_volatility > 1:
                    realized_regime = "moderate_volatility"
                else:
                    realized_regime = "low_volatility"
                volatility_regime["realized_volatility_regime"] = realized_regime

            # Overall volatility regime
            volatility_regime["overall_volatility_regime"] = self._determine_overall_volatility_regime(
                volatility_regime
            )
            return volatility_regime
        except Exception as e:
            return {"error": f"Error analyzing volatility regime: {str(e)}"}

    async def _analyze_correlation_regime(self, data: dict[str, any]) -> dict[str, any]:
        """Analyze correlation regime between assets and sectors"""
        try:
            market_summary = data.get('market_summary', {})
            sector_performance = data.get('sector_performance', {})
            correlation_regime = {
                "index_correlation": None,
                "sector_correlation": None,
                "overall_correlation_regime": None
            }

            # Index correlation analysis
            index_changes = {}
            for index in ['^GSPC', '^DJI', '^IXIC', '^RUT']:
                if index in market_summary and not market_summary[index].get('error'):
                    change_pct = market_summary[index].get('change_percent', 0)
                    index_changes[index] = change_pct

            if len(index_changes) >= 3:
                # Calculate correlation between indices (simplified - single day)
                changes_list = list(index_changes.values())
                correlation_strength = self._estimate_correlation_strength(changes_list)
                correlation_regime["index_correlation"] = correlation_strength
                if correlation_strength > 0.8:
                    index_corr_regime = "high_correlation"
                elif correlation_strength > 0.5:
                    index_corr_regime = "moderate_correlation"
                else:
                    index_corr_regime = "low_correlation"
                correlation_regime["index_correlation_regime"] = index_corr_regime

            # Sector correlation analysis
            if sector_performance and not sector_performance.get('error'):
                sector_changes = []
                for sector, sector_data in sector_performance.items():
                    if isinstance(sector_data, dict) and 'change_percent' in sector_data:
                        sector_changes.append(sector_data['change_percent'])

                if len(sector_changes) >= 5:
                    sector_correlation_strength = self._estimate_correlation_strength(sector_changes)
                    correlation_regime["sector_correlation"] = sector_correlation_strength
                    if sector_correlation_strength > 0.7:
                        sector_corr_regime = "high_sector_correlation"
                    elif sector_correlation_strength > 0.4:
                        sector_corr_regime = "moderate_sector_correlation"
                    else:
                        sector_corr_regime = "low_sector_correlation"
                    correlation_regime["sector_correlation_regime"] = sector_corr_regime

            # Overall correlation regime
            correlation_regime["overall_correlation_regime"] = self._determine_overall_correlation_regime(
                correlation_regime
            )
            return correlation_regime
        except Exception as e:
            return {"error": f"Error analyzing correlation regime: {str(e)}"}

    async def _analyze_sector_rotation_regime(self, data: dict[str, any]) -> dict[str, any]:
        """Analyze sector rotation patterns and regime"""
        try:
            sector_performance = data.get('sector_performance', {})
            if not sector_performance or sector_performance.get('error'):
                return {"error": "No sector performance data available"}

            # Extract sector performance data
            sector_changes = {}
            for sector, sector_data in sector_performance.items():
                if isinstance(sector_data, dict) and 'change_percent' in sector_data:
                    sector_changes[sector] = sector_data['change_percent']

            if len(sector_changes) < 5:
                return {"error": "Insufficient sector data for rotation analysis"}

            # Sort sectors by performance
            sorted_sectors = sorted(sector_changes.items(), key=lambda x: x[1], reverse=True)

            # Identify sector rotation patterns
            rotation_analysis = {
                "top_performing_sectors": sorted_sectors[:3],
                "worst_performing_sectors": sorted_sectors[-3:],
                "performance_spread": sorted_sectors[0][1] - sorted_sectors[-1][1]
            }

            # Classify rotation regime
            performance_spread = rotation_analysis["performance_spread"]
            top_sectors = [sector for sector, _ in sorted_sectors[:3]]

            # Identify regime type based on leading sectors
            if any(sector in ['Technology', 'Communication Services', 'Consumer Discretionary'] for sector in
                   top_sectors):
                if performance_spread > 3:
                    rotation_regime = "strong_growth_rotation"
                else:
                    rotation_regime = "mild_growth_rotation"
            elif any(sector in ['Utilities', 'Consumer Staples', 'Healthcare'] for sector in top_sectors):
                if performance_spread > 3:
                    rotation_regime = "strong_defensive_rotation"
                else:
                    rotation_regime = "mild_defensive_rotation"
            elif any(sector in ['Financial', 'Energy', 'Industrial'] for sector in top_sectors):
                if performance_spread > 3:
                    rotation_regime = "strong_value_rotation"
                else:
                    rotation_regime = "mild_value_rotation"
            else:
                rotation_regime = "mixed_rotation"

            rotation_analysis["rotation_regime"] = rotation_regime
            rotation_analysis[
                "rotation_strength"] = "strong" if performance_spread > 3 else "moderate" if performance_spread > 1.5 else "weak"
            return rotation_analysis
        except Exception as e:
            return {"error": f"Error analyzing sector rotation regime: {str(e)}"}

    async def _analyze_economic_regime(self, economic_data: dict[str, any]) -> dict[str, any]:
        """Analyze economic regime based on economic indicators"""
        try:
            if economic_data.get('status') != 'success':
                return {"error": "Invalid economic data"}

            economic_health = economic_data.get('economic_health_score', {})
            economic_signals = economic_data.get('economic_signals', [])

            # Extract key economic metrics
            economic_regime = {
                "economic_health_score": economic_health.get('overall_score', 50),
                "health_status": economic_health.get('health_status', 'unknown')
            }

            # Analyze economic signals for regime classification
            growth_signals = len([s for s in economic_signals if s.get('type') == 'positive'])
            recession_signals = len([s for s in economic_signals if s.get('type') == 'negative'])
            economic_regime["growth_signals"] = growth_signals
            economic_regime["recession_signals"] = recession_signals

            # Economic regime classification
            health_score = economic_regime["economic_health_score"]
            if health_score >= 70 and growth_signals > recession_signals:
                econ_regime = "expansion"
            elif health_score >= 50 and growth_signals >= recession_signals:
                econ_regime = "stable_growth"
            elif health_score >= 40:
                econ_regime = "slow_growth"
            elif recession_signals > growth_signals:
                econ_regime = "contraction"
            else:
                econ_regime = "uncertain"

            economic_regime["economic_regime"] = econ_regime
            return economic_regime
        except Exception as e:
            return {"error": f"Error analyzing economic regime: {str(e)}"}

    async def _classify_overall_regime(self, market_regime, volatility_regime, correlation_regime,
                                       sector_regime, economic_regime) -> dict[str, any]:
        """Classify overall market regime combining all factors"""
        try:
            # Extract key regime indicators
            trend_regime = market_regime.get('regime', 'unknown')
            vol_regime = volatility_regime.get('overall_volatility_regime', 'unknown')
            corr_regime = correlation_regime.get('overall_correlation_regime', 'unknown')
            rotation_regime = sector_regime.get('rotation_regime', 'unknown')
            econ_regime = economic_regime.get('economic_regime', 'unknown') if economic_regime else 'unknown'

            # Scoring system for overall regime
            regime_scores = {
                "risk_on": 0,
                "risk_off": 0,
                "transition": 0,
                "crisis": 0
            }

            # Market trend contribution
            if 'bull' in trend_regime:
                regime_scores["risk_on"] += 2
            elif 'bear' in trend_regime:
                regime_scores["risk_off"] += 2
            else:
                regime_scores["transition"] += 1

            # Volatility contribution
            if vol_regime in ['low_volatility', 'normal_volatility']:
                regime_scores["risk_on"] += 1
            elif vol_regime in ['high_volatility']:
                regime_scores["risk_off"] += 1
            elif vol_regime in ['extreme_fear']:
                regime_scores["crisis"] += 2

            # Correlation contribution
            if corr_regime == 'low_correlation':
                regime_scores["risk_on"] += 1
            elif corr_regime == 'high_correlation':
                regime_scores["risk_off"] += 1

            # Sector rotation contribution
            if 'growth' in rotation_regime:
                regime_scores["risk_on"] += 1
            elif 'defensive' in rotation_regime:
                regime_scores["risk_off"] += 1
            elif 'value' in rotation_regime:
                regime_scores["transition"] += 1

            # Economic regime contribution
            if econ_regime in ['expansion', 'stable_growth']:
                regime_scores["risk_on"] += 1
            elif econ_regime in ['contraction']:
                regime_scores["risk_off"] += 1
            elif econ_regime in ['uncertain']:
                regime_scores["transition"] += 1

            # Determine overall regime
            max_score = max(regime_scores.values())
            overall_regime = max(regime_scores, key=regime_scores.get)

            # Confidence calculation
            total_score = sum(regime_scores.values())
            confidence = (max_score / total_score * 100) if total_score > 0 else 0

            return {
                "overall_regime": overall_regime,
                "regime_scores": regime_scores,
                "confidence": confidence,
                "regime_strength": "strong" if confidence > 60 else "moderate" if confidence > 40 else "weak",
                "component_regimes": {
                    "trend": trend_regime,
                    "volatility": vol_regime,
                    "correlation": corr_regime,
                    "sector_rotation": rotation_regime,
                    "economic": econ_regime
                }
            }
        except Exception as e:
            return {"error": f"Error classifying overall regime: {str(e)}"}

    async def _detect_regime_transitions(self, market_regime, volatility_regime, correlation_regime) -> dict[str, any]:
        """Detect potential regime transition signals"""
        try:
            transition_signals = []
            transition_score = 0

            # Market trend transition signals
            trend_strength = market_regime.get('trend_strength', 0)
            if trend_strength < 0.3:  # Weak trend strength may indicate transition
                transition_signals.append("Weak market trend strength - potential transition")
                transition_score += 1

            # Volatility transition signals
            vix_level = volatility_regime.get('vix_level', 0)
            if vix_level and 15 < vix_level < 25:  # Transition zone for VIX
                transition_signals.append("VIX in transition zone (15-25)")
                transition_score += 1

            # Correlation breakdown signals
            index_correlation = correlation_regime.get('index_correlation')
            if index_correlation and index_correlation < 0.5:
                transition_signals.append("Low index correlation - potential regime change")
                transition_score += 1

            # Determine transition probability
            if transition_score >= 2:
                transition_probability = "high"
            elif transition_score >= 1:
                transition_probability = "moderate"
            else:
                transition_probability = "low"

            return {
                "transition_signals": transition_signals,
                "transition_score": transition_score,
                "transition_probability": transition_probability,
                "monitoring_indicators": [
                    "VIX levels and volatility spikes",
                    "Cross-asset correlation changes",
                    "Sector rotation patterns",
                    "Economic indicator shifts"
                ]
            }
        except Exception as e:
            return {"error": f"Error detecting regime transitions: {str(e)}"}

    def _generate_regime_summary(self, overall_regime: dict[str, any], transition_signals: dict[str, any]) -> dict[
        str, any]:
        """Generate comprehensive regime summary"""
        try:
            regime_type = overall_regime.get('overall_regime', 'unknown')
            confidence = overall_regime.get('confidence', 0)
            transition_prob = transition_signals.get('transition_probability', 'unknown')

            # Regime implications
            regime_implications = self._get_regime_implications(regime_type)

            # Investment recommendations
            investment_recommendations = self._get_regime_investment_recommendations(regime_type, confidence)

            return {
                "current_regime": regime_type,
                "regime_confidence": confidence,
                "regime_strength": overall_regime.get('regime_strength', 'unknown'),
                "transition_probability": transition_prob,
                "regime_implications": regime_implications,
                "investment_recommendations": investment_recommendations,
                "key_monitoring_factors": transition_signals.get('monitoring_indicators', []),
                "analysis_timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            return {"error": f"Error generating regime summary: {str(e)}"}

    # Helper methods
    def _calculate_trend_confidence(self, trend_signals: list[float]) -> float:
        """Calculate confidence in trend assessment"""
        if not trend_signals:
            return 0
        # Calculate consistency of trend signals
        std_dev = np.std(trend_signals)
        mean_signal = np.mean(trend_signals)
        # Higher confidence for consistent signals
        consistency = max(0, 1 - std_dev)
        strength = min(1, abs(mean_signal))
        confidence = (consistency * strength) * 100
        return round(confidence, 2)

    def _determine_overall_volatility_regime(self, volatility_regime: dict[str, any]) -> str:
        """Determine overall volatility regime"""
        vix_regime = volatility_regime.get('vix_regime')
        realized_regime = volatility_regime.get('realized_volatility_regime')
        # Combine VIX and realized volatility
        if vix_regime in ['extreme_fear', 'high_volatility'] or realized_regime == 'high_volatility':
            return 'high_volatility'
        elif vix_regime in ['low_volatility'] and realized_regime == 'low_volatility':
            return 'low_volatility'
        else:
            return 'moderate_volatility'

    def _estimate_correlation_strength(self, values: list[float]) -> float:
        """Estimate correlation strength from a list of values"""
        if len(values) < 2:
            return 0
        # Simple correlation estimation based on value dispersion
        mean_val = np.mean(values)
        std_val = np.std(values)
        # Low standard deviation relative to mean suggests high correlation
        if std_val == 0:
            return 1.0
        coefficient_of_variation = std_val / abs(mean_val) if mean_val != 0 else std_val
        # Convert to correlation estimate (inverse relationship)
        correlation_estimate = max(0, 1 - coefficient_of_variation)
        return min(1, correlation_estimate)

    def _determine_overall_correlation_regime(self, correlation_regime: dict[str, any]) -> str:
        """Determine overall correlation regime"""
        index_regime = correlation_regime.get('index_correlation_regime')
        sector_regime = correlation_regime.get('sector_correlation_regime')
        # Combine index and sector correlation regimes
        if 'high' in str(index_regime) or 'high' in str(sector_regime):
            return 'high_correlation'
        elif 'low' in str(index_regime) and 'low' in str(sector_regime):
            return 'low_correlation'
        else:
            return 'moderate_correlation'

    def _get_regime_implications(self, regime_type: str) -> list[str]:
        """Get implications for different regime types"""
        implications = {
            "risk_on": [
                "Favorable environment for growth assets",
                "Low volatility and strong correlations",
                "Credit spreads likely tight",
                "Momentum strategies may outperform"
            ],
            "risk_off": [
                "Flight to quality assets expected",
                "Higher volatility and correlation",
                "Defensive assets likely outperform",
                "Mean reversion strategies may work better"
            ],
            "transition": [
                "Market regime uncertainty high",
                "Mixed signals across asset classes",
                "Increased importance of security selection",
                "Tactical allocation adjustments recommended"
            ],
            "crisis": [
                "Extreme risk aversion and high volatility",
                "Correlations approaching 1.0",
                "Liquidity concerns paramount",
                "Capital preservation primary objective"
            ]
        }
        return implications.get(regime_type, ["Unknown regime - analysis required"])

    def _get_regime_investment_recommendations(self, regime_type: str, confidence: float) -> list[str]:
        """Get investment recommendations based on regime"""
        base_recommendations = {
            "risk_on": [
                "Overweight growth and cyclical sectors",
                "Consider momentum strategies",
                "Reduce hedging positions",
                "Focus on beta and leverage"
            ],
            "risk_off": [
                "Overweight defensive sectors and bonds",
                "Increase hedging and downside protection",
                "Focus on quality and dividend yield",
                "Reduce position sizes and leverage"
            ],
            "transition": [
                "Maintain balanced portfolio allocation",
                "Increase active management and flexibility",
                "Monitor regime indicators closely",
                "Prepare for regime change"
            ],
            "crisis": [
                "Maximum defensive positioning",
                "Maintain high cash levels",
                "Avoid leverage and illiquid assets",
                "Focus on capital preservation"
            ]
        }
        recommendations = base_recommendations.get(regime_type, ["Unknown regime"])
        # Adjust recommendations based on confidence
        if confidence < 50:
            recommendations.append("Low confidence - maintain cautious positioning")
        elif confidence > 80:
            recommendations.append("High confidence - consider more aggressive positioning")
        return recommendations


class MarketRegimeAgent:
    """Specialized agent for market regime identification and analysis"""

    def __init__(self, gcp_services, config):
        self.gcp_services = gcp_services
        self.config = config
        self.tool = MarketRegimeTool()
        logger.info("MarketRegimeAgent initialized")

    async def analyze_market_regime(self, market_data: dict[str, any], economic_data: dict[str, any] = None) -> dict[
        str, any]:
        """Main method for market regime analysis"""
        try:
            logger.info("Starting market regime analysis")
            regime_analysis = await self.tool.call(
                market_data=market_data,
                economic_data=economic_data
            )

            # Store results
            if self.gcp_services:
                if hasattr(self.gcp_services, 'store_regime_analysis'):
                    await self.gcp_services.store_regime_analysis(regime_analysis)
                if hasattr(self.gcp_services, 'store_bigquery_regime_data'):
                    await self.gcp_services.store_bigquery_regime_data(regime_analysis)

            # Add metadata
            regime_analysis["analysis_metadata"] = {
                "agent": "market_regime_analyst",
                "analysis_time": datetime.utcnow().isoformat(),
                "economic_data_included": economic_data is not None,
                "quality_score": self._calculate_regime_quality_score(regime_analysis)
            }

            logger.info("Market regime analysis completed successfully")
            return regime_analysis
        except Exception as e:
            error_msg = f"Error in market regime analysis: {str(e)}"
            logger.error(error_msg)
            return {
                "status": "error",
                "error": error_msg,
                "timestamp": datetime.utcnow().isoformat(),
                "agent": "market_regime_analyst"
            }

    def _calculate_regime_quality_score(self, regime_analysis: dict[str, any]) -> float:
        """Calculate quality score for regime analysis"""
        if regime_analysis.get('status') != 'success':
            return 0.0

        score = 0
        # Base score for successful analysis
        score += 30

        # Regime completeness score
        regime_components = [
            'market_trend_regime',
            'volatility_regime',
            'correlation_regime',
            'sector_rotation_regime'
        ]
        complete_components = sum(1 for comp in regime_components
                                  if regime_analysis.get(comp) and not regime_analysis[comp].get('error'))
        score += (complete_components / len(regime_components)) * 30

        # Overall regime classification score
        overall_regime = regime_analysis.get('overall_market_regime')
        if overall_regime and not overall_regime.get('error'):
            confidence = overall_regime.get('confidence', 0)
            score += (confidence / 100) * 25

        # Transition analysis score
        transition_signals = regime_analysis.get('regime_transition_signals')
        if transition_signals and not transition_signals.get('error'):
            score += 10

        # Economic integration bonus
        if regime_analysis.get('economic_regime'):
            score += 5

        return round(min(100.0, score), 2)
