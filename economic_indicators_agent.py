import logging
import os
import sys
from datetime import datetime, timezone
from typing import Dict, Any, List
import aiohttp
import asyncio
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


class EconomicIndicatorsTool(Tool):
    """Tool for collecting economic indicators and macro data"""

    def __init__(self):
        super().__init__(
            name="economic_indicators_collector",
            description="Collect economic indicators and macroeconomic data affecting markets"
        )
        self.alpha_vantage_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        self.fred_api_key = os.getenv('FRED_API_KEY')  # Federal Reserve Economic Data

    async def call(self, indicators: List[str] = None) -> Dict[str, Any]:
        """
        Collect economic indicators and macro data
        Args:
            indicators: List of specific indicators to collect
        Returns:
            Economic indicators data
        """
        try:
            logger.info("Collecting economic indicators data")
            # Default indicators if none specified
            if not indicators:
                indicators = [
                    'GDP', 'INFLATION', 'UNEMPLOYMENT', 'INTEREST_RATES',
                    'CONSUMER_CONFIDENCE', 'RETAIL_SALES', 'INDUSTRIAL_PRODUCTION'
                ]

            economic_data = {}
            async with aiohttp.ClientSession() as session:
                tasks = []
                # Collect key economic indicators
                tasks.append(self._fetch_gdp_data(session))
                tasks.append(self._fetch_inflation_data(session))
                tasks.append(self._fetch_unemployment_data(session))
                tasks.append(self._fetch_interest_rates(session))
                tasks.append(self._fetch_consumer_confidence(session))
                tasks.append(self._fetch_retail_sales(session))
                tasks.append(self._fetch_industrial_production(session))
                tasks.append(self._fetch_market_indicators(session))

                # Wait for all data collection
                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Process results
                indicator_names = [
                    'gdp', 'inflation', 'unemployment', 'interest_rates',
                    'consumer_confidence', 'retail_sales', 'industrial_production', 'market_indicators'
                ]

                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        logger.warning(f"Error collecting {indicator_names[i]}: {str(result)}")
                        economic_data[indicator_names[i]] = {"error": str(result)}
                    else:
                        economic_data[indicator_names[i]] = result

            # Calculate economic health score
            economic_health = self._calculate_economic_health_score(economic_data)
            # Identify key trends and signals
            economic_signals = self._identify_economic_signals(economic_data)

            return {
                "status": "success",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "economic_data": economic_data,
                "economic_health_score": economic_health,
                "economic_signals": economic_signals,
                "market_impact_assessment": self._assess_market_impact(economic_data),
                "data_freshness": self._assess_data_freshness(economic_data)
            }

        except Exception as e:
            error_msg = f"Error collecting economic indicators: {str(e)}"
            logger.error(error_msg)
            return {
                "status": "error",
                "error": error_msg,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

    async def _fetch_gdp_data(self, session: aiohttp.ClientSession) -> Dict[str, Any]:
        """Fetch GDP data"""
        try:
            if not self.alpha_vantage_key:
                return {"error": "Alpha Vantage API key not configured"}

            url = "https://www.alphavantage.co/query"
            params = {
                'function': 'REAL_GDP',
                'interval': 'quarterly',
                'apikey': self.alpha_vantage_key
            }

            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if 'data' in data:
                        gdp_data = data['data'][:4]  # Last 4 quarters
                        latest_gdp = float(gdp_data[0]['value']) if gdp_data else 0
                        previous_gdp = float(gdp_data[1]['value']) if len(gdp_data) > 1 else latest_gdp
                        gdp_growth = ((latest_gdp - previous_gdp) / previous_gdp * 100) if previous_gdp != 0 else 0
                        return {
                            "current_gdp": latest_gdp,
                            "previous_gdp": previous_gdp,
                            "growth_rate": gdp_growth,
                            "trend": "positive" if gdp_growth > 0 else "negative" if gdp_growth < 0 else "stable",
                            "last_updated": gdp_data[0]['date'] if gdp_data else None,
                            "historical_data": gdp_data
                        }
                    else:
                        return {"error": "No GDP data available"}
                else:
                    return {"error": f"GDP API request failed with status {response.status}"}
        except Exception as e:
            return {"error": f"Error fetching GDP data: {str(e)}"}

    async def _fetch_inflation_data(self, session: aiohttp.ClientSession) -> Dict[str, Any]:
        """Fetch inflation data (CPI)"""
        try:
            if not self.alpha_vantage_key:
                return {"error": "Alpha Vantage API key not configured"}

            url = "https://www.alphavantage.co/query"
            params = {
                'function': 'CPI',
                'interval': 'monthly',
                'apikey': self.alpha_vantage_key
            }

            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if 'data' in data:
                        cpi_data = data['data'][:12]  # Last 12 months
                        latest_cpi = float(cpi_data[0]['value']) if cpi_data else 0
                        year_ago_cpi = float(cpi_data[11]['value']) if len(cpi_data) >= 12 else latest_cpi
                        inflation_rate = ((latest_cpi - year_ago_cpi) / year_ago_cpi * 100) if year_ago_cpi != 0 else 0
                        return {
                            "current_cpi": latest_cpi,
                            "year_ago_cpi": year_ago_cpi,
                            "inflation_rate": inflation_rate,
                            "trend": "increasing" if inflation_rate > 2.5 else "decreasing" if inflation_rate < 1.5 else "stable",
                            "target_comparison": inflation_rate - 2.0,  # Fed target is 2%
                            "last_updated": cpi_data[0]['date'] if cpi_data else None,
                            "historical_data": cpi_data
                        }
                    else:
                        return {"error": "No CPI data available"}
                else:
                    return {"error": f"CPI API request failed with status {response.status}"}
        except Exception as e:
            return {"error": f"Error fetching inflation data: {str(e)}"}

    async def _fetch_unemployment_data(self, session: aiohttp.ClientSession) -> Dict[str, Any]:
        """Fetch unemployment data"""
        try:
            if not self.alpha_vantage_key:
                return {"error": "Alpha Vantage API key not configured"}

            url = "https://www.alphavantage.co/query"
            params = {
                'function': 'UNEMPLOYMENT',
                'apikey': self.alpha_vantage_key
            }

            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if 'data' in data:
                        unemployment_data = data['data'][:12]  # Last 12 months
                        current_rate = float(unemployment_data[0]['value']) if unemployment_data else 0
                        previous_rate = float(unemployment_data[1]['value']) if len(
                            unemployment_data) > 1 else current_rate
                        rate_change = current_rate - previous_rate
                        return {
                            "current_rate": current_rate,
                            "previous_rate": previous_rate,
                            "rate_change": rate_change,
                            "trend": "increasing" if rate_change > 0.1 else "decreasing" if rate_change < -0.1 else "stable",
                            "health_indicator": "good" if current_rate < 4.0 else "concerning" if current_rate > 6.0 else "moderate",
                            "last_updated": unemployment_data[0]['date'] if unemployment_data else None,
                            "historical_data": unemployment_data
                        }
                    else:
                        return {"error": "No unemployment data available"}
                else:
                    return {"error": f"Unemployment API request failed with status {response.status}"}
        except Exception as e:
            return {"error": f"Error fetching unemployment data: {str(e)}"}

    async def _fetch_interest_rates(self, session: aiohttp.ClientSession) -> Dict[str, Any]:
        """Fetch interest rates data"""
        try:
            if not self.alpha_vantage_key:
                return {"error": "Alpha Vantage API key not configured"}

            url = "https://www.alphavantage.co/query"
            params = {
                'function': 'FEDERAL_FUNDS_RATE',
                'interval': 'monthly',
                'apikey': self.alpha_vantage_key
            }

            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if 'data' in data:
                        rate_data = data['data'][:12]  # Last 12 months
                        current_rate = float(rate_data[0]['value']) if rate_data else 0
                        previous_rate = float(rate_data[1]['value']) if len(rate_data) > 1 else current_rate
                        rate_change = current_rate - previous_rate
                        return {
                            "federal_funds_rate": current_rate,
                            "previous_rate": previous_rate,
                            "rate_change": rate_change,
                            "trend": "rising" if rate_change > 0.1 else "falling" if rate_change < -0.1 else "stable",
                            "policy_stance": "hawkish" if current_rate > 4.0 else "dovish" if current_rate < 2.0 else "neutral",
                            "last_updated": rate_data[0]['date'] if rate_data else None,
                            "historical_data": rate_data
                        }
                    else:
                        return {"error": "No interest rate data available"}
                else:
                    return {"error": f"Interest rate API request failed with status {response.status}"}
        except Exception as e:
            return {"error": f"Error fetching interest rate data: {str(e)}"}

    async def _fetch_consumer_confidence(self, session: aiohttp.ClientSession) -> Dict[str, Any]:
        """Fetch consumer confidence data"""
        # Simplified implementation - in production, integrate with proper data source
        try:
            # Using a mock implementation since consumer confidence requires specialized data sources
            current_confidence = 110.5  # Mock value
            historical_avg = 100.0
            return {
                "current_index": current_confidence,
                "historical_average": historical_avg,
                "relative_position": ((current_confidence - historical_avg) / historical_avg) * 100,
                "sentiment": "optimistic" if current_confidence > 105 else "pessimistic" if current_confidence < 95 else "neutral",
                "last_updated": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                "data_source": "mock_data"
            }
        except Exception as e:
            return {"error": f"Error fetching consumer confidence data: {str(e)}"}

    async def _fetch_retail_sales(self, session: aiohttp.ClientSession) -> Dict[str, Any]:
        """Fetch retail sales data"""
        try:
            # Mock implementation - integrate with proper data source in production
            current_sales = 105.2  # Mock index value
            previous_sales = 103.8
            growth_rate = ((current_sales - previous_sales) / previous_sales) * 100
            return {
                "current_index": current_sales,
                "previous_index": previous_sales,
                "growth_rate": growth_rate,
                "trend": "growing" if growth_rate > 1.0 else "declining" if growth_rate < -1.0 else "stable",
                "consumer_health": "strong" if growth_rate > 2.0 else "weak" if growth_rate < 0 else "moderate",
                "last_updated": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                "data_source": "mock_data"
            }
        except Exception as e:
            return {"error": f"Error fetching retail sales data: {str(e)}"}

    async def _fetch_industrial_production(self, session: aiohttp.ClientSession) -> Dict[str, Any]:
        """Fetch industrial production data"""
        try:
            # Mock implementation
            current_production = 102.3
            previous_production = 101.5
            growth_rate = ((current_production - previous_production) / previous_production) * 100
            return {
                "current_index": current_production,
                "previous_index": previous_production,
                "growth_rate": growth_rate,
                "trend": "expanding" if growth_rate > 0.5 else "contracting" if growth_rate < -0.5 else "stable",
                "economic_health": "strong" if growth_rate > 1.0 else "weak" if growth_rate < 0 else "moderate",
                "last_updated": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                "data_source": "mock_data"
            }
        except Exception as e:
            return {"error": f"Error fetching industrial production data: {str(e)}"}

    @staticmethod
    async def _fetch_market_indicators(session: aiohttp.ClientSession) -> Dict[str, Any]:
        """Fetch key market indicators"""
        try:
            # This would integrate with market data APIs to get VIX, yield curve, etc.
            return {
                "vix_level": 18.5,
                "vix_interpretation": "moderate_volatility",
                "yield_curve_10y_2y": 0.45,
                "yield_curve_signal": "normal",
                "dollar_index": 103.2,
                "dollar_trend": "strong",
                "commodity_index": 95.8,
                "commodity_trend": "stable"
            }
        except Exception as e:
            return {"error": f"Error fetching market indicators: {str(e)}"}

    def _calculate_economic_health_score(self, economic_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall economic health score"""
        try:
            score = 50  # Neutral baseline
            factors = []

            # GDP contribution
            gdp_data = economic_data.get('gdp', {})
            if not gdp_data.get('error'):
                gdp_growth = gdp_data.get('growth_rate', 0)
                gdp_score = min(20, max(-20, gdp_growth * 5))  # Scale GDP growth
                score += gdp_score
                factors.append(f"GDP: {gdp_score:+.1f}")

            # Inflation contribution
            inflation_data = economic_data.get('inflation', {})
            if not inflation_data.get('error'):
                inflation_rate = inflation_data.get('inflation_rate', 2.0)
                # Optimal inflation around 2%, penalize deviations
                inflation_deviation = abs(inflation_rate - 2.0)
                inflation_score = max(-15, 10 - inflation_deviation * 3)
                score += inflation_score
                factors.append(f"Inflation: {inflation_score:+.1f}")

            # Unemployment contribution
            unemployment_data = economic_data.get('unemployment', {})
            if not unemployment_data.get('error'):
                unemployment_rate = unemployment_data.get('current_rate', 5.0)
                # Lower unemployment is better, but very low can indicate overheating
                if unemployment_rate < 3.5:
                    unemployment_score = 5  # Overheating concern
                elif unemployment_rate < 5.0:
                    unemployment_score = 15  # Good level
                elif unemployment_rate < 7.0:
                    unemployment_score = 0  # Moderate
                else:
                    unemployment_score = -15  # High unemployment
                score += unemployment_score
                factors.append(f"Unemployment: {unemployment_score:+.1f}")

            # Interest rates contribution
            rates_data = economic_data.get('interest_rates', {})
            if not rates_data.get('error'):
                fed_rate = rates_data.get('federal_funds_rate', 3.0)
                # Moderate rates are generally good for markets
                if 2.0 <= fed_rate <= 4.0:
                    rates_score = 10
                elif fed_rate < 1.0:
                    rates_score = -5  # Too low may indicate crisis
                elif fed_rate > 6.0:
                    rates_score = -10  # Too high may slow growth
                else:
                    rates_score = 5
                score += rates_score
                factors.append(f"Rates: {rates_score:+.1f}")

            # Cap score between 0-100
            final_score = max(0, min(100, score))
            if final_score >= 70:
                health_status = "strong"
            elif final_score >= 50:
                health_status = "moderate"
            elif final_score >= 30:
                health_status = "weak"
            else:
                health_status = "concerning"

            return {
                "overall_score": final_score,
                "health_status": health_status,
                "contributing_factors": factors,
                "calculation_timestamp": datetime.now(timezone.utc).isoformat()
            }
        except Exception as e:
            return {
                "overall_score": 50,
                "health_status": "unknown",
                "error": f"Error calculating economic health: {str(e)}"
            }

    def _identify_economic_signals(self, economic_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify key economic signals and trends"""
        signals = []
        try:
            # GDP signals
            gdp_data = economic_data.get('gdp', {})
            if not gdp_data.get('error'):
                growth_rate = gdp_data.get('growth_rate', 0)
                if growth_rate > 3.0:
                    signals.append({
                        "signal": "Strong GDP Growth",
                        "type": "positive",
                        "description": f"GDP growing at {growth_rate:.1f}%, indicating robust economic expansion",
                        "market_impact": "bullish"
                    })
                elif growth_rate < 0:
                    signals.append({
                        "signal": "GDP Contraction",
                        "type": "negative",
                        "description": f"GDP contracting at {growth_rate:.1f}%, recession risk elevated",
                        "market_impact": "bearish"
                    })

            # Inflation signals
            inflation_data = economic_data.get('inflation', {})
            if not inflation_data.get('error'):
                inflation_rate = inflation_data.get('inflation_rate', 2.0)
                target_comparison = inflation_data.get('target_comparison', 0)
                if inflation_rate > 4.0:
                    signals.append({
                        "signal": "High Inflation",
                        "type": "negative",
                        "description": f"Inflation at {inflation_rate:.1f}%, well above Fed target",
                        "market_impact": "bearish_for_bonds_mixed_for_stocks"
                    })
                elif inflation_rate < 1.0:
                    signals.append({
                        "signal": "Low Inflation",
                        "type": "mixed",
                        "description": f"Inflation at {inflation_rate:.1f}%, below Fed target, deflationary concerns",
                        "market_impact": "bullish_for_bonds_mixed_for_stocks"
                    })

            # Interest rate signals
            rates_data = economic_data.get('interest_rates', {})
            if not rates_data.get('error'):
                trend = rates_data.get('trend', 'stable')
                current_rate = rates_data.get('federal_funds_rate', 3.0)
                if trend == 'rising' and current_rate > 4.0:
                    signals.append({
                        "signal": "Rising Interest Rates",
                        "type": "negative",
                        "description": f"Fed funds rate at {current_rate:.1f}% and rising, tightening conditions",
                        "market_impact": "bearish_for_growth_stocks"
                    })
                elif trend == 'falling':
                    signals.append({
                        "signal": "Falling Interest Rates",
                        "type": "positive",
                        "description": f"Fed funds rate falling, accommodative policy stance",
                        "market_impact": "bullish_for_stocks_and_bonds"
                    })

            # Unemployment signals
            unemployment_data = economic_data.get('unemployment', {})
            if not unemployment_data.get('error'):
                current_rate = unemployment_data.get('current_rate', 5.0)
                trend = unemployment_data.get('trend', 'stable')
                if current_rate < 3.5:
                    signals.append({
                        "signal": "Very Low Unemployment",
                        "type": "mixed",
                        "description": f"Unemployment at {current_rate:.1f}%, potential wage pressure and overheating",
                        "market_impact": "inflationary_pressure"
                    })
                elif current_rate > 6.0 and trend == 'increasing':
                    signals.append({
                        "signal": "Rising Unemployment",
                        "type": "negative",
                        "description": f"Unemployment rising to {current_rate:.1f}%, economic weakness",
                        "market_impact": "bearish"
                    })

            return signals
        except Exception as e:
            logger.error(f"Error identifying economic signals: {str(e)}")
            return [{
                "signal": "Analysis Error",
                "type": "error",
                "description": f"Error in signal analysis: {str(e)}",
                "market_impact": "unknown"
            }]

    def _assess_market_impact(self, economic_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall market impact of economic conditions"""
        try:
            impact_scores = {
                "equities": 0,
                "bonds": 0,
                "commodities": 0,
                "currency": 0
            }

            # GDP impact
            gdp_data = economic_data.get('gdp', {})
            if not gdp_data.get('error'):
                growth_rate = gdp_data.get('growth_rate', 0)
                impact_scores["equities"] += min(20, max(-20, growth_rate * 3))
                impact_scores["commodities"] += min(15, max(-15, growth_rate * 2))

            # Inflation impact
            inflation_data = economic_data.get('inflation', {})
            if not inflation_data.get('error'):
                inflation_rate = inflation_data.get('inflation_rate', 2.0)
                # High inflation negative for bonds, mixed for stocks
                impact_scores["bonds"] -= min(25, max(0, (inflation_rate - 2.0) * 5))
                impact_scores["commodities"] += min(15, max(-10, (inflation_rate - 2.0) * 3))

            # Interest rates impact
            rates_data = economic_data.get('interest_rates', {})
            if not rates_data.get('error'):
                fed_rate = rates_data.get('federal_funds_rate', 3.0)
                trend = rates_data.get('trend', 'stable')
                if trend == 'rising':
                    impact_scores["equities"] -= 10
                    impact_scores["bonds"] -= 15
                    impact_scores["currency"] += 10
                elif trend == 'falling':
                    impact_scores["equities"] += 15
                    impact_scores["bonds"] += 10
                    impact_scores["currency"] -= 5

            # Convert scores to interpretations
            market_outlook = {}
            for asset, score in impact_scores.items():
                if score > 10:
                    outlook = "bullish"
                elif score > 5:
                    outlook = "slightly_bullish"
                elif score > -5:
                    outlook = "neutral"
                elif score > -10:
                    outlook = "slightly_bearish"
                else:
                    outlook = "bearish"
                market_outlook[asset] = {
                    "score": score,
                    "outlook": outlook
                }

            return {
                "market_outlook": market_outlook,
                "overall_sentiment": "risk_on" if sum(impact_scores.values()) > 10 else "risk_off" if sum(
                    impact_scores.values()) < -10 else "neutral",
                "assessment_timestamp": datetime.now(timezone.utc).isoformat()
            }
        except Exception as e:
            return {
                "error": f"Error assessing market impact: {str(e)}",
                "assessment_timestamp": datetime.now(timezone.utc).isoformat()
            }

    def _assess_data_freshness(self, economic_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess how fresh the economic data is"""
        freshness_report = {}
        for indicator, data in economic_data.items():
            if isinstance(data, dict) and not data.get('error'):
                last_updated = data.get('last_updated')
                if last_updated:
                    try:
                        update_date = datetime.fromisoformat(last_updated.replace('Z', '+00:00'))
                        days_old = (datetime.now(timezone.utc) - update_date.replace(tzinfo=None)).days
                        if days_old <= 7:
                            freshness = "fresh"
                        elif days_old <= 30:
                            freshness = "recent"
                        elif days_old <= 90:
                            freshness = "stale"
                        else:
                            freshness = "very_stale"
                        freshness_report[indicator] = {
                            "last_updated": last_updated,
                            "days_old": days_old,
                            "freshness": freshness
                        }
                    except:
                        freshness_report[indicator] = {"freshness": "unknown"}
                else:
                    freshness_report[indicator] = {"freshness": "no_date"}
            else:
                freshness_report[indicator] = {"freshness": "error"}
        return freshness_report


class EconomicIndicatorsAgent(Agent):
    """Specialized agent for economic indicators and macro analysis"""

    def __init__(self, gcp_services, config):
        super().__init__(
            model="gemini-2.0-flash-exp",
            name="economic_indicators_analyst",
            description="Expert macroeconomic analyst specializing in economic indicators",
            instructions="""You are a specialized economic indicators analysis agent with expertise in:
1. Macroeconomic data collection and analysis
2. Economic health assessment and scoring
3. Market impact evaluation of economic trends
4. Economic signal identification and interpretation
5. Cross-indicator correlation analysis

Your primary responsibilities:
- Collect key economic indicators (GDP, inflation, employment, etc.)
- Calculate comprehensive economic health scores
- Identify significant economic signals and trends
- Assess market impact across asset classes
- Provide forward-looking economic insights

Focus on providing actionable insights that consider:
- Economic cycle positioning
- Policy implications and central bank actions
- Cross-asset market impacts
- Leading vs lagging indicator relationships""",
            tools=[EconomicIndicatorsTool()]
        )
        self.gcp_services = gcp_services
        self.config = config
        logger.info("EconomicIndicatorsAgent initialized")

    async def analyze_economic_indicators(self, specific_indicators: List[str] = None) -> Dict[str, Any]:
        """Main method to analyze economic indicators"""
        try:
            logger.info("Starting economic indicators analysis")
            economic_data = await self.tools[0].call(indicators=specific_indicators)

            # Store results
            if self.gcp_services:
                if hasattr(self.gcp_services, 'store_economic_data'):
                    await self.gcp_services.store_economic_data(economic_data)
                if hasattr(self.gcp_services, 'store_bigquery_economic_data'):
                    await self.gcp_services.store_bigquery_economic_data(economic_data)

            # Add metadata
            economic_data["analysis_metadata"] = {
                "agent": self.name,
                "analysis_time": datetime.now(timezone.utc).isoformat(),
                "indicators_analyzed": specific_indicators or "all_default",
                "quality_score": self._calculate_economic_quality_score(economic_data)
            }

            logger.info("Economic indicators analysis completed")
            return economic_data
        except Exception as e:
            error_msg = f"Error in economic indicators analysis: {str(e)}"
            logger.error(error_msg)
            return {
                "status": "error",
                "error": error_msg,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "agent": self.name
            }

    def _calculate_economic_quality_score(self, economic_data: Dict[str, Any]) -> float:
        """Calculate quality score for economic analysis"""
        if economic_data.get('status') != 'success':
            return 0.0

        data = economic_data.get('economic_data', {})
        total_indicators = len(data)
        successful_indicators = len([k for k, v in data.items() if not v.get('error')])
        if total_indicators == 0:
            return 0.0

        # Base score from data availability
        base_score = (successful_indicators / total_indicators) * 70

        # Bonus for data freshness
        freshness_data = economic_data.get('data_freshness', {})
        fresh_count = len([k for k, v in freshness_data.items() if v.get('freshness') == 'fresh'])
        freshness_bonus = (fresh_count / total_indicators) * 20 if total_indicators > 0 else 0

        # Bonus for economic health calculation
        health_bonus = 10 if economic_data.get('economic_health_score', {}).get('overall_score') is not None else 0

        total_score = base_score + freshness_bonus + health_bonus
        return round(min(100.0, total_score), 2)
