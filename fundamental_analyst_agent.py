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


class FundamentalAnalysisTool(Tool):
    """Tool for comprehensive fundamental analysis"""

    def __init__(self):
        super().__init__(
            name="fundamental_analyzer",
            description="Perform comprehensive fundamental analysis including financial ratios, valuation metrics, and growth analysis"
        )

    async def call(self, market_data: dict[str, any], symbols: list[str]) -> dict[str, any]:
        """
        Perform fundamental analysis on securities
        Args:
            market_data: Market data from data collection agents
            symbols: List of symbols to analyze
        Returns:
            Comprehensive fundamental analysis results
        """
        try:
            logger.info("Starting fundamental analysis")
            if market_data.get('status') != 'success':
                return {
                    "status": "error",
                    "error": "Invalid market data provided",
                    "timestamp": datetime.utcnow().isoformat()
                }

            data = market_data.get('data', {})
            analysis_results = {}

            # Analyze each symbol
            for symbol in symbols:
                if symbol in data and isinstance(data[symbol], dict):
                    symbol_data = data[symbol]
                    if not symbol_data.get('error'):
                        logger.info(f"Analyzing fundamental data for {symbol}")
                        symbol_analysis = await self._analyze_symbol_fundamentals(symbol, symbol_data)
                        analysis_results[symbol] = symbol_analysis

            # Industry and sector analysis
            sector_analysis = await self._analyze_sector_fundamentals(analysis_results)

            # Market valuation analysis
            market_valuation = await self._analyze_market_valuation(analysis_results)

            return {
                "status": "success",
                "timestamp": datetime.utcnow().isoformat(),
                "symbol_analysis": analysis_results,
                "sector_analysis": sector_analysis,
                "market_valuation": market_valuation,
                "summary": self._generate_fundamental_summary(analysis_results)
            }

        except Exception as e:
            error_msg = f"Error in fundamental analysis: {str(e)}"
            logger.error(error_msg)
            return {
                "status": "error",
                "error": error_msg,
                "timestamp": datetime.utcnow().isoformat()
            }

    async def _analyze_symbol_fundamentals(self, symbol: str, symbol_data: dict[str, any]) -> dict[str, any]:
        """Perform comprehensive fundamental analysis for a single symbol"""
        try:
            financial_metrics = symbol_data.get('financial_metrics', {})
            company_info = symbol_data.get('company_info', {})
            current_price = symbol_data.get('current_price', 0)

            analysis_result = {
                "symbol": symbol,
                "company_name": company_info.get('longName', symbol),
                "sector": company_info.get('sector', 'Unknown'),
                "industry": company_info.get('industry', 'Unknown'),
                "current_price": current_price
            }

            # Valuation Analysis
            valuation_analysis = self._analyze_valuation_metrics(financial_metrics, current_price)
            analysis_result["valuation_analysis"] = valuation_analysis

            # Profitability Analysis
            profitability_analysis = self._analyze_profitability(financial_metrics)
            analysis_result["profitability_analysis"] = profitability_analysis

            # Financial Health Analysis
            financial_health = self._analyze_f极inancial_health(financial_metrics)
            analysis_result["financial_health"] = financial_health

            # Growth Analysis
            growth_analysis = self._analyze_growth_metrics(financial_metrics)
            analysis_result["growth_analysis"] = growth_analysis

            # Dividend Analysis
            dividend_analysis = self._analyze_dividend_metrics(financial_metrics)
            analysis_result["dividend_analysis"] = dividend_analysis

            # Overall fundamental rating
            fundamental_rating = self._calculate_fundamental_rating(analysis_result)
            analysis_result["fundamental_rating"] = fundamental_rating

            return analysis_result
        except Exception as e:
            return {"error": f"Error analyzing {symbol}: {str(e)}"}

    def _analyze_valuation_metrics(self, metrics: dict[str, any], current_price: float) -> dict[str, any]:
        """Analyze valuation metrics"""
        try:
            pe_ratio = metrics.get('pe_ratio')
            forward_pe = metrics.get('forward_pe')
            price_to_book = metrics.get('price_to_book')
            price_to_sales = metrics.get('price_to_sales')
            peg_ratio = metrics.get('peg_ratio')
            enterprise_value = metrics.get('enterprise_value')
            market_cap = metrics.get('market_cap')

            valuation = {
                "pe_ratio": pe_ratio,
                "forward_pe": forward_pe,
                "price_to_book": price_to_book,
                "price_to_sales": price_to_sales,
                "peg_ratio": peg_ratio,
                "enterprise_value": enterprise_value,
                "market_cap": market_cap
            }

            # Valuation assessment
            valuation_signals = []

            # P/E Ratio Analysis
            if pe_ratio:
                if pe_ratio < 15:
                    valuation_signals.append("Potentially undervalued (Low P/E)")
                elif pe_ratio > 25:
                    valuation_signals.append("Potentially overvalued (High P/E)")
                else:
                    valuation_signals.append("Fairly valued (Moderate P/E)")

            # PEG Ratio Analysis
            if peg_ratio:
                if peg_ratio < 1.0:
                    valuation_signals.append("Good value relative to growth (PEG < 1)")
                elif peg_ratio > 2.0:
                    valuation_signals.append("Expensive relative to growth (PEG > 2)")

            # Price-to-Book Analysis
            if price_to_book:
                if price_to_book < 1.0:
                    valuation_signals.append("Trading below book value")
                elif price_to_book > 3.0:
                    valuation_signals.append("High premium to book value")

            # Overall valuation assessment
            if len([s for s in valuation_signals if "undervalued" in s or "Good value" in s or "below book" in s]) >= 2:
                overall_valuation = "undervalued"
            elif len([s for s in valuation_signals if
                      "overvalued" in s or "Expensive" in s or "High premium" in s]) >= 2:
                overall_valuation = "overvalued"
            else:
                overall_valuation = "fairly_valued"

            valuation["valuation_signals"] = valuation_signals
            valuation["overall_valuation"] = overall_valuation

            return valuation
        except Exception as e:
            return {"error": f"Error in valuation analysis: {str(e)}"}

    def _analyze_profitability(self, metrics: dict[str, any]) -> dict[str, any]:
        """Analyze profitability metrics"""
        try:
            profitability = {
                "profit_margin_estimate": "Not available - requires income statement data",
                "roe_estimate": "Not available - requires detailed financial data",
                "roa_estimate": "Not available - requires detailed financial data",
                "gross_margin_estimate": "Not available - requires revenue and cost data"
            }

            # Basic profitability assessment based on available data
            market_cap = metrics.get('market_cap', 0)
            if market_cap:
                if market_cap > 10_000_000_000:  # Large cap
                    profitability["company_size"] = "Large Cap (Generally stable profitability)"
                elif market_cap > 2_000_000_000:  # Mid cap
                    profitability["company_size"] = "Mid Cap (Moderate growth potential)"
                else:  # Small cap
                    profitability["company_size"] = "Small Cap (Higher risk/reward)"

            return profitability
        except Exception as e:
            return {"error": f"Error in profitability analysis: {str(e)}"}

    def _analyze_financial_health(self, metrics: dict[str, any]) -> dict[str, any]:
        """Analyze financial health indicators"""
        try:
            beta = metrics.get('beta')
            market_cap = metrics.get('market_cap')

            health_indicators = {
                "beta": beta,
                "market_cap": market_cap,
                "volatility_assessment": self._assess_volatility(beta),
                "size_stability": self._assess_size_stability(market_cap)
            }

            # Overall health score
            health_score = 50  # Neutral baseline
            if beta is not None:
                if 0.8 <= beta <= 1.2:
                    health_score += 10  # Market-like volatility
                elif beta < 0.8:
                    health_score += 5  # Lower volatility
                else:
                    health_score -= 5  # Higher volatility

            if market_cap:
                if market_cap > 10_000_000_000:
                    health_score += 15  # Large, stable companies
                elif market_cap > 2_000_000_000:
                    health_score += 5  # Mid-cap stability
                else:
                    health_score -= 5  # Small-cap risk

            health_indicators["financial_health_score"] = max(0, min(100, health_score))
            health_indicators["health_rating"] = self._rate_financial_health(health_score)

            return health_indicators
        except Exception as e:
            return {"error": f"Error in financial health analysis: {str(e)}"}

    def _analyze_growth_metrics(self, metrics: dict[str, any]) -> dict[str, any]:
        """Analyze growth potential"""
        try:
            # Basic growth analysis based on available metrics
            pe_ratio = metrics.get('pe_ratio')
            forward_pe = metrics.get('forward_pe')
            peg_ratio = metrics.get('peg_ratio')

            growth_analysis = {
                "pe_ratio": pe_ratio,
                "forward_pe": forward_pe,
                "peg_ratio": peg_ratio
            }

            growth_signals = []

            # Forward P/E vs Current P/E
            if pe_ratio and forward_pe:
                if forward_pe < pe_ratio:
                    growth_signals.append("Earnings expected to grow (Forward P/E < Current P/E)")
                    growth_analysis["earnings_growth_expected"] = True
                else:
                    growth_signals.append("Earnings growth may be slowing")
                    growth_analysis["earnings_growth_expected"] = False

            # PEG Ratio for growth assessment
            if peg_ratio:
                if peg_ratio < 1.0:
                    growth_signals.append("Strong growth relative to valuation")
                elif peg_ratio < 1.5:
                    growth_signals.append("Reasonable growth at current valuation")
                else:
                    growth_signals.append("Growth may not justify current valuation")

            growth_analysis["growth_signals"] = growth_signals
            growth_analysis["growth_assessment"] = self._assess_overall_growth(growth_signals)

            return growth_analysis
        except Exception as e:
            return {"error": f"Error in growth analysis: {str(e)}"}

    def _analyze_dividend_metrics(self, metrics: dict[str, any]) -> dict[str, any]:
        """Analyze dividend-related metrics"""
        try:
            dividend_yield = metrics.get('dividend_yield')
            dividend_rate = metrics.get('dividend_rate')

            dividend_analysis = {
                "dividend_yield": dividend_yield,
                "dividend_rate": dividend_rate
            }

            if dividend_yield:
                dividend_yield_pct = dividend_yield * 100
                if dividend_yield_pct > 4:
                    dividend_analysis["yield_assessment"] = "High yield (>4%)"
                    dividend_analysis["income_potential"] = "Strong"
                elif dividend_yield_pct > 2:
                    dividend_analysis["yield_assessment"] = "Moderate yield (2-4%)"
                    dividend_analysis["income_potential"] = "Moderate"
                elif dividend_yield_pct > 0:
                    dividend_analysis["yield_assessment"] = "Low yield (<2%)"
                    dividend_analysis["income_potential"] = "Low"
                else:
                    dividend_analysis["yield_assessment"] = "No dividend"
                    dividend_analysis["income_potential"] = "None"
            else:
                dividend_analysis["yield_assessment"] = "No dividend information"
                dividend_analysis["income_potential"] = "Unknown"

            return dividend_analysis
        except Exception as e:
            return {"error": f"Error in dividend analysis: {str(e)}"}

    def _calculate_fundamental_rating(self, analysis_result: dict[str, any]) -> dict[str, any]:
        """Calculate overall fundamental rating"""
        try:
            score = 50  # Neutral baseline
            factors = []

            # Valuation score
            valuation = analysis_result.get('valuation_analysis', {})
            overall_valuation = valuation.get('overall_valuation')
            if overall_valuation == 'undervalued':
                score += 15
                factors.append("Undervalued (+15)")
            elif overall_valuation == 'overvalued':
                score -= 10
                factors.append("Overvalued (-10)")

            # Financial health score
            health = analysis_result.get('financial_health', {})
            health_score = health.get('financial_health_score', 50)
            health_contribution = (health_score - 50) / 5  # Scale to -10 to +10
            score += health_contribution
            factors.append(f"Financial Health ({health_contribution:+.1f})")

            # Growth score
            growth = analysis_result.get('growth_analysis', {})
            growth_assessment = growth.get('growth_assessment', 'neutral')
            if growth_assessment == 'strong':
                score += 10
                factors.append("Strong Growth (+10)")
            elif growth_assessment == 'weak':
                score -= 5
                factors.append("Weak Growth (-5)")

            # Dividend score
            dividend = analysis_result.get('dividend_analysis', {})
            income_potential = dividend.get('income_potential', 'None')
            if income_potential == 'Strong':
                score += 5
                factors.append("Strong Dividend (+5)")
            elif income_potential == 'Moderate':
                score += 2
                factors.append("Moderate Dividend (+2)")

            # Normalize score
            final_score = max(0, min(100, score))

            # Rating categories
            if final_score >= 75:
                rating = "strong_buy"
            elif final_score >= 65:
                rating = "buy"
            elif final_score >= 55:
                rating = "hold"
            elif final_score >= 45:
                rating = "weak_hold"
            elif final_score >= 35:
                rating = "sell"
            else:
                rating = "strong_sell"

            return {
                "overall_score": final_score,
                "rating": rating,
                "contributing_factors": factors,
                "confidence_level": min(100, len(factors) * 15)
            }
        except Exception as e:
            return {"error": f"Error calculating fundamental rating: {str(e)}"}

    def _assess_volatility(self, beta: float) -> str:
        """Assess volatility based on beta"""
        if not beta:
            return "Unknown"
        if beta < 0.5:
            return "Very Low Volatility"
        elif beta < 0.8:
            return "Low Volatility"
        elif beta <= 1.2:
            return "Market-like Volatility"
        elif beta <= 1.5:
            return "High Volatility"
        else:
            return "Very High Volatility"

    def _assess_size_stability(self, market_cap: float) -> str:
        """Assess stability based on market cap"""
        if not market_cap:
            return "Unknown"
        if market_cap > 50_000_000_000:
            return "Mega Cap - Very Stable"
        elif market_cap > 10_000_000_000:
            return "Large Cap - Stable"
        elif market_cap > 2_000_000_000:
            return "Mid Cap - Moderate Stability"
        elif market_cap > 300_000_000:
            return "Small Cap - Lower Stability"
        else:
            return "Micro Cap - High Risk"

    def _rate_financial_health(self, score: float) -> str:
        """Rate financial health based on score"""
        if score >= 75:
            return "Excellent"
        elif score >= 65:
            return "Good"
        elif score >= 55:
            return "Fair"
        elif score >= 45:
            return "Poor"
        else:
            return "Very Poor"

    def _assess_overall_growth(self, growth_signals: list[str]) -> str:
        """Assess overall growth potential"""
        positive_signals = len(
            [s for s in growth_signals if "Strong" in s or "expected to grow" in s or "Reasonable" in s])
        negative_signals = len([s for s in growth_signals if "slowing" in s or "not justify" in s])

        if positive_signals > negative_signals:
            return "strong"
        elif negative_signals > positive_signals:
            return "weak"
        else:
            return "neutral"

    async def _analyze_sector_fundamentals(self, symbol_analyses: dict[str, any]) -> dict[str, any]:
        """Analyze fundamentals by sector"""
        try:
            sector_data = {}
            for symbol, analysis in symbol_analyses.items():
                if analysis.get('error'):
                    continue

                sector = analysis.get('sector', 'Unknown')
                if sector not in sector_data:
                    sector_data[sector] = {
                        'symbols': [],
                        'avg_pe_ratio': [],
                        'avg_valuation_score': [],
                        'ratings': []
                    }

                sector_data[sector]['symbols'].append(symbol)

                # Collect metrics
                valuation = analysis.get('valuation_analysis', {})
                pe_ratio = valuation.get('pe_ratio')
                if pe_ratio:
                    sector_data[sector]['avg_pe_ratio'].append(pe_ratio)

                rating = analysis.get('fundamental_rating', {})
                rating_score = rating.get('overall_score')
                if rating_score:
                    sector_data[sector]['avg_valuation_score'].append(rating_score)

                rating_name = rating.get('rating')
                if rating_name:
                    sector_data[sector]['ratings'].append(rating_name)

            # Calculate sector averages
            sector_summary = {}
            for sector, data in sector_data.items():
                avg_pe = np.mean(data['avg_pe_ratio']) if data['avg_pe_ratio'] else None
                avg_score = np.mean(data['avg_valuation_score']) if data['avg_valuation_score'] else None

                # Most common rating
                ratings = data['ratings']
                most_common_rating = max(set(ratings), key=ratings.count) if ratings else 'Unknown'

                sector_summary[sector] = {
                    'symbol_count': len(data['symbols']),
                    'symbols': data['symbols'],
                    'average_pe_ratio': avg_pe,
                    'average_fundamental_score': avg_score,
                    'predominant_rating': most_common_rating
                }

            return sector_summary
        except Exception as e:
            return {"error": f"Error in sector analysis: {str(e)}"}

    async def _analyze_market_valuation(self, symbol_analyses: dict[str, any]) -> dict[str, any]:
        """Analyze overall market valuation"""
        try:
            all_pe_ratios = []
            all_scores = []
            rating_counts = {}

            for symbol, analysis in symbol_analyses.items():
                if analysis.get('error'):
                    continue

                # Collect P/E ratios
                valuation = analysis.get('valuation_analysis', {})
                pe_ratio = valuation.get('pe_ratio')
                if pe_ratio and pe_ratio > 0:
                    all_pe_ratios.append(pe_ratio)

                # Collect fundamental scores
                rating = analysis.get('fundamental_rating', {})
                score = rating.get('overall_score')
                if score:
                    all_scores.append(score)

                # Count ratings
                rating_name = rating.get('rating', 'Unknown')
                rating_counts[rating_name] = rating_counts.get(rating_name, 0) + 1

            market_valuation = {
                'average_pe_ratio': np.mean(all_pe_ratios) if all_pe_ratios else None,
                'median_pe_ratio': np.median(all_pe_ratios) if all_pe_ratios else None,
                'average_fundamental_score': np.mean(all_scores) if all_scores else None,
                'rating_distribution': rating_counts,
                'total_analyzed': len([a for a in symbol_analyses.values() if not a.get('error')])
            }

            # Market assessment
            avg_score = market_valuation['average_fundamental_score']
            if avg_score:
                if avg_score >= 65:
                    market_valuation['market_assessment'] = 'Fundamentally Strong'
                elif avg_score >= 55:
                    market_valuation['market_assessment'] = 'Fundamentally Neutral'
                elif avg_score >= 45:
                    market_valuation['market_assessment'] = 'Fundamentally Weak'
                else:
                    market_valuation['market_assessment'] = 'Fundamentally Poor'
            else:
                market_valuation['market_assessment'] = 'Insufficient Data'

            return market_valuation
        except Exception as e:
            return {"error": f"Error in market valuation analysis: {str(e)}"}

    def _generate_fundamental_summary(self, symbol_analyses: dict[str, any]) -> dict[str, any]:
        """Generate summary of fundamental analysis"""
        try:
            total_symbols = len(symbol_analyses)
            successful_analyses = len([a for a in symbol_analyses.values() if not a.get('error')])

            # Count ratings
            buy_ratings = 0
            sell_ratings = 0
            hold_ratings = 0

            for analysis in symbol_analyses.values():
                if analysis.get('error'):
                    continue
                rating = analysis.get('fundamental_rating', {}).get('rating', '')
                if 'buy' in rating:
                    buy_ratings += 1
                elif 'sell' in rating:
                    sell_ratings += 1
                else:
                    hold_ratings += 1

            return {
                "total_symbols_analyzed": total_symbols,
                "successful_analyses": successful_analyses,
                "buy_recommendations": buy_ratings,
                "sell_recommendations": sell_ratings,
                "hold_recommendations": hold_ratings,
                "analysis_timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            return {"error": f"Error generating summary: {str(e)}"}


class FundamentalAnalystAgent:
    """Specialized agent for fundamental analysis"""

    def __init__(self, gcp_services, config):
        self.gcp_services = gcp_services
        self.config = config
        self.tool = FundamentalAnalysisTool()
        logger.info("FundamentalAnalystAgent initialized")

    async def analyze_securities_fundamental(self, market_data: dict[str, any], symbols: list[str]) -> dict[str, any]:
        """Main method for comprehensive fundamental analysis"""
        try:
            logger.info(f"Starting fundamental analysis for {len(symbols)} symbols")
            fundamental_analysis = await self.tool.call(
                market_data=market_data,
                symbols=symbols
            )

            # Store results
            if self.gcp_services:
                if hasattr(self.gcp_services, 'store_fundamental_analysis'):
                    await self.gcp_services.store_fundamental_analysis(fundamental_analysis)
                if hasattr(self.gcp_services, 'store_bigquery_fundamental_data'):
                    await self.gcp_services.store_bigquery_fundamental_data(fundamental_analysis)

            # Add metadata
            fundamental_analysis["analysis_metadata"] = {
                "agent": "fundamental_analyst_expert",
                "analysis_time": datetime.utcnow().isoformat(),
                "symbols_analyzed": symbols,
                "quality_score": self._calculate_fundamental_quality_score(fundamental_analysis)
            }

            logger.info("Fundamental analysis completed successfully")
            return fundamental_analysis
        except Exception as e:
            error_msg = f"Error in fundamental analysis: {str(e)}"
            logger.error(error_msg)
            return {
                "status": "error",
                "error": error_msg,
                "timestamp": datetime.utcnow().isoformat(),
                "agent": "fundamental_analyst_expert"
            }

    def _calculate_fundamental_quality_score(self, fundamental_analysis: dict[str, any]) -> float:
        """Calculate quality score for fundamental analysis"""
        if fundamental_analysis.get('status') != 'success':
            return 0.0

        symbol_analysis = fundamental_analysis.get('symbol_analysis', {})
        total_symbols = len(symbol_analysis)
        if total_symbols == 0:
            return 0.0

        # Base score from successful analysis
        successful_analysis = len([s for s in symbol_analysis.values() if not s.get('error')])
        base_score = (successful_analysis / total_symbols) * 60

        # Data completeness score
        complete_analyses = 0
        for analysis in symbol_analysis.values():
            if not analysis.get('error'):
                completeness = 0
                if analysis.get('valuation_analysis'):
                    completeness += 1
                if analysis.get('financial_health'):
                    completeness += 1
                if analysis.get('growth_analysis'):
                    completeness += 1
                if analysis.get('dividend_analysis'):
                    completeness += 1
                if completeness >= 3:
                    complete_analyses += 1

        completeness_score = (complete_analyses / total_symbols * 25) if total_symbols > 0 else 0

        # Analysis depth score
        depth_score = 0
        if fundamental_analysis.get('sector_analysis'):
            depth_score += 7
        if fundamental_analysis.get('market_valuation'):
            depth_score += 8

        total_score = base_score + completeness_score + depth_score
        return round(min(100.0, total_score), 2)
