import logging
import sys
from datetime import datetime, timezone

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
        def __init__(self, model=None, name=None, description=None, tools=None):
            self.model = model
            self.name = name
            self.description = description
            self.tools = tools or []

# Import numpy for calculations
import numpy as np

logger = logging.getLogger(__name__)


class StrategyCoordinationTool(Tool):
    """Tool for coordinating trading strategies based on multi-agent analysis"""

    def __init__(self):
        # Use explicit parent class reference for Python 2.7 compatibility
        Tool.__init__(
            self,
            name="strategy_coordinator",
            description="Coordinate trading strategies based on technical, fundamental, sentiment, and risk analysis"
        )

    async def call(self, analysis_data: dict, market_conditions: dict) -> dict:
        """
        Coordinate trading strategies based on comprehensive analysis
        Args:
            analysis_data: Combined analysis from all analysis agents
            market_conditions: Current market regime and conditions
        Returns:
            Coordinated trading strategy recommendations
        """
        try:
            logger.info("Starting strategy coordination")
            # Extract analysis components
            technical_analysis = analysis_data.get('technical_analysis', {})
            fundamental_analysis = analysis_data.get('fundamental_analysis', {})
            sentiment_analysis = analysis_data.get('sentiment_analysis', {})
            risk_analysis = analysis_data.get('risk_analysis', {})
            economic_analysis = analysis_data.get('economic_analysis', {})

            # Extract market regime information
            market_regime = market_conditions.get('market_regime', {})
            volatility_regime = market_conditions.get('volatility_regime', 'normal')

            # Generate strategy recommendations for each symbol
            symbol_strategies = {}

            # Get symbols from technical analysis
            symbols = list(technical_analysis.get('symbol_analysis', {}).keys())

            for symbol in symbols:
                strategy = await self._generate_symbol_strategy(
                    symbol, analysis_data, market_conditions
                )

                symbol_strategies[symbol] = strategy

            # Generate portfolio-level strategies
            portfolio_strategies = await self._generate_portfolio_strategies(
                symbol_strategies, market_conditions, analysis_data
            )

            # Risk-adjusted strategy recommendations
            risk_adjusted_strategies = await self._apply_risk_adjustments(
                symbol_strategies, portfolio_strategies, risk_analysis
            )

            # Market regime specific strategies
            regime_strategies = await self._generate_regime_strategies(
                market_regime, volatility_regime
            )

            return {
                "status": "success",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "symbol_strategies": symbol_strategies,
                "portfolio_strategies": portfolio_strategies,
                "risk_adjusted_strategies": risk_adjusted_strategies,
                "regime_strategies": regime_strategies,
                "market_context": {
                    "market_regime": market_regime.get('overall_regime', 'unknown'),
                    "volatility_regime": volatility_regime,
                    "strategy_confidence": self._calculate_strategy_confidence(analysis_data)
                },
                "execution_priority": self._prioritize_strategies(symbol_strategies),
                "strategy_summary": self._generate_strategy_summary(symbol_strategies, portfolio_strategies)
            }

        except Exception as e:
            error_msg = f"Error in strategy coordination: {str(e)}"
            logger.error(error_msg)
            return {
                "status": "error",
                "error": error_msg,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

    async def _generate_symbol_strategy(self, symbol: str, analysis_data: dict,
                                        market_conditions: dict) -> dict:
        """Generate strategy for individual symbol"""
        try:
            # Extract symbol-specific analysis
            technical = analysis_data.get('technical_analysis', {}).get('symbol_analysis', {}).get(symbol, {})
            fundamental = analysis_data.get('fundamental_analysis', {}).get('symbol_analysis', {}).get(symbol, {})
            risk = analysis_data.get('risk_analysis', {}).get('symbol_risk_analysis', {}).get(symbol, {})

            # Technical signals
            tech_signal = technical.get('signals', {}).get('overall_signal', 'neutral')
            tech_strength = technical.get('signals', {}).get('signal_strength', 0)

            # Fundamental signals
            fund_rating = fundamental.get('fundamental_rating', {}).get('rating', 'hold')
            fund_score = fundamental.get('fundamental_rating', {}).get('overall_score', 50)

            # Risk assessment
            risk_score = risk.get('overall_risk_score', {}).get('overall_risk_score', 50)
            risk_level = risk.get('overall_risk_score', {}).get('risk_level', 'moderate')

            # Strategy synthesis
            strategy = {
                "symbol": symbol,
                "primary_strategy": self._determine_primary_strategy(tech_signal, fund_rating, risk_level),
                "confidence_level": self._calculate_symbol_confidence(tech_strength, fund_score, risk_score),
                "position_sizing": self._calculate_position_size(risk_score, tech_strength),
                "entry_conditions": self._define_entry_conditions(technical, fundamental),
                "exit_conditions": self._define_exit_conditions(technical, risk),
                "time_horizon": self._determine_time_horizon(fund_rating, tech_signal),
                "risk_management": self._define_risk_management(risk, technical)
            }

            # Market regime adjustments
            market_regime = market_conditions.get('market_regime', {}).get('overall_regime', 'neutral')
            strategy = self._adjust_for_market_regime(strategy, market_regime)

            return strategy

        except Exception as e:
            return {"error": f"Error generating strategy for {symbol}: {str(e)}"}

    async def _generate_portfolio_strategies(self, symbol_strategies: dict,
                                             market_conditions: dict,
                                             analysis_data: dict) -> dict:
        """Generate portfolio-level strategies"""
        try:
            # Aggregate symbol strategies
            buy_count = len([s for s in symbol_strategies.values() if 'buy' in s.get('primary_strategy', '')])
            sell_count = len([s for s in symbol_strategies.values() if 'sell' in s.get('primary_strategy', '')])
            hold_count = len([s for s in symbol_strategies.values() if 'hold' in s.get('primary_strategy', '')])

            # Portfolio allocation strategy
            market_regime = market_conditions.get('market_regime', {}).get('overall_regime', 'neutral')

            portfolio_strategies = {
                "allocation_strategy": self._determine_allocation_strategy(market_regime, buy_count, sell_count),
                "diversification_strategy": self._generate_diversification_strategy(symbol_strategies),
                "hedging_strategy": self._generate_hedging_strategy(market_conditions, analysis_data),
                "rebalancing_strategy": self._generate_rebalancing_strategy(symbol_strategies),
                "cash_management": self._generate_cash_strategy(market_regime, buy_count, sell_count),
                "sector_allocation": self._generate_sector_allocation(symbol_strategies, market_conditions)
            }

            return portfolio_strategies

        except Exception as e:
            return {"error": f"Error generating portfolio strategies: {str(e)}"}

    async def _apply_risk_adjustments(self, symbol_strategies: dict,
                                      portfolio_strategies: dict,
                                      risk_analysis: dict) -> dict:
        """Apply risk-based adjustments to strategies"""
        try:
            # Portfolio risk metrics
            portfolio_risk = risk_analysis.get('portfolio_risk_analysis', {})
            market_risk = risk_analysis.get('market_risk_analysis', {})

            risk_adjustments = {
                "position_size_adjustments": {},
                "stop_loss_adjustments": {},
                "leverage_recommendations": {},
                "hedging_adjustments": {}
            }

            # VIX-based adjustments
            vix_level = market_risk.get('vix_level', 20)

            for symbol, strategy in symbol_strategies.items():
                # Adjust position sizes based on volatility
                base_position = strategy.get('position_sizing', {}).get('recommended_weight', 0.05)

                if vix_level > 30:  # High volatility
                    adjusted_position = base_position * 0.7  # Reduce position size
                elif vix_level < 15:  # Low volatility
                    adjusted_position = base_position * 1.2  # Increase position size
                else:
                    adjusted_position = base_position

                risk_adjustments["position_size_adjustments"][symbol] = {
                    "original_weight": base_position,
                    "adjusted_weight": min(0.10, adjusted_position),  # Cap at 10%
                    "adjustment_reason": f"VIX-based adjustment (VIX: {vix_level})"
                }

                # Adjust stop losses based on volatility
                symbol_risk = risk_analysis.get('symbol_risk_analysis', {}).get(symbol, {})
                volatility = symbol_risk.get('volatility_risk', {}).get('annualized_volatility', 0.20)

                # Dynamic stop loss based on volatility
                stop_loss_pct = max(0.05, min(0.15, volatility * 0.5))  # 5-15% range

                risk_adjustments["stop_loss_adjustments"][symbol] = {
                    "stop_loss_percentage": stop_loss_pct,
                    "volatility_based": True,
                    "annualized_volatility": volatility
                }

            # Portfolio-level leverage recommendations
            portfolio_risk_score = portfolio_risk.get('portfolio_risk_score', 50)

            if portfolio_risk_score > 70:
                leverage_recommendation = "no_leverage"
            elif portfolio_risk_score > 50:
                leverage_recommendation = "conservative_leverage"
            else:
                leverage_recommendation = "moderate_leverage"

            risk_adjustments["leverage_recommendations"] = {
                "recommendation": leverage_recommendation,
                "max_leverage": self._get_max_leverage(leverage_recommendation),
                "portfolio_risk_score": portfolio_risk_score
            }

            return risk_adjustments

        except Exception as e:
            return {"error": f"Error applying risk adjustments: {str(e)}"}

    async def _generate_regime_strategies(self, market_regime: dict,
                                          volatility_regime: str) -> dict:
        """Generate market regime-specific strategies"""
        try:
            regime_type = market_regime.get('overall_regime', 'neutral')
            regime_confidence = market_regime.get('confidence', 50)

            regime_strategies = {
                "regime_type": regime_type,
                "regime_confidence": regime_confidence,
                "recommended_strategies": [],
                "asset_class_preferences": {},
                "sector_preferences": {},
                "style_preferences": {}
            }

            # Risk-on regime strategies
            if regime_type == 'risk_on':
                regime_strategies["recommended_strategies"] = [
                    "Growth momentum strategy",
                    "Cyclical sector overweight",
                    "High beta stock selection",
                    "Reduced hedging positions"
                ]

                regime_strategies["asset_class_preferences"] = {
                    "equities": "overweight",
                    "bonds": "underweight",
                    "commodities": "neutral",
                    "cash": "underweight"
                }

                regime_strategies["sector_preferences"] = {
                    "technology": "overweight",
                    "consumer_discretionary": "overweight",
                    "financials": "overweight",
                    "utilities": "underweight",
                    "consumer_staples": "underweight"
                }

                regime_strategies["style_preferences"] = {
                    "growth": "overweight",
                    "value": "underweight",
                    "momentum": "overweight",
                    "low_volatility": "underweight"
                }

            # Risk-off regime strategies
            elif regime_type == 'risk_off':
                regime_strategies["recommended_strategies"] = [
                    "Defensive positioning",
                    "Quality stock selection",
                    "Increased hedging",
                    "Flight to quality bonds"
                ]

                regime_strategies["asset_class_preferences"] = {
                    "equities": "underweight",
                    "bonds": "overweight",
                    "commodities": "underweight",
                    "cash": "overweight"
                }

                regime_strategies["sector_preferences"] = {
                    "utilities": "overweight",
                    "consumer_staples": "overweight",
                    "healthcare": "overweight",
                    "technology": "underweight",
                    "energy": "underweight"
                }

                regime_strategies["style_preferences"] = {
                    "growth": "underweight",
                    "value": "neutral",
                    "momentum": "underweight",
                    "low_volatility": "overweight"
                }

            # Crisis regime strategies
            elif regime_type == 'crisis':
                regime_strategies["recommended_strategies"] = [
                    "Capital preservation",
                    "Maximum defensive positioning",
                    "High cash allocation",
                    "Avoid leverage completely"
                ]

                regime_strategies["asset_class_preferences"] = {
                    "equities": "underweight",
                    "bonds": "overweight",
                    "commodities": "underweight",
                    "cash": "overweight"
                }

            # Transition regime strategies
            else:
                regime_strategies["recommended_strategies"] = [
                    "Balanced positioning",
                    "Increased flexibility",
                    "Tactical allocation",
                    "Monitor regime indicators"
                ]

                regime_strategies["asset_class_preferences"] = {
                    "equities": "neutral",
                    "bonds": "neutral",
                    "commodities": "neutral",
                    "cash": "neutral"
                }

            # Volatility regime adjustments
            if volatility_regime == 'high_volatility':
                regime_strategies["volatility_adjustments"] = [
                    "Reduce position sizes",
                    "Increase stop-loss usage",
                    "Consider volatility hedging",
                    "Focus on quality names"
                ]

            elif volatility_regime == 'low_volatility':
                regime_strategies["volatility_adjustments"] = [
                    "Consider larger positions",
                    "Look for momentum plays",
                    "Reduce hedging costs",
                    "Watch for complacency"
                ]

            return regime_strategies

        except Exception as e:
            return {"error": f"Error generating regime strategies: {str(e)}"}

    # Helper methods for strategy generation
    def _determine_primary_strategy(self, tech_signal: str, fund_rating: str, risk_level: str) -> str:
        """Determine primary strategy based on signals"""
        # Weight technical and fundamental signals
        tech_weight = 0.6
        fund_weight = 0.4

        # Score technical signal
        tech_score = 0
        if 'strong_buy' in tech_signal:
            tech_score = 2
        elif 'buy' in tech_signal:
            tech_score = 1
        elif 'sell' in tech_signal:
            tech_score = -1
        elif 'strong_sell' in tech_signal:
            tech_score = -2

        # Score fundamental rating
        fund_score = 0
        if 'strong_buy' in fund_rating:
            fund_score = 2
        elif 'buy' in fund_rating:
            fund_score = 1
        elif 'sell' in fund_rating:
            fund_score = -1
        elif 'strong_sell' in fund_rating:
            fund_score = -2

        # Combined score
        combined_score = (tech_score * tech_weight) + (fund_score * fund_weight)

        # Risk adjustment
        if risk_level in ['very_high', 'high']:
            combined_score *= 0.7  # Reduce conviction for high-risk stocks

        # Determine strategy
        if combined_score >= 1.5:
            return "aggressive_buy"
        elif combined_score >= 0.8:
            return "buy"
        elif combined_score >= 0.3:
            return "accumulate"
        elif combined_score <= -1.5:
            return "strong_sell"
        elif combined_score <= -0.8:
            return "sell"
        elif combined_score <= -0.3:
            return "reduce"
        else:
            return "hold"

    def _calculate_symbol_confidence(self, tech_strength: int, fund_score: float, risk_score: float) -> dict:
        """Calculate confidence level for symbol strategy"""
        # Normalize scores
        tech_conf = min(100, tech_strength * 10)  # Convert to 0-100
        fund_conf = max(0, min(100, (fund_score - 25) * 2))  # Convert 25-75 to 0-100
        risk_conf = max(0, min(100, 100 - risk_score))  # Invert risk score

        # Weighted average
        overall_confidence = (tech_conf * 0.4) + (fund_conf * 0.3) + (risk_conf * 0.3)

        return {
            "overall_confidence": round(overall_confidence, 1),
            "technical_confidence": tech_conf,
            "fundamental_confidence": fund_conf,
            "risk_confidence": risk_conf,
            "confidence_level": "high" if overall_confidence > 75 else "medium" if overall_confidence > 50 else "low"
        }

    def _calculate_position_size(self, risk_score: float, tech_strength: int) -> dict:
        """Calculate recommended position sizing"""
        # Base position size (2-8% range)
        base_size = 0.05  # 5% default

        # Risk adjustment
        risk_multiplier = max(0.4, min(1.6, (100 - risk_score) / 50))

        # Technical strength adjustment
        tech_multiplier = max(0.7, min(1.3, 1 + (tech_strength - 5) / 10))

        recommended_weight = base_size * risk_multiplier * tech_multiplier
        recommended_weight = max(0.02, min(0.08, recommended_weight))  # Cap between 2-8%

        return {
            "recommended_weight": round(recommended_weight, 4),
            "min_weight": round(recommended_weight * 0.5, 4),
            "max_weight": round(recommended_weight * 1.5, 4),
            "risk_multiplier": risk_multiplier,
            "technical_multiplier": tech_multiplier
        }

    def _define_entry_conditions(self, technical: dict, fundamental: dict) -> list:
        """Define entry conditions based on analysis"""
        conditions = []

        # Technical conditions
        signals = technical.get('signals', {})
        if signals.get('buy_signals'):
            conditions.extend([f"Technical: {signal}" for signal in signals['buy_signals'][:2]])

        # Support/resistance conditions
        support_resistance = technical.get('support_resistance', {})
        if support_resistance.get('nearest_support'):
            conditions.append(f"Price above support: ${support_resistance['nearest_support']:.2f}")

        # Fundamental conditions
        valuation = fundamental.get('valuation_analysis', {})
        if valuation.get('overall_valuation') == 'undervalued':
            conditions.append("Fundamental: Undervalued on metrics")

        # Volume confirmation
        volume_analysis = technical.get('volume_analysis', {})
        if volume_analysis.get('volume_confirmation') == 'bullish':
            conditions.append("Volume confirms upward movement")

        return conditions[:4]  # Limit to 4 conditions

    def _define_exit_conditions(self, technical: dict, risk: dict) -> dict:
        """Define exit conditions"""
        # Stop loss based on volatility
        volatility_risk = risk.get('volatility_risk', {})
        annualized_vol = volatility_risk.get('annualized_volatility', 0.20)

        # Dynamic stop loss
        stop_loss_pct = max(0.05, min(0.15, annualized_vol * 0.5))

        # Take profit levels
        support_resistance = technical.get('support_resistance', {})
        nearest_resistance = support_resistance.get('nearest_resistance')
        current_price = support_resistance.get('current_price', 0)

        take_profit_1 = None
        take_profit_2 = None

        if nearest_resistance and current_price:
            upside_potential = (nearest_resistance - current_price) / current_price
            if upside_potential > 0.05:  # At least 5% upside
                take_profit_1 = current_price * 1.05  # 5% gain
                take_profit_2 = nearest_resistance * 0.95  # 95% of resistance

        return {
            "stop_loss": {
                "percentage": round(stop_loss_pct * 100, 1),
                "type": "trailing_stop",
                "volatility_based": True
            },
            "take_profit": {
                "target_1": take_profit_1,
                "target_2": take_profit_2,
                "profit_taking_strategy": "scale_out"
            },
            "time_based_exit": "Review position in 30 days",
            "fundamental_exit": "Exit if fundamental rating drops to sell"
        }

    def _determine_time_horizon(self, fund_rating: str, tech_signal: str) -> str:
        """Determine appropriate time horizon"""
        if 'strong' in fund_rating and 'buy' in fund_rating:
            return "long_term"  # 6-12 months
        elif 'buy' in fund_rating:
            return "medium_term"  # 3-6 months
        elif 'strong' in tech_signal:
            return "short_term"  # 1-3 months
        else:
            return "tactical"  # 2-8 weeks

    def _define_risk_management(self, risk: dict, technical: dict) -> dict:
        """Define risk management rules"""
        overall_risk = risk.get('overall_risk_score', {})
        risk_level = overall_risk.get('risk_level', 'moderate')

        # Position sizing rules
        if risk_level == 'very_high':
            max_position = 0.03  # 3% max
            correlation_limit = 0.15  # 15% in correlated positions
        elif risk_level == 'high':
            max_position = 0.05  # 5% max
            correlation_limit = 0.20  # 20% in correlated positions
        else:
            max_position = 0.08  # 8% max
            correlation_limit = 0.30  # 30% in correlated positions

        return {
            "max_position_size": max_position,
            "correlation_limit": correlation_limit,
            "var_limit": "2% daily VaR",
            "sector_concentration": "Max 25% per sector",
            "rebalancing_trigger": "10% deviation from target",
            "stress_test_frequency": "Weekly"
        }

    def _adjust_for_market_regime(self, strategy: dict, market_regime: str) -> dict:
        """Adjust strategy based on market regime"""
        if market_regime == 'risk_off' or market_regime == 'crisis':
            # Reduce position sizes in risk-off environments
            position_sizing = strategy.get('position_sizing', {})
            if 'recommended_weight' in position_sizing:
                original_weight = position_sizing['recommended_weight']
                adjusted_weight = original_weight * 0.7  # 30% reduction
                position_sizing['recommended_weight'] = adjusted_weight
                position_sizing['regime_adjustment'] = f"Reduced due to {market_regime} regime"
        elif market_regime == 'risk_on':
            # Slightly increase conviction in risk-on environments
            confidence = strategy.get('confidence_level', {})
            if 'overall_confidence' in confidence:
                confidence['overall_confidence'] = min(100, confidence['overall_confidence'] * 1.1)
                confidence['regime_boost'] = "Increased confidence in risk-on regime"

        return strategy

    def _determine_allocation_strategy(self, market_regime: str, buy_count: int, sell_count: int) -> dict:
        """Determine portfolio allocation strategy"""
        total_signals = buy_count + sell_count

        if market_regime == 'risk_on' and buy_count > sell_count:
            return {
                "strategy": "aggressive_growth",
                "equity_allocation": "85-95%",
                "cash_allocation": "5-15%",
                "rationale": "Risk-on regime with positive signals"
            }
        elif market_regime == 'risk_off' or sell_count > buy_count:
            return {
                "strategy": "defensive",
                "equity_allocation": "60-75%",
                "cash_allocation": "25-40%",
                "rationale": "Risk-off regime or negative signals"
            }
        else:
            return {
                "strategy": "balanced",
                "equity_allocation": "75-85%",
                "cash_allocation": "15-25%",
                "rationale": "Mixed signals or transition regime"
            }

    def _generate_diversification_strategy(self, symbol_strategies: dict) -> dict:
        """Generate diversification strategy"""
        return {
            "max_single_position": "8%",
            "max_sector_concentration": "25%",
            "min_number_of_positions": "15",
            "correlation_monitoring": "Weekly correlation analysis",
            "geographic_diversification": "Consider international exposure",
            "style_diversification": "Mix of growth and value"
        }

    def _generate_hedging_strategy(self, market_conditions: dict, analysis_data: dict) -> dict:
        """Generate hedging strategy"""
        market_regime = market_conditions.get('market_regime', {}).get('overall_regime', 'neutral')

        if market_regime in ['risk_off', 'crisis']:
            return {
                "hedging_level": "high",
                "recommended_hedges": ["VIX calls", "Put options on SPY", "Inverse ETFs"],
                "hedge_ratio": "15-25% of portfolio",
                "rationale": f"High hedging due to {market_regime} regime"
            }
        elif market_regime == 'transition':
            return {
                "hedging_level": "moderate",
                "recommended_hedges": ["Protective puts", "Collar strategies"],
                "hedge_ratio": "5-15% of portfolio",
                "rationale": "Moderate hedging during regime uncertainty"
            }
        else:
            return {
                "hedging_level": "low",
                "recommended_hedges": ["Opportunistic put buying"],
                "hedge_ratio": "0-5% of portfolio",
                "rationale": "Minimal hedging in favorable regime"
            }

    def _generate_rebalancing_strategy(self, symbol_strategies: dict) -> dict:
        """Generate rebalancing strategy"""
        return {
            "frequency": "Monthly with 10% threshold trigger",
            "methodology": "Strategic rebalancing with tactical overlays",
            "triggers": [
                "10% deviation from target weights",
                "Regime change signals",
                "Significant fundamental changes"
            ],
            "costs_consideration": "Minimize transaction costs",
            "tax_efficiency": "Consider tax implications for taxable accounts"
        }

    def _generate_cash_strategy(self, market_regime: str, buy_count: int, sell_count: int) -> dict:
        """Generate cash management strategy"""
        if market_regime == 'crisis':
            target_cash = "30-50%"
        elif market_regime == 'risk_off':
            target_cash = "20-30%"
        elif market_regime == 'risk_on' and buy_count > sell_count * 2:
            target_cash = "5-15%"
        else:
            target_cash = "15-25%"

        return {
            "target_cash_allocation": target_cash,
            "cash_deployment_strategy": "Deploy gradually on market weakness",
            "emergency_reserve": "Maintain 10% minimum cash",
            "opportunity_fund": "5% for special situations"
        }

    def _generate_sector_allocation(self, symbol_strategies: dict, market_conditions: dict) -> dict:
        """Generate sector allocation strategy"""
        # This would require sector information from symbols
        # Simplified implementation
        return {
            "approach": "Sector neutral with tactical tilts",
            "max_sector_weight": "25%",
            "min_sector_weight": "5%",
            "overweight_sectors": "Based on regime analysis",
            "underweight_sectors": "Based on regime analysis",
            "sector_rotation": "Monitor for regime changes"
        }

    def _get_max_leverage(self, leverage_recommendation: str) -> float:
        """Get maximum leverage based on recommendation"""
        leverage_map = {
            "no_leverage": 1.0,
            "conservative_leverage": 1.2,
            "moderate_leverage": 1.5,
            "aggressive_leverage": 2.0
        }

        return leverage_map.get(leverage_recommendation, 1.0)

    def _calculate_strategy_confidence(self, analysis_data: dict) -> float:
        """Calculate overall strategy confidence"""
        confidences = []

        # Technical analysis confidence
        tech_analysis = analysis_data.get('technical_analysis', {})
        if tech_analysis.get('analysis_metadata', {}).get('quality_score'):
            confidences.append(tech_analysis['analysis_metadata']['quality_score'])

        # Fundamental analysis confidence
        fund_analysis = analysis_data.get('fundamental_analysis', {})
        if fund_analysis.get('analysis_metadata', {}).get('quality_score'):
            confidences.append(fund_analysis['analysis_metadata']['quality_score'])

        # Risk analysis confidence
        risk_analysis = analysis_data.get('risk_analysis', {})
        if risk_analysis.get('analysis_metadata', {}).get('quality_score'):
            confidences.append(risk_analysis['analysis_metadata']['quality_score'])

        # Convert numpy mean to float if needed
        result = float(np.mean(confidences)) if confidences else 50.0
        return result

    def _prioritize_strategies(self, symbol_strategies: dict) -> list:
        """Prioritize strategies for execution"""
        strategy_priorities = []

        for symbol, strategy in symbol_strategies.items():
            if strategy.get('error'):
                continue

            primary_strategy = strategy.get('primary_strategy', 'hold')
            confidence = strategy.get('confidence_level', {}).get('overall_confidence', 50)

            # Assign priority score
            priority_score = 0
            if 'strong' in primary_strategy:
                priority_score += 3
            elif 'buy' in primary_strategy or 'sell' in primary_strategy:
                priority_score += 2
            elif 'accumulate' in primary_strategy or 'reduce' in primary_strategy:
                priority_score += 1

            # Adjust for confidence
            priority_score += (confidence - 50) / 25  # -2 to +2 adjustment

            strategy_priorities.append({
                "symbol": symbol,
                "strategy": primary_strategy,
                "priority_score": priority_score,
                "confidence": confidence
            })

        # Sort by priority score
        strategy_priorities.sort(key=lambda x: x['priority_score'], reverse=True)

        return strategy_priorities[:10]  # Top 10 priorities

    def _generate_strategy_summary(self, symbol_strategies: dict,
                                   portfolio_strategies: dict) -> dict:
        """Generate strategy summary"""
        total_symbols = len([s for s in symbol_strategies.values() if not s.get('error')])

        strategy_distribution = {}
        for strategy in symbol_strategies.values():
            if not strategy.get('error'):
                primary = strategy.get('primary_strategy', 'hold')
                strategy_distribution[primary] = strategy_distribution.get(primary, 0) + 1

        return {
            "total_symbols_analyzed": total_symbols,
            "strategy_distribution": strategy_distribution,
            "portfolio_approach": portfolio_strategies.get('allocation_strategy', {}).get('strategy', 'balanced'),
            "market_positioning": self._determine_market_positioning(strategy_distribution),
            "risk_level": self._assess_portfolio_risk_level(symbol_strategies),
            "execution_timeline": "Implement over 5-10 trading days",
            "review_frequency": "Weekly strategy review with monthly rebalancing"
        }

    def _determine_market_positioning(self, strategy_distribution: dict) -> str:
        """Determine overall market positioning"""
        buy_strategies = strategy_distribution.get('buy', 0) + strategy_distribution.get('aggressive_buy',
                                                                                         0) + strategy_distribution.get(
            'accumulate', 0)
        sell_strategies = strategy_distribution.get('sell', 0) + strategy_distribution.get('strong_sell',
                                                                                           0) + strategy_distribution.get(
            'reduce', 0)

        if buy_strategies > sell_strategies * 2:
            return "bullish"
        elif sell_strategies > buy_strategies * 2:
            return "bearish"
        else:
            return "neutral"

    def _assess_portfolio_risk_level(self, symbol_strategies: dict) -> str:
        """Assess overall portfolio risk level"""
        risk_scores = []

        for strategy in symbol_strategies.values():
            if not strategy.get('error'):
                risk_mgmt = strategy.get('risk_management', {})
                max_position = risk_mgmt.get('max_position_size', 0.05)
                risk_scores.append(max_position)

        avg_position_size = float(np.mean(risk_scores)) if risk_scores else 0.05

        if avg_position_size > 0.06:
            return "aggressive"
        elif avg_position_size > 0.04:
            return "moderate"
        else:
            return "conservative"


class StrategyCoordinatorAgent(Agent):
    """Specialized agent for coordinating trading strategies"""

    def __init__(self, gcp_services, config):
        # Use explicit parent class reference for Python 2.7 compatibility
        Agent.__init__(
            self,
            model="gemini-2.0-flash-exp",
            name="strategy_coordinator",
            description="Expert strategy coordinator for multi-agent trading system",
            tools=[StrategyCoordinationTool()]
        )

        self.gcp_services = gcp_services
        self.config = config
        logger.info("StrategyCoordinatorAgent initialized")

    async def coordinate_strategies(self, analysis_data: dict, market_conditions: dict) -> dict:
        """Main method for strategy coordination"""
        try:
            logger.info("Starting strategy coordination process")

            strategy_coordination = await self.tools[0].call(
                analysis_data=analysis_data,
                market_conditions=market_conditions
            )

            # Store results if GCP services available
            if hasattr(self.gcp_services, 'store_strategy_coordination'):
                await self.gcp_services.store_strategy_coordination(strategy_coordination)

            if hasattr(self.gcp_services, 'store_bigquery_strategy_data'):
                await self.gcp_services.store_bigquery_strategy_data(strategy_coordination)

            # Add metadata
            strategy_coordination["coordination_metadata"] = {
                "agent": self.name,
                "coordination_time": datetime.now(timezone.utc).isoformat(),
                "analysis_components": list(analysis_data.keys()),
                "quality_score": self._calculate_coordination_quality_score(strategy_coordination)
            }

            logger.info("Strategy coordination completed successfully")
            return strategy_coordination

        except Exception as e:
            error_msg = f"Error in strategy coordination: {str(e)}"
            logger.error(error_msg)
            return {
                "status": "error",
                "error": error_msg,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "agent": self.name
            }

    def _calculate_coordination_quality_score(self, coordination_result: dict) -> float:
        """Calculate quality score for strategy coordination"""
        if coordination_result.get('status') != 'success':
            return 0.0

        score = 0

        # Symbol strategies completeness
        symbol_strategies = coordination_result.get('symbol_strategies', {})
        if symbol_strategies:
            successful_strategies = len([s for s in symbol_strategies.values() if not s.get('error')])
            total_strategies = len(symbol_strategies)
            completeness_score = (successful_strategies / total_strategies * 40) if total_strategies > 0 else 0
            score += completeness_score

        # Portfolio strategies
        portfolio_strategies = coordination_result.get('portfolio_strategies', {})
        if portfolio_strategies and not portfolio_strategies.get('error'):
            score += 25

        # Risk adjustments
        risk_adjustments = coordination_result.get('risk_adjusted_strategies', {})
        if risk_adjustments and not risk_adjustments.get('error'):
            score += 20

        # Regime strategies
        regime_strategies = coordination_result.get('regime_strategies', {})
        if regime_strategies and not regime_strategies.get('error'):
            score += 15

        return round(min(100.0, score), 2)
