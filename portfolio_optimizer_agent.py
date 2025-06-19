import logging
import sys
from datetime import datetime, timezone
from typing import Dict, Any, List
import numpy as np
import cvxpy as cp
from scipy.optimize import minimize

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


class PortfolioOptimizationTool(Tool):
    """Tool for portfolio optimization using modern portfolio theory and advanced techniques"""

    def __init__(self):
        # Use explicit parent class reference for Python 2.7 compatibility
        Tool.__init__(
            self,
            name="portfolio_optimizer",
            description="Optimize portfolio allocation using mean-variance optimization, risk parity, and other advanced techniques"
        )

    async def call(self, strategy_data: Dict[str, Any], risk_data: Dict[str, Any],
                   constraints: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Optimize portfolio allocation based on strategies and risk analysis
        Args:
            strategy_data: Strategy recommendations from strategy coordinator
            risk_data: Risk analysis data for optimization
            constraints: Optional portfolio constraints
        Returns:
            Optimized portfolio allocation
        """
        try:
            logger.info("Starting portfolio optimization")
            # Extract strategy information
            symbol_strategies = strategy_data.get('symbol_strategies', {})
            portfolio_strategies = strategy_data.get('portfolio_strategies', {})

            # Set default constraints if None provided
            if constraints is None:
                constraints = {}

            # Prepare optimization inputs
            optimization_inputs = await self._prepare_optimization_inputs(
                symbol_strategies, risk_data, constraints
            )

            if optimization_inputs.get('error'):
                return optimization_inputs

            # Run multiple optimization approaches
            optimizations = {}

            # Mean-Variance Optimization
            mv_optimization = await self._mean_variance_optimization(optimization_inputs)
            optimizations['mean_variance'] = mv_optimization

            # Risk Parity Optimization
            rp_optimization = await self._risk_parity_optimization(optimization_inputs)
            optimizations['risk_parity'] = rp_optimization

            # Black-Litterman Optimization
            bl_optimization = await self._black_litterman_optimization(optimization_inputs)
            optimizations['black_litterman'] = bl_optimization

            # Strategic Allocation (Strategy-based)
            strategic_optimization = await self._strategic_allocation(
                optimization_inputs, symbol_strategies
            )
            optimizations['strategic'] = strategic_optimization

            # Ensemble optimization (combine approaches)
            ensemble_optimization = await self._ensemble_optimization(optimizations)

            # Portfolio analytics
            portfolio_analytics = await self._calculate_portfolio_analytics(
                ensemble_optimization, optimization_inputs
            )

            # Rebalancing recommendations
            rebalancing_plan = await self._generate_rebalancing_plan(
                ensemble_optimization, strategy_data
            )

            return {
                "status": "success",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "optimization_approaches": optimizations,
                "recommended_allocation": ensemble_optimization,
                "portfolio_analytics": portfolio_analytics,
                "rebalancing_plan": rebalancing_plan,
                "optimization_summary": self._generate_optimization_summary(optimizations, ensemble_optimization)
            }

        except Exception as e:
            error_msg = f"Error in portfolio optimization: {str(e)}"
            logger.error(error_msg)
            return {
                "status": "error",
                "error": error_msg,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

    async def _prepare_optimization_inputs(self, symbol_strategies: Dict[str, Any],
                                           risk_data: Dict[str, Any],
                                           constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare inputs for portfolio optimization"""
        try:
            # Extract symbols and expected returns
            symbols = []
            expected_returns = []
            risk_scores = []
            strategy_weights = []

            for symbol, strategy in symbol_strategies.items():
                if strategy.get('error'):
                    continue

                symbols.append(symbol)

                # Expected return estimation
                confidence = strategy.get('confidence_level', {}).get('overall_confidence', 50)
                primary_strategy = strategy.get('primary_strategy', 'hold')

                # Convert strategy to expected return
                expected_return = self._strategy_to_expected_return(primary_strategy, confidence)
                expected_returns.append(expected_return)

                # Risk score
                symbol_risk = risk_data.get('symbol_risk_analysis', {}).get(symbol, {})
                risk_score = symbol_risk.get('overall_risk_score', {}).get('overall_risk_score', 50)
                risk_scores.append(risk_score)

                # Strategy weight preference
                position_sizing = strategy.get('position_sizing', {})
                strategy_weight = position_sizing.get('recommended_weight', 0.05)
                strategy_weights.append(strategy_weight)

            if len(symbols) == 0:
                return {"error": "No valid symbols for optimization"}

            # Create correlation matrix (simplified - using risk scores as proxy)
            correlation_matrix = self._estimate_correlation_matrix(symbols, risk_scores)

            # Create covariance matrix
            volatilities = np.array([self._risk_score_to_volatility(rs) for rs in risk_scores])
            covariance_matrix = np.outer(volatilities, volatilities) * correlation_matrix

            # Default constraints
            default_constraints = {
                'min_weight': 0.01,  # 1% minimum
                'max_weight': 0.10,  # 10% maximum
                'max_sector_weight': 0.25,  # 25% per sector
                'min_cash': 0.05,  # 5% minimum cash
                'max_leverage': 1.0,  # No leverage
                'target_return': None,
                'max_volatility': None
            }

            default_constraints.update(constraints)

            return {
                "symbols": symbols,
                "expected_returns": np.array(expected_returns),
                "covariance_matrix": covariance_matrix,
                "risk_scores": np.array(risk_scores),
                "strategy_weights": np.array(strategy_weights),
                "constraints": default_constraints,
                "num_assets": len(symbols)
            }

        except Exception as e:
            return {"error": f"Error preparing optimization inputs: {str(e)}"}

    async def _mean_variance_optimization(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Mean-variance portfolio optimization"""
        try:
            symbols = inputs['symbols']
            expected_returns = inputs['expected_returns']
            cov_matrix = inputs['covariance_matrix']
            constraints = inputs['constraints']
            n_assets = inputs['num_assets']

            # Define optimization variables
            weights = cp.Variable(n_assets)

            # Objective: minimize variance for given return or maximize Sharpe ratio
            portfolio_variance = cp.quad_form(weights, cov_matrix)
            portfolio_return = expected_returns.T @ weights

            # Constraints
            optimization_constraints = [
                cp.sum(weights) <= constraints['max_leverage'],  # Leverage constraint
                weights >= constraints['min_weight'],  # Minimum weights
                weights <= constraints['max_weight']  # Maximum weights
            ]

            # Solve for different risk levels
            efficient_frontier = []
            target_returns = np.linspace(expected_returns.min(), expected_returns.max(), 10)
            optimal_weights = None
            optimal_sharpe = -np.inf

            for target_return in target_returns:
                # Add return constraint
                current_constraints = optimization_constraints + [portfolio_return >= target_return]

                # Solve optimization
                problem = cp.Problem(cp.Minimize(portfolio_variance), current_constraints)
                problem.solve(solver=cp.OSQP, verbose=False)

                if problem.status == cp.OPTIMAL and weights.value is not None:
                    portfolio_vol = np.sqrt(portfolio_variance.value)
                    sharpe_ratio = target_return / portfolio_vol if portfolio_vol > 0 else 0

                    efficient_frontier.append({
                        'return': float(target_return),
                        'volatility': float(portfolio_vol),
                        'sharpe_ratio': float(sharpe_ratio),
                        'weights': weights.value.copy()
                    })

                    # Track best Sharpe ratio
                    if sharpe_ratio > optimal_sharpe:
                        optimal_sharpe = sharpe_ratio
                        optimal_weights = weights.value.copy()

            # Create allocation dictionary
            if optimal_weights is not None:
                allocation = {symbols[i]: max(0, float(optimal_weights[i])) for i in range(n_assets)}
                # Normalize to sum to target leverage
                total_weight = sum(allocation.values())
                if total_weight > 0:
                    target_leverage = constraints['max_leverage']
                    allocation = {k: v / total_weight * target_leverage for k, v in allocation.items()}

                expected_return = float(np.dot(optimal_weights, expected_returns))
                expected_volatility = float(np.sqrt(np.dot(optimal_weights, np.dot(cov_matrix, optimal_weights))))
            else:
                # Fallback to equal weights
                equal_weight = constraints['max_leverage'] / n_assets
                allocation = {symbol: equal_weight for symbol in symbols}
                expected_return = float(np.mean(expected_returns))
                expected_volatility = float(np.sqrt(np.mean(np.diag(cov_matrix))))
                optimal_sharpe = expected_return / expected_volatility if expected_volatility > 0 else 0

            return {
                "allocation": allocation,
                "expected_return": expected_return,
                "expected_volatility": expected_volatility,
                "sharpe_ratio": float(optimal_sharpe),
                "efficient_frontier": efficient_frontier,
                "optimization_method": "mean_variance"
            }

        except Exception as e:
            return {"error": f"Error in mean-variance optimization: {str(e)}"}

    async def _risk_parity_optimization(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Risk parity portfolio optimization"""
        try:
            symbols = inputs['symbols']
            cov_matrix = inputs['covariance_matrix']
            constraints = inputs['constraints']
            n_assets = inputs['num_assets']

            # Risk parity objective: equal risk contribution
            def risk_parity_objective(weights):
                portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
                if portfolio_vol == 0:
                    return 1e6  # Large penalty for zero volatility
                # Risk contributions
                marginal_contrib = np.dot(cov_matrix, weights) / portfolio_vol
                contrib = weights * marginal_contrib
                # Minimize difference from equal risk contribution
                target_contrib = portfolio_vol / n_assets
                return np.sum((contrib - target_contrib) ** 2)

            # Constraints
            constraints_scipy = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - constraints['max_leverage']},
            ]

            # Bounds
            bounds = [(constraints['min_weight'], constraints['max_weight']) for _ in range(n_assets)]

            # Initial guess (equal weights)
            initial_weights = np.ones(n_assets) / n_assets * constraints['max_leverage']

            # Optimize
            result = minimize(
                risk_parity_objective,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints_scipy,
                options={'ftol': 1e-9, 'disp': False}
            )

            if result.success and result.x is not None:
                optimal_weights = result.x
                allocation = {symbols[i]: max(0, float(optimal_weights[i])) for i in range(n_assets)}

                # Calculate portfolio metrics
                portfolio_return = float(np.dot(optimal_weights, inputs['expected_returns']))
                portfolio_vol = float(np.sqrt(np.dot(optimal_weights, np.dot(cov_matrix, optimal_weights))))
                sharpe_ratio = portfolio_return / portfolio_vol if portfolio_vol > 0 else 0
            else:
                # Fallback to equal weights
                equal_weight = constraints['max_leverage'] / n_assets
                allocation = {symbol: equal_weight for symbol in symbols}
                portfolio_return = float(np.mean(inputs['expected_returns']))
                portfolio_vol = float(np.sqrt(np.mean(np.diag(cov_matrix))))
                sharpe_ratio = portfolio_return / portfolio_vol if portfolio_vol > 0 else 0

            return {
                "allocation": allocation,
                "expected_return": portfolio_return,
                "expected_volatility": portfolio_vol,
                "sharpe_ratio": float(sharpe_ratio),
                "optimization_method": "risk_parity"
            }

        except Exception as e:
            return {"error": f"Error in risk parity optimization: {str(e)}"}

    async def _black_litterman_optimization(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Black-Litterman portfolio optimization"""
        try:
            symbols = inputs['symbols']
            expected_returns = inputs['expected_returns']
            cov_matrix = inputs['covariance_matrix']
            strategy_weights = inputs['strategy_weights']
            constraints = inputs['constraints']
            n_assets = inputs['num_assets']

            # Market equilibrium returns (reverse optimization)
            risk_aversion = 3.0  # Typical risk aversion parameter
            market_weights = strategy_weights / np.sum(strategy_weights)  # Use strategy weights as market proxy
            equilibrium_returns = risk_aversion * np.dot(cov_matrix, market_weights)

            # Investor views (based on strategy recommendations)
            # Create views matrix P and view returns Q
            P = np.eye(n_assets)  # Views on each asset
            Q = expected_returns - equilibrium_returns  # Excess returns from analysis

            # Uncertainty matrix (higher uncertainty for lower confidence)
            tau = 0.1  # Scaling factor
            omega = tau * np.diag(np.diag(cov_matrix))

            # Black-Litterman formula
            M1 = np.linalg.inv(tau * cov_matrix)
            M2 = np.dot(P.T, np.dot(np.linalg.inv(omega), P))
            M3 = np.dot(np.linalg.inv(tau * cov_matrix), equilibrium_returns)
            M4 = np.dot(P.T, np.dot(np.linalg.inv(omega), Q))

            bl_returns = np.dot(np.linalg.inv(M1 + M2), M3 + M4)
            bl_cov = np.linalg.inv(M1 + M2)

            # Optimize with Black-Litterman inputs
            weights = cp.Variable(n_assets)
            portfolio_variance = cp.quad_form(weights, bl_cov)
            portfolio_return = bl_returns.T @ weights

            # Constraints
            optimization_constraints = [
                cp.sum(weights) == constraints['max_leverage'],
                weights >= constraints['min_weight'],
                weights <= constraints['max_weight']
            ]

            # Maximize Sharpe ratio (minimize -return/volatility proxy)
            objective = cp.Minimize(portfolio_variance - 2 * portfolio_return)
            problem = cp.Problem(objective, optimization_constraints)
            problem.solve(solver=cp.OSQP, verbose=False)

            if problem.status == cp.OPTIMAL and weights.value is not None:
                optimal_weights = weights.value
                allocation = {symbols[i]: max(0, float(optimal_weights[i])) for i in range(n_assets)}
                portfolio_return_val = float(np.dot(optimal_weights, bl_returns))
                portfolio_vol = float(np.sqrt(np.dot(optimal_weights, np.dot(bl_cov, optimal_weights))))
                sharpe_ratio = portfolio_return_val / portfolio_vol if portfolio_vol > 0 else 0
            else:
                # Fallback allocation
                equal_weight = constraints['max_leverage'] / n_assets
                allocation = {symbol: equal_weight for symbol in symbols}
                portfolio_return_val = float(np.mean(bl_returns))
                portfolio_vol = float(np.sqrt(np.mean(np.diag(bl_cov))))
                sharpe_ratio = portfolio_return_val / portfolio_vol if portfolio_vol > 0 else 0

            return {
                "allocation": allocation,
                "expected_return": portfolio_return_val,
                "expected_volatility": portfolio_vol,
                "sharpe_ratio": float(sharpe_ratio),
                "black_litterman_returns": bl_returns.tolist(),
                "optimization_method": "black_litterman"
            }

        except Exception as e:
            return {"error": f"Error in Black-Litterman optimization: {str(e)}"}

    async def _strategic_allocation(self, inputs: Dict[str, Any],
                                    symbol_strategies: Dict[str, Any]) -> Dict[str, Any]:
        """Strategic allocation based on strategy recommendations"""
        try:
            symbols = inputs['symbols']
            constraints = inputs['constraints']

            # Start with strategy-recommended weights
            allocation = {}
            total_weight = 0

            for symbol in symbols:
                strategy = symbol_strategies.get(symbol, {})
                if strategy.get('error'):
                    continue

                # Get recommended weight
                position_sizing = strategy.get('position_sizing', {})
                base_weight = position_sizing.get('recommended_weight', 0.05)

                # Adjust based on strategy conviction
                primary_strategy = strategy.get('primary_strategy', 'hold')
                confidence = strategy.get('confidence_level', {}).get('overall_confidence', 50)

                # Strategy multiplier
                strategy_multiplier = self._get_strategy_multiplier(primary_strategy)
                confidence_multiplier = 0.5 + (confidence / 100)  # 0.5 to 1.5 range

                adjusted_weight = base_weight * strategy_multiplier * confidence_multiplier

                # Apply constraints
                adjusted_weight = max(constraints['min_weight'],
                                      min(constraints['max_weight'], adjusted_weight))

                allocation[symbol] = adjusted_weight
                total_weight += adjusted_weight

            # Normalize to target leverage
            if total_weight > 0:
                target_leverage = min(constraints['max_leverage'], 0.95)  # Leave 5% cash
                normalization_factor = target_leverage / total_weight
                allocation = {k: v * normalization_factor for k, v in allocation.items()}

            # Calculate expected metrics
            expected_returns = inputs['expected_returns']
            cov_matrix = inputs['covariance_matrix']

            weights_array = np.array([allocation.get(symbol, 0) for symbol in symbols])
            portfolio_return = float(np.dot(weights_array, expected_returns))
            portfolio_vol = float(np.sqrt(np.dot(weights_array, np.dot(cov_matrix, weights_array))))
            sharpe_ratio = portfolio_return / portfolio_vol if portfolio_vol > 0 else 0

            return {
                "allocation": allocation,
                "expected_return": portfolio_return,
                "expected_volatility": portfolio_vol,
                "sharpe_ratio": float(sharpe_ratio),
                "cash_allocation": max(0.05, 1.0 - sum(allocation.values())),
                "optimization_method": "strategic"
            }

        except Exception as e:
            return {"error": f"Error in strategic allocation: {str(e)}"}

    async def _ensemble_optimization(self, optimizations: Dict[str, Any]) -> Dict[str, Any]:
        """Combine multiple optimization approaches"""
        try:
            # Extract valid optimizations
            valid_optimizations = {k: v for k, v in optimizations.items()
                                   if not v.get('error') and v.get('allocation')}

            if not valid_optimizations:
                return {"error": "No valid optimizations to ensemble"}

            # Weights for different approaches
            approach_weights = {
                'strategic': 0.4,  # Highest weight to strategy-based
                'black_litterman': 0.3,  # Second highest to BL
                'mean_variance': 0.2,  # Moderate weight to MV
                'risk_parity': 0.1  # Lowest weight to risk parity
            }

            # Get all unique symbols
            all_symbols = set()
            for opt in valid_optimizations.values():
                all_symbols.update(opt['allocation'].keys())

            # Ensemble allocation
            ensemble_allocation = {}

            for symbol in all_symbols:
                weighted_allocation = 0
                total_approach_weight = 0

                for approach, opt_result in valid_optimizations.items():
                    if approach in approach_weights:
                        symbol_weight = opt_result['allocation'].get(symbol, 0)
                        weighted_allocation += symbol_weight * approach_weights[approach]
                        total_approach_weight += approach_weights[approach]

                if total_approach_weight > 0:
                    ensemble_allocation[symbol] = weighted_allocation / total_approach_weight
                else:
                    ensemble_allocation[symbol] = 0

            # Normalize ensemble allocation
            total_ensemble_weight = sum(ensemble_allocation.values())
            if total_ensemble_weight > 0:
                target_leverage = 0.95  # Leave 5% cash
                ensemble_allocation = {k: v / total_ensemble_weight * target_leverage
                                       for k, v in ensemble_allocation.items()}

            # Calculate ensemble metrics
            valid_returns = [opt.get('expected_return', 0) for opt in valid_optimizations.values()
                             if opt.get('expected_return') is not None]
            valid_vols = [opt.get('expected_volatility', 0) for opt in valid_optimizations.values()
                          if opt.get('expected_volatility') is not None]

            ensemble_return = float(np.mean(valid_returns)) if valid_returns else 0.0
            ensemble_vol = float(np.mean(valid_vols)) if valid_vols else 0.0
            ensemble_sharpe = ensemble_return / ensemble_vol if ensemble_vol > 0 else 0

            return {
                "allocation": ensemble_allocation,
                "expected_return": ensemble_return,
                "expected_volatility": ensemble_vol,
                "sharpe_ratio": float(ensemble_sharpe),
                "cash_allocation": max(0.05, 1.0 - sum(ensemble_allocation.values())),
                "optimization_method": "ensemble",
                "component_weights": approach_weights,
                "valid_approaches": list(valid_optimizations.keys())
            }

        except Exception as e:
            return {"error": f"Error in ensemble optimization: {str(e)}"}

    async def _calculate_portfolio_analytics(self, allocation: Dict[str, Any],
                                             inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive portfolio analytics"""
        try:
            if allocation.get('error'):
                return {"error": "Cannot calculate analytics for invalid allocation"}

            portfolio_allocation = allocation.get('allocation', {})
            symbols = inputs['symbols']
            expected_returns = inputs['expected_returns']
            cov_matrix = inputs['covariance_matrix']
            risk_scores = inputs['risk_scores']

            # Portfolio weights array
            weights = np.array([portfolio_allocation.get(symbol, 0) for symbol in symbols])

            # Basic portfolio metrics
            portfolio_return = float(np.dot(weights, expected_returns))
            portfolio_variance = float(np.dot(weights, np.dot(cov_matrix, weights)))
            portfolio_vol = float(np.sqrt(portfolio_variance))
            sharpe_ratio = portfolio_return / portfolio_vol if portfolio_vol > 0 else 0

            # Risk decomposition
            marginal_contrib = np.dot(cov_matrix, weights)
            contrib_to_risk = weights * marginal_contrib
            percent_contrib = contrib_to_risk / portfolio_variance if portfolio_variance > 0 else np.zeros_like(
                contrib_to_risk)

            # Diversification metrics
            diversification_ratio = self._calculate_diversification_ratio(weights, cov_matrix)
            effective_num_assets = self._calculate_effective_num_assets(weights)

            # Concentration metrics
            herfindahl_index = float(np.sum(weights ** 2))
            max_weight = float(np.max(weights)) if len(weights) > 0 else 0

            # Risk analytics by symbol
            symbol_analytics = {}
            for i, symbol in enumerate(symbols):
                if weights[i] > 0:
                    symbol_analytics[symbol] = {
                        "weight": float(weights[i]),
                        "expected_return": float(expected_returns[i]),
                        "risk_contribution": float(percent_contrib[i]),
                        "marginal_risk": float(marginal_contrib[i]),
                        "risk_score": float(risk_scores[i])
                    }

            # Value at Risk estimation (parametric)
            var_95 = portfolio_return - 1.645 * portfolio_vol  # 95% VaR
            var_99 = portfolio_return - 2.326 * portfolio_vol  # 99% VaR

            return {
                "portfolio_metrics": {
                    "expected_return": portfolio_return,
                    "expected_volatility": portfolio_vol,
                    "sharpe_ratio": float(sharpe_ratio),
                    "var_95": float(var_95),
                    "var_99": float(var_99)
                },
                "diversification_metrics": {
                    "diversification_ratio": float(diversification_ratio),
                    "effective_num_assets": float(effective_num_assets),
                    "herfindahl_index": herfindahl_index,
                    "max_weight": max_weight
                },
                "risk_decomposition": {
                    "symbol_contributions": symbol_analytics,
                    "total_risk": portfolio_variance
                },
                "allocation_summary": {
                    "number_of_positions": len([w for w in weights if w > 0.01]),
                    "total_equity_weight": sum(portfolio_allocation.values()),
                    "cash_weight": allocation.get('cash_allocation', 0),
                    "largest_position": max(portfolio_allocation.values()) if portfolio_allocation else 0
                }
            }

        except Exception as e:
            return {"error": f"Error calculating portfolio analytics: {str(e)}"}

    async def _generate_rebalancing_plan(self, allocation: Dict[str, Any],
                                         strategy_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate rebalancing plan and recommendations"""
        try:
            if allocation.get('error'):
                return {"error": "Cannot generate rebalancing plan for invalid allocation"}

            target_allocation = allocation.get('allocation', {})

            # Rebalancing strategy
            rebalancing_plan = {
                "rebalancing_frequency": "Monthly with 10% threshold trigger",
                "rebalancing_method": "Threshold-based with calendar override",
                "transaction_cost_consideration": True,
                "tax_loss_harvesting": True
            }

            # Rebalancing triggers
            triggers = [
                "Individual position deviates >10% from target",
                "Cash allocation falls below 5%",
                "Market regime change detected",
                "Fundamental rating change for major holding",
                "Monthly calendar rebalancing"
            ]

            # Implementation guidelines
            implementation = {
                "trade_execution": "Spread over 2-3 days to minimize market impact",
                "order_types": "Use limit orders with 5-minute time limits",
                "cost_minimization": "Prioritize trades with highest deviation",
                "cash_management": "Maintain 5-10% cash buffer"
            }

            # Monitoring requirements
            monitoring = {
                "daily_monitoring": [
                    "Position deviations from target",
                    "Cash levels and liquidity",
                    "Market regime indicators"
                ],
                "weekly_review": [
                    "Strategy performance attribution",
                    "Risk metrics and VaR",
                    "Correlation changes"
                ],
                "monthly_review": [
                    "Full portfolio rebalancing",
                    "Strategy updates",
                    "Performance evaluation"
                ]
            }

            return {
                "rebalancing_strategy": rebalancing_plan,
                "triggers": triggers,
                "implementation_guidelines": implementation,
                "monitoring_requirements": monitoring,
                "target_allocation": target_allocation,
                "cash_target": allocation.get('cash_allocation', 0.05)
            }

        except Exception as e:
            return {"error": f"Error generating rebalancing plan: {str(e)}"}

    # Helper methods
    def _strategy_to_expected_return(self, strategy: str, confidence: float) -> float:
        """Convert strategy recommendation to expected return"""
        base_returns = {
            'strong_sell': -0.15,
            'sell': -0.08,
            'reduce': -0.03,
            'hold': 0.02,
            'accumulate': 0.05,
            'buy': 0.10,
            'aggressive_buy': 0.18
        }

        base_return = base_returns.get(strategy, 0.02)
        confidence_multiplier = 0.5 + (confidence / 100)  # 0.5 to 1.5
        return base_return * confidence_multiplier

    def _risk_score_to_volatility(self, risk_score: float) -> float:
        """Convert risk score to annualized volatility"""
        # Risk score 0-100 maps to volatility 10%-50%
        min_vol = 0.10
        max_vol = 0.50

        # Normalize risk score to 0-1
        normalized_risk = risk_score / 100

        # Map to volatility range
        volatility = min_vol + (max_vol - min_vol) * normalized_risk
        return volatility

    def _estimate_correlation_matrix(self, symbols: List[str], risk_scores: List[float]) -> np.ndarray:
        """Estimate correlation matrix based on risk scores and sector similarity"""
        n = len(symbols)
        correlation_matrix = np.eye(n)

        # Base correlation (simplified approach)
        base_correlation = 0.3  # Typical equity correlation

        for i in range(n):
            for j in range(i + 1, n):
                # Correlation based on risk similarity
                risk_similarity = 1 - abs(risk_scores[i] - risk_scores[j]) / 100
                correlation = base_correlation * (0.5 + 0.5 * risk_similarity)
                correlation_matrix[i, j] = correlation
                correlation_matrix[j, i] = correlation

        return correlation_matrix

    def _get_strategy_multiplier(self, strategy: str) -> float:
        """Get multiplier based on strategy conviction"""
        multipliers = {
            'strong_sell': 0.2,
            'sell': 0.4,
            'reduce': 0.6,
            'hold': 1.0,
            'accumulate': 1.2,
            'buy': 1.5,
            'aggressive_buy': 2.0
        }

        return multipliers.get(strategy, 1.0)

    def _calculate_diversification_ratio(self, weights: np.ndarray, cov_matrix: np.ndarray) -> float:
        """Calculate diversification ratio"""
        try:
            # Portfolio volatility
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))

            # Weighted average individual volatilities
            individual_vols = np.sqrt(np.diag(cov_matrix))
            weighted_avg_vol = np.dot(weights, individual_vols)

            # Diversification ratio
            diversification_ratio = weighted_avg_vol / portfolio_vol if portfolio_vol > 0 else 1.0
            return diversification_ratio

        except Exception:
            return 1.0

    def _calculate_effective_num_assets(self, weights: np.ndarray) -> float:
        """Calculate effective number of assets (inverse Herfindahl index)"""
        try:
            herfindahl = np.sum(weights ** 2)
            effective_num = 1 / herfindahl if herfindahl > 0 else len(weights)
            return effective_num
        except Exception:
            return float(len(weights))

    def _generate_optimization_summary(self, optimizations: Dict[str, Any],
                                       ensemble: Dict[str, Any]) -> Dict[str, Any]:
        """Generate optimization summary"""
        try:
            valid_optimizations = {k: v for k, v in optimizations.items() if not v.get('error')}

            # Compare Sharpe ratios
            sharpe_comparison = {}
            for method, result in valid_optimizations.items():
                sharpe_comparison[method] = result.get('sharpe_ratio', 0)

            best_sharpe_method = max(sharpe_comparison.keys(),
                                     key=lambda k: sharpe_comparison[k]) if sharpe_comparison else 'ensemble'

            return {
                "total_optimizations": len(optimizations),
                "successful_optimizations": len(valid_optimizations),
                "optimization_methods": list(valid_optimizations.keys()),
                "sharpe_ratio_comparison": sharpe_comparison,
                "best_individual_method": best_sharpe_method,
                "ensemble_sharpe": ensemble.get('sharpe_ratio', 0),
                "recommended_approach": "ensemble",
                "optimization_timestamp": datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            return {"error": f"Error generating optimization summary: {str(e)}"}


class PortfolioOptimizerAgent(Agent):
    """Specialized agent for portfolio optimization"""

    def __init__(self, gcp_services, config):
        # Use explicit parent class reference for Python 2.7 compatibility
        Agent.__init__(
            self,
            model="gemini-2.0-flash-exp",
            name="portfolio_optimizer",
            description="Expert portfolio optimizer using advanced optimization techniques",
            tools=[PortfolioOptimizationTool()]
        )

        self.gcp_services = gcp_services
        self.config = config
        logger.info("PortfolioOptimizerAgent initialized")

    async def optimize_portfolio(self, strategy_data: Dict[str, Any],
                                 risk_data: Dict[str, Any],
                                 constraints: Dict[str, Any] = None) -> Dict[str, Any]:
        """Main method for portfolio optimization"""
        try:
            logger.info("Starting portfolio optimization process")

            # Set default constraints if None provided
            if constraints is None:
                constraints = {}

            optimization_result = await self.tools[0].call(
                strategy_data=strategy_data,
                risk_data=risk_data,
                constraints=constraints
            )

            # Store results
            if hasattr(self.gcp_services, 'store_portfolio_optimization'):
                await self.gcp_services.store_portfolio_optimization(optimization_result)
            if hasattr(self.gcp_services, 'store_bigquery_optimization_data'):
                await self.gcp_services.store_bigquery_optimization_data(optimization_result)

            # Add metadata
            optimization_result["optimization_metadata"] = {
                "agent": self.name,
                "optimization_time": datetime.now(timezone.utc).isoformat(),
                "constraints_applied": constraints is not None and len(constraints) > 0,
                "quality_score": self._calculate_optimization_quality_score(optimization_result)
            }

            logger.info("Portfolio optimization completed successfully")
            return optimization_result

        except Exception as e:
            error_msg = f"Error in portfolio optimization: {str(e)}"
            logger.error(error_msg)
            return {
                "status": "error",
                "error": error_msg,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "agent": self.name
            }

    def _calculate_optimization_quality_score(self, optimization_result: Dict[str, Any]) -> float:
        """Calculate quality score for portfolio optimization"""
        if optimization_result.get('status') != 'success':
            return 0.0

        score = 0

        # Successful optimization approaches
        optimizations = optimization_result.get('optimization_approaches', {})
        successful_opts = len([opt for opt in optimizations.values() if not opt.get('error')])
        total_opts = len(optimizations)
        if total_opts > 0:
            approach_score = (successful_opts / total_opts) * 40
            score += approach_score

        # Ensemble optimization quality
        ensemble = optimization_result.get('recommended_allocation', {})
        if ensemble and not ensemble.get('error'):
            score += 25

        # Sharpe ratio quality
        sharpe = ensemble.get('sharpe_ratio', 0)
        if sharpe > 0.5:
            score += 15
        elif sharpe > 0.3:
            score += 10
        elif sharpe > 0.1:
            score += 5

        # Portfolio analytics
        analytics = optimization_result.get('portfolio_analytics', {})
        if analytics and not analytics.get('error'):
            score += 15

        # Rebalancing plan
        rebalancing = optimization_result.get('rebalancing_plan', {})
        if rebalancing and not rebalancing.get('error'):
            score += 5

        return round(min(100.0, score), 2)
