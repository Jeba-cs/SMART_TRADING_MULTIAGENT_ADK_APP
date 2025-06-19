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
        def __init__(self, model=None, name=None, description=None, instructions=None, tools=None):
            self.model = model
            self.name = name
            self.description = description
            self.instructions = instructions
            self.tools = tools or []

logger = logging.getLogger(__name__)


class ExecutionPlanningTool(Tool):
    """Tool for planning and optimizing trade execution"""

    def __init__(self):
        # Use explicit parent class reference for Python 2.7 compatibility
        Tool.__init__(
            self,
            name="execution_planner",
            description="Plan and optimize trade execution based on portfolio allocation, market conditions, and liquidity"
        )

    async def call(self, portfolio_allocation: dict, market_data: dict,
                   current_portfolio: dict = None) -> dict:
        """
        Plan and optimize trade execution
        Args:
            portfolio_allocation: Target portfolio allocation
            market_data: Current market data for execution planning
            current_portfolio: Optional current portfolio holdings
        Returns:
            Detailed execution plan
        """
        try:
            logger.info("Starting execution planning")
            # Extract target allocation
            target_allocation = portfolio_allocation.get('recommended_allocation', {}).get('allocation', {})
            if not target_allocation:
                return {
                    "status": "error",
                    "error": "No valid target allocation provided",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }

            # Extract market data
            market_data_dict = market_data.get('data', {})
            # Generate trade list
            trade_list = await self._generate_trade_list(
                target_allocation, current_portfolio, market_data_dict
            )

            # Optimize execution strategy
            execution_strategy = await self._optimize_execution_strategy(
                trade_list, market_data_dict
            )

            # Generate order specifications
            order_specifications = await self._generate_order_specifications(
                trade_list, execution_strategy, market_data_dict
            )

            # Risk controls
            risk_controls = await self._define_execution_risk_controls(
                trade_list, market_data_dict
            )

            # Execution schedule
            execution_schedule = await self._create_execution_schedule(
                trade_list, execution_strategy
            )

            # Post-trade analysis plan
            post_trade_plan = await self._create_post_trade_analysis_plan(
                trade_list, execution_strategy
            )

            return {
                "status": "success",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "trade_list": trade_list,
                "execution_strategy": execution_strategy,
                "order_specifications": order_specifications,
                "risk_controls": risk_controls,
                "execution_schedule": execution_schedule,
                "post_trade_analysis_plan": post_trade_plan,
                "execution_summary": self._generate_execution_summary(trade_list, execution_strategy)
            }

        except Exception as e:
            error_msg = f"Error in execution planning: {str(e)}"
            logger.error(error_msg)
            return {
                "status": "error",
                "error": error_msg,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

    async def _generate_trade_list(self, target_allocation: dict,
                                   current_portfolio: dict,
                                   market_data: dict) -> dict:
        """Generate trade list based on target allocation and current portfolio"""
        try:
            trades = {
                "buy_trades": [],
                "sell_trades": [],
                "total_buy_value": 0,
                "total_sell_value": 0,
                "trade_count": 0
            }

            # Default portfolio value if not provided
            portfolio_value = 1000000  # $1M default
            # Current holdings (default to empty)
            current_holdings = {}
            if current_portfolio:
                current_holdings = current_portfolio.get('holdings', {})
                portfolio_value = current_portfolio.get('total_value', portfolio_value)

            # Calculate target position values
            target_positions = {}
            for symbol, weight in target_allocation.items():
                target_positions[symbol] = weight * portfolio_value

            # Calculate trades needed
            for symbol, target_value in target_positions.items():
                current_value = current_holdings.get(symbol, 0)
                # Get current price
                symbol_data = market_data.get(symbol, {})
                current_price = symbol_data.get('current_price', 0)
                if current_price <= 0:
                    continue  # Skip if no valid price

                # Calculate trade
                trade_value = target_value - current_value
                trade_shares = int(trade_value / current_price)
                if abs(trade_shares) < 1:
                    continue  # Skip tiny trades

                trade_info = {
                    "symbol": symbol,
                    "shares": abs(trade_shares),
                    "price": current_price,
                    "value": abs(trade_value),
                    "target_weight": target_allocation.get(symbol, 0),
                    "current_weight": current_value / portfolio_value if portfolio_value > 0 else 0,
                    "weight_change": (target_value - current_value) / portfolio_value if portfolio_value > 0 else 0
                }

                if trade_shares > 0:
                    trades["buy_trades"].append(trade_info)
                    trades["total_buy_value"] += abs(trade_value)
                else:
                    trades["sell_trades"].append(trade_info)
                    trades["total_sell_value"] += abs(trade_value)

            # Sort trades by value (largest first)
            trades["buy_trades"].sort(key=lambda x: x["value"], reverse=True)
            trades["sell_trades"].sort(key=lambda x: x["value"], reverse=True)
            trades["trade_count"] = len(trades["buy_trades"]) + len(trades["sell_trades"])

            return trades

        except Exception as e:
            return {"error": f"Error generating trade list: {str(e)}"}

    async def _optimize_execution_strategy(self, trade_list: dict,
                                           market_data: dict) -> dict:
        """Optimize execution strategy based on trade list and market conditions"""
        try:
            buy_trades = trade_list.get("buy_trades", [])
            sell_trades = trade_list.get("sell_trades", [])
            total_buy_value = trade_list.get("total_buy_value", 0)
            total_sell_value = trade_list.get("total_sell_value", 0)

            # Determine execution urgency
            execution_urgency = self._determine_execution_urgency(trade_list, market_data)

            # Determine execution method for each trade
            buy_execution_methods = []
            for trade in buy_trades:
                method = self._determine_execution_method(trade, market_data, "buy")
                buy_execution_methods.append({
                    "symbol": trade["symbol"],
                    "method": method,
                    "reason": self._get_method_reason(method, trade)
                })

            sell_execution_methods = []
            for trade in sell_trades:
                method = self._determine_execution_method(trade, market_data, "sell")
                sell_execution_methods.append({
                    "symbol": trade["symbol"],
                    "method": method,
                    "reason": self._get_method_reason(method, trade)
                })

            # Determine overall execution approach
            execution_approach = self._determine_execution_approach(
                buy_trades, sell_trades, execution_urgency
            )

            # Determine execution timing
            execution_timing = self._determine_execution_timing(
                buy_trades, sell_trades, execution_urgency
            )

            return {
                "execution_urgency": execution_urgency,
                "buy_execution_methods": buy_execution_methods,
                "sell_execution_methods": sell_execution_methods,
                "execution_approach": execution_approach,
                "execution_timing": execution_timing,
                "cash_management": self._determine_cash_management(total_buy_value, total_sell_value)
            }

        except Exception as e:
            return {"error": f"Error optimizing execution strategy: {str(e)}"}

    async def _generate_order_specifications(self, trade_list: dict,
                                             execution_strategy: dict,
                                             market_data: dict) -> dict:
        """Generate detailed order specifications"""
        try:
            buy_trades = trade_list.get("buy_trades", [])
            sell_trades = trade_list.get("sell_trades", [])
            buy_methods = execution_strategy.get("buy_execution_methods", [])
            sell_methods = execution_strategy.get("sell_execution_methods", [])

            # Create method lookup
            buy_method_lookup = {m["symbol"]: m["method"] for m in buy_methods}
            sell_method_lookup = {m["symbol"]: m["method"] for m in sell_methods}

            # Generate buy orders
            buy_orders = []
            for trade in buy_trades:
                symbol = trade["symbol"]
                method = buy_method_lookup.get(symbol, "market")
                order_spec = self._create_order_specification(
                    symbol, trade["shares"], "buy", method, market_data.get(symbol, {})
                )

                buy_orders.append(order_spec)

            # Generate sell orders
            sell_orders = []
            for trade in sell_trades:
                symbol = trade["symbol"]
                method = sell_method_lookup.get(symbol, "market")
                order_spec = self._create_order_specification(
                    symbol, trade["shares"], "sell", method, market_data.get(symbol, {})
                )

                sell_orders.append(order_spec)

            return {
                "buy_orders": buy_orders,
                "sell_orders": sell_orders,
                "order_count": len(buy_orders) + len(sell_orders)
            }

        except Exception as e:
            return {"error": f"Error generating order specifications: {str(e)}"}

    async def _define_execution_risk_controls(self, trade_list: dict,
                                              market_data: dict) -> dict:
        """Define risk controls for trade execution"""
        try:
            # Price deviation limits
            price_deviation_limits = {
                "market_orders": 0.02,  # 2% max deviation
                "limit_orders": 0.01,  # 1% from limit price
                "vwap_orders": 0.005  # 0.5% from VWAP
            }

            # Maximum order size limits
            max_order_size_limits = {
                "percent_of_adv": 0.10,  # 10% of ADV
                "percent_of_portfolio": 0.05,  # 5% of portfolio
                "absolute_value": 1000000  # $1M max single order
            }

            # Circuit breakers
            circuit_breakers = {
                "market_decline": 0.05,  # Pause if market drops 5%
                "volatility_spike": 0.03,  # Pause if volatility spikes 3%
                "spread_widening": 0.01  # Pause if spread widens 1%
            }

            # Execution quality monitoring
            execution_quality = {
                "slippage_threshold": 0.003,  # 30bps max acceptable slippage
                "implementation_shortfall_limit": 0.005,  # 50bps max shortfall
                "participation_rate_limit": 0.15  # 15% max participation
            }

            return {
                "price_deviation_limits": price_deviation_limits,
                "max_order_size_limits": max_order_size_limits,
                "circuit_breakers": circuit_breakers,
                "execution_quality_thresholds": execution_quality,
                "kill_switch_conditions": [
                    "Cumulative slippage exceeds 0.5%",
                    "Market declines more than 7%",
                    "Liquidity disappears (spread > 2%)",
                    "Execution rate falls below 25% of expected"
                ]
            }

        except Exception as e:
            return {"error": f"Error defining execution risk controls: {str(e)}"}

    async def _create_execution_schedule(self, trade_list: dict,
                                         execution_strategy: dict) -> dict:
        """Create execution schedule"""
        try:
            execution_approach = execution_strategy.get("execution_approach", "balanced")
            execution_timing = execution_strategy.get("execution_timing", {})

            # Default schedule
            default_schedule = {
                "timeframe": "1 day",
                "trading_hours": "9:30 AM - 4:00 PM ET",
                "execution_phases": [
                    {"time": "9:30 AM - 10:00 AM", "allocation": 0.15, "focus": "Sell orders"},
                    {"time": "10:00 AM - 11:30 AM", "allocation": 0.25, "focus": "Mixed orders"},
                    {"time": "11:30 AM - 2:00 PM", "allocation": 0.20, "focus": "Low urgency orders"},
                    {"time": "2:00 PM - 3:30 PM", "allocation": 0.25, "focus": "Mixed orders"},
                    {"time": "3:30 PM - 4:00 PM", "allocation": 0.15, "focus": "Buy orders"}
                ]
            }

            # Adjust based on execution approach
            if execution_approach == "aggressive":
                schedule = {
                    "timeframe": "Same day",
                    "trading_hours": "9:30 AM - 4:00 PM ET",
                    "execution_phases": [
                        {"time": "9:30 AM - 10:30 AM", "allocation": 0.30, "focus": "High priority orders"},
                        {"time": "10:30 AM - 12:00 PM", "allocation": 0.30, "focus": "Medium priority orders"},
                        {"time": "12:00 PM - 2:00 PM", "allocation": 0.20, "focus": "Low priority orders"},
                        {"time": "2:00 PM - 4:00 PM", "allocation": 0.20, "focus": "Remaining orders"}
                    ]
                }
            elif execution_approach == "passive":
                schedule = {
                    "timeframe": "1-3 days",
                    "trading_hours": "9:30 AM - 4:00 PM ET",
                    "execution_phases": [
                        {"time": "Day 1", "allocation": 0.40, "focus": "High priority orders"},
                        {"time": "Day 2", "allocation": 0.40, "focus": "Medium priority orders"},
                        {"time": "Day 3", "allocation": 0.20, "focus": "Low priority orders"}
                    ]
                }
            else:
                schedule = default_schedule

            # Add specific timing recommendations
            schedule["timing_recommendations"] = execution_timing

            # Add order prioritization
            schedule["order_prioritization"] = self._prioritize_orders(trade_list)

            return schedule

        except Exception as e:
            return {"error": f"Error creating execution schedule: {str(e)}"}

    async def _create_post_trade_analysis_plan(self, trade_list: dict,
                                               execution_strategy: dict) -> dict:
        """Create post-trade analysis plan"""
        try:
            return {
                "performance_metrics": [
                    "Implementation shortfall",
                    "VWAP slippage",
                    "Participation rate",
                    "Market impact",
                    "Timing cost",
                    "Opportunity cost"
                ],
                "analysis_timeframes": [
                    "Intraday (execution quality)",
                    "T+1 (immediate impact)",
                    "T+5 (short-term impact)",
                    "T+20 (medium-term impact)"
                ],
                "benchmarks": [
                    "Arrival price",
                    "VWAP",
                    "Close price",
                    "Implementation shortfall model"
                ],
                "reporting_requirements": [
                    "Trade execution summary",
                    "Slippage analysis by order type",
                    "Market condition impact",
                    "Strategy effectiveness evaluation"
                ]
            }

        except Exception as e:
            return {"error": f"Error creating post-trade analysis plan: {str(e)}"}

    # Helper methods
    def _determine_execution_urgency(self, trade_list: dict,
                                     market_data: dict) -> str:
        """Determine execution urgency based on trade list and market conditions"""
        buy_trades = trade_list.get("buy_trades", [])
        sell_trades = trade_list.get("sell_trades", [])

        # Check trade size
        total_trades = len(buy_trades) + len(sell_trades)
        large_trades = sum(1 for t in buy_trades + sell_trades if t.get("value", 0) > 500000)

        # Check market conditions
        market_summary = market_data.get("market_summary", {})
        vix_data = market_summary.get("^VIX", {})
        vix_level = vix_data.get("current", 20)

        # Determine urgency
        if large_trades > 5 or vix_level > 30:
            return "high"
        elif large_trades > 2 or vix_level > 20:
            return "medium"
        else:
            return "low"

    def _determine_execution_method(self, trade: dict,
                                    market_data: dict,
                                    side: str) -> str:
        """Determine execution method for a trade"""
        symbol = trade.get("symbol", "")
        shares = trade.get("shares", 0)
        value = trade.get("value", 0)
        symbol_data = market_data.get(symbol, {})
        avg_volume = symbol_data.get("trading_metrics", {}).get("avg_volume", 0)

        # Calculate participation rate
        participation = shares / avg_volume if avg_volume > 0 else 0

        # Large order (>5% of ADV)
        if participation > 0.05:
            return "vwap"
        # Medium order (1-5% of ADV)
        elif participation > 0.01:
            return "twap"
        # Small order with high value
        elif value > 100000:
            return "limit"
        # Very small order
        else:
            return "market"

    def _get_method_reason(self, method: str, trade: dict) -> str:
        """Get reason for execution method selection"""
        if method == "vwap":
            return f"Large order size ({trade.get('shares', 0)} shares) requires volume-weighted execution"
        elif method == "twap":
            return f"Medium order size with value ${trade.get('value', 0):,.0f} benefits from time-weighted execution"
        elif method == "limit":
            return f"Order value ${trade.get('value', 0):,.0f} warrants limit order to control execution price"
        else:
            return f"Small order size appropriate for market execution"

    def _determine_execution_approach(self, buy_trades: list,
                                      sell_trades: list,
                                      urgency: str) -> str:
        """Determine overall execution approach"""
        total_trades = len(buy_trades) + len(sell_trades)
        if urgency == "high" or total_trades > 20:
            return "aggressive"
        elif urgency == "low" and total_trades < 10:
            return "passive"
        else:
            return "balanced"

    def _determine_execution_timing(self, buy_trades: list,
                                    sell_trades: list,
                                    urgency: str) -> dict:
        """Determine execution timing recommendations"""
        if urgency == "high":
            return {
                "timeframe": "Same day execution",
                "preferred_timing": "Execute 70% in morning, 30% in afternoon",
                "avoid_periods": ["First 15 minutes", "Last 15 minutes"]
            }
        elif urgency == "medium":
            return {
                "timeframe": "1-2 day execution",
                "preferred_timing": "Balanced throughout day with focus on liquid periods",
                "avoid_periods": ["Market open", "Lunch hour", "Market close"]
            }
        else:
            return {
                "timeframe": "2-3 day execution",
                "preferred_timing": "Focus on midday liquidity",
                "avoid_periods": ["High volatility periods", "Economic announcements"]
            }

    @staticmethod
    def _determine_cash_management(total_buy_value: float, total_sell_value: float) -> dict:
        """Determine cash management approach"""
        net_cash_flow = total_sell_value - total_buy_value
        if net_cash_flow > 100000:  # Significant cash inflow
            return {
                "cash_flow": "positive",
                "net_amount": net_cash_flow,
                "recommendation": "Stage buy orders, invest excess cash in short-term instruments"
            }
        elif net_cash_flow < -100000:  # Significant cash outflow
            return {
                "cash_flow": "negative",
                "net_amount": net_cash_flow,
                "recommendation": "Execute sells first, ensure sufficient liquidity before buys"
            }
        else:
            return {
                "cash_flow": "neutral",
                "net_amount": net_cash_flow,
                "recommendation": "Balance buys and sells throughout execution window"
            }

    def _create_order_specification(self, symbol: str, shares: int, side: str,
                                    method: str, symbol_data: dict) -> dict:
        """Create detailed order specification"""
        current_price = symbol_data.get("current_price", 0)

        # Base order
        order_spec = {
            "symbol": symbol,
            "side": side,
            "quantity": shares,
            "order_type": method if method != "vwap" and method != "twap" else "algorithm",
            "algorithm": method if method == "vwap" or method == "twap" else None,
            "time_in_force": "day"
        }

        # Add price for limit orders
        if method == "limit":
            # Set limit price slightly aggressive
            limit_price = current_price * 1.01 if side == "buy" else current_price * 0.99
            order_spec["limit_price"] = limit_price

        # Add parameters for algorithmic orders
        if method == "vwap" or method == "twap":
            order_spec["parameters"] = {
                "start_time": "market_open",
                "end_time": "market_close",
                "max_participation_rate": 0.15,  # 15% max participation
                "min_completion_ratio": 0.95  # 95% minimum completion
            }

        return order_spec

    def _prioritize_orders(self, trade_list: dict) -> dict:
        """Prioritize orders for execution"""
        buy_trades = trade_list.get("buy_trades", [])
        sell_trades = trade_list.get("sell_trades", [])

        # Prioritize sells by value
        prioritized_sells = sorted(sell_trades, key=lambda x: x.get("value", 0), reverse=True)
        sell_priorities = [{"symbol": t["symbol"], "priority": i + 1, "reason": "Value-based"}
                           for i, t in enumerate(prioritized_sells[:10])]

        # Prioritize buys by value
        prioritized_buys = sorted(buy_trades, key=lambda x: x.get("value", 0), reverse=True)
        buy_priorities = [{"symbol": t["symbol"], "priority": i + 1, "reason": "Value-based"}
                          for i, t in enumerate(prioritized_buys[:10])]

        return {
            "sell_priorities": sell_priorities,
            "buy_priorities": buy_priorities,
            "execution_sequence": "Sells first, then buys"
        }

    def _generate_execution_summary(self, trade_list: dict,
                                    execution_strategy: dict) -> dict:
        """Generate execution summary"""
        buy_trades = trade_list.get("buy_trades", [])
        sell_trades = trade_list.get("sell_trades", [])

        return {
            "total_trades": len(buy_trades) + len(sell_trades),
            "buy_orders": len(buy_trades),
            "sell_orders": len(sell_trades),
            "total_buy_value": trade_list.get("total_buy_value", 0),
            "total_sell_value": trade_list.get("total_sell_value", 0),
            "execution_approach": execution_strategy.get("execution_approach", "balanced"),
            "execution_urgency": execution_strategy.get("execution_urgency", "medium"),
            "estimated_completion_time": self._estimate_completion_time(execution_strategy),
            "execution_timestamp": datetime.now(timezone.utc).isoformat()
        }

    def _estimate_completion_time(self, execution_strategy: dict) -> str:
        """Estimate execution completion time"""
        approach = execution_strategy.get("execution_approach", "balanced")
        if approach == "aggressive":
            return "Same trading day"
        elif approach == "passive":
            return "2-3 trading days"
        else:
            return "1-2 trading days"


class ExecutionPlannerAgent(Agent):
    """Specialized agent for trade execution planning"""

    def __init__(self, gcp_services, config):
        # Use explicit parent class reference for Python 2.7 compatibility
        Agent.__init__(
            self,
            model="gemini-2.0-flash-exp",
            name="execution_planner",
            description="Expert execution planner for optimal trade execution",
            instructions="""You are a senior execution trader with expertise in:
1. Optimal trade execution strategy design
2. Market microstructure and liquidity analysis
3. Transaction cost analysis and minimization
4. Order type selection and algorithmic trading
5. Execution timing and scheduling
6. Post-trade analysis and performance measurement

Your primary responsibilities:
- Design optimal execution strategies for portfolio trades
- Minimize market impact and transaction costs
- Select appropriate order types and algorithms
- Create detailed execution schedules and timing
- Implement risk controls and circuit breakers
- Plan post-trade analysis and performance measurement

Focus on creating execution plans that:
- Minimize market impact and information leakage
- Adapt to market conditions and liquidity
- Balance urgency with cost minimization
- Incorporate appropriate risk controls
- Enable comprehensive performance measurement""",
            tools=[ExecutionPlanningTool()]
        )

        self.gcp_services = gcp_services
        self.config = config
        logger.info("ExecutionPlannerAgent initialized")

    async def plan_execution(self, portfolio_allocation: dict,
                             market_data: dict,
                             current_portfolio: dict = None) -> dict:
        """Main method for execution planning"""
        try:
            logger.info("Starting execution planning process")

            execution_plan = await self.tools[0].call(
                portfolio_allocation=portfolio_allocation,
                market_data=market_data,
                current_portfolio=current_portfolio
            )

            # Store results
            if hasattr(self.gcp_services, 'store_execution_plan'):
                await self.gcp_services.store_execution_plan(execution_plan)
            if hasattr(self.gcp_services, 'store_bigquery_execution_data'):
                await self.gcp_services.store_bigquery_execution_data(execution_plan)

            # Add metadata
            execution_plan["execution_metadata"] = {
                "agent": self.name,
                "planning_time": datetime.now(timezone.utc).isoformat(),
                "current_portfolio_provided": current_portfolio is not None,
                "quality_score": self._calculate_execution_quality_score(execution_plan)
            }

            logger.info("Execution planning completed successfully")
            return execution_plan

        except Exception as e:
            error_msg = f"Error in execution planning: {str(e)}"
            logger.error(error_msg)
            return {
                "status": "error",
                "error": error_msg,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "agent": self.name
            }

    @staticmethod
    def _calculate_execution_quality_score(execution_plan: dict) -> float:
        """Calculate quality score for execution planning"""
        if execution_plan.get('status') != 'success':
            return 0.0

        score = 0

        # Trade list generation
        trade_list = execution_plan.get('trade_list', {})
        if trade_list and not trade_list.get('error'):
            trade_count = trade_list.get('trade_count', 0)
            if trade_count > 0:
                score += 25
            else:
                score += 10

        # Execution strategy
        strategy = execution_plan.get('execution_strategy', {})
        if strategy and not strategy.get('error'):
            score += 25

        # Order specifications
        orders = execution_plan.get('order_specifications', {})
        if orders and not orders.get('error'):
            score += 20

        # Risk controls
        controls = execution_plan.get('risk_controls', {})
        if controls and not controls.get('error'):
            score += 15

        # Execution schedule
        schedule = execution_plan.get('execution_schedule', {})
        if schedule and not schedule.get('error'):
            score += 15

        return round(min(100.0, score), 2)
