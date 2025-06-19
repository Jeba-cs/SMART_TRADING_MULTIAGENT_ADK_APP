import logging
from datetime import datetime, timezone

# Python version compatibility check (requires 3.7+)
import sys

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


class ComplianceMonitoringTool(Tool):
    """Tool for monitoring trading compliance and regulatory requirements"""

    def __init__(self):
        # Use explicit parent class reference for compatibility
        Tool.__init__(
            self,
            name="compliance_monitor",
            description="Monitor trading compliance, regulatory requirements, and internal risk limits"
        )

    async def call(self, portfolio_data: dict[str, any], execution_plan: dict[str, any],
                   risk_data: dict[str, any]) -> dict[str, any]:
        """
        Monitor trading compliance and regulatory requirements
        Args:
            portfolio_data: Current and target portfolio data
            execution_plan: Planned trade execution
            risk_data: Risk analysis data
        Returns:
            Compliance monitoring results
        """
        try:
            logger.info("Starting compliance monitoring")
            # Extract portfolio data
            current_portfolio = portfolio_data.get('current_portfolio', {})
            target_portfolio = portfolio_data.get('target_portfolio', {})

            # Extract execution data
            trade_list = execution_plan.get('trade_list', {})

            # Regulatory compliance checks
            regulatory_compliance = await self._check_regulatory_compliance(
                current_portfolio, target_portfolio, trade_list
            )

            # Internal policy compliance
            internal_compliance = await self._check_internal_compliance(
                current_portfolio, target_portfolio, trade_list, risk_data
            )

            # Risk limit compliance
            risk_compliance = await self._check_risk_limits(
                current_portfolio, target_portfolio, risk_data
            )

            # Trading restrictions
            trading_restrictions = await self._check_trading_restrictions(
                trade_list
            )

            # Compliance issues and recommendations
            compliance_issues = await self._identify_compliance_issues(
                regulatory_compliance, internal_compliance, risk_compliance, trading_restrictions
            )

            # Compliance recommendations
            compliance_recommendations = await self._generate_compliance_recommendations(
                compliance_issues, trade_list, target_portfolio
            )

            return {
                "status": "success",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "regulatory_compliance": regulatory_compliance,
                "internal_compliance": internal_compliance,
                "risk_compliance": risk_compliance,
                "trading_restrictions": trading_restrictions,
                "compliance_issues": compliance_issues,
                "compliance_recommendations": compliance_recommendations,
                "compliance_summary": self._generate_compliance_summary(compliance_issues)
            }

        except Exception as e:
            error_msg = f"Error in compliance monitoring: {str(e)}"
            logger.error(error_msg)
            return {
                "status": "error",
                "error": error_msg,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

    async def _check_regulatory_compliance(self, current_portfolio: dict[str, any],
                                           target_portfolio: dict[str, any],
                                           trade_list: dict[str, any]) -> dict[str, any]:
        """Check regulatory compliance"""
        try:
            # Position limits check
            position_limits = {
                "excessive_concentration": False,
                "reportable_positions": [],
                "ownership_thresholds": []
            }

            # Market manipulation check
            market_manipulation = {
                "wash_trades": False,
                "marking_the_close": False,
                "spoofing_risk": False
            }

            # Insider trading check
            insider_trading = {
                "blackout_period_trades": [],
                "material_nonpublic_info_risk": False
            }

            # Short selling restrictions
            short_selling = {
                "locate_requirements": True,
                "circuit_breaker_restrictions": False,
                "hard_to_borrow_securities": []
            }

            # Regulatory reporting requirements
            reporting_requirements = {
                "large_trader_reporting": False,
                "form_13f_requirements": False,
                "form_13d_requirements": False,
                "form_13g_requirements": False
            }

            return {
                "position_limits": position_limits,
                "market_manipulation": market_manipulation,
                "insider_trading": insider_trading,
                "short_selling": short_selling,
                "reporting_requirements": reporting_requirements,
                "compliance_status": "compliant"
            }

        except Exception as e:
            return {"error": f"Error checking regulatory compliance: {str(e)}"}

    async def _check_internal_compliance(self, current_portfolio: dict[str, any],
                                         target_portfolio: dict[str, any],
                                         trade_list: dict[str, any],
                                         risk_data: dict[str, any]) -> dict[str, any]:
        """Check internal compliance policies"""
        try:
            # Concentration limits
            concentration_limits = {
                "single_position_limit": 0.10,  # 10% max
                "sector_concentration_limit": 0.25,  # 25% max
                "violations": []
            }

            # Check single position concentration
            target_positions = target_portfolio.get('positions', {})
            for symbol, position in target_positions.items():
                weight = position.get('weight', 0)
                if weight > concentration_limits["single_position_limit"]:
                    concentration_limits["violations"].append({
                        "type": "single_position",
                        "symbol": symbol,
                        "weight": weight,
                        "limit": concentration_limits["single_position_limit"]
                    })

            # Liquidity requirements
            liquidity_requirements = {
                "max_days_to_liquidate": 5,
                "max_market_impact": 0.03,  # 3% max impact
                "violations": []
            }

            # Trading restrictions
            trading_restrictions = {
                "restricted_securities": [],
                "prohibited_transactions": [],
                "violations": []
            }

            # Best execution requirements
            best_execution = {
                "price_improvement_required": True,
                "broker_diversification_required": True,
                "transaction_cost_analysis_required": True,
                "violations": []
            }

            return {
                "concentration_limits": concentration_limits,
                "liquidity_requirements": liquidity_requirements,
                "trading_restrictions": trading_restrictions,
                "best_execution": best_execution,
                "compliance_status": "compliant" if not concentration_limits["violations"] else "violations_detected"
            }

        except Exception as e:
            return {"error": f"Error checking internal compliance: {str(e)}"}

    async def _check_risk_limits(self, current_portfolio: dict[str, any],
                                 target_portfolio: dict[str, any],
                                 risk_data: dict[str, any]) -> dict[str, any]:
        """Check risk limit compliance"""
        try:
            # Portfolio risk limits
            portfolio_risk_limits = {
                "var_limit": 0.02,  # 2% daily VaR limit
                "volatility_limit": 0.15,  # 15% annualized volatility limit
                "drawdown_limit": 0.10,  # 10% max drawdown limit
                "violations": []
            }

            # Check VaR limit
            portfolio_analytics = risk_data.get('portfolio_risk_analysis', {}).get('portfolio_metrics', {})
            var_95 = portfolio_analytics.get('var_95', 0)
            if abs(var_95) > portfolio_risk_limits["var_limit"]:
                portfolio_risk_limits["violations"].append({
                    "type": "var_limit",
                    "value": abs(var_95),
                    "limit": portfolio_risk_limits["var_limit"]
                })

            # Leverage limits
            leverage_limits = {
                "gross_leverage_limit": 1.5,  # 1.5x max gross leverage
                "net_leverage_limit": 1.0,  # 1.0x max net leverage
                "violations": []
            }

            # Counterparty risk limits
            counterparty_limits = {
                "max_exposure_per_counterparty": 0.20,  # 20% max exposure
                "violations": []
            }

            # Liquidity risk limits
            liquidity_limits = {
                "min_cash_reserve": 0.05,  # 5% minimum cash
                "max_illiquid_assets": 0.15,  # 15% max illiquid assets
                "violations": []
            }

            return {
                "portfolio_risk_limits": portfolio_risk_limits,
                "leverage_limits": leverage_limits,
                "counterparty_limits": counterparty_limits,
                "liquidity_limits": liquidity_limits,
                "compliance_status": "compliant" if not portfolio_risk_limits["violations"] else "violations_detected"
            }

        except Exception as e:
            return {"error": f"Error checking risk limits: {str(e)}"}

    async def _check_trading_restrictions(self, trade_list: dict[str, any]) -> dict[str, any]:
        """Check trading restrictions"""
        try:
            # Restricted securities list (example)
            restricted_securities = []
            # Prohibited transactions
            prohibited_transactions = []

            # Check buy trades
            buy_trades = trade_list.get('buy_trades', [])
            for trade in buy_trades:
                symbol = trade.get('symbol', '')
                if symbol in restricted_securities:
                    prohibited_transactions.append({
                        "symbol": symbol,
                        "side": "buy",
                        "reason": "Security on restricted list"
                    })

            # Check sell trades
            sell_trades = trade_list.get('sell_trades', [])
            for trade in sell_trades:
                symbol = trade.get('symbol', '')
                if symbol in restricted_securities:
                    prohibited_transactions.append({
                        "symbol": symbol,
                        "side": "sell",
                        "reason": "Security on restricted list"
                    })

            return {
                "restricted_securities": restricted_securities,
                "prohibited_transactions": prohibited_transactions,
                "compliance_status": "compliant" if not prohibited_transactions else "violations_detected"
            }

        except Exception as e:
            return {"error": f"Error checking trading restrictions: {str(e)}"}

    async def _identify_compliance_issues(self, regulatory_compliance: dict[str, any],
                                          internal_compliance: dict[str, any],
                                          risk_compliance: dict[str, any],
                                          trading_restrictions: dict[str, any]) -> list[dict[str, any]]:
        """Identify compliance issues"""
        try:
            compliance_issues = []

            # Check regulatory compliance
            if regulatory_compliance.get('compliance_status') != 'compliant':
                # Position limits
                position_limits = regulatory_compliance.get('position_limits', {})
                if position_limits.get('excessive_concentration'):
                    compliance_issues.append({
                        "category": "regulatory",
                        "type": "position_limits",
                        "severity": "high",
                        "description": "Excessive concentration detected in portfolio"
                    })

                # Short selling
                short_selling = regulatory_compliance.get('short_selling', {})
                if not short_selling.get('locate_requirements'):
                    compliance_issues.append({
                        "category": "regulatory",
                        "type": "short_selling",
                        "severity": "high",
                        "description": "Short sale locate requirements not met"
                    })

            # Check internal compliance
            concentration_limits = internal_compliance.get('concentration_limits', {})
            violations = concentration_limits.get('violations', [])
            for violation in violations:
                compliance_issues.append({
                    "category": "internal",
                    "type": "concentration_limit",
                    "severity": "medium",
                    "description": f"Position {violation.get('symbol')} exceeds concentration limit of {violation.get('limit') * 100}%",
                    "details": violation
                })

            # Check risk compliance
            portfolio_risk_limits = risk_compliance.get('portfolio_risk_limits', {})
            risk_violations = portfolio_risk_limits.get('violations', [])
            for violation in risk_violations:
                compliance_issues.append({
                    "category": "risk",
                    "type": violation.get('type'),
                    "severity": "high",
                    "description": f"Portfolio {violation.get('type')} of {violation.get('value') * 100:.2f}% exceeds limit of {violation.get('limit') * 100:.2f}%",
                    "details": violation
                })

            # Check trading restrictions
            prohibited = trading_restrictions.get('prohibited_transactions', [])
            for transaction in prohibited:
                compliance_issues.append({
                    "category": "trading",
                    "type": "prohibited_transaction",
                    "severity": "high",
                    "description": f"Prohibited {transaction.get('side')} transaction for {transaction.get('symbol')}: {transaction.get('reason')}",
                    "details": transaction
                })

            return compliance_issues

        except Exception as e:
            logger.error(f"Error identifying compliance issues: {str(e)}")
            return []

    async def _generate_compliance_recommendations(self, compliance_issues: list[dict[str, any]],
                                                   trade_list: dict[str, any],
                                                   target_portfolio: dict[str, any]) -> dict[str, any]:
        """Generate compliance recommendations"""
        try:
            if not compliance_issues:
                return {
                    "status": "compliant",
                    "actions_required": [],
                    "documentation_required": ["Standard trade documentation"]
                }

            # Group issues by category
            regulatory_issues = [i for i in compliance_issues if i.get('category') == 'regulatory']
            internal_issues = [i for i in compliance_issues if i.get('category') == 'internal']
            risk_issues = [i for i in compliance_issues if i.get('category') == 'risk']
            trading_issues = [i for i in compliance_issues if i.get('category') == 'trading']

            # Generate recommendations
            actions_required = []
            documentation_required = ["Standard trade documentation"]

            # Regulatory recommendations
            for issue in regulatory_issues:
                if issue.get('type') == 'position_limits':
                    actions_required.append("Reduce position sizes to comply with regulatory limits")
                elif issue.get('type') == 'short_selling':
                    actions_required.append("Ensure proper locate requirements before short selling")
                    documentation_required.append("Short sale locate documentation")

            # Internal policy recommendations
            for issue in internal_issues:
                if issue.get('type') == 'concentration_limit':
                    symbol = issue.get('details', {}).get('symbol')
                    limit = issue.get('details', {}).get('limit', 0.10)
                    actions_required.append(f"Reduce position in {symbol} to below {limit * 100:.1f}% concentration")

            # Risk limit recommendations
            for issue in risk_issues:
                if issue.get('type') == 'var_limit':
                    actions_required.append("Reduce portfolio risk by adjusting position sizes or adding hedges")
                    documentation_required.append("Risk reduction plan documentation")

            # Trading restriction recommendations
            for issue in trading_issues:
                if issue.get('type') == 'prohibited_transaction':
                    symbol = issue.get('details', {}).get('symbol')
                    actions_required.append(f"Remove {symbol} from trade list due to trading restrictions")

            return {
                "status": "violations_detected",
                "actions_required": actions_required,
                "documentation_required": documentation_required,
                "escalation_required": len(regulatory_issues) > 0 or len(trading_issues) > 0,
                "pre_clearance_required": len(regulatory_issues) > 0 or len(internal_issues) > 0
            }

        except Exception as e:
            return {"error": f"Error generating compliance recommendations: {str(e)}"}

    def _generate_compliance_summary(self, compliance_issues: list[dict[str, any]]) -> dict[str, any]:
        """Generate compliance summary"""
        try:
            # Count issues by category and severity
            regulatory_count = len([i for i in compliance_issues if i.get('category') == 'regulatory'])
            internal_count = len([i for i in compliance_issues if i.get('category') == 'internal'])
            risk_count = len([i for i in compliance_issues if i.get('category') == 'risk'])
            trading_count = len([i for i in compliance_issues if i.get('category') == 'trading'])

            high_severity = len([i for i in compliance_issues if i.get('severity') == 'high'])
            medium_severity = len([i for i in compliance_issues if i.get('severity') == 'medium'])
            low_severity = len([i for i in compliance_issues if i.get('severity') == 'low'])

            # Overall compliance status
            if high_severity > 0:
                compliance_status = "critical_violations"
            elif medium_severity > 0:
                compliance_status = "moderate_violations"
            elif low_severity > 0:
                compliance_status = "minor_violations"
            else:
                compliance_status = "compliant"

            # Execution approval
            if high_severity > 0:
                execution_approval = "blocked"
            elif medium_severity > 0:
                execution_approval = "requires_modification"
            else:
                execution_approval = "approved"

            return {
                "total_issues": len(compliance_issues),
                "issues_by_category": {
                    "regulatory": regulatory_count,
                    "internal": internal_count,
                    "risk": risk_count,
                    "trading": trading_count
                },
                "issues_by_severity": {
                    "high": high_severity,
                    "medium": medium_severity,
                    "low": low_severity
                },
                "compliance_status": compliance_status,
                "execution_approval": execution_approval,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            return {"error": f"Error generating compliance summary: {str(e)}"}


class ComplianceMonitorAgent(Agent):
    """Specialized agent for compliance monitoring"""

    def __init__(self, gcp_services, config):
        # Use explicit parent class reference for compatibility
        Agent.__init__(
            self,
            model="gemini-2.0-flash-exp",
            name="compliance_monitor",
            description="Expert compliance monitor for trading regulatory and internal policy compliance",
            tools=[ComplianceMonitoringTool()]
        )

        self.gcp_services = gcp_services
        self.config = config
        logger.info("ComplianceMonitorAgent initialized")

    async def monitor_compliance(self, portfolio_data: dict[str, any],
                                 execution_plan: dict[str, any],
                                 risk_data: dict[str, any]) -> dict[str, any]:
        """Main method for compliance monitoring"""
        try:
            logger.info("Starting compliance monitoring process")

            compliance_result = await self.tools[0].call(
                portfolio_data=portfolio_data,
                execution_plan=execution_plan,
                risk_data=risk_data
            )

            # Store results if GCP services available
            if self.gcp_services:
                if hasattr(self.gcp_services, 'store_compliance_monitoring'):
                    await self.gcp_services.store_compliance_monitoring(compliance_result)
                if hasattr(self.gcp_services, 'store_bigquery_compliance_data'):
                    await self.gcp_services.store_bigquery_compliance_data(compliance_result)

            # Add metadata
            compliance_result["monitoring_metadata"] = {
                "agent": self.name,
                "monitoring_time": datetime.now(timezone.utc).isoformat(),
                "quality_score": self._calculate_compliance_quality_score(compliance_result)
            }

            logger.info("Compliance monitoring completed successfully")
            return compliance_result

        except Exception as e:
            error_msg = f"Error in compliance monitoring: {str(e)}"
            logger.error(error_msg)
            return {
                "status": "error",
                "error": error_msg,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "agent": self.name
            }

    @staticmethod
    def _calculate_compliance_quality_score(compliance_result: dict[str, any]) -> float:
        """Calculate quality score for compliance monitoring"""
        if compliance_result.get('status') != 'success':
            return 0.0

        score = 0

        # Regulatory compliance
        regulatory = compliance_result.get('regulatory_compliance', {})
        if regulatory and not regulatory.get('error'):
            score += 25

        # Internal compliance
        internal = compliance_result.get('internal_compliance', {})
        if internal and not internal.get('error'):
            score += 25

        # Risk compliance
        risk = compliance_result.get('risk_compliance', {})
        if risk and not risk.get('error'):
            score += 20

        # Trading restrictions
        restrictions = compliance_result.get('trading_restrictions', {})
        if restrictions and not restrictions.get('error'):
            score += 15

        # Recommendations
        recommendations = compliance_result.get('compliance_recommendations', {})
        if recommendations and not recommendations.get('error'):
            score += 15

        return round(min(100.0, score), 2)
