import asyncio
import logging
import sys
import os
from datetime import datetime, timezone
from typing import Dict, Any, Optional

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import with proper error handling
try:
    from agents import Agent, Tool
except ImportError:
    print("Warning: Google ADK agents not available, using fallback classes")


    class Agent:
        def __init__(self, **kwargs):
            pass


    class Tool:
        def __init__(self, **kwargs):
            pass

try:
    from agents.orchestration.workflow_manager_agent import WorkflowManagerAgent
except ImportError:
    print("Warning: WorkflowManagerAgent not available")
    WorkflowManagerAgent = None

from config.agent_config import AgentConfig
from config.gcp_config import GCPConfig, validate_config
from services.gcp_services import GCPServices

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SmartTraderADK:
    """
    Main Smart Trader Application using Google Agent Development Kit
    """

    def __init__(self):
        self.agent_config = AgentConfig()
        self.gcp_config = GCPConfig()
        self.gcp_services = None
        self.workflow_manager = None
        self.agents = {}
        self.is_initialized = False

        logger.info("SmartTraderADK initialized")

    async def initialize(self):
        """Initialize all services and agents"""
        try:
            logger.info("Initializing Smart Trader ADK...")

            # Validate configuration
            if not validate_config():
                raise ValueError("Configuration validation failed")

            # Initialize GCP services
            self.gcp_services = GCPServices(self.gcp_config)

            # Initialize workflow manager if available
            if WorkflowManagerAgent:
                self.workflow_manager = WorkflowManagerAgent(self.gcp_services, self.agent_config)

            # Initialize agents
            await self._initialize_agents()

            self.is_initialized = True
            logger.info("Smart Trader ADK initialization completed successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Smart Trader ADK: {str(e)}")
            raise

    async def _initialize_agents(self):
        """Initialize all trading agents"""
        try:
            # Data collection agents
            await self._init_data_collection_agents()

            # Analysis agents
            await self._init_analysis_agents()

            # Decision making agents
            await self._init_decision_agents()

            # Orchestration agents
            await self._init_orchestration_agents()

            logger.info(f"Initialized {len(self.agents)} agents")

        except Exception as e:
            logger.error(f"Failed to initialize agents: {str(e)}")
            raise

    async def _init_data_collection_agents(self):
        """Initialize data collection agents"""
        try:
            from agents.data_collectors.market_data_agent import MarketDataAgent
            self.agents['market_data'] = MarketDataAgent(self.gcp_services, self.agent_config)
        except ImportError:
            logger.warning("MarketDataAgent not available")

        try:
            from agents.data_collectors.news_sentiment_agent import NewsSentimentAgent
            self.agents['news_sentiment'] = NewsSentimentAgent(self.gcp_services, self.agent_config)
        except ImportError:
            logger.warning("NewsSentimentAgent not available")

        try:
            from agents.data_collectors.social_media_agent import SocialMediaAgent
            self.agents['social_media'] = SocialMediaAgent(self.gcp_services, self.agent_config)
        except ImportError:
            logger.warning("SocialMediaAgent not available")

        try:
            from agents.data_collectors.economic_indicators_agent import EconomicIndicatorsAgent
            self.agents['economic_indicators'] = EconomicIndicatorsAgent(self.gcp_services, self.agent_config)
        except ImportError:
            logger.warning("EconomicIndicatorsAgent not available")

    async def _init_analysis_agents(self):
        """Initialize analysis agents"""
        try:
            from agents.analyzers.technical_analyst_agent import TechnicalAnalystAgent
            self.agents['technical_analyst'] = TechnicalAnalystAgent(self.gcp_services, self.agent_config)
        except ImportError:
            logger.warning("TechnicalAnalystAgent not available")

        try:
            from agents.analyzers.fundamental_analyst_agent import FundamentalAnalystAgent
            self.agents['fundamental_analyst'] = FundamentalAnalystAgent(self.gcp_services, self.agent_config)
        except ImportError:
            logger.warning("FundamentalAnalystAgent not available")

        try:
            from agents.analyzers.risk_analyst_agent import RiskAnalystAgent
            self.agents['risk_analyst'] = RiskAnalystAgent(self.gcp_services, self.agent_config)
        except ImportError:
            logger.warning("RiskAnalystAgent not available")

        try:
            from agents.analyzers.market_regime_agent import MarketRegimeAgent
            self.agents['market_regime'] = MarketRegimeAgent(self.gcp_services, self.agent_config)
        except ImportError:
            logger.warning("MarketRegimeAgent not available")

    async def _init_decision_agents(self):
        """Initialize decision making agents"""
        try:
            from agents.decision_makers.strategy_coordinator_agent import StrategyCoordinatorAgent
            self.agents['strategy_coordinator'] = StrategyCoordinatorAgent(self.gcp_services, self.agent_config)
        except ImportError:
            logger.warning("StrategyCoordinatorAgent not available")

        try:
            from agents.decision_makers.portfolio_optimizer_agent import PortfolioOptimizerAgent
            self.agents['portfolio_optimizer'] = PortfolioOptimizerAgent(self.gcp_services, self.agent_config)
        except ImportError:
            logger.warning("PortfolioOptimizerAgent not available")

        try:
            from agents.decision_makers.execution_planner_agent import ExecutionPlannerAgent
            self.agents['execution_planner'] = ExecutionPlannerAgent(self.gcp_services, self.agent_config)
        except ImportError:
            logger.warning("ExecutionPlannerAgent not available")

    async def _init_orchestration_agents(self):
        """Initialize orchestration agents"""
        try:
            from agents.orchestration.compliance_monitor_agent import ComplianceMonitorAgent
            self.agents['compliance_monitor'] = ComplianceMonitorAgent(self.gcp_services, self.agent_config)
        except ImportError:
            logger.warning("ComplianceMonitorAgent not available")

        try:
            from agents.orchestration.performance_evaluator_agent import PerformanceEvaluatorAgent
            self.agents['performance_evaluator'] = PerformanceEvaluatorAgent(self.gcp_services, self.agent_config)
        except ImportError:
            logger.warning("PerformanceEvaluatorAgent not available")

    async def run_trading_workflow(self, symbols: list, strategy: str = "comprehensive") -> Dict[str, Any]:
        """Run the complete trading workflow"""
        if not self.is_initialized:
            await self.initialize()

        try:
            logger.info(f"Starting trading workflow for symbols: {symbols}")

            if self.workflow_manager:
                result = await self.workflow_manager.execute_workflow(symbols, strategy)
            else:
                # Fallback to manual workflow execution
                result = await self._execute_manual_workflow(symbols, strategy)

            logger.info("Trading workflow completed successfully")
            return result

        except Exception as e:
            logger.error(f"Trading workflow failed: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

    async def _execute_manual_workflow(self, symbols: list, strategy: str) -> Dict[str, Any]:
        """Manual workflow execution when WorkflowManager is not available"""
        results = {
            "status": "success",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbols": symbols,
            "strategy": strategy,
            "results": {}
        }

        # Step 1: Collect market data
        if 'market_data' in self.agents:
            try:
                market_data = await self.agents['market_data'].collect_market_data(symbols)
                results['results']['market_data'] = market_data
            except Exception as e:
                logger.error(f"Market data collection failed: {str(e)}")
                results['results']['market_data'] = {"error": str(e)}

        # Step 2: Technical analysis
        if 'technical_analyst' in self.agents and 'market_data' in results['results']:
            try:
                technical_analysis = await self.agents['technical_analyst'].analyze_securities_technical(
                    results['results']['market_data']
                )
                results['results']['technical_analysis'] = technical_analysis
            except Exception as e:
                logger.error(f"Technical analysis failed: {str(e)}")
                results['results']['technical_analysis'] = {"error": str(e)}

        # Step 3: Risk analysis
        if 'risk_analyst' in self.agents and 'market_data' in results['results']:
            try:
                risk_analysis = await self.agents['risk_analyst'].analyze_portfolio_risk(
                    results['results']['market_data']
                )
                results['results']['risk_analysis'] = risk_analysis
            except Exception as e:
                logger.error(f"Risk analysis failed: {str(e)}")
                results['results']['risk_analysis'] = {"error": str(e)}

        return results

    async def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all agents"""
        status = {
            "total_agents": len(self.agents),
            "agents": {},
            "gcp_services_status": "unknown",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        for name, agent in self.agents.items():
            status["agents"][name] = {
                "name": agent.name if hasattr(agent, 'name') else name,
                "status": "active"
            }

        if self.gcp_services:
            try:
                health_check = await self.gcp_services.health_check()
                status["gcp_services_status"] = "healthy" if health_check.get("overall") else "unhealthy"
            except Exception as e:
                status["gcp_services_status"] = f"error: {str(e)}"

        return status

    async def shutdown(self):
        """Shutdown all services and agents"""
        try:
            logger.info("Shutting down Smart Trader ADK...")

            if self.gcp_services:
                await self.gcp_services.close()

            self.agents.clear()
            self.is_initialized = False

            logger.info("Smart Trader ADK shutdown completed")

        except Exception as e:
            logger.error(f"Error during shutdown: {str(e)}")


# Main execution
async def main():
    """Main function for running the Smart Trader ADK"""
    trader = SmartTraderADK()

    try:
        await trader.initialize()

        # Example workflow execution
        symbols = ["AAPL", "GOOGL", "MSFT", "TSLA"]
        result = await trader.run_trading_workflow(symbols)

        print(f"Workflow result: {result}")

        # Get agent status
        status = await trader.get_agent_status()
        print(f"Agent status: {status}")

    except Exception as e:
        logger.error(f"Main execution failed: {str(e)}")
    finally:
        await trader.shutdown()


if __name__ == "__main__":
    asyncio.run(main())

# Export for streamlit_app.py
__all__ = ['SmartTraderADK']
