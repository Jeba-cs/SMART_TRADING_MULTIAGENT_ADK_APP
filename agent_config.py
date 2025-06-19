import os
from datetime import datetime, timezone

# Agent configuration parameters for ADK Smart Trader

AGENT_CONFIG = {
    "market_data_agent": {
        "fetch_interval_seconds": 120,
        "symbols_batch_size": 10,
        "max_retries": 3,
        "timeout_seconds": 30,
        "data_sources": ["alpaca", "yfinance", "finnhub"],
        "fallback_enabled": True,
        "cache_duration": 300
    },
    "news_sentiment_agent": {
        "sources": ["newsapi", "google_news", "bing_news"],
        "max_articles_per_source": 100,
        "sentiment_model": "gemini-2.0-flash-exp",
        "language_filter": "en",
        "relevance_threshold": 0.7,
        "update_interval_minutes": 30,
        "historical_days": 7
    },
    "social_media_agent": {
        "platforms": ["reddit"],
        "max_posts_per_symbol": 200,
        "sentiment_weighting": True,
        "engagement_threshold": 10,
        "subreddits": ["stocks", "investing", "SecurityAnalysis", "StockMarket"],
        "time_window_hours": 24
    },
    "economic_indicators_agent": {
        "indicators": ["GDP", "Inflation", "Unemployment", "Interest_Rates", "Consumer_Confidence"],
        "update_frequency_hours": 24,
        "data_sources": ["alpha_vantage", "fred"],
        "cache_duration_hours": 12,
        "importance_weights": {
            "GDP": 0.25,
            "Inflation": 0.25,
            "Unemployment": 0.20,
            "Interest_Rates": 0.20,
            "Consumer_Confidence": 0.10
        }
    },
    "technical_analyst_agent": {
        "indicators": ["SMA", "EMA", "RSI", "MACD", "Bollinger_Bands", "Stochastic", "Williams_R"],
        "pattern_recognition_enabled": True,
        "timeframes": ["1d", "1h", "15m"],
        "lookback_periods": {
            "short": 14,
            "medium": 50,
            "long": 200
        },
        "signal_confidence_threshold": 0.6,
        "pattern_types": ["head_shoulders", "double_top", "triangle", "flag"]
    },
    "fundamental_analyst_agent": {
        "valuation_models": ["DCF", "Comparable", "Asset_Based"],
        "financial_ratios": ["PE", "PEG", "ROE", "Debt_Equity", "Current_Ratio", "Quick_Ratio"],
        "data_sources": ["yfinance", "alpha_vantage"],
        "analysis_depth": "comprehensive",
        "sector_comparison_enabled": True,
        "growth_analysis_years": 5,
        "quality_filters": {
            "min_market_cap": 1000000000,
            "min_daily_volume": 100000
        }
    },
    "risk_analyst_agent": {
        "var_confidence_level": 0.95,
        "stress_test_scenarios": ["2008_Crisis", "COVID_Crash", "Dot_Com_Bubble"],
        "max_portfolio_var": 0.02,
        "risk_metrics": ["VaR", "CVaR", "Max_Drawdown", "Beta", "Volatility"],
        "correlation_threshold": 0.7,
        "rebalance_trigger": 0.10,
        "risk_limits": {
            "single_position": 0.10,
            "sector_concentration": 0.25,
            "leverage": 1.0
        }
    },
    "market_regime_agent": {
        "regime_types": ["bull", "bear", "transition", "crisis"],
        "volatility_thresholds": [15, 25, 40],
        "regime_indicators": ["VIX", "Market_Breadth", "Sector_Rotation"],
        "detection_algorithms": ["HMM", "Threshold", "Machine_Learning"],
        "confidence_threshold": 0.75,
        "regime_persistence_days": 5
    },
    "strategy_coordinator_agent": {
        "max_positions": 30,
        "max_position_size": 0.08,
        "min_position_size": 0.01,
        "strategy_types": ["momentum", "mean_reversion", "trend_following", "contrarian"],
        "time_horizons": ["short", "medium", "long"],
        "conviction_levels": ["high", "medium", "low"],
        "diversification_requirements": {
            "min_sectors": 5,
            "max_sector_weight": 0.25,
            "geographic_diversification": True
        }
    },
    "portfolio_optimizer_agent": {
        "max_leverage": 1.0,
        "min_weight": 0.01,
        "max_weight": 0.10,
        "optimization_methods": ["mean_variance", "risk_parity", "black_litterman", "strategic"],
        "rebalancing_frequency": "monthly",
        "transaction_cost_model": True,
        "constraints": {
            "turnover_limit": 0.30,
            "sector_limits": 0.25,
            "cash_buffer": 0.05
        },
        "risk_budgeting": {
            "equity_risk_budget": 0.80,
            "sector_risk_budget": 0.15,
            "idiosyncratic_risk_budget": 0.05
        }
    },
    "execution_planner_agent": {
        "max_participation_rate": 0.15,
        "max_order_size_percent": 0.05,
        "execution_timeframe_minutes": 60,
        "order_types": ["market", "limit", "vwap", "twap"],
        "slippage_model": "linear",
        "market_impact_model": "square_root",
        "execution_strategies": {
            "aggressive": {"timeframe": 30, "participation": 0.20},
            "moderate": {"timeframe": 60, "participation": 0.15},
            "passive": {"timeframe": 120, "participation": 0.10}
        }
    },
    "compliance_monitor_agent": {
        "max_single_position": 0.10,
        "max_sector_exposure": 0.25,
        "max_leverage": 1.5,
        "monitoring_frequency_minutes": 15,
        "regulatory_frameworks": ["SEC", "FINRA"],
        "violation_actions": ["alert", "block", "force_liquidate"],
        "reporting_requirements": {
            "daily_reports": True,
            "weekly_summaries": True,
            "monthly_compliance_review": True
        }
    },
    "performance_evaluator_agent": {
        "evaluation_frequency_days": 7,
        "benchmarks": ["SPY", "QQQ", "IWM", "AGG"],
        "metrics": ["Total_Return", "Sharpe_Ratio", "Max_Drawdown", "Alpha", "Beta", "Information_Ratio"],
        "attribution_analysis": ["security_selection", "asset_allocation", "timing"],
        "performance_periods": ["1d", "1w", "1m", "3m", "6m", "1y", "ytd", "inception"],
        "risk_adjusted_metrics": True,
        "peer_comparison": False
    },
    "workflow_manager_agent": {
        "max_concurrent_workflows": 5,
        "workflow_timeout_minutes": 30,
        "retry_policy": {
            "max_retries": 3,
            "backoff_factor": 2,
            "initial_delay_seconds": 1
        },
        "health_check_interval_minutes": 5,
        "logging_level": "INFO",
        "performance_monitoring": True
    }
}

# Environment-based overrides
if os.getenv("DEBUG", "false").lower() == "true":
    AGENT_CONFIG["market_data_agent"]["fetch_interval_seconds"] = 30
    AGENT_CONFIG["news_sentiment_agent"]["update_interval_minutes"] = 15
    AGENT_CONFIG["compliance_monitor_agent"]["monitoring_frequency_minutes"] = 5

if os.getenv("ENVIRONMENT", "development") == "production":
    AGENT_CONFIG["market_data_agent"]["symbols_batch_size"] = 100
    AGENT_CONFIG["news_sentiment_agent"]["max_articles_per_source"] = 200
    AGENT_CONFIG["workflow_manager_agent"]["max_concurrent_workflows"] = 10


class AgentConfig:
    """Agent Configuration Class"""

    def __init__(self):
        self.config = AGENT_CONFIG
        self.last_updated = datetime.now(timezone.utc)

    def get_config(self, agent_name: str = None) -> dict:
        """Get configuration for specific agent or all agents"""
        if agent_name:
            return self.config.get(agent_name, {})
        return self.config

    def update_config(self, agent_name: str, updates: dict):
        """Update configuration for specific agent"""
        if agent_name in self.config:
            self.config[agent_name].update(updates)
            self.last_updated = datetime.now(timezone.utc)

    def validate_config(self) -> bool:
        """Validate configuration parameters"""
        required_agents = [
            "market_data_agent", "technical_analyst_agent",
            "fundamental_analyst_agent", "risk_analyst_agent"
        ]

        for agent in required_agents:
            if agent not in self.config:
                return False

        return True


__all__ = ['AGENT_CONFIG', 'AgentConfig']
