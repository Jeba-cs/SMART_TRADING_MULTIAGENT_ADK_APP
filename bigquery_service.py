import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
import json
from google.cloud import bigquery
from google.cloud.exceptions import NotFound

logger = logging.getLogger(__name__)


class BigQueryService:
    """
    BigQuery service for analytics and data warehousing
    """

    def __init__(self, config):
        self.config = config
        self.project_id = config.GOOGLE_CLOUD_PROJECT
        self.dataset_id = getattr(config, 'BIGQUERY_DATASET_ID', 'smart_trader_analytics')

        try:
            self.client = bigquery.Client(project=self.project_id)
            self.dataset_ref = self.client.dataset(self.dataset_id)

            # Create dataset if it doesn't exist
            self._ensure_dataset_exists()

            # Create tables if they don't exist
            self._ensure_tables_exist()

            logger.info(f"BigQuery client initialized for project: {self.project_id}")

        except Exception as e:
            logger.error(f"Failed to initialize BigQuery: {str(e)}")
            self.client = None

    def _ensure_dataset_exists(self):
        """Ensure the dataset exists"""
        try:
            self.client.get_dataset(self.dataset_ref)
            logger.info(f"Dataset {self.dataset_id} already exists")
        except NotFound:
            dataset = bigquery.Dataset(self.dataset_ref)
            dataset.location = self.config.GOOGLE_CLOUD_LOCATION
            dataset.description = "Smart Trader Analytics Dataset"

            dataset = self.client.create_dataset(dataset, timeout=30)
            logger.info(f"Created dataset {self.dataset_id}")

    def _ensure_tables_exist(self):
        """Ensure all required tables exist"""
        tables_schema = {
            'market_data': [
                bigquery.SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED"),
                bigquery.SchemaField("symbol", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("price", "FLOAT", mode="NULLABLE"),
                bigquery.SchemaField("volume", "INTEGER", mode="NULLABLE"),
                bigquery.SchemaField("change_percent", "FLOAT", mode="NULLABLE"),
                bigquery.SchemaField("data", "JSON", mode="NULLABLE"),
            ],
            'analysis_results': [
                bigquery.SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED"),
                bigquery.SchemaField("agent_name", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("symbol", "STRING", mode="NULLABLE"),
                bigquery.SchemaField("analysis_type", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("result", "JSON", mode="NULLABLE"),
                bigquery.SchemaField("quality_score", "FLOAT", mode="NULLABLE"),
            ],
            'strategy_decisions': [
                bigquery.SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED"),
                bigquery.SchemaField("symbol", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("strategy", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("confidence", "FLOAT", mode="NULLABLE"),
                bigquery.SchemaField("position_size", "FLOAT", mode="NULLABLE"),
                bigquery.SchemaField("decision_data", "JSON", mode="NULLABLE"),
            ],
            'portfolio_optimization': [
                bigquery.SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED"),
                bigquery.SchemaField("optimization_method", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("expected_return", "FLOAT", mode="NULLABLE"),
                bigquery.SchemaField("expected_volatility", "FLOAT", mode="NULLABLE"),
                bigquery.SchemaField("sharpe_ratio", "FLOAT", mode="NULLABLE"),
                bigquery.SchemaField("allocation", "JSON", mode="NULLABLE"),
            ],
            'execution_data': [
                bigquery.SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED"),
                bigquery.SchemaField("symbol", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("side", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("quantity", "INTEGER", mode="REQUIRED"),
                bigquery.SchemaField("execution_method", "STRING", mode="NULLABLE"),
                bigquery.SchemaField("execution_data", "JSON", mode="NULLABLE"),
            ],
            'compliance_monitoring': [
                bigquery.SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED"),
                bigquery.SchemaField("compliance_type", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("status", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("violations", "JSON", mode="NULLABLE"),
                bigquery.SchemaField("recommendations", "JSON", mode="NULLABLE"),
            ],
            'performance_metrics': [
                bigquery.SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED"),
                bigquery.SchemaField("metric_type", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("timeframe", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("value", "FLOAT", mode="NULLABLE"),
                bigquery.SchemaField("benchmark_value", "FLOAT", mode="NULLABLE"),
                bigquery.SchemaField("performance_data", "JSON", mode="NULLABLE"),
            ],
            'risk_metrics': [
                bigquery.SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED"),
                bigquery.SchemaField("symbol", "STRING", mode="NULLABLE"),
                bigquery.SchemaField("risk_type", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("risk_score", "FLOAT", mode="NULLABLE"),
                bigquery.SchemaField("var_95", "FLOAT", mode="NULLABLE"),
                bigquery.SchemaField("risk_data", "JSON", mode="NULLABLE"),
            ],
            'sentiment_data': [
                bigquery.SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED"),
                bigquery.SchemaField("symbol", "STRING", mode="NULLABLE"),
                bigquery.SchemaField("sentiment_source", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("sentiment_score", "FLOAT", mode="NULLABLE"),
                bigquery.SchemaField("confidence", "FLOAT", mode="NULLABLE"),
                bigquery.SchemaField("sentiment_details", "JSON", mode="NULLABLE"),
            ],
            'economic_indicators': [
                bigquery.SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED"),
                bigquery.SchemaField("indicator_name", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("value", "FLOAT", mode="NULLABLE"),
                bigquery.SchemaField("previous_value", "FLOAT", mode="NULLABLE"),
                bigquery.SchemaField("change_percent", "FLOAT", mode="NULLABLE"),
                bigquery.SchemaField("indicator_data", "JSON", mode="NULLABLE"),
            ]
        }

        for table_name, schema in tables_schema.items():
            self._create_table_if_not_exists(table_name, schema)

    def _create_table_if_not_exists(self, table_name: str, schema: List[bigquery.SchemaField]):
        """Create table if it doesn't exist"""
        try:
            table_ref = self.dataset_ref.table(table_name)

            try:
                self.client.get_table(table_ref)
                logger.debug(f"Table {table_name} already exists")
            except NotFound:
                table = bigquery.Table(table_ref, schema=schema)
                table = self.client.create_table(table)
                logger.info(f"Created table {table_name}")

        except Exception as e:
            logger.error(f"Failed to create table {table_name}: {str(e)}")

    async def insert_market_data(self, data: Dict[str, Any]) -> bool:
        """Insert market data into BigQuery"""
        try:
            if not self.client:
                return False

            table_ref = self.dataset_ref.table('market_data')
            table = self.client.get_table(table_ref)

            rows_to_insert = []

            # Process market data
            timestamp = data.get('timestamp', datetime.utcnow().isoformat())
            symbols_data = data.get('data', {})

            for symbol, symbol_data in symbols_data.items():
                if isinstance(symbol_data, dict) and not symbol_data.get('error'):
                    row = {
                        'timestamp': timestamp,
                        'symbol': symbol,
                        'price': symbol_data.get('current_price', 0),
                        'volume': symbol_data.get('trading_metrics', {}).get('volume', 0),
                        'change_percent': symbol_data.get('change_percent', 0),
                        'data': json.dumps(symbol_data)
                    }
                    rows_to_insert.append(row)

            if rows_to_insert:
                errors = self.client.insert_rows_json(table, rows_to_insert)
                if errors:
                    logger.error(f"Failed to insert market data: {errors}")
                    return False

                logger.debug(f"Inserted {len(rows_to_insert)} market data rows")
                return True

            return True

        except Exception as e:
            logger.error(f"Failed to insert market data: {str(e)}")
            return False

    async def insert_analysis_data(self, data: Dict[str, Any]) -> bool:
        """Insert analysis data into BigQuery"""
        try:
            if not self.client:
                return False

            table_ref = self.dataset_ref.table('analysis_results')
            table = self.client.get_table(table_ref)

            timestamp = data.get('timestamp', datetime.utcnow().isoformat())
            agent_name = data.get('analysis_metadata', {}).get('agent', 'unknown')
            analysis_type = agent_name.replace('_agent', '').replace('_', ' ')

            # Process symbol analysis if available
            rows_to_insert = []
            symbol_analysis = data.get('symbol_analysis', {})

            if symbol_analysis:
                for symbol, analysis in symbol_analysis.items():
                    if not analysis.get('error'):
                        row = {
                            'timestamp': timestamp,
                            'agent_name': agent_name,
                            'symbol': symbol,
                            'analysis_type': analysis_type,
                            'result': json.dumps(analysis),
                            'quality_score': data.get('analysis_metadata', {}).get('quality_score', 0)
                        }
                        rows_to_insert.append(row)
            else:
                # General analysis without symbol breakdown
                row = {
                    'timestamp': timestamp,
                    'agent_name': agent_name,
                    'symbol': None,
                    'analysis_type': analysis_type,
                    'result': json.dumps(data),
                    'quality_score': data.get('analysis_metadata', {}).get('quality_score', 0)
                }
                rows_to_insert.append(row)

            if rows_to_insert:
                errors = self.client.insert_rows_json(table, rows_to_insert)
                if errors:
                    logger.error(f"Failed to insert analysis data: {errors}")
                    return False

                logger.debug(f"Inserted {len(rows_to_insert)} analysis data rows")

            return True

        except Exception as e:
            logger.error(f"Failed to insert analysis data: {str(e)}")
            return False

    async def insert_strategy_data(self, data: Dict[str, Any]) -> bool:
        """Insert strategy data into BigQuery"""
        try:
            if not self.client:
                return False

            table_ref = self.dataset_ref.table('strategy_decisions')
            table = self.client.get_table(table_ref)

            timestamp = data.get('timestamp', datetime.utcnow().isoformat())
            symbol_strategies = data.get('symbol_strategies', {})

            rows_to_insert = []

            for symbol, strategy in symbol_strategies.items():
                if not strategy.get('error'):
                    row = {
                        'timestamp': timestamp,
                        'symbol': symbol,
                        'strategy': strategy.get('primary_strategy', 'hold'),
                        'confidence': strategy.get('confidence_level', {}).get('overall_confidence', 0),
                        'position_size': strategy.get('position_sizing', {}).get('recommended_weight', 0),
                        'decision_data': json.dumps(strategy)
                    }
                    rows_to_insert.append(row)

            if rows_to_insert:
                errors = self.client.insert_rows_json(table, rows_to_insert)
                if errors:
                    logger.error(f"Failed to insert strategy data: {errors}")
                    return False

                logger.debug(f"Inserted {len(rows_to_insert)} strategy data rows")

            return True

        except Exception as e:
            logger.error(f"Failed to insert strategy data: {str(e)}")
            return False

    async def insert_optimization_data(self, data: Dict[str, Any]) -> bool:
        """Insert optimization data into BigQuery"""
        try:
            if not self.client:
                return False

            table_ref = self.dataset_ref.table('portfolio_optimization')
            table = self.client.get_table(table_ref)

            timestamp = data.get('timestamp', datetime.utcnow().isoformat())
            optimizations = data.get('optimization_approaches', {})

            rows_to_insert = []

            for method, optimization in optimizations.items():
                if not optimization.get('error'):
                    row = {
                        'timestamp': timestamp,
                        'optimization_method': method,
                        'expected_return': optimization.get('expected_return', 0),
                        'expected_volatility': optimization.get('expected_volatility', 0),
                        'sharpe_ratio': optimization.get('sharpe_ratio', 0),
                        'allocation': json.dumps(optimization.get('allocation', {}))
                    }
                    rows_to_insert.append(row)

            if rows_to_insert:
                errors = self.client.insert_rows_json(table, rows_to_insert)
                if errors:
                    logger.error(f"Failed to insert optimization data: {errors}")
                    return False

                logger.debug(f"Inserted {len(rows_to_insert)} optimization data rows")

            return True

        except Exception as e:
            logger.error(f"Failed to insert optimization data: {str(e)}")
            return False

    async def insert_execution_data(self, data: Dict[str, Any]) -> bool:
        """Insert execution data into BigQuery"""
        try:
            if not self.client:
                return False

            table_ref = self.dataset_ref.table('execution_data')
            table = self.client.get_table(table_ref)

            timestamp = data.get('timestamp', datetime.utcnow().isoformat())
            trade_list = data.get('trade_list', {})

            rows_to_insert = []

            # Process buy trades
            buy_trades = trade_list.get('buy_trades', [])
            for trade in buy_trades:
                row = {
                    'timestamp': timestamp,
                    'symbol': trade.get('symbol', ''),
                    'side': 'buy',
                    'quantity': trade.get('shares', 0),
                    'execution_method': 'planned',
                    'execution_data': json.dumps(trade)
                }
                rows_to_insert.append(row)

            # Process sell trades
            sell_trades = trade_list.get('sell_trades', [])
            for trade in sell_trades:
                row = {
                    'timestamp': timestamp,
                    'symbol': trade.get('symbol', ''),
                    'side': 'sell',
                    'quantity': trade.get('shares', 0),
                    'execution_method': 'planned',
                    'execution_data': json.dumps(trade)
                }
                rows_to_insert.append(row)

            if rows_to_insert:
                errors = self.client.insert_rows_json(table, rows_to_insert)
                if errors:
                    logger.error(f"Failed to insert execution data: {errors}")
                    return False

                logger.debug(f"Inserted {len(rows_to_insert)} execution data rows")

            return True

        except Exception as e:
            logger.error(f"Failed to insert execution data: {str(e)}")
            return False

    async def insert_compliance_data(self, data: Dict[str, Any]) -> bool:
        """Insert compliance data into BigQuery"""
        try:
            if not self.client:
                return False

            table_ref = self.dataset_ref.table('compliance_monitoring')
            table = self.client.get_table(table_ref)

            timestamp = data.get('timestamp', datetime.utcnow().isoformat())
            compliance_summary = data.get('compliance_summary', {})

            row = {
                'timestamp': timestamp,
                'compliance_type': 'comprehensive',
                'status': compliance_summary.get('compliance_status', 'unknown'),
                'violations': json.dumps(data.get('compliance_issues', [])),
                'recommendations': json.dumps(data.get('compliance_recommendations', {}))
            }

            errors = self.client.insert_rows_json(table, [row])
            if errors:
                logger.error(f"Failed to insert compliance data: {errors}")
                return False

            logger.debug("Inserted compliance data row")
            return True

        except Exception as e:
            logger.error(f"Failed to insert compliance data: {str(e)}")
            return False

    async def insert_performance_data(self, data: Dict[str, Any]) -> bool:
        """Insert performance data into BigQuery"""
        try:
            if not self.client:
                return False

            table_ref = self.dataset_ref.table('performance_metrics')
            table = self.client.get_table(table_ref)

            timestamp = data.get('timestamp', datetime.utcnow().isoformat())
            timeframe = data.get('timeframe', 'daily')

            rows_to_insert = []

            # Performance metrics
            performance_metrics = data.get('performance_metrics', {})
            for metric, value in performance_metrics.items():
                if isinstance(value, (int, float)):
                    row = {
                        'timestamp': timestamp,
                        'metric_type': metric,
                        'timeframe': timeframe,
                        'value': float(value),
                        'benchmark_value': None,
                        'performance_data': json.dumps(performance_metrics)
                    }
                    rows_to_insert.append(row)

            # Risk-adjusted metrics
            risk_metrics = data.get('risk_adjusted_metrics', {})
            for metric, value in risk_metrics.items():
                if isinstance(value, (int, float)):
                    row = {
                        'timestamp': timestamp,
                        'metric_type': f"risk_adjusted_{metric}",
                        'timeframe': timeframe,
                        'value': float(value),
                        'benchmark_value': None,
                        'performance_data': json.dumps(risk_metrics)
                    }
                    rows_to_insert.append(row)

            if rows_to_insert:
                errors = self.client.insert_rows_json(table, rows_to_insert)
                if errors:
                    logger.error(f"Failed to insert performance data: {errors}")
                    return False

                logger.debug(f"Inserted {len(rows_to_insert)} performance data rows")

            return True

        except Exception as e:
            logger.error(f"Failed to insert performance data: {str(e)}")
            return False

    async def insert_technical_data(self, data: Dict[str, Any]) -> bool:
        """Insert technical analysis data into BigQuery"""
        return await self.insert_analysis_data(data)

    async def insert_fundamental_data(self, data: Dict[str, Any]) -> bool:
        """Insert fundamental analysis data into BigQuery"""
        return await self.insert_analysis_data(data)

    async def insert_risk_data(self, data: Dict[str, Any]) -> bool:
        """Insert risk analysis data into BigQuery"""
        try:
            if not self.client:
                return False

            table_ref = self.dataset_ref.table('risk_metrics')
            table = self.client.get_table(table_ref)

            timestamp = data.get('timestamp', datetime.utcnow().isoformat())
            symbol_risks = data.get('symbol_risk_analysis', {})

            rows_to_insert = []

            for symbol, risk_data in symbol_risks.items():
                if not risk_data.get('error'):
                    overall_risk = risk_data.get('overall_risk_score', {})
                    var_data = risk_data.get('value_at_risk', {})

                    row = {
                        'timestamp': timestamp,
                        'symbol': symbol,
                        'risk_type': 'comprehensive',
                        'risk_score': overall_risk.get('overall_risk_score', 0),
                        'var_95': var_data.get('historical_var_1d', 0),
                        'risk_data': json.dumps(risk_data)
                    }
                    rows_to_insert.append(row)

            if rows_to_insert:
                errors = self.client.insert_rows_json(table, rows_to_insert)
                if errors:
                    logger.error(f"Failed to insert risk data: {errors}")
                    return False

                logger.debug(f"Inserted {len(rows_to_insert)} risk data rows")

            return True

        except Exception as e:
            logger.error(f"Failed to insert risk data: {str(e)}")
            return False

    async def insert_regime_data(self, data: Dict[str, Any]) -> bool:
        """Insert regime analysis data into BigQuery"""
        return await self.insert_analysis_data(data)

    async def insert_sentiment_data(self, data: Dict[str, Any]) -> bool:
        """Insert sentiment analysis data into BigQuery"""
        try:
            if not self.client:
                return False

            table_ref = self.dataset_ref.table('sentiment_data')
            table = self.client.get_table(table_ref)

            timestamp = data.get('timestamp', datetime.utcnow().isoformat())

            rows_to_insert = []

            # News sentiment
            if 'sentiment_by_symbol' in data:
                for symbol, sentiment in data['sentiment_by_symbol'].items():
                    row = {
                        'timestamp': timestamp,
                        'symbol': symbol,
                        'sentiment_source': 'news',
                        'sentiment_score': sentiment.get('weighted_sentiment', 0),
                        'confidence': sentiment.get('confidence', 0),
                        'sentiment_details': json.dumps(sentiment)
                    }
                    rows_to_insert.append(row)

            # Social sentiment
            if 'symbol_sentiment' in data:
                for symbol, sentiment in data['symbol_sentiment'].items():
                    row = {
                        'timestamp': timestamp,
                        'symbol': symbol,
                        'sentiment_source': 'social',
                        'sentiment_score': sentiment.get('sentiment_score', 0),
                        'confidence': sentiment.get('confidence', 0),
                        'sentiment_details': json.dumps(sentiment)
                    }
                    rows_to_insert.append(row)

            if rows_to_insert:
                errors = self.client.insert_rows_json(table, rows_to_insert)
                if errors:
                    logger.error(f"Failed to insert sentiment data: {errors}")
                    return False

                logger.debug(f"Inserted {len(rows_to_insert)} sentiment data rows")

            return True

        except Exception as e:
            logger.error(f"Failed to insert sentiment data: {str(e)}")
            return False

    async def insert_social_data(self, data: Dict[str, Any]) -> bool:
        """Insert social media data into BigQuery"""
        return await self.insert_sentiment_data(data)

    async def insert_economic_data(self, data: Dict[str, Any]) -> bool:
        """Insert economic indicators data into BigQuery"""
        try:
            if not self.client:
                return False

            table_ref = self.dataset_ref.table('economic_indicators')
            table = self.client.get_table(table_ref)

            timestamp = data.get('timestamp', datetime.utcnow().isoformat())
            economic_data = data.get('economic_data', {})

            rows_to_insert = []

            for indicator, indicator_data in economic_data.items():
                if isinstance(indicator_data, dict) and not indicator_data.get('error'):
                    # Extract values based on indicator type
                    current_value = None
                    previous_value = None
                    change_percent = None

                    if indicator == 'gdp':
                        current_value = indicator_data.get('current_gdp')
                        previous_value = indicator_data.get('previous_gdp')
                        change_percent = indicator_data.get('growth_rate')
                    elif indicator == 'inflation':
                        current_value = indicator_data.get('inflation_rate')
                        change_percent = indicator_data.get('target_comparison')
                    elif indicator == 'unemployment':
                        current_value = indicator_data.get('current_rate')
                        previous_value = indicator_data.get('previous_rate')
                        change_percent = indicator_data.get('rate_change')
                    elif indicator == 'interest_rates':
                        current_value = indicator_data.get('federal_funds_rate')
                        previous_value = indicator_data.get('previous_rate')
                        change_percent = indicator_data.get('rate_change')

                    if current_value is not None:
                        row = {
                            'timestamp': timestamp,
                            'indicator_name': indicator,
                            'value': float(current_value),
                            'previous_value': float(previous_value) if previous_value is not None else None,
                            'change_percent': float(change_percent) if change_percent is not None else None,
                            'indicator_data': json.dumps(indicator_data)
                        }
                        rows_to_insert.append(row)

            if rows_to_insert:
                errors = self.client.insert_rows_json(table, rows_to_insert)
                if errors:
                    logger.error(f"Failed to insert economic data: {errors}")
                    return False

                logger.debug(f"Inserted {len(rows_to_insert)} economic data rows")

            return True

        except Exception as e:
            logger.error(f"Failed to insert economic data: {str(e)}")
            return False

    async def execute_query(self, query: str) -> List[Dict[str, Any]]:
        """Execute BigQuery SQL query"""
        try:
            if not self.client:
                return []

            query_job = self.client.query(query)
            results = query_job.result()

            rows = []
            for row in results:
                rows.append(dict(row))

            logger.debug(f"Query executed successfully, returned {len(rows)} rows")
            return rows

        except Exception as e:
            logger.error(f"Failed to execute query: {str(e)}")
            return []

    async def get_market_data_analytics(self, symbol: str = None, days: int = 30) -> Dict[str, Any]:
        """Get market data analytics"""
        try:
            base_query = f"""
            SELECT 
                symbol,
                AVG(price) as avg_price,
                STDDEV(price) as price_volatility,
                AVG(volume) as avg_volume,
                COUNT(*) as data_points,
                MIN(timestamp) as start_date,
                MAX(timestamp) as end_date
            FROM `{self.project_id}.{self.dataset_id}.market_data`
            WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {days} DAY)
            """

            if symbol:
                base_query += f" AND symbol = '{symbol}'"

            base_query += " GROUP BY symbol ORDER BY symbol"

            results = await self.execute_query(base_query)
            return {"analytics": results, "symbol": symbol, "days": days}

        except Exception as e:
            logger.error(f"Failed to get market data analytics: {str(e)}")
            return {}

    async def get_performance_analytics(self, timeframe: str = "daily", days: int = 30) -> Dict[str, Any]:
        """Get performance analytics"""
        try:
            query = f"""
            SELECT 
                metric_type,
                timeframe,
                AVG(value) as avg_value,
                STDDEV(value) as value_volatility,
                MIN(value) as min_value,
                MAX(value) as max_value,
                COUNT(*) as data_points
            FROM `{self.project_id}.{self.dataset_id}.performance_metrics`
            WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {days} DAY)
            AND timeframe = '{timeframe}'
            GROUP BY metric_type, timeframe
            ORDER BY metric_type
            """

            results = await self.execute_query(query)
            return {"analytics": results, "timeframe": timeframe, "days": days}

        except Exception as e:
            logger.error(f"Failed to get performance analytics: {str(e)}")
            return {}

    async def health_check(self) -> bool:
        """Perform health check on BigQuery service"""
        try:
            if not self.client:
                return False

            # Try a simple query
            query = f"SELECT 1 as health_check"
            results = await self.execute_query(query)

            return len(results) > 0 and results[0].get('health_check') == 1

        except Exception as e:
            logger.error(f"BigQuery health check failed: {str(e)}")
            return False

    async def close(self):
        """Close BigQuery client connections"""
        try:
            if self.client:
                # BigQuery client doesn't need explicit closing
                logger.info("BigQuery service closed")
        except Exception as e:
            logger.error(f"Error closing BigQuery service: {str(e)}")
