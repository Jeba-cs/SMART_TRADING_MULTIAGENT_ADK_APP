import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
import json
import os
from google.cloud import firestore
from google.cloud import bigquery
from google.cloud import storage
from google.auth import default
import vertexai
from vertexai.generative_models import GenerativeModel

from .firestore_service import FirestoreService
from .bigquery_service import BigQueryService

logger = logging.getLogger(__name__)


class GCPServices:
    """
    Comprehensive Google Cloud Platform services integration for ADK Smart Trader
    """

    def __init__(self, config):
        self.config = config
        self.project_id = config.GOOGLE_CLOUD_PROJECT
        self.location = config.GOOGLE_CLOUD_LOCATION

        # Initialize services
        self.firestore_service = FirestoreService(config)
        self.bigquery_service = BigQueryService(config)

        # Initialize other GCP services
        self._initialize_storage()
        self._initialize_vertex_ai()

        # Session management
        self.session_id = None

        logger.info(f"GCP Services initialized for project: {self.project_id}")

    def _initialize_storage(self):
        """Initialize Google Cloud Storage"""
        try:
            self.storage_client = storage.Client(project=self.project_id)
            self.bucket_name = f"{self.project_id}-smart-trader-data"

            # Create bucket if it doesn't exist
            try:
                self.bucket = self.storage_client.bucket(self.bucket_name)
                if not self.bucket.exists():
                    self.bucket = self.storage_client.create_bucket(self.bucket_name, location=self.location)
                    logger.info(f"Created Storage bucket: {self.bucket_name}")
                else:
                    logger.info(f"Using existing Storage bucket: {self.bucket_name}")
            except Exception as e:
                logger.warning(f"Storage initialization warning: {str(e)}")
                self.bucket = None

        except Exception as e:
            logger.error(f"Failed to initialize Cloud Storage: {str(e)}")
            self.storage_client = None
            self.bucket = None

    def _initialize_vertex_ai(self):
        """Initialize Vertex AI"""
        try:
            vertexai.init(project=self.project_id, location=self.location)
            self.vertex_model = GenerativeModel("gemini-2.0-flash-exp")
            logger.info("Vertex AI initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Vertex AI: {str(e)}")
            self.vertex_model = None

    def get_session_id(self) -> str:
        """Get or create session ID"""
        if not self.session_id:
            self.session_id = f"session_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        return self.session_id

    # Firestore operations
    async def store_market_data(self, data: Dict[str, Any]) -> bool:
        """Store market data in Firestore"""
        return await self.firestore_service.store_market_data(data)

    async def store_analysis_results(self, agent_name: str, analysis_data: Dict[str, Any]) -> bool:
        """Store analysis results in Firestore"""
        return await self.firestore_service.store_analysis_results(agent_name, analysis_data)

    async def store_strategy_coordination(self, strategy_data: Dict[str, Any]) -> bool:
        """Store strategy coordination results"""
        return await self.firestore_service.store_document("strategy_coordination", strategy_data)

    async def store_portfolio_optimization(self, optimization_data: Dict[str, Any]) -> bool:
        """Store portfolio optimization results"""
        return await self.firestore_service.store_document("portfolio_optimization", optimization_data)

    async def store_execution_plan(self, execution_data: Dict[str, Any]) -> bool:
        """Store execution plan"""
        return await self.firestore_service.store_document("execution_plans", execution_data)

    async def store_compliance_monitoring(self, compliance_data: Dict[str, Any]) -> bool:
        """Store compliance monitoring results"""
        return await self.firestore_service.store_document("compliance_monitoring", compliance_data)

    async def store_performance_evaluation(self, performance_data: Dict[str, Any]) -> bool:
        """Store performance evaluation results"""
        return await self.firestore_service.store_document("performance_evaluation", performance_data)

    async def store_technical_analysis(self, technical_data: Dict[str, Any]) -> bool:
        """Store technical analysis results"""
        return await self.firestore_service.store_document("technical_analysis", technical_data)

    async def store_fundamental_analysis(self, fundamental_data: Dict[str, Any]) -> bool:
        """Store fundamental analysis results"""
        return await self.firestore_service.store_document("fundamental_analysis", fundamental_data)

    async def store_risk_analysis(self, risk_data: Dict[str, Any]) -> bool:
        """Store risk analysis results"""
        return await self.firestore_service.store_document("risk_analysis", risk_data)

    async def store_regime_analysis(self, regime_data: Dict[str, Any]) -> bool:
        """Store market regime analysis results"""
        return await self.firestore_service.store_document("regime_analysis", regime_data)

    async def store_news_sentiment(self, sentiment_data: Dict[str, Any]) -> bool:
        """Store news sentiment analysis results"""
        return await self.firestore_service.store_document("news_sentiment", sentiment_data)

    async def store_social_sentiment(self, sentiment_data: Dict[str, Any]) -> bool:
        """Store social media sentiment analysis results"""
        return await self.firestore_service.store_document("social_sentiment", sentiment_data)

    async def store_economic_data(self, economic_data: Dict[str, Any]) -> bool:
        """Store economic indicators data"""
        return await self.firestore_service.store_document("economic_data", economic_data)

    async def store_workflow_results(self, workflow_data: Dict[str, Any]) -> bool:
        """Store workflow results"""
        return await self.firestore_service.store_document("workflow_results", workflow_data)

    async def store_request_context(self, context_data: Dict[str, Any]) -> bool:
        """Store request context"""
        return await self.firestore_service.store_document("request_contexts", context_data)

    # BigQuery operations
    async def store_bigquery_market_data(self, data: Dict[str, Any]) -> bool:
        """Store market data in BigQuery for analytics"""
        return await self.bigquery_service.insert_market_data(data)

    async def store_bigquery_analysis_data(self, data: Dict[str, Any]) -> bool:
        """Store analysis data in BigQuery"""
        return await self.bigquery_service.insert_analysis_data(data)

    async def store_bigquery_strategy_data(self, data: Dict[str, Any]) -> bool:
        """Store strategy data in BigQuery"""
        return await self.bigquery_service.insert_strategy_data(data)

    async def store_bigquery_optimization_data(self, data: Dict[str, Any]) -> bool:
        """Store optimization data in BigQuery"""
        return await self.bigquery_service.insert_optimization_data(data)

    async def store_bigquery_execution_data(self, data: Dict[str, Any]) -> bool:
        """Store execution data in BigQuery"""
        return await self.bigquery_service.insert_execution_data(data)

    async def store_bigquery_compliance_data(self, data: Dict[str, Any]) -> bool:
        """Store compliance data in BigQuery"""
        return await self.bigquery_service.insert_compliance_data(data)

    async def store_bigquery_performance_data(self, data: Dict[str, Any]) -> bool:
        """Store performance data in BigQuery"""
        return await self.bigquery_service.insert_performance_data(data)

    async def store_bigquery_technical_data(self, data: Dict[str, Any]) -> bool:
        """Store technical analysis data in BigQuery"""
        return await self.bigquery_service.insert_technical_data(data)

    async def store_bigquery_fundamental_data(self, data: Dict[str, Any]) -> bool:
        """Store fundamental analysis data in BigQuery"""
        return await self.bigquery_service.insert_fundamental_data(data)

    async def store_bigquery_risk_data(self, data: Dict[str, Any]) -> bool:
        """Store risk analysis data in BigQuery"""
        return await self.bigquery_service.insert_risk_data(data)

    async def store_bigquery_regime_data(self, data: Dict[str, Any]) -> bool:
        """Store regime analysis data in BigQuery"""
        return await self.bigquery_service.insert_regime_data(data)

    async def store_bigquery_sentiment_data(self, data: Dict[str, Any]) -> bool:
        """Store sentiment analysis data in BigQuery"""
        return await self.bigquery_service.insert_sentiment_data(data)

    async def store_bigquery_social_data(self, data: Dict[str, Any]) -> bool:
        """Store social media data in BigQuery"""
        return await self.bigquery_service.insert_social_data(data)

    async def store_bigquery_economic_data(self, data: Dict[str, Any]) -> bool:
        """Store economic data in BigQuery"""
        return await self.bigquery_service.insert_economic_data(data)

    # Retrieval operations
    async def get_historical_data(self, collection: str, symbol: str = None,
                                  days: int = 30) -> List[Dict[str, Any]]:
        """Get historical data from Firestore"""
        return await self.firestore_service.get_historical_data(collection, symbol, days)

    async def get_latest_analysis(self, agent_name: str) -> Optional[Dict[str, Any]]:
        """Get latest analysis from specific agent"""
        return await self.firestore_service.get_latest_analysis(agent_name)

    async def query_bigquery_data(self, query: str) -> List[Dict[str, Any]]:
        """Execute BigQuery query and return results"""
        return await self.bigquery_service.execute_query(query)

    # Cloud Storage operations
    async def upload_file(self, file_path: str, blob_name: str) -> bool:
        """Upload file to Cloud Storage"""
        try:
            if not self.bucket:
                logger.error("Cloud Storage not initialized")
                return False

            blob = self.bucket.blob(blob_name)
            blob.upload_from_filename(file_path)
            logger.info(f"File {file_path} uploaded to {blob_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to upload file: {str(e)}")
            return False

    async def download_file(self, blob_name: str, file_path: str) -> bool:
        """Download file from Cloud Storage"""
        try:
            if not self.bucket:
                logger.error("Cloud Storage not initialized")
                return False

            blob = self.bucket.blob(blob_name)
            blob.download_to_filename(file_path)
            logger.info(f"File {blob_name} downloaded to {file_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to download file: {str(e)}")
            return False

    async def store_blob_data(self, data: bytes, blob_name: str) -> bool:
        """Store binary data in Cloud Storage"""
        try:
            if not self.bucket:
                logger.error("Cloud Storage not initialized")
                return False

            blob = self.bucket.blob(blob_name)
            blob.upload_from_string(data)
            logger.info(f"Data uploaded to {blob_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to store blob data: {str(e)}")
            return False

    # Error logging
    async def log_error(self, error_data: Dict[str, Any]) -> bool:
        """Log error to Firestore"""
        try:
            error_doc = {
                "error": error_data,
                "timestamp": datetime.utcnow(),
                "session_id": self.get_session_id()
            }
            return await self.firestore_service.store_document("errors", error_doc)
        except Exception as e:
            logger.error(f"Failed to log error: {str(e)}")
            return False

    # Vertex AI operations
    async def generate_content(self, prompt: str) -> Optional[str]:
        """Generate content using Vertex AI"""
        try:
            if not self.vertex_model:
                logger.error("Vertex AI not initialized")
                return None

            response = await self.vertex_model.generate_content_async(prompt)
            return response.text

        except Exception as e:
            logger.error(f"Failed to generate content: {str(e)}")
            return None

    # Health check
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all services"""
        health_status = {
            "firestore": await self.firestore_service.health_check(),
            "bigquery": await self.bigquery_service.health_check(),
            "storage": self.bucket is not None,
            "vertex_ai": self.vertex_model is not None,
            "timestamp": datetime.utcnow().isoformat()
        }

        health_status["overall"] = all([
            health_status["firestore"],
            health_status["bigquery"],
            health_status["storage"],
            health_status["vertex_ai"]
        ])

        return health_status

    async def close(self):
        """Close all service connections"""
        try:
            await self.firestore_service.close()
            await self.bigquery_service.close()
            logger.info("GCP Services closed successfully")
        except Exception as e:
            logger.error(f"Error closing GCP Services: {str(e)}")
