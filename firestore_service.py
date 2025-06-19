import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import json
from google.cloud import firestore
from google.cloud.firestore_v1.base_query import FieldFilter

logger = logging.getLogger(__name__)


class FirestoreService:
    """
    Firestore service for storing and retrieving trading data
    """

    def __init__(self, config):
        self.config = config
        self.project_id = config.GOOGLE_CLOUD_PROJECT

        try:
            self.db = firestore.Client(project=self.project_id)
            logger.info(f"Firestore client initialized for project: {self.project_id}")
        except Exception as e:
            logger.error(f"Failed to initialize Firestore: {str(e)}")
            self.db = None

    async def store_document(self, collection: str, data: Dict[str, Any],
                             document_id: str = None) -> bool:
        """Store document in Firestore collection"""
        try:
            if not self.db:
                logger.error("Firestore client not initialized")
                return False

            # Add timestamp if not present
            if 'timestamp' not in data:
                data['timestamp'] = datetime.utcnow()

            # Generate document ID if not provided
            if not document_id:
                document_id = f"{collection}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}"

            doc_ref = self.db.collection(collection).document(document_id)
            doc_ref.set(data)

            logger.debug(f"Document stored in {collection}/{document_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to store document in {collection}: {str(e)}")
            return False

    async def store_market_data(self, data: Dict[str, Any]) -> bool:
        """Store market data with symbol-based organization"""
        try:
            if not self.db:
                return False

            # Store main market data
            success = await self.store_document("market_data", data)

            # Store individual symbol data for easier querying
            symbols_data = data.get('data', {})
            for symbol, symbol_data in symbols_data.items():
                if isinstance(symbol_data, dict) and not symbol_data.get('error'):
                    symbol_doc = {
                        "symbol": symbol,
                        "data": symbol_data,
                        "timestamp": data.get('timestamp', datetime.utcnow()),
                        "collection_time": datetime.utcnow()
                    }

                    symbol_collection = f"market_data_by_symbol"
                    symbol_doc_id = f"{symbol}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

                    await self.store_document(symbol_collection, symbol_doc, symbol_doc_id)

            return success

        except Exception as e:
            logger.error(f"Failed to store market data: {str(e)}")
            return False

    async def store_analysis_results(self, agent_name: str, analysis_data: Dict[str, Any]) -> bool:
        """Store analysis results with agent-based organization"""
        try:
            if not self.db:
                return False

            # Store in general analysis collection
            analysis_doc = {
                "agent_name": agent_name,
                "analysis_data": analysis_data,
                "timestamp": analysis_data.get('timestamp', datetime.utcnow()),
                "storage_time": datetime.utcnow()
            }

            doc_id = f"{agent_name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            success = await self.store_document("analysis_results", analysis_doc, doc_id)

            # Store in agent-specific collection
            agent_collection = f"analysis_{agent_name.replace(' ', '_').lower()}"
            await self.store_document(agent_collection, analysis_data)

            return success

        except Exception as e:
            logger.error(f"Failed to store analysis results for {agent_name}: {str(e)}")
            return False

    async def get_document(self, collection: str, document_id: str) -> Optional[Dict[str, Any]]:
        """Get document from Firestore"""
        try:
            if not self.db:
                return None

            doc_ref = self.db.collection(collection).document(document_id)
            doc = doc_ref.get()

            if doc.exists:
                return doc.to_dict()
            else:
                logger.warning(f"Document {collection}/{document_id} not found")
                return None

        except Exception as e:
            logger.error(f"Failed to get document {collection}/{document_id}: {str(e)}")
            return None

    async def get_latest_document(self, collection: str, order_by: str = "timestamp") -> Optional[Dict[str, Any]]:
        """Get latest document from collection"""
        try:
            if not self.db:
                return None

            query = self.db.collection(collection).order_by(
                order_by, direction=firestore.Query.DESCENDING
            ).limit(1)

            docs = query.stream()
            for doc in docs:
                return doc.to_dict()

            return None

        except Exception as e:
            logger.error(f"Failed to get latest document from {collection}: {str(e)}")
            return None

    async def get_historical_data(self, collection: str, symbol: str = None,
                                  days: int = 30) -> List[Dict[str, Any]]:
        """Get historical data from collection"""
        try:
            if not self.db:
                return []

            # Calculate date range
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days)

            # Build query
            query = self.db.collection(collection)
            query = query.where(filter=FieldFilter("timestamp", ">=", start_date))
            query = query.where(filter=FieldFilter("timestamp", "<=", end_date))

            # Add symbol filter if provided
            if symbol:
                query = query.where(filter=FieldFilter("symbol", "==", symbol))

            # Order by timestamp
            query = query.order_by("timestamp", direction=firestore.Query.DESCENDING)

            # Execute query
            docs = query.stream()
            results = [doc.to_dict() for doc in docs]

            logger.info(f"Retrieved {len(results)} historical documents from {collection}")
            return results

        except Exception as e:
            logger.error(f"Failed to get historical data from {collection}: {str(e)}")
            return []

    async def get_latest_analysis(self, agent_name: str) -> Optional[Dict[str, Any]]:
        """Get latest analysis from specific agent"""
        try:
            if not self.db:
                return None

            # Try agent-specific collection first
            agent_collection = f"analysis_{agent_name.replace(' ', '_').lower()}"
            latest_doc = await self.get_latest_document(agent_collection)

            if latest_doc:
                return latest_doc

            # Fall back to general analysis collection
            query = self.db.collection("analysis_results")
            query = query.where(filter=FieldFilter("agent_name", "==", agent_name))
            query = query.order_by("timestamp", direction=firestore.Query.DESCENDING)
            query = query.limit(1)

            docs = query.stream()
            for doc in docs:
                doc_data = doc.to_dict()
                return doc_data.get('analysis_data', doc_data)

            return None

        except Exception as e:
            logger.error(f"Failed to get latest analysis for {agent_name}: {str(e)}")
            return None

    async def query_documents(self, collection: str, filters: List[Dict[str, Any]] = None,
                              order_by: str = None, limit: int = None) -> List[Dict[str, Any]]:
        """Query documents with filters"""
        try:
            if not self.db:
                return []

            query = self.db.collection(collection)

            # Apply filters
            if filters:
                for filter_config in filters:
                    field = filter_config.get('field')
                    operator = filter_config.get('operator', '==')
                    value = filter_config.get('value')

                    if field and value is not None:
                        query = query.where(filter=FieldFilter(field, operator, value))

            # Apply ordering
            if order_by:
                direction = firestore.Query.DESCENDING
                if order_by.startswith('+'):
                    direction = firestore.Query.ASCENDING
                    order_by = order_by[1:]
                elif order_by.startswith('-'):
                    direction = firestore.Query.DESCENDING
                    order_by = order_by[1:]

                query = query.order_by(order_by, direction=direction)

            # Apply limit
            if limit:
                query = query.limit(limit)

            # Execute query
            docs = query.stream()
            results = [doc.to_dict() for doc in docs]

            return results

        except Exception as e:
            logger.error(f"Failed to query documents from {collection}: {str(e)}")
            return []

    async def update_document(self, collection: str, document_id: str,
                              updates: Dict[str, Any]) -> bool:
        """Update document in Firestore"""
        try:
            if not self.db:
                return False

            doc_ref = self.db.collection(collection).document(document_id)

            # Add update timestamp
            updates['last_updated'] = datetime.utcnow()

            doc_ref.update(updates)
            logger.debug(f"Document {collection}/{document_id} updated")
            return True

        except Exception as e:
            logger.error(f"Failed to update document {collection}/{document_id}: {str(e)}")
            return False

    async def delete_document(self, collection: str, document_id: str) -> bool:
        """Delete document from Firestore"""
        try:
            if not self.db:
                return False

            doc_ref = self.db.collection(collection).document(document_id)
            doc_ref.delete()

            logger.debug(f"Document {collection}/{document_id} deleted")
            return True

        except Exception as e:
            logger.error(f"Failed to delete document {collection}/{document_id}: {str(e)}")
            return False

    async def batch_write(self, operations: List[Dict[str, Any]]) -> bool:
        """Perform batch write operations"""
        try:
            if not self.db:
                return False

            batch = self.db.batch()

            for operation in operations:
                op_type = operation.get('type')
                collection = operation.get('collection')
                document_id = operation.get('document_id')
                data = operation.get('data', {})

                if not all([op_type, collection, document_id]):
                    continue

                doc_ref = self.db.collection(collection).document(document_id)

                if op_type == 'set':
                    batch.set(doc_ref, data)
                elif op_type == 'update':
                    data['last_updated'] = datetime.utcnow()
                    batch.update(doc_ref, data)
                elif op_type == 'delete':
                    batch.delete(doc_ref)

            batch.commit()
            logger.info(f"Batch write completed with {len(operations)} operations")
            return True

        except Exception as e:
            logger.error(f"Failed to perform batch write: {str(e)}")
            return False

    async def get_collection_stats(self, collection: str) -> Dict[str, Any]:
        """Get statistics for a collection"""
        try:
            if not self.db:
                return {}

            # Count documents (simple approach for small collections)
            docs = self.db.collection(collection).stream()
            doc_count = len(list(docs))

            # Get latest document
            latest_doc = await self.get_latest_document(collection)
            latest_timestamp = latest_doc.get('timestamp') if latest_doc else None

            return {
                "collection": collection,
                "document_count": doc_count,
                "latest_timestamp": latest_timestamp,
                "stats_generated_at": datetime.utcnow()
            }

        except Exception as e:
            logger.error(f"Failed to get stats for collection {collection}: {str(e)}")
            return {}

    async def cleanup_old_data(self, collection: str, days_to_keep: int = 90) -> int:
        """Clean up old data from collection"""
        try:
            if not self.db:
                return 0

            cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)

            # Query old documents
            query = self.db.collection(collection)
            query = query.where(filter=FieldFilter("timestamp", "<", cutoff_date))

            docs = query.stream()
            deleted_count = 0

            # Delete in batches
            batch = self.db.batch()
            batch_size = 0
            max_batch_size = 500

            for doc in docs:
                batch.delete(doc.reference)
                batch_size += 1
                deleted_count += 1

                if batch_size >= max_batch_size:
                    batch.commit()
                    batch = self.db.batch()
                    batch_size = 0

            # Commit remaining deletes
            if batch_size > 0:
                batch.commit()

            logger.info(f"Cleaned up {deleted_count} old documents from {collection}")
            return deleted_count

        except Exception as e:
            logger.error(f"Failed to cleanup old data from {collection}: {str(e)}")
            return 0

    async def health_check(self) -> bool:
        """Perform health check on Firestore service"""
        try:
            if not self.db:
                return False

            # Try to read from a test collection
            test_collection = "health_check"
            test_doc_id = "test_doc"

            # Try to write
            test_data = {
                "health_check": True,
                "timestamp": datetime.utcnow()
            }

            success = await self.store_document(test_collection, test_data, test_doc_id)

            if success:
                # Try to read
                read_doc = await self.get_document(test_collection, test_doc_id)
                if read_doc:
                    # Clean up test document
                    await self.delete_document(test_collection, test_doc_id)
                    return True

            return False

        except Exception as e:
            logger.error(f"Firestore health check failed: {str(e)}")
            return False

    async def close(self):
        """Close Firestore client connections"""
        try:
            if self.db:
                # Firestore client doesn't need explicit closing
                logger.info("Firestore service closed")
        except Exception as e:
            logger.error(f"Error closing Firestore service: {str(e)}")
