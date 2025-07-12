"""
Qdrant storage for embeddings
"""
import logging
from typing import Dict, List, Any, Optional, Union
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter, 
    FieldCondition, MatchValue, SearchRequest
)
import numpy as np

from ..config import settings

logger = logging.getLogger(__name__)

class QdrantStorage:
    """Qdrant vector database storage"""
    
    def __init__(self):
        self.client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY
        )
        self.collection_name = settings.QDRANT_COLLECTION_NAME
        self._ensure_collection()
    
    def _ensure_collection(self) -> None:
        """Ensure collection exists"""
        try:
            # Check if collection exists
            collections = self.client.get_collections()
            collection_exists = any(
                collection.name == self.collection_name 
                for collection in collections.collections
            )
            
            if not collection_exists:
                # Create collection
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=settings.EMBEDDING_DIMENSION,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created collection: {self.collection_name}")
            else:
                logger.info(f"Collection already exists: {self.collection_name}")
                
        except Exception as e:
            logger.error(f"Error ensuring collection: {e}")
            raise
    
    def store_embeddings(self, embeddings: Dict[str, Dict[str, Any]]) -> bool:
        """Store embeddings in Qdrant"""
        try:
            # Check if embeddings dictionary is empty
            if not embeddings:
                logger.warning("No embeddings to store - embeddings dictionary is empty")
                return True
            
            points = []
            
            for key, embedding_data in embeddings.items():
                # Create point
                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding_data['embedding'].tolist(),
                    payload={
                        'key': key,
                        'text': embedding_data['text'],
                        'metadata': embedding_data['metadata'],
                        **embedding_data['metadata']  # Flatten metadata for easier filtering
                    }
                )
                points.append(point)
            
            # Final check before batch insert
            if not points:
                logger.warning("No valid points to store")
                return True
            
            # Batch insert
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
            logger.info(f"Stored {len(points)} embeddings in Qdrant")
            return True
            
        except Exception as e:
            logger.error(f"Error storing embeddings: {e}")
            return False
    
    def search_similar(
        self, 
        query_embedding: np.ndarray, 
        limit: int = 10,
        filter_conditions: Optional[Dict[str, Any]] = None,
        score_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """Search for similar embeddings"""
        try:
            # Build filter if provided
            query_filter = None
            if filter_conditions:
                conditions = []
                for field, value in filter_conditions.items():
                    conditions.append(
                        FieldCondition(
                            key=field,
                            match=MatchValue(value=value)
                        )
                    )
                query_filter = Filter(must=conditions)
            
            # Search
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding.tolist(),
                query_filter=query_filter,
                limit=limit,
                score_threshold=score_threshold
            )
            
            # Format results
            results = []
            for hit in search_result:
                results.append({
                    'id': hit.id,
                    'score': hit.score,
                    'payload': hit.payload
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching embeddings: {e}")
            return []
    
    def search_customers(
        self, 
        query_embedding: np.ndarray, 
        territory: Optional[str] = None,
        min_sales: Optional[float] = None,
        customer_status: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search for similar customers with filters"""
        filter_conditions = {'type': 'customer'}
        
        if territory:
            filter_conditions['territory'] = territory
        if customer_status:
            filter_conditions['customer_status'] = customer_status
            
        results = self.search_similar(
            query_embedding=query_embedding,
            limit=limit,
            filter_conditions=filter_conditions
        )
        
        # Additional filtering for numeric values
        if min_sales is not None:
            results = [
                r for r in results 
                if r['payload'].get('total_sales', 0) >= min_sales
            ]
        
        return results
    
    def search_products(
        self, 
        query_embedding: np.ndarray, 
        product_line: Optional[str] = None,
        deal_size: Optional[str] = None,
        min_performance: Optional[float] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search for similar products with filters"""
        filter_conditions = {'type': 'product'}
        
        if product_line:
            filter_conditions['product_line'] = product_line
        if deal_size:
            filter_conditions['typical_deal_size'] = deal_size
            
        results = self.search_similar(
            query_embedding=query_embedding,
            limit=limit,
            filter_conditions=filter_conditions
        )
        
        # Additional filtering for numeric values
        if min_performance is not None:
            results = [
                r for r in results 
                if r['payload'].get('performance_score', 0) >= min_performance
            ]
        
        return results
    
    def search_territories(
        self, 
        query_embedding: np.ndarray, 
        min_market_share: Optional[float] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search for similar territories with filters"""
        filter_conditions = {'type': 'territory'}
        
        results = self.search_similar(
            query_embedding=query_embedding,
            limit=limit,
            filter_conditions=filter_conditions
        )
        
        # Additional filtering for numeric values
        if min_market_share is not None:
            results = [
                r for r in results 
                if r['payload'].get('market_share', 0) >= min_market_share
            ]
        
        return results
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        try:
            info = self.client.get_collection(self.collection_name)
            
            # Get counts by type
            type_counts = {}
            for doc_type in ['customer', 'product', 'territory']:
                count_result = self.client.count(
                    collection_name=self.collection_name,
                    count_filter=Filter(
                        must=[
                            FieldCondition(
                                key='type',
                                match=MatchValue(value=doc_type)
                            )
                        ]
                    )
                )
                type_counts[doc_type] = count_result.count
            
            return {
                'total_points': info.points_count,
                'vectors_count': info.vectors_count,
                'type_counts': type_counts,
                'status': info.status
            }
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {}
    
    def clear_collection(self) -> bool:
        """Clear all data from collection"""
        try:
            self.client.delete_collection(self.collection_name)
            self._ensure_collection()
            logger.info(f"Cleared collection: {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            return False
    
    def health_check(self) -> Dict[str, Any]:
        """Check Qdrant health"""
        try:
            collections = self.client.get_collections()
            return {
                'status': 'healthy',
                'collections_count': len(collections.collections),
                'target_collection_exists': any(
                    c.name == self.collection_name for c in collections.collections
                )
            }
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                'status': 'unhealthy',
                'error': str(e)
            } 