"""
API routes for the Smart Sales Agent
"""
from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any
import logging

from .models import (
    QueryRequest, CustomerAnalysisRequest, ProductRecommendationRequest,
    TerritoryAnalysisRequest, SalesPitchRequest, ApiResponse, 
    SalesResponse, CustomerAnalysisResponse, ProductRecommendationResponse,
    TerritoryAnalysisResponse, SalesPitchResponse, HealthResponse, StatsResponse
)
from ..agent import SmartSalesAgent
from ..data import SalesDataProcessor
from ..embeddings import QdrantStorage
from ..config import settings

logger = logging.getLogger(__name__)

# Global instances (will be initialized on startup)
sales_agent: SmartSalesAgent = None
data_processor: SalesDataProcessor = None
vector_store: QdrantStorage = None

router = APIRouter()

def get_sales_agent() -> SmartSalesAgent:
    """Dependency to get sales agent instance"""
    if sales_agent is None:
        raise HTTPException(status_code=500, detail="Sales agent not initialized")
    return sales_agent

def get_data_processor() -> SalesDataProcessor:
    """Dependency to get data processor instance"""
    if data_processor is None:
        raise HTTPException(status_code=500, detail="Data processor not initialized")
    return data_processor

def get_vector_store() -> QdrantStorage:
    """Dependency to get vector store instance"""
    if vector_store is None:
        raise HTTPException(status_code=500, detail="Vector store not initialized")
    return vector_store

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        qdrant_health = vector_store.health_check() if vector_store else {"status": "not_initialized"}
        
        return HealthResponse(
            status="healthy",
            version=settings.APP_VERSION,
            qdrant_status=qdrant_health
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@router.get("/stats", response_model=StatsResponse)
async def get_stats(
    processor: SalesDataProcessor = Depends(get_data_processor),
    store: QdrantStorage = Depends(get_vector_store)
):
    """Get system statistics"""
    try:
        qdrant_stats = store.get_collection_stats()
        
        return StatsResponse(
            total_customers=len(processor.customer_profiles),
            total_products=len(processor.product_catalog),
            total_territories=len(processor.territory_analysis),
            qdrant_stats=qdrant_stats
        )
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")

@router.post("/query", response_model=ApiResponse)
async def query_sales_agent(
    request: QueryRequest,
    agent: SmartSalesAgent = Depends(get_sales_agent)
):
    """Query the sales agent"""
    try:
        result = agent.generate_sales_response(
            user_query=request.query,
            context_type=request.context_type
        )
        
        return ApiResponse(
            success=True,
            message="Query processed successfully",
            data=result
        )
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@router.post("/analyze/customer", response_model=ApiResponse)
async def analyze_customer(
    request: CustomerAnalysisRequest,
    agent: SmartSalesAgent = Depends(get_sales_agent)
):
    """Analyze a specific customer"""
    try:
        result = agent.analyze_customer(request.customer_name)
        
        if 'error' in result:
            return ApiResponse(
                success=False,
                message=result['error'],
                data=result
            )
        
        return ApiResponse(
            success=True,
            message="Customer analysis completed",
            data=result
        )
    except Exception as e:
        logger.error(f"Error analyzing customer: {e}")
        raise HTTPException(status_code=500, detail=f"Error analyzing customer: {str(e)}")

@router.post("/recommend/products", response_model=ApiResponse)
async def recommend_products(
    request: ProductRecommendationRequest,
    agent: SmartSalesAgent = Depends(get_sales_agent)
):
    """Get product recommendations"""
    try:
        result = agent.recommend_products(request.customer_criteria)
        
        return ApiResponse(
            success=True,
            message="Product recommendations generated",
            data=result
        )
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")

@router.post("/analyze/territory", response_model=ApiResponse)
async def analyze_territory(
    request: TerritoryAnalysisRequest,
    agent: SmartSalesAgent = Depends(get_sales_agent)
):
    """Analyze territory performance"""
    try:
        result = agent.analyze_territory(request.territory_name)
        
        return ApiResponse(
            success=True,
            message="Territory analysis completed",
            data=result
        )
    except Exception as e:
        logger.error(f"Error analyzing territory: {e}")
        raise HTTPException(status_code=500, detail=f"Error analyzing territory: {str(e)}")

@router.post("/generate/pitch", response_model=ApiResponse)
async def generate_sales_pitch(
    request: SalesPitchRequest,
    agent: SmartSalesAgent = Depends(get_sales_agent)
):
    """Generate personalized sales pitch"""
    try:
        result = agent.generate_sales_pitch(
            customer_name=request.customer_name,
            product_focus=request.product_focus
        )
        
        return ApiResponse(
            success=True,
            message="Sales pitch generated",
            data=result
        )
    except Exception as e:
        logger.error(f"Error generating sales pitch: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating sales pitch: {str(e)}")

@router.post("/insights", response_model=ApiResponse)
async def get_sales_insights(
    request: QueryRequest,
    agent: SmartSalesAgent = Depends(get_sales_agent)
):
    """Get general sales insights"""
    try:
        result = agent.get_sales_insights(request.query)
        
        return ApiResponse(
            success=True,
            message="Sales insights generated",
            data=result
        )
    except Exception as e:
        logger.error(f"Error generating insights: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating insights: {str(e)}")

@router.get("/customers/top/{limit}")
async def get_top_customers(
    limit: int = 10,
    processor: SalesDataProcessor = Depends(get_data_processor)
):
    """Get top customers by sales"""
    try:
        top_customers = processor.get_top_customers(limit)
        
        customers_data = []
        for customer in top_customers:
            customers_data.append({
                'name': customer.name,
                'total_sales': customer.total_sales,
                'total_orders': customer.total_orders,
                'avg_order_value': customer.avg_order_value,
                'territory': customer.territory,
                'status': customer.customer_status
            })
        
        return ApiResponse(
            success=True,
            message=f"Top {limit} customers retrieved",
            data={'customers': customers_data}
        )
    except Exception as e:
        logger.error(f"Error getting top customers: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting top customers: {str(e)}")

@router.get("/products/top/{limit}")
async def get_top_products(
    limit: int = 10,
    processor: SalesDataProcessor = Depends(get_data_processor)
):
    """Get top products by performance"""
    try:
        top_products = processor.get_top_products(limit)
        
        return ApiResponse(
            success=True,
            message=f"Top {limit} products retrieved",
            data={'products': top_products}
        )
    except Exception as e:
        logger.error(f"Error getting top products: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting top products: {str(e)}")

@router.get("/territories/insights")
async def get_territory_insights(
    processor: SalesDataProcessor = Depends(get_data_processor)
):
    """Get territory insights"""
    try:
        insights = processor.get_territory_insights()
        
        return ApiResponse(
            success=True,
            message="Territory insights retrieved",
            data=insights
        )
    except Exception as e:
        logger.error(f"Error getting territory insights: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting territory insights: {str(e)}")

# Initialization function (to be called on startup)
def initialize_services():
    """Initialize all services"""
    global sales_agent, data_processor, vector_store
    
    try:
        # Initialize data processor
        data_processor = SalesDataProcessor(settings.DATA_PATH)
        logger.info("Processing sales data...")
        processing_result = data_processor.process_all()
        logger.info(f"Data processing completed: {processing_result}")
        
        # Initialize vector store
        vector_store = QdrantStorage()
        logger.info("Vector store initialized")
        
        # Initialize sales agent
        sales_agent = SmartSalesAgent()
        sales_agent.set_data_processor(data_processor)
        logger.info("Sales agent initialized")
        
        logger.info("All services initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error initializing services: {e}")
        return False 