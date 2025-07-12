"""
Pydantic models for API
"""
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from datetime import datetime

class QueryRequest(BaseModel):
    """Request model for queries"""
    query: str = Field(..., description="User query")
    context_type: str = Field(default="all", description="Type of context to retrieve")

class CustomerAnalysisRequest(BaseModel):
    """Request model for customer analysis"""
    customer_name: str = Field(..., description="Name of the customer to analyze")

class ProductRecommendationRequest(BaseModel):
    """Request model for product recommendations"""
    customer_criteria: str = Field(..., description="Customer criteria for recommendations")

class TerritoryAnalysisRequest(BaseModel):
    """Request model for territory analysis"""
    territory_name: str = Field(..., description="Name of the territory to analyze")

class SalesPitchRequest(BaseModel):
    """Request model for sales pitch generation"""
    customer_name: str = Field(..., description="Target customer name")
    product_focus: str = Field(default="", description="Product focus for the pitch")

class ApiResponse(BaseModel):
    """Generic API response model"""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.now)

class SalesResponse(BaseModel):
    """Response model for sales queries"""
    response: str
    context_used: int
    timestamp: str

class CustomerAnalysisResponse(BaseModel):
    """Response model for customer analysis"""
    customer_analysis: str
    customer_data: Dict[str, Any]
    timestamp: str

class ProductRecommendationResponse(BaseModel):
    """Response model for product recommendations"""
    recommendations: str
    products_analyzed: int
    timestamp: str

class TerritoryAnalysisResponse(BaseModel):
    """Response model for territory analysis"""
    territory_analysis: str
    territory_data: Dict[str, Any]
    timestamp: str

class SalesPitchResponse(BaseModel):
    """Response model for sales pitch"""
    sales_pitch: str
    personalization_level: int
    timestamp: str

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    qdrant_status: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)

class StatsResponse(BaseModel):
    """Statistics response"""
    total_customers: int
    total_products: int
    total_territories: int
    qdrant_stats: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now) 