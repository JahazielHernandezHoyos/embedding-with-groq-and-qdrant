"""
Smart Sales Agent - Main Application
"""
import logging
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from src.config import settings
from src.api import router, initialize_services
from src.data import SalesDataProcessor
from src.embeddings import EmbeddingGenerator, QdrantStorage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("Starting Smart Sales Agent...")
    
    # Validate configuration
    config_errors = settings.validate()
    if config_errors:
        logger.error(f"Configuration errors: {config_errors}")
        raise RuntimeError(f"Configuration errors: {config_errors}")
    
    # Initialize services
    logger.info("Initializing services...")
    success = initialize_services()
    if not success:
        logger.error("Failed to initialize services")
        raise RuntimeError("Failed to initialize services")
    
    # # Process data and generate embeddings
    # logger.info("Processing data and generating embeddings...")
    # try:
    #     await initialize_embeddings()
    # except Exception as e:
    #     logger.error(f"Failed to initialize embeddings: {e}")
    #     raise RuntimeError(f"Failed to initialize embeddings: {e}")
    
    logger.info("Smart Sales Agent started successfully!")
    yield
    
    logger.info("Shutting down Smart Sales Agent...")

async def initialize_embeddings():
    """Initialize embeddings and populate vector store"""
    try:
        # Get global instances
        from src.api.routes import data_processor, vector_store
        
        if not data_processor or not vector_store:
            raise RuntimeError("Services not properly initialized")
        
        # Generate embeddings
        embedding_generator = EmbeddingGenerator()
        
        # Generate customer embeddings
        logger.info("Generating customer embeddings...")
        customer_embeddings = await embedding_generator.generate_customer_embeddings(
            data_processor.customer_profiles
        )
        
        # Generate product embeddings
        logger.info("Generating product embeddings...")
        product_embeddings = await embedding_generator.generate_product_embeddings(
            data_processor.product_catalog
        )
        
        # Generate territory embeddings
        logger.info("Generating territory embeddings...")
        territory_embeddings = await embedding_generator.generate_territory_embeddings(
            data_processor.territory_analysis
        )
        
        # Store all embeddings
        logger.info("Storing embeddings in Qdrant...")
        all_embeddings = {
            **customer_embeddings,
            **product_embeddings,
            **territory_embeddings
        }
        
        success = vector_store.store_embeddings(all_embeddings)
        if not success:
            raise RuntimeError("Failed to store embeddings in Qdrant")
        
        logger.info(f"Successfully stored {len(all_embeddings)} embeddings")
        
        # Get final stats
        stats = vector_store.get_collection_stats()
        logger.info(f"Vector store stats: {stats}")
        
    except Exception as e:
        logger.error(f"Error initializing embeddings: {e}")
        raise

# Create FastAPI app
app = FastAPI(
    title="Smart Sales Agent",
    description="Intelligent sales agent with RAG capabilities using Groq and Qdrant",
    version=settings.APP_VERSION,
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api/v1")

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Smart Sales Agent API",
        "version": settings.APP_VERSION,
        "status": "running"
    }

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Global exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "message": "Internal server error",
            "detail": str(exc) if settings.DEBUG else "An error occurred"
        }
    )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
        log_level="info"
    ) 