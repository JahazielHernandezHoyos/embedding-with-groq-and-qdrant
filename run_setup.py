#!/usr/bin/env python3
"""
Setup script for Smart Sales Agent
"""
import asyncio
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config import settings
from src.data import SalesDataProcessor
from src.embeddings import EmbeddingGenerator, QdrantStorage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    """Main setup function"""
    try:
        logger.info("🚀 Starting Smart Sales Agent Setup")
        
        # Validate configuration
        logger.info("Validating configuration...")
        config_errors = settings.validate()
        if config_errors:
            logger.error(f"Configuration errors: {config_errors}")
            return False
        
        # Initialize data processor
        logger.info("Initializing data processor...")
        processor = SalesDataProcessor(settings.DATA_PATH)
        result = processor.process_all()
        logger.info(f"Data processing completed: {result}")
        
        # Initialize vector store
        logger.info("Initializing vector store...")
        vector_store = QdrantStorage()
        health = vector_store.health_check()
        logger.info(f"Vector store health: {health}")
        
        # Generate embeddings
        logger.info("Generating embeddings...")
        embedding_generator = EmbeddingGenerator()
        
        # Generate all embeddings
        logger.info("Generating customer embeddings...")
        customer_embeddings = await embedding_generator.generate_customer_embeddings(
            processor.customer_profiles
        )
        
        logger.info("Generating product embeddings...")
        product_embeddings = await embedding_generator.generate_product_embeddings(
            processor.product_catalog
        )
        
        logger.info("Generating territory embeddings...")
        territory_embeddings = await embedding_generator.generate_territory_embeddings(
            processor.territory_analysis
        )
        
        # Store embeddings
        logger.info("Storing embeddings in Qdrant...")
        all_embeddings = {
            **customer_embeddings,
            **product_embeddings,
            **territory_embeddings
        }
        
        success = vector_store.store_embeddings(all_embeddings)
        if not success:
            logger.error("Failed to store embeddings")
            return False
        
        logger.info(f"Successfully stored {len(all_embeddings)} embeddings")
        
        # Final stats
        stats = vector_store.get_collection_stats()
        logger.info(f"Final vector store stats: {stats}")
        
        logger.info("✅ Setup completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1) 