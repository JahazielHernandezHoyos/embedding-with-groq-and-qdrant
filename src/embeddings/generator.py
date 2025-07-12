"""
Embedding generation using Groq and sentence-transformers
"""
import asyncio
from typing import List, Dict, Any, Optional
import logging
from sentence_transformers import SentenceTransformer
from groq import Groq
import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential
import time

from ..config import settings

logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """Generate embeddings for sales data"""
    
    def __init__(self):
        self.groq_client = Groq(api_key=settings.GROQ_API_KEY)
        self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
        self.rate_limit_tracker = []
        
    def _check_rate_limit(self) -> None:
        """Check Groq API rate limits"""
        current_time = time.time()
        # Remove requests older than 1 minute
        self.rate_limit_tracker = [
            req_time for req_time in self.rate_limit_tracker 
            if current_time - req_time < 60
        ]
        
        if len(self.rate_limit_tracker) >= settings.RATE_LIMIT_REQUESTS:
            sleep_time = 60 - (current_time - self.rate_limit_tracker[0])
            logger.info(f"Rate limit reached, sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
        
        self.rate_limit_tracker.append(current_time)
    
    def generate_text_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text using sentence-transformers"""
        try:
            embedding = self.embedding_model.encode(text)
            return embedding.astype(np.float32)
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return np.zeros(settings.EMBEDDING_DIMENSION, dtype=np.float32)
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def enhance_text_with_groq(self, text: str, context: str = "") -> str:
        """Enhance text using Groq for better embeddings"""
        self._check_rate_limit()
        
        try:
            prompt = f"""
            Enhance this customer/product description for better semantic search:
            
            Context: {context}
            Original text: {text}
            
            Create a rich, descriptive text that captures:
            1. Key business characteristics
            2. Sales potential
            3. Market segment
            4. Product preferences
            5. Geographic and demographic insights
            
            Keep it concise but informative (max 200 words):
            """
            
            response = self.groq_client.chat.completions.create(
                model=settings.GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=settings.GROQ_TEMPERATURE,
                max_tokens=settings.GROQ_MAX_TOKENS
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error enhancing text with Groq: {e}")
            return text
    
    def create_customer_text(self, customer_profile: Dict[str, Any]) -> str:
        """Create comprehensive text for customer embedding"""
        text = f"""
        Customer: {customer_profile['name']}
        Location: {customer_profile['city']}, {customer_profile['state']}, {customer_profile['country']}
        Territory: {customer_profile['territory']}
        Contact: {customer_profile['contact_name']}
        
        Sales Profile:
        - Total Orders: {customer_profile['total_orders']}
        - Total Sales: ${customer_profile['total_sales']:,.2f}
        - Average Order Value: ${customer_profile['avg_order_value']:,.2f}
        - Preferred Products: {', '.join(customer_profile['preferred_products'])}
        - Deal Sizes: {', '.join(customer_profile['deal_sizes'])}
        - Status: {customer_profile['customer_status']}
        - Last Order: {customer_profile['last_order_date']}
        
        Customer Segment: {customer_profile['territory']} market with focus on {customer_profile['preferred_products'][0]} products
        """
        
        return text.strip()
    
    def create_product_text(self, product_data: Dict[str, Any]) -> str:
        """Create comprehensive text for product embedding"""
        text = f"""
        Product: {product_data['product_line']} - {product_data['product_code']}
        
        Performance Metrics:
        - Total Sales: ${product_data['total_sales']:,.2f}
        - Average Sales per Order: ${product_data['avg_sales']:,.2f}
        - Total Orders: {product_data['order_count']}
        - Average Price: ${product_data['avg_price']:,.2f}
        - Total Quantity Sold: {product_data['total_quantity']}
        - Typical Deal Size: {product_data['typical_deal_size']}
        - Performance Score: {product_data['performance_score']:.3f}
        
        Product Category: {product_data['product_line']} with strong performance in {product_data['typical_deal_size']} market segment
        """
        
        return text.strip()
    
    def create_territory_text(self, territory_name: str, territory_data: Dict[str, Any]) -> str:
        """Create comprehensive text for territory embedding"""
        top_products = list(territory_data['top_products'].keys())[:3]
        
        text = f"""
        Territory: {territory_name}
        
        Market Performance:
        - Total Sales: ${territory_data['total_sales']:,.2f}
        - Average Sales: ${territory_data['avg_sales']:,.2f}
        - Total Orders: {territory_data['total_orders']}
        - Unique Customers: {territory_data['unique_customers']}
        - Market Share: {territory_data['market_share']:.2f}%
        
        Top Products: {', '.join(top_products)}
        Deal Distribution: {territory_data['deal_distribution']}
        
        Market Characteristics: {territory_name} region with strong demand for {top_products[0] if top_products else 'various'} products
        """
        
        return text.strip()
    
    async def generate_customer_embeddings(self, customer_profiles: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Generate embeddings for all customers"""
        embeddings = {}
        
        for customer_name, profile in customer_profiles.items():
            logger.info(f"Generating embedding for customer: {customer_name}")
            
            # Create base text
            base_text = self.create_customer_text(profile.__dict__)
            
            # Enhance with Groq (optional, can be disabled to save API calls)
            if settings.GROQ_API_KEY:
                enhanced_text = self.enhance_text_with_groq(
                    base_text, 
                    f"Customer in {profile.territory} territory"
                )
            else:
                enhanced_text = base_text
            
            # Generate embedding
            embedding = self.generate_text_embedding(enhanced_text)
            
            embeddings[customer_name] = {
                'profile': profile,
                'text': enhanced_text,
                'embedding': embedding,
                'metadata': {
                    'type': 'customer',
                    'territory': profile.territory,
                    'total_sales': profile.total_sales,
                    'customer_status': profile.customer_status
                }
            }
            
            # Small delay to avoid overwhelming the system
            await asyncio.sleep(0.1)
        
        logger.info(f"Generated embeddings for {len(embeddings)} customers")
        return embeddings
    
    async def generate_product_embeddings(self, product_catalog: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Generate embeddings for all products"""
        embeddings = {}
        
        for product_key, product_data in product_catalog.items():
            logger.info(f"Generating embedding for product: {product_key}")
            
            # Create base text
            base_text = self.create_product_text(product_data)
            
            # Enhance with Groq
            if settings.GROQ_API_KEY:
                enhanced_text = self.enhance_text_with_groq(
                    base_text,
                    f"Product in {product_data['product_line']} category"
                )
            else:
                enhanced_text = base_text
            
            # Generate embedding
            embedding = self.generate_text_embedding(enhanced_text)
            
            embeddings[product_key] = {
                'data': product_data,
                'text': enhanced_text,
                'embedding': embedding,
                'metadata': {
                    'type': 'product',
                    'product_line': product_data['product_line'],
                    'performance_score': product_data['performance_score'],
                    'typical_deal_size': product_data['typical_deal_size']
                }
            }
            
            await asyncio.sleep(0.1)
        
        logger.info(f"Generated embeddings for {len(embeddings)} products")
        return embeddings
    
    async def generate_territory_embeddings(self, territory_analysis: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Generate embeddings for all territories"""
        embeddings = {}
        
        for territory_name, territory_data in territory_analysis.items():
            logger.info(f"Generating embedding for territory: {territory_name}")
            
            # Create base text
            base_text = self.create_territory_text(territory_name, territory_data)
            
            # Enhance with Groq
            if settings.GROQ_API_KEY:
                enhanced_text = self.enhance_text_with_groq(
                    base_text,
                    f"Sales territory analysis for {territory_name}"
                )
            else:
                enhanced_text = base_text
            
            # Generate embedding
            embedding = self.generate_text_embedding(enhanced_text)
            
            embeddings[territory_name] = {
                'data': territory_data,
                'text': enhanced_text,
                'embedding': embedding,
                'metadata': {
                    'type': 'territory',
                    'market_share': territory_data['market_share'],
                    'total_sales': territory_data['total_sales'],
                    'unique_customers': territory_data['unique_customers']
                }
            }
            
            await asyncio.sleep(0.1)
        
        logger.info(f"Generated embeddings for {len(embeddings)} territories")
        return embeddings 