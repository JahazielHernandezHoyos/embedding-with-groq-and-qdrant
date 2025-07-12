"""
Smart Sales Agent with RAG capabilities
"""
import logging
from typing import Dict, List, Any, Optional, Tuple
from groq import Groq
from tenacity import retry, stop_after_attempt, wait_exponential
import numpy as np
from datetime import datetime

from ..config import settings
from ..embeddings import EmbeddingGenerator, QdrantStorage
from ..data import SalesDataProcessor

logger = logging.getLogger(__name__)

class SmartSalesAgent:
    """Intelligent sales agent with RAG capabilities"""
    
    def __init__(self):
        self.groq_client = Groq(api_key=settings.GROQ_API_KEY)
        self.embedding_generator = EmbeddingGenerator()
        self.vector_store = QdrantStorage()
        self.data_processor = None
        
    def set_data_processor(self, processor: SalesDataProcessor):
        """Set the data processor"""
        self.data_processor = processor
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _query_groq(self, prompt: str, system_prompt: str = "") -> str:
        """Query Groq with retry logic"""
        try:
            messages = []
            
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            messages.append({"role": "user", "content": prompt})
            
            response = self.groq_client.chat.completions.create(
                model=settings.GROQ_MODEL,
                messages=messages,
                temperature=settings.GROQ_TEMPERATURE,
                max_tokens=settings.GROQ_MAX_TOKENS
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error querying Groq: {e}")
            return f"Error generating response: {str(e)}"
    
    def retrieve_context(self, query: str, context_type: str = "all") -> List[Dict[str, Any]]:
        """Retrieve relevant context from vector store"""
        # Generate embedding for query
        query_embedding = self.embedding_generator.generate_text_embedding(query)
        
        context_results = []
        
        if context_type in ["all", "customer"]:
            # Search customers
            customer_results = self.vector_store.search_customers(
                query_embedding=query_embedding,
                limit=3
            )
            context_results.extend(customer_results)
        
        if context_type in ["all", "product"]:
            # Search products
            product_results = self.vector_store.search_products(
                query_embedding=query_embedding,
                limit=3
            )
            context_results.extend(product_results)
        
        if context_type in ["all", "territory"]:
            # Search territories
            territory_results = self.vector_store.search_territories(
                query_embedding=query_embedding,
                limit=2
            )
            context_results.extend(territory_results)
        
        # Sort by relevance score
        context_results.sort(key=lambda x: x['score'], reverse=True)
        
        return context_results
    
    def format_context(self, context_results: List[Dict[str, Any]]) -> str:
        """Format context for prompt"""
        if not context_results:
            return "No relevant context found."
        
        formatted_context = []
        
        for result in context_results:
            payload = result['payload']
            score = result['score']
            
            context_item = f"**Relevance Score: {score:.3f}**\n"
            context_item += f"**Type:** {payload.get('type', 'unknown')}\n"
            context_item += f"**Key:** {payload.get('key', 'unknown')}\n"
            context_item += f"**Description:**\n{payload.get('text', 'No description available')}\n"
            
            # Add specific metadata based on type
            if payload.get('type') == 'customer':
                context_item += f"- Territory: {payload.get('territory', 'N/A')}\n"
                context_item += f"- Total Sales: ${payload.get('total_sales', 0):,.2f}\n"
                context_item += f"- Status: {payload.get('customer_status', 'N/A')}\n"
            elif payload.get('type') == 'product':
                context_item += f"- Product Line: {payload.get('product_line', 'N/A')}\n"
                context_item += f"- Performance Score: {payload.get('performance_score', 0):.3f}\n"
                context_item += f"- Deal Size: {payload.get('typical_deal_size', 'N/A')}\n"
            elif payload.get('type') == 'territory':
                context_item += f"- Market Share: {payload.get('market_share', 0):.2f}%\n"
                context_item += f"- Total Sales: ${payload.get('total_sales', 0):,.2f}\n"
                context_item += f"- Unique Customers: {payload.get('unique_customers', 0)}\n"
            
            formatted_context.append(context_item)
        
        return "\n\n---\n\n".join(formatted_context)
    
    def generate_sales_response(self, user_query: str, context_type: str = "all") -> Dict[str, Any]:
        """Generate sales response using RAG"""
        # Retrieve relevant context
        context_results = self.retrieve_context(user_query, context_type)
        formatted_context = self.format_context(context_results)
        
        # System prompt for sales agent
        system_prompt = """
        Eres un agente de ventas experto e inteligente con acceso a datos completos de ventas. 
        Tu trabajo es:
        
        1. Analizar la consulta del usuario y proporcionar insights valiosos
        2. Hacer recomendaciones basadas en datos reales
        3. Identificar oportunidades de venta
        4. Proporcionar estrategias personalizadas
        5. Usar un tono profesional pero amigable
        
        Siempre basa tus respuestas en los datos proporcionados y sé específico con números y ejemplos.
        Si no tienes información suficiente, menciona qué datos adicionales necesitarías.
        """
        
        # User prompt with context
        user_prompt = f"""
        Consulta del usuario: {user_query}
        
        Información relevante de la base de datos:
        {formatted_context}
        
        Por favor, proporciona una respuesta completa y útil basada en esta información.
        """
        
        # Generate response
        response = self._query_groq(user_prompt, system_prompt)
        
        return {
            'response': response,
            'context_used': len(context_results),
            'context_details': context_results,
            'timestamp': datetime.now().isoformat()
        }
    
    def analyze_customer(self, customer_name: str) -> Dict[str, Any]:
        """Analyze a specific customer"""
        query = f"Análisis completo del cliente {customer_name}"
        context_results = self.retrieve_context(query, "customer")
        
        # Filter for the specific customer
        customer_context = [
            r for r in context_results 
            if customer_name.lower() in r['payload'].get('key', '').lower()
        ]
        
        if not customer_context:
            return {
                'error': f'No se encontró información del cliente {customer_name}',
                'suggestions': [
                    'Verificar el nombre del cliente',
                    'Buscar por nombre parcial',
                    'Revisar la base de datos'
                ]
            }
        
        formatted_context = self.format_context(customer_context)
        
        system_prompt = """
        Eres un analista de ventas experto. Analiza el perfil del cliente y proporciona:
        1. Resumen del cliente
        2. Historial de compras
        3. Potencial de crecimiento
        4. Productos recomendados
        5. Estrategia de ventas sugerida
        6. Riesgos y oportunidades
        """
        
        user_prompt = f"""
        Analiza este cliente en detalle:
        {formatted_context}
        
        Proporciona un análisis completo y recomendaciones accionables.
        """
        
        response = self._query_groq(user_prompt, system_prompt)
        
        return {
            'customer_analysis': response,
            'customer_data': customer_context[0]['payload'] if customer_context else {},
            'timestamp': datetime.now().isoformat()
        }
    
    def recommend_products(self, customer_criteria: str) -> Dict[str, Any]:
        """Recommend products based on criteria"""
        query = f"Productos recomendados para {customer_criteria}"
        context_results = self.retrieve_context(query, "product")
        formatted_context = self.format_context(context_results)
        
        system_prompt = """
        Eres un experto en productos que hace recomendaciones basadas en datos.
        Proporciona:
        1. Top 3 productos recomendados
        2. Razones para cada recomendación
        3. Segmentos de clientes ideales
        4. Estrategias de pricing
        5. Métricas de rendimiento
        """
        
        user_prompt = f"""
        Criterios del cliente: {customer_criteria}
        
        Productos disponibles:
        {formatted_context}
        
        Proporciona recomendaciones específicas con justificación basada en datos.
        """
        
        response = self._query_groq(user_prompt, system_prompt)
        
        return {
            'recommendations': response,
            'products_analyzed': len(context_results),
            'timestamp': datetime.now().isoformat()
        }
    
    def analyze_territory(self, territory_name: str) -> Dict[str, Any]:
        """Analyze territory performance"""
        query = f"Análisis del territorio {territory_name}"
        context_results = self.retrieve_context(query, "territory")
        
        # Filter for specific territory
        territory_context = [
            r for r in context_results 
            if territory_name.lower() in r['payload'].get('key', '').lower()
        ]
        
        formatted_context = self.format_context(territory_context)
        
        system_prompt = """
        Eres un analista de territorios de ventas. Proporciona:
        1. Performance del territorio
        2. Comparación con otros territorios
        3. Oportunidades de crecimiento
        4. Productos con mejor desempeño
        5. Estrategias de expansión
        6. Riesgos del mercado
        """
        
        user_prompt = f"""
        Analiza este territorio:
        {formatted_context}
        
        Proporciona insights estratégicos y recomendaciones accionables.
        """
        
        response = self._query_groq(user_prompt, system_prompt)
        
        return {
            'territory_analysis': response,
            'territory_data': territory_context[0]['payload'] if territory_context else {},
            'timestamp': datetime.now().isoformat()
        }
    
    def generate_sales_pitch(self, customer_name: str, product_focus: str = "") -> Dict[str, Any]:
        """Generate personalized sales pitch"""
        query = f"Pitch de ventas para {customer_name} {product_focus}"
        context_results = self.retrieve_context(query, "all")
        formatted_context = self.format_context(context_results)
        
        system_prompt = """
        Eres un vendedor experto que crea pitches personalizados y persuasivos.
        Crea un pitch que:
        1. Se dirija específicamente al cliente
        2. Destaque beneficios relevantes
        3. Use datos concretos
        4. Incluya llamadas a la acción
        5. Aborde posibles objeciones
        6. Sea profesional pero convincente
        """
        
        user_prompt = f"""
        Cliente objetivo: {customer_name}
        Enfoque del producto: {product_focus}
        
        Información del cliente y mercado:
        {formatted_context}
        
        Crea un pitch de ventas personalizado y persuasivo.
        """
        
        response = self._query_groq(user_prompt, system_prompt)
        
        return {
            'sales_pitch': response,
            'personalization_level': len(context_results),
            'timestamp': datetime.now().isoformat()
        }
    
    def get_sales_insights(self, query: str) -> Dict[str, Any]:
        """Get general sales insights"""
        context_results = self.retrieve_context(query, "all")
        formatted_context = self.format_context(context_results)
        
        system_prompt = """
        Eres un consultor de ventas senior que proporciona insights estratégicos.
        Analiza los datos y proporciona:
        1. Tendencias identificadas
        2. Oportunidades de mercado
        3. Recomendaciones estratégicas
        4. Métricas clave
        5. Próximos pasos sugeridos
        """
        
        user_prompt = f"""
        Consulta: {query}
        
        Datos disponibles:
        {formatted_context}
        
        Proporciona insights valiosos y recomendaciones estratégicas.
        """
        
        response = self._query_groq(user_prompt, system_prompt)
        
        return {
            'insights': response,
            'data_points_analyzed': len(context_results),
            'timestamp': datetime.now().isoformat()
        } 