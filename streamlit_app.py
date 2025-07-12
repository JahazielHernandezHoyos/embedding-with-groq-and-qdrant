"""
Streamlit Web Interface for Smart Sales Agent
"""
import streamlit as st
import requests
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any
import time

# Configure Streamlit
st.set_page_config(
    page_title="Smart Sales Agent",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_BASE_URL = "http://localhost:8000/api/v1"

class SalesAgentUI:
    """Streamlit UI for Sales Agent"""
    
    def __init__(self):
        self.session_state_init()
    
    def session_state_init(self):
        """Initialize session state"""
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        if 'api_status' not in st.session_state:
            st.session_state.api_status = None
    
    def check_api_status(self):
        """Check API health status"""
        try:
            response = requests.get(f"{API_BASE_URL}/health", timeout=5)
            if response.status_code == 200:
                st.session_state.api_status = "healthy"
                return True
            else:
                st.session_state.api_status = "unhealthy"
                return False
        except Exception as e:
            st.session_state.api_status = f"error: {str(e)}"
            return False
    
    def api_request(self, endpoint: str, method: str = "GET", data: Dict = None):
        """Make API request"""
        try:
            url = f"{API_BASE_URL}/{endpoint}"
            
            if method == "GET":
                response = requests.get(url, timeout=30)
            elif method == "POST":
                response = requests.post(url, json=data, timeout=30)
            
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"API Error: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            st.error(f"Request Error: {str(e)}")
            return None
    
    def render_sidebar(self):
        """Render sidebar"""
        st.sidebar.title("🚀 Smart Sales Agent")
        
        # API Status
        if self.check_api_status():
            st.sidebar.success("🟢 API Connected")
        else:
            st.sidebar.error(f"🔴 API Disconnected: {st.session_state.api_status}")
        
        # Navigation
        st.sidebar.markdown("---")
        page = st.sidebar.selectbox(
            "Selecciona una página",
            ["Chat", "Análisis de Clientes", "Recomendaciones", "Territorios", "Estadísticas"]
        )
        
        # Settings
        st.sidebar.markdown("---")
        st.sidebar.subheader("Configuración")
        
        context_type = st.sidebar.selectbox(
            "Tipo de contexto",
            ["all", "customer", "product", "territory"],
            help="Tipo de información a considerar en las consultas"
        )
        
        return page, context_type
    
    def render_chat_page(self, context_type: str):
        """Render chat interface"""
        st.header("💬 Chat con el Agente de Ventas")
        
        # Chat input
        user_input = st.chat_input("Escribe tu consulta aquí...")
        
        if user_input:
            # Add user message to chat history
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_input,
                "timestamp": time.time()
            })
            
            # Get response from API
            with st.spinner("Pensando..."):
                response = self.api_request(
                    "query",
                    method="POST",
                    data={"query": user_input, "context_type": context_type}
                )
            
            if response and response.get("success"):
                # Add assistant response to chat history
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response["data"]["response"],
                    "context_used": response["data"]["context_used"],
                    "timestamp": time.time()
                })
        
        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                
                if message["role"] == "assistant" and "context_used" in message:
                    st.caption(f"Contexto utilizado: {message['context_used']} elementos")
        
        # Clear chat button
        if st.button("🗑️ Limpiar Chat"):
            st.session_state.chat_history = []
            st.rerun()
    
    def render_customer_analysis_page(self):
        """Render customer analysis page"""
        st.header("👥 Análisis de Clientes")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Analizar Cliente Específico")
            customer_name = st.text_input("Nombre del cliente")
            
            if st.button("Analizar Cliente"):
                if customer_name:
                    with st.spinner("Analizando cliente..."):
                        response = self.api_request(
                            "analyze/customer",
                            method="POST",
                            data={"customer_name": customer_name}
                        )
                    
                    if response and response.get("success"):
                        st.success("Análisis completado")
                        st.markdown("### Análisis del Cliente")
                        st.write(response["data"]["customer_analysis"])
                        
                        if response["data"]["customer_data"]:
                            st.markdown("### Datos del Cliente")
                            st.json(response["data"]["customer_data"])
                    else:
                        st.error("No se pudo analizar el cliente")
        
        with col2:
            st.subheader("Top Clientes")
            limit = st.slider("Número de clientes", 5, 20, 10)
            
            if st.button("Obtener Top Clientes"):
                response = self.api_request(f"customers/top/{limit}")
                
                if response and response.get("success"):
                    customers = response["data"]["customers"]
                    df = pd.DataFrame(customers)
                    
                    # Display table
                    st.dataframe(df, use_container_width=True)
                    
                    # Create chart
                    fig = px.bar(
                        df.head(10),
                        x='name',
                        y='total_sales',
                        title='Top 10 Clientes por Ventas',
                        labels={'total_sales': 'Ventas Totales', 'name': 'Cliente'}
                    )
                    fig.update_xaxis(tickangle=45)
                    st.plotly_chart(fig, use_container_width=True)
    
    def render_recommendations_page(self):
        """Render product recommendations page"""
        st.header("🎯 Recomendaciones de Productos")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Recomendar Productos")
            customer_criteria = st.text_area(
                "Criterios del cliente",
                placeholder="Ej: Cliente en Europa interesado en motocicletas con presupuesto alto"
            )
            
            if st.button("Generar Recomendaciones"):
                if customer_criteria:
                    with st.spinner("Generando recomendaciones..."):
                        response = self.api_request(
                            "recommend/products",
                            method="POST",
                            data={"customer_criteria": customer_criteria}
                        )
                    
                    if response and response.get("success"):
                        st.success("Recomendaciones generadas")
                        st.markdown("### Recomendaciones")
                        st.write(response["data"]["recommendations"])
                        st.caption(f"Productos analizados: {response['data']['products_analyzed']}")
        
        with col2:
            st.subheader("Top Productos")
            limit = st.slider("Número de productos", 5, 20, 10)
            
            if st.button("Obtener Top Productos"):
                response = self.api_request(f"products/top/{limit}")
                
                if response and response.get("success"):
                    products = response["data"]["products"]
                    df = pd.DataFrame(products)
                    
                    # Display table
                    st.dataframe(df, use_container_width=True)
                    
                    # Create performance chart
                    fig = px.scatter(
                        df.head(10),
                        x='total_sales',
                        y='performance_score',
                        size='order_count',
                        hover_data=['product_line', 'product_code'],
                        title='Rendimiento de Productos',
                        labels={
                            'total_sales': 'Ventas Totales',
                            'performance_score': 'Score de Rendimiento'
                        }
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    def render_territory_analysis_page(self):
        """Render territory analysis page"""
        st.header("🌍 Análisis de Territorios")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Analizar Territorio")
            territory_name = st.selectbox(
                "Selecciona territorio",
                ["EMEA", "NA", "APAC", "Japan"]
            )
            
            if st.button("Analizar Territorio"):
                with st.spinner("Analizando territorio..."):
                    response = self.api_request(
                        "analyze/territory",
                        method="POST",
                        data={"territory_name": territory_name}
                    )
                
                if response and response.get("success"):
                    st.success("Análisis completado")
                    st.markdown("### Análisis del Territorio")
                    st.write(response["data"]["territory_analysis"])
                    
                    if response["data"]["territory_data"]:
                        st.markdown("### Datos del Territorio")
                        st.json(response["data"]["territory_data"])
        
        with col2:
            st.subheader("Insights de Territorios")
            
            if st.button("Obtener Insights"):
                response = self.api_request("territories/insights")
                
                if response and response.get("success"):
                    insights = response["data"]
                    
                    # Display metrics
                    st.metric("Total Territorios", insights["total_territories"])
                    
                    if "top_territory" in insights:
                        top_territory = insights["top_territory"]
                        st.metric(
                            f"Top Territory: {top_territory[0]}",
                            f"${top_territory[1]['total_sales']:,.2f}"
                        )
                    
                    # Create territory comparison chart
                    if "territory_breakdown" in insights:
                        territories = insights["territory_breakdown"]
                        territory_df = pd.DataFrame([
                            {
                                "territory": name,
                                "total_sales": data["total_sales"],
                                "market_share": data["market_share"],
                                "customers": data["unique_customers"]
                            }
                            for name, data in territories.items()
                        ])
                        
                        fig = px.pie(
                            territory_df,
                            values='total_sales',
                            names='territory',
                            title='Distribución de Ventas por Territorio'
                        )
                        st.plotly_chart(fig, use_container_width=True)
    
    def render_stats_page(self):
        """Render statistics page"""
        st.header("📊 Estadísticas del Sistema")
        
        # Get system stats
        response = self.api_request("stats")
        
        if response and response.get("success"):
            stats = response["data"]
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Clientes", stats["total_customers"])
            
            with col2:
                st.metric("Total Productos", stats["total_products"])
            
            with col3:
                st.metric("Total Territorios", stats["total_territories"])
            
            with col4:
                if "qdrant_stats" in stats and "total_points" in stats["qdrant_stats"]:
                    st.metric("Embeddings Almacenados", stats["qdrant_stats"]["total_points"])
            
            # Qdrant Statistics
            if "qdrant_stats" in stats:
                st.subheader("Estadísticas de Qdrant")
                
                qdrant_stats = stats["qdrant_stats"]
                if "type_counts" in qdrant_stats:
                    type_counts = qdrant_stats["type_counts"]
                    
                    # Create chart
                    fig = px.bar(
                        x=list(type_counts.keys()),
                        y=list(type_counts.values()),
                        title='Distribución de Embeddings por Tipo',
                        labels={'x': 'Tipo', 'y': 'Cantidad'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Raw stats
                st.json(qdrant_stats)
        else:
            st.error("No se pudieron obtener las estadísticas")
    
    def run(self):
        """Run the Streamlit app"""
        page, context_type = self.render_sidebar()
        
        if page == "Chat":
            self.render_chat_page(context_type)
        elif page == "Análisis de Clientes":
            self.render_customer_analysis_page()
        elif page == "Recomendaciones":
            self.render_recommendations_page()
        elif page == "Territorios":
            self.render_territory_analysis_page()
        elif page == "Estadísticas":
            self.render_stats_page()

# Run the app
if __name__ == "__main__":
    app = SalesAgentUI()
    app.run() 