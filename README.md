# 🚀 Smart Sales Agent

Un agente de ventas inteligente que utiliza **Groq** para procesamiento de lenguaje natural, **Qdrant** como base de datos vectorial, y **RAG** (Retrieval-Augmented Generation) para proporcionar insights de ventas basados en datos reales.
https://www.linkedin.com/posts/activity-7349848826574168064-J5Pe?utm_source=share&utm_medium=member_desktop&rcm=ACoAACG9uGQBbKa1oO3QmZLZxs6juCA3bh6mI9I


## ✨ Características

- **Agente de ventas inteligente** con capacidades de chat conversacional
- **Análisis de clientes** personalizado con recomendaciones
- **Búsqueda semántica** avanzada usando embeddings
- **Análisis de productos** y tendencias de mercado
- **Insights de territorios** con visualizaciones interactivas
- **API REST** completa con FastAPI
- **Interfaz web** moderna con Streamlit
- **Arquitectura modular** y escalable

## 🏗️ Arquitectura

```
Smart Sales Agent/
├── src/
│   ├── config/          # Configuración
│   ├── data/           # Procesamiento de datos
│   ├── embeddings/     # Generación y almacenamiento de embeddings
│   ├── agent/          # Agente de ventas con RAG
│   └── api/            # API REST
├── sales_data_sample.csv # Datos de ventas
├── main.py             # Aplicación FastAPI
├── streamlit_app.py    # Interfaz web
├── docker-compose.yml  # Orquestación de servicios
└── requirements.txt    # Dependencias
```

## 🚀 Instalación y Uso

### 1. Configuración del Entorno

```bash
# Clonar el repositorio
git clone <tu-repo>
cd embedding_with_groq

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

### 2. Configuración de Variables de Entorno

Crea un archivo `.env` con tu API key de Groq:

```env
# Configuración de Groq
GROQ_API_KEY=tu_api_key_aquí

# Configuración de Qdrant
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION_NAME=sales_data

# Configuración de datos
DATA_PATH=sales_data_sample.csv
```

### 3. Levantar Qdrant con Docker

```bash
# Levantar Qdrant
docker-compose up -d qdrant

# Verificar que esté funcionando
curl http://localhost:6333/health
```

### 4. Inicializar el Sistema

```bash
# Opción 1: Setup manual
python run_setup.py

# Opción 2: Usar la aplicación principal (se inicializa automáticamente)
python main.py
```

### 5. Usar la Aplicación

#### API REST

```bash
# Levantar la API
python main.py

# La API estará disponible en http://localhost:8000
# Documentación interactiva: http://localhost:8000/docs
```

#### Interfaz Web

```bash
# Levantar Streamlit (en otra terminal)
streamlit run streamlit_app.py

# La interfaz web estará disponible en http://localhost:8501
```

#### Docker Compose (Todo en uno)

```bash
# Levantar todos los servicios
docker-compose up -d

# Ver logs
docker-compose logs -f
```

## 🔧 Uso de la API

### Endpoints Principales

```python
# Consulta general al agente
POST /api/v1/query
{
    "query": "¿Cuáles son los mejores clientes en Europa?",
    "context_type": "customer"
}

# Análisis de cliente específico
POST /api/v1/analyze/customer
{
    "customer_name": "Land of Toys Inc."
}

# Recomendaciones de productos
POST /api/v1/recommend/products
{
    "customer_criteria": "Cliente europeo interesado en motocicletas"
}

# Análisis de territorio
POST /api/v1/analyze/territory
{
    "territory_name": "EMEA"
}

# Generar pitch de ventas
POST /api/v1/generate/pitch
{
    "customer_name": "Land of Toys Inc.",
    "product_focus": "motocicletas"
}
```

## 🎯 Casos de Uso

### 1. Análisis de Clientes
```python
# Encontrar clientes similares
"Busca clientes similares a Land of Toys Inc."

# Análisis de comportamiento
"¿Qué patrones de compra tiene el cliente X?"

# Segmentación
"¿Cómo puedo segmentar mis clientes por territorio?"
```

### 2. Recomendaciones de Productos
```python
# Basado en historial
"¿Qué productos debería ofrecer a clientes de motocicletas?"

# Cross-selling
"¿Qué productos complementarios puedo vender?"

# Análisis de tendencias
"¿Cuáles son los productos más exitosos en Europa?"
```

### 3. Estrategias de Ventas
```python
# Pitch personalizado
"Crea un pitch para el cliente X enfocado en producto Y"

# Análisis de territorio
"¿Qué oportunidades hay en el mercado APAC?"

# Forecasting
"¿Cuál es el potencial de crecimiento en este territorio?"
```

## 🏃‍♂️ Desarrollo

### Estructura del Código

- **`src/config/`**: Configuración centralizada
- **`src/data/`**: Procesamiento y análisis de datos de ventas
- **`src/embeddings/`**: Generación con Groq y almacenamiento en Qdrant
- **`src/agent/`**: Agente de ventas con capacidades RAG
- **`src/api/`**: API REST con FastAPI

### Agregar Nuevas Funcionalidades

1. **Nuevo tipo de análisis**:
   - Agregar método en `SalesDataProcessor`
   - Crear embeddings correspondientes
   - Añadir endpoint en la API

2. **Nuevas fuentes de datos**:
   - Extender `SalesDataProcessor`
   - Actualizar generación de embeddings
   - Modificar el agente según sea necesario

## 📊 Métricas y Monitoreo

### Endpoints de Monitoreo

```bash
# Estado del sistema
GET /api/v1/health

# Estadísticas
GET /api/v1/stats

# Top clientes
GET /api/v1/customers/top/10

# Top productos
GET /api/v1/products/top/10
```

### Métricas de Qdrant

```python
# Estadísticas de la colección
{
    "total_points": 150,
    "vectors_count": 150,
    "type_counts": {
        "customer": 92,
        "product": 48,
        "territory": 10
    }
}
```

## 🔍 Troubleshooting

### Problemas Comunes

1. **Error de conexión con Groq**:
   - Verificar `GROQ_API_KEY`
   - Comprobar límites de rate limiting

2. **Error de conexión con Qdrant**:
   - Verificar que Docker esté ejecutándose
   - Comprobar puerto 6333

3. **Error de procesamiento de datos**:
   - Verificar que `sales_data_sample.csv` existe
   - Comprobar formato de datos

### Logs

```bash
# Ver logs de la aplicación
python main.py  # Logs en consola

# Ver logs de Docker
docker-compose logs -f

# Ver logs específicos de Qdrant
docker-compose logs -f qdrant
```

## 🛠️ Configuración Avanzada

### Variables de Entorno Completas

```env
# Aplicación
APP_NAME=Smart Sales Agent
DEBUG=false

# Groq
GROQ_MODEL=llama3-8b-8192
GROQ_TEMPERATURE=0.7
GROQ_MAX_TOKENS=1024

# Qdrant
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION_NAME=sales_data

# Embeddings
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384

# Rate Limiting
RATE_LIMIT_REQUESTS=30
RATE_LIMIT_WINDOW=60
```

## 🤝 Contribuciones

1. Fork el proyecto
2. Crear branch para feature (`git checkout -b feature/AmazingFeature`)
3. Commit cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push al branch (`git push origin feature/AmazingFeature`)
5. Abrir Pull Request

## 📝 Licencia

Este proyecto está bajo la Licencia MIT.

## 🙏 Agradecimientos

- **Groq** por el acceso a modelos de lenguaje de alta velocidad
- **Qdrant** por la base de datos vectorial
- **Streamlit** por la interfaz web
- **FastAPI** por el framework web moderno 
