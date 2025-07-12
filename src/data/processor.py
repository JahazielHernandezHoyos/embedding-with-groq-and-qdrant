"""
Data processor for sales data analysis
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

@dataclass
class SalesRecord:
    """Sales record data structure"""
    order_number: int
    quantity_ordered: int
    price_each: float
    sales: float
    order_date: str
    status: str
    product_line: str
    product_code: str
    customer_name: str
    phone: str
    city: str
    state: str
    country: str
    territory: str
    contact_name: str
    deal_size: str
    year: int
    quarter: int
    month: int

@dataclass
class CustomerProfile:
    """Customer profile data structure"""
    name: str
    phone: str
    city: str
    state: str
    country: str
    territory: str
    contact_name: str
    total_orders: int
    total_sales: float
    avg_order_value: float
    preferred_products: List[str]
    deal_sizes: List[str]
    last_order_date: str
    customer_status: str

class SalesDataProcessor:
    """Process and analyze sales data"""
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.df: Optional[pd.DataFrame] = None
        self.customer_profiles: Dict[str, CustomerProfile] = {}
        self.product_catalog: Dict[str, Dict] = {}
        self.territory_analysis: Dict[str, Dict] = {}
        
    def load_data(self) -> bool:
        """Load sales data from CSV"""
        try:
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
            
            for encoding in encodings:
                try:
                    self.df = pd.read_csv(self.data_path, encoding=encoding)
                    logger.info(f"Loaded {len(self.df)} sales records with {encoding} encoding")
                    return True
                except UnicodeDecodeError:
                    continue
            
            # If all encodings fail, try with errors='ignore'
            self.df = pd.read_csv(self.data_path, encoding='utf-8', errors='ignore')
            logger.info(f"Loaded {len(self.df)} sales records with error handling")
            return True
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False
    
    def clean_data(self) -> None:
        """Clean and preprocess the data"""
        if self.df is None:
            return
        
        # Convert date columns
        self.df['ORDERDATE'] = pd.to_datetime(self.df['ORDERDATE'])
        
        # Clean numeric columns
        self.df['SALES'] = pd.to_numeric(self.df['SALES'], errors='coerce')
        self.df['PRICEEACH'] = pd.to_numeric(self.df['PRICEEACH'], errors='coerce')
        self.df['QUANTITYORDERED'] = pd.to_numeric(self.df['QUANTITYORDERED'], errors='coerce')
        
        # Fill missing values
        self.df['STATE'] = self.df['STATE'].fillna('Unknown')
        self.df['POSTALCODE'] = self.df['POSTALCODE'].fillna('Unknown')
        
        # Remove duplicates
        initial_count = len(self.df)
        self.df = self.df.drop_duplicates()
        logger.info(f"Removed {initial_count - len(self.df)} duplicate records")
    
    def create_customer_profiles(self) -> None:
        """Create comprehensive customer profiles"""
        if self.df is None:
            return
        
        customer_data = self.df.groupby('CUSTOMERNAME').agg({
            'ORDERNUMBER': 'count',
            'SALES': ['sum', 'mean'],
            'PRODUCTLINE': lambda x: x.value_counts().index[0],  # Most frequent product
            'DEALSIZE': lambda x: list(x.unique()),
            'ORDERDATE': 'max',
            'PHONE': 'first',
            'CITY': 'first',
            'STATE': 'first',
            'COUNTRY': 'first',
            'TERRITORY': 'first',
            'CONTACTFIRSTNAME': 'first',
            'CONTACTLASTNAME': 'first',
            'STATUS': lambda x: 'Active' if 'Shipped' in x.values else 'Inactive'
        }).reset_index()
        
        # Flatten column names
        customer_data.columns = ['customer_name', 'total_orders', 'total_sales', 'avg_order_value',
                               'preferred_product', 'deal_sizes', 'last_order_date', 'phone',
                               'city', 'state', 'country', 'territory', 'contact_first',
                               'contact_last', 'status']
        
        # Create customer profiles
        for _, row in customer_data.iterrows():
            profile = CustomerProfile(
                name=row['customer_name'],
                phone=row['phone'],
                city=row['city'],
                state=row['state'],
                country=row['country'],
                territory=row['territory'],
                contact_name=f"{row['contact_first']} {row['contact_last']}",
                total_orders=row['total_orders'],
                total_sales=row['total_sales'],
                avg_order_value=row['avg_order_value'],
                preferred_products=[row['preferred_product']],
                deal_sizes=row['deal_sizes'],
                last_order_date=str(row['last_order_date']),
                customer_status=row['status']
            )
            self.customer_profiles[row['customer_name']] = profile
        
        logger.info(f"Created {len(self.customer_profiles)} customer profiles")
    
    def analyze_product_catalog(self) -> None:
        """Analyze product catalog and performance"""
        if self.df is None:
            return
        
        product_analysis = self.df.groupby(['PRODUCTLINE', 'PRODUCTCODE']).agg({
            'SALES': ['sum', 'mean', 'count'],
            'PRICEEACH': 'mean',
            'QUANTITYORDERED': 'sum',
            'DEALSIZE': lambda x: x.value_counts().index[0] if len(x) > 0 else 'Unknown'
        }).reset_index()
        
        # Flatten column names
        product_analysis.columns = ['product_line', 'product_code', 'total_sales', 'avg_sales',
                                  'order_count', 'avg_price', 'total_quantity', 'typical_deal_size']
        
        # Create product catalog
        for _, row in product_analysis.iterrows():
            key = f"{row['product_line']}_{row['product_code']}"
            self.product_catalog[key] = {
                'product_line': row['product_line'],
                'product_code': row['product_code'],
                'total_sales': row['total_sales'],
                'avg_sales': row['avg_sales'],
                'order_count': row['order_count'],
                'avg_price': row['avg_price'],
                'total_quantity': row['total_quantity'],
                'typical_deal_size': row['typical_deal_size'],
                'performance_score': self._calculate_product_performance(row)
            }
        
        logger.info(f"Analyzed {len(self.product_catalog)} products")
    
    def analyze_territories(self) -> None:
        """Analyze sales by territory"""
        if self.df is None:
            return
        
        territory_stats = self.df.groupby('TERRITORY').agg({
            'SALES': ['sum', 'mean', 'count'],
            'CUSTOMERNAME': 'nunique',
            'PRODUCTLINE': lambda x: x.value_counts().head(3).to_dict(),
            'DEALSIZE': lambda x: x.value_counts().to_dict()
        }).reset_index()
        
        # Flatten column names
        territory_stats.columns = ['territory', 'total_sales', 'avg_sales', 'total_orders',
                                 'unique_customers', 'top_products', 'deal_distribution']
        
        # Create territory analysis
        for _, row in territory_stats.iterrows():
            self.territory_analysis[row['territory']] = {
                'total_sales': row['total_sales'],
                'avg_sales': row['avg_sales'],
                'total_orders': row['total_orders'],
                'unique_customers': row['unique_customers'],
                'top_products': row['top_products'],
                'deal_distribution': row['deal_distribution'],
                'market_share': row['total_sales'] / self.df['SALES'].sum() * 100
            }
        
        logger.info(f"Analyzed {len(self.territory_analysis)} territories")
    
    def _calculate_product_performance(self, product_row) -> float:
        """Calculate product performance score"""
        # Normalize metrics (0-1 scale)
        max_sales = self.df['SALES'].max()
        max_orders = self.df.groupby('PRODUCTCODE')['ORDERNUMBER'].count().max()
        max_quantity = self.df.groupby('PRODUCTCODE')['QUANTITYORDERED'].sum().max()
        
        sales_score = product_row['total_sales'] / max_sales if max_sales > 0 else 0
        order_score = product_row['order_count'] / max_orders if max_orders > 0 else 0
        quantity_score = product_row['total_quantity'] / max_quantity if max_quantity > 0 else 0
        
        # Weighted average
        return (sales_score * 0.5 + order_score * 0.3 + quantity_score * 0.2)
    
    def get_top_customers(self, n: int = 10) -> List[CustomerProfile]:
        """Get top customers by sales"""
        return sorted(self.customer_profiles.values(), 
                     key=lambda x: x.total_sales, reverse=True)[:n]
    
    def get_top_products(self, n: int = 10) -> List[Dict]:
        """Get top products by performance"""
        return sorted(self.product_catalog.values(), 
                     key=lambda x: x['performance_score'], reverse=True)[:n]
    
    def get_territory_insights(self) -> Dict[str, Any]:
        """Get territory insights"""
        return {
            'total_territories': len(self.territory_analysis),
            'top_territory': max(self.territory_analysis.items(), 
                               key=lambda x: x[1]['total_sales']),
            'territory_breakdown': self.territory_analysis
        }
    
    def process_all(self) -> Dict[str, Any]:
        """Process all data and return summary"""
        if not self.load_data():
            return {'error': 'Failed to load data'}
        
        self.clean_data()
        self.create_customer_profiles()
        self.analyze_product_catalog()
        self.analyze_territories()
        
        return {
            'total_records': len(self.df),
            'total_customers': len(self.customer_profiles),
            'total_products': len(self.product_catalog),
            'total_territories': len(self.territory_analysis),
            'date_range': {
                'start': self.df['ORDERDATE'].min().strftime('%Y-%m-%d'),
                'end': self.df['ORDERDATE'].max().strftime('%Y-%m-%d')
            },
            'total_sales': self.df['SALES'].sum(),
            'avg_order_value': self.df['SALES'].mean()
        } 