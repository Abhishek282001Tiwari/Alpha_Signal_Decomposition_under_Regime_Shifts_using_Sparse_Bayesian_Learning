import pandas as pd
import numpy as np
import sqlite3
import logging
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime, timedelta
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Float, DateTime, Text, Boolean, inspect
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects import sqlite
import json
import pickle
import os
from pathlib import Path
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')

@dataclass
class DatabaseConfig:
    """Configuration for database connection."""
    db_type: str = 'sqlite'  # 'sqlite', 'postgresql', 'mysql'
    db_path: str = 'data/alpha_signals.db'
    host: Optional[str] = None
    port: Optional[int] = None
    username: Optional[str] = None
    password: Optional[str] = None
    database: Optional[str] = None

class DatabaseManager:
    """
    Comprehensive database manager for storing and retrieving financial data,
    features, models, and results with optimized storage and querying capabilities.
    """
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.engine = None
        self.session_maker = None
        self.metadata = MetaData()
        
        # Create database directory if using SQLite
        if config.db_type == 'sqlite':
            db_dir = Path(config.db_path).parent
            db_dir.mkdir(parents=True, exist_ok=True)
        
        self._initialize_connection()
        self._create_tables()
    
    def _initialize_connection(self):
        """Initialize database connection."""
        try:
            if self.config.db_type == 'sqlite':
                connection_string = f'sqlite:///{self.config.db_path}'
            elif self.config.db_type == 'postgresql':
                connection_string = f'postgresql://{self.config.username}:{self.config.password}@{self.config.host}:{self.config.port}/{self.config.database}'
            elif self.config.db_type == 'mysql':
                connection_string = f'mysql+pymysql://{self.config.username}:{self.config.password}@{self.config.host}:{self.config.port}/{self.config.database}'
            else:
                raise ValueError(f"Unsupported database type: {self.config.db_type}")
            
            self.engine = create_engine(connection_string, echo=False)
            self.session_maker = sessionmaker(bind=self.engine)
            
            # Test connection
            with self.engine.connect() as conn:
                conn.execute("SELECT 1")
            
            self.logger.info(f"Database connection established: {self.config.db_type}")
            
        except Exception as e:
            self.logger.error(f"Error initializing database connection: {str(e)}")
            raise
    
    def _create_tables(self):
        """Create database tables if they don't exist."""
        try:
            # Market data table
            self._create_market_data_table()
            
            # Fundamental data table
            self._create_fundamental_data_table()
            
            # Macro data table
            self._create_macro_data_table()
            
            # Features table
            self._create_features_table()
            
            # Regime data table
            self._create_regime_data_table()
            
            # Models table
            self._create_models_table()
            
            # Predictions table
            self._create_predictions_table()
            
            # Performance table
            self._create_performance_table()
            
            # Metadata table
            self._create_metadata_table()
            
            # Create all tables
            self.metadata.create_all(self.engine)
            
            self.logger.info("Database tables created successfully")
            
        except Exception as e:
            self.logger.error(f"Error creating database tables: {str(e)}")
            raise
    
    def _create_market_data_table(self):
        """Create market data table."""
        Table('market_data', self.metadata,
              Column('id', Integer, primary_key=True),
              Column('symbol', String(20), nullable=False),
              Column('date', DateTime, nullable=False),
              Column('open', Float),
              Column('high', Float),
              Column('low', Float),
              Column('close', Float),
              Column('volume', Float),
              Column('adjusted_close', Float),
              Column('returns', Float),
              Column('log_returns', Float),
              Column('created_at', DateTime, default=datetime.utcnow),
              Column('updated_at', DateTime, default=datetime.utcnow)
        )
    
    def _create_fundamental_data_table(self):
        """Create fundamental data table."""
        Table('fundamental_data', self.metadata,
              Column('id', Integer, primary_key=True),
              Column('symbol', String(20), nullable=False),
              Column('date', DateTime, nullable=False),
              Column('metric_name', String(100), nullable=False),
              Column('metric_value', Float),
              Column('period_type', String(20)),  # annual, quarterly
              Column('sector', String(50)),
              Column('industry', String(100)),
              Column('data_source', String(50)),
              Column('created_at', DateTime, default=datetime.utcnow)
        )
    
    def _create_macro_data_table(self):
        """Create macroeconomic data table."""
        Table('macro_data', self.metadata,
              Column('id', Integer, primary_key=True),
              Column('date', DateTime, nullable=False),
              Column('indicator_name', String(100), nullable=False),
              Column('indicator_value', Float),
              Column('frequency', String(20)),  # daily, monthly, quarterly
              Column('category', String(50)),  # interest_rates, inflation, etc.
              Column('data_source', String(50)),
              Column('created_at', DateTime, default=datetime.utcnow)
        )
    
    def _create_features_table(self):
        """Create features table."""
        Table('features', self.metadata,
              Column('id', Integer, primary_key=True),
              Column('symbol', String(20), nullable=False),
              Column('date', DateTime, nullable=False),
              Column('feature_name', String(100), nullable=False),
              Column('feature_value', Float),
              Column('feature_category', String(50)),  # technical, fundamental, macro
              Column('normalization_method', String(20)),
              Column('created_at', DateTime, default=datetime.utcnow)
        )
    
    def _create_regime_data_table(self):
        """Create regime data table."""
        Table('regime_data', self.metadata,
              Column('id', Integer, primary_key=True),
              Column('date', DateTime, nullable=False),
              Column('regime_id', Integer, nullable=False),
              Column('regime_probability', Float, nullable=False),
              Column('regime_name', String(50)),
              Column('model_version', String(50)),
              Column('confidence_score', Float),
              Column('created_at', DateTime, default=datetime.utcnow)
        )
    
    def _create_models_table(self):
        """Create models table."""
        Table('models', self.metadata,
              Column('id', Integer, primary_key=True),
              Column('model_name', String(100), nullable=False),
              Column('model_type', String(50), nullable=False),
              Column('model_version', String(50), nullable=False),
              Column('model_parameters', Text),  # JSON string
              Column('model_binary', Text),  # Serialized model
              Column('training_start_date', DateTime),
              Column('training_end_date', DateTime),
              Column('feature_count', Integer),
              Column('performance_metrics', Text),  # JSON string
              Column('is_active', Boolean, default=True),
              Column('created_at', DateTime, default=datetime.utcnow),
              Column('updated_at', DateTime, default=datetime.utcnow)
        )
    
    def _create_predictions_table(self):
        """Create predictions table."""
        Table('predictions', self.metadata,
              Column('id', Integer, primary_key=True),
              Column('model_id', Integer, nullable=False),
              Column('symbol', String(20), nullable=False),
              Column('prediction_date', DateTime, nullable=False),
              Column('target_date', DateTime, nullable=False),
              Column('prediction_value', Float, nullable=False),
              Column('prediction_uncertainty', Float),
              Column('regime_context', String(50)),
              Column('confidence_score', Float),
              Column('actual_value', Float),
              Column('created_at', DateTime, default=datetime.utcnow)
        )
    
    def _create_performance_table(self):
        """Create performance tracking table."""
        Table('performance', self.metadata,
              Column('id', Integer, primary_key=True),
              Column('model_id', Integer, nullable=False),
              Column('evaluation_date', DateTime, nullable=False),
              Column('evaluation_period_start', DateTime),
              Column('evaluation_period_end', DateTime),
              Column('metric_name', String(50), nullable=False),
              Column('metric_value', Float, nullable=False),
              Column('regime_context', String(50)),
              Column('symbol_universe', Text),  # JSON string of symbols
              Column('created_at', DateTime, default=datetime.utcnow)
        )
    
    def _create_metadata_table(self):
        """Create metadata table for data lineage and versioning."""
        Table('metadata', self.metadata,
              Column('id', Integer, primary_key=True),
              Column('table_name', String(50), nullable=False),
              Column('data_source', String(100)),
              Column('last_updated', DateTime),
              Column('record_count', Integer),
              Column('data_quality_score', Float),
              Column('schema_version', String(20)),
              Column('notes', Text),
              Column('created_at', DateTime, default=datetime.utcnow)
        )
    
    # Data insertion methods
    def store_market_data(self, data: Dict[str, pd.DataFrame], batch_size: int = 1000):
        """Store market data for multiple symbols."""
        try:
            total_records = 0
            
            for symbol, df in data.items():
                if df.empty:
                    continue
                
                self.logger.info(f"Storing market data for {symbol}: {len(df)} records")
                
                # Prepare data for insertion
                df_copy = df.copy()
                df_copy['symbol'] = symbol
                df_copy['date'] = df_copy.index
                
                # Calculate returns if not present
                if 'returns' not in df_copy.columns and 'Close' in df_copy.columns:
                    df_copy['returns'] = df_copy['Close'].pct_change()
                if 'log_returns' not in df_copy.columns and 'Close' in df_copy.columns:
                    df_copy['log_returns'] = np.log(df_copy['Close'] / df_copy['Close'].shift(1))
                
                # Rename columns to match database schema
                column_mapping = {
                    'Open': 'open',
                    'High': 'high', 
                    'Low': 'low',
                    'Close': 'close',
                    'Volume': 'volume',
                    'Adj Close': 'adjusted_close'
                }
                df_copy = df_copy.rename(columns=column_mapping)
                
                # Select only relevant columns
                db_columns = ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume', 
                             'adjusted_close', 'returns', 'log_returns']
                df_copy = df_copy[[col for col in db_columns if col in df_copy.columns]]
                
                # Store in batches
                for i in range(0, len(df_copy), batch_size):
                    batch = df_copy.iloc[i:i+batch_size]
                    batch.to_sql('market_data', self.engine, if_exists='append', index=False)
                
                total_records += len(df_copy)
            
            self._update_metadata('market_data', total_records)
            self.logger.info(f"Stored {total_records} market data records")
            
        except Exception as e:
            self.logger.error(f"Error storing market data: {str(e)}")
            raise
    
    def store_fundamental_data(self, data: Dict[str, Dict], batch_size: int = 1000):
        """Store fundamental data."""
        try:
            records = []
            
            for symbol, symbol_data in data.items():
                company_info = symbol_data.get('company_info', {})
                sector = company_info.get('sector', 'Unknown')
                industry = company_info.get('industry', 'Unknown')
                
                # Store financial ratios
                ratios = symbol_data.get('financial_ratios', {})
                for metric_name, value in ratios.items():
                    if pd.notna(value):
                        records.append({
                            'symbol': symbol,
                            'date': datetime.now().date(),
                            'metric_name': metric_name,
                            'metric_value': float(value),
                            'period_type': 'annual',
                            'sector': sector,
                            'industry': industry,
                            'data_source': 'yfinance'
                        })
                
                # Store growth metrics
                growth = symbol_data.get('growth_metrics', {})
                for metric_name, value in growth.items():
                    if pd.notna(value):
                        records.append({
                            'symbol': symbol,
                            'date': datetime.now().date(),
                            'metric_name': metric_name,
                            'metric_value': float(value),
                            'period_type': 'annual',
                            'sector': sector,
                            'industry': industry,
                            'data_source': 'yfinance'
                        })
            
            if records:
                df = pd.DataFrame(records)
                
                # Store in batches
                for i in range(0, len(df), batch_size):
                    batch = df.iloc[i:i+batch_size]
                    batch.to_sql('fundamental_data', self.engine, if_exists='append', index=False)
                
                self._update_metadata('fundamental_data', len(records))
                self.logger.info(f"Stored {len(records)} fundamental data records")
            
        except Exception as e:
            self.logger.error(f"Error storing fundamental data: {str(e)}")
            raise
    
    def store_macro_data(self, data: Dict[str, pd.DataFrame], batch_size: int = 1000):
        """Store macroeconomic data."""
        try:
            total_records = 0
            
            for category, df in data.items():
                if df.empty:
                    continue
                
                self.logger.info(f"Storing macro data for {category}: {len(df)} records")
                
                records = []
                for date, row in df.iterrows():
                    for indicator_name, value in row.items():
                        if pd.notna(value):
                            records.append({
                                'date': date,
                                'indicator_name': indicator_name,
                                'indicator_value': float(value),
                                'frequency': 'daily',
                                'category': category,
                                'data_source': 'fred_yfinance'
                            })
                
                if records:
                    records_df = pd.DataFrame(records)
                    
                    # Store in batches
                    for i in range(0, len(records_df), batch_size):
                        batch = records_df.iloc[i:i+batch_size]
                        batch.to_sql('macro_data', self.engine, if_exists='append', index=False)
                    
                    total_records += len(records)
            
            self._update_metadata('macro_data', total_records)
            self.logger.info(f"Stored {total_records} macro data records")
            
        except Exception as e:
            self.logger.error(f"Error storing macro data: {str(e)}")
            raise
    
    def store_features(self, features: pd.DataFrame, batch_size: int = 1000):
        """Store engineered features."""
        try:
            if 'Symbol' not in features.columns:
                self.logger.error("Features DataFrame must contain 'Symbol' column")
                return
            
            records = []
            feature_columns = [col for col in features.columns if col not in ['Symbol', 'Sector']]
            
            for idx, row in features.iterrows():
                symbol = row['Symbol']
                date = idx if isinstance(idx, datetime) else datetime.now().date()
                
                for feature_name in feature_columns:
                    value = row[feature_name]
                    if pd.notna(value):
                        # Determine feature category
                        category = 'technical'
                        if 'fundamental' in feature_name:
                            category = 'fundamental'
                        elif 'macro' in feature_name:
                            category = 'macro'
                        elif 'cs_' in feature_name:
                            category = 'cross_sectional'
                        
                        records.append({
                            'symbol': symbol,
                            'date': date,
                            'feature_name': feature_name,
                            'feature_value': float(value),
                            'feature_category': category,
                            'normalization_method': 'standard'
                        })
            
            if records:
                df = pd.DataFrame(records)
                
                # Store in batches
                for i in range(0, len(df), batch_size):
                    batch = df.iloc[i:i+batch_size]
                    batch.to_sql('features', self.engine, if_exists='append', index=False)
                
                self._update_metadata('features', len(records))
                self.logger.info(f"Stored {len(records)} feature records")
            
        except Exception as e:
            self.logger.error(f"Error storing features: {str(e)}")
            raise
    
    def store_regime_data(self, regime_probs: pd.DataFrame, model_version: str = 'v1.0'):
        """Store regime probability data."""
        try:
            records = []
            
            for date, row in regime_probs.iterrows():
                for regime_id, probability in enumerate(row):
                    if pd.notna(probability):
                        records.append({
                            'date': date,
                            'regime_id': regime_id,
                            'regime_probability': float(probability),
                            'regime_name': f'regime_{regime_id}',
                            'model_version': model_version,
                            'confidence_score': float(max(row))  # Highest probability as confidence
                        })
            
            if records:
                df = pd.DataFrame(records)
                df.to_sql('regime_data', self.engine, if_exists='append', index=False)
                
                self._update_metadata('regime_data', len(records))
                self.logger.info(f"Stored {len(records)} regime data records")
            
        except Exception as e:
            self.logger.error(f"Error storing regime data: {str(e)}")
            raise
    
    def store_model(self, model, model_name: str, model_type: str, model_version: str,
                   parameters: Dict, performance_metrics: Dict) -> int:
        """Store a trained model."""
        try:
            # Serialize model
            model_binary = pickle.dumps(model)
            
            # Prepare record
            record = {
                'model_name': model_name,
                'model_type': model_type,
                'model_version': model_version,
                'model_parameters': json.dumps(parameters),
                'model_binary': model_binary.hex(),  # Store as hex string
                'training_start_date': parameters.get('training_start'),
                'training_end_date': parameters.get('training_end'),
                'feature_count': parameters.get('n_features'),
                'performance_metrics': json.dumps(performance_metrics),
                'is_active': True
            }
            
            df = pd.DataFrame([record])
            df.to_sql('models', self.engine, if_exists='append', index=False)
            
            # Get the model ID
            with self.engine.connect() as conn:
                result = conn.execute(
                    "SELECT id FROM models WHERE model_name = ? AND model_version = ? ORDER BY created_at DESC LIMIT 1",
                    (model_name, model_version)
                )
                model_id = result.fetchone()[0]
            
            self.logger.info(f"Stored model {model_name} v{model_version} with ID {model_id}")
            return model_id
            
        except Exception as e:
            self.logger.error(f"Error storing model: {str(e)}")
            raise
    
    def store_predictions(self, predictions: pd.DataFrame, model_id: int):
        """Store model predictions."""
        try:
            predictions_copy = predictions.copy()
            predictions_copy['model_id'] = model_id
            
            # Ensure required columns exist
            required_columns = ['model_id', 'symbol', 'prediction_date', 'target_date', 'prediction_value']
            missing_columns = [col for col in required_columns if col not in predictions_copy.columns]
            
            if missing_columns:
                self.logger.error(f"Missing required columns for predictions: {missing_columns}")
                return
            
            predictions_copy.to_sql('predictions', self.engine, if_exists='append', index=False)
            
            self.logger.info(f"Stored {len(predictions)} prediction records for model {model_id}")
            
        except Exception as e:
            self.logger.error(f"Error storing predictions: {str(e)}")
            raise
    
    # Data retrieval methods
    def get_market_data(self, 
                       symbols: Optional[List[str]] = None,
                       start_date: Optional[datetime] = None,
                       end_date: Optional[datetime] = None) -> Dict[str, pd.DataFrame]:
        """Retrieve market data."""
        try:
            query = "SELECT * FROM market_data WHERE 1=1"
            params = []
            
            if symbols:
                placeholders = ','.join(['?' for _ in symbols])
                query += f" AND symbol IN ({placeholders})"
                params.extend(symbols)
            
            if start_date:
                query += " AND date >= ?"
                params.append(start_date)
            
            if end_date:
                query += " AND date <= ?"
                params.append(end_date)
            
            query += " ORDER BY symbol, date"
            
            df = pd.read_sql_query(query, self.engine, params=params, parse_dates=['date'])
            
            # Group by symbol
            result = {}
            for symbol in df['symbol'].unique():
                symbol_data = df[df['symbol'] == symbol].copy()
                symbol_data.set_index('date', inplace=True)
                symbol_data.drop(['id', 'symbol', 'created_at', 'updated_at'], axis=1, inplace=True, errors='ignore')
                result[symbol] = symbol_data
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error retrieving market data: {str(e)}")
            return {}
    
    def get_features(self, 
                    symbols: Optional[List[str]] = None,
                    start_date: Optional[datetime] = None,
                    end_date: Optional[datetime] = None,
                    feature_categories: Optional[List[str]] = None) -> pd.DataFrame:
        """Retrieve features data."""
        try:
            query = "SELECT * FROM features WHERE 1=1"
            params = []
            
            if symbols:
                placeholders = ','.join(['?' for _ in symbols])
                query += f" AND symbol IN ({placeholders})"
                params.extend(symbols)
            
            if start_date:
                query += " AND date >= ?"
                params.append(start_date)
            
            if end_date:
                query += " AND date <= ?"
                params.append(end_date)
                
            if feature_categories:
                placeholders = ','.join(['?' for _ in feature_categories])
                query += f" AND feature_category IN ({placeholders})"
                params.extend(feature_categories)
            
            df = pd.read_sql_query(query, self.engine, params=params, parse_dates=['date'])
            
            # Pivot to wide format
            if not df.empty:
                pivot_df = df.pivot_table(
                    index=['symbol', 'date'], 
                    columns='feature_name', 
                    values='feature_value',
                    aggfunc='first'
                ).reset_index()
                
                return pivot_df
            
            return pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"Error retrieving features: {str(e)}")
            return pd.DataFrame()
    
    def get_regime_data(self, 
                       start_date: Optional[datetime] = None,
                       end_date: Optional[datetime] = None,
                       model_version: Optional[str] = None) -> pd.DataFrame:
        """Retrieve regime probability data."""
        try:
            query = "SELECT * FROM regime_data WHERE 1=1"
            params = []
            
            if start_date:
                query += " AND date >= ?"
                params.append(start_date)
            
            if end_date:
                query += " AND date <= ?"
                params.append(end_date)
                
            if model_version:
                query += " AND model_version = ?"
                params.append(model_version)
            
            query += " ORDER BY date, regime_id"
            
            df = pd.read_sql_query(query, self.engine, params=params, parse_dates=['date'])
            
            # Pivot to wide format
            if not df.empty:
                pivot_df = df.pivot_table(
                    index='date', 
                    columns='regime_id', 
                    values='regime_probability',
                    aggfunc='first'
                )
                
                return pivot_df
            
            return pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"Error retrieving regime data: {str(e)}")
            return pd.DataFrame()
    
    def get_model(self, model_id: int):
        """Retrieve a stored model."""
        try:
            query = "SELECT * FROM models WHERE id = ?"
            df = pd.read_sql_query(query, self.engine, params=[model_id])
            
            if df.empty:
                return None
            
            record = df.iloc[0]
            
            # Deserialize model
            model_binary = bytes.fromhex(record['model_binary'])
            model = pickle.loads(model_binary)
            
            # Parse parameters and metrics
            parameters = json.loads(record['model_parameters'])
            performance_metrics = json.loads(record['performance_metrics'])
            
            return {
                'model': model,
                'model_name': record['model_name'],
                'model_type': record['model_type'],
                'model_version': record['model_version'],
                'parameters': parameters,
                'performance_metrics': performance_metrics,
                'created_at': record['created_at']
            }
            
        except Exception as e:
            self.logger.error(f"Error retrieving model {model_id}: {str(e)}")
            return None
    
    def _update_metadata(self, table_name: str, record_count: int):
        """Update metadata for a table."""
        try:
            metadata_record = {
                'table_name': table_name,
                'last_updated': datetime.utcnow(),
                'record_count': record_count,
                'data_source': 'alpha_signal_system'
            }
            
            df = pd.DataFrame([metadata_record])
            df.to_sql('metadata', self.engine, if_exists='append', index=False)
            
        except Exception as e:
            self.logger.warning(f"Error updating metadata for {table_name}: {str(e)}")
    
    def get_database_summary(self) -> Dict[str, Any]:
        """Get a summary of the database contents."""
        try:
            summary = {}
            
            # Get table information
            inspector = inspect(self.engine)
            tables = inspector.get_table_names()
            
            for table in tables:
                with self.engine.connect() as conn:
                    # Count records
                    count_result = conn.execute(f"SELECT COUNT(*) FROM {table}")
                    record_count = count_result.fetchone()[0]
                    
                    summary[table] = {
                        'record_count': record_count,
                        'columns': [col['name'] for col in inspector.get_columns(table)]
                    }
                    
                    # Get date range for time-based tables
                    if table in ['market_data', 'features', 'regime_data', 'macro_data']:
                        try:
                            date_result = conn.execute(f"SELECT MIN(date), MAX(date) FROM {table}")
                            min_date, max_date = date_result.fetchone()
                            summary[table]['date_range'] = {
                                'start': min_date,
                                'end': max_date
                            }
                        except:
                            pass
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error getting database summary: {str(e)}")
            return {}
    
    def cleanup_old_data(self, days_to_keep: int = 365):
        """Clean up old data to manage database size."""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            tables_to_clean = ['predictions', 'performance']
            
            for table in tables_to_clean:
                with self.engine.connect() as conn:
                    result = conn.execute(
                        f"DELETE FROM {table} WHERE created_at < ?",
                        (cutoff_date,)
                    )
                    self.logger.info(f"Cleaned {result.rowcount} old records from {table}")
            
        except Exception as e:
            self.logger.error(f"Error cleaning up old data: {str(e)}")
    
    def backup_database(self, backup_path: str):
        """Create a backup of the database."""
        try:
            if self.config.db_type == 'sqlite':
                import shutil
                shutil.copy2(self.config.db_path, backup_path)
                self.logger.info(f"Database backed up to {backup_path}")
            else:
                self.logger.warning("Backup not implemented for non-SQLite databases")
                
        except Exception as e:
            self.logger.error(f"Error backing up database: {str(e)}")
    
    def close(self):
        """Close database connections."""
        try:
            if self.engine:
                self.engine.dispose()
            self.logger.info("Database connections closed")
            
        except Exception as e:
            self.logger.error(f"Error closing database: {str(e)}")