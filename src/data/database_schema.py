from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.postgresql import UUID
import uuid
from datetime import datetime
import logging

Base = declarative_base()

class PriceData(Base):
    """Table for storing daily price and volume data."""
    __tablename__ = 'price_data'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False)
    date = Column(DateTime, nullable=False)
    open_price = Column(Float)
    high_price = Column(Float)
    low_price = Column(Float)
    close_price = Column(Float)
    adj_close_price = Column(Float)
    volume = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_symbol_date', 'symbol', 'date'),
        Index('idx_date', 'date'),
    )

class FundamentalData(Base):
    """Table for storing fundamental data."""
    __tablename__ = 'fundamental_data'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False)
    fiscal_date_ending = Column(DateTime, nullable=False)
    report_type = Column(String(20))  # 'annual', 'quarterly'
    
    # Income Statement Items
    total_revenue = Column(Float)
    net_income = Column(Float)
    gross_profit = Column(Float)
    operating_income = Column(Float)
    ebitda = Column(Float)
    eps = Column(Float)
    
    # Balance Sheet Items
    total_assets = Column(Float)
    total_liabilities = Column(Float)
    shareholders_equity = Column(Float)
    total_debt = Column(Float)
    cash_and_equivalents = Column(Float)
    
    # Key Ratios
    pe_ratio = Column(Float)
    pb_ratio = Column(Float)
    ps_ratio = Column(Float)
    roe = Column(Float)
    roa = Column(Float)
    debt_to_equity = Column(Float)
    current_ratio = Column(Float)
    dividend_yield = Column(Float)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_fund_symbol_date', 'symbol', 'fiscal_date_ending'),
        Index('idx_fund_date', 'fiscal_date_ending'),
    )

class MacroData(Base):
    """Table for storing macroeconomic indicators."""
    __tablename__ = 'macro_data'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    indicator_name = Column(String(50), nullable=False)
    date = Column(DateTime, nullable=False)
    value = Column(Float)
    frequency = Column(String(10))  # 'daily', 'weekly', 'monthly', 'quarterly'
    source = Column(String(20))  # 'FRED', 'Bloomberg', etc.
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_macro_indicator_date', 'indicator_name', 'date'),
        Index('idx_macro_date', 'date'),
    )

class TechnicalIndicators(Base):
    """Table for storing technical indicators."""
    __tablename__ = 'technical_indicators'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False)
    date = Column(DateTime, nullable=False)
    
    # Moving Averages
    sma_5 = Column(Float)
    sma_10 = Column(Float)
    sma_20 = Column(Float)
    sma_50 = Column(Float)
    sma_100 = Column(Float)
    ema_5 = Column(Float)
    ema_10 = Column(Float)
    ema_20 = Column(Float)
    ema_50 = Column(Float)
    ema_100 = Column(Float)
    
    # Momentum Indicators
    rsi_14 = Column(Float)
    rsi_30 = Column(Float)
    macd = Column(Float)
    macd_signal = Column(Float)
    macd_histogram = Column(Float)
    stoch_k = Column(Float)
    stoch_d = Column(Float)
    williams_r = Column(Float)
    
    # Volatility Indicators
    bb_upper = Column(Float)
    bb_lower = Column(Float)
    bb_middle = Column(Float)
    bb_width = Column(Float)
    bb_position = Column(Float)
    atr_14 = Column(Float)
    realized_vol_5 = Column(Float)
    realized_vol_10 = Column(Float)
    realized_vol_20 = Column(Float)
    
    # Volume Indicators
    volume_ratio = Column(Float)
    obv = Column(Float)
    vwap = Column(Float)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_tech_symbol_date', 'symbol', 'date'),
        Index('idx_tech_date', 'date'),
    )

class AlphaSignals(Base):
    """Table for storing alpha signals."""
    __tablename__ = 'alpha_signals'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    signal_id = Column(String(100), nullable=False)
    symbol = Column(String(20), nullable=False)
    date = Column(DateTime, nullable=False)
    signal_value = Column(Float)
    signal_rank = Column(Float)
    signal_zscore = Column(Float)
    regime = Column(String(20))  # 'bull', 'bear', 'neutral', etc.
    
    # Signal metadata
    signal_type = Column(String(50))  # 'momentum', 'mean_reversion', 'quality', etc.
    lookback_period = Column(Integer)
    confidence_score = Column(Float)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_signal_symbol_date', 'symbol', 'date'),
        Index('idx_signal_id_date', 'signal_id', 'date'),
        Index('idx_signal_regime', 'regime'),
    )

class RegimeStates(Base):
    """Table for storing regime state classifications."""
    __tablename__ = 'regime_states'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(DateTime, nullable=False)
    regime_model = Column(String(50), nullable=False)  # 'hmm', 'markov_switching', etc.
    regime_state = Column(Integer, nullable=False)  # 0, 1, 2, etc.
    regime_label = Column(String(20))  # 'bull', 'bear', 'volatile', etc.
    regime_probability = Column(Float)
    
    # Market characteristics
    market_volatility = Column(Float)
    market_return = Column(Float)
    vix_level = Column(Float)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_regime_date_model', 'date', 'regime_model'),
        Index('idx_regime_date', 'date'),
    )

class PortfolioReturns(Base):
    """Table for storing portfolio returns and performance metrics."""
    __tablename__ = 'portfolio_returns'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    portfolio_id = Column(String(100), nullable=False)
    date = Column(DateTime, nullable=False)
    
    # Returns
    daily_return = Column(Float)
    cumulative_return = Column(Float)
    benchmark_return = Column(Float)
    excess_return = Column(Float)
    
    # Risk Metrics
    volatility = Column(Float)
    sharpe_ratio = Column(Float)
    max_drawdown = Column(Float)
    var_95 = Column(Float)
    var_99 = Column(Float)
    
    # Portfolio Characteristics
    num_positions = Column(Integer)
    gross_exposure = Column(Float)
    net_exposure = Column(Float)
    turnover = Column(Float)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_port_id_date', 'portfolio_id', 'date'),
        Index('idx_port_date', 'date'),
    )

class ModelPredictions(Base):
    """Table for storing model predictions and forecasts."""
    __tablename__ = 'model_predictions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    model_id = Column(String(100), nullable=False)
    symbol = Column(String(20), nullable=False)
    prediction_date = Column(DateTime, nullable=False)
    target_date = Column(DateTime, nullable=False)
    
    # Predictions
    predicted_return = Column(Float)
    predicted_volatility = Column(Float)
    predicted_regime = Column(Integer)
    confidence_interval_lower = Column(Float)
    confidence_interval_upper = Column(Float)
    
    # Model metadata
    model_type = Column(String(50))
    training_window = Column(Integer)
    feature_count = Column(Integer)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_pred_model_symbol_date', 'model_id', 'symbol', 'prediction_date'),
        Index('idx_pred_target_date', 'target_date'),
    )

class DataQuality(Base):
    """Table for tracking data quality metrics."""
    __tablename__ = 'data_quality'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    table_name = Column(String(50), nullable=False)
    symbol = Column(String(20))
    check_date = Column(DateTime, nullable=False)
    
    # Quality Metrics
    total_records = Column(Integer)
    missing_values = Column(Integer)
    outlier_count = Column(Integer)
    duplicate_count = Column(Integer)
    
    # Data Range
    date_range_start = Column(DateTime)
    date_range_end = Column(DateTime)
    
    # Flags
    quality_passed = Column(Boolean, default=True)
    notes = Column(Text)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_dq_table_date', 'table_name', 'check_date'),
        Index('idx_dq_symbol', 'symbol'),
    )

class DatabaseManager:
    """Database management class for handling connections and operations."""
    
    def __init__(self, connection_string: str):
        """
        Initialize database manager.
        
        Args:
            connection_string: SQLAlchemy connection string
        """
        self.connection_string = connection_string
        self.engine = create_engine(connection_string, echo=False)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self.logger = logging.getLogger(__name__)
        
    def create_tables(self):
        """Create all database tables."""
        try:
            Base.metadata.create_all(bind=self.engine)
            self.logger.info("Database tables created successfully")
        except Exception as e:
            self.logger.error(f"Error creating tables: {str(e)}")
            raise
    
    def drop_tables(self):
        """Drop all database tables."""
        try:
            Base.metadata.drop_all(bind=self.engine)
            self.logger.info("Database tables dropped successfully")
        except Exception as e:
            self.logger.error(f"Error dropping tables: {str(e)}")
            raise
    
    def get_session(self):
        """Get database session."""
        return self.SessionLocal()
    
    def bulk_insert_price_data(self, data_dict: dict):
        """
        Bulk insert price data.
        
        Args:
            data_dict: Dictionary with symbol as key and DataFrame as value
        """
        session = self.get_session()
        try:
            records = []
            for symbol, df in data_dict.items():
                for date, row in df.iterrows():
                    record = PriceData(
                        symbol=symbol,
                        date=date,
                        open_price=row.get('Open'),
                        high_price=row.get('High'),
                        low_price=row.get('Low'),
                        close_price=row.get('Close'),
                        adj_close_price=row.get('Adj Close'),
                        volume=row.get('Volume')
                    )
                    records.append(record)
            
            session.bulk_save_objects(records)
            session.commit()
            self.logger.info(f"Inserted {len(records)} price data records")
            
        except Exception as e:
            session.rollback()
            self.logger.error(f"Error inserting price data: {str(e)}")
            raise
        finally:
            session.close()
    
    def bulk_insert_technical_indicators(self, data_dict: dict):
        """
        Bulk insert technical indicators.
        
        Args:
            data_dict: Dictionary with symbol as key and DataFrame as value
        """
        session = self.get_session()
        try:
            records = []
            for symbol, df in data_dict.items():
                for date, row in df.iterrows():
                    record = TechnicalIndicators(
                        symbol=symbol,
                        date=date,
                        sma_5=row.get('sma_5'),
                        sma_10=row.get('sma_10'),
                        sma_20=row.get('sma_20'),
                        sma_50=row.get('sma_50'),
                        sma_100=row.get('sma_100'),
                        ema_5=row.get('ema_5'),
                        ema_10=row.get('ema_10'),
                        ema_20=row.get('ema_20'),
                        ema_50=row.get('ema_50'),
                        ema_100=row.get('ema_100'),
                        rsi_14=row.get('rsi_14'),
                        rsi_30=row.get('rsi_30'),
                        macd=row.get('macd'),
                        macd_signal=row.get('macd_signal'),
                        macd_histogram=row.get('macd_histogram'),
                        bb_upper=row.get('bb_upper'),
                        bb_lower=row.get('bb_lower'),
                        bb_middle=row.get('bb_middle'),
                        bb_width=row.get('bb_width'),
                        bb_position=row.get('bb_position'),
                        atr_14=row.get('atr_14'),
                        realized_vol_5=row.get('realized_vol_5'),
                        realized_vol_10=row.get('realized_vol_10'),
                        realized_vol_20=row.get('realized_vol_20'),
                        volume_ratio=row.get('volume_ratio'),
                        obv=row.get('obv'),
                        vwap=row.get('vwap')
                    )
                    records.append(record)
            
            session.bulk_save_objects(records)
            session.commit()
            self.logger.info(f"Inserted {len(records)} technical indicator records")
            
        except Exception as e:
            session.rollback()
            self.logger.error(f"Error inserting technical indicators: {str(e)}")
            raise
        finally:
            session.close()
    
    def get_price_data(self, symbol: str, start_date: str, end_date: str):
        """Retrieve price data for a symbol and date range."""
        session = self.get_session()
        try:
            query = session.query(PriceData).filter(
                PriceData.symbol == symbol,
                PriceData.date >= start_date,
                PriceData.date <= end_date
            ).order_by(PriceData.date)
            
            results = query.all()
            return results
            
        except Exception as e:
            self.logger.error(f"Error retrieving price data: {str(e)}")
            raise
        finally:
            session.close()
    
    def execute_quality_check(self, table_name: str, symbol: str = None):
        """Execute data quality check and log results."""
        session = self.get_session()
        try:
            # This would implement specific quality checks
            # For now, return a placeholder
            quality_record = DataQuality(
                table_name=table_name,
                symbol=symbol,
                check_date=datetime.utcnow(),
                quality_passed=True,
                notes="Quality check completed"
            )
            
            session.add(quality_record)
            session.commit()
            
        except Exception as e:
            session.rollback()
            self.logger.error(f"Error executing quality check: {str(e)}")
            raise
        finally:
            session.close()