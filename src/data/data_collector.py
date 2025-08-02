import yfinance as yf
import pandas as pd
import numpy as np
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.fundamentaldata import FundamentalData
import fredapi
import quandl
import requests
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Union
import logging
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

load_dotenv()

class DataCollector:
    """
    Multi-source financial data collection with robust error handling and rate limiting.
    Supports Yahoo Finance, Alpha Vantage, FRED, and Quandl APIs.
    """
    
    def __init__(self, 
                 alpha_vantage_key: Optional[str] = None,
                 fred_key: Optional[str] = None,
                 quandl_key: Optional[str] = None):
        """
        Initialize data collector with API keys.
        """
        self.alpha_vantage_key = alpha_vantage_key or os.getenv('ALPHA_VANTAGE_API_KEY')
        self.fred_key = fred_key or os.getenv('FRED_API_KEY')
        self.quandl_key = quandl_key or os.getenv('QUANDL_API_KEY')
        
        # Initialize API clients
        if self.alpha_vantage_key:
            self.av_ts = TimeSeries(key=self.alpha_vantage_key, output_format='pandas')
            self.av_fd = FundamentalData(key=self.alpha_vantage_key, output_format='pandas')
        
        if self.fred_key:
            self.fred = fredapi.Fred(api_key=self.fred_key)
        
        if self.quandl_key:
            quandl.ApiConfig.api_key = self.quandl_key
        
        # Rate limiting parameters
        self.last_request_time = {}
        self.min_request_interval = {
            'yfinance': 0.1,
            'alpha_vantage': 12.0,  # 5 requests per minute
            'fred': 0.1,
            'quandl': 0.5
        }
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _rate_limit(self, source: str):
        """Implement rate limiting for API calls."""
        current_time = time.time()
        if source in self.last_request_time:
            time_since_last = current_time - self.last_request_time[source]
            min_interval = self.min_request_interval.get(source, 1.0)
            if time_since_last < min_interval:
                sleep_time = min_interval - time_since_last
                time.sleep(sleep_time)
        self.last_request_time[source] = time.time()
    
    def get_equity_data(self, 
                       symbols: List[str], 
                       start_date: str, 
                       end_date: str,
                       source: str = 'yfinance') -> Dict[str, pd.DataFrame]:
        """
        Fetch historical equity price data for multiple symbols.
        
        Args:
            symbols: List of ticker symbols
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            source: Data source ('yfinance' or 'alpha_vantage')
        
        Returns:
            Dictionary mapping symbols to DataFrames with OHLCV data
        """
        data = {}
        
        for symbol in symbols:
            try:
                self._rate_limit(source)
                
                if source == 'yfinance':
                    ticker = yf.Ticker(symbol)
                    df = ticker.history(start=start_date, end=end_date)
                    if not df.empty:
                        data[symbol] = df
                        self.logger.info(f"Successfully fetched {symbol} from Yahoo Finance")
                    else:
                        self.logger.warning(f"No data found for {symbol}")
                
                elif source == 'alpha_vantage' and self.alpha_vantage_key:
                    df, _ = self.av_ts.get_daily_adjusted(symbol=symbol, outputsize='full')
                    df.index = pd.to_datetime(df.index)
                    df = df[(df.index >= start_date) & (df.index <= end_date)]
                    if not df.empty:
                        data[symbol] = df
                        self.logger.info(f"Successfully fetched {symbol} from Alpha Vantage")
                    else:
                        self.logger.warning(f"No data found for {symbol}")
                
            except Exception as e:
                self.logger.error(f"Error fetching {symbol}: {str(e)}")
                continue
        
        return data
    
    def get_fundamental_data(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Fetch fundamental data for equity symbols.
        
        Args:
            symbols: List of ticker symbols
        
        Returns:
            Dictionary mapping symbols to fundamental data DataFrames
        """
        fundamental_data = {}
        
        for symbol in symbols:
            try:
                self._rate_limit('alpha_vantage')
                
                if self.alpha_vantage_key:
                    # Get company overview
                    overview, _ = self.av_fd.get_company_overview(symbol)
                    
                    # Get income statement
                    income_stmt, _ = self.av_fd.get_income_statement_annual(symbol)
                    
                    # Get balance sheet
                    balance_sheet, _ = self.av_fd.get_balance_sheet_annual(symbol)
                    
                    fundamental_data[symbol] = {
                        'overview': overview,
                        'income_statement': income_stmt,
                        'balance_sheet': balance_sheet
                    }
                    
                    self.logger.info(f"Successfully fetched fundamental data for {symbol}")
                
            except Exception as e:
                self.logger.error(f"Error fetching fundamental data for {symbol}: {str(e)}")
                continue
        
        return fundamental_data
    
    def get_macro_indicators(self, 
                           indicators: List[str], 
                           start_date: str, 
                           end_date: str) -> Dict[str, pd.Series]:
        """
        Fetch macroeconomic indicators from FRED.
        
        Args:
            indicators: List of FRED series IDs
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
        
        Returns:
            Dictionary mapping indicator names to time series data
        """
        macro_data = {}
        
        if not self.fred_key:
            self.logger.warning("FRED API key not provided")
            return macro_data
        
        for indicator in indicators:
            try:
                self._rate_limit('fred')
                series = self.fred.get_series(indicator, start_date, end_date)
                macro_data[indicator] = series
                self.logger.info(f"Successfully fetched {indicator} from FRED")
                
            except Exception as e:
                self.logger.error(f"Error fetching {indicator}: {str(e)}")
                continue
        
        return macro_data
    
    def get_vix_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch VIX volatility index data.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
        
        Returns:
            DataFrame with VIX data
        """
        try:
            self._rate_limit('yfinance')
            vix = yf.Ticker("^VIX")
            vix_data = vix.history(start=start_date, end=end_date)
            self.logger.info("Successfully fetched VIX data")
            return vix_data
        except Exception as e:
            self.logger.error(f"Error fetching VIX data: {str(e)}")
            return pd.DataFrame()
    
    def get_options_data(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """
        Fetch options chain data for a symbol.
        
        Args:
            symbol: Ticker symbol
        
        Returns:
            Dictionary with calls and puts DataFrames
        """
        try:
            self._rate_limit('yfinance')
            ticker = yf.Ticker(symbol)
            expirations = ticker.options
            
            if not expirations:
                return {}
            
            # Get options for the nearest expiration
            exp_date = expirations[0]
            opt_chain = ticker.option_chain(exp_date)
            
            return {
                'calls': opt_chain.calls,
                'puts': opt_chain.puts,
                'expiration': exp_date
            }
            
        except Exception as e:
            self.logger.error(f"Error fetching options data for {symbol}: {str(e)}")
            return {}
    
    def validate_data(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Validate and clean financial data.
        
        Args:
            df: Input DataFrame
            symbol: Symbol name for logging
        
        Returns:
            Cleaned DataFrame
        """
        original_length = len(df)
        
        # Remove rows with all NaN values
        df = df.dropna(how='all')
        
        # Check for negative prices (except for returns)
        price_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close']
        for col in price_columns:
            if col in df.columns:
                negative_prices = df[col] < 0
                if negative_prices.any():
                    self.logger.warning(f"Found {negative_prices.sum()} negative prices for {symbol} in {col}")
                    df.loc[negative_prices, col] = np.nan
        
        # Check for unrealistic price movements (>50% daily change)
        if 'Close' in df.columns:
            returns = df['Close'].pct_change()
            extreme_moves = abs(returns) > 0.5
            if extreme_moves.any():
                self.logger.warning(f"Found {extreme_moves.sum()} extreme price movements for {symbol}")
        
        # Log data quality summary
        final_length = len(df)
        self.logger.info(f"Data validation for {symbol}: {original_length} -> {final_length} rows")
        
        return df
    
    def save_to_database(self, 
                        data: Dict[str, pd.DataFrame], 
                        table_name: str,
                        db_connection_string: str):
        """
        Save data to database.
        
        Args:
            data: Dictionary of DataFrames to save
            table_name: Base table name
            db_connection_string: Database connection string
        """
        try:
            engine = create_engine(db_connection_string)
            
            for symbol, df in data.items():
                df_copy = df.copy()
                df_copy['symbol'] = symbol
                df_copy.to_sql(f"{table_name}_{symbol.lower()}", 
                              engine, 
                              if_exists='replace', 
                              index=True)
            
            self.logger.info(f"Successfully saved {len(data)} datasets to database")
            
        except Exception as e:
            self.logger.error(f"Error saving to database: {str(e)}")