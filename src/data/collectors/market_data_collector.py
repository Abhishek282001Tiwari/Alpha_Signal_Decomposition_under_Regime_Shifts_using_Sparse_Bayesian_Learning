import numpy as np
import pandas as pd
import yfinance as yf
import asyncio
import aiohttp
from typing import List, Dict, Optional, Union, Tuple
from datetime import datetime, timedelta
import logging
import time
from dataclasses import dataclass
from abc import ABC, abstractmethod
import requests

@dataclass
class MarketDataConfig:
    """Configuration for market data collection."""
    symbols: List[str]
    start_date: str
    end_date: str
    interval: str = '1d'
    include_prepost: bool = False
    auto_adjust: bool = True
    back_adjust: bool = False
    proxy: Optional[str] = None
    rounding: bool = False
    show_errors: bool = True
    timeout: int = 30

class BaseDataCollector(ABC):
    """Abstract base class for data collectors."""
    
    def __init__(self, config: MarketDataConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.rate_limit_delay = 0.1  # Seconds between requests
        
    @abstractmethod
    async def collect_data(self) -> pd.DataFrame:
        """Collect data from source."""
        pass
    
    def _handle_rate_limit(self):
        """Handle rate limiting."""
        time.sleep(self.rate_limit_delay)

class MarketDataCollector(BaseDataCollector):
    """
    Comprehensive market data collector supporting multiple data sources
    and asset classes with robust error handling and rate limiting.
    """
    
    def __init__(self, config: MarketDataConfig):
        super().__init__(config)
        self.session = None
        self.collected_data = {}
        
    async def collect_data(self) -> Dict[str, pd.DataFrame]:
        """
        Collect market data for specified symbols.
        
        Returns:
            Dictionary mapping symbols to their market data DataFrames
        """
        self.logger.info(f"Collecting market data for {len(self.config.symbols)} symbols")
        
        # Collect data for all symbols
        all_data = {}
        
        # Use yfinance for primary data collection
        for symbol in self.config.symbols:
            try:
                self.logger.info(f"Collecting data for {symbol}")
                data = await self._collect_symbol_data(symbol)
                if data is not None and not data.empty:
                    all_data[symbol] = data
                    self.logger.info(f"Successfully collected {len(data)} records for {symbol}")
                else:
                    self.logger.warning(f"No data collected for {symbol}")
                
                self._handle_rate_limit()
                
            except Exception as e:
                self.logger.error(f"Error collecting data for {symbol}: {str(e)}")
                continue
        
        self.collected_data = all_data
        return all_data
    
    async def _collect_symbol_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Collect data for a single symbol."""
        try:
            # Use yfinance ticker
            ticker = yf.Ticker(symbol)
            
            # Download historical data
            data = ticker.history(
                start=self.config.start_date,
                end=self.config.end_date,
                interval=self.config.interval,
                prepost=self.config.include_prepost,
                auto_adjust=self.config.auto_adjust,
                back_adjust=self.config.back_adjust,
                proxy=self.config.proxy,
                rounding=self.config.rounding,
                show_errors=self.config.show_errors,
                timeout=self.config.timeout
            )
            
            if data.empty:
                return None
            
            # Add symbol column
            data['Symbol'] = symbol
            
            # Calculate additional metrics
            data = self._calculate_technical_indicators(data)
            
            # Ensure datetime index
            if not isinstance(data.index, pd.DatetimeIndex):
                data.index = pd.to_datetime(data.index)
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error in _collect_symbol_data for {symbol}: {str(e)}")
            return None
    
    def _calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for the market data."""
        try:
            # Returns
            data['Returns'] = data['Close'].pct_change()
            data['LogReturns'] = np.log(data['Close'] / data['Close'].shift(1))
            
            # Volatility (rolling standard deviation)
            data['Volatility_20'] = data['Returns'].rolling(window=20).std() * np.sqrt(252)
            data['Volatility_60'] = data['Returns'].rolling(window=60).std() * np.sqrt(252)
            
            # Moving averages
            data['SMA_20'] = data['Close'].rolling(window=20).mean()
            data['SMA_50'] = data['Close'].rolling(window=50).mean()
            data['SMA_200'] = data['Close'].rolling(window=200).mean()
            
            # Exponential moving averages
            data['EMA_20'] = data['Close'].ewm(span=20).mean()
            data['EMA_50'] = data['Close'].ewm(span=50).mean()
            
            # Relative Strength Index (RSI)
            data['RSI_14'] = self._calculate_rsi(data['Close'], window=14)
            
            # Bollinger Bands
            bb_period = 20
            bb_std = 2
            bb_mean = data['Close'].rolling(window=bb_period).mean()
            bb_stddev = data['Close'].rolling(window=bb_period).std()
            data['BB_Upper'] = bb_mean + (bb_stddev * bb_std)
            data['BB_Lower'] = bb_mean - (bb_stddev * bb_std)
            data['BB_Width'] = (data['BB_Upper'] - data['BB_Lower']) / bb_mean
            data['BB_Position'] = (data['Close'] - data['BB_Lower']) / (data['BB_Upper'] - data['BB_Lower'])
            
            # MACD
            ema_12 = data['Close'].ewm(span=12).mean()
            ema_26 = data['Close'].ewm(span=26).mean()
            data['MACD'] = ema_12 - ema_26
            data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
            data['MACD_Histogram'] = data['MACD'] - data['MACD_Signal']
            
            # Average True Range (ATR)
            data['ATR_14'] = self._calculate_atr(data, window=14)
            
            # Volume indicators
            if 'Volume' in data.columns:
                data['Volume_SMA_20'] = data['Volume'].rolling(window=20).mean()
                data['Volume_Ratio'] = data['Volume'] / data['Volume_SMA_20']
                
                # On Balance Volume (OBV)
                data['OBV'] = self._calculate_obv(data)
            
            # Price-based indicators
            data['High_Low_Ratio'] = data['High'] / data['Low']
            data['Close_Open_Ratio'] = data['Close'] / data['Open']
            
            # Momentum indicators
            data['Momentum_10'] = data['Close'] / data['Close'].shift(10) - 1
            data['Momentum_20'] = data['Close'] / data['Close'].shift(20) - 1
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error calculating technical indicators: {str(e)}")
            return data
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_atr(self, data: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(window=window).mean()
        return atr
    
    def _calculate_obv(self, data: pd.DataFrame) -> pd.Series:
        """Calculate On Balance Volume."""
        obv = np.zeros(len(data))
        for i in range(1, len(data)):
            if data['Close'].iloc[i] > data['Close'].iloc[i-1]:
                obv[i] = obv[i-1] + data['Volume'].iloc[i]
            elif data['Close'].iloc[i] < data['Close'].iloc[i-1]:
                obv[i] = obv[i-1] - data['Volume'].iloc[i]
            else:
                obv[i] = obv[i-1]
        return pd.Series(obv, index=data.index)
    
    def get_sector_data(self, sector_etfs: Dict[str, str]) -> Dict[str, pd.DataFrame]:
        """
        Collect sector ETF data for regime analysis.
        
        Args:
            sector_etfs: Dictionary mapping sector names to ETF symbols
        
        Returns:
            Dictionary of sector data
        """
        self.logger.info("Collecting sector ETF data")
        
        sector_config = MarketDataConfig(
            symbols=list(sector_etfs.values()),
            start_date=self.config.start_date,
            end_date=self.config.end_date,
            interval=self.config.interval
        )
        
        # Collect ETF data
        etf_data = asyncio.run(self._collect_etf_data(sector_config))
        
        # Map back to sector names
        sector_data = {}
        for sector_name, etf_symbol in sector_etfs.items():
            if etf_symbol in etf_data:
                sector_data[sector_name] = etf_data[etf_symbol]
        
        return sector_data
    
    async def _collect_etf_data(self, config: MarketDataConfig) -> Dict[str, pd.DataFrame]:
        """Collect ETF data asynchronously."""
        etf_collector = MarketDataCollector(config)
        return await etf_collector.collect_data()
    
    def get_market_indices(self) -> Dict[str, pd.DataFrame]:
        """Collect major market indices data."""
        indices = {
            'SPY': 'S&P 500',
            'QQQ': 'NASDAQ 100',
            'IWM': 'Russell 2000',
            'VTI': 'Total Stock Market',
            'EFA': 'Developed Markets',
            'EEM': 'Emerging Markets',
            'TLT': '20+ Year Treasury',
            'HYG': 'High Yield Corporate',
            'GLD': 'Gold',
            'VIX': 'Volatility Index'
        }
        
        self.logger.info("Collecting market indices data")
        
        indices_config = MarketDataConfig(
            symbols=list(indices.keys()),
            start_date=self.config.start_date,
            end_date=self.config.end_date,
            interval=self.config.interval
        )
        
        return asyncio.run(self._collect_indices_data(indices_config))
    
    async def _collect_indices_data(self, config: MarketDataConfig) -> Dict[str, pd.DataFrame]:
        """Collect indices data asynchronously."""
        indices_collector = MarketDataCollector(config)
        return await indices_collector.collect_data()
    
    def create_market_summary(self) -> pd.DataFrame:
        """Create a summary DataFrame combining all collected data."""
        if not self.collected_data:
            self.logger.warning("No data collected yet. Run collect_data() first.")
            return pd.DataFrame()
        
        summary_data = []
        
        for symbol, data in self.collected_data.items():
            if data is not None and not data.empty:
                # Latest values
                latest = data.iloc[-1]
                
                # Performance metrics
                total_return = (latest['Close'] / data.iloc[0]['Close']) - 1
                annual_return = total_return * (252 / len(data))
                volatility = data['Returns'].std() * np.sqrt(252)
                sharpe_ratio = annual_return / volatility if volatility > 0 else 0
                
                # Maximum drawdown
                cumulative = (1 + data['Returns']).cumprod()
                peak = cumulative.expanding(min_periods=1).max()
                drawdown = (cumulative / peak) - 1
                max_drawdown = drawdown.min()
                
                summary_data.append({
                    'Symbol': symbol,
                    'Start_Date': data.index[0],
                    'End_Date': data.index[-1],
                    'Total_Return': total_return,
                    'Annual_Return': annual_return,
                    'Volatility': volatility,
                    'Sharpe_Ratio': sharpe_ratio,
                    'Max_Drawdown': max_drawdown,
                    'Latest_Price': latest['Close'],
                    'Latest_Volume': latest.get('Volume', np.nan),
                    'Data_Points': len(data)
                })
        
        return pd.DataFrame(summary_data)
    
    def validate_data_quality(self) -> Dict[str, Dict]:
        """Validate the quality of collected data."""
        validation_results = {}
        
        for symbol, data in self.collected_data.items():
            if data is None or data.empty:
                validation_results[symbol] = {'status': 'FAILED', 'reason': 'No data'}
                continue
            
            issues = []
            
            # Check for missing values
            missing_pct = data.isnull().sum().sum() / (len(data) * len(data.columns))
            if missing_pct > 0.05:  # More than 5% missing
                issues.append(f"High missing data: {missing_pct:.2%}")
            
            # Check for price anomalies
            if 'Close' in data.columns:
                price_changes = data['Close'].pct_change().abs()
                extreme_changes = (price_changes > 0.2).sum()  # 20% daily change
                if extreme_changes > len(data) * 0.01:  # More than 1% of days
                    issues.append(f"Extreme price movements: {extreme_changes} days")
            
            # Check for volume anomalies
            if 'Volume' in data.columns:
                volume_zero = (data['Volume'] == 0).sum()
                if volume_zero > len(data) * 0.05:  # More than 5% zero volume
                    issues.append(f"High zero volume days: {volume_zero}")
            
            # Check data continuity
            expected_days = pd.date_range(data.index[0], data.index[-1], freq='D')
            missing_days = len(expected_days) - len(data)
            if missing_days > len(expected_days) * 0.3:  # More than 30% missing
                issues.append(f"Low data continuity: {missing_days} missing days")
            
            validation_results[symbol] = {
                'status': 'PASSED' if not issues else 'WARNING',
                'issues': issues,
                'data_points': len(data),
                'date_range': f"{data.index[0]} to {data.index[-1]}",
                'missing_pct': missing_pct
            }
        
        return validation_results

class RealTimeMarketData:
    """Real-time market data collector for live trading."""
    
    def __init__(self, symbols: List[str], update_interval: int = 60):
        self.symbols = symbols
        self.update_interval = update_interval
        self.logger = logging.getLogger(__name__)
        self.latest_data = {}
        self.is_running = False
    
    async def start_streaming(self):
        """Start real-time data streaming."""
        self.is_running = True
        self.logger.info(f"Starting real-time data streaming for {len(self.symbols)} symbols")
        
        while self.is_running:
            try:
                await self._update_data()
                await asyncio.sleep(self.update_interval)
            except Exception as e:
                self.logger.error(f"Error in real-time data update: {str(e)}")
                await asyncio.sleep(self.update_interval)
    
    async def _update_data(self):
        """Update latest market data."""
        for symbol in self.symbols:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                
                # Get latest quote
                hist = ticker.history(period="1d", interval="1m")
                if not hist.empty:
                    latest = hist.iloc[-1]
                    
                    self.latest_data[symbol] = {
                        'timestamp': hist.index[-1],
                        'price': latest['Close'],
                        'volume': latest['Volume'],
                        'high': latest['High'],
                        'low': latest['Low'],
                        'open': latest['Open']
                    }
                    
            except Exception as e:
                self.logger.error(f"Error updating {symbol}: {str(e)}")
    
    def stop_streaming(self):
        """Stop real-time data streaming."""
        self.is_running = False
        self.logger.info("Stopped real-time data streaming")
    
    def get_latest_prices(self) -> Dict[str, float]:
        """Get latest prices for all symbols."""
        return {symbol: data['price'] for symbol, data in self.latest_data.items()}