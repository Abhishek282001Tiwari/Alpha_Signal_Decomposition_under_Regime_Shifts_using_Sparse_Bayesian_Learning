import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union
from datetime import datetime, timedelta
import logging
import time
from dataclasses import dataclass
import requests
import yfinance as yf

try:
    from fredapi import Fred
    FRED_AVAILABLE = True
except ImportError:
    FRED_AVAILABLE = False
    logging.warning("fredapi not available. Install with: pip install fredapi")

@dataclass
class MacroDataConfig:
    """Configuration for macroeconomic data collection."""
    start_date: str
    end_date: str
    frequency: str = 'M'  # Monthly, Quarterly, Annual
    fred_api_key: Optional[str] = None
    include_international: bool = True
    include_sentiment: bool = True

class MacroDataCollector:
    """
    Comprehensive macroeconomic data collector for regime analysis
    with multiple data sources and economic indicators.
    """
    
    def __init__(self, config: MacroDataConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.rate_limit_delay = 0.2
        self.collected_data = {}
        
        # Initialize FRED API if available and key provided
        self.fred = None
        if FRED_AVAILABLE and config.fred_api_key:
            try:
                self.fred = Fred(api_key=config.fred_api_key)
                self.logger.info("FRED API initialized successfully")
            except Exception as e:
                self.logger.error(f"Error initializing FRED API: {str(e)}")
    
    async def collect_data(self) -> Dict[str, pd.DataFrame]:
        """
        Collect comprehensive macroeconomic data.
        
        Returns:
            Dictionary mapping categories to their data DataFrames
        """
        self.logger.info("Collecting macroeconomic data")
        
        all_data = {}
        
        # Collect different categories of macro data
        categories = [
            ('interest_rates', self._collect_interest_rates),
            ('inflation', self._collect_inflation_data),
            ('employment', self._collect_employment_data),
            ('gdp_growth', self._collect_gdp_data),
            ('consumer_sentiment', self._collect_sentiment_data),
            ('monetary_policy', self._collect_monetary_data),
            ('international', self._collect_international_data),
            ('commodity_prices', self._collect_commodity_data),
            ('market_indicators', self._collect_market_indicators)
        ]
        
        for category, collect_func in categories:
            try:
                self.logger.info(f"Collecting {category} data")
                data = await collect_func()
                if data is not None and not data.empty:
                    all_data[category] = data
                    self.logger.info(f"Successfully collected {category} data: {len(data)} records")
                else:
                    self.logger.warning(f"No data collected for {category}")
                
                time.sleep(self.rate_limit_delay)
                
            except Exception as e:
                self.logger.error(f"Error collecting {category} data: {str(e)}")
                continue
        
        self.collected_data = all_data
        return all_data
    
    async def _collect_interest_rates(self) -> pd.DataFrame:
        """Collect interest rate data."""
        try:
            interest_data = {}
            
            if self.fred:
                # FRED interest rate series
                fred_series = {
                    'FEDFUNDS': 'Fed_Funds_Rate',
                    'DFF': 'Fed_Funds_Daily',
                    'DGS10': '10Y_Treasury',
                    'DGS2': '2Y_Treasury',
                    'DGS30': '30Y_Treasury',
                    'DGS3MO': '3M_Treasury',
                    'DAAA': 'AAA_Corporate',
                    'DBAA': 'BAA_Corporate',
                    'MORTGAGE30US': '30Y_Mortgage',
                    'TB3MS': '3M_TB_Monthly'
                }
                
                for series_id, column_name in fred_series.items():
                    try:
                        data = self.fred.get_series(
                            series_id,
                            start=self.config.start_date,
                            end=self.config.end_date
                        )
                        if not data.empty:
                            interest_data[column_name] = data
                    except Exception as e:
                        self.logger.warning(f"Error collecting {series_id}: {str(e)}")
            
            # Alternative: Use yfinance for treasury data
            treasury_symbols = {
                '^TNX': '10Y_Treasury_YF',
                '^FVX': '5Y_Treasury_YF',
                '^TYX': '30Y_Treasury_YF',
                '^IRX': '3M_Treasury_YF'
            }
            
            for symbol, name in treasury_symbols.items():
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(
                        start=self.config.start_date,
                        end=self.config.end_date,
                        interval='1d'
                    )
                    if not hist.empty:
                        interest_data[name] = hist['Close']
                except Exception as e:
                    self.logger.warning(f"Error collecting {symbol}: {str(e)}")
            
            if interest_data:
                df = pd.DataFrame(interest_data)
                
                # Calculate derived metrics
                if '10Y_Treasury' in df.columns and '2Y_Treasury' in df.columns:
                    df['Yield_Curve_10Y_2Y'] = df['10Y_Treasury'] - df['2Y_Treasury']
                
                if '30Y_Treasury' in df.columns and '10Y_Treasury' in df.columns:
                    df['Yield_Curve_30Y_10Y'] = df['30Y_Treasury'] - df['10Y_Treasury']
                
                if 'AAA_Corporate' in df.columns and '10Y_Treasury' in df.columns:
                    df['Credit_Spread_AAA'] = df['AAA_Corporate'] - df['10Y_Treasury']
                
                if 'BAA_Corporate' in df.columns and '10Y_Treasury' in df.columns:
                    df['Credit_Spread_BAA'] = df['BAA_Corporate'] - df['10Y_Treasury']
                
                return df
            
            return pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"Error in _collect_interest_rates: {str(e)}")
            return pd.DataFrame()
    
    async def _collect_inflation_data(self) -> pd.DataFrame:
        """Collect inflation and price level data."""
        try:
            inflation_data = {}
            
            if self.fred:
                fred_series = {
                    'CPIAUCSL': 'CPI_All_Urban',
                    'CPILFESL': 'CPI_Core',
                    'PCEPI': 'PCE_Price_Index',
                    'PCEPILFE': 'PCE_Core',
                    'PPIFIS': 'PPI_Final_Demand',
                    'GDPDEF': 'GDP_Deflator',
                    'UNRATE': 'Unemployment_Rate',
                    'HOUST': 'Housing_Starts'
                }
                
                for series_id, column_name in fred_series.items():
                    try:
                        data = self.fred.get_series(
                            series_id,
                            start=self.config.start_date,
                            end=self.config.end_date
                        )
                        if not data.empty:
                            inflation_data[column_name] = data
                    except Exception as e:
                        self.logger.warning(f"Error collecting {series_id}: {str(e)}")
            
            if inflation_data:
                df = pd.DataFrame(inflation_data)
                
                # Calculate inflation rates (YoY change)
                for col in ['CPI_All_Urban', 'CPI_Core', 'PCE_Price_Index', 'PCE_Core']:
                    if col in df.columns:
                        df[f'{col}_YoY'] = df[col].pct_change(12) * 100
                
                # Calculate month-over-month changes
                for col in df.columns:
                    if 'YoY' not in col:
                        df[f'{col}_MoM'] = df[col].pct_change() * 100
                
                return df
            
            return pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"Error in _collect_inflation_data: {str(e)}")
            return pd.DataFrame()
    
    async def _collect_employment_data(self) -> pd.DataFrame:
        """Collect employment and labor market data."""
        try:
            employment_data = {}
            
            if self.fred:
                fred_series = {
                    'UNRATE': 'Unemployment_Rate',
                    'CIVPART': 'Labor_Force_Participation',
                    'EMRATIO': 'Employment_Population_Ratio',
                    'NFCI': 'Financial_Conditions_Index',
                    'AWHMAN': 'Average_Weekly_Hours',
                    'AHETPI': 'Average_Hourly_Earnings',
                    'ICSA': 'Initial_Claims',
                    'CCSA': 'Continued_Claims',
                    'JOLTS': 'Job_Openings',
                    'U6RATE': 'U6_Unemployment_Rate'
                }
                
                for series_id, column_name in fred_series.items():
                    try:
                        data = self.fred.get_series(
                            series_id,
                            start=self.config.start_date,
                            end=self.config.end_date
                        )
                        if not data.empty:
                            employment_data[column_name] = data
                    except Exception as e:
                        self.logger.warning(f"Error collecting {series_id}: {str(e)}")
            
            if employment_data:
                df = pd.DataFrame(employment_data)
                
                # Calculate derived metrics
                if 'Unemployment_Rate' in df.columns:
                    df['Unemployment_Change'] = df['Unemployment_Rate'].diff()
                
                if 'Average_Hourly_Earnings' in df.columns:
                    df['Wage_Growth_YoY'] = df['Average_Hourly_Earnings'].pct_change(12) * 100
                
                return df
            
            return pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"Error in _collect_employment_data: {str(e)}")
            return pd.DataFrame()
    
    async def _collect_gdp_data(self) -> pd.DataFrame:
        """Collect GDP and economic growth data."""
        try:
            gdp_data = {}
            
            if self.fred:
                fred_series = {
                    'GDP': 'Real_GDP',
                    'GDPC1': 'Real_GDP_Per_Capita',
                    'GDPPOT': 'Potential_GDP',
                    'NYGDPMKTPCDWLD': 'World_GDP_Per_Capita',
                    'INDPRO': 'Industrial_Production',
                    'CAPUTLB50001SQ': 'Capacity_Utilization',
                    'RSAFS': 'Retail_Sales',
                    'DSPIC96': 'Real_Disposable_Income',
                    'PSAVERT': 'Personal_Saving_Rate'
                }
                
                for series_id, column_name in fred_series.items():
                    try:
                        data = self.fred.get_series(
                            series_id,
                            start=self.config.start_date,
                            end=self.config.end_date
                        )
                        if not data.empty:
                            gdp_data[column_name] = data
                    except Exception as e:
                        self.logger.warning(f"Error collecting {series_id}: {str(e)}")
            
            if gdp_data:
                df = pd.DataFrame(gdp_data)
                
                # Calculate growth rates
                for col in ['Real_GDP', 'Industrial_Production', 'Retail_Sales']:
                    if col in df.columns:
                        df[f'{col}_Growth_QoQ'] = df[col].pct_change() * 100
                        df[f'{col}_Growth_YoY'] = df[col].pct_change(4) * 100  # Quarterly data
                
                # Output gap
                if 'Real_GDP' in df.columns and 'Potential_GDP' in df.columns:
                    df['Output_Gap'] = ((df['Real_GDP'] / df['Potential_GDP']) - 1) * 100
                
                return df
            
            return pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"Error in _collect_gdp_data: {str(e)}")
            return pd.DataFrame()
    
    async def _collect_sentiment_data(self) -> pd.DataFrame:
        """Collect consumer and business sentiment data."""
        try:
            sentiment_data = {}
            
            if self.fred:
                fred_series = {
                    'UMCSENT': 'Consumer_Sentiment',
                    'USCCI': 'Consumer_Confidence',
                    'NAPM': 'ISM_PMI',
                    'NAPMNOI': 'ISM_New_Orders',
                    'NAPMEI': 'ISM_Employment',
                    'NAPMPRI': 'ISM_Prices_Paid',
                    'USSLIND': 'Leading_Economic_Index',
                    'USPHCI': 'Philadelphia_Fed_Index'
                }
                
                for series_id, column_name in fred_series.items():
                    try:
                        data = self.fred.get_series(
                            series_id,
                            start=self.config.start_date,
                            end=self.config.end_date
                        )
                        if not data.empty:
                            sentiment_data[column_name] = data
                    except Exception as e:
                        self.logger.warning(f"Error collecting {series_id}: {str(e)}")
            
            # Market-based sentiment indicators
            market_sentiment = await self._collect_market_sentiment()
            if market_sentiment:
                sentiment_data.update(market_sentiment)
            
            if sentiment_data:
                df = pd.DataFrame(sentiment_data)
                
                # Calculate sentiment changes
                for col in df.columns:
                    df[f'{col}_Change'] = df[col].diff()
                
                return df
            
            return pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"Error in _collect_sentiment_data: {str(e)}")
            return pd.DataFrame()
    
    async def _collect_market_sentiment(self) -> Dict:
        """Collect market-based sentiment indicators."""
        try:
            market_data = {}
            
            # VIX and volatility indicators
            volatility_symbols = {
                '^VIX': 'VIX',
                '^VXN': 'VXN_NASDAQ',
                '^RVX': 'RVX_Russell'
            }
            
            for symbol, name in volatility_symbols.items():
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(
                        start=self.config.start_date,
                        end=self.config.end_date,
                        interval='1d'
                    )
                    if not hist.empty:
                        market_data[name] = hist['Close']
                except Exception as e:
                    self.logger.warning(f"Error collecting {symbol}: {str(e)}")
            
            # Currency and safe haven indicators
            safe_haven_symbols = {
                'GLD': 'Gold_ETF',
                'TLT': 'Long_Term_Treasury',
                'UUP': 'US_Dollar_Index'
            }
            
            for symbol, name in safe_haven_symbols.items():
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(
                        start=self.config.start_date,
                        end=self.config.end_date,
                        interval='1d'
                    )
                    if not hist.empty:
                        market_data[name] = hist['Close']
                except Exception as e:
                    self.logger.warning(f"Error collecting {symbol}: {str(e)}")
            
            return market_data
            
        except Exception as e:
            self.logger.error(f"Error in _collect_market_sentiment: {str(e)}")
            return {}
    
    async def _collect_monetary_data(self) -> pd.DataFrame:
        """Collect monetary policy and money supply data."""
        try:
            monetary_data = {}
            
            if self.fred:
                fred_series = {
                    'M1SL': 'M1_Money_Supply',
                    'M2SL': 'M2_Money_Supply',
                    'BOGMBASE': 'Monetary_Base',
                    'WALCL': 'Fed_Balance_Sheet',
                    'TOTRESNS': 'Total_Reserves',
                    'EXCSRESNS': 'Excess_Reserves',
                    'FEDFUNDS': 'Fed_Funds_Rate',
                    'DFEDTAR': 'Fed_Target_Rate',
                    'DFEDTARU': 'Fed_Target_Upper',
                    'DFEDTARL': 'Fed_Target_Lower'
                }
                
                for series_id, column_name in fred_series.items():
                    try:
                        data = self.fred.get_series(
                            series_id,
                            start=self.config.start_date,
                            end=self.config.end_date
                        )
                        if not data.empty:
                            monetary_data[column_name] = data
                    except Exception as e:
                        self.logger.warning(f"Error collecting {series_id}: {str(e)}")
            
            if monetary_data:
                df = pd.DataFrame(monetary_data)
                
                # Calculate money supply growth
                for col in ['M1_Money_Supply', 'M2_Money_Supply']:
                    if col in df.columns:
                        df[f'{col}_YoY'] = df[col].pct_change(12) * 100
                
                return df
            
            return pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"Error in _collect_monetary_data: {str(e)}")
            return pd.DataFrame()
    
    async def _collect_international_data(self) -> pd.DataFrame:
        """Collect international economic data."""
        try:
            international_data = {}
            
            if self.fred and self.config.include_international:
                fred_series = {
                    'DEXUSEU': 'USD_EUR_Rate',
                    'DEXJPUS': 'USD_JPY_Rate',
                    'DEXCHUS': 'USD_CNY_Rate',
                    'DEXUSUK': 'USD_GBP_Rate',
                    'POILWTIUSDM': 'WTI_Oil_Price',
                    'GOLDAMGBD228NLBM': 'Gold_Price_London',
                    'DCOILWTICO': 'WTI_Oil_Spot'
                }
                
                for series_id, column_name in fred_series.items():
                    try:
                        data = self.fred.get_series(
                            series_id,
                            start=self.config.start_date,
                            end=self.config.end_date
                        )
                        if not data.empty:
                            international_data[column_name] = data
                    except Exception as e:
                        self.logger.warning(f"Error collecting {series_id}: {str(e)}")
            
            # Alternative international data via yfinance
            fx_symbols = {
                'EURUSD=X': 'EUR_USD',
                'USDJPY=X': 'USD_JPY',
                'GBPUSD=X': 'GBP_USD',
                'USDCNY=X': 'USD_CNY'
            }
            
            for symbol, name in fx_symbols.items():
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(
                        start=self.config.start_date,
                        end=self.config.end_date,
                        interval='1d'
                    )
                    if not hist.empty:
                        international_data[f'{name}_YF'] = hist['Close']
                except Exception as e:
                    self.logger.warning(f"Error collecting {symbol}: {str(e)}")
            
            if international_data:
                df = pd.DataFrame(international_data)
                return df
            
            return pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"Error in _collect_international_data: {str(e)}")
            return pd.DataFrame()
    
    async def _collect_commodity_data(self) -> pd.DataFrame:
        """Collect commodity price data."""
        try:
            commodity_data = {}
            
            # Commodity ETFs and futures via yfinance
            commodity_symbols = {
                'GLD': 'Gold_ETF',
                'SLV': 'Silver_ETF',
                'USO': 'Oil_ETF',
                'UNG': 'Natural_Gas_ETF',
                'CORN': 'Corn_ETF',
                'WEAT': 'Wheat_ETF',
                'DBA': 'Agriculture_ETF',
                'PDBC': 'Commodity_ETF',
                'GSG': 'Commodity_Index'
            }
            
            for symbol, name in commodity_symbols.items():
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(
                        start=self.config.start_date,
                        end=self.config.end_date,
                        interval='1d'
                    )
                    if not hist.empty:
                        commodity_data[name] = hist['Close']
                except Exception as e:
                    self.logger.warning(f"Error collecting {symbol}: {str(e)}")
            
            if commodity_data:
                df = pd.DataFrame(commodity_data)
                
                # Calculate commodity returns and volatility
                for col in df.columns:
                    df[f'{col}_Return'] = df[col].pct_change()
                    df[f'{col}_Volatility'] = df[col].pct_change().rolling(window=20).std() * np.sqrt(252)
                
                return df
            
            return pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"Error in _collect_commodity_data: {str(e)}")
            return pd.DataFrame()
    
    async def _collect_market_indicators(self) -> pd.DataFrame:
        """Collect market-based economic indicators."""
        try:
            market_data = {}
            
            # Market indices for economic insight
            market_symbols = {
                '^GSPC': 'SP500',
                '^DJI': 'Dow_Jones',
                '^IXIC': 'NASDAQ',
                '^RUT': 'Russell_2000',
                '^VIX': 'VIX_Volatility'
            }
            
            for symbol, name in market_symbols.items():
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(
                        start=self.config.start_date,
                        end=self.config.end_date,
                        interval='1d'
                    )
                    if not hist.empty:
                        market_data[name] = hist['Close']
                        market_data[f'{name}_Volume'] = hist['Volume']
                except Exception as e:
                    self.logger.warning(f"Error collecting {symbol}: {str(e)}")
            
            if market_data:
                df = pd.DataFrame(market_data)
                
                # Calculate market-based indicators
                if 'SP500' in df.columns:
                    df['SP500_Return'] = df['SP500'].pct_change()
                    df['SP500_Volatility'] = df['SP500_Return'].rolling(window=20).std() * np.sqrt(252)
                
                # Risk appetite indicators
                if 'Russell_2000' in df.columns and 'SP500' in df.columns:
                    df['Small_Cap_Outperformance'] = df['Russell_2000'].pct_change() - df['SP500'].pct_change()
                
                return df
            
            return pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"Error in _collect_market_indicators: {str(e)}")
            return pd.DataFrame()
    
    def create_macro_features(self) -> pd.DataFrame:
        """Create comprehensive macro features for regime analysis."""
        if not self.collected_data:
            self.logger.warning("No data collected yet. Run collect_data() first.")
            return pd.DataFrame()
        
        all_features = pd.DataFrame()
        
        for category, data in self.collected_data.items():
            if data is not None and not data.empty:
                # Resample to common frequency if needed
                if self.config.frequency == 'M':
                    data_resampled = data.resample('M').last()
                elif self.config.frequency == 'Q':
                    data_resampled = data.resample('Q').last()
                else:
                    data_resampled = data
                
                # Add category prefix to column names
                data_resampled.columns = [f'{category}_{col}' for col in data_resampled.columns]
                
                # Merge with main features DataFrame
                if all_features.empty:
                    all_features = data_resampled
                else:
                    all_features = pd.merge(all_features, data_resampled, 
                                          left_index=True, right_index=True, how='outer')
        
        # Calculate additional derived features
        all_features = self._calculate_derived_features(all_features)
        
        return all_features
    
    def _calculate_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate derived macroeconomic features."""
        try:
            # Economic cycle indicators
            if 'gdp_growth_Real_GDP' in df.columns:
                df['gdp_trend'] = df['gdp_growth_Real_GDP'].rolling(window=4).mean()
                df['gdp_cycle'] = df['gdp_growth_Real_GDP'] - df['gdp_trend']
            
            # Yield curve analysis
            yield_cols = [col for col in df.columns if 'Treasury' in col and 'YF' not in col]
            if len(yield_cols) >= 2:
                # Create yield curve slope and curvature measures
                long_term_cols = [col for col in yield_cols if any(term in col for term in ['10Y', '30Y'])]
                short_term_cols = [col for col in yield_cols if any(term in col for term in ['2Y', '3M'])]
                
                if long_term_cols and short_term_cols:
                    df['yield_curve_slope'] = df[long_term_cols[0]] - df[short_term_cols[0]]
            
            # Inflation expectations (implied from TIPS if available)
            if 'inflation_CPI_All_Urban_YoY' in df.columns:
                df['inflation_trend'] = df['inflation_CPI_All_Urban_YoY'].rolling(window=6).mean()
                df['inflation_volatility'] = df['inflation_CPI_All_Urban_YoY'].rolling(window=12).std()
            
            # Financial conditions index (simplified)
            financial_components = []
            
            if 'interest_rates_Fed_Funds_Rate' in df.columns:
                financial_components.append(df['interest_rates_Fed_Funds_Rate'].fillna(method='ffill'))
            
            if 'market_indicators_VIX_Volatility' in df.columns:
                financial_components.append(df['market_indicators_VIX_Volatility'].fillna(method='ffill'))
            
            if 'international_USD_EUR_Rate' in df.columns:
                usd_strength = df['international_USD_EUR_Rate'].pct_change(periods=12).fillna(method='ffill')
                financial_components.append(usd_strength)
            
            if len(financial_components) >= 2:
                # Normalize components and create index
                normalized_components = []
                for component in financial_components:
                    if component.std() > 0:
                        normalized = (component - component.mean()) / component.std()
                        normalized_components.append(normalized)
                
                if normalized_components:
                    df['financial_conditions_index'] = pd.concat(normalized_components, axis=1).mean(axis=1)
            
            # Economic momentum indicators
            momentum_indicators = [
                'employment_Unemployment_Rate',
                'gdp_growth_Industrial_Production',
                'sentiment_Consumer_Sentiment',
                'sentiment_ISM_PMI'
            ]
            
            available_momentum = [col for col in momentum_indicators if col in df.columns]
            if available_momentum:
                # Calculate 3-month momentum for each indicator
                for col in available_momentum:
                    df[f'{col}_momentum_3m'] = df[col].pct_change(periods=3)
                    df[f'{col}_momentum_6m'] = df[col].pct_change(periods=6)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error calculating derived features: {str(e)}")
            return df
    
    def get_regime_indicators(self) -> pd.DataFrame:
        """Extract key indicators for regime classification."""
        macro_features = self.create_macro_features()
        
        if macro_features.empty:
            return pd.DataFrame()
        
        # Select key regime indicators
        regime_indicators = [
            'interest_rates_Fed_Funds_Rate',
            'interest_rates_Yield_Curve_10Y_2Y',
            'inflation_CPI_All_Urban_YoY',
            'employment_Unemployment_Rate',
            'gdp_growth_Real_GDP_Growth_YoY',
            'sentiment_Consumer_Sentiment',
            'market_indicators_VIX_Volatility',
            'financial_conditions_index',
            'yield_curve_slope'
        ]
        
        available_indicators = [col for col in regime_indicators if col in macro_features.columns]
        
        if available_indicators:
            regime_df = macro_features[available_indicators].copy()
            
            # Add regime signals
            if 'interest_rates_Yield_Curve_10Y_2Y' in regime_df.columns:
                regime_df['recession_signal'] = (regime_df['interest_rates_Yield_Curve_10Y_2Y'] < 0).astype(int)
            
            if 'market_indicators_VIX_Volatility' in regime_df.columns:
                regime_df['high_volatility_regime'] = (regime_df['market_indicators_VIX_Volatility'] > regime_df['market_indicators_VIX_Volatility'].quantile(0.75)).astype(int)
            
            return regime_df
        
        return pd.DataFrame()
    
    def validate_data_quality(self) -> Dict[str, Dict]:
        """Validate the quality of collected macro data."""
        validation_results = {}
        
        for category, data in self.collected_data.items():
            if data is None or data.empty:
                validation_results[category] = {'status': 'FAILED', 'reason': 'No data'}
                continue
            
            issues = []
            
            # Check for missing values
            missing_pct = data.isnull().sum().sum() / (len(data) * len(data.columns))
            if missing_pct > 0.1:  # More than 10% missing
                issues.append(f"High missing data: {missing_pct:.2%}")
            
            # Check data frequency consistency
            if len(data) > 1:
                date_diff = data.index[1] - data.index[0]
                expected_freq = pd.Timedelta(days=30) if self.config.frequency == 'M' else pd.Timedelta(days=1)
                
                if abs(date_diff.total_seconds() - expected_freq.total_seconds()) > 86400 * 7:  # 1 week tolerance
                    issues.append("Inconsistent data frequency")
            
            # Check for extreme outliers
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                series = data[col].dropna()
                if len(series) > 0:
                    q99 = series.quantile(0.99)
                    q01 = series.quantile(0.01)
                    outlier_count = ((series > q99) | (series < q01)).sum()
                    
                    if outlier_count > len(series) * 0.05:  # More than 5% outliers
                        issues.append(f"High outliers in {col}: {outlier_count}")
            
            validation_results[category] = {
                'status': 'PASSED' if not issues else 'WARNING',
                'issues': issues,
                'data_points': len(data),
                'date_range': f"{data.index[0]} to {data.index[-1]}",
                'missing_pct': missing_pct,
                'columns': len(data.columns)
            }
        
        return validation_results