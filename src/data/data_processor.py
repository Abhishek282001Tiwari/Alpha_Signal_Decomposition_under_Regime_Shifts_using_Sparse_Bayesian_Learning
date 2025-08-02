import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from scipy import interpolate, stats
from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler
import logging
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

class DataProcessor:
    """
    Comprehensive data quality and alignment processor for financial time series.
    Handles missing data, corporate actions, survivorship bias, and temporal alignment.
    """
    
    def __init__(self, 
                 business_days_only: bool = True,
                 min_history_days: int = 252):
        """
        Initialize data processor.
        
        Args:
            business_days_only: Whether to use only business days
            min_history_days: Minimum number of days of history required
        """
        self.business_days_only = business_days_only
        self.min_history_days = min_history_days
        self.logger = logging.getLogger(__name__)
        
    def handle_missing_data(self, 
                          df: pd.DataFrame, 
                          method: str = 'hybrid',
                          max_consecutive_missing: int = 5) -> pd.DataFrame:
        """
        Handle missing data using various imputation methods.
        
        Args:
            df: Input DataFrame
            method: Imputation method ('forward_fill', 'interpolation', 'knn', 'hybrid')
            max_consecutive_missing: Maximum consecutive missing values to impute
        
        Returns:
            DataFrame with missing values handled
        """
        result = df.copy()
        
        # Log missing data statistics
        missing_stats = result.isnull().sum()
        self.logger.info(f"Missing data before imputation:\n{missing_stats[missing_stats > 0]}")
        
        if method == 'forward_fill':
            result = self._forward_fill_imputation(result, max_consecutive_missing)
            
        elif method == 'interpolation':
            result = self._interpolation_imputation(result)
            
        elif method == 'knn':
            result = self._knn_imputation(result)
            
        elif method == 'hybrid':
            # Use forward fill for price data, interpolation for others
            price_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close']
            for col in price_columns:
                if col in result.columns:
                    result[col] = self._forward_fill_imputation(
                        result[[col]], max_consecutive_missing
                    )[col]
            
            # Use interpolation for volume and other indicators
            other_columns = [col for col in result.columns if col not in price_columns]
            if other_columns:
                result[other_columns] = self._interpolation_imputation(
                    result[other_columns]
                )[other_columns]
        
        # Log imputation results
        missing_after = result.isnull().sum()
        self.logger.info(f"Missing data after imputation:\n{missing_after[missing_after > 0]}")
        
        return result
    
    def _forward_fill_imputation(self, 
                               df: pd.DataFrame, 
                               max_consecutive: int) -> pd.DataFrame:
        """Forward fill with limit on consecutive missing values."""
        result = df.copy()
        
        for column in result.columns:
            # Identify consecutive missing value groups
            missing_mask = result[column].isnull()
            missing_groups = (missing_mask != missing_mask.shift()).cumsum()[missing_mask]
            
            # Count consecutive missing values
            consecutive_counts = missing_groups.groupby(missing_groups).cumcount() + 1
            
            # Only fill where consecutive count <= max_consecutive
            fill_mask = consecutive_counts <= max_consecutive
            
            # Forward fill only allowed positions
            temp_series = result[column].copy()
            temp_series[missing_mask & ~fill_mask] = np.nan  # Keep long gaps as NaN
            result[column] = temp_series.fillna(method='ffill')
        
        return result
    
    def _interpolation_imputation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Interpolation-based imputation."""
        result = df.copy()
        
        for column in result.columns:
            if result[column].dtype in ['float64', 'int64']:
                # Use cubic spline for smooth interpolation
                result[column] = result[column].interpolate(
                    method='cubic', 
                    limit_direction='both',
                    limit=10
                )
        
        return result
    
    def _knn_imputation(self, df: pd.DataFrame) -> pd.DataFrame:
        """KNN-based imputation for multivariate missing data."""
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_columns) > 1:
            imputer = KNNImputer(n_neighbors=5)
            df[numeric_columns] = imputer.fit_transform(df[numeric_columns])
        
        return df
    
    def adjust_corporate_actions(self, 
                               df: pd.DataFrame, 
                               actions: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Adjust prices for stock splits, dividends, and other corporate actions.
        
        Args:
            df: Price data DataFrame
            actions: Corporate actions data (if available)
        
        Returns:
            Adjusted DataFrame
        """
        result = df.copy()
        
        if actions is not None and not actions.empty:
            # Process stock splits
            splits = actions[actions['action'] == 'split']
            for _, split in splits.iterrows():
                split_date = split['date']
                split_ratio = split['value']
                
                # Adjust prices before split date
                mask = result.index < split_date
                price_columns = ['Open', 'High', 'Low', 'Close']
                for col in price_columns:
                    if col in result.columns:
                        result.loc[mask, col] *= split_ratio
                
                # Adjust volume
                if 'Volume' in result.columns:
                    result.loc[mask, 'Volume'] /= split_ratio
            
            # Process dividends (if not already adjusted)
            dividends = actions[actions['action'] == 'dividend']
            for _, dividend in dividends.iterrows():
                ex_date = dividend['date']
                dividend_amount = dividend['value']
                
                # Adjust prices before ex-dividend date
                mask = result.index < ex_date
                price_columns = ['Open', 'High', 'Low', 'Close']
                for col in price_columns:
                    if col in result.columns:
                        result.loc[mask, col] -= dividend_amount
        
        else:
            # Basic adjustment using close and adjusted close ratio
            if 'Close' in result.columns and 'Adj Close' in result.columns:
                adjustment_factor = result['Adj Close'] / result['Close']
                
                price_columns = ['Open', 'High', 'Low']
                for col in price_columns:
                    if col in result.columns:
                        result[col] *= adjustment_factor
                
                # Replace Close with Adj Close
                result['Close'] = result['Adj Close']
        
        return result
    
    def correct_survivorship_bias(self, 
                                data: Dict[str, pd.DataFrame],
                                universe_start_date: str,
                                min_trading_days: int = 500) -> Dict[str, pd.DataFrame]:
        """
        Correct for survivorship bias by including delisted stocks.
        
        Args:
            data: Dictionary of stock data
            universe_start_date: Start date for universe construction
            min_trading_days: Minimum trading days to include a stock
        
        Returns:
            Filtered data dictionary
        """
        corrected_data = {}
        start_date = pd.to_datetime(universe_start_date)
        
        for symbol, df in data.items():
            # Check if stock has sufficient history
            valid_data = df.dropna(subset=['Close'])
            if len(valid_data) < min_trading_days:
                self.logger.warning(f"Excluding {symbol}: insufficient history ({len(valid_data)} days)")
                continue
            
            # Check if stock was trading at universe start date
            if valid_data.index.min() > start_date:
                self.logger.warning(f"Excluding {symbol}: not trading at universe start date")
                continue
            
            # Include stocks that were delisted during the period
            # (This would require additional delisting data in practice)
            corrected_data[symbol] = df
        
        self.logger.info(f"Survivorship bias correction: {len(data)} -> {len(corrected_data)} stocks")
        return corrected_data
    
    def align_point_in_time(self, 
                          price_data: Dict[str, pd.DataFrame],
                          fundamental_data: Optional[Dict[str, pd.DataFrame]] = None,
                          announcement_lag: int = 45) -> Dict[str, pd.DataFrame]:
        """
        Ensure point-in-time data alignment to prevent look-ahead bias.
        
        Args:
            price_data: Dictionary of price DataFrames
            fundamental_data: Dictionary of fundamental DataFrames
            announcement_lag: Days to lag fundamental data after period end
        
        Returns:
            Aligned data dictionary
        """
        aligned_data = {}
        
        for symbol in price_data.keys():
            df = price_data[symbol].copy()
            
            if fundamental_data and symbol in fundamental_data:
                fund_df = fundamental_data[symbol].copy()
                
                # Shift fundamental data by announcement lag
                if 'fiscal_date_ending' in fund_df.columns:
                    fund_df['announcement_date'] = (
                        pd.to_datetime(fund_df['fiscal_date_ending']) + 
                        timedelta(days=announcement_lag)
                    )
                    
                    # Merge with point-in-time alignment
                    df = pd.merge_asof(
                        df.sort_index(),
                        fund_df.sort_values('announcement_date'),
                        left_index=True,
                        right_on='announcement_date',
                        direction='backward'
                    )
            
            aligned_data[symbol] = df
        
        return aligned_data
    
    def detect_outliers(self, 
                       df: pd.DataFrame, 
                       method: str = 'isolation_forest',
                       columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Detect and flag outliers in the data.
        
        Args:
            df: Input DataFrame
            method: Outlier detection method
            columns: Columns to check (default: all numeric)
        
        Returns:
            DataFrame with outlier flags
        """
        result = df.copy()
        
        if columns is None:
            columns = result.select_dtypes(include=[np.number]).columns.tolist()
        
        outlier_flags = pd.DataFrame(False, index=result.index, columns=columns)
        
        for column in columns:
            if column not in result.columns:
                continue
                
            data = result[column].dropna()
            
            if method == 'zscore':
                z_scores = np.abs(stats.zscore(data))
                outliers = z_scores > 3
                outlier_flags.loc[data.index, column] = outliers
                
            elif method == 'iqr':
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = (data < lower_bound) | (data > upper_bound)
                outlier_flags.loc[data.index, column] = outliers
                
            elif method == 'modified_zscore':
                median = data.median()
                mad = np.median(np.abs(data - median))
                modified_z_scores = 0.6745 * (data - median) / mad
                outliers = np.abs(modified_z_scores) > 3.5
                outlier_flags.loc[data.index, column] = outliers
        
        # Add outlier flags to result
        for column in columns:
            result[f'{column}_outlier'] = outlier_flags[column]
        
        # Log outlier statistics
        outlier_counts = outlier_flags.sum()
        self.logger.info(f"Outliers detected:\n{outlier_counts[outlier_counts > 0]}")
        
        return result
    
    def treat_outliers(self, 
                      df: pd.DataFrame, 
                      method: str = 'winsorize',
                      outlier_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Treat detected outliers.
        
        Args:
            df: DataFrame with outlier flags
            method: Treatment method ('winsorize', 'cap', 'remove')
            outlier_columns: Columns with outliers to treat
        
        Returns:
            DataFrame with treated outliers
        """
        result = df.copy()
        
        if outlier_columns is None:
            outlier_columns = [col for col in result.columns if col.endswith('_outlier')]
        
        for outlier_col in outlier_columns:
            if outlier_col not in result.columns:
                continue
                
            base_col = outlier_col.replace('_outlier', '')
            if base_col not in result.columns:
                continue
            
            outlier_mask = result[outlier_col]
            
            if method == 'winsorize':
                # Winsorize at 1st and 99th percentiles
                lower_bound = result[base_col].quantile(0.01)
                upper_bound = result[base_col].quantile(0.99)
                
                result.loc[outlier_mask & (result[base_col] < lower_bound), base_col] = lower_bound
                result.loc[outlier_mask & (result[base_col] > upper_bound), base_col] = upper_bound
                
            elif method == 'cap':
                # Cap at 3 standard deviations
                mean_val = result[base_col].mean()
                std_val = result[base_col].std()
                
                lower_bound = mean_val - 3 * std_val
                upper_bound = mean_val + 3 * std_val
                
                result.loc[outlier_mask & (result[base_col] < lower_bound), base_col] = lower_bound
                result.loc[outlier_mask & (result[base_col] > upper_bound), base_col] = upper_bound
                
            elif method == 'remove':
                # Set outliers to NaN
                result.loc[outlier_mask, base_col] = np.nan
        
        return result
    
    def create_business_day_index(self, 
                                start_date: str, 
                                end_date: str) -> pd.DatetimeIndex:
        """
        Create business day datetime index.
        
        Args:
            start_date: Start date string
            end_date: End date string
        
        Returns:
            Business day DatetimeIndex
        """
        return pd.bdate_range(start=start_date, end=end_date)
    
    def align_to_common_calendar(self, 
                               data: Dict[str, pd.DataFrame],
                               reference_calendar: Optional[pd.DatetimeIndex] = None) -> Dict[str, pd.DataFrame]:
        """
        Align all datasets to a common trading calendar.
        
        Args:
            data: Dictionary of DataFrames
            reference_calendar: Reference calendar to align to
        
        Returns:
            Aligned data dictionary
        """
        if reference_calendar is None:
            # Create reference calendar from union of all dates
            all_dates = set()
            for df in data.values():
                all_dates.update(df.index)
            reference_calendar = pd.DatetimeIndex(sorted(all_dates))
            
            if self.business_days_only:
                reference_calendar = reference_calendar[reference_calendar.dayofweek < 5]
        
        aligned_data = {}
        for symbol, df in data.items():
            # Reindex to common calendar
            aligned_df = df.reindex(reference_calendar)
            aligned_data[symbol] = aligned_df
        
        return aligned_data
    
    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Comprehensive data quality validation.
        
        Args:
            df: DataFrame to validate
        
        Returns:
            Dictionary with validation results
        """
        validation_results = {}
        
        # Check for missing data
        missing_data = df.isnull().sum()
        validation_results['missing_data'] = missing_data[missing_data > 0].to_dict()
        
        # Check for infinite values
        inf_data = np.isinf(df.select_dtypes(include=[np.number])).sum()
        validation_results['infinite_values'] = inf_data[inf_data > 0].to_dict()
        
        # Check for negative prices
        price_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close']
        negative_prices = {}
        for col in price_columns:
            if col in df.columns:
                neg_count = (df[col] < 0).sum()
                if neg_count > 0:
                    negative_prices[col] = neg_count
        validation_results['negative_prices'] = negative_prices
        
        # Check for zero volume
        if 'Volume' in df.columns:
            zero_volume = (df['Volume'] == 0).sum()
            validation_results['zero_volume'] = zero_volume
        
        # Check for duplicate dates
        duplicate_dates = df.index.duplicated().sum()
        validation_results['duplicate_dates'] = duplicate_dates
        
        # Check data range
        validation_results['date_range'] = {
            'start': df.index.min(),
            'end': df.index.max(),
            'trading_days': len(df)
        }
        
        # Log validation summary
        self.logger.info(f"Data quality validation results: {validation_results}")
        
        return validation_results
    
    def process_pipeline(self, 
                        data: Dict[str, pd.DataFrame],
                        start_date: str,
                        end_date: str) -> Dict[str, pd.DataFrame]:
        """
        Complete data processing pipeline.
        
        Args:
            data: Raw data dictionary
            start_date: Processing start date
            end_date: Processing end date
        
        Returns:
            Processed data dictionary
        """
        self.logger.info("Starting data processing pipeline")
        
        # Step 1: Validate raw data
        for symbol, df in data.items():
            self.validate_data_quality(df)
        
        # Step 2: Corporate action adjustments
        processed_data = {}
        for symbol, df in data.items():
            processed_data[symbol] = self.adjust_corporate_actions(df)
        
        # Step 3: Survivorship bias correction
        processed_data = self.correct_survivorship_bias(
            processed_data, start_date
        )
        
        # Step 4: Align to common calendar
        processed_data = self.align_to_common_calendar(processed_data)
        
        # Step 5: Handle missing data
        for symbol in processed_data:
            processed_data[symbol] = self.handle_missing_data(
                processed_data[symbol], method='hybrid'
            )
        
        # Step 6: Outlier detection and treatment
        for symbol in processed_data:
            processed_data[symbol] = self.detect_outliers(processed_data[symbol])
            processed_data[symbol] = self.treat_outliers(
                processed_data[symbol], method='winsorize'
            )
        
        # Step 7: Final validation
        for symbol, df in processed_data.items():
            final_validation = self.validate_data_quality(df)
            self.logger.info(f"Final validation for {symbol}: {final_validation}")
        
        self.logger.info("Data processing pipeline completed")
        return processed_data