import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod
from scipy import stats
import warnings

@dataclass
class ValidationRule:
    """Defines a validation rule for data quality checks."""
    name: str
    description: str
    severity: str  # 'ERROR', 'WARNING', 'INFO'
    threshold: Optional[float] = None
    
@dataclass
class ValidationResult:
    """Result of a validation check."""
    rule_name: str
    status: str  # 'PASS', 'FAIL', 'WARNING'
    message: str
    details: Dict[str, Any]
    severity: str

class BaseValidator(ABC):
    """Abstract base class for data validators."""
    
    def __init__(self, rules: List[ValidationRule]):
        self.rules = rules
        self.logger = logging.getLogger(__name__)
    
    @abstractmethod
    def validate(self, data: pd.DataFrame) -> List[ValidationResult]:
        """Validate data against rules."""
        pass

class DataValidator:
    """
    Comprehensive data validation system for financial data quality assurance
    with regime-aware validation rules and anomaly detection.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.validation_rules = self._initialize_validation_rules()
        self.validation_results = []
        
    def _initialize_validation_rules(self) -> List[ValidationRule]:
        """Initialize standard validation rules."""
        return [
            ValidationRule(
                name="missing_data_check",
                description="Check for excessive missing data",
                severity="WARNING",
                threshold=0.05  # 5% missing data threshold
            ),
            ValidationRule(
                name="outlier_detection",
                description="Detect statistical outliers",
                severity="WARNING",
                threshold=3.0  # 3 standard deviations
            ),
            ValidationRule(
                name="price_anomaly_check",
                description="Check for unrealistic price movements",
                severity="ERROR",
                threshold=0.5  # 50% daily change threshold
            ),
            ValidationRule(
                name="volume_anomaly_check",
                description="Check for volume anomalies",
                severity="WARNING",
                threshold=10.0  # 10x average volume
            ),
            ValidationRule(
                name="data_continuity_check",
                description="Check for data gaps and continuity",
                severity="WARNING",
                threshold=0.1  # 10% missing days threshold
            ),
            ValidationRule(
                name="duplicate_check",
                description="Check for duplicate records",
                severity="ERROR"
            ),
            ValidationRule(
                name="date_consistency_check",
                description="Check for date ordering and consistency",
                severity="ERROR"
            ),
            ValidationRule(
                name="business_logic_check",
                description="Check business logic constraints",
                severity="ERROR"
            )
        ]
    
    def validate_dataset(self, 
                         data: pd.DataFrame, 
                         data_type: str = 'market_data',
                         symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Comprehensive validation of a dataset.
        
        Args:
            data: DataFrame to validate
            data_type: Type of data ('market_data', 'fundamental', 'macro')
            symbol: Optional symbol identifier
        
        Returns:
            Dictionary containing validation results and summary
        """
        self.logger.info(f"Validating {data_type} dataset" + (f" for {symbol}" if symbol else ""))
        
        if data is None or data.empty:
            return {
                'status': 'FAILED',
                'reason': 'Empty or null dataset',
                'validation_results': [],
                'summary': {'errors': 1, 'warnings': 0, 'info': 0}
            }
        
        validation_results = []
        
        # Run all validation checks
        for rule in self.validation_rules:
            try:
                result = self._execute_validation_rule(data, rule, data_type, symbol)
                if result:
                    validation_results.append(result)
            except Exception as e:
                self.logger.error(f"Error executing rule {rule.name}: {str(e)}")
                validation_results.append(ValidationResult(
                    rule_name=rule.name,
                    status='FAIL',
                    message=f"Validation rule execution failed: {str(e)}",
                    details={},
                    severity='ERROR'
                ))
        
        # Summarize results
        summary = self._summarize_results(validation_results)
        overall_status = self._determine_overall_status(validation_results)
        
        return {
            'status': overall_status,
            'validation_results': validation_results,
            'summary': summary,
            'data_info': {
                'rows': len(data),
                'columns': len(data.columns),
                'date_range': f"{data.index[0]} to {data.index[-1]}" if isinstance(data.index, pd.DatetimeIndex) else "No date index",
                'memory_usage': data.memory_usage(deep=True).sum()
            }
        }
    
    def _execute_validation_rule(self, 
                                data: pd.DataFrame, 
                                rule: ValidationRule,
                                data_type: str,
                                symbol: Optional[str]) -> Optional[ValidationResult]:
        """Execute a specific validation rule."""
        
        if rule.name == "missing_data_check":
            return self._check_missing_data(data, rule)
        elif rule.name == "outlier_detection":
            return self._check_outliers(data, rule)
        elif rule.name == "price_anomaly_check":
            return self._check_price_anomalies(data, rule)
        elif rule.name == "volume_anomaly_check":
            return self._check_volume_anomalies(data, rule)
        elif rule.name == "data_continuity_check":
            return self._check_data_continuity(data, rule)
        elif rule.name == "duplicate_check":
            return self._check_duplicates(data, rule)
        elif rule.name == "date_consistency_check":
            return self._check_date_consistency(data, rule)
        elif rule.name == "business_logic_check":
            return self._check_business_logic(data, rule, data_type)
        
        return None
    
    def _check_missing_data(self, data: pd.DataFrame, rule: ValidationRule) -> ValidationResult:
        """Check for missing data."""
        total_cells = len(data) * len(data.columns)
        missing_cells = data.isnull().sum().sum()
        missing_percentage = missing_cells / total_cells
        
        status = 'PASS'
        message = f"Missing data: {missing_percentage:.2%} of cells"
        
        if missing_percentage > rule.threshold:
            status = 'WARNING' if rule.severity == 'WARNING' else 'FAIL'
            message = f"High missing data: {missing_percentage:.2%} exceeds threshold of {rule.threshold:.2%}"
        
        # Column-wise missing data analysis
        missing_by_column = data.isnull().sum()
        problematic_columns = missing_by_column[missing_by_column > len(data) * rule.threshold].index.tolist()
        
        return ValidationResult(
            rule_name=rule.name,
            status=status,
            message=message,
            details={
                'missing_percentage': missing_percentage,
                'missing_cells': missing_cells,
                'total_cells': total_cells,
                'problematic_columns': problematic_columns,
                'missing_by_column': missing_by_column.to_dict()
            },
            severity=rule.severity
        )
    
    def _check_outliers(self, data: pd.DataFrame, rule: ValidationRule) -> ValidationResult:
        """Check for statistical outliers using multiple methods."""
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        outlier_details = {}
        total_outliers = 0
        
        for col in numeric_columns:
            series = data[col].dropna()
            if len(series) == 0:
                continue
            
            # Z-score method
            z_scores = np.abs(stats.zscore(series))
            z_outliers = (z_scores > rule.threshold).sum()
            
            # IQR method
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            iqr_outliers = ((series < (Q1 - 1.5 * IQR)) | (series > (Q3 + 1.5 * IQR))).sum()
            
            # Modified Z-score (using median)
            median = series.median()
            mad = np.median(np.abs(series - median))
            modified_z_scores = 0.6745 * (series - median) / mad if mad != 0 else np.zeros(len(series))
            modified_z_outliers = (np.abs(modified_z_scores) > rule.threshold).sum()
            
            outlier_details[col] = {
                'z_score_outliers': z_outliers,
                'iqr_outliers': iqr_outliers,
                'modified_z_outliers': modified_z_outliers,
                'total_values': len(series)
            }
            
            total_outliers += max(z_outliers, iqr_outliers, modified_z_outliers)
        
        outlier_percentage = total_outliers / (len(data) * len(numeric_columns)) if len(numeric_columns) > 0 else 0
        
        status = 'PASS'
        message = f"Outliers detected: {outlier_percentage:.2%} of numeric values"
        
        if outlier_percentage > 0.05:  # More than 5% outliers
            status = 'WARNING'
            message = f"High outlier percentage: {outlier_percentage:.2%}"
        
        return ValidationResult(
            rule_name=rule.name,
            status=status,
            message=message,
            details={
                'outlier_percentage': outlier_percentage,
                'total_outliers': total_outliers,
                'column_details': outlier_details
            },
            severity=rule.severity
        )
    
    def _check_price_anomalies(self, data: pd.DataFrame, rule: ValidationRule) -> ValidationResult:
        """Check for unrealistic price movements."""
        price_columns = [col for col in data.columns if any(x in col.lower() for x in ['close', 'price', 'open', 'high', 'low'])]
        
        if not price_columns:
            return ValidationResult(
                rule_name=rule.name,
                status='PASS',
                message="No price columns found to validate",
                details={},
                severity=rule.severity
            )
        
        anomaly_details = {}
        total_anomalies = 0
        
        for col in price_columns:
            if col in data.columns:
                prices = data[col].dropna()
                if len(prices) < 2:
                    continue
                
                # Calculate daily returns
                returns = prices.pct_change().dropna()
                
                # Check for extreme returns
                extreme_returns = np.abs(returns) > rule.threshold
                extreme_count = extreme_returns.sum()
                
                # Check for impossible values
                negative_prices = (prices <= 0).sum()
                
                # Check for price gaps
                price_ratios = prices / prices.shift(1)
                large_gaps = ((price_ratios > 2) | (price_ratios < 0.5)).sum()
                
                anomaly_details[col] = {
                    'extreme_returns': extreme_count,
                    'negative_prices': negative_prices,
                    'large_gaps': large_gaps,
                    'max_return': returns.abs().max(),
                    'total_observations': len(returns)
                }
                
                total_anomalies += extreme_count + negative_prices + large_gaps
        
        status = 'PASS'
        message = f"Price anomalies detected: {total_anomalies} across all price columns"
        
        if total_anomalies > 0:
            status = 'FAIL' if rule.severity == 'ERROR' else 'WARNING'
            message = f"Price anomalies found: {total_anomalies} issues detected"
        
        return ValidationResult(
            rule_name=rule.name,
            status=status,
            message=message,
            details=anomaly_details,
            severity=rule.severity
        )
    
    def _check_volume_anomalies(self, data: pd.DataFrame, rule: ValidationRule) -> ValidationResult:
        """Check for volume anomalies."""
        volume_columns = [col for col in data.columns if 'volume' in col.lower()]
        
        if not volume_columns:
            return ValidationResult(
                rule_name=rule.name,
                status='PASS',
                message="No volume columns found to validate",
                details={},
                severity=rule.severity
            )
        
        volume_details = {}
        total_anomalies = 0
        
        for col in volume_columns:
            if col in data.columns:
                volumes = data[col].dropna()
                if len(volumes) == 0:
                    continue
                
                # Zero volume days
                zero_volume = (volumes == 0).sum()
                
                # Extremely high volume (compared to rolling average)
                avg_volume = volumes.rolling(window=20).mean()
                high_volume = (volumes > avg_volume * rule.threshold).sum()
                
                # Negative volume (should not exist)
                negative_volume = (volumes < 0).sum()
                
                volume_details[col] = {
                    'zero_volume_days': zero_volume,
                    'high_volume_days': high_volume,
                    'negative_volume_days': negative_volume,
                    'avg_volume': avg_volume.mean(),
                    'max_volume': volumes.max(),
                    'total_observations': len(volumes)
                }
                
                total_anomalies += zero_volume + negative_volume
        
        status = 'PASS'
        message = f"Volume anomalies: {total_anomalies} issues detected"
        
        if total_anomalies > len(data) * 0.1:  # More than 10% of data
            status = 'WARNING'
            message = f"High volume anomalies: {total_anomalies} issues detected"
        
        return ValidationResult(
            rule_name=rule.name,
            status=status,
            message=message,
            details=volume_details,
            severity=rule.severity
        )
    
    def _check_data_continuity(self, data: pd.DataFrame, rule: ValidationRule) -> ValidationResult:
        """Check for data continuity and gaps."""
        if not isinstance(data.index, pd.DatetimeIndex):
            return ValidationResult(
                rule_name=rule.name,
                status='WARNING',
                message="Data does not have datetime index for continuity check",
                details={},
                severity=rule.severity
            )
        
        # Expected business days
        start_date = data.index[0]
        end_date = data.index[-1]
        expected_days = pd.bdate_range(start_date, end_date)
        
        # Actual vs expected
        missing_days = len(expected_days) - len(data)
        missing_percentage = missing_days / len(expected_days)
        
        # Find gaps larger than 5 business days
        gaps = data.index.to_series().diff()[1:]
        large_gaps = gaps[gaps > pd.Timedelta(days=7)]  # More than a week
        
        status = 'PASS'
        message = f"Data continuity: {missing_percentage:.2%} missing business days"
        
        if missing_percentage > rule.threshold:
            status = 'WARNING'
            message = f"Poor data continuity: {missing_percentage:.2%} missing business days"
        
        return ValidationResult(
            rule_name=rule.name,
            status=status,
            message=message,
            details={
                'missing_days': missing_days,
                'expected_days': len(expected_days),
                'missing_percentage': missing_percentage,
                'large_gaps': len(large_gaps),
                'largest_gap': gaps.max() if len(gaps) > 0 else pd.Timedelta(0)
            },
            severity=rule.severity
        )
    
    def _check_duplicates(self, data: pd.DataFrame, rule: ValidationRule) -> ValidationResult:
        """Check for duplicate records."""
        # Check for duplicate indices
        duplicate_indices = data.index.duplicated().sum()
        
        # Check for completely duplicate rows
        duplicate_rows = data.duplicated().sum()
        
        # Check for duplicate values in key columns
        duplicate_details = {
            'duplicate_indices': duplicate_indices,
            'duplicate_rows': duplicate_rows
        }
        
        total_duplicates = duplicate_indices + duplicate_rows
        
        status = 'PASS'
        message = f"Duplicates found: {total_duplicates} issues"
        
        if total_duplicates > 0:
            status = 'FAIL' if rule.severity == 'ERROR' else 'WARNING'
            message = f"Duplicates detected: {duplicate_indices} index duplicates, {duplicate_rows} row duplicates"
        
        return ValidationResult(
            rule_name=rule.name,
            status=status,
            message=message,
            details=duplicate_details,
            severity=rule.severity
        )
    
    def _check_date_consistency(self, data: pd.DataFrame, rule: ValidationRule) -> ValidationResult:
        """Check date ordering and consistency."""
        if not isinstance(data.index, pd.DatetimeIndex):
            return ValidationResult(
                rule_name=rule.name,
                status='WARNING',
                message="Data does not have datetime index for date consistency check",
                details={},
                severity=rule.severity
            )
        
        issues = []
        
        # Check if dates are sorted
        if not data.index.is_monotonic_increasing:
            issues.append("Dates are not in ascending order")
        
        # Check for future dates
        future_dates = (data.index > pd.Timestamp.now()).sum()
        if future_dates > 0:
            issues.append(f"{future_dates} future dates found")
        
        # Check for very old dates (before 1900)
        very_old_dates = (data.index < pd.Timestamp('1900-01-01')).sum()
        if very_old_dates > 0:
            issues.append(f"{very_old_dates} dates before 1900 found")
        
        status = 'PASS' if not issues else ('FAIL' if rule.severity == 'ERROR' else 'WARNING')
        message = "Date consistency check passed" if not issues else "; ".join(issues)
        
        return ValidationResult(
            rule_name=rule.name,
            status=status,
            message=message,
            details={
                'is_sorted': data.index.is_monotonic_increasing,
                'future_dates': future_dates,
                'very_old_dates': very_old_dates,
                'date_range': f"{data.index[0]} to {data.index[-1]}"
            },
            severity=rule.severity
        )
    
    def _check_business_logic(self, data: pd.DataFrame, rule: ValidationRule, data_type: str) -> ValidationResult:
        """Check business logic constraints specific to data type."""
        issues = []
        details = {}
        
        if data_type == 'market_data':
            # Market data specific checks
            if 'High' in data.columns and 'Low' in data.columns:
                high_low_violations = (data['High'] < data['Low']).sum()
                if high_low_violations > 0:
                    issues.append(f"{high_low_violations} cases where High < Low")
                details['high_low_violations'] = high_low_violations
            
            if all(col in data.columns for col in ['Open', 'High', 'Low', 'Close']):
                # OHLC consistency checks
                open_violations = ((data['Open'] > data['High']) | (data['Open'] < data['Low'])).sum()
                close_violations = ((data['Close'] > data['High']) | (data['Close'] < data['Low'])).sum()
                
                if open_violations > 0:
                    issues.append(f"{open_violations} cases where Open outside High-Low range")
                if close_violations > 0:
                    issues.append(f"{close_violations} cases where Close outside High-Low range")
                
                details['open_violations'] = open_violations
                details['close_violations'] = close_violations
        
        elif data_type == 'fundamental':
            # Fundamental data specific checks
            ratio_columns = [col for col in data.columns if any(x in col.lower() for x in ['ratio', 'margin', 'roe', 'roa'])]
            for col in ratio_columns:
                if col in data.columns:
                    # Check for impossible ratio values
                    impossible_values = ((data[col] < -10) | (data[col] > 10)).sum()  # Ratios beyond ±1000%
                    if impossible_values > 0:
                        issues.append(f"{impossible_values} impossible values in {col}")
                    details[f'{col}_impossible_values'] = impossible_values
        
        elif data_type == 'macro':
            # Macroeconomic data specific checks
            rate_columns = [col for col in data.columns if any(x in col.lower() for x in ['rate', 'yield', 'inflation'])]
            for col in rate_columns:
                if col in data.columns:
                    # Check for impossible interest rates (negative rates are possible but extreme values are not)
                    extreme_rates = ((data[col] < -10) | (data[col] > 50)).sum()  # Beyond -10% or 50%
                    if extreme_rates > 0:
                        issues.append(f"{extreme_rates} extreme values in {col}")
                    details[f'{col}_extreme_values'] = extreme_rates
        
        status = 'PASS' if not issues else ('FAIL' if rule.severity == 'ERROR' else 'WARNING')
        message = "Business logic check passed" if not issues else "; ".join(issues)
        
        return ValidationResult(
            rule_name=rule.name,
            status=status,
            message=message,
            details=details,
            severity=rule.severity
        )
    
    def _summarize_results(self, results: List[ValidationResult]) -> Dict[str, int]:
        """Summarize validation results."""
        summary = {'errors': 0, 'warnings': 0, 'info': 0, 'passed': 0}
        
        for result in results:
            if result.status == 'FAIL':
                if result.severity == 'ERROR':
                    summary['errors'] += 1
                else:
                    summary['warnings'] += 1
            elif result.status == 'WARNING':
                summary['warnings'] += 1
            else:
                summary['passed'] += 1
        
        return summary
    
    def _determine_overall_status(self, results: List[ValidationResult]) -> str:
        """Determine overall validation status."""
        has_errors = any(r.status == 'FAIL' and r.severity == 'ERROR' for r in results)
        has_warnings = any(r.status in ['FAIL', 'WARNING'] for r in results)
        
        if has_errors:
            return 'FAILED'
        elif has_warnings:
            return 'WARNING'
        else:
            return 'PASSED'
    
    def validate_cross_dataset_consistency(self, 
                                         datasets: Dict[str, pd.DataFrame]) -> ValidationResult:
        """Validate consistency across multiple datasets."""
        issues = []
        details = {}
        
        if len(datasets) < 2:
            return ValidationResult(
                rule_name="cross_dataset_consistency",
                status='PASS',
                message="Single dataset provided, no cross-validation needed",
                details={},
                severity='INFO'
            )
        
        # Check date range consistency
        date_ranges = {}
        for name, data in datasets.items():
            if isinstance(data.index, pd.DatetimeIndex):
                date_ranges[name] = (data.index[0], data.index[-1])
        
        if date_ranges:
            min_start = min(start for start, _ in date_ranges.values())
            max_end = max(end for _, end in date_ranges.values())
            
            for name, (start, end) in date_ranges.items():
                if (max_end - start).days > 30 or (end - min_start).days > 30:  # More than 30 days difference
                    issues.append(f"Date range inconsistency in {name}")
            
            details['date_ranges'] = date_ranges
        
        # Check for common symbols/keys if applicable
        if all('Symbol' in data.columns for data in datasets.values()):
            symbol_sets = {name: set(data['Symbol'].unique()) for name, data in datasets.items()}
            
            # Find common symbols
            common_symbols = set.intersection(*symbol_sets.values()) if symbol_sets else set()
            details['common_symbols'] = len(common_symbols)
            details['symbol_overlap'] = {
                name: len(symbols.intersection(common_symbols)) / len(symbols) if symbols else 0
                for name, symbols in symbol_sets.items()
            }
        
        status = 'PASS' if not issues else 'WARNING'
        message = "Cross-dataset consistency check passed" if not issues else "; ".join(issues)
        
        return ValidationResult(
            rule_name="cross_dataset_consistency",
            status=status,
            message=message,
            details=details,
            severity='WARNING'
        )
    
    def generate_validation_report(self, 
                                  validation_results: Dict[str, Dict[str, Any]]) -> str:
        """Generate a comprehensive validation report."""
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("DATA VALIDATION REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        overall_status = 'PASSED'
        total_errors = 0
        total_warnings = 0
        
        for dataset_name, result in validation_results.items():
            report_lines.append(f"Dataset: {dataset_name}")
            report_lines.append("-" * 40)
            report_lines.append(f"Status: {result['status']}")
            
            if 'data_info' in result:
                info = result['data_info']
                report_lines.append(f"Rows: {info['rows']:,}")
                report_lines.append(f"Columns: {info['columns']}")
                report_lines.append(f"Date Range: {info['date_range']}")
                report_lines.append(f"Memory Usage: {info['memory_usage'] / 1024 / 1024:.2f} MB")
            
            summary = result.get('summary', {})
            errors = summary.get('errors', 0)
            warnings = summary.get('warnings', 0)
            passed = summary.get('passed', 0)
            
            total_errors += errors
            total_warnings += warnings
            
            if result['status'] != 'PASSED':
                overall_status = result['status']
            
            report_lines.append(f"Validation Results: {errors} errors, {warnings} warnings, {passed} passed")
            
            # Add details for failed validations
            failed_validations = [r for r in result.get('validation_results', []) 
                                if r.status in ['FAIL', 'WARNING']]
            
            if failed_validations:
                report_lines.append("\nIssues Found:")
                for validation in failed_validations:
                    report_lines.append(f"  • {validation.message} [{validation.severity}]")
            
            report_lines.append("")
        
        # Overall summary
        report_lines.append("=" * 80)
        report_lines.append("OVERALL SUMMARY")
        report_lines.append("=" * 80)
        report_lines.append(f"Overall Status: {overall_status}")
        report_lines.append(f"Total Datasets: {len(validation_results)}")
        report_lines.append(f"Total Errors: {total_errors}")
        report_lines.append(f"Total Warnings: {total_warnings}")
        
        # Recommendations
        report_lines.append("\nRECOMMENDations:")
        if total_errors > 0:
            report_lines.append("• Address all ERROR-level issues before proceeding with analysis")
        if total_warnings > 0:
            report_lines.append("• Review WARNING-level issues and consider data cleaning")
        if total_errors == 0 and total_warnings == 0:
            report_lines.append("• Data quality is good for analysis")
        
        return "\n".join(report_lines)