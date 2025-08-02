import numpy as np
import pandas as pd
import yfinance as yf
from typing import List, Dict, Optional, Union
from datetime import datetime, timedelta
import logging
import time
from dataclasses import dataclass
import requests
import json

@dataclass
class FundamentalDataConfig:
    """Configuration for fundamental data collection."""
    symbols: List[str]
    metrics: List[str]
    periods: List[str] = None  # ['annual', 'quarterly']
    include_ratios: bool = True
    include_growth: bool = True
    lookback_years: int = 5

class FundamentalDataCollector:
    """
    Comprehensive fundamental data collector for financial statements,
    ratios, and corporate metrics with regime-aware factor construction.
    """
    
    def __init__(self, config: FundamentalDataConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.rate_limit_delay = 0.5  # Longer delay for fundamental data
        self.collected_data = {}
        
        # Default periods if not specified
        if self.config.periods is None:
            self.config.periods = ['annual', 'quarterly']
    
    async def collect_data(self) -> Dict[str, Dict]:
        """
        Collect fundamental data for specified symbols.
        
        Returns:
            Dictionary mapping symbols to their fundamental data
        """
        self.logger.info(f"Collecting fundamental data for {len(self.config.symbols)} symbols")
        
        all_data = {}
        
        for symbol in self.config.symbols:
            try:
                self.logger.info(f"Collecting fundamental data for {symbol}")
                data = await self._collect_symbol_fundamentals(symbol)
                
                if data:
                    all_data[symbol] = data
                    self.logger.info(f"Successfully collected fundamental data for {symbol}")
                else:
                    self.logger.warning(f"No fundamental data collected for {symbol}")
                
                time.sleep(self.rate_limit_delay)
                
            except Exception as e:
                self.logger.error(f"Error collecting fundamental data for {symbol}: {str(e)}")
                continue
        
        self.collected_data = all_data
        return all_data
    
    async def _collect_symbol_fundamentals(self, symbol: str) -> Optional[Dict]:
        """Collect fundamental data for a single symbol."""
        try:
            ticker = yf.Ticker(symbol)
            
            fundamental_data = {
                'symbol': symbol,
                'company_info': self._get_company_info(ticker),
                'financial_statements': self._get_financial_statements(ticker),
                'financial_ratios': self._calculate_financial_ratios(ticker),
                'growth_metrics': self._calculate_growth_metrics(ticker),
                'valuation_metrics': self._get_valuation_metrics(ticker),
                'quality_scores': self._calculate_quality_scores(ticker)
            }
            
            return fundamental_data
            
        except Exception as e:
            self.logger.error(f"Error in _collect_symbol_fundamentals for {symbol}: {str(e)}")
            return None
    
    def _get_company_info(self, ticker) -> Dict:
        """Extract company information and basic metrics."""
        try:
            info = ticker.info
            
            company_data = {
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'market_cap': info.get('marketCap', np.nan),
                'enterprise_value': info.get('enterpriseValue', np.nan),
                'employees': info.get('fullTimeEmployees', np.nan),
                'country': info.get('country', 'Unknown'),
                'exchange': info.get('exchange', 'Unknown'),
                'currency': info.get('currency', 'USD'),
                'business_summary': info.get('businessSummary', ''),
                'website': info.get('website', ''),
                'beta': info.get('beta', np.nan),
                'shares_outstanding': info.get('sharesOutstanding', np.nan),
                'float_shares': info.get('floatShares', np.nan),
                'shares_short': info.get('sharesShort', np.nan),
                'short_ratio': info.get('shortRatio', np.nan),
                'held_percent_institutions': info.get('heldPercentInstitutions', np.nan),
                'held_percent_insiders': info.get('heldPercentInsiders', np.nan)
            }
            
            return company_data
            
        except Exception as e:
            self.logger.error(f"Error getting company info: {str(e)}")
            return {}
    
    def _get_financial_statements(self, ticker) -> Dict:
        """Extract financial statement data."""
        try:
            financial_data = {}
            
            # Income Statement
            try:
                income_stmt = ticker.financials
                quarterly_income = ticker.quarterly_financials
                
                if not income_stmt.empty:
                    financial_data['income_statement_annual'] = income_stmt.to_dict()
                if not quarterly_income.empty:
                    financial_data['income_statement_quarterly'] = quarterly_income.to_dict()
                    
            except Exception as e:
                self.logger.warning(f"Error getting income statement: {str(e)}")
            
            # Balance Sheet
            try:
                balance_sheet = ticker.balance_sheet
                quarterly_balance = ticker.quarterly_balance_sheet
                
                if not balance_sheet.empty:
                    financial_data['balance_sheet_annual'] = balance_sheet.to_dict()
                if not quarterly_balance.empty:
                    financial_data['balance_sheet_quarterly'] = quarterly_balance.to_dict()
                    
            except Exception as e:
                self.logger.warning(f"Error getting balance sheet: {str(e)}")
            
            # Cash Flow Statement
            try:
                cash_flow = ticker.cashflow
                quarterly_cash_flow = ticker.quarterly_cashflow
                
                if not cash_flow.empty:
                    financial_data['cash_flow_annual'] = cash_flow.to_dict()
                if not quarterly_cash_flow.empty:
                    financial_data['cash_flow_quarterly'] = quarterly_cash_flow.to_dict()
                    
            except Exception as e:
                self.logger.warning(f"Error getting cash flow: {str(e)}")
            
            return financial_data
            
        except Exception as e:
            self.logger.error(f"Error getting financial statements: {str(e)}")
            return {}
    
    def _calculate_financial_ratios(self, ticker) -> Dict:
        """Calculate comprehensive financial ratios."""
        try:
            ratios = {}
            
            # Get latest financial data
            income_stmt = ticker.financials
            balance_sheet = ticker.balance_sheet
            cash_flow = ticker.cashflow
            info = ticker.info
            
            if income_stmt.empty or balance_sheet.empty:
                return ratios
            
            # Get latest year data
            latest_income = income_stmt.iloc[:, 0]
            latest_balance = balance_sheet.iloc[:, 0]
            latest_cash_flow = cash_flow.iloc[:, 0] if not cash_flow.empty else pd.Series()
            
            # Profitability Ratios
            revenue = latest_income.get('Total Revenue', np.nan)
            net_income = latest_income.get('Net Income', np.nan)
            gross_profit = latest_income.get('Gross Profit', np.nan)
            operating_income = latest_income.get('Operating Income', np.nan)
            ebitda = latest_income.get('EBITDA', np.nan)
            
            total_assets = latest_balance.get('Total Assets', np.nan)
            total_equity = latest_balance.get('Total Stockholder Equity', np.nan)
            
            if pd.notna(revenue) and revenue != 0:
                ratios['gross_margin'] = gross_profit / revenue if pd.notna(gross_profit) else np.nan
                ratios['operating_margin'] = operating_income / revenue if pd.notna(operating_income) else np.nan
                ratios['net_margin'] = net_income / revenue if pd.notna(net_income) else np.nan
                ratios['ebitda_margin'] = ebitda / revenue if pd.notna(ebitda) else np.nan
            
            if pd.notna(total_assets) and total_assets != 0:
                ratios['roa'] = net_income / total_assets if pd.notna(net_income) else np.nan
            
            if pd.notna(total_equity) and total_equity != 0:
                ratios['roe'] = net_income / total_equity if pd.notna(net_income) else np.nan
            
            # Liquidity Ratios
            current_assets = latest_balance.get('Total Current Assets', np.nan)
            current_liabilities = latest_balance.get('Total Current Liabilities', np.nan)
            cash = latest_balance.get('Cash And Cash Equivalents', np.nan)
            
            if pd.notna(current_liabilities) and current_liabilities != 0:
                ratios['current_ratio'] = current_assets / current_liabilities if pd.notna(current_assets) else np.nan
                ratios['quick_ratio'] = (current_assets - latest_balance.get('Inventory', 0)) / current_liabilities if pd.notna(current_assets) else np.nan
                ratios['cash_ratio'] = cash / current_liabilities if pd.notna(cash) else np.nan
            
            # Leverage Ratios
            total_debt = latest_balance.get('Total Debt', np.nan)
            long_term_debt = latest_balance.get('Long Term Debt', np.nan)
            
            if pd.notna(total_assets) and total_assets != 0:
                ratios['debt_to_assets'] = total_debt / total_assets if pd.notna(total_debt) else np.nan
            
            if pd.notna(total_equity) and total_equity != 0:
                ratios['debt_to_equity'] = total_debt / total_equity if pd.notna(total_debt) else np.nan
            
            # Efficiency Ratios
            if pd.notna(total_assets) and total_assets != 0:
                ratios['asset_turnover'] = revenue / total_assets if pd.notna(revenue) else np.nan
            
            inventory = latest_balance.get('Inventory', np.nan)
            cost_of_goods = latest_income.get('Cost Of Revenue', np.nan)
            
            if pd.notna(inventory) and inventory != 0 and pd.notna(cost_of_goods):
                ratios['inventory_turnover'] = cost_of_goods / inventory
            
            # Market Ratios (from info)
            ratios['pe_ratio'] = info.get('trailingPE', np.nan)
            ratios['forward_pe'] = info.get('forwardPE', np.nan)
            ratios['pb_ratio'] = info.get('priceToBook', np.nan)
            ratios['ps_ratio'] = info.get('priceToSalesTrailing12Months', np.nan)
            ratios['peg_ratio'] = info.get('pegRatio', np.nan)
            ratios['ev_ebitda'] = info.get('enterpriseToEbitda', np.nan)
            ratios['ev_revenue'] = info.get('enterpriseToRevenue', np.nan)
            
            # Dividend Ratios
            ratios['dividend_yield'] = info.get('dividendYield', np.nan)
            ratios['payout_ratio'] = info.get('payoutRatio', np.nan)
            
            return ratios
            
        except Exception as e:
            self.logger.error(f"Error calculating financial ratios: {str(e)}")
            return {}
    
    def _calculate_growth_metrics(self, ticker) -> Dict:
        """Calculate growth metrics over multiple periods."""
        try:
            growth_metrics = {}
            
            # Get multi-year data
            income_stmt = ticker.financials
            balance_sheet = ticker.balance_sheet
            
            if income_stmt.empty or len(income_stmt.columns) < 2:
                return growth_metrics
            
            # Calculate growth rates for key metrics
            metrics_to_analyze = [
                'Total Revenue', 'Gross Profit', 'Operating Income', 
                'Net Income', 'EBITDA', 'Total Assets', 'Total Stockholder Equity'
            ]
            
            for metric in metrics_to_analyze:
                if metric in income_stmt.index:
                    values = income_stmt.loc[metric]
                elif metric in balance_sheet.index:
                    values = balance_sheet.loc[metric]
                else:
                    continue
                
                # Calculate growth rates
                growth_rates = []
                for i in range(len(values) - 1):
                    if pd.notna(values.iloc[i]) and pd.notna(values.iloc[i+1]) and values.iloc[i+1] != 0:
                        growth_rate = (values.iloc[i] / values.iloc[i+1]) - 1
                        growth_rates.append(growth_rate)
                
                if growth_rates:
                    metric_clean = metric.lower().replace(' ', '_')
                    growth_metrics[f'{metric_clean}_growth_1y'] = growth_rates[0] if len(growth_rates) > 0 else np.nan
                    growth_metrics[f'{metric_clean}_growth_3y_avg'] = np.mean(growth_rates[:3]) if len(growth_rates) >= 3 else np.nan
                    growth_metrics[f'{metric_clean}_growth_5y_avg'] = np.mean(growth_rates) if len(growth_rates) >= 5 else np.nan
                    growth_metrics[f'{metric_clean}_growth_volatility'] = np.std(growth_rates) if len(growth_rates) > 1 else np.nan
            
            return growth_metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating growth metrics: {str(e)}")
            return {}
    
    def _get_valuation_metrics(self, ticker) -> Dict:
        """Get comprehensive valuation metrics."""
        try:
            valuation = {}
            info = ticker.info
            
            # Current valuation metrics
            valuation['market_cap'] = info.get('marketCap', np.nan)
            valuation['enterprise_value'] = info.get('enterpriseValue', np.nan)
            valuation['price_to_book'] = info.get('priceToBook', np.nan)
            valuation['price_to_sales'] = info.get('priceToSalesTrailing12Months', np.nan)
            valuation['ev_to_ebitda'] = info.get('enterpriseToEbitda', np.nan)
            valuation['ev_to_revenue'] = info.get('enterpriseToRevenue', np.nan)
            
            # Price metrics
            current_price = info.get('currentPrice', np.nan)
            book_value = info.get('bookValue', np.nan)
            
            if pd.notna(current_price) and pd.notna(book_value) and book_value != 0:
                valuation['price_to_book_calculated'] = current_price / book_value
            
            # Relative valuation (if industry data available)
            sector = info.get('sector')
            if sector:
                valuation['sector'] = sector
                # Note: Industry comparison would require additional data source
            
            return valuation
            
        except Exception as e:
            self.logger.error(f"Error getting valuation metrics: {str(e)}")
            return {}
    
    def _calculate_quality_scores(self, ticker) -> Dict:
        """Calculate quality scores based on financial metrics."""
        try:
            quality_scores = {}
            
            # Get financial data
            income_stmt = ticker.financials
            balance_sheet = ticker.balance_sheet
            cash_flow = ticker.cashflow
            
            if income_stmt.empty or balance_sheet.empty:
                return quality_scores
            
            # Profitability Quality Score
            profitability_factors = []
            
            # Check for consistent profitability
            net_income_series = income_stmt.loc['Net Income'] if 'Net Income' in income_stmt.index else pd.Series()
            if not net_income_series.empty:
                positive_years = (net_income_series > 0).sum()
                profitability_factors.append(positive_years / len(net_income_series))
            
            # ROE stability
            total_equity = balance_sheet.loc['Total Stockholder Equity'] if 'Total Stockholder Equity' in balance_sheet.index else pd.Series()
            if not net_income_series.empty and not total_equity.empty:
                roe_series = net_income_series / total_equity
                roe_volatility = roe_series.std()
                profitability_factors.append(1 / (1 + roe_volatility) if pd.notna(roe_volatility) else 0)
            
            quality_scores['profitability_quality'] = np.mean(profitability_factors) if profitability_factors else np.nan
            
            # Financial Strength Score
            strength_factors = []
            
            # Debt levels
            total_debt = balance_sheet.loc['Total Debt'] if 'Total Debt' in balance_sheet.index else pd.Series()
            total_assets = balance_sheet.loc['Total Assets'] if 'Total Assets' in balance_sheet.index else pd.Series()
            
            if not total_debt.empty and not total_assets.empty:
                debt_to_assets = total_debt.iloc[0] / total_assets.iloc[0]
                strength_factors.append(1 - min(debt_to_assets, 1) if pd.notna(debt_to_assets) else 0)
            
            # Current ratio
            current_assets = balance_sheet.loc['Total Current Assets'] if 'Total Current Assets' in balance_sheet.index else pd.Series()
            current_liabilities = balance_sheet.loc['Total Current Liabilities'] if 'Total Current Liabilities' in balance_sheet.index else pd.Series()
            
            if not current_assets.empty and not current_liabilities.empty:
                current_ratio = current_assets.iloc[0] / current_liabilities.iloc[0]
                strength_factors.append(min(current_ratio / 2, 1) if pd.notna(current_ratio) else 0)
            
            quality_scores['financial_strength'] = np.mean(strength_factors) if strength_factors else np.nan
            
            # Earnings Quality Score
            earnings_factors = []
            
            # Cash flow vs net income
            if not cash_flow.empty:
                operating_cash_flow = cash_flow.loc['Total Cash From Operating Activities'] if 'Total Cash From Operating Activities' in cash_flow.index else pd.Series()
                
                if not operating_cash_flow.empty and not net_income_series.empty:
                    cf_to_ni_ratio = operating_cash_flow.iloc[0] / net_income_series.iloc[0]
                    earnings_factors.append(min(cf_to_ni_ratio, 2) / 2 if pd.notna(cf_to_ni_ratio) and net_income_series.iloc[0] > 0 else 0)
            
            quality_scores['earnings_quality'] = np.mean(earnings_factors) if earnings_factors else np.nan
            
            # Overall Quality Score (weighted average)
            quality_components = [
                quality_scores.get('profitability_quality', 0) * 0.4,
                quality_scores.get('financial_strength', 0) * 0.4,
                quality_scores.get('earnings_quality', 0) * 0.2
            ]
            
            quality_scores['overall_quality'] = np.mean([x for x in quality_components if pd.notna(x)])
            
            return quality_scores
            
        except Exception as e:
            self.logger.error(f"Error calculating quality scores: {str(e)}")
            return {}
    
    def create_fundamental_matrix(self) -> pd.DataFrame:
        """Create a matrix of fundamental metrics for all symbols."""
        if not self.collected_data:
            self.logger.warning("No data collected yet. Run collect_data() first.")
            return pd.DataFrame()
        
        fundamental_matrix = []
        
        for symbol, data in self.collected_data.items():
            if not data:
                continue
            
            row = {'Symbol': symbol}
            
            # Company info
            company_info = data.get('company_info', {})
            row.update({
                'Sector': company_info.get('sector', 'Unknown'),
                'Industry': company_info.get('industry', 'Unknown'),
                'Market_Cap': company_info.get('market_cap', np.nan),
                'Beta': company_info.get('beta', np.nan)
            })
            
            # Financial ratios
            ratios = data.get('financial_ratios', {})
            row.update({
                'PE_Ratio': ratios.get('pe_ratio', np.nan),
                'PB_Ratio': ratios.get('pb_ratio', np.nan),
                'PS_Ratio': ratios.get('ps_ratio', np.nan),
                'ROE': ratios.get('roe', np.nan),
                'ROA': ratios.get('roa', np.nan),
                'Debt_to_Equity': ratios.get('debt_to_equity', np.nan),
                'Current_Ratio': ratios.get('current_ratio', np.nan),
                'Gross_Margin': ratios.get('gross_margin', np.nan),
                'Operating_Margin': ratios.get('operating_margin', np.nan),
                'Net_Margin': ratios.get('net_margin', np.nan)
            })
            
            # Growth metrics
            growth = data.get('growth_metrics', {})
            row.update({
                'Revenue_Growth_1Y': growth.get('total_revenue_growth_1y', np.nan),
                'Revenue_Growth_3Y': growth.get('total_revenue_growth_3y_avg', np.nan),
                'Earnings_Growth_1Y': growth.get('net_income_growth_1y', np.nan),
                'Earnings_Growth_3Y': growth.get('net_income_growth_3y_avg', np.nan)
            })
            
            # Quality scores
            quality = data.get('quality_scores', {})
            row.update({
                'Profitability_Quality': quality.get('profitability_quality', np.nan),
                'Financial_Strength': quality.get('financial_strength', np.nan),
                'Overall_Quality': quality.get('overall_quality', np.nan)
            })
            
            fundamental_matrix.append(row)
        
        return pd.DataFrame(fundamental_matrix)
    
    def get_sector_fundamentals(self) -> Dict[str, pd.DataFrame]:
        """Group fundamental data by sector."""
        fundamental_df = self.create_fundamental_matrix()
        
        if fundamental_df.empty:
            return {}
        
        sector_data = {}
        for sector in fundamental_df['Sector'].unique():
            if pd.notna(sector) and sector != 'Unknown':
                sector_df = fundamental_df[fundamental_df['Sector'] == sector].copy()
                sector_data[sector] = sector_df
        
        return sector_data
    
    def calculate_fundamental_scores(self) -> pd.DataFrame:
        """Calculate composite fundamental scores for regime analysis."""
        fundamental_df = self.create_fundamental_matrix()
        
        if fundamental_df.empty:
            return pd.DataFrame()
        
        # Normalize metrics for scoring (higher is better)
        score_metrics = [
            'ROE', 'ROA', 'Gross_Margin', 'Operating_Margin', 'Net_Margin',
            'Current_Ratio', 'Revenue_Growth_1Y', 'Earnings_Growth_1Y',
            'Profitability_Quality', 'Financial_Strength', 'Overall_Quality'
        ]
        
        # Inverse metrics (lower is better)
        inverse_metrics = ['PE_Ratio', 'PB_Ratio', 'PS_Ratio', 'Debt_to_Equity']
        
        scores_df = fundamental_df[['Symbol', 'Sector']].copy()
        
        # Calculate normalized scores
        for metric in score_metrics:
            if metric in fundamental_df.columns:
                values = fundamental_df[metric]
                # Normalize to 0-1 scale
                if values.std() > 0:
                    normalized = (values - values.min()) / (values.max() - values.min())
                    scores_df[f'{metric}_Score'] = normalized
        
        for metric in inverse_metrics:
            if metric in fundamental_df.columns:
                values = fundamental_df[metric]
                # Inverse normalization
                if values.std() > 0:
                    normalized = 1 - (values - values.min()) / (values.max() - values.min())
                    scores_df[f'{metric}_Score'] = normalized
        
        # Calculate composite scores
        profitability_cols = [col for col in scores_df.columns if any(x in col for x in ['ROE', 'ROA', 'Margin', 'Profitability'])]
        valuation_cols = [col for col in scores_df.columns if any(x in col for x in ['PE', 'PB', 'PS'])]
        growth_cols = [col for col in scores_df.columns if 'Growth' in col]
        quality_cols = [col for col in scores_df.columns if any(x in col for x in ['Quality', 'Strength', 'Current_Ratio'])]
        
        if profitability_cols:
            scores_df['Profitability_Score'] = scores_df[profitability_cols].mean(axis=1)
        if valuation_cols:
            scores_df['Valuation_Score'] = scores_df[valuation_cols].mean(axis=1)
        if growth_cols:
            scores_df['Growth_Score'] = scores_df[growth_cols].mean(axis=1)
        if quality_cols:
            scores_df['Quality_Score'] = scores_df[quality_cols].mean(axis=1)
        
        # Overall fundamental score
        composite_cols = ['Profitability_Score', 'Valuation_Score', 'Growth_Score', 'Quality_Score']
        available_cols = [col for col in composite_cols if col in scores_df.columns]
        
        if available_cols:
            scores_df['Overall_Fundamental_Score'] = scores_df[available_cols].mean(axis=1)
        
        return scores_df