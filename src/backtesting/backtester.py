import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Callable
from datetime import datetime, timedelta
import logging
import warnings
from dataclasses import dataclass
from copy import deepcopy
import pickle

warnings.filterwarnings('ignore')

@dataclass
class BacktestConfig:
    """Backtesting configuration parameters."""
    start_date: str
    end_date: str
    initial_capital: float = 1000000.0
    rebalance_frequency: str = 'monthly'  # 'daily', 'weekly', 'monthly', 'quarterly'
    transaction_costs: float = 0.001  # 10 bps
    market_impact: float = 0.0005  # 5 bps
    commission: float = 0.0001  # 1 bp
    max_leverage: float = 1.0
    benchmark: str = 'SPY'
    regime_aware: bool = True
    walk_forward_analysis: bool = True
    training_window: int = 252  # Days
    min_training_window: int = 126  # Minimum days
    expanding_window: bool = False

@dataclass
class BacktestResults:
    """Comprehensive backtesting results."""
    portfolio_returns: pd.Series
    portfolio_weights: pd.DataFrame
    benchmark_returns: pd.Series
    regime_probabilities: pd.DataFrame
    transaction_costs: pd.Series
    positions: pd.DataFrame
    trades: pd.DataFrame
    performance_metrics: Dict
    regime_attribution: Dict
    drawdown_analysis: Dict
    risk_metrics: Dict

class Backtester:
    """
    Comprehensive backtesting framework with walk-forward analysis,
    regime-aware performance attribution, and Monte Carlo simulation.
    """
    
    def __init__(self, config: BacktestConfig):
        """
        Initialize Backtester.
        
        Args:
            config: Backtesting configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Data storage
        self.price_data = None
        self.volume_data = None
        self.fundamental_data = None
        self.macro_data = None
        
        # Models
        self.regime_detector = None
        self.signal_generator = None
        self.portfolio_optimizer = None
        self.risk_manager = None
        self.execution_engine = None
        
        # Backtesting state
        self.current_date = None
        self.portfolio_value = config.initial_capital
        self.cash = config.initial_capital
        self.positions = {}
        self.portfolio_history = []
        self.trade_history = []
        self.rebalance_dates = []
        
        # Performance tracking
        self.daily_returns = []
        self.daily_weights = []
        self.daily_regime_probs = []
        self.transaction_cost_history = []
        
    def set_data(self, 
                price_data: pd.DataFrame,
                volume_data: Optional[pd.DataFrame] = None,
                fundamental_data: Optional[pd.DataFrame] = None,
                macro_data: Optional[pd.DataFrame] = None):
        """
        Set market data for backtesting.
        
        Args:
            price_data: Price data DataFrame
            volume_data: Volume data DataFrame
            fundamental_data: Fundamental data DataFrame
            macro_data: Macroeconomic data DataFrame
        """
        self.price_data = price_data
        self.volume_data = volume_data
        self.fundamental_data = fundamental_data
        self.macro_data = macro_data
        
        # Filter data by backtest period
        start_date = pd.to_datetime(self.config.start_date)
        end_date = pd.to_datetime(self.config.end_date)
        
        self.price_data = self.price_data.loc[start_date:end_date]
        
        if self.volume_data is not None:
            self.volume_data = self.volume_data.loc[start_date:end_date]
        
        if self.fundamental_data is not None:
            self.fundamental_data = self.fundamental_data.loc[:end_date]
        
        if self.macro_data is not None:
            self.macro_data = self.macro_data.loc[:end_date]
    
    def set_models(self,
                  regime_detector=None,
                  signal_generator=None,
                  portfolio_optimizer=None,
                  risk_manager=None,
                  execution_engine=None):
        """
        Set models for backtesting.
        
        Args:
            regime_detector: Regime detection model
            signal_generator: Signal generation model
            portfolio_optimizer: Portfolio optimization model
            risk_manager: Risk management model
            execution_engine: Execution engine
        """
        self.regime_detector = regime_detector
        self.signal_generator = signal_generator
        self.portfolio_optimizer = portfolio_optimizer
        self.risk_manager = risk_manager
        self.execution_engine = execution_engine
    
    def run_backtest(self) -> BacktestResults:
        """
        Run comprehensive backtesting with walk-forward analysis.
        
        Returns:
            BacktestResults object with comprehensive results
        """
        self.logger.info("Starting backtesting process")
        
        # Initialize backtest
        self._initialize_backtest()
        
        # Generate rebalancing dates
        self._generate_rebalance_dates()
        
        if self.config.walk_forward_analysis:
            results = self._run_walk_forward_backtest()
        else:
            results = self._run_static_backtest()
        
        # Calculate performance metrics
        final_results = self._calculate_final_results(results)
        
        self.logger.info("Backtesting completed")
        return final_results
    
    def _initialize_backtest(self):
        """Initialize backtesting state."""
        self.current_date = pd.to_datetime(self.config.start_date)
        self.portfolio_value = self.config.initial_capital
        self.cash = self.config.initial_capital
        self.positions = {asset: 0.0 for asset in self.price_data.columns}
        
        # Clear history
        self.portfolio_history = []
        self.trade_history = []
        self.daily_returns = []
        self.daily_weights = []
        self.daily_regime_probs = []
        self.transaction_cost_history = []
    
    def _generate_rebalance_dates(self):
        """Generate rebalancing dates based on frequency."""
        start_date = pd.to_datetime(self.config.start_date)
        end_date = pd.to_datetime(self.config.end_date)
        
        if self.config.rebalance_frequency == 'daily':
            self.rebalance_dates = pd.date_range(start_date, end_date, freq='B')
        elif self.config.rebalance_frequency == 'weekly':
            self.rebalance_dates = pd.date_range(start_date, end_date, freq='W-FRI')
        elif self.config.rebalance_frequency == 'monthly':
            self.rebalance_dates = pd.date_range(start_date, end_date, freq='BM')
        elif self.config.rebalance_frequency == 'quarterly':
            self.rebalance_dates = pd.date_range(start_date, end_date, freq='BQ')
        else:
            # Custom frequency
            self.rebalance_dates = pd.date_range(start_date, end_date, freq='BM')
        
        # Filter to actual trading days
        self.rebalance_dates = self.rebalance_dates.intersection(self.price_data.index)
    
    def _run_walk_forward_backtest(self) -> Dict:
        """Run walk-forward backtesting."""
        
        results = {
            'portfolio_returns': [],
            'portfolio_weights': [],
            'regime_probabilities': [],
            'transaction_costs': [],
            'positions': [],
            'trades': []
        }
        
        # Walk forward through time
        for rebalance_date in self.rebalance_dates:
            self.current_date = rebalance_date
            
            # Define training window
            if self.config.expanding_window:
                # Expanding window
                train_start = pd.to_datetime(self.config.start_date)
            else:
                # Rolling window
                train_start = rebalance_date - timedelta(days=self.config.training_window)
            
            train_end = rebalance_date - timedelta(days=1)
            
            # Ensure minimum training window
            if (train_end - train_start).days < self.config.min_training_window:
                continue
            
            # Get training data
            train_data = self._get_training_data(train_start, train_end)
            
            if train_data is None:
                continue
            
            # Retrain models
            self._retrain_models(train_data, train_start, train_end)
            
            # Generate signals and portfolio weights
            target_weights = self._generate_portfolio_weights(rebalance_date)
            
            if target_weights is None:
                continue
            
            # Execute rebalancing
            rebalance_results = self._execute_rebalancing(rebalance_date, target_weights)
            
            # Store results
            results['portfolio_weights'].append({
                'date': rebalance_date,
                'weights': target_weights.copy()
            })
            
            results['transaction_costs'].append({
                'date': rebalance_date,
                'cost': rebalance_results['transaction_cost']
            })
            
            # Update regime probabilities if available
            if self.regime_detector and hasattr(self.regime_detector, 'regime_probabilities'):
                regime_probs = self._get_current_regime_probabilities(rebalance_date)
                results['regime_probabilities'].append({
                    'date': rebalance_date,
                    'probabilities': regime_probs
                })
        
        # Calculate daily portfolio returns
        results['portfolio_returns'] = self._calculate_daily_returns()
        
        return results
    
    def _get_training_data(self, start_date: pd.Timestamp, end_date: pd.Timestamp) -> Optional[Dict]:
        """Get training data for the specified period."""
        
        try:
            # Price data
            price_subset = self.price_data.loc[start_date:end_date]
            
            if price_subset.empty:
                return None
            
            train_data = {'prices': price_subset}
            
            # Volume data
            if self.volume_data is not None:
                volume_subset = self.volume_data.loc[start_date:end_date]
                train_data['volumes'] = volume_subset
            
            # Fundamental data
            if self.fundamental_data is not None:
                fundamental_subset = self.fundamental_data.loc[:end_date]
                train_data['fundamentals'] = fundamental_subset
            
            # Macro data
            if self.macro_data is not None:
                macro_subset = self.macro_data.loc[:end_date]
                train_data['macro'] = macro_subset
            
            return train_data
            
        except Exception as e:
            self.logger.error(f"Error getting training data: {str(e)}")
            return None
    
    def _retrain_models(self, train_data: Dict, start_date: pd.Timestamp, end_date: pd.Timestamp):
        """Retrain models on training data."""
        
        try:
            price_data = train_data['prices']
            returns_data = price_data.pct_change().dropna()
            
            # Retrain regime detector
            if self.regime_detector:
                features = ['returns']
                if 'volumes' in train_data:
                    features.append('volume')
                
                # Create feature matrix
                feature_data = pd.DataFrame(index=returns_data.index)
                feature_data['returns'] = returns_data.mean(axis=1)
                
                if 'volumes' in train_data:
                    feature_data['volume'] = train_data['volumes'].mean(axis=1)
                
                # Add macro features if available
                if 'macro' in train_data and not train_data['macro'].empty:
                    macro_aligned = train_data['macro'].reindex(feature_data.index, method='ffill')
                    for col in macro_aligned.columns:
                        feature_data[f'macro_{col}'] = macro_aligned[col]
                
                # Fit regime detector
                self.regime_detector.fit_hmm_regime(feature_data, list(feature_data.columns))
            
            # Retrain signal generator (if applicable)
            if self.signal_generator and hasattr(self.signal_generator, 'fit'):
                self.signal_generator.fit(returns_data)
            
            # Update risk manager
            if self.risk_manager:
                self.risk_manager.fit_factor_model(returns_data)
            
        except Exception as e:
            self.logger.error(f"Error retraining models: {str(e)}")
    
    def _generate_portfolio_weights(self, date: pd.Timestamp) -> Optional[np.ndarray]:
        """Generate target portfolio weights for the given date."""
        
        try:
            # Get current regime probabilities
            regime_probs = self._get_current_regime_probabilities(date)
            
            # Generate expected returns (simplified)
            expected_returns = self._generate_expected_returns(date, regime_probs)
            
            if expected_returns is None:
                return None
            
            # Get current portfolio weights
            current_weights = self._get_current_weights()
            
            # Set up portfolio optimizer
            if self.portfolio_optimizer:
                # Get recent returns for covariance estimation
                lookback_data = self.price_data.loc[:date].tail(63)  # 3 months
                returns_data = lookback_data.pct_change().dropna()
                
                if len(returns_data) < 20:
                    return None
                
                self.portfolio_optimizer.set_market_data(
                    returns=returns_data,
                    regime_probabilities=pd.DataFrame([regime_probs], 
                                                    columns=[f'regime_{i}' for i in range(len(regime_probs))])
                )
                
                # Optimize portfolio
                optimization_result = self.portfolio_optimizer.optimize_portfolio(
                    current_weights=current_weights
                )
                
                if optimization_result['optimization_status'] == 'optimal':
                    return optimization_result['weights']
            
            # Fallback: equal-weight or signal-based weights
            n_assets = len(self.price_data.columns)
            return np.ones(n_assets) / n_assets
            
        except Exception as e:
            self.logger.error(f"Error generating portfolio weights: {str(e)}")
            return None
    
    def _get_current_regime_probabilities(self, date: pd.Timestamp) -> np.ndarray:
        """Get current regime probabilities."""
        
        if self.regime_detector and hasattr(self.regime_detector, 'regime_probabilities'):
            # Get most recent regime probabilities
            if 'hmm' in self.regime_detector.regime_probabilities:
                regime_df = self.regime_detector.regime_probabilities['hmm']
                
                # Find closest date
                valid_dates = regime_df.index[regime_df.index <= date]
                if len(valid_dates) > 0:
                    latest_date = valid_dates[-1]
                    return regime_df.loc[latest_date].values
        
        # Default: equal probabilities
        return np.ones(3) / 3
    
    def _generate_expected_returns(self, date: pd.Timestamp, regime_probs: np.ndarray) -> Optional[pd.Series]:
        """Generate expected returns for assets."""
        
        try:
            # Get historical returns for estimation
            lookback_data = self.price_data.loc[:date].tail(252)  # 1 year
            returns_data = lookback_data.pct_change().dropna()
            
            if len(returns_data) < 50:
                return None
            
            # Simple expected return estimation
            if self.config.regime_aware and len(regime_probs) > 1:
                # Regime-weighted expected returns
                expected_returns = pd.Series(0.0, index=returns_data.columns)
                
                # For simplicity, use different lookback periods for different regimes
                for i, prob in enumerate(regime_probs):
                    if i == 0:  # Bull regime - shorter lookback
                        regime_returns = returns_data.tail(63).mean()
                    elif i == 1:  # Bear regime - medium lookback
                        regime_returns = returns_data.tail(126).mean()
                    else:  # Neutral regime - long lookback
                        regime_returns = returns_data.tail(252).mean()
                    
                    expected_returns += prob * regime_returns
            else:
                # Simple historical mean
                expected_returns = returns_data.mean()
            
            return expected_returns
            
        except Exception as e:
            self.logger.error(f"Error generating expected returns: {str(e)}")
            return None
    
    def _get_current_weights(self) -> np.ndarray:
        """Get current portfolio weights."""
        
        if not hasattr(self, 'positions') or not self.positions:
            return np.zeros(len(self.price_data.columns))
        
        # Calculate current market values
        current_prices = self.price_data.loc[self.current_date]
        total_value = 0
        
        for asset in self.price_data.columns:
            if asset in self.positions:
                total_value += self.positions[asset] * current_prices[asset]
        
        total_value += self.cash
        
        # Calculate weights
        weights = []
        for asset in self.price_data.columns:
            if asset in self.positions and total_value > 0:
                weight = (self.positions[asset] * current_prices[asset]) / total_value
            else:
                weight = 0.0
            weights.append(weight)
        
        return np.array(weights)
    
    def _execute_rebalancing(self, date: pd.Timestamp, target_weights: np.ndarray) -> Dict:
        """Execute portfolio rebalancing."""
        
        # Get current prices
        current_prices = self.price_data.loc[date]
        
        # Calculate current portfolio value
        portfolio_value = self.cash
        for asset in self.price_data.columns:
            if asset in self.positions:
                portfolio_value += self.positions[asset] * current_prices[asset]
        
        # Calculate target positions
        target_positions = {}
        total_transaction_cost = 0
        trades = []
        
        for i, asset in enumerate(self.price_data.columns):
            target_value = target_weights[i] * portfolio_value
            target_position = target_value / current_prices[asset]
            
            current_position = self.positions.get(asset, 0.0)
            trade_quantity = target_position - current_position
            
            if abs(trade_quantity) > 1e-6:  # Minimum trade size
                # Calculate transaction costs
                trade_value = abs(trade_quantity) * current_prices[asset]
                transaction_cost = trade_value * (
                    self.config.transaction_costs + 
                    self.config.market_impact + 
                    self.config.commission
                )
                
                total_transaction_cost += transaction_cost
                
                # Record trade
                trades.append({
                    'date': date,
                    'asset': asset,
                    'quantity': trade_quantity,
                    'price': current_prices[asset],
                    'value': trade_quantity * current_prices[asset],
                    'transaction_cost': transaction_cost
                })
                
                # Update position
                target_positions[asset] = target_position
            else:
                target_positions[asset] = current_position
        
        # Update portfolio state
        self.positions = target_positions
        self.cash = portfolio_value - sum(
            pos * current_prices[asset] for asset, pos in target_positions.items()
        ) - total_transaction_cost
        
        # Store trade history
        self.trade_history.extend(trades)
        
        return {
            'transaction_cost': total_transaction_cost,
            'trades': trades,
            'new_portfolio_value': portfolio_value - total_transaction_cost
        }
    
    def _calculate_daily_returns(self) -> pd.Series:
        """Calculate daily portfolio returns."""
        
        daily_returns = []
        daily_dates = []
        
        prev_value = self.config.initial_capital
        
        for date in self.price_data.index:
            if date < pd.to_datetime(self.config.start_date):
                continue
            
            # Calculate portfolio value on this date
            current_prices = self.price_data.loc[date]
            portfolio_value = self.cash
            
            for asset in self.price_data.columns:
                if asset in self.positions:
                    portfolio_value += self.positions[asset] * current_prices[asset]
            
            # Calculate return
            if prev_value > 0:
                daily_return = (portfolio_value - prev_value) / prev_value
            else:
                daily_return = 0.0
            
            daily_returns.append(daily_return)
            daily_dates.append(date)
            prev_value = portfolio_value
        
        return pd.Series(daily_returns, index=daily_dates)
    
    def _run_static_backtest(self) -> Dict:
        """Run static backtesting (no walk-forward)."""
        
        # Train models on full dataset
        train_data = {
            'prices': self.price_data,
            'volumes': self.volume_data,
            'fundamentals': self.fundamental_data,
            'macro': self.macro_data
        }
        
        start_date = pd.to_datetime(self.config.start_date)
        end_date = pd.to_datetime(self.config.end_date)
        
        self._retrain_models(train_data, start_date, end_date)
        
        # Run backtesting with fixed models
        return self._run_walk_forward_backtest()
    
    def _calculate_final_results(self, results: Dict) -> BacktestResults:
        """Calculate final comprehensive results."""
        
        # Portfolio returns
        portfolio_returns = results['portfolio_returns']
        
        # Benchmark returns
        benchmark_returns = self._get_benchmark_returns()
        
        # Portfolio weights DataFrame
        weights_data = []
        for weight_entry in results['portfolio_weights']:
            row = {'date': weight_entry['date']}
            for i, asset in enumerate(self.price_data.columns):
                row[asset] = weight_entry['weights'][i] if i < len(weight_entry['weights']) else 0
            weights_data.append(row)
        
        portfolio_weights = pd.DataFrame(weights_data).set_index('date') if weights_data else pd.DataFrame()
        
        # Transaction costs
        transaction_costs = pd.Series([
            tc['cost'] for tc in results['transaction_costs']
        ], index=[tc['date'] for tc in results['transaction_costs']])
        
        # Regime probabilities
        regime_data = []
        for regime_entry in results['regime_probabilities']:
            row = {'date': regime_entry['date']}
            for i, prob in enumerate(regime_entry['probabilities']):
                row[f'regime_{i}'] = prob
            regime_data.append(row)
        
        regime_probabilities = pd.DataFrame(regime_data).set_index('date') if regime_data else pd.DataFrame()
        
        # Performance metrics
        performance_metrics = self._calculate_performance_metrics(portfolio_returns, benchmark_returns)
        
        # Risk metrics
        risk_metrics = self._calculate_risk_metrics(portfolio_returns)
        
        # Drawdown analysis
        drawdown_analysis = self._calculate_drawdown_analysis(portfolio_returns)
        
        # Regime attribution
        regime_attribution = self._calculate_regime_attribution(
            portfolio_returns, regime_probabilities
        )
        
        # Positions and trades
        positions_df = self._create_positions_dataframe()
        trades_df = pd.DataFrame(self.trade_history)
        
        return BacktestResults(
            portfolio_returns=portfolio_returns,
            portfolio_weights=portfolio_weights,
            benchmark_returns=benchmark_returns,
            regime_probabilities=regime_probabilities,
            transaction_costs=transaction_costs,
            positions=positions_df,
            trades=trades_df,
            performance_metrics=performance_metrics,
            regime_attribution=regime_attribution,
            drawdown_analysis=drawdown_analysis,
            risk_metrics=risk_metrics
        )
    
    def _get_benchmark_returns(self) -> pd.Series:
        """Get benchmark returns."""
        
        if self.config.benchmark in self.price_data.columns:
            benchmark_prices = self.price_data[self.config.benchmark]
            return benchmark_prices.pct_change().dropna()
        else:
            # Create a simple equal-weighted benchmark
            equal_weight_returns = self.price_data.pct_change().mean(axis=1)
            return equal_weight_returns.dropna()
    
    def _calculate_performance_metrics(self, 
                                     portfolio_returns: pd.Series,
                                     benchmark_returns: pd.Series) -> Dict:
        """Calculate comprehensive performance metrics."""
        
        # Align returns
        common_index = portfolio_returns.index.intersection(benchmark_returns.index)
        port_ret = portfolio_returns.loc[common_index]
        bench_ret = benchmark_returns.loc[common_index]
        
        # Basic metrics
        total_return = (1 + port_ret).prod() - 1
        annualized_return = (1 + total_return) ** (252 / len(port_ret)) - 1
        annualized_volatility = port_ret.std() * np.sqrt(252)
        
        # Risk-adjusted metrics
        risk_free_rate = 0.02  # Assume 2%
        sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility
        
        # Sortino ratio
        downside_returns = port_ret[port_ret < 0]
        downside_volatility = downside_returns.std() * np.sqrt(252)
        sortino_ratio = (annualized_return - risk_free_rate) / downside_volatility if downside_volatility > 0 else 0
        
        # Calmar ratio
        max_drawdown = self._calculate_max_drawdown(port_ret)
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Information ratio vs benchmark
        excess_returns = port_ret - bench_ret
        tracking_error = excess_returns.std() * np.sqrt(252)
        information_ratio = excess_returns.mean() * np.sqrt(252) / tracking_error if tracking_error > 0 else 0
        
        # Alpha and Beta
        if len(common_index) > 20:
            from scipy import stats
            beta, alpha, r_value, p_value, std_err = stats.linregress(bench_ret, port_ret)
            alpha_annualized = alpha * 252
        else:
            beta = 1.0
            alpha_annualized = 0.0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'information_ratio': information_ratio,
            'tracking_error': tracking_error,
            'alpha': alpha_annualized,
            'beta': beta,
            'win_rate': (port_ret > 0).mean(),
            'best_day': port_ret.max(),
            'worst_day': port_ret.min(),
            'skewness': port_ret.skew(),
            'kurtosis': port_ret.kurtosis()
        }
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def _calculate_risk_metrics(self, portfolio_returns: pd.Series) -> Dict:
        """Calculate comprehensive risk metrics."""
        
        # VaR calculations
        var_95 = np.percentile(portfolio_returns, 5)
        var_99 = np.percentile(portfolio_returns, 1)
        
        # CVaR (Expected Shortfall)
        cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()
        cvar_99 = portfolio_returns[portfolio_returns <= var_99].mean()
        
        # Volatility metrics
        rolling_vol = portfolio_returns.rolling(window=21).std() * np.sqrt(252)
        vol_of_vol = rolling_vol.std()
        
        # Downside risk
        downside_deviation = portfolio_returns[portfolio_returns < 0].std() * np.sqrt(252)
        
        return {
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            'cvar_99': cvar_99,
            'volatility_of_volatility': vol_of_vol,
            'downside_deviation': downside_deviation,
            'max_daily_loss': portfolio_returns.min(),
            'max_daily_gain': portfolio_returns.max()
        }
    
    def _calculate_drawdown_analysis(self, portfolio_returns: pd.Series) -> Dict:
        """Calculate detailed drawdown analysis."""
        
        cumulative = (1 + portfolio_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        # Find drawdown periods
        in_drawdown = drawdown < 0
        drawdown_starts = in_drawdown & ~in_drawdown.shift(1)
        drawdown_ends = ~in_drawdown & in_drawdown.shift(1)
        
        drawdown_periods = []
        start_dates = drawdown_starts[drawdown_starts].index
        end_dates = drawdown_ends[drawdown_ends].index
        
        for i, start_date in enumerate(start_dates):
            # Find corresponding end date
            end_candidates = end_dates[end_dates > start_date]
            if len(end_candidates) > 0:
                end_date = end_candidates[0]
            else:
                end_date = drawdown.index[-1]  # Ongoing drawdown
            
            period_drawdown = drawdown.loc[start_date:end_date]
            max_dd = period_drawdown.min()
            duration = (end_date - start_date).days
            
            drawdown_periods.append({
                'start_date': start_date,
                'end_date': end_date,
                'max_drawdown': max_dd,
                'duration_days': duration
            })
        
        return {
            'max_drawdown': drawdown.min(),
            'drawdown_periods': drawdown_periods,
            'avg_drawdown_duration': np.mean([dd['duration_days'] for dd in drawdown_periods]) if drawdown_periods else 0,
            'current_drawdown': drawdown.iloc[-1],
            'time_underwater_pct': (drawdown < 0).mean()
        }
    
    def _calculate_regime_attribution(self, 
                                    portfolio_returns: pd.Series,
                                    regime_probabilities: pd.DataFrame) -> Dict:
        """Calculate performance attribution by regime."""
        
        if regime_probabilities.empty:
            return {}
        
        # Align data
        common_index = portfolio_returns.index.intersection(regime_probabilities.index)
        
        if len(common_index) == 0:
            return {}
        
        aligned_returns = portfolio_returns.loc[common_index]
        aligned_regimes = regime_probabilities.loc[common_index]
        
        regime_attribution = {}
        
        for regime_col in aligned_regimes.columns:
            regime_weights = aligned_regimes[regime_col]
            
            # Weighted returns for this regime
            weighted_returns = aligned_returns * regime_weights
            
            # Performance metrics for this regime
            regime_return = weighted_returns.sum() / regime_weights.sum() if regime_weights.sum() > 0 else 0
            regime_vol = weighted_returns.std() * np.sqrt(252)
            
            regime_attribution[regime_col] = {
                'contribution_to_return': weighted_returns.sum(),
                'average_return': regime_return,
                'volatility': regime_vol,
                'weight': regime_weights.mean(),
                'sharpe_ratio': regime_return / regime_vol if regime_vol > 0 else 0
            }
        
        return regime_attribution
    
    def _create_positions_dataframe(self) -> pd.DataFrame:
        """Create positions DataFrame from portfolio history."""
        
        # This would be implemented based on stored position history
        # For now, return empty DataFrame
        return pd.DataFrame()
    
    def monte_carlo_simulation(self, 
                             n_simulations: int = 1000,
                             confidence_levels: List[float] = [0.05, 0.95]) -> Dict:
        """
        Run Monte Carlo simulation for robustness testing.
        
        Args:
            n_simulations: Number of Monte Carlo simulations
            confidence_levels: Confidence levels for analysis
        
        Returns:
            Monte Carlo simulation results
        """
        
        self.logger.info(f"Running Monte Carlo simulation with {n_simulations} iterations")
        
        # Get base portfolio returns
        base_returns = self.daily_returns
        
        if len(base_returns) == 0:
            return {}
        
        # Bootstrap simulation
        simulation_results = []
        
        for sim in range(n_simulations):
            # Bootstrap sample of returns
            bootstrap_returns = np.random.choice(base_returns, size=len(base_returns), replace=True)
            
            # Calculate simulation metrics
            total_return = (1 + bootstrap_returns).prod() - 1
            volatility = np.std(bootstrap_returns) * np.sqrt(252)
            max_dd = self._calculate_max_drawdown(pd.Series(bootstrap_returns))
            sharpe = (np.mean(bootstrap_returns) * 252 - 0.02) / volatility
            
            simulation_results.append({
                'total_return': total_return,
                'volatility': volatility,
                'max_drawdown': max_dd,
                'sharpe_ratio': sharpe
            })
        
        # Calculate confidence intervals
        results_df = pd.DataFrame(simulation_results)
        
        confidence_intervals = {}
        for level in confidence_levels:
            confidence_intervals[f'{level:.0%}'] = {
                'total_return': results_df['total_return'].quantile(level),
                'volatility': results_df['volatility'].quantile(level),
                'max_drawdown': results_df['max_drawdown'].quantile(level),
                'sharpe_ratio': results_df['sharpe_ratio'].quantile(level)
            }
        
        return {
            'simulation_results': results_df,
            'confidence_intervals': confidence_intervals,
            'mean_results': results_df.mean().to_dict(),
            'std_results': results_df.std().to_dict(),
            'n_simulations': n_simulations
        }
    
    def save_results(self, results: BacktestResults, filepath: str):
        """Save backtest results to file."""
        
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)
        
        self.logger.info(f"Backtest results saved to {filepath}")
    
    def load_results(self, filepath: str) -> BacktestResults:
        """Load backtest results from file."""
        
        with open(filepath, 'rb') as f:
            results = pickle.load(f)
        
        self.logger.info(f"Backtest results loaded from {filepath}")
        return results