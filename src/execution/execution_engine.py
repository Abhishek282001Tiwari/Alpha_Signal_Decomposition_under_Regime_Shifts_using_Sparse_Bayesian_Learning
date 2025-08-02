import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
import logging
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

warnings.filterwarnings('ignore')

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    TWAP = "twap"
    VWAP = "vwap"

class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"

@dataclass
class Order:
    """Order data structure."""
    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType
    price: Optional[float] = None
    time_in_force: str = "DAY"
    order_id: Optional[str] = None
    parent_order_id: Optional[str] = None
    created_time: Optional[datetime] = None
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    status: str = "NEW"

@dataclass
class Fill:
    """Trade fill data structure."""
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    timestamp: datetime
    commission: float = 0.0
    
class ExecutionEngine:
    """
    Optimal trade execution engine with regime-aware algorithms,
    market impact modeling, and intelligent order scheduling.
    """
    
    def __init__(self, 
                 execution_algorithm: str = 'twap',
                 market_impact_model: str = 'linear',
                 commission_rate: float = 0.001,
                 max_participation_rate: float = 0.20,
                 regime_aware: bool = True):
        """
        Initialize Execution Engine.
        
        Args:
            execution_algorithm: Default execution algorithm ('twap', 'vwap', 'implementation_shortfall')
            market_impact_model: Market impact model ('linear', 'sqrt', 'regime_dependent')
            commission_rate: Commission rate as fraction of notional
            max_participation_rate: Maximum participation rate in daily volume
            regime_aware: Whether to use regime-dependent execution
        """
        self.execution_algorithm = execution_algorithm
        self.market_impact_model = market_impact_model
        self.commission_rate = commission_rate
        self.max_participation_rate = max_participation_rate
        self.regime_aware = regime_aware
        
        # Market data
        self.market_data = {}
        self.volume_data = {}
        self.regime_probabilities = None
        
        # Execution state
        self.pending_orders = {}
        self.executed_orders = {}
        self.fills = []
        self.order_counter = 0
        
        # Market impact parameters
        self.impact_coefficients = {}
        self.regime_impact_models = {}
        
        # Performance tracking
        self.execution_costs = {}
        self.slippage_tracking = {}
        
        self.logger = logging.getLogger(__name__)
        
    def set_market_data(self, 
                       price_data: pd.DataFrame,
                       volume_data: pd.DataFrame,
                       regime_probabilities: Optional[pd.DataFrame] = None):
        """
        Set market data for execution.
        
        Args:
            price_data: Price data DataFrame
            volume_data: Volume data DataFrame
            regime_probabilities: Optional regime probabilities
        """
        self.market_data = price_data
        self.volume_data = volume_data
        self.regime_probabilities = regime_probabilities
        
        # Calibrate market impact models
        self._calibrate_market_impact_models()
        
        if self.regime_aware and regime_probabilities is not None:
            self._calibrate_regime_impact_models()
    
    def _calibrate_market_impact_models(self):
        """Calibrate market impact models for each asset."""
        
        for symbol in self.market_data.columns:
            if symbol in self.volume_data.columns:
                returns = self.market_data[symbol].pct_change().dropna()
                volumes = self.volume_data[symbol].dropna()
                
                # Align data
                common_index = returns.index.intersection(volumes.index)
                if len(common_index) > 100:
                    aligned_returns = returns.loc[common_index]
                    aligned_volumes = volumes.loc[common_index]
                    
                    # Calibrate impact coefficient
                    impact_coeff = self._estimate_impact_coefficient(aligned_returns, aligned_volumes)
                    self.impact_coefficients[symbol] = impact_coeff
                else:
                    # Default impact coefficient
                    self.impact_coefficients[symbol] = 0.1
    
    def _estimate_impact_coefficient(self, returns: pd.Series, volumes: pd.Series) -> float:
        """Estimate market impact coefficient using regression."""
        
        # Market impact model: |return| = alpha * (volume / avg_volume)^beta + epsilon
        avg_volume = volumes.rolling(window=20).mean()
        volume_ratio = volumes / avg_volume
        
        # Remove outliers and align
        abs_returns = np.abs(returns)
        valid_mask = (volume_ratio > 0) & (abs_returns < abs_returns.quantile(0.95))
        
        if valid_mask.sum() > 50:
            y = np.log(abs_returns[valid_mask] + 1e-6)
            x = np.log(volume_ratio[valid_mask])
            
            # Simple linear regression
            from sklearn.linear_model import LinearRegression
            reg = LinearRegression()
            reg.fit(x.values.reshape(-1, 1), y.values)
            
            return np.exp(reg.intercept_)  # Convert back from log space
        else:
            return 0.1  # Default coefficient
    
    def _calibrate_regime_impact_models(self):
        """Calibrate regime-dependent market impact models."""
        
        if self.regime_probabilities is None:
            return
        
        n_regimes = self.regime_probabilities.shape[1]
        
        for regime in range(n_regimes):
            regime_weights = self.regime_probabilities.iloc[:, regime]
            regime_coefficients = {}
            
            for symbol in self.market_data.columns:
                if symbol in self.volume_data.columns:
                    returns = self.market_data[symbol].pct_change().dropna()
                    volumes = self.volume_data[symbol].dropna()
                    
                    # Align with regime weights
                    common_index = returns.index.intersection(volumes.index).intersection(regime_weights.index)
                    
                    if len(common_index) > 50:
                        aligned_returns = returns.loc[common_index]
                        aligned_volumes = volumes.loc[common_index]
                        aligned_weights = regime_weights.loc[common_index]
                        
                        # Weighted impact estimation
                        impact_coeff = self._estimate_weighted_impact_coefficient(
                            aligned_returns, aligned_volumes, aligned_weights
                        )
                        regime_coefficients[symbol] = impact_coeff
            
            self.regime_impact_models[regime] = regime_coefficients
    
    def _estimate_weighted_impact_coefficient(self, 
                                            returns: pd.Series, 
                                            volumes: pd.Series,
                                            weights: pd.Series) -> float:
        """Estimate weighted market impact coefficient."""
        
        # Weighted regression for regime-specific impact
        avg_volume = volumes.rolling(window=20).mean()
        volume_ratio = volumes / avg_volume
        abs_returns = np.abs(returns)
        
        # Filter valid observations
        valid_mask = (volume_ratio > 0) & (abs_returns < abs_returns.quantile(0.95)) & (weights > 0.1)
        
        if valid_mask.sum() > 20:
            y = np.log(abs_returns[valid_mask] + 1e-6)
            x = np.log(volume_ratio[valid_mask])
            w = weights[valid_mask]
            
            # Weighted least squares
            from sklearn.linear_model import LinearRegression
            reg = LinearRegression()
            reg.fit(x.values.reshape(-1, 1), y.values, sample_weight=w.values)
            
            return np.exp(reg.intercept_)
        else:
            return 0.1  # Default
    
    def create_order(self, 
                    symbol: str,
                    side: OrderSide,
                    quantity: float,
                    order_type: OrderType = OrderType.MARKET,
                    price: Optional[float] = None,
                    execution_params: Optional[Dict] = None) -> str:
        """
        Create a new order.
        
        Args:
            symbol: Asset symbol
            side: Order side (BUY/SELL)
            quantity: Order quantity
            order_type: Order type
            price: Limit price (if applicable)
            execution_params: Additional execution parameters
        
        Returns:
            Order ID
        """
        self.order_counter += 1
        order_id = f"ORD_{self.order_counter:06d}"
        
        order = Order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=order_type,
            price=price,
            order_id=order_id,
            created_time=datetime.now()
        )
        
        self.pending_orders[order_id] = order
        
        # Schedule execution based on order type
        if order_type in [OrderType.TWAP, OrderType.VWAP]:
            execution_schedule = self._create_execution_schedule(order, execution_params)
            order.execution_schedule = execution_schedule
        
        self.logger.info(f"Created order {order_id}: {side.value} {quantity} {symbol}")
        return order_id
    
    def _create_execution_schedule(self, order: Order, params: Optional[Dict] = None) -> List[Dict]:
        """Create execution schedule for algorithmic orders."""
        
        if params is None:
            params = {}
        
        duration_hours = params.get('duration_hours', 4)  # Default 4-hour execution
        num_slices = params.get('num_slices', 20)  # Default 20 slices
        
        # Get current regime information
        current_regime = self._get_current_regime()
        
        # Adjust execution based on regime
        if self.regime_aware and current_regime is not None:
            regime_adjustment = self._get_regime_execution_adjustment(current_regime)
            duration_hours *= regime_adjustment.get('duration_multiplier', 1.0)
            num_slices = int(num_slices * regime_adjustment.get('slice_multiplier', 1.0))
        
        slice_size = order.quantity / num_slices
        slice_interval = timedelta(hours=duration_hours / num_slices)
        
        execution_schedule = []
        current_time = datetime.now()
        
        for i in range(num_slices):
            # Time-weighted slice sizing for TWAP
            if order.order_type == OrderType.TWAP:
                weight = 1.0  # Equal weighting for TWAP
            elif order.order_type == OrderType.VWAP:
                # Volume-weighted slice sizing
                weight = self._get_volume_weight(order.symbol, current_time + i * slice_interval)
            else:
                weight = 1.0
            
            # Adjust slice size based on market conditions
            adjusted_slice_size = slice_size * weight
            
            # Don't exceed participation rate
            max_slice_size = self._calculate_max_slice_size(order.symbol, current_time + i * slice_interval)
            adjusted_slice_size = min(adjusted_slice_size, max_slice_size)
            
            execution_schedule.append({
                'slice_id': i + 1,
                'scheduled_time': current_time + i * slice_interval,
                'quantity': adjusted_slice_size,
                'weight': weight,
                'max_participation': max_slice_size
            })
        
        return execution_schedule
    
    def _get_current_regime(self) -> Optional[int]:
        """Get current market regime."""
        
        if self.regime_probabilities is None or self.regime_probabilities.empty:
            return None
        
        # Most recent regime probabilities
        latest_probs = self.regime_probabilities.iloc[-1]
        return latest_probs.idxmax()
    
    def _get_regime_execution_adjustment(self, regime: int) -> Dict:
        """Get execution adjustments based on regime."""
        
        # Regime-dependent execution parameters
        if regime == 0:  # Bull market regime
            return {
                'duration_multiplier': 0.8,  # Faster execution
                'slice_multiplier': 1.2,     # More slices
                'aggression_factor': 1.1     # More aggressive
            }
        elif regime == 1:  # Bear market regime
            return {
                'duration_multiplier': 1.5,  # Slower execution
                'slice_multiplier': 0.8,     # Fewer slices
                'aggression_factor': 0.9     # Less aggressive
            }
        else:  # Neutral/volatile regime
            return {
                'duration_multiplier': 1.0,  # Normal execution
                'slice_multiplier': 1.0,     # Normal slicing
                'aggression_factor': 1.0     # Normal aggression
            }
    
    def _get_volume_weight(self, symbol: str, timestamp: datetime) -> float:
        """Get volume weight for VWAP execution."""
        
        if symbol not in self.volume_data.columns:
            return 1.0
        
        # Get historical intraday volume pattern
        hour = timestamp.hour
        
        # Simple intraday volume pattern (U-shaped)
        if 9 <= hour <= 10 or 15 <= hour <= 16:  # Market open/close
            return 1.5
        elif 11 <= hour <= 14:  # Midday
            return 0.7
        else:
            return 1.0
    
    def _calculate_max_slice_size(self, symbol: str, timestamp: datetime) -> float:
        """Calculate maximum slice size based on participation rate."""
        
        if symbol not in self.volume_data.columns:
            return float('inf')
        
        # Get recent average daily volume
        recent_volumes = self.volume_data[symbol].tail(20)
        avg_daily_volume = recent_volumes.mean()
        
        # Estimated hourly volume (assuming 6.5 trading hours)
        avg_hourly_volume = avg_daily_volume / 6.5
        
        # Maximum participation
        max_slice_size = avg_hourly_volume * self.max_participation_rate
        
        return max_slice_size
    
    def execute_orders(self, current_time: datetime = None) -> List[Fill]:
        """Execute pending orders based on their schedules."""
        
        if current_time is None:
            current_time = datetime.now()
        
        fills = []
        orders_to_remove = []
        
        for order_id, order in self.pending_orders.items():
            
            if order.order_type == OrderType.MARKET:
                # Execute market order immediately
                fill = self._execute_market_order(order, current_time)
                if fill:
                    fills.append(fill)
                    orders_to_remove.append(order_id)
            
            elif order.order_type in [OrderType.TWAP, OrderType.VWAP]:
                # Execute algorithmic order slices
                if hasattr(order, 'execution_schedule'):
                    slice_fills = self._execute_algorithmic_order(order, current_time)
                    fills.extend(slice_fills)
                    
                    # Check if order is fully executed
                    if order.filled_quantity >= order.quantity:
                        orders_to_remove.append(order_id)
        
        # Remove completed orders
        for order_id in orders_to_remove:
            self.executed_orders[order_id] = self.pending_orders.pop(order_id)
        
        # Update fills list
        self.fills.extend(fills)
        
        return fills
    
    def _execute_market_order(self, order: Order, current_time: datetime) -> Optional[Fill]:
        """Execute a market order."""
        
        # Get current market price
        market_price = self._get_market_price(order.symbol, current_time)
        if market_price is None:
            return None
        
        # Calculate market impact
        impact = self._calculate_market_impact(order.symbol, order.quantity, current_time)
        
        # Execution price includes impact
        if order.side == OrderSide.BUY:
            execution_price = market_price * (1 + impact)
        else:
            execution_price = market_price * (1 - impact)
        
        # Create fill
        fill = Fill(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=execution_price,
            timestamp=current_time,
            commission=order.quantity * execution_price * self.commission_rate
        )
        
        # Update order
        order.filled_quantity = order.quantity
        order.avg_fill_price = execution_price
        order.status = "FILLED"
        
        return fill
    
    def _execute_algorithmic_order(self, order: Order, current_time: datetime) -> List[Fill]:
        """Execute slices of an algorithmic order."""
        
        fills = []
        
        if not hasattr(order, 'execution_schedule'):
            return fills
        
        for slice_info in order.execution_schedule:
            slice_time = slice_info['scheduled_time']
            
            # Check if it's time to execute this slice
            if slice_time <= current_time and slice_info.get('executed', False) is False:
                
                slice_quantity = slice_info['quantity']
                remaining_quantity = order.quantity - order.filled_quantity
                
                # Don't over-execute
                actual_quantity = min(slice_quantity, remaining_quantity)
                
                if actual_quantity > 0:
                    # Execute slice
                    market_price = self._get_market_price(order.symbol, current_time)
                    if market_price is not None:
                        
                        # Calculate impact for this slice
                        impact = self._calculate_market_impact(order.symbol, actual_quantity, current_time)
                        
                        # Execution price
                        if order.side == OrderSide.BUY:
                            execution_price = market_price * (1 + impact)
                        else:
                            execution_price = market_price * (1 - impact)
                        
                        # Create fill
                        fill = Fill(
                            order_id=order.order_id,
                            symbol=order.symbol,
                            side=order.side,
                            quantity=actual_quantity,
                            price=execution_price,
                            timestamp=current_time,
                            commission=actual_quantity * execution_price * self.commission_rate
                        )
                        
                        fills.append(fill)
                        
                        # Update order
                        order.filled_quantity += actual_quantity
                        order.avg_fill_price = (
                            (order.avg_fill_price * (order.filled_quantity - actual_quantity) + 
                             execution_price * actual_quantity) / order.filled_quantity
                        )
                        
                        # Mark slice as executed
                        slice_info['executed'] = True
                        slice_info['actual_quantity'] = actual_quantity
                        slice_info['execution_price'] = execution_price
        
        # Update order status
        if order.filled_quantity >= order.quantity:
            order.status = "FILLED"
        elif order.filled_quantity > 0:
            order.status = "PARTIALLY_FILLED"
        
        return fills
    
    def _get_market_price(self, symbol: str, timestamp: datetime) -> Optional[float]:
        """Get market price for symbol at timestamp."""
        
        if symbol not in self.market_data.columns:
            return None
        
        # Get the most recent price
        price_series = self.market_data[symbol].dropna()
        
        if price_series.empty:
            return None
        
        # Find closest price to timestamp
        time_diff = abs(price_series.index - timestamp)
        closest_idx = time_diff.idxmin()
        
        return price_series.loc[closest_idx]
    
    def _calculate_market_impact(self, 
                               symbol: str, 
                               quantity: float, 
                               timestamp: datetime) -> float:
        """Calculate market impact for trade."""
        
        if symbol not in self.impact_coefficients:
            return 0.001  # Default impact
        
        # Get average daily volume
        if symbol in self.volume_data.columns:
            recent_volume = self.volume_data[symbol].tail(20).mean()
        else:
            recent_volume = 1e6  # Default volume
        
        # Participation rate
        participation_rate = quantity / recent_volume
        
        # Base impact coefficient
        base_impact = self.impact_coefficients[symbol]
        
        # Regime adjustment
        if self.regime_aware:
            current_regime = self._get_current_regime()
            if (current_regime is not None and 
                current_regime in self.regime_impact_models and
                symbol in self.regime_impact_models[current_regime]):
                
                regime_impact = self.regime_impact_models[current_regime][symbol]
                base_impact = regime_impact
        
        # Impact models
        if self.market_impact_model == 'linear':
            impact = base_impact * participation_rate
        elif self.market_impact_model == 'sqrt':
            impact = base_impact * np.sqrt(participation_rate)
        else:
            impact = base_impact * participation_rate
        
        return min(impact, 0.05)  # Cap impact at 5%
    
    def calculate_execution_costs(self, order_id: str) -> Dict:
        """Calculate execution costs for completed order."""
        
        if order_id not in self.executed_orders:
            return {}
        
        order = self.executed_orders[order_id]
        order_fills = [f for f in self.fills if f.order_id == order_id]
        
        if not order_fills:
            return {}
        
        # Calculate VWAP
        total_quantity = sum(f.quantity for f in order_fills)
        vwap = sum(f.price * f.quantity for f in order_fills) / total_quantity
        
        # Benchmark price (arrival price)
        arrival_price = self._get_market_price(order.symbol, order.created_time)
        
        if arrival_price is None:
            return {}
        
        # Implementation shortfall
        if order.side == OrderSide.BUY:
            implementation_shortfall = (vwap - arrival_price) / arrival_price
        else:
            implementation_shortfall = (arrival_price - vwap) / arrival_price
        
        # Commission costs
        total_commission = sum(f.commission for f in order_fills)
        commission_bps = (total_commission / (total_quantity * vwap)) * 10000
        
        # Market impact
        market_impact_bps = implementation_shortfall * 10000
        
        # Timing risk (price movement during execution)
        if hasattr(order, 'execution_schedule') and order.execution_schedule:
            completion_time = max(slice_info['scheduled_time'] for slice_info in order.execution_schedule)
            completion_price = self._get_market_price(order.symbol, completion_time)
            
            if completion_price is not None:
                if order.side == OrderSide.BUY:
                    timing_risk = (completion_price - arrival_price) / arrival_price
                else:
                    timing_risk = (arrival_price - completion_price) / arrival_price
                
                timing_risk_bps = timing_risk * 10000
            else:
                timing_risk_bps = 0
        else:
            timing_risk_bps = 0
        
        return {
            'order_id': order_id,
            'symbol': order.symbol,
            'total_quantity': total_quantity,
            'vwap': vwap,
            'arrival_price': arrival_price,
            'implementation_shortfall_bps': market_impact_bps,
            'commission_bps': commission_bps,
            'timing_risk_bps': timing_risk_bps,
            'total_cost_bps': market_impact_bps + commission_bps,
            'fills': len(order_fills)
        }
    
    def get_execution_summary(self) -> Dict:
        """Get comprehensive execution summary."""
        
        total_orders = len(self.executed_orders) + len(self.pending_orders)
        completed_orders = len(self.executed_orders)
        total_fills = len(self.fills)
        
        # Calculate average execution costs
        execution_costs = []
        for order_id in self.executed_orders:
            cost_metrics = self.calculate_execution_costs(order_id)
            if cost_metrics:
                execution_costs.append(cost_metrics)
        
        if execution_costs:
            avg_implementation_shortfall = np.mean([c['implementation_shortfall_bps'] for c in execution_costs])
            avg_commission = np.mean([c['commission_bps'] for c in execution_costs])
            avg_total_cost = np.mean([c['total_cost_bps'] for c in execution_costs])
        else:
            avg_implementation_shortfall = 0
            avg_commission = 0
            avg_total_cost = 0
        
        return {
            'total_orders': total_orders,
            'completed_orders': completed_orders,
            'pending_orders': len(self.pending_orders),
            'total_fills': total_fills,
            'execution_algorithm': self.execution_algorithm,
            'market_impact_model': self.market_impact_model,
            'regime_aware': self.regime_aware,
            'avg_implementation_shortfall_bps': avg_implementation_shortfall,
            'avg_commission_bps': avg_commission,
            'avg_total_cost_bps': avg_total_cost,
            'max_participation_rate': self.max_participation_rate,
            'execution_costs_detail': execution_costs
        }
    
    def optimize_execution_schedule(self, 
                                  symbol: str,
                                  quantity: float,
                                  side: OrderSide,
                                  target_duration_hours: float = 4) -> Dict:
        """
        Optimize execution schedule using implementation shortfall framework.
        
        Args:
            symbol: Asset symbol
            quantity: Total quantity to execute
            side: Order side
            target_duration_hours: Target execution duration
        
        Returns:
            Optimized execution schedule
        """
        
        # Get market parameters
        if symbol in self.volume_data.columns:
            avg_daily_volume = self.volume_data[symbol].tail(20).mean()
            volatility = self.market_data[symbol].pct_change().tail(252).std() * np.sqrt(252)
        else:
            avg_daily_volume = 1e6
            volatility = 0.20
        
        # Impact coefficient
        impact_coeff = self.impact_coefficients.get(symbol, 0.1)
        
        # Optimization: minimize expected implementation shortfall
        # IS = Market Impact + Timing Risk
        # Market Impact = impact_coeff * (quantity / volume) * participation_rate
        # Timing Risk = volatility * sqrt(duration) * remaining_quantity_ratio
        
        from scipy.optimize import minimize_scalar
        
        def implementation_shortfall(duration_hours):
            participation_rate = quantity / (avg_daily_volume * duration_hours / 6.5)
            
            # Market impact cost
            market_impact = impact_coeff * participation_rate
            
            # Timing risk cost
            timing_risk = volatility * np.sqrt(duration_hours / (6.5 * 252)) * 0.5  # Average remaining quantity
            
            return market_impact + timing_risk
        
        # Optimize duration
        result = minimize_scalar(
            implementation_shortfall,
            bounds=(0.5, 8.0),  # Between 30 minutes and 8 hours
            method='bounded'
        )
        
        optimal_duration = result.x
        optimal_cost = result.fun
        
        # Create optimal schedule
        num_slices = max(5, int(optimal_duration * 4))  # At least 5 slices, 4 per hour
        
        optimal_schedule = {
            'optimal_duration_hours': optimal_duration,
            'expected_cost_bps': optimal_cost * 10000,
            'num_slices': num_slices,
            'avg_slice_size': quantity / num_slices,
            'recommended_participation_rate': quantity / (avg_daily_volume * optimal_duration / 6.5),
            'market_impact_component': impact_coeff * quantity / (avg_daily_volume * optimal_duration / 6.5),
            'timing_risk_component': volatility * np.sqrt(optimal_duration / (6.5 * 252)) * 0.5
        }
        
        return optimal_schedule