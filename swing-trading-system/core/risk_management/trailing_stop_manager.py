#!/usr/bin/env python3
"""
Trailing Stop Manager
Implements multiple trailing stop strategies for profit protection
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from typing import Dict, List, Tuple, Optional

class TrailingStopManager:
    """
    Manages various trailing stop strategies:
    - ATR-based trailing stops
    - Percentage-based trailing stops
    - Parabolic SAR
    - Chandelier Exit
    - Dynamic trailing based on profit levels
    """
    
    def __init__(self):
        # Internal methods that return dicts
        self._stop_strategies_full = {
            'atr_trail': self._calculate_atr_trailing_stop_full,
            'percentage_trail': self._calculate_percentage_trailing_stop_full,
            'parabolic_sar': self._calculate_parabolic_sar_full,
            'chandelier_exit': self._calculate_chandelier_exit_full,
            'dynamic_trail': self._calculate_dynamic_trailing_stop_full
        }
        
        # For backward compatibility - return floats
        self.stop_strategies = {
            'atr_trail': self.calculate_atr_trailing_stop,
            'percentage_trail': self.calculate_percentage_trailing_stop,
            'parabolic_sar': self.calculate_parabolic_sar,
            'chandelier_exit': self.calculate_chandelier_exit,
            'dynamic_trail': self.calculate_dynamic_trailing_stop
        }
        
        # Default parameters
        self.default_params = {
            'atr_multiplier': 2.0,
            'percentage': 0.05,  # 5% trailing stop
            'sar_acceleration': 0.02,
            'sar_maximum': 0.2,
            'chandelier_period': 22,
            'chandelier_multiplier': 3.0
        }
        
    def calculate_optimal_trailing_stop(self, data: pd.DataFrame, entry_price: float, 
                                      position_type: str = 'long', market_regime: str = 'NEUTRAL') -> Dict:
        """
        Calculate optimal trailing stop based on market conditions
        """
        results = {}
        
        # Calculate all trailing stop types
        for strategy_name, strategy_func in self._stop_strategies_full.items():
            try:
                stop_levels = strategy_func(data, entry_price, position_type)
                results[strategy_name] = stop_levels
            except Exception as e:
                print(f"Error in {strategy_name}: {str(e)}")
                results[strategy_name] = None
        
        # Select optimal strategy based on market regime
        optimal_strategy = self._select_optimal_strategy(results, market_regime, data)
        
        # Calculate stop loss metrics
        current_price = data['Close'].iloc[-1]
        stop_distance = abs(current_price - optimal_strategy['current_stop'])
        stop_percentage = (stop_distance / current_price) * 100
        
        return {
            'optimal_strategy': optimal_strategy['strategy'],
            'current_stop': optimal_strategy['current_stop'],
            'stop_distance': stop_distance,
            'stop_percentage': stop_percentage,
            'all_strategies': results,
            'recommendation': self._generate_stop_recommendation(optimal_strategy, market_regime)
        }
    
    def calculate_atr_trailing_stop(self, data: pd.DataFrame, entry_price: float, 
                                   position_type: str = 'long') -> float:
        """
        ATR-based trailing stop that adjusts with volatility
        Returns stop price for compatibility with tests
        """
        result = self._calculate_atr_trailing_stop_full(data, entry_price, position_type)
        return result['current_stop']
    
    def _calculate_atr_trailing_stop_full(self, data: pd.DataFrame, entry_price: float, 
                                   position_type: str = 'long') -> Dict:
        """
        ATR-based trailing stop that adjusts with volatility
        """
        # Calculate ATR
        atr = self._calculate_atr(data)
        current_atr = atr.iloc[-1]
        
        # Calculate trailing stop levels
        stops = []
        highest_close = entry_price
        lowest_close = entry_price
        
        for i in range(len(data)):
            if position_type == 'long':
                highest_close = max(highest_close, data['Close'].iloc[i])
                stop = highest_close - (self.default_params['atr_multiplier'] * atr.iloc[i])
                stops.append(stop)
            else:  # short
                lowest_close = min(lowest_close, data['Close'].iloc[i])
                stop = lowest_close + (self.default_params['atr_multiplier'] * atr.iloc[i])
                stops.append(stop)
        
        # Adjust multiplier based on profit
        current_price = data['Close'].iloc[-1]
        profit_percentage = ((current_price - entry_price) / entry_price) * 100 if position_type == 'long' else ((entry_price - current_price) / entry_price) * 100
        
        # Tighten stop as profit increases
        adjusted_multiplier = self._adjust_atr_multiplier(profit_percentage)
        current_stop = stops[-1]
        
        if position_type == 'long':
            adjusted_stop = data['High'].tail(10).max() - (adjusted_multiplier * current_atr)
            current_stop = max(current_stop, adjusted_stop)  # Never lower the stop
        else:
            adjusted_stop = data['Low'].tail(10).min() + (adjusted_multiplier * current_atr)
            current_stop = min(current_stop, adjusted_stop)  # Never raise the stop for shorts
        
        return {
            'strategy': 'atr_trail',
            'current_stop': current_stop,
            'atr_value': current_atr,
            'multiplier': adjusted_multiplier,
            'stop_series': stops,
            'profit_percentage': profit_percentage
        }
    
    def calculate_percentage_trailing_stop(self, data: pd.DataFrame, entry_price: float, 
                                         position_type: str = 'long') -> float:
        """
        Simple percentage-based trailing stop
        Returns stop price for compatibility with tests
        """
        result = self._calculate_percentage_trailing_stop_full(data, entry_price, position_type)
        return result['current_stop']
    
    def _calculate_percentage_trailing_stop_full(self, data: pd.DataFrame, entry_price: float, 
                                         position_type: str = 'long') -> Dict:
        """
        Simple percentage-based trailing stop
        """
        percentage = self.default_params['percentage']
        stops = []
        
        if position_type == 'long':
            highest_price = entry_price
            for i in range(len(data)):
                highest_price = max(highest_price, data['High'].iloc[i])
                stop = highest_price * (1 - percentage)
                stops.append(stop)
        else:  # short
            lowest_price = entry_price
            for i in range(len(data)):
                lowest_price = min(lowest_price, data['Low'].iloc[i])
                stop = lowest_price * (1 + percentage)
                stops.append(stop)
        
        current_stop = stops[-1]
        
        # Dynamic percentage based on volatility
        volatility = data['Close'].pct_change().rolling(20).std().iloc[-1]
        if volatility > 0.03:  # High volatility
            percentage = 0.07  # Wider stop
        elif volatility < 0.01:  # Low volatility
            percentage = 0.03  # Tighter stop
        
        return {
            'strategy': 'percentage_trail',
            'current_stop': current_stop,
            'percentage': percentage,
            'stop_series': stops,
            'volatility': volatility
        }
    
    def calculate_parabolic_sar(self, data: pd.DataFrame, entry_price: float, 
                               position_type: str = 'long') -> float:
        """
        Parabolic SAR trailing stop
        Returns stop price for compatibility with tests
        """
        result = self._calculate_parabolic_sar_full(data, entry_price, position_type)
        return result['current_stop']
    
    def _calculate_parabolic_sar_full(self, data: pd.DataFrame, entry_price: float, 
                               position_type: str = 'long') -> Dict:
        """
        Parabolic SAR trailing stop
        """
        high = data['High']
        low = data['Low']
        close = data['Close']
        
        # Initialize
        af = self.default_params['sar_acceleration']
        max_af = self.default_params['sar_maximum']
        
        # Initialize arrays
        sar = np.zeros(len(data))
        ep = np.zeros(len(data))  # Extreme point
        af_values = np.zeros(len(data))
        trend = np.zeros(len(data))  # 1 for long, -1 for short
        
        # Starting values
        sar[0] = low.iloc[0] if position_type == 'long' else high.iloc[0]
        ep[0] = high.iloc[0] if position_type == 'long' else low.iloc[0]
        af_values[0] = af
        trend[0] = 1 if position_type == 'long' else -1
        
        for i in range(1, len(data)):
            if trend[i-1] == 1:  # Long position
                sar[i] = sar[i-1] + af_values[i-1] * (ep[i-1] - sar[i-1])
                
                # Check if we hit the SAR
                if low.iloc[i] <= sar[i]:
                    trend[i] = -1
                    sar[i] = ep[i-1]
                    ep[i] = low.iloc[i]
                    af_values[i] = af
                else:
                    trend[i] = trend[i-1]
                    if high.iloc[i] > ep[i-1]:
                        ep[i] = high.iloc[i]
                        af_values[i] = min(af_values[i-1] + af, max_af)
                    else:
                        ep[i] = ep[i-1]
                        af_values[i] = af_values[i-1]
                        
            else:  # Short position
                sar[i] = sar[i-1] - af_values[i-1] * (sar[i-1] - ep[i-1])
                
                # Check if we hit the SAR
                if high.iloc[i] >= sar[i]:
                    trend[i] = 1
                    sar[i] = ep[i-1]
                    ep[i] = high.iloc[i]
                    af_values[i] = af
                else:
                    trend[i] = trend[i-1]
                    if low.iloc[i] < ep[i-1]:
                        ep[i] = low.iloc[i]
                        af_values[i] = min(af_values[i-1] + af, max_af)
                    else:
                        ep[i] = ep[i-1]
                        af_values[i] = af_values[i-1]
        
        current_sar = sar[-1]
        current_trend = 'long' if trend[-1] == 1 else 'short'
        
        return {
            'strategy': 'parabolic_sar',
            'current_stop': current_sar,
            'current_trend': current_trend,
            'acceleration_factor': af_values[-1],
            'extreme_point': ep[-1],
            'stop_series': sar.tolist()
        }
    
    def calculate_chandelier_exit(self, data: pd.DataFrame, entry_price: float, 
                                 position_type: str = 'long') -> float:
        """
        Chandelier Exit - trails from highest high or lowest low
        Returns stop price for compatibility with tests
        """
        result = self._calculate_chandelier_exit_full(data, entry_price, position_type)
        return result['current_stop']
    
    def _calculate_chandelier_exit_full(self, data: pd.DataFrame, entry_price: float, 
                                 position_type: str = 'long') -> Dict:
        """
        Chandelier Exit - trails from highest high or lowest low
        """
        period = self.default_params['chandelier_period']
        multiplier = self.default_params['chandelier_multiplier']
        
        # Calculate ATR
        atr = self._calculate_atr(data, period)
        
        stops = []
        
        for i in range(period, len(data)):
            if position_type == 'long':
                highest_high = data['High'].iloc[i-period:i+1].max()
                stop = highest_high - (multiplier * atr.iloc[i])
            else:  # short
                lowest_low = data['Low'].iloc[i-period:i+1].min()
                stop = lowest_low + (multiplier * atr.iloc[i])
            
            stops.append(stop)
        
        # Pad the beginning
        stops = [entry_price] * period + stops
        current_stop = stops[-1]
        
        return {
            'strategy': 'chandelier_exit',
            'current_stop': current_stop,
            'period': period,
            'multiplier': multiplier,
            'current_atr': atr.iloc[-1],
            'stop_series': stops
        }
    
    def calculate_dynamic_trailing_stop(self, data: pd.DataFrame, entry_price: float, 
                                       position_type: str = 'long') -> float:
        """
        Dynamic trailing stop that adjusts based on profit levels
        Returns stop price for compatibility with tests
        """
        result = self._calculate_dynamic_trailing_stop_full(data, entry_price, position_type)
        return result['current_stop']
    
    def _calculate_dynamic_trailing_stop_full(self, data: pd.DataFrame, entry_price: float, 
                                       position_type: str = 'long') -> Dict:
        """
        Dynamic trailing stop that adjusts based on profit levels
        """
        current_price = data['Close'].iloc[-1]
        
        # Calculate profit
        if position_type == 'long':
            profit_points = current_price - entry_price
            profit_percentage = (profit_points / entry_price) * 100
        else:
            profit_points = entry_price - current_price
            profit_percentage = (profit_points / entry_price) * 100
        
        # Define profit tiers and corresponding stop distances
        profit_tiers = [
            {'min_profit': 0, 'max_profit': 2, 'stop_percentage': 0.02},      # 0-2% profit: 2% stop
            {'min_profit': 2, 'max_profit': 5, 'stop_percentage': 0.015},     # 2-5% profit: 1.5% stop
            {'min_profit': 5, 'max_profit': 10, 'stop_percentage': 0.01},     # 5-10% profit: 1% stop
            {'min_profit': 10, 'max_profit': 20, 'stop_percentage': 0.008},   # 10-20% profit: 0.8% stop
            {'min_profit': 20, 'max_profit': float('inf'), 'stop_percentage': 0.005}  # 20%+ profit: 0.5% stop
        ]
        
        # Find appropriate tier
        stop_percentage = 0.02  # Default
        for tier in profit_tiers:
            if tier['min_profit'] <= profit_percentage < tier['max_profit']:
                stop_percentage = tier['stop_percentage']
                break
        
        # Calculate stop based on recent high/low
        lookback = 10
        if position_type == 'long':
            recent_high = data['High'].tail(lookback).max()
            current_stop = recent_high * (1 - stop_percentage)
            # Ensure stop is at breakeven or better after 5% profit
            if profit_percentage >= 5:
                current_stop = max(current_stop, entry_price * 1.001)  # At least breakeven + 0.1%
        else:
            recent_low = data['Low'].tail(lookback).min()
            current_stop = recent_low * (1 + stop_percentage)
            # Ensure stop is at breakeven or better after 5% profit
            if profit_percentage >= 5:
                current_stop = min(current_stop, entry_price * 0.999)  # At least breakeven + 0.1%
        
        return {
            'strategy': 'dynamic_trail',
            'current_stop': current_stop,
            'profit_percentage': profit_percentage,
            'stop_percentage': stop_percentage * 100,  # Convert to percentage
            'tier': self._get_profit_tier_name(profit_percentage),
            'breakeven_locked': profit_percentage >= 5
        }
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high = data['High']
        low = data['Low']
        close = data['Close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        return tr.rolling(period).mean()
    
    def update_trailing_stop(self, data: pd.DataFrame, entry_price: float, current_stop: float,
                           position_type: str = 'long', strategy: str = 'atr_trail') -> float:
        """
        Update trailing stop based on current market data
        Returns new stop price (never moves against the position)
        """
        if strategy not in self._stop_strategies_full:
            return current_stop
        
        # Calculate new stop using specified strategy
        strategy_func = self._stop_strategies_full[strategy]
        result = strategy_func(data, entry_price, position_type)
        
        if isinstance(result, dict):
            new_stop = result.get('current_stop', current_stop)
        else:
            new_stop = result
        
        # Ensure stop only moves in favorable direction
        if position_type == 'long':
            # For long positions, stop can only move up
            return max(current_stop, new_stop)
        else:
            # For short positions, stop can only move down
            return min(current_stop, new_stop)
    
    def _adjust_atr_multiplier(self, profit_percentage: float) -> float:
        """Adjust ATR multiplier based on profit level"""
        if profit_percentage < 0:
            return 2.5  # Wider stop for losing positions
        elif profit_percentage < 5:
            return 2.0  # Standard
        elif profit_percentage < 10:
            return 1.5  # Tighter
        elif profit_percentage < 20:
            return 1.2  # Much tighter
        else:
            return 1.0  # Very tight for big winners
    
    def _select_optimal_strategy(self, results: Dict, market_regime: str, data: pd.DataFrame) -> Dict:
        """Select optimal trailing stop strategy based on conditions"""
        valid_results = {k: v for k, v in results.items() if v is not None}
        
        if not valid_results:
            # Fallback to simple percentage stop
            return {
                'strategy': 'fixed_percentage',
                'current_stop': data['Close'].iloc[-1] * 0.95
            }
        
        # Market regime based selection
        if market_regime in ['STRONG_BULL', 'BULL']:
            # Prefer Parabolic SAR in trending markets
            if 'parabolic_sar' in valid_results and valid_results['parabolic_sar']['current_trend'] == 'long':
                return valid_results['parabolic_sar']
            elif 'chandelier_exit' in valid_results:
                return valid_results['chandelier_exit']
            else:
                return valid_results.get('atr_trail', list(valid_results.values())[0])
                
        elif market_regime in ['BEAR', 'STRONG_BEAR']:
            # Prefer tighter stops in bear markets
            if 'dynamic_trail' in valid_results:
                return valid_results['dynamic_trail']
            else:
                return valid_results.get('percentage_trail', list(valid_results.values())[0])
                
        elif market_regime == 'HIGH_VOLATILITY':
            # Prefer ATR-based stops in volatile markets
            if 'atr_trail' in valid_results:
                return valid_results['atr_trail']
            else:
                return valid_results.get('chandelier_exit', list(valid_results.values())[0])
        
        else:  # NEUTRAL
            # Use dynamic trailing stop
            if 'dynamic_trail' in valid_results:
                return valid_results['dynamic_trail']
            else:
                return valid_results.get('atr_trail', list(valid_results.values())[0])
    
    def _get_profit_tier_name(self, profit_percentage: float) -> str:
        """Get descriptive name for profit tier"""
        if profit_percentage < 0:
            return 'Losing Position'
        elif profit_percentage < 2:
            return 'Minimal Profit'
        elif profit_percentage < 5:
            return 'Small Profit'
        elif profit_percentage < 10:
            return 'Moderate Profit'
        elif profit_percentage < 20:
            return 'Good Profit'
        else:
            return 'Excellent Profit'
    
    def _generate_stop_recommendation(self, optimal_strategy: Dict, market_regime: str) -> str:
        """Generate recommendation for stop loss management"""
        strategy = optimal_strategy.get('strategy', 'unknown')
        
        recommendations = {
            'atr_trail': f"Using ATR-based trailing stop. Automatically adjusts to market volatility.",
            'percentage_trail': f"Using {optimal_strategy.get('percentage', 5)*100:.1f}% trailing stop.",
            'parabolic_sar': f"Using Parabolic SAR. Acceleration factor: {optimal_strategy.get('acceleration_factor', 0.02):.3f}",
            'chandelier_exit': f"Using Chandelier Exit with {optimal_strategy.get('period', 22)}-period lookback.",
            'dynamic_trail': f"Using dynamic trailing stop. Current tier: {optimal_strategy.get('tier', 'Unknown')}"
        }
        
        base_rec = recommendations.get(strategy, "Using default trailing stop strategy.")
        
        # Add regime-specific advice
        if market_regime in ['STRONG_BULL', 'BULL']:
            base_rec += " In bull market - allowing more room for position to breathe."
        elif market_regime in ['BEAR', 'STRONG_BEAR']:
            base_rec += " In bear market - using tighter stops to protect capital."
        elif market_regime == 'HIGH_VOLATILITY':
            base_rec += " High volatility detected - stops adjusted to avoid premature exit."
        
        # Add profit-specific advice
        if 'profit_percentage' in optimal_strategy:
            profit = optimal_strategy['profit_percentage']
            if profit > 20:
                base_rec += " Large profit locked in - stop very close to protect gains."
            elif profit > 10:
                base_rec += " Good profit secured - stop tightened to protect profits."
            elif profit > 5:
                base_rec += " Breakeven stop activated - position cannot turn into a loss."
        
        return base_rec
    
    def update_trailing_stops(self, positions: List[Dict], market_data: Dict, market_regime: str) -> List[Dict]:
        """
        Update trailing stops for all open positions
        """
        updated_positions = []
        
        for position in positions:
            symbol = position['symbol']
            entry_price = position['entry_price']
            position_type = position['position_type']
            
            if symbol in market_data:
                data = market_data[symbol]
                
                # Calculate optimal trailing stop
                stop_info = self.calculate_optimal_trailing_stop(
                    data, entry_price, position_type, market_regime
                )
                
                # Update position with new stop
                position['current_stop'] = stop_info['current_stop']
                position['stop_strategy'] = stop_info['optimal_strategy']
                position['stop_recommendation'] = stop_info['recommendation']
                
                # Check if stop would be triggered
                current_price = data['Close'].iloc[-1]
                if position_type == 'long' and current_price <= stop_info['current_stop']:
                    position['stop_triggered'] = True
                elif position_type == 'short' and current_price >= stop_info['current_stop']:
                    position['stop_triggered'] = True
                else:
                    position['stop_triggered'] = False
                
                updated_positions.append(position)
        
        return updated_positions


def main():
    """Test the trailing stop manager"""
    import yfinance as yf
    
    # Initialize manager
    manager = TrailingStopManager()
    
    # Test with real data
    symbol = 'AAPL'
    ticker = yf.Ticker(symbol)
    data = ticker.history(period='3mo')
    
    # Simulate a position
    entry_price = data['Close'].iloc[-30]  # Entered 30 days ago
    
    print(f"🛡️ TRAILING STOP ANALYSIS FOR {symbol}")
    print("="*60)
    print(f"Entry Price: ${entry_price:.2f}")
    print(f"Current Price: ${data['Close'].iloc[-1]:.2f}")
    print(f"Profit: {((data['Close'].iloc[-1] - entry_price) / entry_price * 100):.2f}%")
    print("\n")
    
    # Calculate stops for different market regimes
    for regime in ['STRONG_BULL', 'NEUTRAL', 'HIGH_VOLATILITY']:
        print(f"Market Regime: {regime}")
        print("-"*40)
        
        stop_info = manager.calculate_optimal_trailing_stop(
            data, entry_price, 'long', regime
        )
        
        print(f"Optimal Strategy: {stop_info['optimal_strategy']}")
        print(f"Current Stop: ${stop_info['current_stop']:.2f}")
        print(f"Stop Distance: ${stop_info['stop_distance']:.2f} ({stop_info['stop_percentage']:.2f}%)")
        print(f"Recommendation: {stop_info['recommendation']}")
        print("\n")
    
    # Save example output
    output = {
        'symbol': symbol,
        'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'entry_price': entry_price,
        'current_price': data['Close'].iloc[-1],
        'regime_analysis': {}
    }
    
    for regime in ['STRONG_BULL', 'NEUTRAL', 'HIGH_VOLATILITY']:
        stop_info = manager.calculate_optimal_trailing_stop(
            data, entry_price, 'long', regime
        )
        output['regime_analysis'][regime] = {
            'optimal_strategy': stop_info['optimal_strategy'],
            'current_stop': stop_info['current_stop'],
            'stop_percentage': stop_info['stop_percentage']
        }
    
    import os
    os.makedirs('outputs/trailing_stops', exist_ok=True)
    
    with open('outputs/trailing_stops/example_analysis.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print("💾 Analysis saved to outputs/trailing_stops/example_analysis.json")


if __name__ == "__main__":
    main()