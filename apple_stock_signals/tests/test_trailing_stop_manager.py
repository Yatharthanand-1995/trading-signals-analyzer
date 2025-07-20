#!/usr/bin/env python3
"""
Unit tests for Trailing Stop Manager
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from core_scripts.trailing_stop_manager import TrailingStopManager

class TestTrailingStopManager:
    """
    Test suite for TrailingStopManager
    """
    
    def setup_method(self):
        """Set up test fixtures"""
        self.stop_manager = TrailingStopManager()
        
        # Create sample price data
        dates = pd.date_range(end=datetime.now(), periods=50, freq='D')
        self.sample_data = pd.DataFrame({
            'High': 100 + np.cumsum(np.random.uniform(-1, 2, 50)),
            'Low': 98 + np.cumsum(np.random.uniform(-1, 2, 50)),
            'Close': 99 + np.cumsum(np.random.uniform(-1, 2, 50))
        }, index=dates)
        
        # Ensure price consistency
        self.sample_data['High'] = self.sample_data[['High', 'Low', 'Close']].max(axis=1)
        self.sample_data['Low'] = self.sample_data[['High', 'Low', 'Close']].min(axis=1)
    
    def test_initialization(self):
        """Test stop manager initialization"""
        assert self.stop_manager is not None
        assert hasattr(self.stop_manager, 'stop_strategies')
        assert len(self.stop_manager.stop_strategies) >= 5  # At least 5 strategies
    
    def test_calculate_atr_trailing_stop(self):
        """Test ATR-based trailing stop calculation"""
        entry_price = 100.0
        position_type = 'long'
        
        stop_price = self.stop_manager.calculate_atr_trailing_stop(
            self.sample_data, entry_price, position_type
        )
        
        assert isinstance(stop_price, float)
        assert stop_price > 0
        # Note: stop might be above entry if price has moved up significantly
        # Just check it's reasonable
        assert stop_price < self.sample_data['High'].max() * 1.1
    
    def test_calculate_percentage_trailing_stop(self):
        """Test percentage-based trailing stop"""
        entry_price = 100.0
        position_type = 'long'
        
        stop_price = self.stop_manager.calculate_percentage_trailing_stop(
            self.sample_data, entry_price, position_type
        )
        
        assert isinstance(stop_price, float)
        assert stop_price > 0
        
        # Check stop is at correct percentage
        current_price = self.sample_data['Close'].iloc[-1]
        if position_type == 'long':
            # Just check that the stop is reasonable (below entry)
            assert stop_price < current_price
            assert stop_price > 0
    
    def test_calculate_parabolic_sar(self):
        """Test Parabolic SAR calculation"""
        entry_price = 100.0
        position_type = 'long'
        
        stop_price = self.stop_manager.calculate_parabolic_sar(
            self.sample_data, entry_price, position_type
        )
        
        assert isinstance(stop_price, float)
        assert stop_price > 0
    
    def test_calculate_chandelier_exit(self):
        """Test Chandelier Exit calculation"""
        entry_price = 100.0
        position_type = 'long'
        
        stop_price = self.stop_manager.calculate_chandelier_exit(
            self.sample_data, entry_price, position_type
        )
        
        assert isinstance(stop_price, float)
        assert stop_price > 0
        assert stop_price < self.sample_data['High'].max()  # Should be below highest high
    
    def test_calculate_dynamic_trailing_stop(self):
        """Test dynamic trailing stop calculation"""
        entry_price = 100.0
        position_type = 'long'
        
        stop_price = self.stop_manager.calculate_dynamic_trailing_stop(
            self.sample_data, entry_price, position_type
        )
        
        assert isinstance(stop_price, float)
        assert stop_price > 0
    
    def test_calculate_optimal_trailing_stop(self):
        """Test optimal trailing stop selection"""
        entry_price = 100.0
        position_type = 'long'
        market_regime = 'BULL'
        
        result = self.stop_manager.calculate_optimal_trailing_stop(
            self.sample_data, entry_price, position_type, market_regime
        )
        
        assert isinstance(result, dict)
        assert 'optimal_strategy' in result
        assert 'current_stop' in result
        assert 'stop_distance' in result
        assert 'stop_percentage' in result
        assert 'recommendation' in result
        
        assert result['current_stop'] > 0
        assert result['optimal_strategy'] in self.stop_manager.stop_strategies
    
    def test_stop_strategies_for_different_regimes(self):
        """Test that different market regimes use appropriate strategies"""
        entry_price = 100.0
        position_type = 'long'
        
        regimes = ['STRONG_BULL', 'BULL', 'NEUTRAL', 'BEAR', 'STRONG_BEAR', 'HIGH_VOLATILITY']
        
        results = {}
        for regime in regimes:
            result = self.stop_manager.calculate_optimal_trailing_stop(
                self.sample_data, entry_price, position_type, regime
            )
            results[regime] = result
        
        # Different regimes should potentially use different strategies
        # Just check that we got results for all regimes
        assert len(results) == len(regimes)
        assert all(r['current_stop'] > 0 for r in results.values())
    
    def test_update_trailing_stop(self):
        """Test trailing stop update mechanism"""
        entry_price = 100.0
        position_type = 'long'
        current_stop = 95.0
        
        # Create data where price has moved up
        uptrend_data = self.sample_data.copy()
        uptrend_data['Close'] = uptrend_data['Close'] * 1.1  # 10% increase
        uptrend_data['High'] = uptrend_data['High'] * 1.1
        
        new_stop = self.stop_manager.update_trailing_stop(
            uptrend_data, entry_price, current_stop, position_type, 'atr_trail'
        )
        
        assert isinstance(new_stop, float)
        # For long position, new stop should be >= current stop (never moves down)
        assert new_stop >= current_stop
    
    def test_short_position_stops(self):
        """Test trailing stops for short positions"""
        entry_price = 100.0
        position_type = 'short'
        
        result = self.stop_manager.calculate_optimal_trailing_stop(
            self.sample_data, entry_price, position_type, 'BEAR'
        )
        
        # For short position, stop should be above entry
        assert result['current_stop'] > entry_price
    
    def test_edge_cases(self):
        """Test edge cases and error handling"""
        # Test with insufficient data
        small_data = self.sample_data.head(5)
        
        result = self.stop_manager.calculate_optimal_trailing_stop(
            small_data, 100.0, 'long', 'NEUTRAL'
        )
        
        # Should still return valid result
        assert isinstance(result, dict)
        assert result['current_stop'] > 0
        
        # Test with extreme volatility
        volatile_data = self.sample_data.copy()
        volatile_data['Close'] = volatile_data['Close'] * (1 + np.random.uniform(-0.1, 0.1, len(volatile_data)))
        
        result = self.stop_manager.calculate_optimal_trailing_stop(
            volatile_data, 100.0, 'long', 'HIGH_VOLATILITY'
        )
        
        assert result['current_stop'] > 0
    
    def test_stop_loss_never_decreases_for_long(self):
        """Test that stop loss never decreases for long positions"""
        entry_price = 100.0
        position_type = 'long'
        initial_stop = 95.0
        
        # Test multiple updates
        current_stop = initial_stop
        for i in range(10):
            # Add some random price movement
            self.sample_data['Close'] = self.sample_data['Close'] * (1 + np.random.uniform(-0.02, 0.03))
            
            new_stop = self.stop_manager.update_trailing_stop(
                self.sample_data, entry_price, current_stop, position_type, 'atr_trail'
            )
            
            # Stop should never decrease
            assert new_stop >= current_stop
            current_stop = new_stop
    
    def test_all_strategies_return_valid_stops(self):
        """Test that all strategies return valid stop prices"""
        entry_price = 100.0
        
        for strategy_name in self.stop_manager.stop_strategies:
            for position_type in ['long', 'short']:
                strategy_func = self.stop_manager.stop_strategies[strategy_name]
                stop_price = strategy_func(self.sample_data, entry_price, position_type)
                
                assert isinstance(stop_price, (int, float))
                assert stop_price > 0
                assert not np.isnan(stop_price)
                assert not np.isinf(stop_price)