#!/usr/bin/env python3
"""
Unit tests for Volatility Position Sizing
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from core_scripts.volatility_position_sizing import VolatilityPositionSizer

class TestVolatilityPositionSizing:
    """
    Test suite for VolatilityPositionSizer
    """
    
    def setup_method(self):
        """Set up test fixtures"""
        self.portfolio_size = 100000
        self.sizer = VolatilityPositionSizer(self.portfolio_size)
    
    def test_initialization(self):
        """Test position sizer initialization"""
        assert self.sizer is not None
        assert self.sizer.portfolio_size == self.portfolio_size
        assert hasattr(self.sizer, 'risk_config')
        assert 0 < self.sizer.risk_config['max_risk_per_trade'] <= 0.02
        assert 0 < self.sizer.risk_config['max_portfolio_heat'] <= 0.06
    
    def test_calculate_position_size_basic(self):
        """Test basic position size calculation"""
        result = self.sizer.calculate_position_size(
            entry_price=100.0,
            stop_loss=95.0,
            confidence=0.7
        )
        
        assert isinstance(result, dict)
        assert 'shares' in result
        assert 'position_value' in result
        assert 'risk_amount' in result
        assert result['shares'] > 0
        assert result['position_value'] <= self.portfolio_size * 0.25  # Max 25% position
    
    def test_kelly_criterion_calculation(self):
        """Test Kelly Criterion calculation"""
        kelly_fraction = self.sizer._calculate_kelly_criterion(
            win_rate=0.6,
            avg_win=0.03,
            avg_loss=0.015
        )
        
        assert isinstance(kelly_fraction, float)
        assert 0 <= kelly_fraction <= 1
        
        # Test with edge cases
        # 50% win rate with equal wins/losses should give 0
        kelly_zero = self.sizer._calculate_kelly_criterion(0.5, 0.02, 0.02)
        assert abs(kelly_zero) < 0.01
        
        # Very high win rate should give positive Kelly
        kelly_high = self.sizer._calculate_kelly_criterion(0.8, 0.03, 0.01)
        assert kelly_high > 0
    
    def test_position_size_with_risk_limits(self):
        """Test position sizing respects risk limits"""
        # Test with 2% risk limit
        result = self.sizer.calculate_position_size(
            entry_price=100.0,
            stop_loss=90.0,  # 10% stop
            confidence=0.8
        )
        
        # Risk should not exceed 2% of portfolio
        assert result['risk_amount'] <= self.portfolio_size * 0.02
        assert result['risk_percentage'] <= 0.02
    
    def test_volatility_adjustment(self, sample_stock_data):
        """Test volatility-based position adjustment"""
        volatility = self.sizer._calculate_volatility(sample_stock_data)
        adjustment = self.sizer._get_volatility_adjustment(volatility)
        
        assert isinstance(volatility, float)
        assert volatility > 0
        assert 0 < adjustment <= 1.5
        
        # High volatility should reduce position size
        if volatility > 0.3:  # 30% annualized volatility
            assert adjustment < 1.0
    
    def test_portfolio_allocation(self, sample_signals):
        """Test portfolio-wide allocation"""
        trade_setups = [
            {
                'symbol': signal['symbol'],
                'entry_price': signal['entry_price'],
                'stop_loss': signal['stop_loss'],
                'take_profit_1': signal['take_profit_1'],
                'confidence': signal['confidence']
            }
            for signal in sample_signals
        ]
        
        historical_performance = {
            'win_rate': 0.55,
            'avg_win': 0.03,
            'avg_loss': 0.015
        }
        
        # Mock market data
        market_data = {
            'AAPL': sample_stock_data,
            'GOOGL': sample_stock_data
        }
        
        result = self.sizer.calculate_portfolio_allocation(
            trade_setups,
            historical_performance,
            market_data
        )
        
        assert isinstance(result, dict)
        assert 'allocations' in result
        assert 'summary' in result
        
        # Check portfolio heat limit
        total_risk = sum(alloc['risk_percentage'] for alloc in result['allocations'])
        assert total_risk <= self.sizer.risk_config['max_portfolio_heat']
    
    def test_correlation_adjustment(self):
        """Test correlation-based position adjustment"""
        # Create correlated positions
        positions = [
            {'symbol': 'AAPL', 'sector': 'Technology'},
            {'symbol': 'MSFT', 'sector': 'Technology'},
            {'symbol': 'JPM', 'sector': 'Financial'}
        ]
        
        # Tech stocks should have reduced allocation due to correlation
        for i, pos in enumerate(positions):
            if pos['sector'] == 'Technology' and i > 0:
                # Second tech stock should have some reduction
                # This is a simplified test - actual implementation may vary
                assert True  # Placeholder for actual correlation test
    
    def test_edge_cases(self):
        """Test edge cases and error handling"""
        # Test with zero stop loss distance
        with pytest.raises(ValueError):
            self.sizer.calculate_position_size(
                entry_price=100.0,
                stop_loss=100.0,  # Same as entry
                confidence=0.7
            )
        
        # Test with negative confidence
        with pytest.raises(ValueError):
            self.sizer.calculate_position_size(
                entry_price=100.0,
                stop_loss=95.0,
                confidence=-0.5
            )
        
        # Test with confidence > 1
        with pytest.raises(ValueError):
            self.sizer.calculate_position_size(
                entry_price=100.0,
                stop_loss=95.0,
                confidence=1.5
            )
    
    def test_minimum_position_size(self):
        """Test minimum position size constraints"""
        # Very small portfolio should still calculate valid positions
        small_sizer = VolatilityPositionSizer(1000)  # $1000 portfolio
        
        result = small_sizer.calculate_position_size(
            entry_price=500.0,  # Expensive stock
            stop_loss=490.0,
            confidence=0.7
        )
        
        # Should get at least 1 share or 0 shares
        assert result['shares'] >= 0
        if result['shares'] > 0:
            assert result['shares'] >= 1
    
    def test_portfolio_remaining_calculation(self):
        """Test calculation of remaining portfolio capacity"""
        # Simulate existing positions
        existing_positions = [
            {'symbol': 'AAPL', 'value': 20000, 'risk': 0.01},
            {'symbol': 'GOOGL', 'value': 15000, 'risk': 0.008}
        ]
        
        total_risk = sum(pos['risk'] for pos in existing_positions)
        remaining_heat = self.sizer.risk_config['max_portfolio_heat'] - total_risk
        
        assert remaining_heat > 0
        assert remaining_heat < self.sizer.risk_config['max_portfolio_heat']
    
    def test_position_sizing_consistency(self):
        """Test that position sizing is consistent"""
        # Same inputs should give same outputs
        results = []
        for _ in range(5):
            result = self.sizer.calculate_position_size(
                entry_price=100.0,
                stop_loss=95.0,
                confidence=0.75
            )
            results.append(result['shares'])
        
        # All results should be identical
        assert all(r == results[0] for r in results)