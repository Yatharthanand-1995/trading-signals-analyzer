#!/usr/bin/env python3
"""
Pytest configuration and fixtures for trading system tests
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set test environment
os.environ['ENVIRONMENT'] = 'test'
os.environ['FINNHUB_API_KEY'] = 'test_api_key'

@pytest.fixture
def sample_stock_data():
    """
    Create sample stock data for testing
    """
    dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
    
    # Generate realistic stock data
    np.random.seed(42)
    close_prices = 100 + np.cumsum(np.random.randn(100) * 2)
    
    data = pd.DataFrame({
        'Open': close_prices * (1 + np.random.uniform(-0.02, 0.02, 100)),
        'High': close_prices * (1 + np.random.uniform(0, 0.03, 100)),
        'Low': close_prices * (1 + np.random.uniform(-0.03, 0, 100)),
        'Close': close_prices,
        'Volume': np.random.randint(1000000, 10000000, 100)
    }, index=dates)
    
    # Ensure OHLC consistency
    data['High'] = data[['Open', 'High', 'Close']].max(axis=1)
    data['Low'] = data[['Open', 'Low', 'Close']].min(axis=1)
    
    return data

@pytest.fixture
def sample_portfolio():
    """
    Create sample portfolio configuration
    """
    return {
        'cash': 100000,
        'positions': {},
        'max_risk_per_trade': 0.02,
        'max_portfolio_heat': 0.06
    }

@pytest.fixture
def sample_market_regime():
    """
    Create sample market regime data
    """
    return {
        'regime': 'BULL',
        'confidence': 75.5,
        'strategy': 'momentum',
        'risk_multiplier': 1.2,
        'scores': {
            'trend': 80.0,
            'volatility': 40.0,
            'breadth': 70.0,
            'momentum': 75.0
        }
    }

@pytest.fixture
def sample_signals():
    """
    Create sample trading signals
    """
    return [
        {
            'symbol': 'AAPL',
            'action': 'BUY',
            'confidence': 75.0,
            'entry_price': 150.50,
            'stop_loss': 145.00,
            'take_profit_1': 155.00,
            'take_profit_2': 160.00,
            'take_profit_3': 165.00,
            'signal_strength': 'MEDIUM',
            'reasons': ['MACD bullish crossover', 'RSI oversold'],
            'combined_score': 72.5
        },
        {
            'symbol': 'GOOGL',
            'action': 'BUY',
            'confidence': 68.0,
            'entry_price': 2800.00,
            'stop_loss': 2750.00,
            'take_profit_1': 2850.00,
            'take_profit_2': 2900.00,
            'take_profit_3': 2950.00,
            'signal_strength': 'MEDIUM',
            'reasons': ['Volume breakout'],
            'combined_score': 65.8
        }
    ]

@pytest.fixture
def mock_api_response():
    """
    Mock API response for testing
    """
    return {
        'c': 150.50,  # Current price
        'pc': 149.00,  # Previous close
        'o': 149.50,  # Open
        'h': 151.00,  # High
        'l': 149.00,  # Low
        'dp': 1.01  # Percent change
    }