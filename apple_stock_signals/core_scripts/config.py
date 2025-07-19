# Configuration settings for Apple Stock Analyzer

import os

# Stock symbols to analyze
STOCKS = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'UNH']

# API Keys (set as environment variables)
API_KEYS = {
    'NEWS_API_KEY': os.getenv('NEWS_API_KEY', 'your_news_api_key_here'),
    'ALPHA_VANTAGE_KEY': os.getenv('ALPHA_VANTAGE_KEY', 'your_alpha_vantage_key_here'),  # Optional - for price verification
}

# Data Verification Settings
VERIFICATION_SETTINGS = {
    'ENABLE_CROSS_VERIFICATION': True,     # Cross-check prices from multiple sources
    'PRICE_DISCREPANCY_THRESHOLD': 1.0,    # % difference threshold for price alerts
    'MIN_DATA_CONFIDENCE': 75.0,           # Minimum confidence score to proceed
    'ANOMALY_DETECTION': True,             # Enable anomaly detection
    'WEB_SCRAPING_BACKUP': True,           # Use web scraping as backup data source
}

# Analysis Settings
ANALYSIS_SETTINGS = {
    'SYMBOL': 'AAPL',
    'LOOKBACK_PERIOD': 252,  # Trading days (1 year)
    'CONFIDENCE_THRESHOLD': 60,  # Minimum confidence for signals
    'MAX_RISK_PERCENT': 2.0,  # Maximum risk per trade
    'MIN_RISK_REWARD_RATIO': 2.0,  # Minimum risk/reward ratio
}

# Risk Management Settings
RISK_PER_TRADE = 2.0  # Percentage of capital to risk per trade
PORTFOLIO_SIZE = 10000  # Default portfolio size for calculations

# Technical Indicator Settings
TECHNICAL_SETTINGS = {
    'RSI_PERIOD': 14,
    'MACD_FAST': 12,
    'MACD_SLOW': 26,
    'MACD_SIGNAL': 9,
    'SMA_PERIODS': [20, 50, 200],
    'ATR_PERIOD': 14,
    'BOLLINGER_PERIOD': 20,
    'BOLLINGER_STD': 2
}

# Signal Generation Weights (must sum to 1.0)
SIGNAL_WEIGHTS = {
    'TECHNICAL': 0.60,     # Technical indicators
    'SENTIMENT': 0.30,     # News and social sentiment
    'FUNDAMENTAL': 0.10    # Fundamental analysis
}

# Signal Thresholds
SIGNAL_THRESHOLDS = {
    'STRONG_BUY': 70,
    'BUY': 60,
    'HOLD_UPPER': 59,
    'HOLD_LOWER': 40,
    'SELL': 30,
    'STRONG_SELL': 0
}