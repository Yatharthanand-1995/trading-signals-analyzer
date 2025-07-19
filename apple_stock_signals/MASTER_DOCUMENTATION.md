# 📚 Master Documentation - Algorithmic Trading System

## Table of Contents
1. [System Overview](#system-overview)
2. [Quick Start Guide](#quick-start-guide)
3. [Installation & Setup](#installation--setup)
4. [System Architecture](#system-architecture)
5. [Core Features](#core-features)
6. [Advanced Features](#advanced-features)
7. [Command Reference](#command-reference)
8. [Daily Workflow](#daily-workflow)
9. [Trading Strategies](#trading-strategies)
10. [Risk Management](#risk-management)
11. [Performance Metrics](#performance-metrics)
12. [Troubleshooting](#troubleshooting)
13. [API Reference](#api-reference)
14. [Development Guide](#development-guide)
15. [Backlog & Future Enhancements](#backlog--future-enhancements)

---

## System Overview

This is a comprehensive algorithmic trading system designed for swing trading (2-15 day positions) with advanced risk management, performance tracking, and market analysis capabilities.

### Key Capabilities
- **Multi-Stock Analysis**: AAPL, GOOGL, MSFT, TSLA, UNH
- **Technical Analysis**: 20+ indicators including RSI, MACD, Bollinger Bands
- **Risk Management**: Position sizing, stop losses, portfolio heat mapping
- **Event-Driven Trading**: Economic calendar integration
- **Performance Analytics**: Trade journaling with pattern recognition
- **Data Quality Monitoring**: Real-time validation and anomaly detection
- **Automated Backtesting**: Historical performance validation

### System Requirements
- Python 3.8+
- 4GB RAM minimum
- 10GB disk space
- Stable internet connection
- macOS/Linux/Windows (with WSL)

---

## Quick Start Guide

### 1. First Time Setup
```bash
# Clone the repository
cd /Users/yatharthanand/genai-assistant-vercel/trading-script/apple_stock_signals

# Install dependencies
pip install -r requirements.txt

# Install TA-Lib (required for technical indicators)
# macOS:
brew install ta-lib && pip install TA-Lib

# Set up aliases
./setup_alias.sh
source ~/.bashrc  # or ~/.zshrc

# Initial data fetch
trade-update
```

### 2. Daily Trading Routine
```bash
# Morning routine (before market open)
trade-health      # Check system health
trade-update      # Update historical data
trade-advanced    # Run full analysis with all features

# During trading hours
trade-risk        # Monitor portfolio risk
trade-monitor     # Check data quality

# End of day
trade-journal     # Review performance
trade-calendar    # Check tomorrow's events
```

### 3. View Results
```bash
trade-log         # View today's analysis
trade-dir         # Navigate to project directory
```

---

## Installation & Setup

### Prerequisites
1. **Python 3.8+**
   ```bash
   python3 --version  # Should be 3.8 or higher
   ```

2. **TA-Lib Installation**
   
   **macOS (Homebrew)**:
   ```bash
   brew install ta-lib
   pip install TA-Lib
   ```
   
   **Ubuntu/Debian**:
   ```bash
   sudo apt-get update
   sudo apt-get install build-essential wget
   wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
   tar -xzf ta-lib-0.4.0-src.tar.gz
   cd ta-lib/
   ./configure --prefix=/usr
   make
   sudo make install
   pip install TA-Lib
   ```
   
   **Windows**:
   - Download wheel from https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
   - Install: `pip install TA_Lib‑0.4.28‑cp39‑cp39‑win_amd64.whl`

3. **Python Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Configuration
The system uses `core_scripts/config.py` for all settings:

```python
# Stock symbols to analyze
STOCKS = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'UNH']

# Risk management
RISK_PER_TRADE = 2.0  # Percentage of capital to risk
PORTFOLIO_SIZE = 10000  # Default portfolio size

# Technical indicators
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
```

---

## System Architecture

### Directory Structure
```
apple_stock_signals/
├── core_scripts/          # Main trading logic
│   ├── config.py         # System configuration
│   ├── enhanced_trading_analyzer.py
│   └── live_swing_signals.py
├── data_modules/          # Data handling
│   ├── data_fetcher.py
│   ├── historical_data_updater.py
│   ├── signal_generator.py
│   └── technical_analyzer.py
├── trading_systems/       # Trading strategies
│   ├── swing_trading_system.py
│   └── ultra_swing_system.py
├── advanced_features/     # Advanced capabilities
│   ├── economic_calendar/
│   ├── trade_journal/
│   ├── risk_management/
│   └── data_monitoring/
├── historical_data/       # CSV data files
├── outputs/              # Analysis results
├── backtest_results/     # Backtesting outputs
└── documentation/        # Additional docs
```

### Data Flow
1. **Data Collection**: Yahoo Finance API → Historical data CSVs
2. **Technical Analysis**: Calculate indicators using TA-Lib
3. **Signal Generation**: Combine technical, sentiment, and fundamental scores
4. **Risk Management**: Apply position sizing and risk filters
5. **Event Adjustment**: Modify based on economic calendar
6. **Output Generation**: JSON reports with trade recommendations

---

## Core Features

### 1. Multi-Stock Analysis
Analyzes 5 major stocks simultaneously:
- **AAPL**: Apple Inc.
- **GOOGL**: Alphabet Inc.
- **MSFT**: Microsoft Corporation
- **TSLA**: Tesla Inc.
- **UNH**: UnitedHealth Group

### 2. Technical Indicators
- **Trend**: SMA (20, 50, 200), EMA, MACD
- **Momentum**: RSI, Stochastic, ROC
- **Volatility**: Bollinger Bands, ATR, Standard Deviation
- **Volume**: OBV, Volume SMA, Volume Ratio
- **Support/Resistance**: Pivot Points, Fibonacci Levels

### 3. Signal Generation
Combines multiple factors with weighted scoring:
- Technical Analysis (60%)
- Sentiment Analysis (30%)
- Fundamental Data (10%)

Signal classifications:
- **STRONG_BUY**: Score ≥ 70
- **BUY**: Score ≥ 60
- **HOLD**: Score 40-59
- **SELL**: Score ≤ 30
- **STRONG_SELL**: Score ≤ 0

### 4. Position Sizing
Risk-based position sizing with:
- 2% risk per trade (configurable)
- Stop loss based on ATR
- Multiple take-profit targets (R/R 2:1, 4:1, 6:1)

---

## Advanced Features

### 1. Economic Calendar Integration
Tracks and adjusts for market events:
- **FOMC Meetings**: -50% position size 3 days before
- **CPI/NFP Releases**: -30% position size 2 days before
- **Earnings**: -50% position size 5 days before
- **GDP Reports**: -25% position size 2 days before

```bash
trade-calendar  # View upcoming events
```

### 2. Trade Journal & Analytics
Comprehensive performance tracking:
- Automated trade logging
- Win/loss pattern analysis
- Common mistake identification
- Performance metrics (Sharpe, Sortino, Calmar ratios)
- Day-of-week and time-of-day analysis

```bash
trade-journal  # View performance analytics
```

Key metrics tracked:
- Win Rate & Profit Factor
- Maximum Drawdown
- Risk-Adjusted Returns
- Average Win/Loss
- Expectancy

### 3. Risk Management Dashboard
Real-time portfolio risk monitoring:
- Portfolio heat mapping
- Correlation analysis
- Value at Risk (VaR) calculations
- Stress testing scenarios
- Sector exposure limits

```bash
trade-risk  # View risk dashboard
```

Risk levels:
- 🟢 LOW (0-25): Safe to trade normally
- 🟡 MEDIUM (26-50): Exercise caution
- 🟠 HIGH (51-75): Reduce position sizes
- 🔴 CRITICAL (76-100): Stop new trades

### 4. Data Quality Monitoring
Ensures data integrity:
- Real-time validation
- Anomaly detection
- Source comparison
- Missing data alerts
- API health checks

```bash
trade-monitor  # Run data quality check
```

---

## Command Reference

### Basic Commands
| Command | Description |
|---------|-------------|
| `trade` | Run daily trading analysis |
| `trade-log` | View today's analysis results |
| `trade-dir` | Navigate to trading directory |

### Data Management
| Command | Description |
|---------|-------------|
| `trade-update` | Update all historical data |
| `trade-update-aapl` | Update specific stock data |
| `trade-validate` | Validate data integrity |

### Advanced Analysis
| Command | Description |
|---------|-------------|
| `trade-advanced` | Run all advanced features |
| `trade-calendar` | View economic events |
| `trade-journal` | View trading performance |
| `trade-risk` | View risk dashboard |

### System Monitoring
| Command | Description |
|---------|-------------|
| `trade-monitor` | Data quality monitoring |
| `trade-health` | Complete health check |
| `trade-test` | Run system tests |

---

## Daily Workflow

### Pre-Market (9:00 AM ET)
1. **System Check**
   ```bash
   trade-health  # Ensure all systems operational
   ```

2. **Data Update**
   ```bash
   trade-update  # Get latest market data
   ```

3. **Run Analysis**
   ```bash
   trade-advanced  # Generate signals with all features
   ```

4. **Review Signals**
   - Check `trade-log` for recommendations
   - Note any risk warnings
   - Review upcoming events

### Market Hours (9:30 AM - 4:00 PM ET)
1. **Execute Trades**
   - Place orders based on signals
   - Set stop losses and take profits
   - Log trades in journal

2. **Monitor Positions**
   ```bash
   trade-risk  # Check portfolio heat
   ```

3. **Data Quality**
   ```bash
   trade-monitor  # Ensure data accuracy
   ```

### Post-Market (After 4:00 PM ET)
1. **Performance Review**
   ```bash
   trade-journal  # Analyze today's trades
   ```

2. **Tomorrow's Prep**
   ```bash
   trade-calendar  # Check tomorrow's events
   ```

---

## Trading Strategies

### 1. Enhanced Swing Trading (Primary)
**Timeframe**: 2-15 days
**Win Rate**: 37-43%
**Profit Factor**: 1.9-2.5

Entry criteria:
- RSI oversold bounce or breakout
- MACD bullish crossover
- Price above key moving averages
- Volume confirmation

Exit criteria:
- Target hit (2:1, 4:1, 6:1 R/R)
- Stop loss triggered
- Reversal signal
- Time-based exit (15 days max)

### 2. Ultra Swing System (ML-Enhanced)
**Features**:
- Machine learning pattern recognition
- Adaptive indicators
- Multi-timeframe analysis
- Sentiment integration

**Performance**:
- Higher win rate (45-50%)
- Better risk-adjusted returns
- Reduced drawdowns

### 3. Event-Driven Adjustments
**Before major events**:
- Reduce position sizes
- Tighten stops
- Consider hedging
- Increase cash allocation

---

## Risk Management

### Position Sizing Formula
```
Shares = (Account_Size × Risk_Per_Trade) / (Entry_Price - Stop_Loss)
```

Example:
- Account: $10,000
- Risk: 2% = $200
- Entry: $100
- Stop: $95
- Shares = $200 / $5 = 40 shares

### Portfolio Rules
1. **Maximum Positions**: 5 stocks
2. **Sector Limits**: Max 40% in one sector
3. **Correlation Limits**: Avoid highly correlated positions
4. **Cash Reserve**: Minimum 20% cash
5. **Drawdown Limits**: Stop at -10% monthly loss

### Risk Metrics
- **Maximum Loss per Trade**: 2%
- **Maximum Daily Loss**: 6%
- **Maximum Weekly Loss**: 10%
- **Maximum Monthly Loss**: 15%

---

## Performance Metrics

### Expected Performance (Based on Backtesting)
| Metric | Range |
|--------|-------|
| Win Rate | 37-43% |
| Profit Factor | 1.2-1.9 |
| Sharpe Ratio | 1.0-2.0 |
| Max Drawdown | 10-15% |
| Annual Return | 15-30% |

### Key Performance Indicators
1. **Sharpe Ratio**: Risk-adjusted returns (target > 1.5)
2. **Profit Factor**: Gross profit / Gross loss (target > 1.5)
3. **Win Rate**: Not critical if R/R is good (35%+ acceptable)
4. **Expectancy**: Average profit per trade (must be positive)
5. **Recovery Factor**: Net profit / Max drawdown (target > 3)

---

## Troubleshooting

### Common Issues and Solutions

#### 1. Import Errors
```bash
# Fix: Install dependencies
pip install -r requirements.txt

# Fix: Add __init__.py files
touch core_scripts/__init__.py
```

#### 2. TA-Lib Installation Failed
```bash
# macOS fix:
brew install ta-lib
export TA_LIB_PATH=/opt/homebrew/opt/ta-lib
pip install TA-Lib

# Alternative: Use 'ta' package
pip install ta
```

#### 3. Data Not Updating
```bash
# Check internet connection
ping yahoo.com

# Force update
rm historical_data/*.csv
trade-update

# Validate data
trade-validate
```

#### 4. Timezone Errors
- Fixed in data_quality_monitor.py
- System handles both timezone-aware and naive datetime objects

#### 5. No Signals Generated
- Check if market is open
- Verify data is current
- Review signal thresholds in config
- Run `trade-health` for diagnostics

### Error Logs
- System test reports: `test_reports/`
- Monitoring logs: `advanced_features/data_monitoring/monitoring_log.json`
- Trade journal: `advanced_features/trade_journal/trades.db`

---

## API Reference

### Core Classes

#### DataFetcher
```python
from data_modules.data_fetcher import AppleDataFetcher
fetcher = AppleDataFetcher()
data = fetcher.fetch_stock_data()
```

#### SignalGenerator
```python
from data_modules.signal_generator import AppleSignalGenerator
generator = AppleSignalGenerator()
signal = generator.generate_signal(data)
```

#### TechnicalAnalyzer
```python
from data_modules.technical_analyzer import AppleTechnicalAnalyzer
analyzer = AppleTechnicalAnalyzer()
indicators = analyzer.calculate_all_indicators(stock_data)
```

### Advanced Features

#### EconomicCalendar
```python
from advanced_features.economic_calendar.economic_events import EconomicCalendar
calendar = EconomicCalendar()
events = calendar.get_upcoming_events(days_ahead=7)
adjusted_size = calendar.calculate_event_adjusted_position_size(base_size, symbol)
```

#### TradeJournal
```python
from advanced_features.trade_journal.trade_journal import TradeJournal, Trade
journal = TradeJournal()
journal.log_trade(trade)
analytics = journal.generate_analytics_report()
```

#### RiskDashboard
```python
from advanced_features.risk_management.risk_dashboard import RiskManagementDashboard
dashboard = RiskManagementDashboard()
risk_analysis = dashboard.analyze_portfolio_risk(positions)
```

---

## Development Guide

### Adding a New Stock
1. Add symbol to `STOCKS` list in `config.py`
2. Run `trade-update` to fetch historical data
3. Verify with `trade-validate`

### Creating a New Strategy
1. Create new file in `trading_systems/`
2. Inherit from base strategy class
3. Implement `generate_signals()` method
4. Add to strategy selector in main analyzer

### Adding a New Indicator
1. Update `technical_analyzer.py`
2. Add calculation in `calculate_all_indicators()`
3. Update signal generator weights if needed

### Testing Changes
```bash
# Run comprehensive tests
trade-test

# Check specific module
python3 -m pytest tests/test_module.py

# Backtest strategy
python3 backtest_apple_signals.py
```

---

## Backlog & Future Enhancements

### Planned Features
1. **Machine Learning Enhancements**
   - LSTM price prediction
   - Random Forest signal validation
   - Reinforcement learning optimizer

2. **Additional Data Sources**
   - Options flow analysis
   - Social media sentiment
   - Insider trading data
   - Analyst recommendations

3. **Portfolio Management**
   - Multi-strategy allocation
   - Dynamic rebalancing
   - Pairs trading
   - Market neutral strategies

4. **Automation**
   - Broker API integration
   - Automated order execution
   - Position monitoring alerts
   - Telegram/Discord notifications

5. **Advanced Analytics**
   - Monte Carlo simulations
   - Walk-forward optimization
   - Market regime detection
   - Correlation breakout detection

### Contributing
1. Fork the repository
2. Create feature branch
3. Add tests for new features
4. Update documentation
5. Submit pull request

---

## Disclaimer

**IMPORTANT**: This trading system is for educational and research purposes only. 

- Past performance does not guarantee future results
- All trading involves risk of loss
- Never trade with money you cannot afford to lose
- Always verify signals before executing trades
- Consider paper trading before using real money
- The developers assume no liability for trading losses

**Risk Warning**: Trading stocks involves substantial risk of loss and is not suitable for every investor. The valuation of stocks may fluctuate, and as a result, you may lose more than your original investment.

---

## Support & Resources

### Getting Help
- Check troubleshooting section first
- Review error logs in `test_reports/`
- Run `trade-health` for diagnostics

### Additional Resources
- [TA-Lib Documentation](https://mrjbq7.github.io/ta-lib/)
- [yfinance Documentation](https://pypi.org/project/yfinance/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)

### Version History
- v2.0 (Current): Advanced features, ML integration
- v1.5: Risk management, trade journal
- v1.0: Basic swing trading system

---

*Last Updated: July 2025*
*System Version: 2.0 - Advanced Features Edition*