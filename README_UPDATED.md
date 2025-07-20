# Algorithmic Trading System - Updated Documentation

**Last Updated:** July 20, 2025  
**Version:** 2.0

## Overview

This is a comprehensive algorithmic trading system designed for swing trading (2-15 day positions) across the top 50 US stocks. The system uses technical analysis, machine learning, and advanced risk management to generate trading signals.

## Current System Status

### Performance Summary
- **ML Model Accuracy:** 62.7% (test set)
- **Average Backtest Return:** -5.57%
- **Top Performer:** NVDA (+99.74%)
- **Active Stocks:** 50 (Top US stocks by market cap)

### Latest Recommendations (as of July 20, 2025)
- **Strong Buy:** GOOGL
- **Buy:** TSLA
- **Hold:** AAPL, NVDA, JNJ
- **Sell:** MSFT, AMZN, META, BRK-B, JPM

## Quick Start

```bash
# Navigate to the project directory
cd /Users/yatharthanand/genai-assistant-vercel/trading-script/apple_stock_signals

# Run the complete pipeline for top 50 stocks
python3 core_scripts/top50_complete_pipeline.py

# Or run the master pipeline
python3 master_pipeline.py

# View latest results
cat outputs/master_analysis_*.json
```

## Project Structure (After Cleanup)

```
trading-script/
├── apple_stock_signals/        # Main application directory
│   ├── core_scripts/          # Core trading logic
│   ├── data_modules/          # Data handling modules
│   ├── trading_systems/       # Trading strategies
│   ├── advanced_features/     # Advanced capabilities
│   ├── ml_models/            # Machine learning models
│   ├── historical_data/      # Stock data (CSV files)
│   ├── outputs/              # Analysis results
│   ├── backtest_results/     # Backtesting outputs
│   └── reports/              # Performance reports
├── swing-trading-system/      # Reorganized clean structure
├── SYSTEM_PERFORMANCE_REPORT.md
└── README_UPDATED.md
```

## Key Features

### 1. Technical Analysis
- RSI, MACD, Bollinger Bands, SMA/EMA
- Volume analysis and momentum indicators
- Support/resistance detection
- Multi-timeframe analysis

### 2. Machine Learning
- Random Forest classifier
- 15 technical features
- Real-time prediction updates
- Feature importance analysis

### 3. Risk Management
- 2% risk per trade
- ATR-based stop losses
- Multiple take-profit targets
- Portfolio heat mapping

### 4. Advanced Features
- Economic calendar integration
- Trade journaling
- Data quality monitoring
- Paper trading simulation

## Installation

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install TA-Lib (macOS)
brew install ta-lib
pip install TA-Lib

# Set up aliases (optional)
./setup_alias.sh
source ~/.bashrc
```

## Usage Examples

### Run Complete Analysis
```bash
# Analyze all 50 stocks
python3 core_scripts/top50_complete_pipeline.py

# Run specific stock analysis
python3 data_modules/simple_apple_analyzer.py
```

### View Results
```bash
# Latest signals
cat outputs/top50_signals/signals_*.json

# Backtest results
cat backtest_results/top50/backtest_*.json

# ML performance
cat ml_models/performance/training_report_*.json
```

## Current Issues & Improvements Needed

1. **Strategy Performance**
   - Negative average returns indicate strategy needs optimization
   - Consider adding market regime detection
   - Implement adaptive parameters

2. **Machine Learning**
   - Some overfitting (86.8% train vs 62.7% test)
   - Add cross-validation
   - Try ensemble methods

3. **Risk Management**
   - Add sector rotation
   - Implement correlation-based sizing
   - Add drawdown limits

## Recommended Next Steps

1. **Use the swing-trading-system folder** - It has a cleaner, more organized structure
2. **Optimize the trading strategy** - Current momentum strategy underperforms
3. **Add real-time monitoring** - Build a dashboard for live tracking
4. **Implement broker integration** - For live trading capabilities
5. **Enhance ML models** - Add LSTM for time series prediction

## Files Cleaned Up

During the latest cleanup (July 20, 2025), the following were removed:
- 8 old test reports
- 18 duplicate documentation files
- Old backtest results (kept only latest)
- Redundant data folders
- Empty directories

See `CLEANUP_SUMMARY.md` for full details.

## Disclaimer

This system is for educational and research purposes only. Past performance does not guarantee future results. The system shows mixed results in backtesting and should not be used for actual trading without further optimization and risk assessment.

---

For detailed documentation, see `apple_stock_signals/MASTER_DOCUMENTATION.md`