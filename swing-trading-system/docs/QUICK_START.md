# 🚀 Quick Start Guide - Swing Trading System

## New Project Structure

The project has been reorganized with a cleaner, more professional structure:

```
swing-trading-system/         (renamed from apple_stock_signals)
├── core/                    (renamed from core_scripts)
│   ├── analysis/           - Trading analysis modules
│   ├── risk_management/    - Risk & position management
│   └── indicators/         - Technical indicators
├── automation/             - Automation scripts
├── data/                   - All data storage
├── config/                 - Configuration files
└── docs/                   - Documentation
```

## Running the System

### 1. Navigate to New Location
```bash
cd /Users/yatharthanand/genai-assistant-vercel/trading-script/swing-trading-system
```

### 2. Run Analysis
```bash
# Analyze top 5 stocks
./automation/run_analysis.sh top5

# Analyze top 10 stocks
./automation/run_analysis.sh top10

# Analyze top 50 stocks
./automation/run_analysis.sh top50
```

### 3. Check Results
Results are now organized in the `data/` directory:
- **Analysis Results**: `data/analysis_results/`
- **Reports**: `data/reports/`
- **Backtest Data**: `data/backtest_results/`
- **Historical Data**: `data/historical/`

## Configuration

Stock lists are now in `config/stocks.json` (renamed from stocks_config.json)

### View Current Config
```bash
cat config/stocks.json | grep -A 10 '"active": true'
```

### Add/Remove Stocks
```bash
# Add a stock
./automation/run_analysis.sh add NVDA

# Remove a stock
./automation/run_analysis.sh remove TSLA
```

## Testing the System

### Run System Test
```bash
cd tests
python3 test_system.py
```

### Health Check
```bash
cd tests
python3 health_check.py
```

## Key Changes

1. **Project Name**: `apple_stock_signals` → `swing-trading-system`
2. **Core Scripts**: `core_scripts/` → `core/`
3. **Outputs**: `outputs/` → `data/analysis_results/`
4. **Config File**: `stocks_config.json` → `stocks.json`

## Create Alias for Easy Access

Add this to your shell profile:
```bash
echo "alias swing='cd /Users/yatharthanand/genai-assistant-vercel/trading-script/swing-trading-system && ./automation/run_analysis.sh'" >> ~/.zshrc
source ~/.zshrc
```

Now you can run:
```bash
swing top50
swing add AAPL
swing config
```

## What's Inside Each Folder

- **core/analysis/**: Multi-timeframe, volume, entry filters, swing analyzer
- **core/risk_management/**: Stop loss, position sizing, profit targets
- **core/indicators/**: Technical indicators and wrappers
- **automation/**: Scripts for running analysis, training, reports
- **data/**: All outputs, results, and historical data
- **config/**: Stock lists and system settings
- **tests/**: Test scripts and health checks
- **docs/**: All documentation

The system is now better organized and more scalable for future enhancements!