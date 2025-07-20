# 🚀 Automated Trading Analysis Pipeline

## Overview

The system now includes a fully automated pipeline that detects when you add or remove stocks and automatically runs all necessary analyses. No more manual steps!

## ✨ Key Features

1. **Automatic Stock List Detection**: The system detects when you change the stock list
2. **Complete Pipeline Execution**: Runs all analyses automatically:
   - Historical data fetching
   - Technical analysis
   - ML model training
   - Backtesting
   - Paper trading
   - Report generation
3. **Configuration Management**: Easy JSON-based configuration
4. **One-Command Execution**: Single command to run everything

## 📋 Stock Lists Available

- **top_5**: Top 5 tech stocks (AAPL, MSFT, GOOGL, TSLA, UNH)
- **top_10**: Top 10 US stocks by market cap
- **top_50**: Top 50 US stocks by market cap (currently active)
- **tech_sector**: Technology sector focus (12 stocks)
- **healthcare_sector**: Healthcare sector focus (11 stocks)
- **custom**: Your own custom list

## 🎯 Quick Start

### Run with current stock list:
```bash
./run_analysis.sh
```

### Switch to different stock list:
```bash
./run_analysis.sh top50    # Analyze top 50 stocks
./run_analysis.sh top10    # Analyze top 10 stocks
./run_analysis.sh tech     # Analyze tech sector
```

### Manage custom stock list:
```bash
./run_analysis.sh add AAPL      # Add AAPL to custom list
./run_analysis.sh add TSLA      # Add TSLA to custom list
./run_analysis.sh remove AAPL   # Remove AAPL from custom list
./run_analysis.sh custom        # Run analysis on custom list
```

### Other commands:
```bash
./run_analysis.sh config   # Show current configuration
./run_analysis.sh force    # Force full pipeline run
./run_analysis.sh quick    # Run quick analysis only
./run_analysis.sh help     # Show help
```

## 🔄 How It Works

1. **Configuration File** (`config/stocks_config.json`):
   - Contains all stock lists
   - Marks which list is currently active
   - Stores analysis settings

2. **Master Pipeline** (`master_pipeline.py`):
   - Detects if stock list has changed
   - Runs complete analysis pipeline
   - Saves state between runs
   - Only re-runs when necessary

3. **Run Script** (`run_analysis.sh`):
   - User-friendly interface
   - Handles all commands
   - Shows colored output
   - Displays results summary

## 📊 What Happens When You Change Stocks

When you add/remove stocks or switch lists:

1. System detects the change automatically
2. Fetches historical data for new stocks
3. Runs technical analysis on all stocks
4. Trains ML models with new data
5. Backtests trading strategies
6. Runs paper trading simulation
7. Generates comprehensive reports

All this happens automatically with one command!

## 📁 Output Files

After running, find your results in:

- `outputs/` - Current analysis results
- `backtest_results/` - Historical performance data
- `reports/` - Comprehensive reports
- `historical_data/` - Cached stock data
- `ml_models/` - Trained ML models

## 🎨 Examples

### Example 1: Analyze Your Watchlist
```bash
# Add your favorite stocks
./run_analysis.sh add NVDA
./run_analysis.sh add AMD
./run_analysis.sh add INTC

# Run analysis on your custom list
./run_analysis.sh custom
```

### Example 2: Sector Analysis
```bash
# Analyze tech sector
./run_analysis.sh tech

# Later, switch to healthcare
./run_analysis.sh health
```

### Example 3: Quick Market Check
```bash
# Just run technical analysis (fast)
./run_analysis.sh quick
```

## 🔧 Advanced Configuration

Edit `config/stocks_config.json` to:

- Add new stock lists
- Change analysis parameters
- Enable/disable pipeline steps
- Adjust position sizing
- Configure output formats

## 🚨 Important Notes

1. **First Run**: Takes longer as it downloads all historical data
2. **Subsequent Runs**: Much faster, only updates recent data
3. **Stock Changes**: Automatically triggers full analysis
4. **Force Run**: Use `./run_analysis.sh force` to override cache

## 📈 Performance Tips

- Start with smaller lists (top_5 or top_10) for testing
- Use `quick` mode for rapid market checks
- Run full pipeline overnight for large lists
- Check `config/pipeline_state.json` for last run info

## 🛡️ Error Handling

If something fails:
1. Check the console output for error messages
2. Look at `config/pipeline_state.json` for failed steps
3. Use `force` flag to retry: `./run_analysis.sh force`
4. Individual logs are in respective module directories

## 📚 Next Steps

1. Review results in `outputs/enhanced_analysis.json`
2. Check backtesting performance in `backtest_results/`
3. Monitor paper trading in `paper_trading/trades/`
4. Read comprehensive reports in `reports/`

Remember: This is for educational purposes only. Always do your own research!