# 📁 Organized Trading Script Structure

## 🗂️ Directory Structure

```
apple_stock_signals/
│
├── core_scripts/          # Main executable scripts
│   ├── main.py           # Original Apple stock analyzer
│   ├── config.py         # Configuration settings
│   ├── enhanced_trading_analyzer.py  # Multi-stock analyzer
│   ├── live_swing_signals.py       # Live signal generator
│   └── live_data_analyzer.py       # Real-time data analyzer
│
├── trading_systems/       # Different trading strategies
│   ├── swing_trading_system.py     # Basic swing trading
│   ├── improved_swing_system.py    # Enhanced swing trading
│   ├── final_swing_system.py       # Optimized swing trading
│   ├── ultra_swing_system.py       # Advanced ML-based system
│   └── optimized_swing_system.py   # Latest optimizations
│
├── data_modules/          # Data processing modules
│   ├── data_fetcher.py            # Real-time data fetching
│   ├── historical_data_fetcher.py # Historical data download
│   ├── technical_analyzer.py      # Technical indicators
│   ├── sentiment_analyzer.py      # News/social sentiment
│   ├── signal_generator.py        # Signal generation logic
│   └── *_analyzer.py             # Various analysis modules
│
├── backtest_results/      # Historical backtest outputs
│   ├── swing_trades_*.json        # Trade records by stock
│   ├── swing_trades_*.csv         # CSV trade summaries
│   └── *_results_*.json          # System performance results
│
├── outputs/              # Current analysis outputs
│   ├── enhanced_analysis.json     # Latest analysis
│   └── daily_YYYYMMDD/           # Daily archived results
│
├── historical_data/      # Downloaded historical data
│   ├── AAPL_historical_data.csv
│   ├── GOOGL_historical_data.csv
│   └── ...
│
├── documentation/        # All documentation files
│   ├── README.md
│   ├── SYSTEM_DOCUMENTATION.md
│   ├── TRADING_LOGIC_FLOWCHART.md
│   └── ...
│
├── utils/               # Utility scripts
│   ├── run_trading_analysis.sh
│   └── fetch_historical_data.sh
│
├── archived_outputs/    # Old output files
│
├── daily_trading_analysis.sh  # Daily run script
├── setup_alias.sh            # Alias setup script
└── requirements.txt          # Python dependencies
```

## 🚀 Quick Start with Aliases

After running `./setup_alias.sh`, you have access to these commands:

### `trade`
Run the daily trading analysis for all stocks (includes automatic data update):
```bash
trade
```

### `trade-log`
View today's analysis results in formatted JSON:
```bash
trade-log
```

### `trade-dir`
Navigate to the trading script directory:
```bash
trade-dir
```

### `trade-update`
Update historical data for all stocks:
```bash
trade-update
```

### `trade-update-aapl`
Update historical data for specific stock (example: AAPL):
```bash
trade-update-aapl
# For other stocks: ./update_historical_data.sh -s GOOGL
```

### `trade-validate`
Validate data integrity for all stocks:
```bash
trade-validate
```

## 📅 Daily Usage

1. **Run Daily Analysis**:
   ```bash
   trade
   ```
   This will:
   - Update historical data files with latest market data
   - Analyze AAPL, GOOGL, TSLA, MSFT, UNH
   - Generate trading signals
   - Save results to `outputs/daily_YYYYMMDD/`
   - Clean up files older than 30 days

2. **View Results**:
   ```bash
   trade-log
   ```

3. **Access Historical Results**:
   ```bash
   trade-dir
   cd outputs/
   ls -la daily_*
   ```

## 🔧 Configuration

Edit `core_scripts/config.py` to modify:
- Trading parameters
- Risk management settings
- Technical indicator periods
- API keys (optional)

## 📊 Output Files

Daily analysis creates:
- `analysis_YYYYMMDD.json` - Detailed technical analysis
- `live_signals_*.json` - Real-time trading signals

## 🧹 Maintenance

The daily script automatically:
- Archives daily results
- Cleans up files older than 30 days
- Organizes outputs by date

## 📊 Historical Data Management

### Features:
- **Automatic Updates**: Daily analysis automatically updates historical data
- **Incremental Updates**: Only fetches new data since last update
- **Data Validation**: Checks for duplicates and missing dates
- **Integrity Checks**: Validates data quality and consistency

### Manual Data Operations:
```bash
# Check data status
python3 data_modules/data_status.py

# Update specific stock
./update_historical_data.sh -s TSLA

# Validate all data
trade-validate
```

### Data Files Location:
- Historical data: `historical_data/`
- Update metadata: `historical_data/update_metadata.json`
- Update reports: `historical_data/update_report.txt`

## 💡 Tips

1. **Schedule Daily Runs**: Add to crontab for automated analysis
   ```bash
   0 9 * * 1-5 /Users/yatharthanand/genai-assistant-vercel/trading-script/apple_stock_signals/daily_trading_analysis.sh
   ```

2. **Review Backtest Results**: Check `backtest_results/` for historical performance

3. **Monitor Live Signals**: Use `trade-log` to quickly check today's recommendations

4. **Keep Data Fresh**: Historical data is automatically updated, but you can run `trade-update` manually anytime

## ⚠️ Important Notes

- Always verify signals before trading
- This is for educational/analytical purposes only
- Past performance doesn't guarantee future results
- Use proper risk management