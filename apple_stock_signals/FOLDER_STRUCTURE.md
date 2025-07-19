# 📁 Project Folder Structure

```
apple_stock_signals/
│
├── 📚 Documentation (Root Level)
│   ├── README.md                    # Quick start guide
│   ├── MASTER_DOCUMENTATION.md      # Complete system documentation
│   ├── COMPLETE_SYSTEM_GUIDE.md     # Command reference & workflow
│   ├── ADVANCED_FEATURES_GUIDE.md   # Advanced features details
│   ├── TALIB_INSTALLATION_GUIDE.md  # TA-Lib installation
│   └── FOLDER_STRUCTURE.md          # This file
│
├── 🎯 Core Scripts
│   ├── core_scripts/
│   │   ├── __init__.py
│   │   ├── config.py                # System configuration
│   │   ├── enhanced_trading_analyzer.py
│   │   ├── live_swing_signals.py
│   │   └── main.py
│   │
│   └── data_modules/
│       ├── __init__.py
│       ├── data_fetcher.py          # Yahoo Finance data
│       ├── historical_data_updater.py
│       ├── signal_generator.py      # Trading signals
│       ├── technical_analyzer.py    # TA-Lib indicators
│       └── sentiment_analyzer.py
│
├── 📈 Trading Systems
│   └── trading_systems/
│       ├── __init__.py
│       ├── final_swing_system.py    # Production system
│       ├── ultra_swing_system.py    # ML-enhanced
│       └── optimized_swing_system.py
│
├── 🚀 Advanced Features
│   └── advanced_features/
│       ├── __init__.py
│       ├── integrated_trading_system.py
│       │
│       ├── economic_calendar/       # Event tracking
│       │   ├── __init__.py
│       │   └── economic_events.py
│       │
│       ├── trade_journal/           # Performance tracking
│       │   ├── __init__.py
│       │   ├── trade_journal.py
│       │   └── trades.db
│       │
│       ├── risk_management/         # Portfolio risk
│       │   ├── __init__.py
│       │   └── risk_dashboard.py
│       │
│       └── data_monitoring/         # Data quality
│           ├── __init__.py
│           └── data_quality_monitor.py
│
├── 📊 Data Storage
│   ├── historical_data/
│   │   ├── AAPL_historical_data.csv
│   │   ├── GOOGL_historical_data.csv
│   │   ├── MSFT_historical_data.csv
│   │   ├── TSLA_historical_data.csv
│   │   ├── UNH_historical_data.csv
│   │   └── fetch_metadata.json
│   │
│   ├── outputs/                     # Analysis results
│   │   ├── enhanced_analysis.json
│   │   └── daily_YYYYMMDD/
│   │
│   └── backtest_results/            # Strategy testing
│       └── *.json
│
├── 🛠️ Scripts & Utilities
│   ├── scripts/                     # Shell scripts
│   │   ├── daily_trading_analysis.sh
│   │   ├── update_historical_data.sh
│   │   ├── system_health_check.sh
│   │   ├── setup_alias.sh
│   │   └── run_advanced_analysis.sh
│   │
│   └── utils/                       # Additional utilities
│       ├── fetch_historical_data.sh
│       └── run_trading_analysis.sh
│
├── 🧪 Testing
│   ├── tests/
│   │   └── test_system.py
│   │
│   └── test_reports/                # Test results
│       └── system_test_*.txt
│
├── 📦 Archives
│   ├── archive/
│   │   ├── old_swing_systems/       # Previous versions
│   │   └── old_docs/
│   │
│   └── archived_outputs/            # Historical outputs
│
├── 📄 Configuration Files
│   ├── requirements.txt             # Python dependencies
│   └── .gitignore                  # Git ignore rules
│
└── 📋 Other
    ├── documentation/               # Additional docs
    ├── status/                     # System status files
    └── backtest_apple_signals.py   # Backtesting script
```

## 📌 Key Directories Explained

### Core Scripts
- **config.py**: Central configuration for stocks, indicators, risk settings
- **enhanced_trading_analyzer.py**: Main analysis engine
- **signal_generator.py**: Combines technical/sentiment/fundamental signals

### Advanced Features
- **economic_calendar/**: Tracks FOMC, CPI, earnings for position adjustment
- **trade_journal/**: SQLite database for trade logging and analytics
- **risk_management/**: Portfolio heat maps, VaR, correlation analysis
- **data_monitoring/**: Real-time data quality and anomaly detection

### Data Storage
- **historical_data/**: CSV files with price history (updated daily)
- **outputs/**: JSON analysis results organized by date
- **backtest_results/**: Strategy performance testing results

### Scripts
All shell scripts are now organized in the `scripts/` directory for cleaner root

## 🔄 Data Flow

1. **Input**: Historical data CSVs → 
2. **Processing**: Technical analysis + Signal generation →
3. **Risk Management**: Position sizing + Event adjustments →
4. **Output**: JSON signals + Trading recommendations

## 🏗️ Recent Improvements

1. Created `scripts/` directory for all shell scripts
2. Created `tests/` directory for test files
3. Created `archive/` for older versions
4. Consolidated documentation in root
5. Removed empty directories (analytics/, reports/)
6. Added `MASTER_DOCUMENTATION.md` as single source of truth

## 💡 Best Practices

1. **Daily Updates**: Run `trade-update` before analysis
2. **Health Checks**: Run `trade-health` weekly
3. **Backups**: Archive important outputs monthly
4. **Testing**: Run `trade-test` after any code changes
5. **Documentation**: Update docs when adding features