# Project Reorganization Plan

## Current Issues
- Project named "apple_stock_signals" but handles multiple stocks
- Unclear folder structure with mixed purposes
- Some legacy folders that aren't used

## Proposed New Structure

```
trading-script/
├── swing-trading-system/           # Main project folder (renamed from apple_stock_signals)
│   ├── config/                     # Configuration files
│   │   ├── stocks.json            # Stock lists configuration
│   │   ├── settings.json          # System settings
│   │   └── strategies.json        # Trading strategies config
│   │
│   ├── core/                      # Core trading modules (renamed from core_scripts)
│   │   ├── analysis/              # Analysis modules
│   │   │   ├── multi_timeframe.py
│   │   │   ├── volume_analyzer.py
│   │   │   ├── entry_filters.py
│   │   │   └── swing_analyzer.py
│   │   │
│   │   ├── risk_management/       # Risk management modules
│   │   │   ├── stop_loss.py
│   │   │   ├── position_sizing.py
│   │   │   ├── profit_targets.py
│   │   │   └── risk_manager.py
│   │   │
│   │   ├── indicators/            # Technical indicators
│   │   │   ├── technical_wrapper.py
│   │   │   └── enhanced_analyzer.py
│   │   │
│   │   └── utils/                 # Utility functions
│   │       └── data_fetcher.py
│   │
│   ├── automation/                # Automation scripts (renamed from scripts)
│   │   ├── run_analysis.sh       # Main runner script
│   │   ├── pipeline.py           # Master pipeline
│   │   ├── train_models.py
│   │   ├── backtest.py
│   │   └── generate_reports.py
│   │
│   ├── ml_models/                 # Machine learning models
│   │   ├── models/
│   │   ├── training/
│   │   └── predictions/
│   │
│   ├── data/                      # All data storage
│   │   ├── historical/            # Historical price data
│   │   ├── analysis_results/      # Analysis outputs
│   │   ├── backtest_results/      # Backtest results
│   │   └── reports/               # Generated reports
│   │
│   ├── tests/                     # Test scripts
│   │   ├── test_system.py
│   │   ├── test_integration.py
│   │   └── health_check.py
│   │
│   ├── docs/                      # Documentation
│   │   ├── README.md
│   │   ├── SETUP_GUIDE.md
│   │   ├── USER_GUIDE.md
│   │   └── API_REFERENCE.md
│   │
│   └── archive/                   # Old/unused files
│
└── README.md                      # Main project README
```

## Key Renamings

1. **apple_stock_signals** → **swing-trading-system**
2. **core_scripts** → **core**
3. **scripts** → **automation**
4. **outputs** → **data/analysis_results**
5. **historical_data** → **data/historical**
6. **reports** → **data/reports**

## Benefits
- Clear, professional structure
- Logical grouping of related modules
- Easy to understand purpose of each folder
- Scalable for future additions
- Follows Python best practices