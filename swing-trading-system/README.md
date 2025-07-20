# Swing Trading System

A comprehensive automated trading system optimized for 2-15 day swing trading strategies.

## Features

- **Multi-Timeframe Analysis**: Analyzes weekly, daily, and 4-hour charts
- **Volume Confirmation**: OBV, VWAP, MFI, and volume pattern detection
- **Entry Filters**: Liquidity, trend, and momentum filters
- **Risk Management**: Dynamic stops, position sizing, and profit targets
- **Automation**: One-command analysis for multiple stock lists

## Quick Start

```bash
cd swing-trading-system
./automation/run_analysis.sh top50
```

## Documentation

- [User Guide](docs/USER_GUIDE.md)
- [System Status](docs/SYSTEM_STATUS.md)
- [API Reference](docs/API_REFERENCE.md)

## Structure

- `core/`: Core trading modules
- `automation/`: Automation scripts
- `data/`: Historical data and results
- `config/`: Configuration files
- `ml_models/`: Machine learning models
- `tests/`: Test scripts
- `docs/`: Documentation
