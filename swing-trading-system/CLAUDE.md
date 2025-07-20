# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Development Commands

**Primary Analysis Commands:**
- **Run Top 5 stocks**: `./automation/run_analysis.sh top5`
- **Run Top 10 stocks**: `./automation/run_analysis.sh top10`
- **Run Top 50 stocks**: `./automation/run_analysis.sh top50`
- **Run Tech sector**: `./automation/run_analysis.sh tech`
- **Run Healthcare sector**: `./automation/run_analysis.sh health`

**Stock Management:**
- **Add stock**: `./automation/run_analysis.sh add AAPL`
- **Remove stock**: `./automation/run_analysis.sh remove TSLA`
- **Show config**: `./automation/run_analysis.sh config`

**Manual Pipeline Operations:**
- **Run pipeline directly**: `python3 automation/pipeline.py --list top_50`
- **Train ML models**: `python3 automation/train_models.py`
- **Generate reports**: `python3 automation/generate_reports.py`

**Testing Commands:**
- **System health check**: `python3 tests/health_check.py` (comprehensive test of all components)
- **Integration tests**: `python3 tests/test_integration.py`
- **Phase 1 tests**: `python3 tests/test_phase1_integration.py`
- **System tests**: `python3 tests/test_system.py`

## Development Guidelines

**Coding Standards:**
- Always use python3
- Use explicit python3 command for all Python scripts

## Architecture Overview

This is a **Python-based automated swing trading system** designed for 2-15 day trading strategies with comprehensive risk management and machine learning integration.

### Core Architecture

**Technology Stack**: Python 3, pandas, numpy, yfinance, scikit-learn, joblib
**Trading Focus**: Multi-timeframe swing trading with volume confirmation
**Data Source**: Yahoo Finance via yfinance API
**ML Framework**: Random Forest classifiers with StandardScaler

### Key Modules

**Core Trading Logic (`/core/`)**:
- `analysis/`: Multi-timeframe analyzer, swing analyzer, volume analyzer, market regime detector
- `risk_management/`: Position sizing, stop loss systems, profit targets, integrated risk management
- `indicators/`: Enhanced technical indicators (RSI, MACD, SMA, EMA, Bollinger Bands, ATR, Stochastic)

**Automation System (`/automation/`)**:
- `run_analysis.sh`: Master bash script with color-coded output
- `pipeline.py`: Core trading pipeline orchestrator
- `train_models.py`: ML model training automation
- `generate_reports.py`: Automated report generation

**Configuration (`/config/`)**:
- `stocks.json`: Master configuration with pre-defined stock lists (top5, top10, top50, sectors)
- `active_stocks.txt`: Currently active stock list
- `pipeline_state.json`: Pipeline execution state tracking

### Trading System Features

**Multi-Layered Analysis**:
- Weekly, daily, and 4-hour timeframe analysis
- Volume confirmation using OBV, VWAP, MFI
- Entry filters for liquidity, trend, and momentum
- Market regime detection

**Risk Management System**:
- Dynamic stop losses and profit targets
- Position sizing (10% max position, 2% risk per trade)
- Integrated risk assessment across all positions

**Automation Pipeline**:
- State-based execution (only runs when stock lists change)
- Comprehensive data caching (2-year historical data)
- 365-day backtest periods
- Parallel processing capabilities

### Data Organization

**Data Storage (`/data/`)**:
- `historical/`: Cached stock data for 50+ major US stocks
- `analysis_results/`: Daily analysis outputs and trading signals
- `backtest_results/`: Strategy performance data
- `reports/`: Generated summary reports

**Model Storage (`/ml_models/`)**:
- Trained Random Forest models
- Model performance metrics
- Feature importance data

### Stock Lists Available

The system comes with pre-configured stock lists:
- **top_5**: AAPL, MSFT, GOOGL, TSLA, UNH
- **top_10**: Top 10 US stocks by market cap
- **top_50**: Top 50 US stocks by market cap
- **tech_sector**: Technology focused stocks
- **health_sector**: Healthcare focused stocks

### Development Notes

**State Management**: The pipeline uses `pipeline_state.json` to track execution state and only runs analysis when stock lists change, making it efficient for repeated runs.

**Error Handling**: The system includes comprehensive error handling with fallback mechanisms and detailed logging.

**Testing Framework**: Uses Python's built-in `unittest` framework. Tests include system health checks, unit tests, integration tests, and performance benchmarks. Run `python3 tests/health_check.py` for comprehensive testing.

**Dependencies**: Core dependencies include yfinance for data, pandas/numpy for analysis, scikit-learn for ML, and joblib for model persistence. No pip requirements file - dependencies managed manually.