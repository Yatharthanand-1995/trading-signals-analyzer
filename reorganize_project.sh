#!/bin/bash

# Reorganize Trading System Project
# This script will reorganize the project structure

echo "🔄 Starting Project Reorganization..."
echo "===================================="

# Set base directory
BASE_DIR="/Users/yatharthanand/genai-assistant-vercel/trading-script"
OLD_DIR="$BASE_DIR/apple_stock_signals"
NEW_DIR="$BASE_DIR/swing-trading-system"

# Check if old directory exists
if [ ! -d "$OLD_DIR" ]; then
    echo "❌ Error: $OLD_DIR not found!"
    exit 1
fi

# Create new directory structure
echo "📁 Creating new directory structure..."
mkdir -p "$NEW_DIR"/{config,core/{analysis,risk_management,indicators,utils},automation,ml_models/{models,training,predictions},data/{historical,analysis_results,backtest_results,reports},tests,docs,archive}

# Copy configuration files
echo "📋 Moving configuration files..."
if [ -d "$OLD_DIR/config" ]; then
    cp -r "$OLD_DIR/config/stocks_config.json" "$NEW_DIR/config/stocks.json" 2>/dev/null || true
fi

# Copy core analysis modules
echo "🔍 Moving analysis modules..."
if [ -d "$OLD_DIR/core_scripts" ]; then
    # Analysis modules
    cp "$OLD_DIR/core_scripts/multi_timeframe_analyzer.py" "$NEW_DIR/core/analysis/multi_timeframe.py" 2>/dev/null || true
    cp "$OLD_DIR/core_scripts/volume_analyzer.py" "$NEW_DIR/core/analysis/volume_analyzer.py" 2>/dev/null || true
    cp "$OLD_DIR/core_scripts/entry_filter_system.py" "$NEW_DIR/core/analysis/entry_filters.py" 2>/dev/null || true
    cp "$OLD_DIR/core_scripts/swing_trading_analyzer.py" "$NEW_DIR/core/analysis/swing_analyzer.py" 2>/dev/null || true
    
    # Risk management modules
    cp "$OLD_DIR/core_scripts/dynamic_stop_loss_system.py" "$NEW_DIR/core/risk_management/stop_loss.py" 2>/dev/null || true
    cp "$OLD_DIR/core_scripts/advanced_position_sizing.py" "$NEW_DIR/core/risk_management/position_sizing.py" 2>/dev/null || true
    cp "$OLD_DIR/core_scripts/profit_taking_strategy.py" "$NEW_DIR/core/risk_management/profit_targets.py" 2>/dev/null || true
    cp "$OLD_DIR/core_scripts/integrated_risk_management.py" "$NEW_DIR/core/risk_management/risk_manager.py" 2>/dev/null || true
    
    # Indicators
    cp "$OLD_DIR/core_scripts/technical_indicators_wrapper.py" "$NEW_DIR/core/indicators/technical_wrapper.py" 2>/dev/null || true
    cp "$OLD_DIR/core_scripts/enhanced_trading_analyzer.py" "$NEW_DIR/core/indicators/enhanced_analyzer.py" 2>/dev/null || true
fi

# Copy automation scripts
echo "🤖 Moving automation scripts..."
cp "$OLD_DIR/run_analysis.sh" "$NEW_DIR/automation/run_analysis.sh" 2>/dev/null || true
cp "$OLD_DIR/master_pipeline.py" "$NEW_DIR/automation/pipeline.py" 2>/dev/null || true
if [ -d "$OLD_DIR/scripts" ]; then
    cp "$OLD_DIR/scripts/train_ml_model.py" "$NEW_DIR/automation/train_models.py" 2>/dev/null || true
    cp "$OLD_DIR/scripts/generate_master_report.py" "$NEW_DIR/automation/generate_reports.py" 2>/dev/null || true
fi

# Copy ML models
echo "🧠 Moving ML models..."
if [ -d "$OLD_DIR/ml_models" ]; then
    cp -r "$OLD_DIR/ml_models/"* "$NEW_DIR/ml_models/" 2>/dev/null || true
fi

# Copy data files
echo "📊 Moving data files..."
if [ -d "$OLD_DIR/historical_data" ]; then
    cp -r "$OLD_DIR/historical_data/"* "$NEW_DIR/data/historical/" 2>/dev/null || true
fi
if [ -d "$OLD_DIR/outputs" ]; then
    cp -r "$OLD_DIR/outputs/"* "$NEW_DIR/data/analysis_results/" 2>/dev/null || true
fi
if [ -d "$OLD_DIR/backtest_results" ]; then
    cp -r "$OLD_DIR/backtest_results/"* "$NEW_DIR/data/backtest_results/" 2>/dev/null || true
fi
if [ -d "$OLD_DIR/reports" ]; then
    cp -r "$OLD_DIR/reports/"* "$NEW_DIR/data/reports/" 2>/dev/null || true
fi

# Copy test files
echo "🧪 Moving test files..."
if [ -d "$OLD_DIR/core_scripts" ]; then
    cp "$OLD_DIR/core_scripts/test_complete_system.py" "$NEW_DIR/tests/test_system.py" 2>/dev/null || true
    cp "$OLD_DIR/core_scripts/test_phase2_integration.py" "$NEW_DIR/tests/test_integration.py" 2>/dev/null || true
    cp "$OLD_DIR/core_scripts/system_health_check.py" "$NEW_DIR/tests/health_check.py" 2>/dev/null || true
fi

# Copy documentation
echo "📚 Moving documentation..."
cp "$OLD_DIR/README.md" "$NEW_DIR/docs/README.md" 2>/dev/null || true
cp "$OLD_DIR/HOW_TO_RUN_GUIDE.md" "$NEW_DIR/docs/USER_GUIDE.md" 2>/dev/null || true
cp "$OLD_DIR/SYSTEM_STATUS_REPORT.md" "$NEW_DIR/docs/SYSTEM_STATUS.md" 2>/dev/null || true
cp "$OLD_DIR/FIXES_COMPLETED.md" "$NEW_DIR/docs/CHANGELOG.md" 2>/dev/null || true

# Create __init__.py files
echo "🐍 Creating Python package files..."
touch "$NEW_DIR/core/__init__.py"
touch "$NEW_DIR/core/analysis/__init__.py"
touch "$NEW_DIR/core/risk_management/__init__.py"
touch "$NEW_DIR/core/indicators/__init__.py"
touch "$NEW_DIR/core/utils/__init__.py"

# Create new README
echo "📝 Creating new README..."
cat > "$NEW_DIR/README.md" << 'EOF'
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
EOF

echo ""
echo "✅ Reorganization Complete!"
echo ""
echo "📁 New project location: $NEW_DIR"
echo ""
echo "⚠️  Next steps:"
echo "1. Update import paths in Python files"
echo "2. Update paths in shell scripts"
echo "3. Test the system"
echo ""
echo "Old directory preserved at: $OLD_DIR"