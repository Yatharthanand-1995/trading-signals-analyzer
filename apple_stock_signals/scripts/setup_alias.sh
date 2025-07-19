#!/bin/bash

# Setup alias for daily trading analysis

TRADING_SCRIPT_PATH="/Users/yatharthanand/genai-assistant-vercel/trading-script/apple_stock_signals/daily_trading_analysis.sh"

echo "🔧 Setting up trading analysis alias..."

# Detect shell
if [ -n "$ZSH_VERSION" ]; then
    SHELL_RC="$HOME/.zshrc"
    echo "Detected zsh shell"
elif [ -n "$BASH_VERSION" ]; then
    SHELL_RC="$HOME/.bashrc"
    echo "Detected bash shell"
else
    echo "⚠️  Unknown shell. Please add the alias manually to your shell configuration."
    exit 1
fi

# Check if alias already exists
if grep -q "alias trade=" "$SHELL_RC" 2>/dev/null; then
    echo "⚠️  Alias 'trade' already exists in $SHELL_RC"
    echo "Updating existing alias..."
    # Remove old alias
    sed -i '' '/alias trade=/d' "$SHELL_RC"
fi

# Add new alias
echo "" >> "$SHELL_RC"
echo "# Trading Analysis Alias" >> "$SHELL_RC"
echo "alias trade='$TRADING_SCRIPT_PATH'" >> "$SHELL_RC"
echo "alias trade-log='cat /Users/yatharthanand/genai-assistant-vercel/trading-script/apple_stock_signals/outputs/daily_\$(date +%Y%m%d)/analysis_\$(date +%Y%m%d).json | python3 -m json.tool'" >> "$SHELL_RC"
echo "alias trade-dir='cd /Users/yatharthanand/genai-assistant-vercel/trading-script/apple_stock_signals'" >> "$SHELL_RC"
echo "alias trade-update='/Users/yatharthanand/genai-assistant-vercel/trading-script/apple_stock_signals/update_historical_data.sh'" >> "$SHELL_RC"
echo "alias trade-update-aapl='/Users/yatharthanand/genai-assistant-vercel/trading-script/apple_stock_signals/update_historical_data.sh -s AAPL'" >> "$SHELL_RC"
echo "alias trade-validate='/Users/yatharthanand/genai-assistant-vercel/trading-script/apple_stock_signals/update_historical_data.sh -v'" >> "$SHELL_RC"
echo "alias trade-advanced='/Users/yatharthanand/genai-assistant-vercel/trading-script/apple_stock_signals/run_advanced_analysis.sh'" >> "$SHELL_RC"
echo "alias trade-calendar='python3 /Users/yatharthanand/genai-assistant-vercel/trading-script/apple_stock_signals/advanced_features/economic_calendar/economic_events.py'" >> "$SHELL_RC"
echo "alias trade-journal='python3 /Users/yatharthanand/genai-assistant-vercel/trading-script/apple_stock_signals/advanced_features/trade_journal/trade_journal.py'" >> "$SHELL_RC"
echo "alias trade-risk='python3 /Users/yatharthanand/genai-assistant-vercel/trading-script/apple_stock_signals/advanced_features/risk_management/risk_dashboard.py'" >> "$SHELL_RC"
echo "alias trade-monitor='python3 /Users/yatharthanand/genai-assistant-vercel/trading-script/apple_stock_signals/advanced_features/data_monitoring/data_quality_monitor.py'" >> "$SHELL_RC"
echo "alias trade-health='/Users/yatharthanand/genai-assistant-vercel/trading-script/apple_stock_signals/system_health_check.sh'" >> "$SHELL_RC"
echo "alias trade-test='python3 /Users/yatharthanand/genai-assistant-vercel/trading-script/apple_stock_signals/test_system.py'" >> "$SHELL_RC"

# ML and Paper Trading aliases
echo "" >> "$SHELL_RC"
echo "# ML-Enhanced Trading and Paper Trading" >> "$SHELL_RC"
echo "alias trade-ml='/Users/yatharthanand/genai-assistant-vercel/trading-script/apple_stock_signals/scripts/ml_trading_analysis.sh'" >> "$SHELL_RC"
echo "alias trade-ml-live='/Users/yatharthanand/genai-assistant-vercel/trading-script/apple_stock_signals/scripts/ml_trading_analysis.sh --live'" >> "$SHELL_RC"
echo "alias trade-ml-train='python3 /Users/yatharthanand/genai-assistant-vercel/trading-script/apple_stock_signals/train_ml_model.py'" >> "$SHELL_RC"
echo "alias trade-paper='/Users/yatharthanand/genai-assistant-vercel/trading-script/apple_stock_signals/scripts/view_paper_trading.sh'" >> "$SHELL_RC"
echo "alias trade-paper-reset='python3 -c \"from paper_trading.paper_trader import PaperTradingAccount; PaperTradingAccount().reset_account()\"'" >> "$SHELL_RC"

echo ""
echo "✅ Aliases added successfully!"
echo ""
echo "Available commands:"
echo "  trade           - Run daily trading analysis (includes data update)"
echo "  trade-log       - View today's analysis in formatted JSON"
echo "  trade-dir       - Navigate to trading script directory"
echo "  trade-update    - Update historical data for all stocks"
echo "  trade-update-aapl - Update historical data for AAPL only"
echo "  trade-validate  - Validate data integrity for all stocks"
echo ""
echo "Advanced Features:"
echo "  trade-advanced  - Run full advanced analysis (all features)"
echo "  trade-calendar  - View economic calendar and events"
echo "  trade-journal   - View trade journal analytics"
echo "  trade-risk      - View risk management dashboard"
echo ""
echo "Monitoring & Testing:"
echo "  trade-monitor   - Run data quality monitoring"
echo "  trade-health    - Complete system health check"
echo "  trade-test      - Run comprehensive system tests"
echo ""
echo "ML-Enhanced Trading:"
echo "  trade-ml        - Run ML-enhanced analysis (paper trading)"
echo "  trade-ml-live   - Run ML-enhanced analysis (live signals)"
echo "  trade-ml-train  - Train/retrain ML model"
echo "  trade-paper     - View paper trading performance"
echo "  trade-paper-reset - Reset paper trading account"
echo ""
echo "To activate the aliases, run:"
echo "  source $SHELL_RC"
echo ""
echo "Or restart your terminal."