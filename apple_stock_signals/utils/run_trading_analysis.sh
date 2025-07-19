#!/bin/bash
# Trading Analysis Runner Script

echo "🚀 Starting Enhanced Trading Analysis..."
echo "Analyzing: AAPL, GOOGL, TSLA, MSFT, UNH"
echo ""

# Navigate to the script directory
cd "$(dirname "$0")"

# Run the enhanced trading analyzer
python3 enhanced_trading_analyzer.py

echo ""
echo "✅ Analysis complete!"