#!/bin/bash
# Script to fetch and update historical data

echo "📊 Fetching 3 years of historical data for analysis..."
echo ""

# Run the historical data fetcher
python3 historical_data_fetcher.py

echo ""
echo "✅ Historical data fetch complete!"
echo ""
echo "📁 Data saved in: historical_data/"
echo ""
echo "Files created:"
ls -lh historical_data/*.csv 2>/dev/null | awk '{print "  • " $9 " (" $5 ")"}'
echo ""
echo "To analyze historical data, run: python3 enhanced_trading_analyzer.py"