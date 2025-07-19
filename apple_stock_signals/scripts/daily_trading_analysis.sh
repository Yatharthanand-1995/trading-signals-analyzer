#!/bin/bash

# Daily Trading Analysis Script
# This script runs the enhanced trading analyzer and archives results

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# Create daily output directory
TODAY=$(date +%Y%m%d)
OUTPUT_DIR="outputs/daily_${TODAY}"
mkdir -p "$OUTPUT_DIR"

echo -e "${GREEN}🚀 Starting Daily Trading Analysis...${NC}"
echo -e "${GREEN}============================================================${NC}"
echo "Date: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Analyzing: AAPL, GOOGL, TSLA, MSFT, UNH"
echo ""

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo -e "${YELLOW}Activating virtual environment...${NC}"
    source venv/bin/activate
else
    echo -e "${YELLOW}No virtual environment found, using system Python${NC}"
fi

# Update historical data first
echo -e "${GREEN}📊 Updating Historical Data...${NC}"
python3 data_modules/historical_data_updater.py
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Historical data updated successfully!${NC}"
else
    echo -e "${YELLOW}⚠️ Historical data update had issues, continuing with analysis...${NC}"
fi
echo ""

# Run the enhanced trading analyzer
echo -e "${GREEN}Running Enhanced Trading Analyzer...${NC}"
python3 core_scripts/enhanced_trading_analyzer.py

# Check if analysis was successful
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Analysis completed successfully!${NC}"
    
    # Move today's output files to daily directory
    if [ -f "outputs/enhanced_analysis.json" ]; then
        cp outputs/enhanced_analysis.json "$OUTPUT_DIR/analysis_${TODAY}.json"
        echo "Results saved to: $OUTPUT_DIR/analysis_${TODAY}.json"
    fi
    
    # Also run live signals
    echo ""
    echo -e "${GREEN}Generating Live Trading Signals...${NC}"
    python3 core_scripts/live_swing_signals.py
    
    if [ -f "live_signals_*.json" ]; then
        mv live_signals_*.json "$OUTPUT_DIR/"
    fi
    
    # Clean up old files (keep last 30 days)
    echo ""
    echo -e "${YELLOW}Cleaning up old files...${NC}"
    find outputs/daily_* -type d -mtime +30 -exec rm -rf {} + 2>/dev/null || true
    
    # Show summary
    echo ""
    echo -e "${GREEN}============================================================${NC}"
    echo -e "${GREEN}📊 Analysis Summary:${NC}"
    echo "Output directory: $OUTPUT_DIR"
    echo "Files generated:"
    ls -la "$OUTPUT_DIR" 2>/dev/null | grep -E "\.(json|csv)$" | awk '{print "  - " $9}'
    
else
    echo -e "${RED}❌ Analysis failed! Check the logs for details.${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}✨ Daily trading analysis complete!${NC}"
echo -e "${GREEN}============================================================${NC}"