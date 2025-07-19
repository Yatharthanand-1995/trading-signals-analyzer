#!/bin/bash

# View Paper Trading Performance Script

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR/.."

echo -e "${BLUE}📊 Paper Trading Performance Viewer${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""

# Check for account name argument
ACCOUNT="ml_strategy"
if [ -n "$1" ]; then
    ACCOUNT="$1"
fi

# Check if virtual environment exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Generate and display report
python3 -c "
from paper_trading.paper_trader import create_paper_trading_report
create_paper_trading_report('$ACCOUNT')
"

# Show recent trades
echo -e "\n${YELLOW}Recent Trades:${NC}"
TRADES_FILE="paper_trading/trades/trades_$(date +%Y%m%d).json"
if [ -f "$TRADES_FILE" ]; then
    python3 -c "
import json
with open('$TRADES_FILE', 'r') as f:
    trades = json.load(f)
    for trade in trades[-5:]:
        print(f\"{trade['timestamp']}: {trade['action']} {trade['quantity']} {trade['symbol']} @ \${trade['price']:.2f}\")
    "
else
    echo "No trades today"
fi

# Show available reports
echo -e "\n${YELLOW}Available Performance Reports:${NC}"
ls -lt paper_trading/performance/performance_report_*.json 2>/dev/null | head -5 | awk '{print "  " $9}'

echo -e "\n${BLUE}============================================================${NC}"