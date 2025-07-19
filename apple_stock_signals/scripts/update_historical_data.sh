#!/bin/bash

# Historical Data Update Script
# Updates historical stock data files with latest data

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

echo -e "${BLUE}📊 Historical Data Update Tool${NC}"
echo -e "${BLUE}============================================================${NC}"
echo "This tool updates historical data for: AAPL, GOOGL, TSLA, MSFT, UNH"
echo ""

# Parse command line arguments
SYMBOL=""
VALIDATE_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -s|--symbol)
            SYMBOL="$2"
            shift 2
            ;;
        -v|--validate)
            VALIDATE_ONLY=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  -s, --symbol SYMBOL   Update specific symbol only"
            echo "  -v, --validate        Validate data integrity only"
            echo "  -h, --help           Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo -e "${YELLOW}Activating virtual environment...${NC}"
    source venv/bin/activate
else
    echo -e "${YELLOW}No virtual environment found, using system Python${NC}"
fi

# Create Python script for single symbol update if needed
if [ ! -z "$SYMBOL" ]; then
    cat > /tmp/update_single_symbol.py << 'EOF'
import sys
sys.path.append('.')
from data_modules.historical_data_updater import HistoricalDataUpdater

symbol = sys.argv[1]
updater = HistoricalDataUpdater()

if symbol.upper() in updater.stocks:
    success, message = updater.update_historical_file(symbol.upper())
    print(message)
    sys.exit(0 if success else 1)
else:
    print(f"Error: {symbol} is not in the supported stocks list")
    print(f"Supported stocks: {', '.join(updater.stocks)}")
    sys.exit(1)
EOF
fi

# Create Python script for validation if needed
if [ "$VALIDATE_ONLY" = true ]; then
    cat > /tmp/validate_data.py << 'EOF'
import sys
sys.path.append('.')
from data_modules.historical_data_updater import HistoricalDataUpdater

updater = HistoricalDataUpdater()
all_valid = True

for symbol in updater.stocks:
    if updater.validate_data_integrity(symbol):
        print(f"✓ {symbol}: Data integrity verified")
    else:
        print(f"✗ {symbol}: Data integrity issues found")
        all_valid = False

sys.exit(0 if all_valid else 1)
EOF
fi

# Execute based on options
if [ "$VALIDATE_ONLY" = true ]; then
    echo -e "${GREEN}Validating data integrity...${NC}"
    python3 /tmp/validate_data.py
    rm /tmp/validate_data.py
elif [ ! -z "$SYMBOL" ]; then
    echo -e "${GREEN}Updating historical data for $SYMBOL...${NC}"
    python3 /tmp/update_single_symbol.py "$SYMBOL"
    rm /tmp/update_single_symbol.py
else
    echo -e "${GREEN}Updating all historical data files...${NC}"
    python3 data_modules/historical_data_updater.py
fi

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✅ Operation completed successfully!${NC}"
    
    # Show update report if it exists
    if [ -f "historical_data/update_report.txt" ] && [ "$VALIDATE_ONLY" = false ]; then
        echo ""
        echo -e "${BLUE}Update Report:${NC}"
        echo -e "${BLUE}============================================================${NC}"
        tail -n 20 historical_data/update_report.txt
    fi
else
    echo ""
    echo -e "${RED}❌ Operation failed! Check the logs for details.${NC}"
    exit 1
fi

echo ""
echo -e "${BLUE}============================================================${NC}"