#!/bin/bash

# System Health Check Script
# Comprehensive health monitoring for the trading system

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

echo -e "${PURPLE}🏥 TRADING SYSTEM HEALTH CHECK${NC}"
echo -e "${PURPLE}============================================================${NC}"
echo "Date: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo -e "${YELLOW}Activating virtual environment...${NC}"
    source venv/bin/activate
else
    echo -e "${YELLOW}No virtual environment found, using system Python${NC}"
fi

# Create necessary directories
mkdir -p test_reports advanced_features/data_monitoring 2>/dev/null

# Function to check component
check_component() {
    local name=$1
    local command=$2
    
    echo -ne "Checking $name... "
    
    if eval "$command" &>/dev/null; then
        echo -e "${GREEN}✅ OK${NC}"
        return 0
    else
        echo -e "${RED}❌ FAIL${NC}"
        return 1
    fi
}

echo -e "\n${BLUE}1️⃣ SYSTEM PREREQUISITES${NC}"
echo -e "${BLUE}------------------------------------------------------------${NC}"

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo -e "Python Version: ${GREEN}$PYTHON_VERSION${NC}"

# Check critical directories
echo -e "\n📁 Directory Structure:"
DIRS=("historical_data" "outputs" "core_scripts" "data_modules" "advanced_features")
for dir in "${DIRS[@]}"; do
    if [ -d "$dir" ]; then
        echo -e "  $dir: ${GREEN}✅${NC}"
    else
        echo -e "  $dir: ${RED}❌ Missing${NC}"
    fi
done

echo -e "\n${BLUE}2️⃣ RUNNING SYSTEM TESTS${NC}"
echo -e "${BLUE}------------------------------------------------------------${NC}"

# Run comprehensive system test
python3 test_system.py
TEST_RESULT=$?

echo -e "\n${BLUE}3️⃣ DATA QUALITY MONITORING${NC}"
echo -e "${BLUE}------------------------------------------------------------${NC}"

# Run data quality monitor
python3 advanced_features/data_monitoring/data_quality_monitor.py
MONITOR_RESULT=$?

echo -e "\n${BLUE}4️⃣ QUICK COMPONENT CHECKS${NC}"
echo -e "${BLUE}------------------------------------------------------------${NC}"

# Test key components
check_component "Configuration" "python3 -c 'from core_scripts.config import STOCKS; print(STOCKS)'"
check_component "Data Fetcher" "python3 -c 'from data_modules.data_fetcher import DataFetcher'"
check_component "Signal Generator" "python3 -c 'from data_modules.signal_generator import SignalGenerator'"
check_component "Economic Calendar" "python3 -c 'from advanced_features.economic_calendar.economic_events import EconomicCalendar'"
check_component "Trade Journal" "python3 -c 'from advanced_features.trade_journal.trade_journal import TradeJournal'"
check_component "Risk Dashboard" "python3 -c 'from advanced_features.risk_management.risk_dashboard import RiskManagementDashboard'"

echo -e "\n${BLUE}5️⃣ API CONNECTIVITY${NC}"
echo -e "${BLUE}------------------------------------------------------------${NC}"

# Test Yahoo Finance API
echo -ne "Yahoo Finance API... "
if python3 -c "import yfinance as yf; t=yf.Ticker('SPY'); p=t.info.get('regularMarketPrice', 0); exit(0 if p > 0 else 1)" 2>/dev/null; then
    echo -e "${GREEN}✅ Connected${NC}"
else
    echo -e "${RED}❌ Not Available${NC}"
fi

echo -e "\n${BLUE}6️⃣ HISTORICAL DATA STATUS${NC}"
echo -e "${BLUE}------------------------------------------------------------${NC}"

# Check data files
for symbol in AAPL GOOGL MSFT TSLA UNH; do
    FILE="historical_data/${symbol}_historical_data.csv"
    if [ -f "$FILE" ]; then
        LINES=$(wc -l < "$FILE")
        MODIFIED=$(date -r "$FILE" "+%Y-%m-%d %H:%M")
        echo -e "${symbol}: ${GREEN}✅${NC} ($LINES records, last updated: $MODIFIED)"
    else
        echo -e "${symbol}: ${RED}❌ No data file${NC}"
    fi
done

echo -e "\n${BLUE}7️⃣ RECENT ERROR CHECK${NC}"
echo -e "${BLUE}------------------------------------------------------------${NC}"

# Check for recent errors in logs
ERROR_COUNT=0
if [ -f "advanced_features/data_monitoring/monitoring_log.json" ]; then
    # Count errors in monitoring log (simplified check)
    ERROR_COUNT=$(grep -c "error\|Error\|ERROR" advanced_features/data_monitoring/monitoring_log.json 2>/dev/null || echo "0")
fi

if [ "$ERROR_COUNT" -eq 0 ]; then
    echo -e "Recent Errors: ${GREEN}None detected${NC}"
else
    echo -e "Recent Errors: ${YELLOW}$ERROR_COUNT errors found in logs${NC}"
fi

echo -e "\n${PURPLE}============================================================${NC}"
echo -e "${PURPLE}📊 HEALTH CHECK SUMMARY${NC}"
echo -e "${PURPLE}============================================================${NC}"

# Determine overall health status
HEALTH_SCORE=100
HEALTH_STATUS="HEALTHY"
HEALTH_COLOR=$GREEN

if [ "$TEST_RESULT" -ne 0 ]; then
    HEALTH_SCORE=$((HEALTH_SCORE - 30))
fi

if [ "$MONITOR_RESULT" -ne 0 ]; then
    HEALTH_SCORE=$((HEALTH_SCORE - 20))
fi

if [ "$ERROR_COUNT" -gt 5 ]; then
    HEALTH_SCORE=$((HEALTH_SCORE - 10))
fi

# Determine status based on score
if [ "$HEALTH_SCORE" -ge 90 ]; then
    HEALTH_STATUS="EXCELLENT"
    HEALTH_COLOR=$GREEN
    EMOJI="🟢"
elif [ "$HEALTH_SCORE" -ge 70 ]; then
    HEALTH_STATUS="GOOD"
    HEALTH_COLOR=$YELLOW
    EMOJI="🟡"
elif [ "$HEALTH_SCORE" -ge 50 ]; then
    HEALTH_STATUS="WARNING"
    HEALTH_COLOR=$YELLOW
    EMOJI="🟠"
else
    HEALTH_STATUS="CRITICAL"
    HEALTH_COLOR=$RED
    EMOJI="🔴"
fi

echo -e "\n${EMOJI} Overall Health: ${HEALTH_COLOR}${HEALTH_STATUS}${NC} (Score: ${HEALTH_SCORE}/100)"

# Show recent reports
echo -e "\n${BLUE}📄 Recent Reports:${NC}"
if [ -f "test_reports/system_test_"*.txt ]; then
    echo "• System Test: $(ls -t test_reports/system_test_*.txt 2>/dev/null | head -1)"
fi
if [ -f "advanced_features/data_monitoring/monitoring_report.txt" ]; then
    echo "• Data Monitor: advanced_features/data_monitoring/monitoring_report.txt"
fi

# Recommendations
echo -e "\n${BLUE}💡 Recommendations:${NC}"
if [ "$HEALTH_SCORE" -lt 90 ]; then
    if [ "$TEST_RESULT" -ne 0 ]; then
        echo "• Fix failing tests - check test_reports/ for details"
    fi
    if [ "$MONITOR_RESULT" -ne 0 ]; then
        echo "• Address data quality issues - run 'trade-update'"
    fi
    if [ "$ERROR_COUNT" -gt 5 ]; then
        echo "• Review error logs and fix recurring issues"
    fi
else
    echo "• ✅ System is healthy and ready for trading!"
fi

echo -e "\n${PURPLE}============================================================${NC}"
echo -e "${GREEN}✨ Health check complete!${NC}"
echo -e "${PURPLE}============================================================${NC}"

# Create health status file for other scripts to check
echo "{
    \"timestamp\": \"$(date -u +"%Y-%m-%dT%H:%M:%SZ")\",
    \"health_score\": $HEALTH_SCORE,
    \"health_status\": \"$HEALTH_STATUS\",
    \"test_result\": $TEST_RESULT,
    \"monitor_result\": $MONITOR_RESULT,
    \"error_count\": $ERROR_COUNT
}" > health_status.json

exit 0