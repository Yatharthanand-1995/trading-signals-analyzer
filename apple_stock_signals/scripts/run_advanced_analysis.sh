#!/bin/bash

# Advanced Trading Analysis Script
# Runs all advanced features: Economic Calendar, Trade Journal, Risk Dashboard

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

echo -e "${BLUE}🚀 Advanced Trading Analysis System${NC}"
echo -e "${BLUE}============================================================${NC}"
echo "Date: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo -e "${YELLOW}Activating virtual environment...${NC}"
    source venv/bin/activate
else
    echo -e "${YELLOW}No virtual environment found, using system Python${NC}"
fi

# Create output directories
mkdir -p advanced_features/{economic_calendar,trade_journal,risk_management,daily_reports} 2>/dev/null

# Function to run a module and check status
run_module() {
    local module_name=$1
    local module_path=$2
    
    echo -e "\n${GREEN}Running $module_name...${NC}"
    python3 "$module_path"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ $module_name completed successfully${NC}"
        return 0
    else
        echo -e "${RED}❌ $module_name failed${NC}"
        return 1
    fi
}

# 1. Economic Calendar Analysis
run_module "Economic Calendar Integration" "advanced_features/economic_calendar/economic_events.py"

# 2. Trade Journal Analytics
run_module "Trade Journal Analytics" "advanced_features/trade_journal/trade_journal.py"

# 3. Risk Management Dashboard
run_module "Risk Management Dashboard" "advanced_features/risk_management/risk_dashboard.py"

# 4. Integrated Trading System
echo -e "\n${BLUE}============================================================${NC}"
echo -e "${BLUE}Running Integrated Trading System...${NC}"
echo -e "${BLUE}============================================================${NC}"
python3 advanced_features/integrated_trading_system.py

if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}✅ All advanced analyses completed successfully!${NC}"
    
    # Show summary of generated reports
    echo -e "\n${BLUE}📊 Generated Reports:${NC}"
    echo -e "${BLUE}------------------------------------------------------------${NC}"
    
    # List recent reports
    if [ -f "advanced_features/economic_calendar/event_report.txt" ]; then
        echo "• Economic Calendar: advanced_features/economic_calendar/event_report.txt"
    fi
    
    if [ -f "advanced_features/trade_journal/journal_report.txt" ]; then
        echo "• Trade Journal: advanced_features/trade_journal/journal_report.txt"
    fi
    
    if [ -f "advanced_features/risk_management/risk_report.txt" ]; then
        echo "• Risk Dashboard: advanced_features/risk_management/risk_report.txt"
    fi
    
    # Find today's integrated report
    TODAY=$(date +%Y%m%d)
    if [ -f "advanced_features/daily_reports/integrated_report_${TODAY}.txt" ]; then
        echo "• Integrated Report: advanced_features/daily_reports/integrated_report_${TODAY}.txt"
        
        # Show a preview of the integrated report
        echo -e "\n${BLUE}📋 Integrated Report Preview:${NC}"
        echo -e "${BLUE}------------------------------------------------------------${NC}"
        head -n 30 "advanced_features/daily_reports/integrated_report_${TODAY}.txt"
        echo -e "\n${YELLOW}... (truncated, see full report in file)${NC}"
    fi
else
    echo -e "\n${RED}❌ Some analyses failed. Check the logs for details.${NC}"
fi

echo -e "\n${BLUE}============================================================${NC}"
echo -e "${GREEN}✨ Advanced analysis complete!${NC}"
echo -e "${BLUE}============================================================${NC}"