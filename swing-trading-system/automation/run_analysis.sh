#!/bin/bash

# Master Trading Analysis Runner
# One command to run complete analysis

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Display banner
echo -e "${BLUE}"
echo "=================================================="
echo "    📊 TRADING ANALYSIS SYSTEM"
echo "    Automated Pipeline Runner v1.0"
echo "=================================================="
echo -e "${NC}"

# Check Python installation
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 is not installed${NC}"
    exit 1
fi

# Parse command line arguments
case "$1" in
    "top5")
        echo -e "${GREEN}Switching to Top 5 stocks...${NC}"
        python3 automation/pipeline.py --list top_5
        ;;
    "top10")
        echo -e "${GREEN}Switching to Top 10 stocks...${NC}"
        python3 automation/pipeline.py --list top_10
        ;;
    "top50")
        echo -e "${GREEN}Switching to Top 50 stocks...${NC}"
        python3 automation/pipeline.py --list top_50
        ;;
    "tech")
        echo -e "${GREEN}Switching to Tech sector stocks...${NC}"
        python3 automation/pipeline.py --list tech_sector
        ;;
    "health")
        echo -e "${GREEN}Switching to Healthcare sector stocks...${NC}"
        python3 automation/pipeline.py --list healthcare_sector
        ;;
    "custom")
        echo -e "${GREEN}Switching to Custom stock list...${NC}"
        python3 automation/pipeline.py --list custom
        ;;
    "add")
        if [ -z "$2" ]; then
            echo -e "${RED}Usage: ./run_analysis.sh add SYMBOL${NC}"
            exit 1
        fi
        echo -e "${GREEN}Adding $2 to custom list...${NC}"
        python3 automation/pipeline.py --add-stock "$2"
        ;;
    "remove")
        if [ -z "$2" ]; then
            echo -e "${RED}Usage: ./run_analysis.sh remove SYMBOL${NC}"
            exit 1
        fi
        echo -e "${GREEN}Removing $2 from custom list...${NC}"
        python3 automation/pipeline.py --remove-stock "$2"
        ;;
    "config")
        echo -e "${BLUE}Current Configuration:${NC}"
        python3 automation/run_pipeline.py --show-config
        ;;
    "force")
        echo -e "${YELLOW}Force running complete pipeline...${NC}"
        python3 automation/pipeline.py --force
        ;;
    "quick")
        echo -e "${GREEN}Running quick analysis (current list)...${NC}"
        python3 core/enhanced_trading_analyzer.py
        ;;
    "help"|"-h"|"--help")
        echo "Usage: ./run_analysis.sh [command] [options]"
        echo ""
        echo "Commands:"
        echo "  (no args)    Run analysis with current stock list"
        echo "  top5         Switch to and analyze top 5 stocks"
        echo "  top10        Switch to and analyze top 10 stocks"
        echo "  top50        Switch to and analyze top 50 stocks"
        echo "  tech         Switch to and analyze tech sector"
        echo "  health       Switch to and analyze healthcare sector"
        echo "  custom       Switch to and analyze custom list"
        echo "  add SYMBOL   Add stock to custom list"
        echo "  remove SYMBOL Remove stock from custom list"
        echo "  config       Show current configuration"
        echo "  force        Force run complete pipeline"
        echo "  quick        Run quick analysis only"
        echo "  help         Show this help message"
        echo ""
        echo "Examples:"
        echo "  ./run_analysis.sh                # Run with current list"
        echo "  ./run_analysis.sh top50          # Switch to top 50 and run"
        echo "  ./run_analysis.sh add AAPL       # Add AAPL to custom list"
        echo "  ./run_analysis.sh custom         # Run custom list"
        ;;
    *)
        echo -e "${GREEN}Running analysis with current stock list...${NC}"
        python3 automation/pipeline.py
        ;;
esac

# Check if analysis completed successfully
if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}✅ Analysis completed successfully!${NC}"
    
    # Show latest results summary
    if [ -f "data/analysis_results/enhanced_analysis.json" ]; then
        echo -e "\n${BLUE}📊 Latest Results Summary:${NC}"
        python3 -c "
import json
with open('data/analysis_results/enhanced_analysis.json', 'r') as f:
    data = json.load(f)
    if 'results' in data:
        signals = {}
        for stock in data['results'][:10]:
            signal = stock.get('signal', 'UNKNOWN')
            signals[signal] = signals.get(signal, 0) + 1
        
        print('Signal Distribution:')
        for signal, count in sorted(signals.items()):
            print(f'  {signal}: {count} stocks')
        "
    fi
    
    echo -e "\n${BLUE}📁 Results saved in:${NC}"
    echo "  • data/analysis_results/         - Analysis results"
    echo "  • data/backtest_results/ - Backtesting data"
    echo "  • data/reports/         - Comprehensive reports"
    echo "  • data/historical/ - Stock data cache"
else
    echo -e "\n${RED}❌ Analysis failed! Check the logs for details.${NC}"
    exit 1
fi

echo -e "\n${YELLOW}⚠️  Disclaimer: For educational purposes only. Not financial advice.${NC}"