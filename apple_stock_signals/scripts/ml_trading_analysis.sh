#!/bin/bash

# ML-Enhanced Trading Analysis Script
# Runs trading analysis with machine learning and paper trading

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR/.."

echo -e "${BLUE}🚀 ML-Enhanced Trading Analysis${NC}"
echo -e "${BLUE}============================================================${NC}"
echo "Date: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# Check for command line arguments
PAPER_TRADING=true
TRAIN_MODEL=false
SYMBOL=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --live)
            PAPER_TRADING=false
            shift
            ;;
        --train)
            TRAIN_MODEL=true
            shift
            ;;
        --symbol)
            SYMBOL="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--live] [--train] [--symbol SYMBOL]"
            exit 1
            ;;
    esac
done

# Display mode
if [ "$PAPER_TRADING" = true ]; then
    echo -e "${YELLOW}Mode: Paper Trading (Safe Mode)${NC}"
else
    echo -e "${RED}Mode: Live Trading Signals${NC}"
fi

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo -e "${YELLOW}Activating virtual environment...${NC}"
    source venv/bin/activate
else
    echo -e "${YELLOW}No virtual environment found, using system Python${NC}"
fi

# Check if ML model exists
if [ ! -f "ml_models/saved_models/ml_model_latest.pkl" ] || [ "$TRAIN_MODEL" = true ]; then
    echo -e "\n${YELLOW}Training ML Model...${NC}"
    python3 train_ml_model.py
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}❌ ML model training failed!${NC}"
        exit 1
    fi
    echo ""
fi

# Update historical data
echo -e "${GREEN}📊 Updating Historical Data...${NC}"
python3 data_modules/historical_data_updater.py

# Run ML-enhanced analysis
echo -e "\n${GREEN}Running ML-Enhanced Analysis...${NC}"

if [ "$PAPER_TRADING" = true ]; then
    if [ -n "$SYMBOL" ]; then
        python3 core_scripts/ml_enhanced_analyzer.py --symbol "$SYMBOL"
    else
        python3 core_scripts/ml_enhanced_analyzer.py
    fi
else
    if [ -n "$SYMBOL" ]; then
        python3 core_scripts/ml_enhanced_analyzer.py --live --symbol "$SYMBOL"
    else
        python3 core_scripts/ml_enhanced_analyzer.py --live
    fi
fi

# Check if analysis was successful
if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}✅ ML-Enhanced analysis completed successfully!${NC}"
    
    # Show latest results location
    echo -e "\nResults saved to:"
    echo -e "  - ML Analysis: ml_models/analysis_results/ml_analysis_latest.json"
    
    if [ "$PAPER_TRADING" = true ]; then
        echo -e "  - Paper Trading: paper_trading/performance/"
    fi
else
    echo -e "\n${RED}❌ Analysis failed!${NC}"
    exit 1
fi

echo -e "\n${BLUE}============================================================${NC}"
echo -e "${GREEN}✨ ML-Enhanced trading analysis complete!${NC}"
echo -e "${BLUE}============================================================${NC}"