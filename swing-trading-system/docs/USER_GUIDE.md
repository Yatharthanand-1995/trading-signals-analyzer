# 🚀 How to Run the Trading System

## Quick Start Commands

### 1. Navigate to the Project Directory
```bash
cd /Users/yatharthanand/genai-assistant-vercel/trading-script/apple_stock_signals
```

### 2. Run Analysis Using the Automation Script

#### Analyze Top 5 Stocks
```bash
./run_analysis.sh top5
```

#### Analyze Top 10 Stocks
```bash
./run_analysis.sh top10
```

#### Analyze Top 50 Stocks
```bash
./run_analysis.sh top50
```

#### Add a New Stock
```bash
./run_analysis.sh add AAPL
```

#### Remove a Stock
```bash
./run_analysis.sh remove TSLA
```

#### Force Run (even if no changes)
```bash
./run_analysis.sh force
```

#### Show Current Configuration
```bash
./run_analysis.sh config
```

## Manual Python Scripts

### 1. Test Complete System
```bash
cd core_scripts
python3 test_complete_system.py
```

### 2. Run Individual Analysis
```bash
cd core_scripts
python3 test_phase2_integration.py
```

### 3. Check System Health
```bash
cd core_scripts
python3 system_health_check.py
```

## Step-by-Step Example

### Example 1: Complete Analysis for Top 5 Stocks

1. **Open Terminal**

2. **Navigate to the project**:
```bash
cd /Users/yatharthanand/genai-assistant-vercel/trading-script/apple_stock_signals
```

3. **Run the analysis**:
```bash
./run_analysis.sh top5
```

4. **What happens**:
   - Fetches latest stock data
   - Runs technical analysis
   - Applies swing trading filters
   - Calculates risk management
   - Generates trade setups
   - Creates reports

5. **Check results**:
```bash
# View latest signals
ls -la outputs/

# View reports
ls -la reports/

# View backtest results
ls -la backtest_results/
```

### Example 2: Manual Python Analysis

1. **Navigate to core scripts**:
```bash
cd /Users/yatharthanand/genai-assistant-vercel/trading-script/apple_stock_signals/core_scripts
```

2. **Run Python analysis**:
```bash
# Test a single stock
python3 -c "
from swing_trading_analyzer import SwingTradingAnalyzer
from integrated_risk_management import IntegratedRiskManagement

analyzer = SwingTradingAnalyzer()
risk_mgr = IntegratedRiskManagement(10000)

# Analyze NVDA
result = analyzer.analyze_stock_complete('NVDA')
print(f'Signal: {result.get(\"signal\", \"N/A\")}')
print(f'Score: {result.get(\"total_score\", 0)}/100')
"
```

## Understanding the Output

### When you run `./run_analysis.sh top5`, you'll see:

```
==================================================
    📊 TRADING ANALYSIS SYSTEM
    Automated Pipeline Runner v1.0
==================================================

✅ Running pipeline steps:
1. Update Historical Data ✅
2. Technical Analysis ✅
3. ML Model Training ✅
4. Backtesting ✅
5. Paper Trading ✅
6. Generate Reports ✅
```

### Results Location:

- **Analysis Results**: `outputs/enhanced_analysis.json`
- **Trade Signals**: `outputs/swing_trades/`
- **Reports**: `reports/master_report_*.json`
- **Backtest Data**: `backtest_results/`

## Reading the Results

### 1. Check Latest Signals
```bash
# View the latest analysis
cat outputs/enhanced_analysis.json | python3 -m json.tool
```

### 2. View Summary Report
```bash
# Find latest report
ls -la reports/summary_*.txt

# Read it
cat reports/summary_20250720_020847.txt
```

### 3. Check Specific Stock Results
```bash
# Look for stock-specific outputs
grep -r "AAPL" outputs/
```

## Common Commands

### Check What's Configured
```bash
# View active stock list
cat config/stocks_config.json | grep -A 10 '"active": true'
```

### Switch Stock Lists
```bash
# Switch to tech stocks
./run_analysis.sh tech

# Switch to healthcare
./run_analysis.sh healthcare
```

### Run Everything Fresh
```bash
# Clean run with force flag
./run_analysis.sh force
```

## Troubleshooting

### If you get permission denied:
```bash
chmod +x run_analysis.sh
```

### If Python modules not found:
```bash
# Make sure you're in the right directory
cd /Users/yatharthanand/genai-assistant-vercel/trading-script/apple_stock_signals
```

### To see detailed output:
```bash
# Run with Python directly for more details
cd core_scripts
python3 test_complete_system.py
```

## Daily Workflow

1. **Morning Analysis**:
```bash
cd /Users/yatharthanand/genai-assistant-vercel/trading-script/apple_stock_signals
./run_analysis.sh top50
```

2. **Check Opportunities**:
```bash
# Look at latest report
cat reports/summary_*.txt | tail -50
```

3. **Focus on Specific Stocks**:
```bash
# Add stocks you're interested in
./run_analysis.sh add NVDA
./run_analysis.sh add GOOGL
```

4. **Run Focused Analysis**:
```bash
./run_analysis.sh force
```

## Key Features You're Using

1. **Phase 1**: Multi-timeframe analysis, volume confirmation, entry filters
2. **Phase 2**: Dynamic stops, position sizing, profit targets
3. **Risk Management**: Max 2% risk per trade, 6% portfolio heat
4. **Swing Trading**: Optimized for 2-15 day holds

---

💡 **Pro Tip**: Save this command as an alias in your shell:
```bash
echo "alias trade='cd /Users/yatharthanand/genai-assistant-vercel/trading-script/apple_stock_signals && ./run_analysis.sh'" >> ~/.zshrc
source ~/.zshrc

# Now just type:
trade top5
```