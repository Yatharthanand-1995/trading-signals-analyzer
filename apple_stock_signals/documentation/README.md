# 🍎 Apple Stock Signal Generator

A comprehensive Python application that analyzes Apple (AAPL) stock using technical indicators, news sentiment, and fundamental analysis to generate clear buy/sell signals with data verification.

## 🚀 Features

- **Real-time Data Collection**: Fetches current AAPL stock data from multiple sources
- **Data Verification System**: Cross-verifies prices and detects anomalies
- **Technical Analysis**: 20+ indicators including RSI, MACD, Moving Averages, Bollinger Bands
- **Sentiment Analysis**: News and social media sentiment from multiple sources
- **Fundamental Analysis**: P/E ratio, growth metrics, profitability indicators
- **Clear Signal Generation**: STRONG_BUY, BUY, HOLD, SELL, STRONG_SELL with confidence scores
- **Risk Management**: Automatic stop-loss and take-profit calculations

## 📋 Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Internet connection for data fetching

### Installing TA-Lib

TA-Lib is required for technical indicators. Installation varies by OS:

**macOS:**
```bash
brew install ta-lib
```

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install ta-lib
```

**Windows:**
Download and install from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib

## 🔧 Installation

1. Clone or download this repository:
```bash
cd /Users/yatharthanand/genai-assistant-vercel/trading-script/apple_stock_signals
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Get free API keys for enhanced functionality:
   - **News API**: https://newsapi.org (Required for real news sentiment)
   - **Alpha Vantage**: https://www.alphavantage.co (Optional for price verification)

4. Set environment variables (optional):
```bash
export NEWS_API_KEY="your_news_api_key_here"
export ALPHA_VANTAGE_KEY="your_alpha_vantage_key_here"
```

## 🚀 Usage

Run the analysis:
```bash
python main.py
```

The script will:
1. Fetch current Apple stock data
2. Verify data accuracy from multiple sources
3. Calculate technical indicators
4. Analyze news and social sentiment
5. Generate a trading signal with confidence score
6. Save results to the `outputs/` directory

## 📊 Understanding the Output

### Signal Types
- **STRONG_BUY**: Score ≥ 70 - Multiple strong bullish indicators
- **BUY**: Score ≥ 60 - Bullish indicators outweigh bearish
- **HOLD**: Score 40-59 - Mixed signals
- **SELL**: Score 30-39 - Bearish indicators outweigh bullish
- **STRONG_SELL**: Score < 30 - Multiple strong bearish indicators

### Data Quality Indicators
- 🟢 **EXCELLENT** (90-100%): Proceed with full confidence
- 🟡 **GOOD** (75-89%): Safe to proceed
- 🟠 **FAIR** (50-74%): Proceed with caution
- 🔴 **POOR** (<50%): Manual verification recommended

### Key Metrics Explained
- **Signal Confidence**: How confident the system is in the signal (0-100%)
- **Data Quality**: Reliability of the data sources (0-100%)
- **Adjusted Confidence**: Signal confidence × Data quality
- **Risk/Reward Ratio**: Potential profit ÷ Potential loss

## 📁 Output Files

Results are saved in the `outputs/` directory:
- **CSV File**: Summary of analysis with key metrics
- **JSON File**: Detailed analysis data including all calculations

## ⚙️ Configuration

Edit `config.py` to customize:
- Analysis thresholds
- Technical indicator periods
- Signal generation weights
- Data verification settings

## 🔍 Data Verification Features

The system automatically:
- Compares prices from multiple sources
- Detects unusual price movements
- Validates data ranges
- Checks data freshness
- Provides confidence scores

## ⚠️ Important Notes

1. **No API Keys Required**: The system works without API keys using mock data
2. **Real Data Recommended**: For best results, use real API keys
3. **Risk Warning**: This tool is for informational purposes only
4. **Not Financial Advice**: Always do your own research

## 🐛 Troubleshooting

### Common Issues

1. **TA-Lib Installation Error**:
   - Make sure to install system dependencies first (see Prerequisites)
   - On M1 Macs: `arch -arm64 brew install ta-lib`

2. **No Data Available**:
   - Check internet connection
   - Verify API keys are set correctly
   - Market might be closed (weekends/holidays)

3. **Import Errors**:
   - Run `pip install -r requirements.txt` again
   - Check Python version (3.8+ required)

## 📈 Example Output

```
🍎 APPLE STOCK ANALYSIS STARTING...
============================================================

🔄 Fetching Apple Stock Data...
🔍 Verifying data accuracy...
📰 Fetching Apple news...
💬 Fetching social media sentiment...

🟢 DATA QUALITY: EXCELLENT
📊 CONFIDENCE SCORE: 95.0/100

📈 CURRENT STOCK DATA:
Current Price: $182.50
Volume Ratio: 1.25x (vs 20-day avg)

🎯 FINAL TRADING RECOMMENDATION
============================================================
🟢 SIGNAL: BUY
📊 SIGNAL CONFIDENCE: 72.5%
🔍 DATA QUALITY: 95.0%
📈 ADJUSTED CONFIDENCE: 68.9%

💰 PRICE TARGETS:
Entry Price: $182.50
Stop Loss: $176.80
Take Profit 1: $194.90
Risk/Reward Ratio: 2.18:1
```

## 🤝 Support

For issues or questions:
1. Check the troubleshooting section
2. Review the log file: `apple_analysis.log`
3. Ensure all dependencies are correctly installed

## 📄 License

This project is for educational and informational purposes only. Use at your own risk.