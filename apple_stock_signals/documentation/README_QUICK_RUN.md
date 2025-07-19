# Quick Run Commands for Trading Analysis

## 📊 Run the Enhanced Trading Analyzer (with Buy/Sell/Stop Loss levels)

### One-line command to run from anywhere:

```bash
python3 ~/genai-assistant-vercel/trading-script/apple_stock_signals/enhanced_trading_analyzer.py
```

### Or use the bash script:

```bash
~/genai-assistant-vercel/trading-script/apple_stock_signals/run_trading_analysis.sh
```

### Create an alias for easy access (add to ~/.zshrc or ~/.bash_profile):

```bash
echo "alias trading='python3 ~/genai-assistant-vercel/trading-script/apple_stock_signals/enhanced_trading_analyzer.py'" >> ~/.zshrc
source ~/.zshrc
```

Then you can simply type:
```bash
trading
```

## 📈 Other Available Scripts:

1. **Simple Demo Analyzer** (uses mock data):
   ```bash
   python3 ~/genai-assistant-vercel/trading-script/apple_stock_signals/simple_apple_analyzer.py
   ```

2. **Live Data Analyzer** (basic live data):
   ```bash
   python3 ~/genai-assistant-vercel/trading-script/apple_stock_signals/live_data_analyzer.py
   ```

3. **Multi-Stock Analyzer** (mock data for 5 stocks):
   ```bash
   python3 ~/genai-assistant-vercel/trading-script/apple_stock_signals/multi_stock_analyzer.py
   ```

## 📊 What the Enhanced Analyzer Provides:

- ✅ Real-time stock prices with timestamps
- ✅ Buy/Sell signals with confidence scores
- ✅ Entry price, Stop Loss, and 3 Take Profit levels
- ✅ Risk/Reward ratios for each target
- ✅ Position sizing recommendations
- ✅ Support and Resistance levels
- ✅ Technical indicators (RSI, MACD, Moving Averages)
- ✅ Key metrics (P/E, Market Cap, 52-week range)
- ✅ Analysis for AAPL, GOOGL, TSLA, MSFT, UNH

## 📁 Output Location:

Results are saved to: `~/genai-assistant-vercel/trading-script/apple_stock_signals/outputs/enhanced_analysis.json`