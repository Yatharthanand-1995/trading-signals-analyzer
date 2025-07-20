#!/usr/bin/env python3
"""
Enhanced Trading Analyzer with Entry/Exit Points
Provides buy, sell, stop loss, and take profit levels
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import time

class EnhancedTradingAnalyzer:
    def __init__(self):
        # Updated stock list to include Microsoft
        self.symbols = ['AAPL', 'GOOGL', 'TSLA', 'MSFT', 'UNH']
        
    def fetch_stock_data(self, symbol):
        """Fetch comprehensive stock data"""
        try:
            ticker = yf.Ticker(symbol)
            
            # Get historical data for calculations
            history = ticker.history(period="3mo")
            week_history = ticker.history(period="5d")
            
            if history.empty:
                return None
                
            # Current data
            info = ticker.info
            current_price = week_history['Close'].iloc[-1]
            previous_close = week_history['Close'].iloc[-2] if len(week_history) > 1 else current_price
            
            # Get company name
            company_names = {
                'AAPL': 'Apple Inc.',
                'GOOGL': 'Alphabet Inc.',
                'TSLA': 'Tesla Inc.',
                'MSFT': 'Microsoft Corporation',
                'UNH': 'UnitedHealth Group Inc.'
            }
            
            data = {
                'symbol': symbol,
                'company_name': company_names.get(symbol, info.get('longName', symbol)),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'current_price': round(current_price, 2),
                'previous_close': round(previous_close, 2),
                'open': round(week_history['Open'].iloc[-1], 2),
                'high': round(week_history['High'].iloc[-1], 2),
                'low': round(week_history['Low'].iloc[-1], 2),
                'volume': int(week_history['Volume'].iloc[-1]),
                'change': round(current_price - previous_close, 2),
                'change_percent': round(((current_price - previous_close) / previous_close) * 100, 2),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': round(info.get('trailingPE', 0), 2),
                'dividend_yield': round(info.get('dividendYield', 0) * 100, 2) if info.get('dividendYield') else 0,
                '52_week_high': round(info.get('fiftyTwoWeekHigh', 0), 2),
                '52_week_low': round(info.get('fiftyTwoWeekLow', 0), 2),
                'avg_volume': info.get('averageVolume', 0),
                'beta': round(info.get('beta', 0), 2),
                'eps': round(info.get('trailingEps', 0), 2),
                'forward_pe': round(info.get('forwardPE', 0), 2),
                'price_to_book': round(info.get('priceToBook', 0), 2),
                'history': history
            }
            
            return data
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None
    
    def calculate_technical_indicators(self, data):
        """Calculate comprehensive technical indicators"""
        history = data['history']
        close_prices = history['Close']
        
        # RSI
        delta = close_prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # Moving averages
        sma_20 = close_prices.rolling(window=20).mean()
        sma_50 = close_prices.rolling(window=50).mean()
        sma_200 = close_prices.rolling(window=200).mean()
        ema_12 = close_prices.ewm(span=12).mean()
        ema_26 = close_prices.ewm(span=26).mean()
        
        # MACD
        macd = ema_12 - ema_26
        macd_signal = macd.ewm(span=9).mean()
        macd_histogram = macd - macd_signal
        
        # Bollinger Bands
        bb_sma = close_prices.rolling(window=20).mean()
        bb_std = close_prices.rolling(window=20).std()
        bb_upper = bb_sma + (bb_std * 2)
        bb_lower = bb_sma - (bb_std * 2)
        
        # ATR (Average True Range)
        high_low = history['High'] - history['Low']
        high_close = np.abs(history['High'] - history['Close'].shift())
        low_close = np.abs(history['Low'] - history['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        atr = true_range.rolling(14).mean()
        
        # Stochastic
        low_14 = history['Low'].rolling(window=14).min()
        high_14 = history['High'].rolling(window=14).max()
        stoch_k = 100 * ((history['Close'] - low_14) / (high_14 - low_14))
        stoch_d = stoch_k.rolling(window=3).mean()
        
        # Support and Resistance levels
        pivots = self.calculate_pivot_points(history)
        
        # Fibonacci levels
        fib_levels = self.calculate_fibonacci_levels(history)
        
        return {
            'RSI': round(rsi.iloc[-1], 2),
            'MACD': round(macd.iloc[-1], 3),
            'MACD_signal': round(macd_signal.iloc[-1], 3),
            'MACD_histogram': round(macd_histogram.iloc[-1], 3),
            'SMA_20': round(sma_20.iloc[-1], 2),
            'SMA_50': round(sma_50.iloc[-1], 2),
            'SMA_200': round(sma_200.iloc[-1], 2) if not pd.isna(sma_200.iloc[-1]) else None,
            'EMA_12': round(ema_12.iloc[-1], 2),
            'EMA_26': round(ema_26.iloc[-1], 2),
            'BB_upper': round(bb_upper.iloc[-1], 2),
            'BB_middle': round(bb_sma.iloc[-1], 2),
            'BB_lower': round(bb_lower.iloc[-1], 2),
            'ATR': round(atr.iloc[-1], 2),
            'ATR_percent': round((atr.iloc[-1] / close_prices.iloc[-1]) * 100, 2),
            'Stoch_K': round(stoch_k.iloc[-1], 2),
            'Stoch_D': round(stoch_d.iloc[-1], 2),
            'pivots': pivots,
            'fibonacci': fib_levels
        }
    
    def calculate_pivot_points(self, history):
        """Calculate pivot points for support/resistance"""
        last_day = history.iloc[-1]
        prev_day = history.iloc[-2] if len(history) > 1 else last_day
        
        # Classic Pivot Points
        pivot = (last_day['High'] + last_day['Low'] + last_day['Close']) / 3
        r1 = (2 * pivot) - last_day['Low']
        r2 = pivot + (last_day['High'] - last_day['Low'])
        r3 = r1 + (last_day['High'] - last_day['Low'])
        s1 = (2 * pivot) - last_day['High']
        s2 = pivot - (last_day['High'] - last_day['Low'])
        s3 = s1 - (last_day['High'] - last_day['Low'])
        
        return {
            'pivot': round(pivot, 2),
            'resistance_1': round(r1, 2),
            'resistance_2': round(r2, 2),
            'resistance_3': round(r3, 2),
            'support_1': round(s1, 2),
            'support_2': round(s2, 2),
            'support_3': round(s3, 2)
        }
    
    def calculate_fibonacci_levels(self, history):
        """Calculate Fibonacci retracement levels"""
        recent_high = history['High'].rolling(window=20).max().iloc[-1]
        recent_low = history['Low'].rolling(window=20).min().iloc[-1]
        diff = recent_high - recent_low
        
        return {
            'high': round(recent_high, 2),
            'low': round(recent_low, 2),
            'fib_236': round(recent_high - (diff * 0.236), 2),
            'fib_382': round(recent_high - (diff * 0.382), 2),
            'fib_500': round(recent_high - (diff * 0.500), 2),
            'fib_618': round(recent_high - (diff * 0.618), 2),
            'fib_786': round(recent_high - (diff * 0.786), 2)
        }
    
    def calculate_trading_levels(self, data, indicators):
        """Calculate buy, sell, stop loss, and take profit levels"""
        current_price = data['current_price']
        atr = indicators['ATR']
        
        # Risk management parameters
        risk_reward_ratio = 2.0  # 1:2 risk/reward
        position_risk_percent = 2.0  # 2% risk per trade
        
        # Determine trend and signal
        signal = self.determine_signal(data, indicators)
        
        if signal['action'] in ['BUY', 'STRONG_BUY']:
            # For bullish signals
            entry_price = current_price
            stop_loss = current_price - (atr * 2)  # 2 ATR stop
            take_profit_1 = current_price + (atr * 2)  # 1:1 R/R
            take_profit_2 = current_price + (atr * 4)  # 1:2 R/R
            take_profit_3 = current_price + (atr * 6)  # 1:3 R/R
            
            # Alternative stop using support levels
            alt_stop = max(indicators['pivots']['support_1'], indicators['BB_lower'])
            stop_loss = max(stop_loss, alt_stop - (atr * 0.5))
            
        elif signal['action'] in ['SELL', 'STRONG_SELL']:
            # For bearish signals
            entry_price = current_price
            stop_loss = current_price + (atr * 2)  # 2 ATR stop
            take_profit_1 = current_price - (atr * 2)  # 1:1 R/R
            take_profit_2 = current_price - (atr * 4)  # 1:2 R/R
            take_profit_3 = current_price - (atr * 6)  # 1:3 R/R
            
            # Alternative stop using resistance levels
            alt_stop = min(indicators['pivots']['resistance_1'], indicators['BB_upper'])
            stop_loss = min(stop_loss, alt_stop + (atr * 0.5))
            
        else:  # HOLD
            entry_price = current_price
            stop_loss = current_price - (atr * 2)
            take_profit_1 = current_price + (atr * 2)
            take_profit_2 = current_price + (atr * 4)
            take_profit_3 = current_price + (atr * 6)
        
        # Calculate position sizing
        risk_amount = abs(entry_price - stop_loss)
        risk_percent = (risk_amount / entry_price) * 100
        
        # For $10,000 account
        account_size = 10000
        position_risk_dollars = account_size * (position_risk_percent / 100)
        shares_to_buy = int(position_risk_dollars / risk_amount)
        position_value = shares_to_buy * entry_price
        
        return {
            'signal': signal,
            'entry_price': round(entry_price, 2),
            'stop_loss': round(stop_loss, 2),
            'take_profit_1': round(take_profit_1, 2),
            'take_profit_2': round(take_profit_2, 2),
            'take_profit_3': round(take_profit_3, 2),
            'risk_amount': round(risk_amount, 2),
            'risk_percent': round(risk_percent, 2),
            'potential_profit_1': round(take_profit_1 - entry_price, 2),
            'potential_profit_2': round(take_profit_2 - entry_price, 2),
            'potential_profit_3': round(take_profit_3 - entry_price, 2),
            'risk_reward_1': round(abs(take_profit_1 - entry_price) / risk_amount, 2),
            'risk_reward_2': round(abs(take_profit_2 - entry_price) / risk_amount, 2),
            'risk_reward_3': round(abs(take_profit_3 - entry_price) / risk_amount, 2),
            'position_sizing': {
                'shares_to_buy': shares_to_buy,
                'position_value': round(position_value, 2),
                'position_risk_dollars': round(position_risk_dollars, 2),
                'position_risk_percent': position_risk_percent
            }
        }
    
    def determine_signal(self, data, indicators):
        """Determine trading signal based on multiple factors"""
        score = 50  # Start neutral
        reasons = []
        
        # RSI Analysis
        rsi = indicators['RSI']
        if rsi < 30:
            score += 20
            reasons.append(f"RSI oversold ({rsi})")
        elif rsi > 70:
            score -= 20
            reasons.append(f"RSI overbought ({rsi})")
        elif rsi < 40:
            score += 10
            reasons.append(f"RSI approaching oversold ({rsi})")
        elif rsi > 60:
            score -= 10
            reasons.append(f"RSI approaching overbought ({rsi})")
        
        # MACD Analysis
        if indicators['MACD'] > indicators['MACD_signal']:
            score += 15
            reasons.append("MACD bullish crossover")
            if indicators['MACD_histogram'] > 0:
                score += 5
                reasons.append("MACD histogram positive")
        else:
            score -= 15
            reasons.append("MACD bearish")
        
        # Moving Average Analysis
        current_price = data['current_price']
        if indicators['SMA_20'] and indicators['SMA_50']:
            if current_price > indicators['SMA_20'] > indicators['SMA_50']:
                score += 20
                reasons.append("Price above key moving averages")
            elif current_price < indicators['SMA_20'] < indicators['SMA_50']:
                score -= 20
                reasons.append("Price below key moving averages")
        
        # Bollinger Bands
        if current_price <= indicators['BB_lower']:
            score += 10
            reasons.append("Price at lower Bollinger Band")
        elif current_price >= indicators['BB_upper']:
            score -= 10
            reasons.append("Price at upper Bollinger Band")
        
        # Stochastic
        if indicators['Stoch_K'] < 20:
            score += 10
            reasons.append("Stochastic oversold")
        elif indicators['Stoch_K'] > 80:
            score -= 10
            reasons.append("Stochastic overbought")
        
        # Volume analysis
        if data['volume'] > data['avg_volume'] * 1.5:
            score += 5
            reasons.append("High volume confirmation")
        
        # Determine action
        if score >= 70:
            action = 'STRONG_BUY'
        elif score >= 60:
            action = 'BUY'
        elif score <= 30:
            action = 'STRONG_SELL'
        elif score <= 40:
            action = 'SELL'
        else:
            action = 'HOLD'
        
        return {
            'action': action,
            'score': score,
            'reasons': reasons
        }
    
    def analyze_stock(self, symbol):
        """Complete analysis for a single stock"""
        print(f"\n🔍 Analyzing {symbol}...")
        
        # Fetch data
        data = self.fetch_stock_data(symbol)
        if not data:
            return None
        
        # Calculate indicators
        indicators = self.calculate_technical_indicators(data)
        
        # Calculate trading levels
        trading_levels = self.calculate_trading_levels(data, indicators)
        
        # Remove history from data to avoid serialization issues
        analysis_data = data.copy()
        del analysis_data['history']
        
        return {
            'data': analysis_data,
            'indicators': indicators,
            'trading': trading_levels
        }
    
    def display_results(self, results):
        """Display comprehensive analysis results"""
        print("\n" + "="*120)
        print(f"📊 ENHANCED TRADING ANALYSIS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*120)
        
        for symbol, analysis in results.items():
            if not analysis:
                continue
            
            data = analysis['data']
            indicators = analysis['indicators']
            trading = analysis['trading']
            signal = trading['signal']
            
            # Determine emoji
            if 'BUY' in signal['action']:
                emoji = '🟢🟢' if 'STRONG' in signal['action'] else '🟢'
            elif 'SELL' in signal['action']:
                emoji = '🔴🔴' if 'STRONG' in signal['action'] else '🔴'
            else:
                emoji = '🟡'
            
            print(f"\n{'='*120}")
            print(f"📈 {symbol} - {data['company_name']}")
            print(f"{'='*120}")
            
            # Current Market Data
            print(f"\n📊 MARKET DATA (as of {data['timestamp']}):")
            print(f"Current Price: ${data['current_price']:.2f}")
            print(f"Change: ${data['change']:.2f} ({data['change_percent']:+.2f}%)")
            print(f"Day Range: ${data['low']:.2f} - ${data['high']:.2f}")
            print(f"Volume: {data['volume']:,} (Avg: {data['avg_volume']:,})")
            print(f"52-Week Range: ${data['52_week_low']:.2f} - ${data['52_week_high']:.2f}")
            
            # Key Metrics
            print(f"\n📈 KEY METRICS:")
            print(f"Market Cap: ${data['market_cap']/1e9:.1f}B")
            print(f"P/E Ratio: {data['pe_ratio']}")
            print(f"Forward P/E: {data['forward_pe']}")
            print(f"EPS: ${data['eps']}")
            print(f"Beta: {data['beta']}")
            print(f"Dividend Yield: {data['dividend_yield']:.2f}%")
            
            # Technical Indicators
            print(f"\n📊 TECHNICAL INDICATORS:")
            print(f"RSI(14): {indicators['RSI']}")
            print(f"MACD: {indicators['MACD']} (Signal: {indicators['MACD_signal']})")
            print(f"Stochastic: K={indicators['Stoch_K']}, D={indicators['Stoch_D']}")
            print(f"ATR: ${indicators['ATR']:.2f} ({indicators['ATR_percent']:.1f}% of price)")
            print(f"Moving Averages: SMA20=${indicators['SMA_20']:.2f}, SMA50=${indicators['SMA_50']:.2f}")
            print(f"Bollinger Bands: Lower=${indicators['BB_lower']:.2f}, Upper=${indicators['BB_upper']:.2f}")
            
            # Support & Resistance
            print(f"\n📊 SUPPORT & RESISTANCE LEVELS:")
            pivots = indicators['pivots']
            print(f"Resistance: R3=${pivots['resistance_3']:.2f}, R2=${pivots['resistance_2']:.2f}, R1=${pivots['resistance_1']:.2f}")
            print(f"Pivot: ${pivots['pivot']:.2f}")
            print(f"Support: S1=${pivots['support_1']:.2f}, S2=${pivots['support_2']:.2f}, S3=${pivots['support_3']:.2f}")
            
            # Trading Signal
            print(f"\n🎯 TRADING SIGNAL: {emoji} {signal['action']} (Score: {signal['score']}/100)")
            print("Reasons:")
            for reason in signal['reasons']:
                print(f"  • {reason}")
            
            # Trading Levels
            print(f"\n💰 TRADING LEVELS:")
            print(f"Entry Price: ${trading['entry_price']:.2f}")
            print(f"Stop Loss: ${trading['stop_loss']:.2f} (Risk: {trading['risk_percent']:.1f}%)")
            print(f"Take Profit 1: ${trading['take_profit_1']:.2f} (R/R: {trading['risk_reward_1']:.1f})")
            print(f"Take Profit 2: ${trading['take_profit_2']:.2f} (R/R: {trading['risk_reward_2']:.1f})")
            print(f"Take Profit 3: ${trading['take_profit_3']:.2f} (R/R: {trading['risk_reward_3']:.1f})")
            
            # Position Sizing
            pos = trading['position_sizing']
            print(f"\n📊 POSITION SIZING (for $10,000 account):")
            print(f"Shares to Buy: {pos['shares_to_buy']}")
            print(f"Position Value: ${pos['position_value']:.2f}")
            print(f"Risk Amount: ${pos['position_risk_dollars']:.2f} ({pos['position_risk_percent']:.1f}% of account)")
    
    def save_results(self, results):
        """Save comprehensive results to JSON"""
        output = {
            'analysis_date': datetime.now().strftime('%Y-%m-%d'),
            'analysis_time': datetime.now().strftime('%H:%M:%S'),
            'stocks': {}
        }
        
        for symbol, analysis in results.items():
            if analysis:
                output['stocks'][symbol] = {
                    'company_name': analysis['data']['company_name'],
                    'current_price': analysis['data']['current_price'],
                    'change': analysis['data']['change'],
                    'change_percent': analysis['data']['change_percent'],
                    'signal': analysis['trading']['signal']['action'],
                    'score': analysis['trading']['signal']['score'],
                    'entry_price': analysis['trading']['entry_price'],
                    'stop_loss': analysis['trading']['stop_loss'],
                    'take_profit_1': analysis['trading']['take_profit_1'],
                    'take_profit_2': analysis['trading']['take_profit_2'],
                    'take_profit_3': analysis['trading']['take_profit_3'],
                    'risk_reward_ratios': {
                        'tp1': analysis['trading']['risk_reward_1'],
                        'tp2': analysis['trading']['risk_reward_2'],
                        'tp3': analysis['trading']['risk_reward_3']
                    },
                    'position_sizing': analysis['trading']['position_sizing'],
                    'technical_indicators': {
                        'RSI': analysis['indicators']['RSI'],
                        'MACD': analysis['indicators']['MACD'],
                        'SMA_20': analysis['indicators']['SMA_20'],
                        'SMA_50': analysis['indicators']['SMA_50']
                    },
                    'support_resistance': analysis['indicators']['pivots']
                }
        
        try:
            with open('data/analysis_results/enhanced_analysis.json', 'w') as f:
                json.dump(output, f, indent=2)
            print(f"\n💾 Results saved to data/analysis_results/enhanced_analysis.json")
        except:
            pass
    
    def run_analysis(self):
        """Run complete analysis for all stocks"""
        results = {}
        
        for symbol in self.symbols:
            analysis = self.analyze_stock(symbol)
            if analysis:
                results[symbol] = analysis
            time.sleep(0.5)  # Rate limiting
        
        self.display_results(results)
        self.save_results(results)
        
        return results

def main():
    """Main function"""
    print("\n🚀 Enhanced Trading Analyzer")
    print("="*60)
    print("Analyzing: AAPL, GOOGL, TSLA, MSFT, UNH")
    print("Calculating entry/exit points and position sizing...")
    print("="*60)
    
    analyzer = EnhancedTradingAnalyzer()
    results = analyzer.run_analysis()
    
    print("\n" + "="*120)
    print("⚠️ RISK DISCLAIMER:")
    print("="*120)
    print("• All trading involves risk. Past performance doesn't guarantee future results.")
    print("• Use stop losses and proper position sizing to manage risk.")
    print("• These are analytical tools only - not financial advice.")
    print("• Always verify prices with your broker before trading.")
    print("="*120)
    
    # Load and return the saved results
    try:
        with open('data/analysis_results/enhanced_analysis.json', 'r') as f:
            return json.load(f)
    except:
        return results

if __name__ == "__main__":
    main()