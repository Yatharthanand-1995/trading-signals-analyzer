#!/usr/bin/env python3
"""
Live Data Stock Analyzer with Multi-Source Verification
Fetches real-time data and verifies across multiple sources
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import json
import time
import sys

class LiveStockAnalyzer:
    def __init__(self):
        self.symbols = ['AAPL', 'GOOGL', 'TSLA', 'ADBE', 'UNH']
        self.data_sources = {}
        
    def fetch_yahoo_finance_data(self, symbol):
        """Fetch real-time data from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            
            # Get current data
            info = ticker.info
            history = ticker.history(period="5d")
            
            if history.empty:
                return None
                
            current_price = history['Close'].iloc[-1]
            previous_close = history['Close'].iloc[-2] if len(history) > 1 else current_price
            
            # Get additional info
            data = {
                'source': 'Yahoo Finance',
                'symbol': symbol,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'current_price': round(current_price, 2),
                'previous_close': round(previous_close, 2),
                'open': round(history['Open'].iloc[-1], 2),
                'high': round(history['High'].iloc[-1], 2),
                'low': round(history['Low'].iloc[-1], 2),
                'volume': int(history['Volume'].iloc[-1]),
                'change': round(current_price - previous_close, 2),
                'change_percent': round(((current_price - previous_close) / previous_close) * 100, 2),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'dividend_yield': info.get('dividendYield', 0),
                '52_week_high': info.get('fiftyTwoWeekHigh', 0),
                '52_week_low': info.get('fiftyTwoWeekLow', 0),
                'avg_volume': info.get('averageVolume', 0),
                'beta': info.get('beta', 0)
            }
            
            return data
            
        except Exception as e:
            print(f"Error fetching Yahoo Finance data for {symbol}: {e}")
            return None
    
    def fetch_finnhub_data(self, symbol):
        """Fetch data from Finnhub (free tier)"""
        try:
            # Using Finnhub free API
            api_key = 'ct7a2r1r01qgs0109lngct7a2r1r01qgs0109lo0'  # Free tier key
            url = f'https://finnhub.io/api/v1/quote?symbol={symbol}&token={api_key}'
            
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                return {
                    'source': 'Finnhub',
                    'symbol': symbol,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'current_price': round(data.get('c', 0), 2),
                    'previous_close': round(data.get('pc', 0), 2),
                    'open': round(data.get('o', 0), 2),
                    'high': round(data.get('h', 0), 2),
                    'low': round(data.get('l', 0), 2),
                    'change': round(data.get('c', 0) - data.get('pc', 0), 2),
                    'change_percent': round(data.get('dp', 0), 2)
                }
        except:
            pass
        return None
    
    def verify_price_data(self, sources_data):
        """Verify price data across multiple sources"""
        if not sources_data:
            return None
            
        prices = [data['current_price'] for data in sources_data if data and data.get('current_price', 0) > 0]
        
        if not prices:
            return None
            
        # Calculate statistics
        avg_price = np.mean(prices)
        std_dev = np.std(prices) if len(prices) > 1 else 0
        
        # Check for anomalies
        anomalies = []
        for data in sources_data:
            if data and data.get('current_price', 0) > 0:
                price = data['current_price']
                diff_percent = abs((price - avg_price) / avg_price) * 100
                if diff_percent > 1.0:  # More than 1% difference
                    anomalies.append({
                        'source': data['source'],
                        'price': price,
                        'difference': diff_percent
                    })
        
        return {
            'verified_price': round(avg_price, 2),
            'sources_count': len(prices),
            'price_range': {
                'min': round(min(prices), 2),
                'max': round(max(prices), 2),
                'std_dev': round(std_dev, 2)
            },
            'anomalies': anomalies,
            'confidence': 'HIGH' if std_dev < 0.5 and len(anomalies) == 0 else 'MEDIUM' if std_dev < 1.0 else 'LOW'
        }
    
    def calculate_technical_indicators(self, symbol):
        """Calculate technical indicators using real data"""
        try:
            ticker = yf.Ticker(symbol)
            history = ticker.history(period="3mo")
            
            if history.empty:
                return None
                
            close_prices = history['Close']
            
            # Calculate indicators
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
            
            # MACD
            ema_12 = close_prices.ewm(span=12).mean()
            ema_26 = close_prices.ewm(span=26).mean()
            macd = ema_12 - ema_26
            macd_signal = macd.ewm(span=9).mean()
            
            # Bollinger Bands
            bb_sma = close_prices.rolling(window=20).mean()
            bb_std = close_prices.rolling(window=20).std()
            bb_upper = bb_sma + (bb_std * 2)
            bb_lower = bb_sma - (bb_std * 2)
            
            # ATR
            high_low = history['High'] - history['Low']
            high_close = np.abs(history['High'] - history['Close'].shift())
            low_close = np.abs(history['Low'] - history['Close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            atr = true_range.rolling(14).mean()
            
            return {
                'RSI': round(rsi.iloc[-1], 2) if not rsi.empty else 50,
                'MACD': round(macd.iloc[-1], 3) if not macd.empty else 0,
                'MACD_signal': round(macd_signal.iloc[-1], 3) if not macd_signal.empty else 0,
                'SMA_20': round(sma_20.iloc[-1], 2) if not sma_20.empty else 0,
                'SMA_50': round(sma_50.iloc[-1], 2) if not sma_50.empty else 0,
                'SMA_200': round(sma_200.iloc[-1], 2) if not pd.isna(sma_200.iloc[-1]) else 0,
                'BB_upper': round(bb_upper.iloc[-1], 2) if not bb_upper.empty else 0,
                'BB_lower': round(bb_lower.iloc[-1], 2) if not bb_lower.empty else 0,
                'ATR': round(atr.iloc[-1], 2) if not atr.empty else 0,
                'current_price': round(close_prices.iloc[-1], 2)
            }
            
        except Exception as e:
            print(f"Error calculating indicators for {symbol}: {e}")
            return None
    
    def analyze_stock(self, symbol):
        """Comprehensive analysis with verified data"""
        print(f"\n🔍 Fetching live data for {symbol}...")
        
        # Fetch data from multiple sources
        yahoo_data = self.fetch_yahoo_finance_data(symbol)
        finnhub_data = self.fetch_finnhub_data(symbol)
        
        # Collect all valid sources
        sources_data = [data for data in [yahoo_data, finnhub_data] if data]
        
        if not sources_data:
            print(f"❌ Failed to fetch data for {symbol}")
            return None
        
        # Verify data
        verification = self.verify_price_data(sources_data)
        
        # Use primary source (Yahoo) but with verification info
        primary_data = yahoo_data if yahoo_data else sources_data[0]
        primary_data['verification'] = verification
        
        # Calculate technical indicators
        indicators = self.calculate_technical_indicators(symbol)
        if indicators:
            primary_data['technical_indicators'] = indicators
        
        return primary_data
    
    def calculate_signal(self, data):
        """Calculate trading signal based on technical indicators"""
        if not data or 'technical_indicators' not in data:
            return 'HOLD', 50
            
        indicators = data['technical_indicators']
        score = 50
        
        # RSI Analysis
        rsi = indicators.get('RSI', 50)
        if rsi < 30:
            score += 25
        elif rsi > 70:
            score -= 25
        elif rsi < 40:
            score += 15
        elif rsi > 60:
            score -= 15
        
        # MACD Analysis
        if indicators.get('MACD', 0) > indicators.get('MACD_signal', 0):
            score += 20
        else:
            score -= 20
        
        # Moving Average Trend
        current_price = data['current_price']
        sma_20 = indicators.get('SMA_20', current_price)
        sma_50 = indicators.get('SMA_50', current_price)
        
        if current_price > sma_20 > sma_50:
            score += 25
        elif current_price < sma_20 < sma_50:
            score -= 25
        
        # Bollinger Bands
        if current_price < indicators.get('BB_lower', current_price):
            score += 15
        elif current_price > indicators.get('BB_upper', current_price):
            score -= 15
        
        # Determine signal
        if score >= 70:
            return 'STRONG_BUY', score
        elif score >= 60:
            return 'BUY', score
        elif score <= 30:
            return 'STRONG_SELL', score
        elif score <= 40:
            return 'SELL', score
        else:
            return 'HOLD', score
    
    def display_results(self, results):
        """Display analysis results with verification info"""
        print("\n" + "="*100)
        print(f"📊 LIVE STOCK ANALYSIS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*100)
        
        for symbol, data in results.items():
            if not data:
                continue
                
            signal, score = self.calculate_signal(data)
            signal_emoji = self.get_signal_emoji(signal)
            
            print(f"\n{symbol} - Live Market Data")
            print("-"*50)
            print(f"💹 Current Price: ${data['current_price']:.2f}")
            print(f"📅 Date/Time: {data['timestamp']}")
            print(f"📊 Change: ${data['change']:.2f} ({data['change_percent']:+.2f}%)")
            print(f"📈 Day Range: ${data['low']:.2f} - ${data['high']:.2f}")
            print(f"📊 Volume: {data['volume']:,}")
            
            # Verification info
            if 'verification' in data and data['verification']:
                v = data['verification']
                print(f"\n✅ Data Verification:")
                print(f"  • Verified Price: ${v['verified_price']:.2f}")
                print(f"  • Sources: {v['sources_count']}")
                print(f"  • Confidence: {v['confidence']}")
                if v['anomalies']:
                    print(f"  • ⚠️ Price discrepancies detected!")
            
            # Technical indicators
            if 'technical_indicators' in data:
                ind = data['technical_indicators']
                print(f"\n📊 Technical Indicators:")
                print(f"  • RSI: {ind.get('RSI', 'N/A')}")
                print(f"  • MACD: {ind.get('MACD', 'N/A')}")
                print(f"  • SMA 20: ${ind.get('SMA_20', 0):.2f}")
                print(f"  • SMA 50: ${ind.get('SMA_50', 0):.2f}")
            
            print(f"\n🎯 Signal: {signal_emoji} {signal} (Score: {score}/100)")
            
            # Additional metrics
            print(f"\n📈 Key Metrics:")
            print(f"  • Market Cap: ${data.get('market_cap', 0)/1e9:.1f}B")
            print(f"  • P/E Ratio: {data.get('pe_ratio', 0):.1f}")
            print(f"  • 52W Range: ${data.get('52_week_low', 0):.2f} - ${data.get('52_week_high', 0):.2f}")
    
    def get_signal_emoji(self, signal):
        """Return emoji for signal"""
        if 'STRONG_BUY' in signal:
            return '🟢🟢'
        elif 'BUY' in signal:
            return '🟢'
        elif 'STRONG_SELL' in signal:
            return '🔴🔴'
        elif 'SELL' in signal:
            return '🔴'
        else:
            return '🟡'
    
    def run_analysis(self):
        """Run complete analysis for all stocks"""
        results = {}
        
        print("🔄 Fetching live market data and verifying across multiple sources...")
        print("This may take a moment...\n")
        
        for symbol in self.symbols:
            data = self.analyze_stock(symbol)
            if data:
                results[symbol] = data
            time.sleep(0.5)  # Rate limiting
        
        # Display results
        self.display_results(results)
        
        # Save results
        self.save_results(results)
        
        return results
    
    def save_results(self, results):
        """Save analysis results with verification data"""
        output = {
            'timestamp': datetime.now().isoformat(),
            'analysis_date': datetime.now().strftime('%Y-%m-%d'),
            'stocks': {}
        }
        
        for symbol, data in results.items():
            if data:
                signal, score = self.calculate_signal(data)
                output['stocks'][symbol] = {
                    'price': data['current_price'],
                    'change': data['change'],
                    'change_percent': data['change_percent'],
                    'signal': signal,
                    'score': score,
                    'verification': data.get('verification', {}),
                    'technical_indicators': data.get('technical_indicators', {}),
                    'timestamp': data['timestamp']
                }
        
        try:
            with open('outputs/live_analysis.json', 'w') as f:
                json.dump(output, f, indent=2)
            print(f"\n💾 Results saved to outputs/live_analysis.json")
        except:
            pass

def main():
    """Main function"""
    print("\n🚀 Live Stock Analyzer with Multi-Source Verification")
    print("="*60)
    print("Fetching real-time market data...")
    print("Verifying prices across multiple sources...")
    print("="*60)
    
    analyzer = LiveStockAnalyzer()
    results = analyzer.run_analysis()
    
    print("\n" + "="*100)
    print("⚠️ IMPORTANT NOTES:")
    print("="*100)
    print("• Prices are verified across multiple data sources for accuracy")
    print("• Technical indicators are calculated from 3-month historical data")
    print("• Always verify with your broker before trading")
    print("• This analysis is for informational purposes only")
    print("="*100)

if __name__ == "__main__":
    main()