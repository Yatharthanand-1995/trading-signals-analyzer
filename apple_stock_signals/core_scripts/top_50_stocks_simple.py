#!/usr/bin/env python3
"""
Top 50 US Stocks Simple Analyzer
Analyzes the top 50 US stocks with built-in technical indicators
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Top 50 US stocks by market cap
TOP_50_STOCKS = [
    # Top 10
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 'JPM', 'JNJ',
    # 11-20
    'V', 'WMT', 'PG', 'MA', 'UNH', 'HD', 'DIS', 'BAC', 'XOM', 'ADBE',
    # 21-30
    'NFLX', 'CRM', 'CVX', 'ABBV', 'KO', 'PEP', 'TMO', 'COST', 'CSCO', 'MRK',
    # 31-40
    'ACN', 'LLY', 'AVGO', 'DHR', 'VZ', 'CMCSA', 'ABT', 'WFC', 'INTC', 'TXN',
    # 41-50
    'PFE', 'PM', 'NEE', 'RTX', 'HON', 'UNP', 'QCOM', 'BMY', 'UPS', 'AMGN'
]

class SimpleStockAnalyzer:
    def __init__(self):
        self.results = []
        
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI"""
        deltas = prices.diff()
        gain = deltas.where(deltas > 0, 0).rolling(window=period).mean()
        loss = -deltas.where(deltas < 0, 0).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if len(rsi) > 0 else 50
        
    def calculate_macd(self, prices):
        """Calculate MACD"""
        exp1 = prices.ewm(span=12, adjust=False).mean()
        exp2 = prices.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        histogram = macd - signal
        return {
            'macd': macd.iloc[-1] if len(macd) > 0 else 0,
            'signal': signal.iloc[-1] if len(signal) > 0 else 0,
            'histogram': histogram.iloc[-1] if len(histogram) > 0 else 0
        }
        
    def calculate_bollinger_bands(self, prices, period=20):
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = sma + (std * 2)
        lower = sma - (std * 2)
        return {
            'upper': upper.iloc[-1] if len(upper) > 0 else prices.iloc[-1],
            'middle': sma.iloc[-1] if len(sma) > 0 else prices.iloc[-1],
            'lower': lower.iloc[-1] if len(lower) > 0 else prices.iloc[-1]
        }
    
    def calculate_technical_score(self, data):
        """Calculate technical score based on indicators"""
        score = 50  # Base score
        current_price = data['current_price']
        
        # RSI scoring (0-30: oversold/buy, 70-100: overbought/sell)
        rsi = data['rsi']
        if 30 < rsi < 70:
            if rsi > 50:
                score += (rsi - 50) * 0.3  # Mild bullish
            else:
                score -= (50 - rsi) * 0.3  # Mild bearish
        elif rsi <= 30:
            score += 15  # Oversold - strong buy signal
        elif rsi >= 70:
            score -= 15  # Overbought - strong sell signal
            
        # MACD scoring
        if data['macd'] > data['macd_signal']:
            score += 15  # Bullish crossover
            if data['macd_histogram'] > 0:
                score += 5  # Strengthening bullish momentum
        else:
            score -= 15  # Bearish crossover
            if data['macd_histogram'] < 0:
                score -= 5  # Strengthening bearish momentum
                
        # Moving average scoring
        if current_price > data['sma_20'] and current_price > data['sma_50']:
            score += 10  # Above both MAs - bullish
        elif current_price < data['sma_20'] and current_price < data['sma_50']:
            score -= 10  # Below both MAs - bearish
            
        # Bollinger Bands scoring
        if current_price <= data['bb_lower']:
            score += 10  # At lower band - potential bounce
        elif current_price >= data['bb_upper']:
            score -= 10  # At upper band - potential reversal
            
        # Price trend scoring (5-day momentum)
        if data['price_change_5d_pct'] > 3:
            score += 5  # Strong upward momentum
        elif data['price_change_5d_pct'] < -3:
            score -= 5  # Strong downward momentum
            
        return max(0, min(100, score))
    
    def analyze_stock(self, symbol):
        """Analyze a single stock"""
        try:
            print(f"  Analyzing {symbol}...", end='', flush=True)
            
            # Fetch stock data
            stock = yf.Ticker(symbol)
            
            # Get historical data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=100)
            df = stock.history(start=start_date, end=end_date)
            
            if df.empty or len(df) < 50:
                print(f" ❌ Insufficient data")
                return None
                
            # Get stock info
            info = stock.info
            
            # Current price and basic data
            current_price = df['Close'].iloc[-1]
            
            # Calculate technical indicators
            rsi = self.calculate_rsi(df['Close'])
            macd_data = self.calculate_macd(df['Close'])
            bb_data = self.calculate_bollinger_bands(df['Close'])
            
            # Moving averages
            sma_20 = df['Close'].rolling(window=20).mean().iloc[-1]
            sma_50 = df['Close'].rolling(window=50).mean().iloc[-1]
            
            # Price changes
            price_change_1d = current_price - df['Close'].iloc[-2]
            price_change_1d_pct = (price_change_1d / df['Close'].iloc[-2]) * 100
            price_change_5d = current_price - df['Close'].iloc[-6]
            price_change_5d_pct = (price_change_5d / df['Close'].iloc[-6]) * 100
            
            # Volume analysis
            avg_volume = df['Volume'].rolling(window=20).mean().iloc[-1]
            volume_ratio = df['Volume'].iloc[-1] / avg_volume
            
            # ATR for stop loss/take profit
            high_low = df['High'] - df['Low']
            high_close = abs(df['High'] - df['Close'].shift())
            low_close = abs(df['Low'] - df['Close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            atr = true_range.rolling(14).mean().iloc[-1]
            
            # Compile data
            data = {
                'symbol': symbol,
                'company_name': info.get('longName', symbol),
                'sector': info.get('sector', 'Unknown'),
                'current_price': current_price,
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'rsi': rsi,
                'macd': macd_data['macd'],
                'macd_signal': macd_data['signal'],
                'macd_histogram': macd_data['histogram'],
                'sma_20': sma_20,
                'sma_50': sma_50,
                'bb_upper': bb_data['upper'],
                'bb_middle': bb_data['middle'],
                'bb_lower': bb_data['lower'],
                'price_change_1d': price_change_1d,
                'price_change_1d_pct': price_change_1d_pct,
                'price_change_5d_pct': price_change_5d_pct,
                'volume': df['Volume'].iloc[-1],
                'avg_volume': avg_volume,
                'volume_ratio': volume_ratio,
                'atr': atr
            }
            
            # Calculate technical score
            tech_score = self.calculate_technical_score(data)
            data['technical_score'] = tech_score
            
            # Determine signal
            if tech_score >= 70:
                signal = 'STRONG_BUY'
                emoji = '🟢🟢'
            elif tech_score >= 60:
                signal = 'BUY'
                emoji = '🟢'
            elif tech_score <= 30:
                signal = 'STRONG_SELL'
                emoji = '🔴🔴'
            elif tech_score <= 40:
                signal = 'SELL'
                emoji = '🔴'
            else:
                signal = 'HOLD'
                emoji = '🟡'
                
            data['signal'] = signal
            
            # Calculate price targets
            data['stop_loss'] = current_price - (2 * atr)
            data['take_profit_1'] = current_price + (2 * atr)
            data['take_profit_2'] = current_price + (4 * atr)
            data['risk_reward_ratio'] = 2.0
            
            print(f" ✅ {emoji} {signal} (Score: {tech_score:.1f})")
            return data
            
        except Exception as e:
            print(f" ❌ Error: {str(e)}")
            return None
    
    def analyze_all_stocks(self, max_workers=10):
        """Analyze all stocks in parallel"""
        print(f"\n🚀 ANALYZING TOP 50 US STOCKS")
        print(f"="*80)
        print(f"Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Using {max_workers} parallel workers")
        print(f"="*80)
        
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_stock = {executor.submit(self.analyze_stock, symbol): symbol 
                              for symbol in TOP_50_STOCKS}
            
            for future in as_completed(future_to_stock):
                result = future.result()
                if result:
                    results.append(result)
        
        # Sort by technical score
        results.sort(key=lambda x: x['technical_score'], reverse=True)
        self.results = results
        
        return results
    
    def display_results(self):
        """Display analysis results"""
        if not self.results:
            print("No results to display")
            return
            
        print(f"\n{'='*120}")
        print(f"📊 TOP 50 US STOCKS ANALYSIS RESULTS")
        print(f"{'='*120}")
        
        # Top picks
        print(f"\n🎯 TOP BULLISH PICKS (Score >= 60):")
        print(f"{'Symbol':<8} {'Company':<25} {'Price':<10} {'Change%':<8} {'Score':<8} {'RSI':<8} {'Signal':<12} {'Target':<10}")
        print(f"{'-'*103}")
        
        for stock in self.results:
            if stock['technical_score'] >= 60:
                print(f"{stock['symbol']:<8} {stock['company_name'][:24]:<25} "
                      f"${stock['current_price']:<9.2f} {stock['price_change_1d_pct']:<7.2f}% "
                      f"{stock['technical_score']:<8.1f} {stock['rsi']:<8.2f} "
                      f"{stock['signal']:<12} ${stock['take_profit_1']:<9.2f}")
        
        # Market overview
        signals_count = {'STRONG_BUY': 0, 'BUY': 0, 'HOLD': 0, 'SELL': 0, 'STRONG_SELL': 0}
        sector_data = {}
        
        for stock in self.results:
            signals_count[stock['signal']] += 1
            
            sector = stock['sector']
            if sector not in sector_data:
                sector_data[sector] = {'count': 0, 'total_score': 0}
            sector_data[sector]['count'] += 1
            sector_data[sector]['total_score'] += stock['technical_score']
        
        # Signal distribution
        print(f"\n📈 MARKET SIGNAL DISTRIBUTION:")
        total = len(self.results)
        print(f"{'Signal':<15} {'Count':<8} {'Percentage':<12}")
        print(f"{'-'*35}")
        
        for signal, count in signals_count.items():
            pct = (count / total * 100) if total > 0 else 0
            print(f"{signal:<15} {count:<8} {pct:<11.1f}%")
        
        # Market sentiment
        bullish = signals_count['STRONG_BUY'] + signals_count['BUY']
        bearish = signals_count['STRONG_SELL'] + signals_count['SELL']
        neutral = signals_count['HOLD']
        
        print(f"\n🎯 MARKET SENTIMENT:")
        print(f"Bullish: {bullish} stocks ({bullish/total*100:.1f}%)")
        print(f"Bearish: {bearish} stocks ({bearish/total*100:.1f}%)")
        print(f"Neutral: {neutral} stocks ({neutral/total*100:.1f}%)")
        
        # Sector performance
        print(f"\n📊 SECTOR ANALYSIS:")
        print(f"{'Sector':<25} {'Stocks':<8} {'Avg Score':<12}")
        print(f"{'-'*45}")
        
        for sector, data in sorted(sector_data.items()):
            avg_score = data['total_score'] / data['count']
            print(f"{sector:<25} {data['count']:<8} {avg_score:<11.1f}")
        
        # Save results
        self.save_results()
    
    def save_results(self):
        """Save analysis results"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        os.makedirs('outputs/top50_analysis', exist_ok=True)
        
        # Save JSON
        json_file = f'outputs/top50_analysis/analysis_{timestamp}.json'
        with open(json_file, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'total_stocks': len(self.results),
                'results': self.results
            }, f, indent=2, default=str)
        
        # Save CSV
        csv_file = f'outputs/top50_analysis/analysis_{timestamp}.csv'
        df = pd.DataFrame(self.results)
        
        # Select key columns for CSV
        key_columns = ['symbol', 'company_name', 'sector', 'current_price', 
                      'price_change_1d_pct', 'technical_score', 'signal', 
                      'rsi', 'macd', 'stop_loss', 'take_profit_1']
        
        df[key_columns].to_csv(csv_file, index=False)
        
        print(f"\n💾 Results saved to:")
        print(f"  • {json_file}")
        print(f"  • {csv_file}")

def main():
    analyzer = SimpleStockAnalyzer()
    analyzer.analyze_all_stocks(max_workers=10)
    analyzer.display_results()
    
    print(f"\n{'='*120}")
    print("⚠️ DISCLAIMER: This analysis is for educational purposes only. Not financial advice.")
    print("Always do your own research and consult with financial professionals.")
    print(f"{'='*120}")

if __name__ == "__main__":
    main()