#!/usr/bin/env python3
"""
Top 50 US Stocks Trading Analyzer
Analyzes the top 50 US stocks by market cap with technical indicators
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import sys
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_modules.technical_analyzer import AppleTechnicalAnalyzer

# Top 50 US stocks by market cap (as of 2025)
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

class Top50StocksAnalyzer:
    def __init__(self):
        self.technical_analyzer = AppleTechnicalAnalyzer()
        self.results = {}
        
    def calculate_technical_score(self, indicators):
        """Calculate technical score based on indicators"""
        score = 50  # Base score
        
        # RSI scoring
        rsi = indicators.get('RSI', 50)
        if 30 < rsi < 70:
            if rsi > 50:
                score += (rsi - 50) * 0.5  # Bullish momentum
            else:
                score -= (50 - rsi) * 0.5  # Bearish momentum
        elif rsi <= 30:
            score += 10  # Oversold - potential buy
        elif rsi >= 70:
            score -= 10  # Overbought - potential sell
            
        # MACD scoring
        macd = indicators.get('MACD', 0)
        macd_signal = indicators.get('MACD_Signal', 0)
        if macd > macd_signal:
            score += 15  # Bullish crossover
        else:
            score -= 15  # Bearish crossover
            
        # Moving average scoring
        current_price = indicators.get('Close', 0)
        sma_20 = indicators.get('SMA_20', current_price)
        sma_50 = indicators.get('SMA_50', current_price)
        
        if current_price > sma_20 and current_price > sma_50:
            score += 10
        elif current_price < sma_20 and current_price < sma_50:
            score -= 10
            
        # Bollinger Bands scoring
        bb_upper = indicators.get('BB_Upper', current_price)
        bb_lower = indicators.get('BB_Lower', current_price)
        
        if current_price < bb_lower:
            score += 5  # Potential bounce
        elif current_price > bb_upper:
            score -= 5  # Potential reversal
            
        return max(0, min(100, score))
    
    def analyze_single_stock(self, symbol):
        """Analyze a single stock"""
        try:
            print(f"  Analyzing {symbol}...", end='', flush=True)
            
            # Fetch data
            stock = yf.Ticker(symbol)
            
            # Get historical data (1 month for analysis)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=90)
            df = stock.history(start=start_date, end=end_date)
            
            if df.empty:
                print(f" ❌ No data")
                return None
                
            # Get current info
            info = stock.info
            current_price = df['Close'].iloc[-1]
            
            # Calculate technical indicators
            indicators = self.technical_analyzer.calculate_all_indicators(df)
            
            # Get latest values
            latest_indicators = {}
            for key, value in indicators.items():
                if isinstance(value, pd.Series) and len(value) > 0:
                    latest_indicators[key] = value.iloc[-1]
                else:
                    latest_indicators[key] = value
                    
            latest_indicators['Close'] = current_price
            
            # Calculate technical score
            tech_score = self.calculate_technical_score(latest_indicators)
            
            # Determine signal
            if tech_score >= 70:
                signal = 'STRONG_BUY'
            elif tech_score >= 60:
                signal = 'BUY'
            elif tech_score <= 30:
                signal = 'STRONG_SELL'
            elif tech_score <= 40:
                signal = 'SELL'
            else:
                signal = 'HOLD'
                
            # Calculate price targets
            atr = latest_indicators.get('ATR', current_price * 0.02)
            stop_loss = current_price - (2 * atr)
            take_profit_1 = current_price + (2 * atr)
            take_profit_2 = current_price + (4 * atr)
            
            result = {
                'symbol': symbol,
                'company_name': info.get('longName', symbol),
                'sector': info.get('sector', 'Unknown'),
                'current_price': round(current_price, 2),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'volume': int(df['Volume'].iloc[-1]),
                'avg_volume': int(df['Volume'].mean()),
                'day_change': round(df['Close'].iloc[-1] - df['Close'].iloc[-2], 2),
                'day_change_pct': round((df['Close'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2] * 100, 2),
                'technical_score': round(tech_score, 1),
                'signal': signal,
                'rsi': round(latest_indicators.get('RSI', 50), 2),
                'macd': round(latest_indicators.get('MACD', 0), 3),
                'macd_signal': round(latest_indicators.get('MACD_Signal', 0), 3),
                'sma_20': round(latest_indicators.get('SMA_20', 0), 2),
                'sma_50': round(latest_indicators.get('SMA_50', 0), 2),
                'stop_loss': round(stop_loss, 2),
                'take_profit_1': round(take_profit_1, 2),
                'take_profit_2': round(take_profit_2, 2),
                'analysis_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            print(f" ✅ {signal} (Score: {tech_score:.1f})")
            return result
            
        except Exception as e:
            print(f" ❌ Error: {str(e)}")
            return None
    
    def analyze_all_stocks(self, max_workers=10):
        """Analyze all top 50 stocks in parallel"""
        print(f"\n🚀 ANALYZING TOP 50 US STOCKS")
        print(f"="*80)
        print(f"Starting analysis at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Using {max_workers} parallel workers")
        print(f"="*80)
        
        results = []
        
        # Process stocks in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_stock = {executor.submit(self.analyze_single_stock, symbol): symbol 
                              for symbol in TOP_50_STOCKS}
            
            # Process completed tasks
            for future in as_completed(future_to_stock):
                symbol = future_to_stock[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    print(f"Error processing {symbol}: {e}")
        
        # Sort results by technical score
        results.sort(key=lambda x: x['technical_score'], reverse=True)
        
        return results
    
    def display_summary(self, results):
        """Display analysis summary"""
        print(f"\n{'='*120}")
        print(f"📊 TOP 50 US STOCKS ANALYSIS SUMMARY - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*120}")
        
        # Group by signal
        signals = {}
        for r in results:
            signal = r['signal']
            if signal not in signals:
                signals[signal] = []
            signals[signal].append(r)
        
        # Display top picks
        print(f"\n🎯 TOP PICKS (STRONG BUY/BUY):")
        print(f"{'Symbol':<8} {'Company':<30} {'Price':<10} {'Score':<8} {'RSI':<8} {'Signal':<12} {'Target 1':<10}")
        print(f"{'-'*96}")
        
        for signal in ['STRONG_BUY', 'BUY']:
            if signal in signals:
                for stock in signals[signal][:10]:  # Top 10 for each signal
                    print(f"{stock['symbol']:<8} {stock['company_name'][:29]:<30} "
                          f"${stock['current_price']:<9.2f} {stock['technical_score']:<8.1f} "
                          f"{stock['rsi']:<8.2f} {stock['signal']:<12} ${stock['take_profit_1']:<9.2f}")
        
        # Display sector analysis
        print(f"\n📊 SECTOR BREAKDOWN:")
        sector_counts = {}
        sector_scores = {}
        
        for r in results:
            sector = r['sector']
            if sector not in sector_counts:
                sector_counts[sector] = 0
                sector_scores[sector] = []
            sector_counts[sector] += 1
            sector_scores[sector].append(r['technical_score'])
        
        print(f"{'Sector':<25} {'Count':<8} {'Avg Score':<10}")
        print(f"{'-'*43}")
        
        for sector in sorted(sector_counts.keys()):
            avg_score = sum(sector_scores[sector]) / len(sector_scores[sector])
            print(f"{sector:<25} {sector_counts[sector]:<8} {avg_score:<10.1f}")
        
        # Display signal summary
        print(f"\n📈 SIGNAL SUMMARY:")
        print(f"{'Signal':<15} {'Count':<8} {'Percentage':<12}")
        print(f"{'-'*35}")
        
        total = len(results)
        for signal in ['STRONG_BUY', 'BUY', 'HOLD', 'SELL', 'STRONG_SELL']:
            count = len(signals.get(signal, []))
            pct = (count / total * 100) if total > 0 else 0
            print(f"{signal:<15} {count:<8} {pct:<12.1f}%")
        
        # Market sentiment
        bullish = len(signals.get('STRONG_BUY', [])) + len(signals.get('BUY', []))
        bearish = len(signals.get('STRONG_SELL', [])) + len(signals.get('SELL', []))
        
        print(f"\n🎯 MARKET SENTIMENT:")
        print(f"Bullish Stocks: {bullish} ({bullish/total*100:.1f}%)")
        print(f"Bearish Stocks: {bearish} ({bearish/total*100:.1f}%)")
        print(f"Neutral Stocks: {total - bullish - bearish} ({(total-bullish-bearish)/total*100:.1f}%)")
        
        # Save results
        self.save_results(results)
        
    def save_results(self, results):
        """Save analysis results"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create output directory
        os.makedirs('outputs/top50_analysis', exist_ok=True)
        
        # Save detailed JSON
        json_file = f'outputs/top50_analysis/top50_analysis_{timestamp}.json'
        with open(json_file, 'w') as f:
            json.dump({
                'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'total_stocks': len(results),
                'results': results
            }, f, indent=2)
        
        # Save CSV summary
        csv_file = f'outputs/top50_analysis/top50_summary_{timestamp}.csv'
        df = pd.DataFrame(results)
        df.to_csv(csv_file, index=False)
        
        print(f"\n💾 Results saved to:")
        print(f"  • {json_file}")
        print(f"  • {csv_file}")

def main():
    """Main function"""
    analyzer = Top50StocksAnalyzer()
    
    # Analyze all stocks
    results = analyzer.analyze_all_stocks(max_workers=10)
    
    # Display summary
    if results:
        analyzer.display_summary(results)
    else:
        print("❌ No results obtained")
    
    print(f"\n{'='*120}")
    print("⚠️ DISCLAIMER: This is for educational purposes only. Not financial advice.")
    print(f"{'='*120}")

if __name__ == "__main__":
    main()