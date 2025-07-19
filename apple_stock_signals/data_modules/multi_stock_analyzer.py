#!/usr/bin/env python3
"""
Multi-Stock Analyzer - Analyzes AAPL, GOOGL, TSLA, ADBE, UNH
"""

import json
from datetime import datetime

# Mock data for multiple stocks
def get_stock_data(symbol):
    """Return mock stock data for different symbols"""
    stock_data = {
        'AAPL': {
            'name': 'Apple Inc.',
            'current_price': 210.25,
            'previous_close': 208.75,
            'volume': 55000000,
            'volume_avg': 51000000,
            'high': 211.50,
            'low': 208.90,
            'open': 209.15,
            'change': 1.50,
            'change_percent': 0.72,
            'market_cap': 3.25e12,
            'pe_ratio': 32.8,
            'dividend_yield': 0.0042,
            'date': datetime.now().strftime('%Y-%m-%d')
        },
        'GOOGL': {
            'name': 'Alphabet Inc.',
            'current_price': 178.45,
            'previous_close': 176.20,
            'volume': 28000000,
            'volume_avg': 25000000,
            'high': 179.80,
            'low': 176.90,
            'open': 177.00,
            'change': 2.25,
            'change_percent': 1.28,
            'market_cap': 2.28e12,
            'pe_ratio': 28.5,
            'dividend_yield': 0.0,
            'date': datetime.now().strftime('%Y-%m-%d')
        },
        'TSLA': {
            'name': 'Tesla Inc.',
            'current_price': 265.30,
            'previous_close': 262.50,
            'volume': 125000000,
            'volume_avg': 110000000,
            'high': 268.90,
            'low': 261.20,
            'open': 263.00,
            'change': 2.80,
            'change_percent': 1.07,
            'market_cap': 842e9,
            'pe_ratio': 75.2,
            'dividend_yield': 0.0,
            'date': datetime.now().strftime('%Y-%m-%d')
        },
        'ADBE': {
            'name': 'Adobe Inc.',
            'current_price': 592.75,
            'previous_close': 588.50,
            'volume': 3200000,
            'volume_avg': 2900000,
            'high': 596.20,
            'low': 587.30,
            'open': 589.00,
            'change': 4.25,
            'change_percent': 0.72,
            'market_cap': 268e9,
            'pe_ratio': 45.3,
            'dividend_yield': 0.0,
            'date': datetime.now().strftime('%Y-%m-%d')
        },
        'UNH': {
            'name': 'UnitedHealth Group',
            'current_price': 485.60,
            'previous_close': 482.20,
            'volume': 3500000,
            'volume_avg': 3200000,
            'high': 487.90,
            'low': 481.50,
            'open': 483.00,
            'change': 3.40,
            'change_percent': 0.71,
            'market_cap': 445e9,
            'pe_ratio': 21.8,
            'dividend_yield': 0.0145,
            'date': datetime.now().strftime('%Y-%m-%d')
        }
    }
    
    return stock_data.get(symbol, stock_data['AAPL'])

def get_technical_indicators(symbol):
    """Return mock technical indicators for different symbols"""
    indicators = {
        'AAPL': {
            'RSI': 58.5,
            'MACD': 0.82,
            'MACD_signal': 0.65,
            'SMA_20': 207.80,
            'SMA_50': 205.50,
            'SMA_200': 195.20,
            'BB_upper': 213.50,
            'BB_middle': 207.80,
            'BB_lower': 202.10,
            'ATR': 3.25,
            'volume_ratio': 1.08
        },
        'GOOGL': {
            'RSI': 62.3,
            'MACD': 1.15,
            'MACD_signal': 0.98,
            'SMA_20': 175.60,
            'SMA_50': 172.30,
            'SMA_200': 165.80,
            'BB_upper': 181.20,
            'BB_middle': 175.60,
            'BB_lower': 170.00,
            'ATR': 2.85,
            'volume_ratio': 1.12
        },
        'TSLA': {
            'RSI': 55.8,
            'MACD': 2.45,
            'MACD_signal': 2.10,
            'SMA_20': 261.20,
            'SMA_50': 255.80,
            'SMA_200': 235.50,
            'BB_upper': 272.50,
            'BB_middle': 261.20,
            'BB_lower': 249.90,
            'ATR': 8.75,
            'volume_ratio': 1.14
        },
        'ADBE': {
            'RSI': 48.2,
            'MACD': -1.25,
            'MACD_signal': -0.85,
            'SMA_20': 595.20,
            'SMA_50': 598.50,
            'SMA_200': 575.30,
            'BB_upper': 605.80,
            'BB_middle': 595.20,
            'BB_lower': 584.60,
            'ATR': 9.25,
            'volume_ratio': 1.10
        },
        'UNH': {
            'RSI': 61.5,
            'MACD': 2.15,
            'MACD_signal': 1.85,
            'SMA_20': 481.50,
            'SMA_50': 475.20,
            'SMA_200': 458.60,
            'BB_upper': 492.30,
            'BB_middle': 481.50,
            'BB_lower': 470.70,
            'ATR': 6.45,
            'volume_ratio': 1.09
        }
    }
    
    return indicators.get(symbol, indicators['AAPL'])

def get_sentiment_data(symbol):
    """Return mock sentiment data for different symbols"""
    sentiment = {
        'AAPL': {
            'news_sentiment': 65.5,
            'social_sentiment': 58.2,
            'combined_sentiment': 62.3,
            'news_articles': 15,
            'bullish_percent': 60
        },
        'GOOGL': {
            'news_sentiment': 72.3,
            'social_sentiment': 68.5,
            'combined_sentiment': 70.8,
            'news_articles': 12,
            'bullish_percent': 70
        },
        'TSLA': {
            'news_sentiment': 58.5,
            'social_sentiment': 62.8,
            'combined_sentiment': 60.2,
            'news_articles': 25,
            'bullish_percent': 55
        },
        'ADBE': {
            'news_sentiment': 45.2,
            'social_sentiment': 48.5,
            'combined_sentiment': 46.5,
            'news_articles': 8,
            'bullish_percent': 40
        },
        'UNH': {
            'news_sentiment': 68.5,
            'social_sentiment': 55.2,
            'combined_sentiment': 63.2,
            'news_articles': 6,
            'bullish_percent': 65
        }
    }
    
    return sentiment.get(symbol, sentiment['AAPL'])

def calculate_technical_score(indicators, stock_data):
    """Calculate technical analysis score"""
    score = 50
    
    # RSI Analysis
    rsi = indicators['RSI']
    if rsi < 30:
        score += 25
    elif rsi > 70:
        score -= 25
    elif rsi < 40:
        score += 15
    elif rsi > 60:
        score -= 15
    
    # MACD Analysis
    if indicators['MACD'] > indicators['MACD_signal']:
        score += 20
    else:
        score -= 20
    
    # Moving Average Trend
    current_price = stock_data['current_price']
    sma_20 = indicators['SMA_20']
    sma_50 = indicators['SMA_50']
    sma_200 = indicators['SMA_200']
    
    if sma_20 > sma_50 > sma_200:
        score += 30
    elif sma_20 > sma_50:
        score += 20
    elif sma_20 < sma_50 < sma_200:
        score -= 30
    elif sma_20 < sma_50:
        score -= 20
    
    # Bollinger Bands
    if current_price < indicators['BB_lower']:
        score += 15
    elif current_price > indicators['BB_upper']:
        score -= 15
    
    # Volume confirmation
    if indicators['volume_ratio'] > 1.5:
        score += 10
    
    return max(0, min(100, score))

def generate_signal(technical_score, sentiment_score):
    """Generate trading signal based on scores"""
    tech_weight = 0.6
    sentiment_weight = 0.4
    
    final_score = (technical_score * tech_weight) + (sentiment_score * sentiment_weight)
    
    if final_score >= 70:
        return 'STRONG_BUY', final_score
    elif final_score >= 60:
        return 'BUY', final_score
    elif final_score >= 40:
        return 'HOLD', final_score
    elif final_score >= 30:
        return 'SELL', final_score
    else:
        return 'STRONG_SELL', final_score

def analyze_stock(symbol):
    """Analyze a single stock"""
    stock_data = get_stock_data(symbol)
    indicators = get_technical_indicators(symbol)
    sentiment = get_sentiment_data(symbol)
    
    technical_score = calculate_technical_score(indicators, stock_data)
    sentiment_score = sentiment['combined_sentiment']
    
    signal, final_score = generate_signal(technical_score, sentiment_score)
    
    return {
        'symbol': symbol,
        'name': stock_data['name'],
        'price': stock_data['current_price'],
        'change': stock_data['change'],
        'change_percent': stock_data['change_percent'],
        'signal': signal,
        'final_score': final_score,
        'technical_score': technical_score,
        'sentiment_score': sentiment_score,
        'rsi': indicators['RSI'],
        'pe_ratio': stock_data['pe_ratio'],
        'volume_ratio': indicators['volume_ratio']
    }

def get_signal_emoji(signal):
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

def main():
    """Run multi-stock analysis"""
    print("\n📊 MULTI-STOCK ANALYZER")
    print("=" * 80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Analyzing: AAPL, GOOGL, TSLA, ADBE, UNH")
    print("=" * 80)
    
    # Analyze all stocks
    symbols = ['AAPL', 'GOOGL', 'TSLA', 'ADBE', 'UNH']
    results = []
    
    for symbol in symbols:
        result = analyze_stock(symbol)
        results.append(result)
    
    # Sort by final score (best opportunities first)
    results.sort(key=lambda x: x['final_score'], reverse=True)
    
    # Display individual analyses
    print("\n📈 INDIVIDUAL STOCK ANALYSIS:")
    print("-" * 80)
    
    for result in results:
        signal_emoji = get_signal_emoji(result['signal'])
        print(f"\n{result['symbol']} - {result['name']}")
        print(f"Price: ${result['price']:.2f} ({result['change']:+.2f}, {result['change_percent']:+.2f}%)")
        print(f"Signal: {signal_emoji} {result['signal']} (Score: {result['final_score']:.1f}/100)")
        print(f"Technical: {result['technical_score']:.0f}/100 | Sentiment: {result['sentiment_score']:.1f}/100")
        print(f"RSI: {result['rsi']:.1f} | P/E: {result['pe_ratio']:.1f} | Volume: {result['volume_ratio']:.2f}x")
    
    # Create comparison table
    print("\n📊 COMPARISON TABLE:")
    print("-" * 80)
    
    # Print header
    print(f"{'Symbol':<8} {'Price':<10} {'Change%':<10} {'RSI':<8} {'Signal':<12} {'Score':<8} {'Action':<8}")
    print("-" * 80)
    
    # Print data rows
    for r in results:
        print(f"{r['symbol']:<8} ${r['price']:<9.2f} {r['change_percent']:>+7.2f}%   {r['rsi']:<8.1f} {r['signal']:<12} {r['final_score']:<8.1f} {get_signal_emoji(r['signal']):<8}")
    
    # Portfolio recommendations
    print("\n💼 PORTFOLIO RECOMMENDATIONS:")
    print("-" * 80)
    
    strong_buys = [r for r in results if r['signal'] == 'STRONG_BUY']
    buys = [r for r in results if r['signal'] == 'BUY']
    holds = [r for r in results if r['signal'] == 'HOLD']
    sells = [r for r in results if r['signal'] in ['SELL', 'STRONG_SELL']]
    
    if strong_buys:
        print("\n🟢🟢 STRONG BUY (High Conviction):")
        for r in strong_buys:
            print(f"  • {r['symbol']} - Score: {r['final_score']:.1f}/100")
    
    if buys:
        print("\n🟢 BUY (Moderate Conviction):")
        for r in buys:
            print(f"  • {r['symbol']} - Score: {r['final_score']:.1f}/100")
    
    if holds:
        print("\n🟡 HOLD (Neutral):")
        for r in holds:
            print(f"  • {r['symbol']} - Score: {r['final_score']:.1f}/100")
    
    if sells:
        print("\n🔴 AVOID/SELL:")
        for r in sells:
            print(f"  • {r['symbol']} - Score: {r['final_score']:.1f}/100")
    
    # Best opportunity
    best = results[0]
    print(f"\n🎯 BEST OPPORTUNITY: {best['symbol']} ({best['signal']}, Score: {best['final_score']:.1f}/100)")
    
    # Risk diversification
    print("\n📊 SUGGESTED PORTFOLIO ALLOCATION:")
    print("-" * 80)
    
    total_score = sum(r['final_score'] for r in results if r['signal'] in ['BUY', 'STRONG_BUY'])
    
    if total_score > 0:
        for r in results:
            if r['signal'] in ['BUY', 'STRONG_BUY']:
                allocation = (r['final_score'] / total_score) * 100
                print(f"{r['symbol']}: {allocation:.1f}%")
    else:
        print("No buy signals - consider staying in cash")
    
    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'date': datetime.now().strftime('%Y-%m-%d'),
        'stocks': results,
        'best_opportunity': best['symbol'],
        'strong_buys': [r['symbol'] for r in strong_buys],
        'buys': [r['symbol'] for r in buys],
        'holds': [r['symbol'] for r in holds],
        'sells': [r['symbol'] for r in sells]
    }
    
    try:
        with open('outputs/multi_stock_analysis.json', 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\n💾 Results saved to outputs/multi_stock_analysis.json")
    except:
        pass
    
    # Risk warning
    print("\n" + "="*80)
    print("⚠️ RISK WARNING")
    print("="*80)
    print("This analysis uses mock data for demonstration purposes.")
    print("Always conduct your own research before making investment decisions.")
    print("Past performance does not guarantee future results.")
    print("="*80)

if __name__ == "__main__":
    main()