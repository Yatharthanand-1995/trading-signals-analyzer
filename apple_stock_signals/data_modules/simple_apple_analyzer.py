#!/usr/bin/env python3
"""
Simple Apple Stock Analyzer - Works without complex dependencies
"""

import json
import sys
from datetime import datetime

# Mock data for demonstration - Updated with current prices
def get_mock_stock_data():
    """Return mock Apple stock data"""
    return {
        'symbol': 'AAPL',
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
    }

def get_mock_technical_indicators():
    """Return mock technical indicators"""
    return {
        'RSI': 58.5,
        'MACD': 0.82,
        'MACD_signal': 0.65,
        'SMA_20': 207.80,
        'SMA_50': 205.50,
        'SMA_200': 195.20,
        'EMA_12': 209.10,
        'EMA_26': 206.80,
        'BB_upper': 213.50,
        'BB_middle': 207.80,
        'BB_lower': 202.10,
        'ATR': 3.25,
        'volume_ratio': 1.08
    }

def get_mock_sentiment_data():
    """Return mock sentiment data"""
    return {
        'news_sentiment': 65.5,
        'social_sentiment': 58.2,
        'combined_sentiment': 62.3,
        'news_articles': 15,
        'positive_articles': 9,
        'negative_articles': 3,
        'neutral_articles': 3
    }

def calculate_technical_score(indicators, price_data):
    """Calculate technical analysis score"""
    score = 50  # Start neutral
    reasons = []
    
    # RSI Analysis
    rsi = indicators['RSI']
    if rsi < 30:
        score += 25
        reasons.append(f"RSI oversold at {rsi:.1f}")
    elif rsi > 70:
        score -= 25
        reasons.append(f"RSI overbought at {rsi:.1f}")
    elif rsi < 40:
        score += 15
        reasons.append(f"RSI approaching oversold at {rsi:.1f}")
    elif rsi > 60:
        score -= 15
        reasons.append(f"RSI approaching overbought at {rsi:.1f}")
    
    # MACD Analysis
    if indicators['MACD'] > indicators['MACD_signal']:
        score += 20
        reasons.append("MACD bullish crossover")
    else:
        score -= 20
        reasons.append("MACD bearish crossover")
    
    # Moving Average Trend
    current_price = price_data['current_price']
    sma_20 = indicators['SMA_20']
    sma_50 = indicators['SMA_50']
    sma_200 = indicators['SMA_200']
    
    if sma_20 > sma_50 > sma_200:
        score += 30
        reasons.append("Strong uptrend: SMA20 > SMA50 > SMA200")
    elif sma_20 > sma_50:
        score += 20
        reasons.append("Short-term uptrend: SMA20 > SMA50")
    elif sma_20 < sma_50 < sma_200:
        score -= 30
        reasons.append("Strong downtrend: SMA20 < SMA50 < SMA200")
    elif sma_20 < sma_50:
        score -= 20
        reasons.append("Short-term downtrend: SMA20 < SMA50")
    
    # Bollinger Bands
    if current_price < indicators['BB_lower']:
        score += 15
        reasons.append("Price below lower Bollinger Band - oversold")
    elif current_price > indicators['BB_upper']:
        score -= 15
        reasons.append("Price above upper Bollinger Band - overbought")
    
    # Volume confirmation
    if indicators['volume_ratio'] > 1.5:
        score += 10
        reasons.append(f"High volume confirmation: {indicators['volume_ratio']:.2f}x")
    
    return max(0, min(100, score)), reasons

def generate_signal(technical_score, sentiment_score):
    """Generate trading signal based on scores"""
    # Weights
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

def calculate_price_targets(current_price, atr, signal):
    """Calculate stop loss and take profit levels"""
    if signal in ['BUY', 'STRONG_BUY']:
        stop_loss = current_price - (atr * 2)
        take_profit_1 = current_price + (atr * 2)
        take_profit_2 = current_price + (atr * 3)
    elif signal in ['SELL', 'STRONG_SELL']:
        stop_loss = current_price + (atr * 2)
        take_profit_1 = current_price - (atr * 2)
        take_profit_2 = current_price - (atr * 3)
    else:
        return None
    
    return {
        'stop_loss': round(stop_loss, 2),
        'take_profit_1': round(take_profit_1, 2),
        'take_profit_2': round(take_profit_2, 2),
        'risk_reward_ratio': 1.0
    }

def display_analysis():
    """Run and display the analysis"""
    print("\n🍎 APPLE STOCK ANALYSIS (Simplified Version)")
    print("=" * 60)
    print("📊 Using mock data for demonstration")
    print("=" * 60)
    
    # Get mock data
    stock_data = get_mock_stock_data()
    indicators = get_mock_technical_indicators()
    sentiment = get_mock_sentiment_data()
    
    # Display current data
    print(f"\n📈 CURRENT STOCK DATA:")
    print(f"Symbol: {stock_data['symbol']}")
    print(f"Date: {stock_data['date']}")
    print(f"Current Price: ${stock_data['current_price']:.2f}")
    print(f"Previous Close: ${stock_data['previous_close']:.2f}")
    print(f"Change: ${stock_data['change']:.2f} ({stock_data['change_percent']:.2f}%)")
    print(f"Day Range: ${stock_data['low']:.2f} - ${stock_data['high']:.2f}")
    print(f"Volume: {stock_data['volume']:,} (Ratio: {indicators['volume_ratio']:.2f}x)")
    print(f"P/E Ratio: {stock_data['pe_ratio']:.1f}")
    
    # Display technical indicators
    print(f"\n📊 TECHNICAL INDICATORS:")
    print(f"RSI (14): {indicators['RSI']:.1f}")
    print(f"MACD: {indicators['MACD']:.3f} / Signal: {indicators['MACD_signal']:.3f}")
    print(f"SMA 20: ${indicators['SMA_20']:.2f}")
    print(f"SMA 50: ${indicators['SMA_50']:.2f}")
    print(f"SMA 200: ${indicators['SMA_200']:.2f}")
    
    # Calculate scores
    technical_score, tech_reasons = calculate_technical_score(indicators, stock_data)
    sentiment_score = sentiment['combined_sentiment']
    
    print(f"\n📈 TECHNICAL ANALYSIS:")
    for reason in tech_reasons:
        print(f"  • {reason}")
    
    print(f"\n📰 SENTIMENT ANALYSIS:")
    print(f"News Sentiment: {sentiment['news_sentiment']:.1f}/100")
    print(f"Social Sentiment: {sentiment['social_sentiment']:.1f}/100")
    print(f"Articles Analyzed: {sentiment['news_articles']}")
    print(f"  • Positive: {sentiment['positive_articles']}")
    print(f"  • Negative: {sentiment['negative_articles']}")
    print(f"  • Neutral: {sentiment['neutral_articles']}")
    
    # Generate signal
    signal, final_score = generate_signal(technical_score, sentiment_score)
    
    # Calculate targets
    targets = calculate_price_targets(
        stock_data['current_price'], 
        indicators['ATR'],
        signal
    )
    
    # Display results
    print(f"\n{'='*60}")
    print("🎯 FINAL TRADING RECOMMENDATION")
    print("="*60)
    
    signal_icon = "🟢" if "BUY" in signal else "🔴" if "SELL" in signal else "🟡"
    print(f"\n{signal_icon} SIGNAL: {signal}")
    print(f"📊 Technical Score: {technical_score:.1f}/100")
    print(f"📰 Sentiment Score: {sentiment_score:.1f}/100")
    print(f"📈 Final Score: {final_score:.1f}/100")
    
    if targets:
        print(f"\n💰 PRICE TARGETS:")
        print(f"Entry Price: ${stock_data['current_price']:.2f}")
        print(f"Stop Loss: ${targets['stop_loss']:.2f}")
        print(f"Take Profit 1: ${targets['take_profit_1']:.2f}")
        print(f"Take Profit 2: ${targets['take_profit_2']:.2f}")
    
    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'date': stock_data['date'],
        'symbol': 'AAPL',
        'signal': signal,
        'final_score': final_score,
        'technical_score': technical_score,
        'sentiment_score': sentiment_score,
        'current_price': stock_data['current_price'],
        'targets': targets
    }
    
    try:
        with open('outputs/simple_analysis.json', 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to outputs/simple_analysis.json")
    except:
        pass
    
    print(f"\n⏰ Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Risk warning
    print("\n" + "="*60)
    print("⚠️ RISK WARNING")
    print("="*60)
    print("This is a simplified demo using mock data.")
    print("For real trading, use the full version with live data.")
    print("Always do your own research before trading.")
    print("="*60)

if __name__ == "__main__":
    display_analysis()