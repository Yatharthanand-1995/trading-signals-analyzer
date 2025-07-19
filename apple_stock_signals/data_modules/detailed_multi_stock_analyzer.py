#!/usr/bin/env python3
"""
Detailed Multi-Stock Analyzer - Shows detailed analysis for each stock
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
            'EMA_12': 209.10,
            'EMA_26': 206.80,
            'BB_upper': 213.50,
            'BB_middle': 207.80,
            'BB_lower': 202.10,
            'ATR': 3.25,
            'volume_ratio': 1.08,
            'STOCH_K': 65.2,
            'STOCH_D': 62.8,
            'ADX': 28.5,
            'CCI': 45.2,
            'MFI': 58.7
        },
        'GOOGL': {
            'RSI': 62.3,
            'MACD': 1.15,
            'MACD_signal': 0.98,
            'SMA_20': 175.60,
            'SMA_50': 172.30,
            'SMA_200': 165.80,
            'EMA_12': 177.80,
            'EMA_26': 174.50,
            'BB_upper': 181.20,
            'BB_middle': 175.60,
            'BB_lower': 170.00,
            'ATR': 2.85,
            'volume_ratio': 1.12,
            'STOCH_K': 72.5,
            'STOCH_D': 70.1,
            'ADX': 32.1,
            'CCI': 85.3,
            'MFI': 65.4
        },
        'TSLA': {
            'RSI': 55.8,
            'MACD': 2.45,
            'MACD_signal': 2.10,
            'SMA_20': 261.20,
            'SMA_50': 255.80,
            'SMA_200': 235.50,
            'EMA_12': 263.50,
            'EMA_26': 260.10,
            'BB_upper': 272.50,
            'BB_middle': 261.20,
            'BB_lower': 249.90,
            'ATR': 8.75,
            'volume_ratio': 1.14,
            'STOCH_K': 58.3,
            'STOCH_D': 55.7,
            'ADX': 26.8,
            'CCI': 28.5,
            'MFI': 52.3
        },
        'ADBE': {
            'RSI': 48.2,
            'MACD': -1.25,
            'MACD_signal': -0.85,
            'SMA_20': 595.20,
            'SMA_50': 598.50,
            'SMA_200': 575.30,
            'EMA_12': 590.80,
            'EMA_26': 593.20,
            'BB_upper': 605.80,
            'BB_middle': 595.20,
            'BB_lower': 584.60,
            'ATR': 9.25,
            'volume_ratio': 1.10,
            'STOCH_K': 35.8,
            'STOCH_D': 38.2,
            'ADX': 22.5,
            'CCI': -75.2,
            'MFI': 42.1
        },
        'UNH': {
            'RSI': 61.5,
            'MACD': 2.15,
            'MACD_signal': 1.85,
            'SMA_20': 481.50,
            'SMA_50': 475.20,
            'SMA_200': 458.60,
            'EMA_12': 483.80,
            'EMA_26': 480.20,
            'BB_upper': 492.30,
            'BB_middle': 481.50,
            'BB_lower': 470.70,
            'ATR': 6.45,
            'volume_ratio': 1.09,
            'STOCH_K': 68.5,
            'STOCH_D': 66.2,
            'ADX': 30.2,
            'CCI': 62.8,
            'MFI': 60.5
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
            'positive_articles': 9,
            'negative_articles': 3,
            'neutral_articles': 3,
            'bullish_percent': 60,
            'recent_headlines': [
                "Apple's AI Strategy Shows Promise for Future Growth",
                "iPhone 16 Demand Exceeds Expectations in Key Markets",
                "Apple Services Revenue Hits New Record High"
            ]
        },
        'GOOGL': {
            'news_sentiment': 72.3,
            'social_sentiment': 68.5,
            'combined_sentiment': 70.8,
            'news_articles': 12,
            'positive_articles': 8,
            'negative_articles': 2,
            'neutral_articles': 2,
            'bullish_percent': 70,
            'recent_headlines': [
                "Google's Gemini AI Gains Market Share",
                "Cloud Revenue Growth Accelerates for Alphabet",
                "YouTube Ad Revenue Beats Analyst Estimates"
            ]
        },
        'TSLA': {
            'news_sentiment': 58.5,
            'social_sentiment': 62.8,
            'combined_sentiment': 60.2,
            'news_articles': 25,
            'positive_articles': 13,
            'negative_articles': 7,
            'neutral_articles': 5,
            'bullish_percent': 55,
            'recent_headlines': [
                "Tesla Cybertruck Production Ramps Up",
                "FSD Beta Shows Significant Improvements",
                "Competition Intensifies in EV Market"
            ]
        },
        'ADBE': {
            'news_sentiment': 45.2,
            'social_sentiment': 48.5,
            'combined_sentiment': 46.5,
            'news_articles': 8,
            'positive_articles': 2,
            'negative_articles': 4,
            'neutral_articles': 2,
            'bullish_percent': 40,
            'recent_headlines': [
                "Adobe Faces Increased Competition from AI Startups",
                "Subscription Growth Slows Amid Market Saturation",
                "Creative Cloud Pricing Under Pressure"
            ]
        },
        'UNH': {
            'news_sentiment': 68.5,
            'social_sentiment': 55.2,
            'combined_sentiment': 63.2,
            'news_articles': 6,
            'positive_articles': 4,
            'negative_articles': 1,
            'neutral_articles': 1,
            'bullish_percent': 65,
            'recent_headlines': [
                "UnitedHealth Reports Strong Q4 Earnings",
                "Medicare Advantage Enrollment Grows",
                "Healthcare Sector Shows Resilience"
            ]
        }
    }
    
    return sentiment.get(symbol, sentiment['AAPL'])

def get_fundamental_data(symbol):
    """Return mock fundamental data for different symbols"""
    fundamentals = {
        'AAPL': {
            'revenue_growth': 0.123,
            'earnings_growth': 0.145,
            'profit_margin': 0.268,
            'roe': 0.325,
            'debt_to_equity': 0.52,
            'forward_pe': 30.2,
            'peg_ratio': 2.1,
            'price_to_book': 45.2,
            'free_cash_flow': 95.8e9,
            'beta': 1.15
        },
        'GOOGL': {
            'revenue_growth': 0.152,
            'earnings_growth': 0.178,
            'profit_margin': 0.245,
            'roe': 0.285,
            'debt_to_equity': 0.12,
            'forward_pe': 26.8,
            'peg_ratio': 1.5,
            'price_to_book': 6.8,
            'free_cash_flow': 78.5e9,
            'beta': 1.08
        },
        'TSLA': {
            'revenue_growth': 0.285,
            'earnings_growth': 0.325,
            'profit_margin': 0.195,
            'roe': 0.225,
            'debt_to_equity': 0.35,
            'forward_pe': 68.5,
            'peg_ratio': 2.8,
            'price_to_book': 12.5,
            'free_cash_flow': 12.5e9,
            'beta': 1.85
        },
        'ADBE': {
            'revenue_growth': 0.085,
            'earnings_growth': 0.062,
            'profit_margin': 0.285,
            'roe': 0.385,
            'debt_to_equity': 0.28,
            'forward_pe': 42.5,
            'peg_ratio': 4.5,
            'price_to_book': 15.8,
            'free_cash_flow': 6.8e9,
            'beta': 1.25
        },
        'UNH': {
            'revenue_growth': 0.125,
            'earnings_growth': 0.138,
            'profit_margin': 0.062,
            'roe': 0.245,
            'debt_to_equity': 0.65,
            'forward_pe': 20.5,
            'peg_ratio': 1.4,
            'price_to_book': 5.2,
            'free_cash_flow': 22.5e9,
            'beta': 0.75
        }
    }
    
    return fundamentals.get(symbol, fundamentals['AAPL'])

def calculate_technical_score(indicators, stock_data, symbol):
    """Calculate technical analysis score with detailed reasoning"""
    score = 50
    reasons = []
    
    # RSI Analysis
    rsi = indicators['RSI']
    if rsi < 30:
        score += 25
        reasons.append(f"RSI oversold at {rsi:.1f} (+25 points)")
    elif rsi > 70:
        score -= 25
        reasons.append(f"RSI overbought at {rsi:.1f} (-25 points)")
    elif rsi < 40:
        score += 15
        reasons.append(f"RSI approaching oversold at {rsi:.1f} (+15 points)")
    elif rsi > 60:
        score -= 15
        reasons.append(f"RSI approaching overbought at {rsi:.1f} (-15 points)")
    else:
        reasons.append(f"RSI neutral at {rsi:.1f} (0 points)")
    
    # MACD Analysis
    macd = indicators['MACD']
    macd_signal = indicators['MACD_signal']
    if macd > macd_signal:
        score += 20
        reasons.append(f"MACD bullish crossover (+20 points)")
    else:
        score -= 20
        reasons.append(f"MACD bearish crossover (-20 points)")
    
    # Moving Average Trend
    current_price = stock_data['current_price']
    sma_20 = indicators['SMA_20']
    sma_50 = indicators['SMA_50']
    sma_200 = indicators['SMA_200']
    
    if sma_20 > sma_50 > sma_200:
        score += 30
        reasons.append("Strong uptrend: SMA20 > SMA50 > SMA200 (+30 points)")
    elif sma_20 > sma_50:
        score += 20
        reasons.append("Short-term uptrend: SMA20 > SMA50 (+20 points)")
    elif sma_20 < sma_50 < sma_200:
        score -= 30
        reasons.append("Strong downtrend: SMA20 < SMA50 < SMA200 (-30 points)")
    elif sma_20 < sma_50:
        score -= 20
        reasons.append("Short-term downtrend: SMA20 < SMA50 (-20 points)")
    else:
        reasons.append("Sideways trend (0 points)")
    
    # Bollinger Bands
    if current_price < indicators['BB_lower']:
        score += 15
        reasons.append("Price below lower Bollinger Band - oversold (+15 points)")
    elif current_price > indicators['BB_upper']:
        score -= 15
        reasons.append("Price above upper Bollinger Band - overbought (-15 points)")
    elif current_price < indicators['BB_middle']:
        score -= 5
        reasons.append("Price below middle Bollinger Band (-5 points)")
    else:
        score += 5
        reasons.append("Price above middle Bollinger Band (+5 points)")
    
    # Volume confirmation
    if indicators['volume_ratio'] > 1.5:
        if score > 50:
            score += 10
            reasons.append(f"High volume confirmation: {indicators['volume_ratio']:.2f}x (+10 points)")
        else:
            score -= 5
            reasons.append(f"High volume on bearish signal: {indicators['volume_ratio']:.2f}x (-5 points)")
    elif indicators['volume_ratio'] < 0.5:
        score -= 5
        reasons.append(f"Low volume: {indicators['volume_ratio']:.2f}x (-5 points)")
    else:
        reasons.append(f"Normal volume: {indicators['volume_ratio']:.2f}x (0 points)")
    
    # ADX Trend Strength
    adx = indicators['ADX']
    if adx > 25:
        if score > 50:
            score += 10
            reasons.append(f"Strong trend confirmed by ADX: {adx:.1f} (+10 points)")
        else:
            score -= 10
            reasons.append(f"Strong bearish trend by ADX: {adx:.1f} (-10 points)")
    else:
        reasons.append(f"Weak trend by ADX: {adx:.1f} (0 points)")
    
    return max(0, min(100, score)), reasons

def calculate_sentiment_score(sentiment_data):
    """Calculate sentiment score with details"""
    news_score = sentiment_data['news_sentiment']
    social_score = sentiment_data['social_sentiment']
    combined_score = (news_score * 0.7) + (social_score * 0.3)
    
    return combined_score

def calculate_fundamental_score(fundamental_data, stock_data):
    """Calculate fundamental analysis score with reasoning"""
    score = 50
    reasons = []
    
    # P/E Ratio Analysis
    pe_ratio = stock_data['pe_ratio']
    if pe_ratio > 0:
        if pe_ratio < 20:
            score += 15
            reasons.append(f"P/E ratio undervalued at {pe_ratio:.1f} (+15 points)")
        elif pe_ratio > 40:
            score -= 15
            reasons.append(f"P/E ratio overvalued at {pe_ratio:.1f} (-15 points)")
        elif pe_ratio < 25:
            score += 5
            reasons.append(f"P/E ratio slightly undervalued at {pe_ratio:.1f} (+5 points)")
        elif pe_ratio > 35:
            score -= 5
            reasons.append(f"P/E ratio slightly overvalued at {pe_ratio:.1f} (-5 points)")
        else:
            reasons.append(f"P/E ratio fair at {pe_ratio:.1f} (0 points)")
    
    # Revenue Growth
    revenue_growth = fundamental_data['revenue_growth']
    if revenue_growth > 0.15:
        score += 20
        reasons.append(f"Strong revenue growth: {revenue_growth*100:.1f}% (+20 points)")
    elif revenue_growth > 0.05:
        score += 10
        reasons.append(f"Moderate revenue growth: {revenue_growth*100:.1f}% (+10 points)")
    elif revenue_growth < 0:
        score -= 20
        reasons.append(f"Revenue decline: {revenue_growth*100:.1f}% (-20 points)")
    else:
        reasons.append(f"Low revenue growth: {revenue_growth*100:.1f}% (0 points)")
    
    # Profit Margin
    profit_margin = fundamental_data['profit_margin']
    if profit_margin > 0.25:
        score += 10
        reasons.append(f"High profit margin: {profit_margin*100:.1f}% (+10 points)")
    elif profit_margin < 0.10:
        score -= 10
        reasons.append(f"Low profit margin: {profit_margin*100:.1f}% (-10 points)")
    else:
        reasons.append(f"Decent profit margin: {profit_margin*100:.1f}% (0 points)")
    
    # PEG Ratio
    peg_ratio = fundamental_data['peg_ratio']
    if 0 < peg_ratio < 1:
        score += 10
        reasons.append(f"Excellent PEG ratio: {peg_ratio:.2f} (+10 points)")
    elif 1 <= peg_ratio < 1.5:
        score += 5
        reasons.append(f"Good PEG ratio: {peg_ratio:.2f} (+5 points)")
    elif peg_ratio > 3:
        score -= 10
        reasons.append(f"Poor PEG ratio: {peg_ratio:.2f} (-10 points)")
    
    # Beta Analysis
    beta = fundamental_data['beta']
    if 0.8 <= beta <= 1.2:
        score += 5
        reasons.append(f"Stable beta: {beta:.2f} (+5 points)")
    elif beta > 1.5:
        score -= 5
        reasons.append(f"High beta (more risky): {beta:.2f} (-5 points)")
    else:
        reasons.append(f"Beta: {beta:.2f} (0 points)")
    
    return max(0, min(100, score)), reasons

def generate_signal(technical_score, sentiment_score, fundamental_score):
    """Generate trading signal based on scores"""
    tech_weight = 0.6
    sentiment_weight = 0.3
    fundamental_weight = 0.1
    
    final_score = (
        technical_score * tech_weight +
        sentiment_score * sentiment_weight +
        fundamental_score * fundamental_weight
    )
    
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
    
    risk_reward = 1.0 if atr > 0 else 0
    
    return {
        'stop_loss': round(stop_loss, 2),
        'take_profit_1': round(take_profit_1, 2),
        'take_profit_2': round(take_profit_2, 2),
        'risk_reward_ratio': risk_reward
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

def display_detailed_analysis(symbol):
    """Display detailed analysis for a single stock"""
    # Get all data
    stock_data = get_stock_data(symbol)
    indicators = get_technical_indicators(symbol)
    sentiment = get_sentiment_data(symbol)
    fundamentals = get_fundamental_data(symbol)
    
    print(f"\n{'='*80}")
    print(f"📊 {symbol} - {stock_data['name']} DETAILED ANALYSIS")
    print(f"{'='*80}")
    
    # Current Stock Data
    print(f"\n📈 CURRENT STOCK DATA:")
    print(f"Date: {stock_data['date']}")
    print(f"Current Price: ${stock_data['current_price']:.2f}")
    print(f"Previous Close: ${stock_data['previous_close']:.2f}")
    print(f"Change: ${stock_data['change']:.2f} ({stock_data['change_percent']:+.2f}%)")
    print(f"Day Range: ${stock_data['low']:.2f} - ${stock_data['high']:.2f}")
    print(f"Volume: {stock_data['volume']:,} (Ratio: {indicators['volume_ratio']:.2f}x)")
    print(f"Market Cap: ${stock_data['market_cap']/1e9:.1f}B")
    print(f"P/E Ratio: {stock_data['pe_ratio']:.1f}")
    if stock_data['dividend_yield'] > 0:
        print(f"Dividend Yield: {stock_data['dividend_yield']*100:.2f}%")
    
    # Technical Indicators
    print(f"\n📊 TECHNICAL INDICATORS:")
    print(f"\n🔄 Moving Averages:")
    print(f"SMA 20: ${indicators['SMA_20']:.2f}")
    print(f"SMA 50: ${indicators['SMA_50']:.2f}")
    print(f"SMA 200: ${indicators['SMA_200']:.2f}")
    print(f"EMA 12: ${indicators['EMA_12']:.2f}")
    print(f"EMA 26: ${indicators['EMA_26']:.2f}")
    
    # Trend Analysis
    sma_20 = indicators['SMA_20']
    sma_50 = indicators['SMA_50']
    sma_200 = indicators['SMA_200']
    
    if sma_20 > sma_50 > sma_200:
        trend = "🟢 Strong Uptrend"
    elif sma_20 > sma_50:
        trend = "🟡 Short-term Uptrend"
    elif sma_20 < sma_50 < sma_200:
        trend = "🔴 Strong Downtrend"
    elif sma_20 < sma_50:
        trend = "🟡 Short-term Downtrend"
    else:
        trend = "🟡 Sideways/Mixed"
    
    print(f"Overall Trend: {trend}")
    
    # Momentum Indicators
    print(f"\n⚡ Momentum Indicators:")
    rsi = indicators['RSI']
    print(f"RSI (14): {rsi:.1f}", end='')
    if rsi < 30:
        print(" - 🟢 Oversold")
    elif rsi > 70:
        print(" - 🔴 Overbought")
    elif rsi < 40:
        print(" - 🟡 Approaching Oversold")
    elif rsi > 60:
        print(" - 🟡 Approaching Overbought")
    else:
        print(" - 🟡 Neutral")
    
    print(f"MACD: {indicators['MACD']:.3f} / Signal: {indicators['MACD_signal']:.3f}")
    if indicators['MACD'] > indicators['MACD_signal']:
        print("MACD Signal: 🟢 Bullish (MACD > Signal)")
    else:
        print("MACD Signal: 🔴 Bearish (MACD < Signal)")
    
    print(f"Stochastic %K: {indicators['STOCH_K']:.1f} / %D: {indicators['STOCH_D']:.1f}")
    
    # Bollinger Bands
    print(f"\n📊 Bollinger Bands:")
    print(f"Upper Band: ${indicators['BB_upper']:.2f}")
    print(f"Middle Band: ${indicators['BB_middle']:.2f}")
    print(f"Lower Band: ${indicators['BB_lower']:.2f}")
    
    current_price = stock_data['current_price']
    if current_price > indicators['BB_upper']:
        bb_position = "🔴 Above Upper Band (Overbought)"
    elif current_price < indicators['BB_lower']:
        bb_position = "🟢 Below Lower Band (Oversold)"
    elif current_price > indicators['BB_middle']:
        bb_position = "🟡 Above Middle (Bullish Bias)"
    else:
        bb_position = "🟡 Below Middle (Bearish Bias)"
    print(f"Price Position: {bb_position}")
    
    # Additional Indicators
    print(f"\n📊 Additional Indicators:")
    print(f"ADX (14): {indicators['ADX']:.1f} - ", end='')
    if indicators['ADX'] > 25:
        print("Strong Trend")
    else:
        print("Weak/No Trend")
    
    print(f"CCI (20): {indicators['CCI']:.1f} - ", end='')
    if indicators['CCI'] > 100:
        print("Overbought")
    elif indicators['CCI'] < -100:
        print("Oversold")
    else:
        print("Neutral")
    
    print(f"MFI (14): {indicators['MFI']:.1f} - ", end='')
    if indicators['MFI'] > 80:
        print("Overbought")
    elif indicators['MFI'] < 20:
        print("Oversold")
    else:
        print("Neutral")
    
    print(f"ATR (14): ${indicators['ATR']:.2f} ({(indicators['ATR']/current_price)*100:.2f}% of price)")
    
    # Calculate scores
    technical_score, tech_reasons = calculate_technical_score(indicators, stock_data, symbol)
    sentiment_score = calculate_sentiment_score(sentiment)
    fundamental_score, fund_reasons = calculate_fundamental_score(fundamentals, stock_data)
    
    # Technical Analysis Breakdown
    print(f"\n🔧 CALCULATING TECHNICAL SCORE...")
    print("📈 Technical Analysis Breakdown:")
    for reason in tech_reasons:
        print(f"  • {reason}")
    print(f"📊 Technical Score: {technical_score:.1f}/100")
    
    # Sentiment Analysis
    print(f"\n📰 SENTIMENT ANALYSIS:")
    print(f"News Sentiment: {sentiment['news_sentiment']:.1f}/100")
    print(f"Social Sentiment: {sentiment['social_sentiment']:.1f}/100")
    print(f"Combined Sentiment: {sentiment_score:.1f}/100")
    print(f"📊 News Analysis:")
    print(f"  • Total Articles: {sentiment['news_articles']}")
    print(f"  • Positive: {sentiment['positive_articles']}")
    print(f"  • Negative: {sentiment['negative_articles']}")
    print(f"  • Neutral: {sentiment['neutral_articles']}")
    print(f"\n📰 Recent Headlines:")
    for headline in sentiment['recent_headlines']:
        print(f"  • {headline}")
    
    # Fundamental Analysis
    print(f"\n📊 FUNDAMENTAL ANALYSIS:")
    print("📊 Fundamental Analysis Breakdown:")
    for reason in fund_reasons:
        print(f"  • {reason}")
    print(f"📊 Fundamental Score: {fundamental_score:.1f}/100")
    
    # Generate signal
    signal, final_score = generate_signal(technical_score, sentiment_score, fundamental_score)
    
    # Calculate targets
    targets = calculate_price_targets(current_price, indicators['ATR'], signal)
    
    # Final Trading Recommendation
    print(f"\n{'='*60}")
    print("🎯 FINAL TRADING RECOMMENDATION")
    print("="*60)
    
    signal_emoji = get_signal_emoji(signal)
    print(f"\n{signal_emoji} SIGNAL: {signal}")
    print(f"📊 COMPONENT SCORES:")
    print(f"Technical Score: {technical_score:.1f}/100 (Weight: 60%)")
    print(f"Sentiment Score: {sentiment_score:.1f}/100 (Weight: 30%)")
    print(f"Fundamental Score: {fundamental_score:.1f}/100 (Weight: 10%)")
    print(f"\nFINAL SCORE: {final_score:.1f}/100")
    
    if targets:
        print(f"\n💰 PRICE TARGETS:")
        print(f"Entry Price: ${current_price:.2f}")
        print(f"Stop Loss: ${targets['stop_loss']:.2f}")
        print(f"Take Profit 1: ${targets['take_profit_1']:.2f}")
        print(f"Take Profit 2: ${targets['take_profit_2']:.2f}")
        print(f"Risk/Reward Ratio: {targets['risk_reward_ratio']:.2f}:1")
    
    # Position sizing
    if signal in ['BUY', 'STRONG_BUY']:
        print(f"\n💡 POSITION SIZING RECOMMENDATION:")
        print(f"• Risk no more than 2% of portfolio on this trade")
        print(f"• With stop-loss at ${targets['stop_loss']:.2f}, risk per share is ${abs(current_price - targets['stop_loss']):.2f}")
        print(f"• For $10,000 portfolio, max risk = $200")
        print(f"• Suggested position size: {int(200 / abs(current_price - targets['stop_loss']))} shares")
    
    return {
        'symbol': symbol,
        'name': stock_data['name'],
        'signal': signal,
        'final_score': final_score,
        'technical_score': technical_score,
        'sentiment_score': sentiment_score,
        'fundamental_score': fundamental_score,
        'current_price': current_price,
        'targets': targets
    }

def display_summary(results):
    """Display summary of all stocks"""
    print(f"\n\n{'='*80}")
    print("📊 MULTI-STOCK ANALYSIS SUMMARY")
    print(f"{'='*80}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Sort by final score
    results.sort(key=lambda x: x['final_score'], reverse=True)
    
    # Summary table
    print(f"\n📈 COMPARISON TABLE:")
    print("-" * 80)
    print(f"{'Symbol':<8} {'Price':<10} {'Signal':<12} {'Score':<8} {'Tech':<8} {'Sent':<8} {'Fund':<8}")
    print("-" * 80)
    
    for r in results:
        print(f"{r['symbol']:<8} ${r['current_price']:<9.2f} {r['signal']:<12} {r['final_score']:<8.1f} "
              f"{r['technical_score']:<8.0f} {r['sentiment_score']:<8.1f} {r['fundamental_score']:<8.0f}")
    
    # Portfolio recommendations
    print(f"\n💼 PORTFOLIO RECOMMENDATIONS:")
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
    
    # Portfolio allocation
    print(f"\n📊 SUGGESTED PORTFOLIO ALLOCATION:")
    print("-" * 80)
    
    buyable = [r for r in results if r['signal'] in ['BUY', 'STRONG_BUY']]
    if buyable:
        total_score = sum(r['final_score'] for r in buyable)
        for r in buyable:
            allocation = (r['final_score'] / total_score) * 100
            print(f"{r['symbol']}: {allocation:.1f}%")
    else:
        print("No buy signals - consider staying in cash")

def main():
    """Run detailed multi-stock analysis"""
    print("\n📊 DETAILED MULTI-STOCK ANALYZER")
    print("=" * 80)
    print(f"Analyzing: AAPL, GOOGL, TSLA, ADBE, UNH")
    print("=" * 80)
    
    # Analyze all stocks
    symbols = ['AAPL', 'GOOGL', 'TSLA', 'ADBE', 'UNH']
    results = []
    
    for symbol in symbols:
        result = display_detailed_analysis(symbol)
        results.append(result)
    
    # Display summary
    display_summary(results)
    
    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'date': datetime.now().strftime('%Y-%m-%d'),
        'detailed_results': results,
        'best_opportunity': max(results, key=lambda x: x['final_score'])['symbol']
    }
    
    try:
        with open('outputs/detailed_multi_stock_analysis.json', 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\n💾 Results saved to outputs/detailed_multi_stock_analysis.json")
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