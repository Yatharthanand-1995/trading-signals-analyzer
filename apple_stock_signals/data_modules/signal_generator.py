import pandas as pd
import numpy as np
from datetime import datetime
from core_scripts.config import SIGNAL_WEIGHTS, SIGNAL_THRESHOLDS, ANALYSIS_SETTINGS

class AppleSignalGenerator:
    def __init__(self):
        # Use weights from config
        self.weights = SIGNAL_WEIGHTS
        
    def generate_signal(self, all_data):
        """Generate buy/sell signal for Apple stock"""
        print("\n🎯 GENERATING TRADING SIGNAL...")
        
        # Calculate component scores
        technical_score = self.calculate_technical_score(all_data['technical_indicators'])
        sentiment_score = self.calculate_sentiment_score(all_data['news_data'], all_data['social_data'])
        fundamental_score = self.calculate_fundamental_score(all_data['fundamental_data'])
        
        # Adjust scores based on data quality
        data_confidence = all_data['verification_results']['confidence_score']
        if data_confidence < ANALYSIS_SETTINGS['CONFIDENCE_THRESHOLD']:
            print(f"\n⚠️ WARNING: Low data confidence ({data_confidence:.1f}%). Adjusting scores...")
        
        # Display component scores
        print(f"\n📊 COMPONENT SCORES:")
        print(f"Technical Score: {technical_score:.1f}/100 (Weight: {self.weights['TECHNICAL']*100:.0f}%)")
        print(f"Sentiment Score: {sentiment_score:.1f}/100 (Weight: {self.weights['SENTIMENT']*100:.0f}%)")
        print(f"Fundamental Score: {fundamental_score:.1f}/100 (Weight: {self.weights['FUNDAMENTAL']*100:.0f}%)")
        
        # Calculate final score
        final_score = (
            technical_score * self.weights['TECHNICAL'] +
            sentiment_score * self.weights['SENTIMENT'] +
            fundamental_score * self.weights['FUNDAMENTAL']
        )
        
        print(f"\nFINAL SCORE: {final_score:.1f}/100")
        
        # Generate signal
        signal_result = self.score_to_signal(final_score)
        
        # Calculate price targets
        price_targets = self.calculate_price_targets(all_data, signal_result)
        
        # Create detailed analysis
        analysis_details = self.create_analysis_details(all_data, technical_score, sentiment_score, fundamental_score)
        
        return {
            'symbol': 'AAPL',
            'signal': signal_result['signal'],
            'confidence': signal_result['confidence'],
            'final_score': final_score,
            'component_scores': {
                'technical': technical_score,
                'sentiment': sentiment_score,
                'fundamental': fundamental_score
            },
            'price_targets': price_targets,
            'analysis_details': analysis_details,
            'timestamp': datetime.now().isoformat(),
            'data_quality': data_confidence
        }
    
    def calculate_technical_score(self, indicators):
        """Calculate technical analysis score (0-100)"""
        print("\n🔧 CALCULATING TECHNICAL SCORE...")
        
        score = 50  # Start neutral
        reasons = []
        
        # 1. RSI Analysis (+/-25 points)
        rsi = indicators['RSI'][-1]
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
        
        # 2. MACD Analysis (+/-20 points)
        macd = indicators['MACD'][-1]
        macd_signal = indicators['MACD_signal'][-1]
        macd_hist = indicators['MACD_hist'][-1]
        
        if macd > macd_signal:
            if macd_hist > indicators['MACD_hist'][-2]:  # Strengthening
                score += 20
                reasons.append(f"MACD bullish crossover strengthening (+20 points)")
            else:
                score += 15
                reasons.append(f"MACD bullish crossover (+15 points)")
        else:
            if macd_hist < indicators['MACD_hist'][-2]:  # Weakening
                score -= 20
                reasons.append(f"MACD bearish crossover weakening (-20 points)")
            else:
                score -= 15
                reasons.append(f"MACD bearish crossover (-15 points)")
        
        # 3. Moving Average Trend (+/-30 points)
        sma_20 = indicators['SMA_20'][-1]
        sma_50 = indicators['SMA_50'][-1]
        sma_200 = indicators['SMA_200'][-1]
        current_price = indicators['current_price']
        
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
        
        # 4. Bollinger Bands (+/-15 points)
        bb_upper = indicators['BB_upper'][-1]
        bb_lower = indicators['BB_lower'][-1]
        bb_middle = indicators['BB_middle'][-1]
        
        if current_price < bb_lower:
            score += 15
            reasons.append("Price below lower Bollinger Band - oversold (+15 points)")
        elif current_price > bb_upper:
            score -= 15
            reasons.append("Price above upper Bollinger Band - overbought (-15 points)")
        elif current_price < bb_middle:
            score -= 5
            reasons.append("Price below middle Bollinger Band (-5 points)")
        else:
            score += 5
            reasons.append("Price above middle Bollinger Band (+5 points)")
        
        # 5. Volume Confirmation (+/-10 points)
        volume_ratio = indicators['volume_ratio']
        if volume_ratio > 1.5:
            if score > 50:  # Volume confirms bullish signal
                score += 10
                reasons.append(f"High volume confirmation: {volume_ratio:.2f}x (+10 points)")
            else:  # High volume on bearish signal
                score -= 5
                reasons.append(f"High volume on bearish signal: {volume_ratio:.2f}x (-5 points)")
        elif volume_ratio < 0.5:
            score -= 5
            reasons.append(f"Low volume: {volume_ratio:.2f}x (-5 points)")
        else:
            reasons.append(f"Normal volume: {volume_ratio:.2f}x (0 points)")
        
        # 6. ADX Trend Strength (+/-10 points)
        adx = indicators['ADX'][-1]
        if adx > 25:
            if score > 50:
                score += 10
                reasons.append(f"Strong trend confirmed by ADX: {adx:.1f} (+10 points)")
            else:
                score -= 10
                reasons.append(f"Strong bearish trend by ADX: {adx:.1f} (-10 points)")
        else:
            reasons.append(f"Weak trend by ADX: {adx:.1f} (0 points)")
        
        # Display technical analysis reasoning
        print("📈 Technical Analysis Breakdown:")
        for reason in reasons:
            print(f"  • {reason}")
        
        final_score = max(0, min(100, score))
        print(f"📊 Technical Score: {final_score:.1f}/100")
        
        return final_score
    
    def calculate_sentiment_score(self, news_data, social_data):
        """Calculate sentiment analysis score (0-100)"""
        print("\n📰 CALCULATING SENTIMENT SCORE...")
        
        # News sentiment (70% weight)
        news_score = news_data['avg_sentiment']
        
        # Social sentiment (30% weight)
        social_score = social_data['combined_sentiment']
        
        # Combine with weights
        combined_score = (news_score * 0.7) + (social_score * 0.3)
        
        # Adjust based on volume of sentiment data
        if news_data['total_articles'] < 5:
            print("⚠️ Limited news data available")
            combined_score = combined_score * 0.8 + 50 * 0.2  # Move towards neutral
        
        print(f"📰 News Sentiment: {news_score:.1f}/100")
        print(f"💬 Social Sentiment: {social_score:.1f}/100")
        print(f"📊 Combined Sentiment: {combined_score:.1f}/100")
        
        # Show sentiment breakdown
        print(f"📊 News Analysis:")
        print(f"  • Total Articles: {news_data['total_articles']}")
        print(f"  • Positive: {news_data['positive_count']}")
        print(f"  • Negative: {news_data['negative_count']}")
        print(f"  • Neutral: {news_data['neutral_count']}")
        
        if social_data['reddit_data']['post_count'] > 0:
            print(f"💬 Reddit Analysis:")
            print(f"  • Total Posts: {social_data['reddit_data']['post_count']}")
            print(f"  • Bullish: {social_data['reddit_data']['bullish_posts']}")
            print(f"  • Bearish: {social_data['reddit_data']['bearish_posts']}")
        
        return combined_score
    
    def calculate_fundamental_score(self, fundamental_data):
        """Calculate fundamental analysis score (0-100)"""
        print("\n📊 CALCULATING FUNDAMENTAL SCORE...")
        
        if not fundamental_data:
            print("No fundamental data available")
            return 50
        
        score = 50  # Start neutral
        reasons = []
        
        # 1. P/E Ratio Analysis (+/-15 points)
        pe_ratio = fundamental_data.get('pe_ratio', 0)
        if pe_ratio > 0:
            # Apple's historical average P/E is around 25
            if pe_ratio < 20:
                score += 15
                reasons.append(f"P/E ratio undervalued at {pe_ratio:.1f} (+15 points)")
            elif pe_ratio > 35:
                score -= 15
                reasons.append(f"P/E ratio overvalued at {pe_ratio:.1f} (-15 points)")
            elif pe_ratio < 25:
                score += 5
                reasons.append(f"P/E ratio slightly undervalued at {pe_ratio:.1f} (+5 points)")
            elif pe_ratio > 30:
                score -= 5
                reasons.append(f"P/E ratio slightly overvalued at {pe_ratio:.1f} (-5 points)")
            else:
                reasons.append(f"P/E ratio fair at {pe_ratio:.1f} (0 points)")
        
        # 2. Revenue Growth (+/-20 points)
        revenue_growth = fundamental_data.get('revenue_growth', 0)
        if revenue_growth > 0.15:  # 15% growth
            score += 20
            reasons.append(f"Strong revenue growth: {revenue_growth*100:.1f}% (+20 points)")
        elif revenue_growth > 0.05:  # 5% growth
            score += 10
            reasons.append(f"Moderate revenue growth: {revenue_growth*100:.1f}% (+10 points)")
        elif revenue_growth < 0:
            score -= 20
            reasons.append(f"Revenue decline: {revenue_growth*100:.1f}% (-20 points)")
        else:
            reasons.append(f"Low revenue growth: {revenue_growth*100:.1f}% (0 points)")
        
        # 3. Profit Margin (+/-10 points)
        profit_margin = fundamental_data.get('profit_margin', 0)
        if profit_margin > 0.25:  # 25% margin
            score += 10
            reasons.append(f"High profit margin: {profit_margin*100:.1f}% (+10 points)")
        elif profit_margin < 0.15:  # 15% margin
            score -= 10
            reasons.append(f"Low profit margin: {profit_margin*100:.1f}% (-10 points)")
        else:
            reasons.append(f"Decent profit margin: {profit_margin*100:.1f}% (0 points)")
        
        # 4. Beta Analysis (+/-5 points)
        beta = fundamental_data.get('beta', 1.0)
        if 0.8 <= beta <= 1.2:
            score += 5
            reasons.append(f"Stable beta: {beta:.2f} (+5 points)")
        elif beta > 1.5:
            score -= 5
            reasons.append(f"High beta (more risky): {beta:.2f} (-5 points)")
        else:
            reasons.append(f"Beta: {beta:.2f} (0 points)")
        
        # 5. PEG Ratio (+/-10 points)
        peg_ratio = fundamental_data.get('peg_ratio', 0)
        if 0 < peg_ratio < 1:
            score += 10
            reasons.append(f"Excellent PEG ratio: {peg_ratio:.2f} (+10 points)")
        elif 1 <= peg_ratio < 1.5:
            score += 5
            reasons.append(f"Good PEG ratio: {peg_ratio:.2f} (+5 points)")
        elif peg_ratio > 2:
            score -= 10
            reasons.append(f"Poor PEG ratio: {peg_ratio:.2f} (-10 points)")
        
        # Display fundamental analysis reasoning
        print("📊 Fundamental Analysis Breakdown:")
        for reason in reasons:
            print(f"  • {reason}")
        
        final_score = max(0, min(100, score))
        print(f"📊 Fundamental Score: {final_score:.1f}/100")
        
        return final_score
    
    def score_to_signal(self, combined_score):
        """Convert combined score to trading signal"""
        if combined_score >= SIGNAL_THRESHOLDS['STRONG_BUY']:
            return {
                'signal': 'STRONG_BUY',
                'confidence': combined_score,
                'reasoning': 'Multiple strong bullish indicators aligned'
            }
        elif combined_score >= SIGNAL_THRESHOLDS['BUY']:
            return {
                'signal': 'BUY',
                'confidence': combined_score,
                'reasoning': 'Bullish indicators outweigh bearish factors'
            }
        elif combined_score >= SIGNAL_THRESHOLDS['HOLD_LOWER']:
            return {
                'signal': 'HOLD',
                'confidence': abs(combined_score - 50) * 2,
                'reasoning': 'Mixed signals or neutral market conditions'
            }
        elif combined_score >= SIGNAL_THRESHOLDS['SELL']:
            return {
                'signal': 'SELL',
                'confidence': 100 - combined_score,
                'reasoning': 'Bearish indicators outweigh bullish factors'
            }
        else:
            return {
                'signal': 'STRONG_SELL',
                'confidence': 100 - combined_score,
                'reasoning': 'Multiple strong bearish indicators aligned'
            }
    
    def calculate_price_targets(self, all_data, signal_result):
        """Calculate entry, stop-loss, and take-profit levels"""
        current_price = all_data['stock_data']['current_price']
        atr = all_data['technical_indicators']['ATR'][-1]
        
        # Use ATR-based stops and targets
        stop_multiplier = 2.0
        target1_multiplier = 2.0
        target2_multiplier = 3.0
        
        # Adjust based on volatility
        atr_percent = (atr / current_price) * 100
        if atr_percent > 3:  # High volatility
            stop_multiplier = 2.5
            target1_multiplier = 3.0
            target2_multiplier = 4.0
        elif atr_percent < 1:  # Low volatility
            stop_multiplier = 1.5
            target1_multiplier = 1.5
            target2_multiplier = 2.5
        
        stop_distance = atr * stop_multiplier
        profit_distance1 = atr * target1_multiplier
        profit_distance2 = atr * target2_multiplier
        
        if signal_result['signal'] in ['BUY', 'STRONG_BUY']:
            return {
                'entry_price': current_price,
                'stop_loss': round(current_price - stop_distance, 2),
                'take_profit_1': round(current_price + profit_distance1, 2),
                'take_profit_2': round(current_price + profit_distance2, 2),
                'risk_amount': round(stop_distance, 2),
                'reward_amount': round(profit_distance1, 2),
                'risk_reward_ratio': round(profit_distance1 / stop_distance, 2)
            }
        elif signal_result['signal'] in ['SELL', 'STRONG_SELL']:
            return {
                'entry_price': current_price,
                'stop_loss': round(current_price + stop_distance, 2),
                'take_profit_1': round(current_price - profit_distance1, 2),
                'take_profit_2': round(current_price - profit_distance2, 2),
                'risk_amount': round(stop_distance, 2),
                'reward_amount': round(profit_distance1, 2),
                'risk_reward_ratio': round(profit_distance1 / stop_distance, 2)
            }
        else:
            return {
                'entry_price': current_price,
                'stop_loss': None,
                'take_profit_1': None,
                'take_profit_2': None,
                'risk_amount': 0,
                'reward_amount': 0,
                'risk_reward_ratio': 0
            }
    
    def create_analysis_details(self, all_data, technical_score, sentiment_score, fundamental_score):
        """Create detailed analysis summary"""
        details = []
        
        # Technical details
        indicators = all_data['technical_indicators']
        rsi = indicators['RSI'][-1]
        
        if rsi < 30:
            details.append(f"RSI oversold at {rsi:.1f} - potential bounce")
        elif rsi > 70:
            details.append(f"RSI overbought at {rsi:.1f} - potential correction")
        
        # MACD details
        macd = indicators['MACD'][-1]
        macd_signal = indicators['MACD_signal'][-1]
        if macd > macd_signal:
            details.append("MACD bullish crossover")
        else:
            details.append("MACD bearish crossover")
        
        # Moving average trend
        sma_20 = indicators['SMA_20'][-1]
        sma_50 = indicators['SMA_50'][-1]
        if sma_20 > sma_50:
            details.append("Short-term uptrend (SMA20 > SMA50)")
        else:
            details.append("Short-term downtrend (SMA20 < SMA50)")
        
        # Sentiment details
        news_data = all_data['news_data']
        if news_data['avg_sentiment'] > 65:
            details.append("Positive news sentiment")
        elif news_data['avg_sentiment'] < 35:
            details.append("Negative news sentiment")
        
        # Volume details
        volume_ratio = indicators['volume_ratio']
        if volume_ratio > 1.5:
            details.append(f"High volume confirmation ({volume_ratio:.2f}x)")
        
        # Data quality
        if all_data['verification_results']['confidence_score'] < 75:
            details.append("⚠️ Data quality concerns - trade with caution")
        
        return details