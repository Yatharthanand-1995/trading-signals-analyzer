#!/usr/bin/env python3
"""
Swing Trading Analyzer - Enhanced Version
Integrates multi-timeframe, volume, and entry filters
Optimized for 2-15 day swing trading
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our modules
from .multi_timeframe import MultiTimeframeAnalyzer
from .volume_analyzer import VolumeAnalyzer
from .entry_filters import EntryFilterSystem

# Import existing modules
try:
    from ..indicators.technical_wrapper import TechnicalAnalyzer
except ImportError:
    # Fallback for testing or if technical wrapper is not available
    class TechnicalAnalyzer:
        def __init__(self):
            pass

class SwingTradingAnalyzer:
    """
    Complete swing trading analysis system with Phase 1 enhancements
    """
    
    def __init__(self):
        # Initialize all components
        self.mtf_analyzer = MultiTimeframeAnalyzer()
        self.volume_analyzer = VolumeAnalyzer()
        self.filter_system = EntryFilterSystem()
        self.tech_analyzer = TechnicalAnalyzer()
        
        # Configuration
        self.config = {
            'min_filter_score': 60,  # Minimum filter score to consider
            'min_alignment_score': 70,  # Minimum timeframe alignment
            'min_volume_score': 50,  # Minimum volume confirmation
            'risk_per_trade': 0.02,  # 2% risk per trade
            'position_sizing_atr_multiplier': 2  # ATR multiplier for stops
        }
        
    def analyze_stock_complete(self, symbol, market_symbol='SPY'):
        """Complete analysis pipeline for a stock"""
        print(f"\n{'='*60}")
        print(f"🎯 SWING TRADING ANALYSIS: {symbol}")
        print(f"{'='*60}")
        
        analysis_results = {
            'symbol': symbol,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'passes_filters': False,
            'signal': 'NO_TRADE',
            'score': 0
        }
        
        try:
            # Step 1: Fetch basic data
            print(f"\n📊 Step 1: Fetching data...")
            stock = yf.Ticker(symbol)
            df = stock.history(period='6mo')
            
            if df.empty or len(df) < 50:
                print(f"❌ Insufficient data for {symbol}")
                return analysis_results
            
            # Get market data for relative strength
            market = yf.Ticker(market_symbol)
            market_df = market.history(period='6mo')
            
            # Step 2: Apply entry filters
            print(f"\n🔍 Step 2: Applying entry filters...")
            filters = self.filter_system.apply_filters(symbol, df, market_df, stock.info)
            analysis_results['filters'] = filters
            
            if not filters['passes_filters']:
                print(f"❌ {symbol} fails entry filters (Score: {filters['overall']['filter_percentage']:.1f}%)")
                analysis_results['reason'] = 'Failed entry filters'
                return analysis_results
            
            print(f"✅ Passes entry filters (Score: {filters['overall']['filter_percentage']:.1f}%)")
            
            # Step 3: Multi-timeframe analysis
            print(f"\n📈 Step 3: Multi-timeframe analysis...")
            mtf_analysis = self.mtf_analyzer.analyze_stock(symbol)
            analysis_results['multi_timeframe'] = mtf_analysis
            
            if mtf_analysis and 'alignment' in mtf_analysis:
                alignment_score = mtf_analysis['alignment']['alignment_percentage']
                if alignment_score < self.config['min_alignment_score']:
                    print(f"⚠️ Poor timeframe alignment ({alignment_score:.1f}%)")
                else:
                    print(f"✅ Good timeframe alignment ({alignment_score:.1f}%)")
            
            # Step 4: Volume analysis
            print(f"\n📊 Step 4: Volume analysis...")
            volume_analysis = self.volume_analyzer.analyze_volume(df)
            analysis_results['volume'] = volume_analysis
            
            if volume_analysis:
                volume_score = volume_analysis['volume_score']
                if volume_score < self.config['min_volume_score']:
                    print(f"⚠️ Weak volume confirmation ({volume_score})")
                else:
                    print(f"✅ Strong volume confirmation ({volume_score})")
            
            # Step 5: Calculate final signal and score
            print(f"\n🎯 Step 5: Calculating final signal...")
            final_analysis = self._calculate_final_signal(
                filters, mtf_analysis, volume_analysis, df
            )
            
            analysis_results.update(final_analysis)
            analysis_results['passes_filters'] = True
            
            # Step 6: Calculate position sizing and targets
            if final_analysis['signal'] in ['BUY', 'STRONG_BUY']:
                position_data = self._calculate_position_sizing(df, stock.info)
                analysis_results['position_sizing'] = position_data
            
            return analysis_results
            
        except Exception as e:
            print(f"❌ Error analyzing {symbol}: {e}")
            analysis_results['error'] = str(e)
            return analysis_results
    
    def _calculate_final_signal(self, filters, mtf_analysis, volume_analysis, df):
        """Calculate final trading signal based on all analyses"""
        score = 0
        max_score = 100
        signal_reasons = []
        
        # Filter score (30% weight)
        filter_score = filters['overall']['filter_percentage']
        weighted_filter_score = (filter_score / 100) * 30
        score += weighted_filter_score
        
        if filter_score >= 80:
            signal_reasons.append("Excellent entry filters")
        elif filter_score >= 70:
            signal_reasons.append("Good entry filters")
        
        # Timeframe alignment (30% weight)
        if mtf_analysis and 'alignment' in mtf_analysis:
            alignment_score = mtf_analysis['alignment']['alignment_percentage']
            weighted_alignment_score = (alignment_score / 100) * 30
            score += weighted_alignment_score
            
            if alignment_score >= 80:
                signal_reasons.append("Strong timeframe alignment")
            elif alignment_score >= 70:
                signal_reasons.append("Good timeframe alignment")
        
        # Volume confirmation (20% weight)
        if volume_analysis:
            volume_score = volume_analysis['volume_score']
            weighted_volume_score = (volume_score / 100) * 20
            score += weighted_volume_score
            
            if volume_score >= 70:
                signal_reasons.append("Strong volume confirmation")
            elif volume_score >= 60:
                signal_reasons.append("Good volume confirmation")
        
        # Technical setup (20% weight)
        tech_score = self._calculate_technical_score(df)
        weighted_tech_score = (tech_score / 100) * 20
        score += weighted_tech_score
        
        if tech_score >= 70:
            signal_reasons.append("Bullish technical setup")
        
        # Determine final signal
        if score >= 80:
            signal = 'STRONG_BUY'
        elif score >= 70:
            signal = 'BUY'
        elif score >= 60:
            signal = 'HOLD_BULLISH'
        elif score <= 20:
            signal = 'STRONG_SELL'
        elif score <= 30:
            signal = 'SELL'
        elif score <= 40:
            signal = 'HOLD_BEARISH'
        else:
            signal = 'HOLD'
        
        # Entry timing check for BUY signals
        entry_quality = 'standard'
        if signal in ['BUY', 'STRONG_BUY'] and mtf_analysis and 'entry_timing' in mtf_analysis:
            entry_timing = mtf_analysis['entry_timing']['entry_quality']
            if entry_timing == 'excellent':
                signal_reasons.append("Excellent entry timing (pullback in uptrend)")
                entry_quality = 'excellent'
            elif entry_timing == 'overbought_wait':
                signal = 'HOLD_BULLISH'
                signal_reasons.append("Wait for pullback - currently overbought")
                entry_quality = 'wait'
        
        return {
            'signal': signal,
            'score': score,
            'signal_reasons': signal_reasons,
            'entry_quality': entry_quality,
            'confidence': self._calculate_confidence(score, len(signal_reasons))
        }
    
    def _calculate_technical_score(self, df):
        """Calculate technical analysis score"""
        result = self.tech_analyzer.calculate_all_indicators(df)
        
        # Handle different return formats
        if isinstance(result, dict) and 'indicators' in result:
            indicators = result['indicators']
        else:
            indicators = result if result else {}
        
        score = 50
        
        # RSI
        rsi = indicators.get('RSI', 50)
        if 40 < rsi < 60:
            score += 10
        elif rsi < 30:
            score += 15  # Oversold bounce
        elif rsi > 70:
            score -= 15  # Overbought
        
        # MACD
        macd = indicators.get('MACD', 0)
        macd_signal = indicators.get('MACD_Signal', 0)
        if macd > macd_signal:
            score += 15
        else:
            score -= 15
        
        # Price vs SMA
        current_price = df['Close'].iloc[-1]
        sma_20 = indicators.get('SMA_20', current_price)
        sma_50 = indicators.get('SMA_50', current_price)
        
        if current_price > sma_20 and current_price > sma_50:
            score += 10
        elif current_price < sma_20 and current_price < sma_50:
            score -= 10
        
        return max(0, min(100, score))
    
    def _calculate_confidence(self, score, reason_count):
        """Calculate confidence level based on score and reasons"""
        base_confidence = score
        
        # Boost confidence for multiple confirming reasons
        if reason_count >= 4:
            base_confidence = min(100, base_confidence + 10)
        elif reason_count >= 3:
            base_confidence = min(100, base_confidence + 5)
        
        if base_confidence >= 80:
            return 'HIGH'
        elif base_confidence >= 60:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _calculate_position_sizing(self, df, info):
        """Calculate position sizing based on ATR and account risk"""
        current_price = df['Close'].iloc[-1]
        
        # Calculate ATR
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=14).mean().iloc[-1]
        
        # Calculate stop loss based on ATR
        stop_distance = atr * self.config['position_sizing_atr_multiplier']
        stop_loss = current_price - stop_distance
        
        # Calculate targets
        risk_amount = current_price - stop_loss
        take_profit_1 = current_price + (risk_amount * 2)  # 2:1 R/R
        take_profit_2 = current_price + (risk_amount * 3)  # 3:1 R/R
        take_profit_3 = current_price + (risk_amount * 5)  # 5:1 R/R
        
        # Position sizing for $10,000 account
        account_size = 10000
        risk_per_trade = account_size * self.config['risk_per_trade']
        shares = int(risk_per_trade / stop_distance)
        position_value = shares * current_price
        
        return {
            'current_price': current_price,
            'atr': atr,
            'stop_loss': stop_loss,
            'stop_distance': stop_distance,
            'stop_distance_pct': (stop_distance / current_price) * 100,
            'take_profit_1': take_profit_1,
            'take_profit_2': take_profit_2,
            'take_profit_3': take_profit_3,
            'shares': shares,
            'position_value': position_value,
            'risk_amount': risk_per_trade,
            'risk_reward_ratios': [2, 3, 5],
            'position_pct_of_account': (position_value / account_size) * 100
        }
    
    def generate_trade_summary(self, analysis):
        """Generate a comprehensive trade summary"""
        summary = f"\n{'='*60}\n"
        summary += f"📊 SWING TRADE ANALYSIS SUMMARY: {analysis['symbol']}\n"
        summary += f"{'='*60}\n"
        
        # Signal and score
        summary += f"\n🎯 SIGNAL: {analysis['signal']} (Score: {analysis['score']:.1f}/100)\n"
        summary += f"🔒 Confidence: {analysis.get('confidence', 'N/A')}\n"
        
        # Reasons
        if 'signal_reasons' in analysis:
            summary += f"\n📋 Key Factors:\n"
            for reason in analysis['signal_reasons']:
                summary += f"  ✓ {reason}\n"
        
        # Entry filters
        if 'filters' in analysis:
            filters = analysis['filters']
            summary += f"\n🔍 Entry Filters: {filters['overall']['filter_percentage']:.1f}% - {filters['overall']['recommendation']}\n"
        
        # Timeframe alignment
        if 'multi_timeframe' in analysis and analysis['multi_timeframe']:
            mtf = analysis['multi_timeframe']
            if 'alignment' in mtf:
                summary += f"📈 Timeframe Alignment: {mtf['alignment']['alignment_percentage']:.1f}%\n"
        
        # Volume analysis
        if 'volume' in analysis and analysis['volume']:
            vol = analysis['volume']
            summary += f"📊 Volume Score: {vol['volume_score']:.0f} ({vol['volume_signal']})\n"
        
        # Position sizing
        if 'position_sizing' in analysis:
            pos = analysis['position_sizing']
            summary += f"\n💰 POSITION SIZING (for $10,000 account):\n"
            summary += f"  Entry: ${pos['current_price']:.2f}\n"
            summary += f"  Stop Loss: ${pos['stop_loss']:.2f} ({pos['stop_distance_pct']:.1f}%)\n"
            summary += f"  Shares: {pos['shares']}\n"
            summary += f"  Position Size: ${pos['position_value']:.2f} ({pos['position_pct_of_account']:.1f}% of account)\n"
            summary += f"  Risk: ${pos['risk_amount']:.2f}\n"
            summary += f"\n  TARGETS:\n"
            summary += f"  TP1 (2:1): ${pos['take_profit_1']:.2f}\n"
            summary += f"  TP2 (3:1): ${pos['take_profit_2']:.2f}\n"
            summary += f"  TP3 (5:1): ${pos['take_profit_3']:.2f}\n"
        
        # Entry quality
        if 'entry_quality' in analysis:
            if analysis['entry_quality'] == 'excellent':
                summary += f"\n⭐ ENTRY QUALITY: EXCELLENT - Ideal entry point detected\n"
            elif analysis['entry_quality'] == 'wait':
                summary += f"\n⏳ ENTRY QUALITY: WAIT - Better entry coming soon\n"
        
        summary += f"\n⏰ Analysis Time: {analysis['timestamp']}\n"
        
        return summary
    
    def analyze_multiple_stocks(self, symbols, max_positions=5):
        """Analyze multiple stocks and rank by score"""
        print(f"\n🚀 ANALYZING {len(symbols)} STOCKS FOR SWING TRADES")
        print("="*60)
        
        all_results = []
        
        for symbol in symbols:
            try:
                result = self.analyze_stock_complete(symbol)
                all_results.append(result)
            except Exception as e:
                print(f"❌ Error analyzing {symbol}: {e}")
        
        # Filter and sort by score
        valid_trades = [r for r in all_results if r['passes_filters'] and r['signal'] in ['BUY', 'STRONG_BUY']]
        valid_trades.sort(key=lambda x: x['score'], reverse=True)
        
        # Generate portfolio summary
        print(f"\n{'='*60}")
        print(f"📊 TOP SWING TRADE OPPORTUNITIES")
        print(f"{'='*60}")
        
        if valid_trades:
            print(f"\nFound {len(valid_trades)} valid trade setups:\n")
            
            for i, trade in enumerate(valid_trades[:max_positions]):
                print(f"{i+1}. {trade['symbol']} - {trade['signal']} (Score: {trade['score']:.1f})")
                print(f"   Reasons: {', '.join(trade['signal_reasons'][:2])}")
                if 'position_sizing' in trade:
                    pos = trade['position_sizing']
                    print(f"   Entry: ${pos['current_price']:.2f}, Stop: ${pos['stop_loss']:.2f}, Target: ${pos['take_profit_1']:.2f}")
                print()
        else:
            print("No valid trade setups found meeting all criteria.")
        
        # Save results
        self._save_analysis_results(all_results, valid_trades)
        
        return valid_trades
    
    def _save_analysis_results(self, all_results, valid_trades):
        """Save analysis results to file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        os.makedirs('data/analysis_results/swing_trades', exist_ok=True)
        
        # Save all results
        filename = f'data/analysis_results/swing_trades/analysis_{timestamp}.json'
        with open(filename, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'total_analyzed': len(all_results),
                'valid_trades': len(valid_trades),
                'all_results': all_results,
                'top_trades': valid_trades[:5]
            }, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to: {filename}")


def main():
    """Test the swing trading analyzer"""
    analyzer = SwingTradingAnalyzer()
    
    # Test with active stocks from config
    config_path = 'config/stocks.json'
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Find active stock list
        active_stocks = []
        for list_name, list_data in config['stock_lists'].items():
            if list_data['active']:
                active_stocks = list_data['symbols'][:10]  # Top 10 for testing
                print(f"Using {list_name}: {len(active_stocks)} stocks")
                break
    else:
        # Default test stocks
        active_stocks = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
    
    # Analyze stocks
    valid_trades = analyzer.analyze_multiple_stocks(active_stocks)
    
    # Show detailed analysis for top pick
    if valid_trades:
        print(f"\n{'='*60}")
        print("📋 DETAILED ANALYSIS - TOP PICK")
        print(analyzer.generate_trade_summary(valid_trades[0]))

if __name__ == "__main__":
    main()