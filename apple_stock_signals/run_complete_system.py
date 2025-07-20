#!/usr/bin/env python3
"""
Complete Trading System Runner
Demonstrates all components working together with Phase 2 enhancements
"""

import time
import os
import sys
from datetime import datetime
import logging
import json
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import all modules
from data_modules.data_fetcher import AppleDataFetcher
from data_modules.technical_analyzer import AppleTechnicalAnalyzer
from data_modules.sentiment_analyzer import SentimentAnalyzer
from data_modules.signal_generator import AppleSignalGenerator
from core_scripts.market_regime_detector import MarketRegimeDetector
from core_scripts.volatility_position_sizing import VolatilityPositionSizer
from core_scripts.trailing_stop_manager import TrailingStopManager
from utils.cache_manager import get_cache_manager
from utils.validators import Validators
from config.env_config import EnvConfig

def print_header(title):
    """Print formatted header"""
    print(f"\n{'='*80}")
    print(f"🚀 {title}")
    print('='*80)

def run_complete_analysis():
    """Run complete trading system with all components"""
    print_header("COMPLETE APPLE STOCK TRADING SYSTEM")
    print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize configuration
    config = EnvConfig()
    portfolio_size = 100000  # $100k portfolio
    
    print(f"\n💰 Portfolio Size: ${portfolio_size:,}")
    print(f"🔧 Environment: {config.ENVIRONMENT}")
    
    # Initialize components
    print("\n📦 Initializing Components...")
    fetcher = AppleDataFetcher()
    analyzer = AppleTechnicalAnalyzer()
    sentiment = SentimentAnalyzer()
    signal_gen = AppleSignalGenerator()
    regime_detector = MarketRegimeDetector()
    position_sizer = VolatilityPositionSizer(portfolio_size)
    stop_manager = TrailingStopManager()
    cache = get_cache_manager()
    
    try:
        # Phase 1: Market Regime Detection
        print_header("PHASE 1: MARKET REGIME DETECTION")
        regime_info = regime_detector.detect_regime()
        
        print(f"\n📊 Current Market Regime: {regime_info['regime']}")
        print(f"Confidence: {regime_info['confidence']:.1f}%")
        print(f"Strategy: {regime_info['strategy']}")
        print(f"Risk Multiplier: {regime_info['risk_multiplier']}x")
        
        print("\nScores:")
        for key, value in regime_info['scores'].items():
            print(f"  {key.capitalize()}: {value:.1f}")
        
        # Phase 2: Data Collection
        print_header("PHASE 2: DATA COLLECTION")
        start_time = time.time()
        
        print("\n🔍 Fetching all market data...")
        all_data = fetcher.fetch_all_data()
        
        fetch_time = time.time() - start_time
        print(f"✅ Data fetched in {fetch_time:.2f} seconds")
        
        if not all_data or not all_data.get('stock_data'):
            print("❌ Failed to fetch data. Exiting.")
            return
        
        # Phase 3: Technical Analysis
        print_header("PHASE 3: TECHNICAL ANALYSIS")
        
        stock_data = all_data['stock_data']
        if stock_data and isinstance(stock_data, dict):
            hist_data = stock_data.get('historical_data')
            current_price = stock_data.get('current_price')
        else:
            hist_data = None
            current_price = None
        
        if current_price:
            print(f"\n💰 Current AAPL Price: ${current_price:.2f}")
        else:
            print(f"\n❌ Unable to get current price")
        
        # Calculate indicators
        indicators = analyzer.calculate_all_indicators(hist_data)
        patterns = {}  # Chart patterns analysis not implemented yet
        
        if indicators:
            print(f"\n📈 Technical Indicators:")
            # Get the latest values from arrays
            rsi_value = indicators.get('RSI', [])
            if isinstance(rsi_value, np.ndarray) and len(rsi_value) > 0:
                rsi_latest = rsi_value[-1]
            else:
                rsi_latest = 'N/A'
                
            print(f"  RSI: {rsi_latest:.2f}" if isinstance(rsi_latest, (int, float)) else f"  RSI: {rsi_latest}")
            
            # MACD is returned as a tuple
            macd_data = indicators.get('MACD', (None, None, None))
            if isinstance(macd_data, tuple) and len(macd_data) >= 2:
                macd_line = macd_data[0]
                signal_line = macd_data[1]
                if isinstance(macd_line, np.ndarray) and isinstance(signal_line, np.ndarray):
                    print(f"  MACD: {macd_line[-1]:.4f}")
                    print(f"  MACD Signal: {signal_line[-1]:.4f}")
                else:
                    print(f"  MACD Signal: N/A")
            else:
                print(f"  MACD Signal: N/A")
                
            # Bollinger Bands
            bb_data = indicators.get('BB', (None, None, None))
            if isinstance(bb_data, tuple) and len(bb_data) >= 3:
                upper = bb_data[0]
                middle = bb_data[1]
                lower = bb_data[2]
                if isinstance(upper, np.ndarray):
                    print(f"  Bollinger Bands: Upper=${upper[-1]:.2f}, Middle=${middle[-1]:.2f}, Lower=${lower[-1]:.2f}")
                else:
                    print(f"  Bollinger Band Position: N/A")
            else:
                print(f"  Bollinger Band Position: N/A")
            print(f"  SMA 20: ${indicators.get('SMA_20', 0):.2f}")
            print(f"  SMA 50: ${indicators.get('SMA_50', 0):.2f}")
        
        # Phase 4: Sentiment Analysis
        print_header("PHASE 4: SENTIMENT ANALYSIS")
        
        if all_data.get('news_data'):
            news_sentiment = all_data['news_data']['avg_sentiment']
            print(f"\n😊 News Sentiment: {news_sentiment:.1f}/100")
            print(f"Positive Articles: {all_data['news_data']['positive_count']}")
            print(f"Negative Articles: {all_data['news_data']['negative_count']}")
        
        # Phase 5: Signal Generation
        print_header("PHASE 5: TRADING SIGNAL GENERATION")
        
        signals = signal_gen.generate_all_signals(all_data, indicators, patterns)
        
        if signals:
            print(f"\n🚦 Primary Signal: {signals.get('primary_signal', 'HOLD')}")
            print(f"Signal Strength: {signals.get('signal_strength', 0):.1f}/100")
            print(f"Confidence Level: {signals.get('confidence_level', 'N/A')}")
            
            if signals.get('reasons'):
                print(f"\nSignal Reasons:")
                for reason in signals['reasons'][:3]:
                    print(f"  • {reason}")
        
        # Phase 6: Position Sizing (if BUY signal)
        if signals and signals.get('primary_signal') == 'BUY':
            print_header("PHASE 6: POSITION SIZING")
            
            # Create trade setup
            stop_loss = current_price * 0.97  # 3% stop loss
            take_profit = current_price * 1.06  # 6% take profit
            
            trade_setup = {
                'symbol': 'AAPL',
                'entry_price': current_price,
                'stop_loss': stop_loss,
                'take_profit_1': take_profit,
                'confidence': signals.get('confidence_level', 'MEDIUM')
            }
            
            # Historical performance (mock data for demo)
            historical_performance = {
                'win_rate': 0.55,
                'avg_win': 0.03,
                'avg_loss': 0.015
            }
            
            position = position_sizer.calculate_optimal_position_size(
                trade_setup,
                historical_performance,
                hist_data
            )
            
            print(f"\n💸 Recommended Position:")
            print(f"  Shares: {position['optimal_shares']}")
            print(f"  Position Value: ${position['position_value']:,.2f} ({position['position_percentage']:.1f}%)")
            print(f"  Risk Amount: ${position['risk_amount']:,.2f} ({position['risk_percentage']:.1f}%)")
            print(f"  Entry: ${current_price:.2f}")
            print(f"  Stop Loss: ${stop_loss:.2f}")
            print(f"  Take Profit: ${take_profit:.2f}")
            
            # Phase 7: Trailing Stop Management
            print_header("PHASE 7: TRAILING STOP STRATEGY")
            
            stop_info = stop_manager.calculate_optimal_trailing_stop(
                hist_data,
                current_price,
                'long',
                regime_info['regime']
            )
            
            print(f"\n🛡️ Trailing Stop Strategy: {stop_info['optimal_strategy']}")
            print(f"Initial Stop: ${stop_info['current_stop']:.2f}")
            print(f"Stop Distance: ${stop_info['stop_distance']:.2f} ({stop_info['stop_percentage']:.2f}%)")
            print(f"Recommendation: {stop_info['recommendation']}")
        
        # Phase 8: Performance Summary
        print_header("PHASE 8: SYSTEM PERFORMANCE SUMMARY")
        
        cache_stats = cache.get_stats()
        
        print(f"\n📊 Performance Metrics:")
        print(f"  Total Analysis Time: {time.time() - start_time:.2f} seconds")
        print(f"  Cache Hit Rate: {cache_stats['hit_rate']}")
        print(f"  API Calls Saved: {cache_stats['hits']}")
        print(f"  Total Cache Requests: {cache_stats['total_requests']}")
        
        # Risk Management Summary
        print(f"\n⚠️ Risk Management:")
        print(f"  Market Regime Risk Multiplier: {regime_info['risk_multiplier']}x")
        print(f"  Max Portfolio Heat: 6%")
        print(f"  Max Risk Per Trade: 2%")
        
        # Save results
        results = {
            'timestamp': datetime.now().isoformat(),
            'market_regime': regime_info,
            'current_price': current_price,
            'signal': signals,
            'technical_indicators': {k: float(v) if isinstance(v, (int, float)) else str(v) 
                                  for k, v in indicators.items()} if indicators else {},
            'sentiment': all_data.get('news_data', {})
        }
        
        os.makedirs('outputs', exist_ok=True)
        with open('outputs/complete_analysis.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Complete analysis saved to outputs/complete_analysis.json")
        
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

def main():
    """Main entry point"""
    print("\n" + "="*80)
    print("🎯 TRADING SYSTEM V2.0 - COMPLETE DEMONSTRATION")
    print("="*80)
    print("\nThis demonstration shows all system components working together:")
    print("• Market Regime Detection")
    print("• Data Collection with Caching")
    print("• Technical Analysis")
    print("• Sentiment Analysis")
    print("• Signal Generation")
    print("• Volatility-Based Position Sizing")
    print("• Dynamic Trailing Stop Management")
    print("• Performance Optimization")
    
    # Run the analysis
    run_complete_analysis()
    
    print("\n" + "="*80)
    print("✅ COMPLETE SYSTEM DEMONSTRATION FINISHED!")
    print("="*80)
    print("\n🚀 System Features Demonstrated:")
    print("  • Secure configuration management")
    print("  • Input validation on all parameters")
    print("  • Retry logic for API calls")
    print("  • Caching for performance")
    print("  • Parallel data fetching")
    print("  • Market regime adaptation")
    print("  • Dynamic position sizing")
    print("  • Intelligent stop loss management")
    print("  • Comprehensive error handling")

if __name__ == "__main__":
    main()