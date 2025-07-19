#!/usr/bin/env python3
"""
Apple Stock Signal Generator - Main Script
Analyzes AAPL using technical indicators and sentiment analysis
"""

import sys
import os
import json
import pandas as pd
from datetime import datetime
import logging

# Import our modules
from data_fetcher import AppleDataFetcher
from technical_analyzer import AppleTechnicalAnalyzer
from sentiment_analyzer import SentimentAnalyzer
from signal_generator import AppleSignalGenerator
from config import ANALYSIS_SETTINGS, VERIFICATION_SETTINGS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('apple_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AppleStockAnalyzer:
    def __init__(self):
        self.data_fetcher = AppleDataFetcher()
        self.technical_analyzer = AppleTechnicalAnalyzer()
        self.signal_generator = AppleSignalGenerator()
        
    def run_analysis(self):
        """Run complete Apple stock analysis"""
        print("🍎 APPLE STOCK ANALYSIS STARTING...")
        print("=" * 60)
        
        try:
            # 1. Fetch all data
            all_data = self.data_fetcher.fetch_all_data()
            
            if not all_data or not all_data.get('stock_data'):
                print("❌ Failed to fetch stock data. Please check your internet connection.")
                return None
            
            # 2. Calculate technical indicators
            technical_indicators = self.technical_analyzer.calculate_all_indicators(all_data['stock_data'])
            
            # 3. Add technical indicators to data
            all_data['technical_indicators'] = technical_indicators
            
            # 4. Generate trading signal
            signal_result = self.signal_generator.generate_signal(all_data)
            
            # 5. Display final results (including verification)
            self.display_final_results(signal_result, all_data['verification_results'])
            
            # 6. Save results
            self.save_results(signal_result, all_data)
            
            return signal_result
            
        except Exception as e:
            logger.error(f"Error in analysis: {e}")
            print(f"\n❌ Analysis failed: {e}")
            return None
    
    def display_final_results(self, signal_result, verification_results):
        """Display final trading recommendation"""
        print("\n" + "=" * 60)
        print("🎯 FINAL TRADING RECOMMENDATION")
        print("=" * 60)
        
        # Data quality warning if needed
        if verification_results['confidence_score'] < VERIFICATION_SETTINGS['MIN_DATA_CONFIDENCE']:
            print("⚠️ WARNING: Data quality issues detected!")
            print(f"Data confidence: {verification_results['confidence_score']:.1f}%")
            print("Consider manual verification before trading.\n")
        
        # Signal summary
        signal = signal_result['signal']
        confidence = signal_result['confidence']
        
        # Adjust confidence based on data quality
        adjusted_confidence = confidence * (verification_results['confidence_score'] / 100)
        
        # Color coding for signal
        if signal in ['STRONG_BUY', 'BUY']:
            signal_color = "🟢"
        elif signal in ['STRONG_SELL', 'SELL']:
            signal_color = "🔴"
        else:
            signal_color = "🟡"
        
        print(f"{signal_color} SIGNAL: {signal}")
        print(f"📊 SIGNAL CONFIDENCE: {confidence:.1f}%")
        print(f"🔍 DATA QUALITY: {verification_results['confidence_score']:.1f}%")
        print(f"📈 ADJUSTED CONFIDENCE: {adjusted_confidence:.1f}%")
        print(f"📈 FINAL SCORE: {signal_result['final_score']:.1f}/100")
        
        # Price targets
        targets = signal_result['price_targets']
        print(f"\n💰 PRICE TARGETS:")
        print(f"Entry Price: ${targets['entry_price']:.2f}")
        
        if targets['stop_loss']:
            print(f"Stop Loss: ${targets['stop_loss']:.2f}")
            print(f"Take Profit 1: ${targets['take_profit_1']:.2f}")
            print(f"Take Profit 2: ${targets['take_profit_2']:.2f}")
            print(f"Risk/Reward Ratio: {targets['risk_reward_ratio']:.2f}:1")
        
        # Component scores
        print(f"\n📊 COMPONENT BREAKDOWN:")
        components = signal_result['component_scores']
        print(f"Technical Score: {components['technical']:.1f}/100")
        print(f"Sentiment Score: {components['sentiment']:.1f}/100")
        print(f"Fundamental Score: {components['fundamental']:.1f}/100")
        
        # Key reasons
        print(f"\n🔍 KEY ANALYSIS POINTS:")
        for detail in signal_result['analysis_details']:
            print(f"  • {detail}")
        
        # Data verification summary
        print(f"\n📋 DATA VERIFICATION:")
        if verification_results['sources_compared'] > 1:
            print(f"  ✅ Cross-verified with {verification_results['sources_compared']} sources")
        else:
            print(f"  ⚠️ Only 1 data source available")
        
        if verification_results['anomalies_detected']:
            high_anomalies = sum(1 for a in verification_results['anomalies_detected'] if a['severity'] == 'HIGH')
            if high_anomalies > 0:
                print(f"  🔴 {high_anomalies} high-severity anomalies detected")
            else:
                print(f"  🟡 {len(verification_results['anomalies_detected'])} minor anomalies detected")
        else:
            print(f"  ✅ No significant anomalies detected")
        
        # Position sizing recommendation
        if signal in ['BUY', 'STRONG_BUY'] and adjusted_confidence > 50:
            print(f"\n💡 POSITION SIZING RECOMMENDATION:")
            print(f"• Risk no more than {ANALYSIS_SETTINGS['MAX_RISK_PERCENT']:.0f}% of portfolio on this trade")
            print(f"• With stop-loss at ${targets['stop_loss']:.2f}, risk per share is ${targets['risk_amount']:.2f}")
            print(f"• For $10,000 portfolio, max risk = ${10000 * ANALYSIS_SETTINGS['MAX_RISK_PERCENT'] / 100:.0f}")
            print(f"• Suggested position size: {int((10000 * ANALYSIS_SETTINGS['MAX_RISK_PERCENT'] / 100) / targets['risk_amount'])} shares")
            
            # Adjust position size based on data quality
            if verification_results['confidence_score'] < VERIFICATION_SETTINGS['MIN_DATA_CONFIDENCE']:
                print(f"• ⚠️ Consider reducing position size by 50% due to data quality concerns")
        
        print(f"\n⏰ Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📊 Overall recommendation reliability: {adjusted_confidence:.1f}%")
    
    def save_results(self, signal_result, all_data):
        """Save analysis results to files"""
        try:
            # Create outputs directory if it doesn't exist
            os.makedirs('outputs', exist_ok=True)
            
            # Save CSV summary
            csv_data = {
                'Date': [datetime.now().strftime('%Y-%m-%d')],
                'Time': [datetime.now().strftime('%H:%M:%S')],
                'Symbol': ['AAPL'],
                'Signal': [signal_result['signal']],
                'Confidence': [signal_result['confidence']],
                'Final_Score': [signal_result['final_score']],
                'Entry_Price': [signal_result['price_targets']['entry_price']],
                'Stop_Loss': [signal_result['price_targets']['stop_loss']],
                'Take_Profit_1': [signal_result['price_targets']['take_profit_1']],
                'Risk_Reward_Ratio': [signal_result['price_targets']['risk_reward_ratio']],
                'Technical_Score': [signal_result['component_scores']['technical']],
                'Sentiment_Score': [signal_result['component_scores']['sentiment']],
                'Fundamental_Score': [signal_result['component_scores']['fundamental']],
                'Data_Quality': [signal_result['data_quality']],
                'Analysis_Details': ['; '.join(signal_result['analysis_details'])]
            }
            
            df = pd.DataFrame(csv_data)
            csv_filename = f"outputs/aapl_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(csv_filename, index=False)
            
            # Save detailed JSON
            json_data = {
                'signal_result': signal_result,
                'verification_results': all_data['verification_results'],
                'raw_data': {
                    'news_data': all_data['news_data'],
                    'social_data': all_data['social_data'],
                    'fundamental_data': all_data['fundamental_data']
                }
            }
            
            json_filename = f"outputs/aapl_detailed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(json_filename, 'w') as f:
                json.dump(json_data, f, indent=2, default=str)
            
            print(f"\n💾 Results saved to:")
            print(f"  • {csv_filename}")
            print(f"  • {json_filename}")
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            print(f"\n⚠️ Warning: Could not save results to file: {e}")

def main():
    """Main function to run Apple stock analysis"""
    # Print startup info
    print("\n🍎 Apple Stock Signal Generator v1.0")
    print("=====================================")
    print("📊 Analyzing AAPL with technical indicators, sentiment, and fundamentals")
    print("🔍 Data verification enabled for maximum accuracy")
    print()
    
    # Check for API keys
    from config import API_KEYS
    if API_KEYS['NEWS_API_KEY'] == 'your_news_api_key_here':
        print("⚠️ Warning: News API key not configured")
        print("  Set NEWS_API_KEY environment variable for real news sentiment")
        print("  Using mock news data for demonstration\n")
    
    # Run analysis
    analyzer = AppleStockAnalyzer()
    result = analyzer.run_analysis()
    
    if result:
        print("\n✅ Analysis completed successfully!")
        
        # Show risk warning
        print("\n" + "="*60)
        print("⚠️ RISK WARNING")
        print("="*60)
        print("This analysis is for informational purposes only.")
        print("Always do your own research and consult with a financial advisor.")
        print("Past performance does not guarantee future results.")
        print("="*60)
    else:
        print("\n❌ Analysis failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()