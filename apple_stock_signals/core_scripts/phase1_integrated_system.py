#!/usr/bin/env python3
"""
Phase 1 Integrated Trading System
Combines: Market Regime Detection, Adaptive Signals, Multi-Timeframe Analysis, Volume Breakouts
"""

import pandas as pd
import numpy as np
from datetime import datetime
import json
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from .market_regime_detector import MarketRegimeDetector
    from .adaptive_signal_generator import AdaptiveSignalGenerator
    from .multi_timeframe_analyzer import MultiTimeframeAnalyzer
    from .volume_breakout_analyzer import VolumeBreakoutAnalyzer
except ImportError:
    from market_regime_detector import MarketRegimeDetector
    from adaptive_signal_generator import AdaptiveSignalGenerator
    from multi_timeframe_analyzer import MultiTimeframeAnalyzer
    from volume_breakout_analyzer import VolumeBreakoutAnalyzer

class Phase1IntegratedSystem:
    """
    Integrated system that combines all Phase 1 components for enhanced trading decisions
    """
    
    def __init__(self):
        self.regime_detector = MarketRegimeDetector()
        self.signal_generator = AdaptiveSignalGenerator()
        self.mtf_analyzer = MultiTimeframeAnalyzer()
        self.volume_analyzer = VolumeBreakoutAnalyzer()
        
        # System configuration
        self.config = {
            'min_combined_score': 60,      # Minimum score to generate signal
            'volume_weight': 0.25,         # Weight for volume breakout score
            'mtf_weight': 0.25,           # Weight for multi-timeframe alignment
            'adaptive_weight': 0.30,       # Weight for adaptive signal score
            'regime_weight': 0.20,         # Weight for regime confidence
            'max_signals': 5               # Maximum signals to generate
        }
        
    def run_complete_analysis(self, symbols):
        """
        Run complete Phase 1 analysis on given symbols
        """
        print("\n🚀 PHASE 1 INTEGRATED TRADING SYSTEM")
        print("="*80)
        print(f"Analyzing {len(symbols)} symbols with enhanced strategy...")
        print("="*80)
        
        # Step 1: Detect Market Regime
        print("\n📊 STEP 1: MARKET REGIME DETECTION")
        regime_info = self.regime_detector.detect_regime()
        self._display_regime_info(regime_info)
        
        # Step 2: Generate Adaptive Signals
        print("\n🎯 STEP 2: ADAPTIVE SIGNAL GENERATION")
        adaptive_signals = self.signal_generator.generate_adaptive_signals(symbols)
        
        # Step 3: Multi-Timeframe Analysis
        print("\n📈 STEP 3: MULTI-TIMEFRAME ANALYSIS")
        mtf_analyses = self._run_mtf_analysis(symbols)
        
        # Step 4: Volume Breakout Analysis
        print("\n📊 STEP 4: VOLUME BREAKOUT ANALYSIS")
        volume_analyses = self._run_volume_analysis(symbols)
        
        # Step 5: Combine and Score
        print("\n🔄 STEP 5: COMBINING ANALYSES")
        combined_signals = self._combine_analyses(
            regime_info, adaptive_signals, mtf_analyses, volume_analyses
        )
        
        # Step 6: Generate Final Recommendations
        print("\n✅ STEP 6: FINAL RECOMMENDATIONS")
        recommendations = self._generate_recommendations(combined_signals)
        
        # Save results
        self._save_results(regime_info, combined_signals, recommendations)
        
        return {
            'regime': regime_info,
            'signals': combined_signals,
            'recommendations': recommendations,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def _display_regime_info(self, regime_info):
        """Display market regime information"""
        print(f"\nMarket Regime: {regime_info['regime']}")
        print(f"Confidence: {regime_info['confidence']:.1f}%")
        print(f"Strategy: {regime_info['strategy']}")
        print(f"Risk Multiplier: {regime_info['risk_multiplier']}x")
        
        print("\nRegime Scores:")
        for metric, score in regime_info['scores'].items():
            print(f"  {metric.capitalize()}: {score:.1f}")
    
    def _run_mtf_analysis(self, symbols):
        """Run multi-timeframe analysis on all symbols"""
        analyses = {}
        
        for symbol in symbols[:10]:  # Limit to avoid API rate limits
            print(f"  Analyzing {symbol}...", end='', flush=True)
            try:
                analysis = self.mtf_analyzer.analyze_stock(symbol)
                if analysis and 'alignment' in analysis:
                    analyses[symbol] = analysis
                    print(f" ✅ Alignment: {analysis['alignment']['alignment_percentage']:.1f}%")
                else:
                    print(f" ❌ Insufficient data")
            except Exception as e:
                print(f" ❌ Error: {str(e)}")
        
        return analyses
    
    def _run_volume_analysis(self, symbols):
        """Run volume breakout analysis on all symbols"""
        analyses = {}
        
        for symbol in symbols[:10]:  # Limit to avoid API rate limits
            print(f"  Analyzing {symbol}...", end='', flush=True)
            try:
                analysis = self.volume_analyzer.analyze_volume_breakout(symbol)
                if analysis:
                    analyses[symbol] = analysis
                    print(f" ✅ Breakout Score: {analysis['breakout_score']:.1f}")
                else:
                    print(f" ❌ Analysis failed")
            except Exception as e:
                print(f" ❌ Error: {str(e)}")
        
        return analyses
    
    def _combine_analyses(self, regime_info, adaptive_signals, mtf_analyses, volume_analyses):
        """Combine all analyses into unified signals"""
        combined_signals = []
        
        # Get adaptive signals as a dictionary for easier lookup
        adaptive_dict = {s['symbol']: s for s in adaptive_signals.get('signals', [])}
        
        # Process each symbol that has adaptive signal
        for symbol, adaptive_signal in adaptive_dict.items():
            combined = {
                'symbol': symbol,
                'action': adaptive_signal['action'],
                'scores': {
                    'adaptive': adaptive_signal['score'],
                    'mtf': 50,  # Default if not available
                    'volume': 50,  # Default if not available
                    'regime': regime_info['confidence']
                },
                'current_price': adaptive_signal['current_price'],
                'indicators': adaptive_signal['indicators'],
                'position_params': {
                    'position_size': adaptive_signal.get('position_size', 0),
                    'stop_loss': adaptive_signal.get('stop_loss', 0),
                    'take_profit_1': adaptive_signal.get('take_profit_1', 0),
                    'take_profit_2': adaptive_signal.get('take_profit_2', 0),
                    'take_profit_3': adaptive_signal.get('take_profit_3', 0)
                }
            }
            
            # Add MTF score if available
            if symbol in mtf_analyses and 'alignment' in mtf_analyses[symbol]:
                combined['scores']['mtf'] = mtf_analyses[symbol]['alignment']['alignment_percentage']
                combined['mtf_recommendation'] = mtf_analyses[symbol]['alignment']['recommendation']
                combined['timeframe_trends'] = {}
                for tf in ['weekly', 'daily', '4hour']:
                    if tf in mtf_analyses[symbol]:
                        combined['timeframe_trends'][tf] = mtf_analyses[symbol][tf]['trend']
            
            # Add volume score if available
            if symbol in volume_analyses:
                combined['scores']['volume'] = volume_analyses[symbol]['breakout_score']
                combined['volume_ratio'] = volume_analyses[symbol]['volume_analysis']['volume_ratio']
                combined['breakout_type'] = volume_analyses[symbol]['breakout_analysis']['breakout_type']
                combined['volume_confirmed'] = volume_analyses[symbol]['breakout_analysis']['volume_confirmed']
            
            # Calculate combined score
            combined['combined_score'] = self._calculate_combined_score(combined['scores'])
            
            # Add confidence level
            combined['confidence'] = self._determine_confidence(combined)
            
            combined_signals.append(combined)
        
        # Sort by combined score
        combined_signals.sort(key=lambda x: x['combined_score'], reverse=True)
        
        return combined_signals
    
    def _calculate_combined_score(self, scores):
        """Calculate weighted combined score"""
        weighted_score = (
            scores['adaptive'] * self.config['adaptive_weight'] +
            scores['mtf'] * self.config['mtf_weight'] +
            scores['volume'] * self.config['volume_weight'] +
            scores['regime'] * self.config['regime_weight']
        )
        
        return round(weighted_score, 1)
    
    def _determine_confidence(self, signal):
        """Determine confidence level for a signal"""
        score = signal['combined_score']
        
        # Check for confirmations
        confirmations = 0
        
        if signal['scores']['adaptive'] >= 65:
            confirmations += 1
        if signal['scores']['mtf'] >= 70:
            confirmations += 1
        if signal['scores']['volume'] >= 60:
            confirmations += 1
        if signal.get('volume_confirmed', False):
            confirmations += 1
        
        # Determine confidence
        if score >= 80 and confirmations >= 3:
            return 'VERY_HIGH'
        elif score >= 70 and confirmations >= 2:
            return 'HIGH'
        elif score >= 60 and confirmations >= 1:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _generate_recommendations(self, combined_signals):
        """Generate final trading recommendations"""
        recommendations = []
        
        # Filter by minimum score
        valid_signals = [s for s in combined_signals 
                        if s['combined_score'] >= self.config['min_combined_score']]
        
        # Limit to max signals
        top_signals = valid_signals[:self.config['max_signals']]
        
        for signal in top_signals:
            rec = {
                'symbol': signal['symbol'],
                'action': signal['action'],
                'confidence': signal['confidence'],
                'combined_score': signal['combined_score'],
                'entry_price': signal['current_price'],
                'stop_loss': signal['position_params']['stop_loss'],
                'take_profit_1': signal['position_params']['take_profit_1'],
                'take_profit_2': signal['position_params']['take_profit_2'],
                'position_size': signal['position_params']['position_size'],
                'reasons': self._generate_reasons(signal),
                'risks': self._identify_risks(signal)
            }
            
            recommendations.append(rec)
        
        return recommendations
    
    def _generate_reasons(self, signal):
        """Generate reasons for the recommendation"""
        reasons = []
        
        # Adaptive signal strength
        if signal['scores']['adaptive'] >= 70:
            reasons.append(f"Strong adaptive signal ({signal['scores']['adaptive']:.1f})")
        
        # Multi-timeframe alignment
        if signal['scores']['mtf'] >= 70:
            reasons.append(f"Excellent timeframe alignment ({signal['scores']['mtf']:.1f}%)")
        elif signal['scores']['mtf'] >= 60:
            reasons.append(f"Good timeframe alignment ({signal['scores']['mtf']:.1f}%)")
        
        # Volume confirmation
        if signal.get('volume_confirmed', False):
            reasons.append(f"Volume breakout confirmed ({signal.get('volume_ratio', 0):.1f}x average)")
        elif signal['scores']['volume'] >= 60:
            reasons.append(f"Strong volume patterns (score: {signal['scores']['volume']:.1f})")
        
        # Technical indicators
        indicators = signal.get('indicators', {})
        if indicators.get('rsi', 50) < 30:
            reasons.append("RSI oversold bounce opportunity")
        elif indicators.get('macd_signal', 0) > 0:
            reasons.append("MACD bullish crossover")
        
        # Breakout type
        if signal.get('breakout_type') == 'resistance':
            reasons.append("Breaking resistance level")
        elif signal.get('breakout_type') == 'consolidation':
            reasons.append("Breaking out from consolidation")
        
        return reasons
    
    def _identify_risks(self, signal):
        """Identify potential risks for the signal"""
        risks = []
        
        # Low confidence
        if signal['confidence'] == 'LOW':
            risks.append("Low overall confidence - consider smaller position")
        
        # Poor volume
        if signal['scores']['volume'] < 40:
            risks.append("Weak volume - potential false breakout")
        
        # Overbought conditions
        indicators = signal.get('indicators', {})
        if indicators.get('rsi', 50) > 70:
            risks.append("RSI overbought - wait for pullback")
        
        # Misaligned timeframes
        if signal['scores']['mtf'] < 50:
            risks.append("Timeframes not aligned - conflicting signals")
        
        # High ATR (volatility)
        if indicators.get('atr', 0) > signal['current_price'] * 0.03:
            risks.append("High volatility - wider stops needed")
        
        return risks if risks else ["Standard market risk applies"]
    
    def _save_results(self, regime_info, combined_signals, recommendations):
        """Save all results to files"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create output directory
        output_dir = 'outputs/phase1_integrated'
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert numpy types to Python native types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj
        
        # Save complete analysis
        complete_results = {
            'timestamp': timestamp,
            'market_regime': convert_to_serializable(regime_info),
            'combined_signals': convert_to_serializable(combined_signals),
            'recommendations': convert_to_serializable(recommendations)
        }
        
        with open(f'{output_dir}/analysis_{timestamp}.json', 'w') as f:
            json.dump(complete_results, f, indent=2, default=str)
        
        # Save summary report
        self._generate_summary_report(recommendations, regime_info, output_dir, timestamp)
        
        print(f"\n💾 Results saved to {output_dir}/")
    
    def _generate_summary_report(self, recommendations, regime_info, output_dir, timestamp):
        """Generate human-readable summary report"""
        report = "="*80 + "\n"
        report += "📊 PHASE 1 INTEGRATED TRADING SYSTEM - SUMMARY REPORT\n"
        report += "="*80 + "\n"
        report += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        report += f"Market Regime: {regime_info['regime']} (Confidence: {regime_info['confidence']:.1f}%)\n"
        report += f"Strategy: {regime_info['strategy']}\n"
        report += f"Risk Multiplier: {regime_info['risk_multiplier']}x\n"
        report += "\n"
        
        if recommendations:
            report += "🎯 TOP TRADING RECOMMENDATIONS:\n"
            report += "-"*80 + "\n"
            
            for i, rec in enumerate(recommendations, 1):
                report += f"\n{i}. {rec['symbol']} - {rec['action']}\n"
                report += f"   Confidence: {rec['confidence']}\n"
                report += f"   Combined Score: {rec['combined_score']:.1f}/100\n"
                report += f"   Entry: ${rec['entry_price']:.2f}\n"
                report += f"   Stop Loss: ${rec['stop_loss']:.2f}\n"
                report += f"   Target 1: ${rec['take_profit_1']:.2f}\n"
                report += f"   Position Size: {rec['position_size']} shares\n"
                
                report += "   \n   Reasons:\n"
                for reason in rec['reasons']:
                    report += f"   • {reason}\n"
                
                report += "   \n   Risks:\n"
                for risk in rec['risks']:
                    report += f"   ⚠️ {risk}\n"
        else:
            report += "\n❌ No signals meet the minimum criteria in current market conditions.\n"
            report += "Consider waiting for better setups or adjusting strategy parameters.\n"
        
        report += "\n" + "="*80 + "\n"
        report += "⚠️ DISCLAIMER: This is for educational purposes only. Not financial advice.\n"
        report += "="*80 + "\n"
        
        # Save report
        with open(f'{output_dir}/summary_{timestamp}.txt', 'w') as f:
            f.write(report)
        
        # Also print to console
        print(report)


def main():
    """Run the Phase 1 integrated system"""
    # Initialize system
    system = Phase1IntegratedSystem()
    
    # Define symbols to analyze
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 
               'META', 'TSLA', 'BRK-B', 'JPM', 'JNJ']
    
    # Run complete analysis
    results = system.run_complete_analysis(symbols)
    
    print("\n✅ Phase 1 Integrated System Analysis Complete!")
    print(f"Generated {len(results['recommendations'])} high-confidence trading signals.")


if __name__ == "__main__":
    main()