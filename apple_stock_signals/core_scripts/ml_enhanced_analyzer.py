#!/usr/bin/env python3
"""
ML-Enhanced Trading Analyzer
Combines traditional signals with machine learning for better accuracy
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import json
from datetime import datetime
from typing import Dict, List, Optional

# Import existing modules
from core_scripts.config import STOCKS
from core_scripts.enhanced_trading_analyzer import main as run_enhanced_analysis
from ml_models.basic_ml_predictor import BasicMLPredictor
from paper_trading.paper_trader import PaperTradingAccount

class MLEnhancedAnalyzer:
    def __init__(self, paper_trading=True):
        self.ml_predictor = BasicMLPredictor()
        self.paper_trading = paper_trading
        
        if paper_trading:
            self.paper_account = PaperTradingAccount(initial_balance=10000, account_name="ml_strategy")
            print("📄 Paper Trading Mode Enabled")
        else:
            print("💰 Live Trading Mode (Signals Only)")
    
    def analyze_with_ml(self, symbol: str = None):
        """Run analysis with ML enhancement"""
        symbols = [symbol] if symbol else STOCKS
        
        print(f"\n🚀 ML-Enhanced Trading Analysis")
        print("=" * 80)
        print(f"Mode: {'Paper Trading' if self.paper_trading else 'Live Signals'}")
        print(f"Analyzing: {', '.join(symbols)}")
        print("=" * 80)
        
        # First, run traditional analysis
        traditional_results = run_enhanced_analysis()
        
        if not traditional_results or 'stocks' not in traditional_results:
            print("❌ Failed to get traditional analysis")
            return None
        
        # Ensure ML model is loaded
        if not self.ml_predictor.load_model():
            print("\n⚠️ No ML model found. Training new model...")
            self._train_models(symbols)
        
        # Enhance signals with ML
        enhanced_results = {
            'analysis_date': datetime.now().strftime('%Y-%m-%d'),
            'analysis_time': datetime.now().strftime('%H:%M:%S'),
            'mode': 'paper_trading' if self.paper_trading else 'live_signals',
            'stocks': {}
        }
        
        for symbol, data in traditional_results.get('stocks', {}).items():
            if symbol not in symbols:
                continue
            
            print(f"\n🔍 Analyzing {symbol} with ML...")
            
            # Load historical data for ML features
            hist_file = f"historical_data/{symbol}_historical_data.csv"
            if os.path.exists(hist_file):
                hist_data = pd.read_csv(hist_file, index_col='Date', parse_dates=True)
                
                # Get ML prediction
                ml_signal = self.ml_predictor.predict_signal(
                    hist_data,
                    traditional_signal={
                        'signal': data.get('signal', 'HOLD'),
                        'score': data.get('score', 50),
                        'confidence': data.get('score', 50) / 100  # Convert to 0-1 range
                    }
                )
                
                # Enhance the traditional data with ML insights
                enhanced_data = data.copy()
                enhanced_data['ml_signal'] = ml_signal
                enhanced_data['final_signal'] = ml_signal.get('signal', data.get('signal'))
                enhanced_data['combined_confidence'] = ml_signal.get('confidence', data.get('score'))
                enhanced_data['ml_buy_probability'] = ml_signal.get('buy_probability', 0.5)
                enhanced_data['ml_source'] = ml_signal.get('source', 'unknown')
                
                # Execute paper trade if conditions met
                if self.paper_trading and self._should_trade(enhanced_data):
                    self._execute_paper_trade(symbol, enhanced_data)
                
                enhanced_results['stocks'][symbol] = enhanced_data
                
                # Display results
                self._display_ml_analysis(symbol, data, ml_signal)
            else:
                print(f"   ⚠️ No historical data for {symbol}")
                enhanced_results['stocks'][symbol] = data
        
        # Save enhanced results
        self._save_results(enhanced_results)
        
        # Generate paper trading report if enabled
        if self.paper_trading:
            self._generate_trading_report()
        
        return enhanced_results
    
    def _train_models(self, symbols: List[str]):
        """Train ML models for specified symbols"""
        for symbol in symbols:
            hist_file = f"historical_data/{symbol}_historical_data.csv"
            if os.path.exists(hist_file):
                print(f"\n🧠 Training ML model for {symbol}...")
                hist_data = pd.read_csv(hist_file, index_col='Date', parse_dates=True)
                self.ml_predictor.train_model(hist_data)
                break  # Train on first available symbol
    
    def _should_trade(self, signal_data: Dict) -> bool:
        """Determine if we should execute a trade"""
        # Only trade on strong signals with high confidence
        signal = signal_data.get('final_signal', 'HOLD')
        confidence = signal_data.get('combined_confidence', 0)
        
        if signal in ['STRONG_BUY', 'BUY'] and confidence > 65:
            return True
        elif signal in ['STRONG_SELL', 'SELL'] and confidence > 65:
            # For now, only handle long positions
            # Could add short selling logic here
            return False
        
        return False
    
    def _execute_paper_trade(self, symbol: str, signal_data: Dict):
        """Execute paper trade based on signal"""
        signal = signal_data.get('final_signal')
        current_price = signal_data.get('current_price', 0)
        
        if not current_price:
            return
        
        # Get position sizing from signal data
        position_data = signal_data.get('position_sizing', {})
        shares = position_data.get('shares_to_buy', 0)
        
        if shares > 0 and signal in ['STRONG_BUY', 'BUY']:
            # Execute buy
            success = self.paper_account.execute_trade(
                symbol=symbol,
                action='BUY',
                quantity=shares,
                price=current_price,
                signal_data={
                    'signal': signal,
                    'traditional_score': signal_data.get('score'),
                    'ml_confidence': signal_data.get('ml_signal', {}).get('confidence'),
                    'combined_confidence': signal_data.get('combined_confidence'),
                    'stop_loss': signal_data.get('stop_loss'),
                    'take_profit_1': signal_data.get('take_profit_1')
                }
            )
            
            if success:
                print(f"\n💼 Paper Trade Executed!")
                print(f"   Stop Loss: ${signal_data.get('stop_loss', 0):.2f}")
                print(f"   Take Profit: ${signal_data.get('take_profit_1', 0):.2f}")
    
    def _display_ml_analysis(self, symbol: str, traditional: Dict, ml_signal: Dict):
        """Display ML analysis results"""
        print(f"\n📊 {symbol} Analysis Results:")
        print(f"   Traditional Signal: {traditional.get('signal')} (Score: {traditional.get('score')})")
        print(f"   ML Prediction: {ml_signal.get('prediction')} (Confidence: {ml_signal.get('confidence', 0):.1f})")
        print(f"   Final Signal: {ml_signal.get('signal')} (Combined Score: {ml_signal.get('confidence', 0):.0f})")
        
        if 'buy_probability' in ml_signal:
            print(f"   Buy Probability: {ml_signal['buy_probability']:.1%}")
            print(f"   ML Strategy: {ml_signal.get('source', 'unknown')}")
    
    def _save_results(self, results: Dict):
        """Save ML-enhanced results"""
        # Create ML results directory
        os.makedirs("ml_models/analysis_results", exist_ok=True)
        
        # Save timestamped results
        filename = f"ml_models/analysis_results/ml_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Also save as latest
        with open("ml_models/analysis_results/ml_analysis_latest.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to {filename}")
    
    def _generate_trading_report(self):
        """Generate paper trading performance report"""
        print("\n" + "=" * 80)
        report = self.paper_account.generate_performance_report()
        
        # Display summary
        summary = report['account_summary']
        current_value = report['current_positions']['total_portfolio_value']
        total_return = current_value - summary['initial_balance']
        total_return_pct = (total_return / summary['initial_balance']) * 100
        
        print("📊 PAPER TRADING SUMMARY")
        print("-" * 40)
        print(f"Initial Balance: ${summary['initial_balance']:,.2f}")
        print(f"Current Value: ${current_value:,.2f}")
        print(f"Total Return: ${total_return:+,.2f} ({total_return_pct:+.1f}%)")
        print(f"Open Positions: {summary['positions_count']}")
        
        # Show current positions
        if report['current_positions']['positions']:
            print("\nCurrent Positions:")
            for symbol, pos in report['current_positions']['positions'].items():
                print(f"  {symbol}: {pos['shares']} @ ${pos['avg_price']:.2f} "
                      f"(P&L: ${pos['unrealized_pnl']:+.2f})")


def run_ml_enhanced_analysis(paper_trading=True):
    """Main function to run ML-enhanced analysis"""
    analyzer = MLEnhancedAnalyzer(paper_trading=paper_trading)
    return analyzer.analyze_with_ml()


if __name__ == "__main__":
    # Check command line arguments
    import argparse
    import sys
    
    # Add parent directory to path
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.validators import Validators, ValidationError
    
    parser = argparse.ArgumentParser(description='ML-Enhanced Trading Analysis')
    parser.add_argument('--live', action='store_true', help='Run in live mode (no paper trading)')
    parser.add_argument('--symbol', type=str, help='Analyze specific symbol')
    parser.add_argument('--train', action='store_true', help='Train ML model first')
    
    args = parser.parse_args()
    
    # Validate symbol if provided
    if args.symbol:
        try:
            args.symbol = Validators.validate_stock_symbol(args.symbol, allow_lowercase=True)
        except ValidationError as e:
            print(f"❌ Error: {e}")
            sys.exit(1)
    
    # Train model if requested
    if args.train:
        from ml_models.basic_ml_predictor import train_ml_model
        print("🧠 Training ML model...")
        train_ml_model(args.symbol or 'AAPL')
    
    # Run analysis
    analyzer = MLEnhancedAnalyzer(paper_trading=not args.live)
    analyzer.analyze_with_ml(symbol=args.symbol)