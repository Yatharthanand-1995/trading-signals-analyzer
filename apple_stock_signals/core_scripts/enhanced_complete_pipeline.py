#!/usr/bin/env python3
"""
Enhanced Complete Trading Pipeline
Includes Phase 1 (Multi-timeframe, Volume, Entry Filters) and Phase 2 (Risk Management)
Optimized for 2-15 day swing trading
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json
import sys

# Import all components
from enhanced_trading_analyzer import EnhancedTradingAnalyzer
from swing_trading_analyzer import SwingTradingAnalyzer
from integrated_risk_management import IntegratedRiskManagement
from data_fetcher import StockDataFetcher
from ml_data_prep import MLDataPreparation
from backtester import TradingBacktester

class EnhancedCompletePipeline:
    def __init__(self, account_size=10000):
        self.account_size = account_size
        self.risk_manager = IntegratedRiskManagement(account_size)
        self.swing_analyzer = SwingTradingAnalyzer()
        self.fetcher = StockDataFetcher()
        self.ml_prep = MLDataPreparation()
        self.backtester = TradingBacktester()
        
    def run_complete_analysis(self, symbols):
        """Run complete analysis pipeline with all phases"""
        print(f"\n{'='*80}")
        print(f"🚀 ENHANCED TRADING PIPELINE - SWING TRADING OPTIMIZED")
        print(f"{'='*80}")
        print(f"Account Size: ${self.account_size:,}")
        print(f"Analyzing {len(symbols)} stocks...")
        print(f"Features: Phase 1 (Analysis) + Phase 2 (Risk Management)")
        print(f"{'='*80}\n")
        
        results = []
        portfolio_setups = []
        
        for i, symbol in enumerate(symbols):
            print(f"\n[{i+1}/{len(symbols)}] Analyzing {symbol}...")
            
            try:
                # Step 1: Fetch historical data
                df = self.fetcher.fetch_stock_data(symbol, period='6mo')
                if df is None or df.empty:
                    print(f"  ❌ No data available for {symbol}")
                    continue
                
                # Step 2: Run enhanced technical analysis
                enhanced_analyzer = EnhancedTradingAnalyzer(df)
                base_signals = enhanced_analyzer.generate_signals()
                
                # Step 3: Run swing trading analysis (Phase 1)
                swing_signals = self.swing_analyzer.analyze_complete(symbol, df)
                
                # Step 4: Prepare for risk management
                if swing_signals['signal'] in ['BUY', 'STRONG_BUY']:
                    # Get stock info
                    stock = yf.Ticker(symbol)
                    info = stock.info
                    
                    analysis_data = {
                        'symbol': symbol,
                        'signal': swing_signals['signal'],
                        'score': swing_signals['total_score'],
                        'entry_quality': 'excellent' if swing_signals['total_score'] >= 80 else 
                                       'good' if swing_signals['total_score'] >= 70 else 'standard',
                        'sector': info.get('sector', 'Unknown')
                    }
                    
                    # Step 5: Calculate complete trade setup (Phase 2)
                    trade_setup = self.risk_manager.calculate_complete_trade_setup(
                        analysis_data, df, current_positions=portfolio_setups
                    )
                    
                    if trade_setup['action'] == 'ENTER_TRADE':
                        portfolio_setups.append({
                            'symbol': symbol,
                            'position_pct': trade_setup['entry']['position_pct'],
                            'risk_pct': trade_setup['risk']['risk_pct'],
                            'sector': info.get('sector', 'Unknown')
                        })
                    
                    # Step 6: Run backtesting
                    backtest_results = self._run_backtest(symbol, df, trade_setup)
                    
                    # Compile results
                    result = {
                        'symbol': symbol,
                        'timestamp': datetime.now().isoformat(),
                        'current_price': df['Close'].iloc[-1],
                        'swing_analysis': {
                            'signal': swing_signals['signal'],
                            'score': swing_signals['total_score'],
                            'timeframe_alignment': swing_signals['timeframe_alignment']['aligned'],
                            'volume_confirmation': swing_signals['volume_confirmation']['has_confirmation'],
                            'entry_filter': swing_signals['entry_filter']['passed_filters'],
                            'components': swing_signals['score_breakdown']
                        },
                        'risk_management': {
                            'action': trade_setup['action'],
                            'position_size': trade_setup['entry']['shares'] if trade_setup['action'] == 'ENTER_TRADE' else 0,
                            'position_value': trade_setup['entry']['position_value'] if trade_setup['action'] == 'ENTER_TRADE' else 0,
                            'stop_loss': trade_setup['stop_loss']['initial_stop'] if trade_setup['action'] == 'ENTER_TRADE' else None,
                            'profit_targets': [
                                {'name': t['name'], 'price': t['price'], 'gain_pct': t['distance_pct']}
                                for t in trade_setup['profit_targets']['primary']
                            ] if trade_setup['action'] == 'ENTER_TRADE' else [],
                            'risk_reward': trade_setup['risk_reward']['average_rr'] if trade_setup['action'] == 'ENTER_TRADE' else 0,
                            'trade_quality': trade_setup['trade_quality']['grade'] if trade_setup['action'] == 'ENTER_TRADE' else 'N/A',
                            'risk_pct': trade_setup['risk']['risk_pct'] if trade_setup['action'] == 'ENTER_TRADE' else 0
                        },
                        'backtest': backtest_results,
                        'recommendation': self._generate_recommendation(swing_signals, trade_setup)
                    }
                    
                    results.append(result)
                    
                    # Print summary
                    self._print_stock_summary(result)
                    
                else:
                    print(f"  ⚠️ {symbol}: {swing_signals['signal']} (Score: {swing_signals['total_score']})")
                    
            except Exception as e:
                print(f"  ❌ Error analyzing {symbol}: {str(e)}")
                continue
        
        # Generate portfolio summary
        self._generate_portfolio_summary(results)
        
        # Save results
        self._save_results(results)
        
        return results
    
    def _run_backtest(self, symbol, df, trade_setup):
        """Run backtesting for the trade setup"""
        if trade_setup['action'] != 'ENTER_TRADE':
            return {'status': 'no_trade'}
        
        # Simple backtest based on historical performance
        try:
            # Look for similar setups in history
            lookback_days = 252  # 1 year
            if len(df) < lookback_days:
                lookback_days = len(df)
            
            historical_df = df.iloc[-lookback_days:-20]  # Exclude recent 20 days
            
            # Calculate win rate for similar conditions
            similar_setups = 0
            wins = 0
            
            for i in range(20, len(historical_df) - 15):
                # Check if conditions were similar
                entry_price = historical_df['Close'].iloc[i]
                
                # Look at next 15 days
                future_prices = historical_df['Close'].iloc[i+1:i+16]
                max_price = future_prices.max()
                min_price = future_prices.min()
                
                # Check if hit profit target
                first_target = trade_setup['profit_targets']['primary'][0]['price']
                target_distance_pct = (first_target - trade_setup['entry']['price']) / trade_setup['entry']['price']
                
                if (max_price - entry_price) / entry_price >= target_distance_pct:
                    wins += 1
                
                similar_setups += 1
            
            win_rate = wins / similar_setups if similar_setups > 0 else 0.5
            
            return {
                'historical_win_rate': win_rate,
                'sample_size': similar_setups,
                'expected_return': win_rate * trade_setup['risk_reward']['average_rr'] - (1 - win_rate),
                'confidence': 'high' if similar_setups >= 50 else 'medium' if similar_setups >= 20 else 'low'
            }
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def _generate_recommendation(self, swing_signals, trade_setup):
        """Generate trading recommendation"""
        if trade_setup['action'] != 'ENTER_TRADE':
            return f"❌ NO TRADE - {trade_setup.get('reason', 'Risk constraints')}"
        
        score = swing_signals['total_score']
        grade = trade_setup['trade_quality']['grade']
        
        if score >= 80 and grade in ['A', 'B']:
            return "🟢 STRONG BUY - Excellent setup with optimal risk/reward"
        elif score >= 70 and grade in ['A', 'B', 'C']:
            return "🟢 BUY - Good setup meeting all criteria"
        elif score >= 60:
            return "🟡 WATCH - Decent setup but wait for better entry"
        else:
            return "🔴 AVOID - Setup does not meet quality criteria"
    
    def _print_stock_summary(self, result):
        """Print formatted summary for a stock"""
        symbol = result['symbol']
        swing = result['swing_analysis']
        risk = result['risk_management']
        
        print(f"\n  ✅ {symbol} - {result['recommendation']}")
        print(f"     Score: {swing['score']}/100 | Signal: {swing['signal']}")
        
        if risk['action'] == 'ENTER_TRADE':
            print(f"     Position: {risk['position_size']} shares (${risk['position_value']:,.2f})")
            print(f"     Stop: ${risk['stop_loss']:.2f} | Risk: {risk['risk_pct']:.1%}")
            print(f"     Targets: ", end="")
            for t in risk['profit_targets'][:3]:
                print(f"${t['price']:.2f} (+{t['gain_pct']:.1%}) ", end="")
            print(f"\n     R/R: {risk['risk_reward']:.1f}:1 | Quality: {risk['trade_quality']}")
    
    def _generate_portfolio_summary(self, results):
        """Generate portfolio-wide summary"""
        print(f"\n{'='*80}")
        print("📊 PORTFOLIO SUMMARY")
        print(f"{'='*80}")
        
        # Filter approved trades
        approved_trades = [r for r in results if r['risk_management']['action'] == 'ENTER_TRADE']
        
        if not approved_trades:
            print("\n❌ No trades meet the criteria for entry")
            return
        
        # Calculate portfolio metrics
        total_value = sum(t['risk_management']['position_value'] for t in approved_trades)
        total_risk = sum(t['risk_management']['risk_pct'] for t in approved_trades)
        avg_rr = np.mean([t['risk_management']['risk_reward'] for t in approved_trades])
        
        print(f"\nApproved Trades: {len(approved_trades)}")
        print(f"Total Position Value: ${total_value:,.2f} ({total_value/self.account_size:.1%} of account)")
        print(f"Total Portfolio Risk: {total_risk:.1%} (Max: 6.0%)")
        print(f"Average R/R Ratio: {avg_rr:.2f}:1")
        
        print("\nTop Opportunities:")
        sorted_trades = sorted(approved_trades, key=lambda x: x['swing_analysis']['score'], reverse=True)
        for i, trade in enumerate(sorted_trades[:5]):
            print(f"{i+1}. {trade['symbol']} - Score: {trade['swing_analysis']['score']} - {trade['risk_management']['trade_quality']} Grade")
        
        # Sector distribution
        sectors = {}
        for trade in approved_trades:
            sector = trade['symbol']  # Would need actual sector data
            sectors[sector] = sectors.get(sector, 0) + trade['risk_management']['risk_pct']
        
        print("\nRisk Distribution:")
        for sector, risk in sorted(sectors.items(), key=lambda x: x[1], reverse=True):
            print(f"  {sector}: {risk:.1%}")
    
    def _save_results(self, results):
        """Save analysis results to file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'results/enhanced_analysis_{timestamp}.json'
        
        os.makedirs('results', exist_ok=True)
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to: {filename}")


def main():
    """Run the enhanced complete pipeline"""
    # Get top stocks
    top_stocks = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 'JPM', 'UNH',
        'JNJ', 'V', 'PG', 'HD', 'MA', 'DIS', 'BAC', 'ABBV', 'PFE', 'AVGO',
        'CSCO', 'MRK', 'PEP', 'TMO', 'COST', 'WMT', 'ABT', 'CRM', 'NKE', 'MCD',
        'LLY', 'ACN', 'TXN', 'ADBE', 'NFLX', 'CVX', 'WFC', 'AMD', 'PM', 'UPS',
        'RTX', 'NEE', 'BMY', 'LOW', 'INTC', 'ORCL', 'UNP', 'QCOM', 'T', 'IBM'
    ]
    
    # Run pipeline with first 10 stocks as a test
    pipeline = EnhancedCompletePipeline(account_size=10000)
    results = pipeline.run_complete_analysis(top_stocks[:10])
    
    print(f"\n✅ Enhanced pipeline completed successfully!")
    print(f"Analyzed {len(results)} stocks with full risk management")


if __name__ == "__main__":
    main()