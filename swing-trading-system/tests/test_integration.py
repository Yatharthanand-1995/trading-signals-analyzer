#!/usr/bin/env python3
"""
Test Phase 2 Integration
Demonstrates complete swing trading system with risk management
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# Import all components
from core.indicators.enhanced_analyzer import EnhancedTradingAnalyzer
from core.analysis.swing_analyzer import SwingTradingAnalyzer
from core.risk_management.risk_manager import IntegratedRiskManagement

def test_complete_system():
    """Test the complete Phase 1 + Phase 2 system"""
    
    # Configuration
    symbols = ['AAPL', 'NVDA', 'TSLA', 'MSFT', 'GOOGL']
    account_size = 10000
    
    print(f"\n{'='*80}")
    print(f"🚀 COMPLETE SWING TRADING SYSTEM TEST")
    print(f"Phase 1: Multi-timeframe Analysis + Volume + Entry Filters")
    print(f"Phase 2: Dynamic Stops + Position Sizing + Profit Taking")
    print(f"{'='*80}")
    print(f"Account Size: ${account_size:,}")
    print(f"Testing with: {', '.join(symbols)}")
    print(f"{'='*80}\n")
    
    # Initialize components
    swing_analyzer = SwingTradingAnalyzer()
    risk_manager = IntegratedRiskManagement(account_size)
    
    # Track portfolio
    portfolio_trades = []
    current_positions = []
    
    for symbol in symbols:
        print(f"\n{'='*60}")
        print(f"📊 ANALYZING {symbol}")
        print(f"{'='*60}")
        
        try:
            # Fetch data
            stock = yf.Ticker(symbol)
            df = stock.history(period='6mo')
            
            if df.empty:
                print(f"❌ No data available for {symbol}")
                continue
            
            # Get stock info
            info = stock.info
            sector = info.get('sector', 'Unknown')
            
            # Phase 1: Swing Trading Analysis
            print("\n📈 PHASE 1: Swing Trading Analysis")
            print("-" * 40)
            swing_signals = swing_analyzer.analyze_stock_complete(symbol)
            
            print(f"Signal: {swing_signals['signal']}")
            print(f"Total Score: {swing_signals['total_score']}/100")
            print(f"Breakdown:")
            for component, score in swing_signals['score_breakdown'].items():
                print(f"  • {component}: {score['score']}/{score['max']}")
            
            # Check if we should take the trade
            if swing_signals['signal'] in ['BUY', 'STRONG_BUY'] and swing_signals['total_score'] >= 60:
                # Phase 2: Risk Management
                print("\n💰 PHASE 2: Risk Management")
                print("-" * 40)
                
                # Prepare analysis data
                analysis_data = {
                    'symbol': symbol,
                    'signal': swing_signals['signal'],
                    'score': swing_signals['total_score'],
                    'entry_quality': 'excellent' if swing_signals['total_score'] >= 80 else 
                                   'good' if swing_signals['total_score'] >= 70 else 'standard',
                    'sector': sector
                }
                
                # Calculate complete trade setup
                trade_setup = risk_manager.calculate_complete_trade_setup(
                    analysis_data, df, current_positions
                )
                
                # Print risk management report
                print(risk_manager.generate_risk_report(trade_setup))
                
                # Add to portfolio if approved
                if trade_setup['action'] == 'ENTER_TRADE':
                    portfolio_trades.append(trade_setup)
                    current_positions.append({
                        'symbol': symbol,
                        'position_pct': trade_setup['entry']['position_pct'],
                        'risk_pct': trade_setup['risk']['risk_pct'],
                        'sector': sector
                    })
            else:
                print(f"\n❌ Trade Rejected: Signal={swing_signals['signal']}, Score={swing_signals['total_score']}")
                
        except Exception as e:
            print(f"\n❌ Error analyzing {symbol}: {str(e)}")
            continue
    
    # Portfolio Summary
    print(f"\n{'='*80}")
    print("📊 PORTFOLIO SUMMARY")
    print(f"{'='*80}")
    
    if portfolio_trades:
        print(f"\nTotal Approved Trades: {len(portfolio_trades)}")
        
        # Calculate portfolio metrics
        total_value = sum(t['entry']['position_value'] for t in portfolio_trades)
        total_risk = sum(t['risk']['risk_pct'] for t in portfolio_trades)
        
        print(f"Total Position Value: ${total_value:,.2f} ({total_value/account_size:.1%} of account)")
        print(f"Total Portfolio Risk: {total_risk:.1%} (Max: 6.0%)")
        
        print("\nPositions:")
        for trade in portfolio_trades:
            print(f"  • {trade['symbol']}: {trade['entry']['shares']} shares @ ${trade['entry']['price']:.2f}")
            print(f"    Stop: ${trade['stop_loss']['initial_stop']:.2f} | Risk: {trade['risk']['risk_pct']:.1%}")
            print(f"    Target 1: ${trade['profit_targets']['primary'][0]['price']:.2f} (+{trade['profit_targets']['primary'][0]['distance_pct']:.1%})")
            print(f"    Quality: {trade['trade_quality']['grade']} | R/R: {trade['risk_reward']['average_rr']:.1f}:1")
    else:
        print("\n❌ No trades met the criteria for entry")
    
    print(f"\n{'='*80}")
    print("✅ COMPLETE SYSTEM TEST FINISHED")
    print(f"{'='*80}")


if __name__ == "__main__":
    test_complete_system()