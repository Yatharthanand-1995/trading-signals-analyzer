#!/usr/bin/env python3
"""
Test script for complete risk management system
Demonstrates all Phase 2 features
"""

import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd

# Import all risk management components
from dynamic_stop_loss_system import DynamicStopLossSystem
from advanced_position_sizing import AdvancedPositionSizing
from profit_taking_strategy import ProfitTakingStrategy
from integrated_risk_management import IntegratedRiskManagement

def test_complete_system():
    """Test all risk management components"""
    
    # Test stocks
    test_symbols = ['AAPL', 'NVDA', 'TSLA']
    
    # Initialize system with $10,000 account
    risk_manager = IntegratedRiskManagement(account_size=10000)
    
    print("🚀 COMPLETE RISK MANAGEMENT SYSTEM TEST")
    print("=" * 70)
    print(f"Account Size: ${risk_manager.account_size:,}")
    print(f"Risk Settings:")
    print(f"  • Max Risk Per Trade: 2%")
    print(f"  • Max Portfolio Risk: 6%")
    print(f"  • Max Positions: 5")
    print("=" * 70)
    
    # Track all setups
    all_setups = []
    
    for symbol in test_symbols:
        print(f"\n{'='*70}")
        print(f"📊 ANALYZING {symbol}")
        print(f"{'='*70}")
        
        # Fetch data
        stock = yf.Ticker(symbol)
        df = stock.history(period='3mo')
        
        if df.empty:
            print(f"❌ No data for {symbol}")
            continue
        
        # Simulate different analysis scores
        if symbol == 'AAPL':
            analysis_data = {
                'symbol': symbol,
                'signal': 'BUY',
                'score': 75,
                'entry_quality': 'good',
                'sector': 'Technology'
            }
        elif symbol == 'NVDA':
            analysis_data = {
                'symbol': symbol,
                'signal': 'STRONG_BUY',
                'score': 85,
                'entry_quality': 'excellent',
                'sector': 'Technology'
            }
        else:
            analysis_data = {
                'symbol': symbol,
                'signal': 'BUY',
                'score': 70,
                'entry_quality': 'standard',
                'sector': 'Automotive'
            }
        
        # Calculate complete trade setup
        current_positions = [setup for setup in all_setups if setup['action'] == 'ENTER_TRADE']
        trade_setup = risk_manager.calculate_complete_trade_setup(
            analysis_data, df, current_positions
        )
        
        all_setups.append(trade_setup)
        
        # Display results
        if trade_setup['action'] == 'ENTER_TRADE':
            print(f"\n✅ TRADE APPROVED: {symbol}")
            print(f"\nEntry Setup:")
            print(f"  Price: ${trade_setup['entry']['price']:.2f}")
            print(f"  Shares: {trade_setup['entry']['shares']}")
            print(f"  Value: ${trade_setup['entry']['position_value']:,.2f}")
            
            print(f"\nRisk Management:")
            print(f"  Stop Loss: ${trade_setup['stop_loss']['initial_stop']:.2f} ({trade_setup['stop_loss']['stop_distance_pct']:.1f}%)")
            print(f"  Risk: ${trade_setup['risk']['risk_amount']:.2f} ({trade_setup['risk']['risk_pct']:.1f}%)")
            
            print(f"\nProfit Targets:")
            for target in trade_setup['profit_targets']['primary']:
                print(f"  {target['name']}: ${target['price']:.2f} (+{target['distance_pct']:.1f}%)")
            
            print(f"\nTrade Quality: {trade_setup['trade_quality']['grade']}")
        else:
            print(f"\n❌ TRADE REJECTED: {symbol}")
            print(f"Reason: {trade_setup['reason']}")
    
    # Portfolio summary
    print(f"\n{'='*70}")
    print("📊 PORTFOLIO SUMMARY")
    print(f"{'='*70}")
    
    approved_trades = [s for s in all_setups if s['action'] == 'ENTER_TRADE']
    
    if approved_trades:
        total_value = sum(t['entry']['position_value'] for t in approved_trades)
        total_risk = sum(t['risk']['risk_amount'] for t in approved_trades)
        total_risk_pct = sum(t['risk']['risk_pct'] for t in approved_trades)
        
        print(f"\nApproved Trades: {len(approved_trades)}")
        print(f"Total Position Value: ${total_value:,.2f}")
        print(f"Total Risk: ${total_risk:.2f} ({total_risk_pct:.1f}% of account)")
        print(f"Portfolio Heat: {total_risk_pct:.1f}% / 6.0% limit")
        
        print(f"\nPositions:")
        for trade in approved_trades:
            print(f"  • {trade['symbol']}: {trade['entry']['shares']} shares @ ${trade['entry']['price']:.2f}")
    
    # Test position monitoring
    if approved_trades:
        print(f"\n{'='*70}")
        print("📊 POSITION MONITORING SIMULATION")
        print(f"{'='*70}")
        
        # Simulate 5 days later
        first_trade = approved_trades[0]
        symbol = first_trade['symbol']
        
        print(f"\n🔍 Monitoring {symbol} after 5 days...")
        
        # Create position data with current_price and stop_loss added
        position = {
            'symbol': symbol,
            'entry_price': first_trade['entry']['price'],
            'current_stop': first_trade['stop_loss']['initial_stop'],
            'initial_stop': first_trade['stop_loss']['initial_stop'],
            'stop_loss': first_trade['stop_loss']['initial_stop'],  # Add stop_loss field
            'shares': first_trade['entry']['shares'],
            'position_type': first_trade['position_type'],
            'entry_date': datetime.now() - timedelta(days=5),
            'profit_targets': first_trade['profit_targets'],
            'current_price': df['Close'].iloc[-1]  # Add current price
        }
        
        # Get fresh data
        stock = yf.Ticker(symbol)
        current_df = stock.history(period='1mo')
        
        if not current_df.empty:
            # Monitor position
            management = risk_manager.monitor_existing_position(position, current_df)
            
            print(f"\nPosition Status:")
            print(f"  Health: {management['position_health']['status'].upper()}")
            print(f"  P&L: ${management['position_health']['unrealized_pnl']:.2f} ({management['position_health']['pnl_pct']:.1f}%)")
            print(f"  R-Multiple: {management['position_health']['r_multiple']:.2f}R")
            print(f"  Days Held: {management['position_health']['days_held']}")
            
            if management['stop_management']['should_update']:
                print(f"\n⚠️ Stop Update Recommended:")
                print(f"  Current Stop: ${position['current_stop']:.2f}")
                print(f"  New Stop: ${management['stop_management']['recommended_stop']:.2f}")
                print(f"  Reason: {management['stop_management']['update_reason']}")
            
            if management['recommended_actions']:
                print(f"\n🎯 Recommended Actions:")
                for action in management['recommended_actions']:
                    print(f"  • {action['type']}: {action['reason']}")

def demonstrate_individual_components():
    """Demonstrate each risk management component"""
    print(f"\n{'='*70}")
    print("🔧 INDIVIDUAL COMPONENT DEMONSTRATIONS")
    print(f"{'='*70}")
    
    # Get sample data
    stock = yf.Ticker('AAPL')
    df = stock.history(period='3mo')
    
    if not df.empty:
        current_price = df['Close'].iloc[-1]
        entry_price = df['Close'].iloc[-10]
        
        # 1. Dynamic Stop Loss
        print("\n1️⃣ DYNAMIC STOP LOSS SYSTEM")
        print("-" * 40)
        stop_system = DynamicStopLossSystem()
        
        initial_stop = stop_system.calculate_initial_stop(entry_price, df)
        print(f"Entry: ${entry_price:.2f}")
        print(f"Initial Stop: ${initial_stop['initial_stop']:.2f} ({initial_stop['stop_type']})")
        print(f"Stop Distance: {initial_stop['stop_distance_pct']:.1f}%")
        print(f"Volatility: {initial_stop['volatility_regime']}")
        
        # 2. Position Sizing
        print("\n2️⃣ ADVANCED POSITION SIZING")
        print("-" * 40)
        sizer = AdvancedPositionSizing(10000)
        
        sizing = sizer.calculate_optimal_position_size(
            {'symbol': 'AAPL', 'price_data': df, 'sector': 'Tech'},
            {'stop_distance': initial_stop['stop_distance']},
            {'win_rate': 0.55, 'avg_win': 2.5, 'avg_loss': -1.0},
            []
        )
        
        print(f"Recommended Shares: {sizing['recommended_shares']}")
        print(f"Position Value: ${sizing['position_value']:,.2f}")
        print(f"Position Size: {sizing['position_pct']:.1%} of account")
        print(f"Risk: ${sizing['actual_risk_amount']:.2f} ({sizing['actual_risk_pct']:.1%})")
        
        # 3. Profit Taking
        print("\n3️⃣ PROFIT TAKING STRATEGY")
        print("-" * 40)
        profit_strategy = ProfitTakingStrategy()
        
        targets = profit_strategy.calculate_optimal_targets(
            {'entry_price': entry_price, 'stop_loss': initial_stop['initial_stop'], 'position_size': 100},
            df
        )
        
        print("Scale-Out Targets:")
        for target in targets['primary']:
            print(f"  {target['name']}: ${target['price']:.2f} (+{target['distance_pct']:.1f}%) - Exit {target['exit_pct']:.0%}")

if __name__ == "__main__":
    # Run complete system test
    test_complete_system()
    
    # Demonstrate individual components
    demonstrate_individual_components()
    
    print(f"\n{'='*70}")
    print("✅ RISK MANAGEMENT TEST COMPLETE")
    print(f"{'='*70}")