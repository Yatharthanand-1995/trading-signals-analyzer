#!/usr/bin/env python3
"""
System Health Check
Verifies all Phase 1 and Phase 2 components are working correctly
"""

import sys
import importlib
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def check_imports():
    """Check if all modules can be imported"""
    print("🔍 Checking module imports...")
    
    modules_to_check = [
        # Phase 1
        ('multi_timeframe_analyzer', 'MultiTimeframeAnalyzer'),
        ('volume_analyzer', 'VolumeAnalyzer'),
        ('entry_filter_system', 'EntryFilterSystem'),
        ('swing_trading_analyzer', 'SwingTradingAnalyzer'),
        
        # Phase 2
        ('dynamic_stop_loss_system', 'DynamicStopLossSystem'),
        ('advanced_position_sizing', 'AdvancedPositionSizing'),
        ('profit_taking_strategy', 'ProfitTakingStrategy'),
        ('integrated_risk_management', 'IntegratedRiskManagement'),
        
        # Base module
        ('enhanced_trading_analyzer', 'EnhancedTradingAnalyzer')
    ]
    
    results = {}
    for module_name, class_name in modules_to_check:
        try:
            module = importlib.import_module(module_name)
            if hasattr(module, class_name):
                results[module_name] = "✅ OK"
            else:
                results[module_name] = f"❌ Class {class_name} not found"
        except Exception as e:
            results[module_name] = f"❌ Import error: {str(e)}"
    
    return results

def test_phase1_components():
    """Test Phase 1 components individually"""
    print("\n📊 Testing Phase 1 Components...")
    
    # Get test data
    symbol = 'AAPL'
    stock = yf.Ticker(symbol)
    df = stock.history(period='3mo')
    
    if df.empty:
        return {"error": "Failed to fetch test data"}
    
    results = {}
    
    # Test 1: Multi-timeframe Analyzer
    try:
        from multi_timeframe_analyzer import MultiTimeframeAnalyzer
        analyzer = MultiTimeframeAnalyzer()
        analysis = analyzer.analyze_timeframes(df)
        results['multi_timeframe'] = f"✅ Alignment: {analysis['alignment_score']:.1f}%"
    except Exception as e:
        results['multi_timeframe'] = f"❌ Error: {str(e)}"
    
    # Test 2: Volume Analyzer
    try:
        from volume_analyzer import VolumeAnalyzer
        volume_analyzer = VolumeAnalyzer()
        volume_analysis = volume_analyzer.analyze_volume(df)
        results['volume'] = f"✅ Score: {volume_analysis['volume_score']}"
    except Exception as e:
        results['volume'] = f"❌ Error: {str(e)}"
    
    # Test 3: Entry Filter
    try:
        from entry_filter_system import EntryFilterSystem
        filter_system = EntryFilterSystem()
        filters = filter_system.apply_filters(symbol, df)
        results['entry_filter'] = f"✅ Score: {filters['total_score']:.0f}%, Passed: {filters['passed_filters']}"
    except Exception as e:
        results['entry_filter'] = f"❌ Error: {str(e)}"
    
    # Test 4: Swing Trading Analyzer
    try:
        from swing_trading_analyzer import SwingTradingAnalyzer
        swing_analyzer = SwingTradingAnalyzer()
        # Note: This requires market data fetch
        results['swing_analyzer'] = "✅ Module loaded (full test requires market data)"
    except Exception as e:
        results['swing_analyzer'] = f"❌ Error: {str(e)}"
    
    return results

def test_phase2_components():
    """Test Phase 2 components individually"""
    print("\n💰 Testing Phase 2 Components...")
    
    # Get test data
    symbol = 'AAPL'
    stock = yf.Ticker(symbol)
    df = stock.history(period='3mo')
    
    if df.empty:
        return {"error": "Failed to fetch test data"}
    
    results = {}
    entry_price = df['Close'].iloc[-1]
    
    # Test 1: Dynamic Stop Loss
    try:
        from dynamic_stop_loss_system import DynamicStopLossSystem
        stop_system = DynamicStopLossSystem()
        stop_data = stop_system.calculate_initial_stop(entry_price, df)
        results['stop_loss'] = f"✅ Stop: ${stop_data['initial_stop']:.2f} ({stop_data['stop_distance_pct']:.1f}%)"
    except Exception as e:
        results['stop_loss'] = f"❌ Error: {str(e)}"
    
    # Test 2: Position Sizing
    try:
        from advanced_position_sizing import AdvancedPositionSizing
        sizer = AdvancedPositionSizing(10000)
        sizing = sizer.calculate_atr_based_size(df, entry_price * 0.02)
        results['position_sizing'] = f"✅ Shares: {sizing['shares']}, Value: ${sizing['position_value']:.2f}"
    except Exception as e:
        results['position_sizing'] = f"❌ Error: {str(e)}"
    
    # Test 3: Profit Taking
    try:
        from profit_taking_strategy import ProfitTakingStrategy
        profit_strategy = ProfitTakingStrategy()
        targets = profit_strategy.calculate_atr_targets(entry_price, entry_price * 0.02)
        results['profit_taking'] = f"✅ Targets: {len(targets)} levels set"
    except Exception as e:
        results['profit_taking'] = f"❌ Error: {str(e)}"
    
    # Test 4: Integrated Risk Management
    try:
        from integrated_risk_management import IntegratedRiskManagement
        risk_manager = IntegratedRiskManagement(10000)
        results['risk_management'] = "✅ System initialized"
    except Exception as e:
        results['risk_management'] = f"❌ Error: {str(e)}"
    
    return results

def test_integration():
    """Test complete integration of Phase 1 and Phase 2"""
    print("\n🔗 Testing Complete Integration...")
    
    try:
        from swing_trading_analyzer import SwingTradingAnalyzer
        from integrated_risk_management import IntegratedRiskManagement
        
        # Test symbol
        symbol = 'NVDA'
        
        # Initialize components
        swing_analyzer = SwingTradingAnalyzer()
        risk_manager = IntegratedRiskManagement(10000)
        
        # Get data
        stock = yf.Ticker(symbol)
        df = stock.history(period='3mo')
        
        if df.empty:
            return {"error": "Failed to fetch test data"}
        
        # Phase 1: Analysis
        print(f"  Analyzing {symbol}...")
        swing_signals = swing_analyzer.analyze_stock_complete(symbol)
        
        # Check if we have a valid signal
        if 'signal' not in swing_signals:
            return {"error": "Swing analysis failed"}
        
        # Phase 2: Risk Management (only if buy signal)
        if swing_signals['signal'] in ['BUY', 'STRONG_BUY']:
            analysis_data = {
                'symbol': symbol,
                'signal': swing_signals['signal'],
                'score': swing_signals.get('total_score', 70),
                'entry_quality': 'good',
                'sector': 'Technology'
            }
            
            trade_setup = risk_manager.calculate_complete_trade_setup(
                analysis_data, df, []
            )
            
            if trade_setup['action'] == 'ENTER_TRADE':
                return {
                    'status': '✅ Full integration working',
                    'signal': swing_signals['signal'],
                    'position_size': trade_setup['entry']['shares'],
                    'stop_loss': f"${trade_setup['stop_loss']['initial_stop']:.2f}",
                    'risk': f"{trade_setup['risk']['risk_pct']:.1%}",
                    'quality': trade_setup['trade_quality']['grade']
                }
            else:
                return {
                    'status': '⚠️ Trade rejected by risk management',
                    'reason': trade_setup.get('reason', 'Unknown')
                }
        else:
            return {
                'status': '⚠️ No buy signal generated',
                'signal': swing_signals['signal']
            }
            
    except Exception as e:
        return {"error": f"Integration test failed: {str(e)}"}

def check_configuration():
    """Check configuration files"""
    print("\n⚙️ Checking Configuration...")
    
    import os
    import json
    
    results = {}
    
    # Check stocks config
    config_path = '../config/stocks_config.json'
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
            results['stocks_config'] = f"✅ Active list: {config['active_list']} ({len(config['stock_lists'][config['active_list']])} stocks)"
    else:
        results['stocks_config'] = "❌ Config file not found"
    
    # Check run script
    script_path = '../run_analysis.sh'
    if os.path.exists(script_path):
        results['run_script'] = "✅ Automation script present"
    else:
        results['run_script'] = "❌ Run script not found"
    
    return results

def main():
    """Run complete system health check"""
    print("="*60)
    print("🏥 TRADING SYSTEM HEALTH CHECK")
    print("="*60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # 1. Check imports
    import_results = check_imports()
    print("\n📦 Module Import Status:")
    for module, status in import_results.items():
        print(f"  {module}: {status}")
    
    # 2. Test Phase 1
    phase1_results = test_phase1_components()
    print("\n🔍 Phase 1 Component Tests:")
    for component, status in phase1_results.items():
        print(f"  {component}: {status}")
    
    # 3. Test Phase 2
    phase2_results = test_phase2_components()
    print("\n💰 Phase 2 Component Tests:")
    for component, status in phase2_results.items():
        print(f"  {component}: {status}")
    
    # 4. Test Integration
    integration_result = test_integration()
    print("\n🔗 Integration Test:")
    if isinstance(integration_result, dict):
        for key, value in integration_result.items():
            print(f"  {key}: {value}")
    
    # 5. Check Configuration
    config_results = check_configuration()
    print("\n⚙️ Configuration Status:")
    for item, status in config_results.items():
        print(f"  {item}: {status}")
    
    # Summary
    print("\n" + "="*60)
    all_good = True
    
    # Check for any failures
    for results in [import_results, phase1_results, phase2_results]:
        for status in results.values():
            if "❌" in str(status):
                all_good = False
                break
    
    if all_good:
        print("✅ SYSTEM STATUS: ALL COMPONENTS OPERATIONAL")
        print("🚀 Ready for swing trading with 2-15 day holding periods!")
    else:
        print("⚠️ SYSTEM STATUS: SOME ISSUES DETECTED")
        print("Please review the errors above and fix them before proceeding.")
    
    print("="*60)

if __name__ == "__main__":
    main()