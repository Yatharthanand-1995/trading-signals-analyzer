#!/usr/bin/env python3
"""
Test script for swing trading analyzer with sample data
"""

import yfinance as yf
from datetime import datetime
import pandas as pd

# Import our analyzers directly
from multi_timeframe_analyzer import MultiTimeframeAnalyzer
from volume_analyzer import VolumeAnalyzer
from entry_filter_system import EntryFilterSystem

def test_complete_analysis():
    """Test the complete analysis pipeline"""
    
    # Test stocks
    test_symbols = ['AAPL', 'NVDA', 'TSLA']
    
    # Initialize analyzers
    mtf = MultiTimeframeAnalyzer()
    vol = VolumeAnalyzer()
    filters = EntryFilterSystem()
    
    # Get market data
    spy = yf.Ticker('SPY')
    market_df = spy.history(period='6mo')
    
    for symbol in test_symbols:
        print(f"\n{'='*70}")
        print(f"🎯 COMPLETE ANALYSIS FOR {symbol}")
        print(f"{'='*70}")
        
        # Fetch data
        stock = yf.Ticker(symbol)
        df = stock.history(period='6mo')
        
        if df.empty:
            print(f"❌ No data for {symbol}")
            continue
        
        # 1. Entry Filters
        print("\n📋 ENTRY FILTERS:")
        filter_results = filters.apply_filters(symbol, df, market_df, stock.info)
        print(f"Filter Score: {filter_results['overall']['filter_percentage']:.1f}%")
        print(f"Recommendation: {filter_results['overall']['recommendation']}")
        
        # Show individual filter results
        print(f"\nFilter Breakdown:")
        print(f"  Liquidity: {'✅' if filter_results['liquidity']['passes_filter'] else '❌'}")
        print(f"  Trend: {'✅' if filter_results['trend']['passes_filter'] else '❌'} ({filter_results['trend']['trend_strength']})")
        print(f"  Relative Strength: {'✅' if filter_results['relative_strength']['passes_filter'] else '❌'} (RS: {filter_results['relative_strength']['rs_vs_market']:.2f})")
        print(f"  Volatility: {'✅' if filter_results['volatility']['passes_filter'] else '❌'} (ATR: {filter_results['volatility']['atr_pct']:.1f}%)")
        print(f"  Momentum: {'✅' if filter_results['momentum']['passes_filter'] else '❌'} (RSI: {filter_results['momentum']['rsi']:.1f})")
        
        if not filter_results['passes_filters']:
            print(f"\n❌ {symbol} FAILS FILTERS - Skip this trade")
            continue
        
        # 2. Multi-timeframe Analysis
        print("\n📈 MULTI-TIMEFRAME ANALYSIS:")
        mtf_results = mtf.analyze_stock(symbol)
        
        if mtf_results and 'alignment' in mtf_results:
            align = mtf_results['alignment']
            print(f"Timeframe Alignment: {align['alignment_percentage']:.1f}%")
            print(f"Recommendation: {align['recommendation']}")
            
            # Show timeframe details
            if 'weekly' in mtf_results:
                weekly = mtf_results['weekly']
                print(f"\nWeekly: {weekly['trend']['trend_direction']} ({weekly['trend']['trend_strength']})")
                
            if 'daily' in mtf_results:
                daily = mtf_results['daily']
                print(f"Daily: {daily['trend']['trend_direction']} ({daily['trend']['trend_strength']})")
                
            if '4hour' in mtf_results:
                hourly = mtf_results['4hour']
                print(f"4-Hour: RSI {hourly['indicators']['rsi']:.1f}")
                
            if 'entry_timing' in mtf_results:
                entry = mtf_results['entry_timing']
                print(f"\nEntry Quality: {entry['entry_quality']}")
        
        # 3. Volume Analysis
        print("\n📊 VOLUME ANALYSIS:")
        vol_results = vol.analyze_volume(df)
        
        if vol_results:
            print(f"Volume Score: {vol_results['volume_score']}")
            print(f"Volume Signal: {vol_results['volume_signal']}")
            
            # Key volume metrics
            print(f"\nVolume Metrics:")
            print(f"  Relative Volume: {vol_results['relative_volume']['relative_volume']:.2f}x average")
            print(f"  OBV Signal: {vol_results['obv']['obv_signal']}")
            print(f"  VWAP Position: {vol_results['vwap']['vwap_signal']}")
            print(f"  Money Flow Index: {vol_results['mfi']['mfi']:.1f}")
            
            if vol_results['patterns']:
                print(f"  Patterns: {', '.join(vol_results['patterns'])}")
        
        # 4. Calculate final score
        print("\n🎯 FINAL ANALYSIS:")
        
        # Simple scoring
        total_score = 0
        
        # Filter score (30%)
        filter_score = filter_results['overall']['filter_percentage']
        total_score += (filter_score / 100) * 30
        
        # Timeframe alignment (30%)
        if mtf_results and 'alignment' in mtf_results:
            alignment_score = mtf_results['alignment']['alignment_percentage']
            total_score += (alignment_score / 100) * 30
        
        # Volume score (20%)
        if vol_results:
            volume_score = vol_results['volume_score']
            total_score += (volume_score / 100) * 20
        
        # Technical score (20%) - simplified
        technical_score = 50  # Base
        if filter_results['momentum']['rsi'] < 70:
            technical_score += 10
        if filter_results['trend']['trend_strength'] == 'strong':
            technical_score += 10
        total_score += (technical_score / 100) * 20
        
        print(f"Total Score: {total_score:.1f}/100")
        
        # Generate signal
        if total_score >= 75:
            signal = "STRONG BUY"
        elif total_score >= 65:
            signal = "BUY"
        elif total_score >= 50:
            signal = "HOLD"
        else:
            signal = "AVOID"
            
        print(f"SIGNAL: {signal}")
        
        # Position sizing example
        if signal in ["STRONG BUY", "BUY"]:
            current_price = df['Close'].iloc[-1]
            atr = filter_results['volatility']['atr']
            stop_loss = current_price - (2 * atr)
            take_profit = current_price + (4 * atr)
            
            print(f"\n💰 TRADE SETUP:")
            print(f"Entry: ${current_price:.2f}")
            print(f"Stop Loss: ${stop_loss:.2f} ({((current_price - stop_loss)/current_price * 100):.1f}%)")
            print(f"Take Profit: ${take_profit:.2f} ({((take_profit - current_price)/current_price * 100):.1f}%)")
            print(f"Risk/Reward: 1:2")

if __name__ == "__main__":
    print("🚀 SWING TRADING ANALYZER TEST")
    print("="*70)
    print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    test_complete_analysis()