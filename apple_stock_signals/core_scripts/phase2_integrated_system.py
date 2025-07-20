#!/usr/bin/env python3
"""
Phase 2 Integrated Trading System
Enhanced Exit Strategies and Risk Management
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import sys
from typing import Dict, List

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Phase 1 components
try:
    from .phase1_integrated_system import Phase1IntegratedSystem
    from .trailing_stop_manager import TrailingStopManager
    from .volatility_position_sizing import VolatilityPositionSizer
    from .dynamic_exit_strategy import DynamicExitStrategy
except ImportError:
    from phase1_integrated_system import Phase1IntegratedSystem
    from trailing_stop_manager import TrailingStopManager
    from volatility_position_sizing import VolatilityPositionSizer
    from dynamic_exit_strategy import DynamicExitStrategy

class Phase2IntegratedSystem:
    """
    Integrates Phase 1 + Phase 2 components for complete trading system
    """
    
    def __init__(self, portfolio_size: float = 100000):
        # Phase 1 system
        self.phase1_system = Phase1IntegratedSystem()
        
        # Phase 2 components
        self.stop_manager = TrailingStopManager()
        self.position_sizer = VolatilityPositionSizer(portfolio_size)
        self.exit_manager = DynamicExitStrategy()
        
        self.portfolio_size = portfolio_size
        self.active_positions = []
        
    def run_complete_analysis(self, symbols: List):
        """
        Run complete Phase 1 + Phase 2 analysis
        """
        print("\n🚀 PHASE 2 INTEGRATED TRADING SYSTEM")
        print("="*80)
        print(f"Portfolio Size: ${self.portfolio_size:,}")
        print(f"Analyzing {len(symbols)} symbols with advanced exit strategies...")
        print("="*80)
        
        # Step 1: Run Phase 1 analysis to get signals
        print("\n📊 RUNNING PHASE 1 ANALYSIS")
        phase1_results = self.phase1_system.run_complete_analysis(symbols)
        
        # Step 2: Apply Phase 2 enhancements
        print("\n🎯 APPLYING PHASE 2 ENHANCEMENTS")
        
        # Get recommendations from Phase 1
        signals = phase1_results['recommendations']
        
        if not signals:
            print("❌ No signals from Phase 1 - no positions to size")
            return {
                'phase1_results': phase1_results,
                'enhanced_positions': [],
                'portfolio_summary': None
            }
        
        # Fetch market data for position sizing
        market_data = self._fetch_market_data_for_signals(signals)
        
        # Step 3: Calculate optimal position sizes
        print("\n💰 CALCULATING OPTIMAL POSITION SIZES")
        sized_positions = self._calculate_position_sizes(signals, market_data)
        
        # Step 4: Set up trailing stops
        print("\n🛡️ SETTING UP TRAILING STOPS")
        positions_with_stops = self._setup_trailing_stops(sized_positions, market_data, phase1_results['regime'])
        
        # Step 5: Define exit strategies
        print("\n🎯 DEFINING EXIT STRATEGIES")
        complete_positions = self._define_exit_strategies(positions_with_stops, market_data, phase1_results['regime'])
        
        # Step 6: Generate portfolio summary
        portfolio_summary = self._generate_portfolio_summary(complete_positions)
        
        # Save results
        self._save_results(phase1_results, complete_positions, portfolio_summary)
        
        return {
            'phase1_results': phase1_results,
            'enhanced_positions': complete_positions,
            'portfolio_summary': portfolio_summary
        }
    
    def _fetch_market_data_for_signals(self, signals: List[Dict]) -> Dict:
        """Fetch market data for all signals"""
        import yfinance as yf
        market_data = {}
        
        for signal in signals:
            symbol = signal['symbol']
            if symbol not in market_data:
                print(f"  Fetching data for {symbol}...", end='', flush=True)
                try:
                    ticker = yf.Ticker(symbol)
                    data = ticker.history(period='3mo')
                    market_data[symbol] = data
                    print(" ✅")
                except Exception as e:
                    print(f" ❌ Error: {str(e)}")
        
        return market_data
    
    def _calculate_position_sizes(self, signals: List[Dict], market_data: Dict) -> List[Dict]:
        """Calculate optimal position sizes for all signals"""
        # Mock historical performance - in real system would use actual data
        historical_performance = {
            'win_rate': 0.55,
            'avg_win': 0.03,
            'avg_loss': 0.015
        }
        
        # Convert signals to trade setups
        trade_setups = []
        for signal in signals:
            setup = {
                'symbol': signal['symbol'],
                'entry_price': signal['entry_price'],
                'stop_loss': signal['stop_loss'],
                'take_profit_1': signal['take_profit_1'],
                'confidence': signal['confidence'],
                'combined_score': signal['combined_score']
            }
            trade_setups.append(setup)
        
        # Calculate portfolio allocation
        allocation_result = self.position_sizer.calculate_portfolio_allocation(
            trade_setups,
            historical_performance,
            market_data
        )
        
        # Enhance signals with position sizing
        sized_positions = []
        for alloc in allocation_result['allocations']:
            # Find original signal
            original = next(s for s in signals if s['symbol'] == alloc['symbol'])
            
            # Combine with allocation
            enhanced = {**original}
            enhanced.update({
                'shares': alloc['shares'],
                'position_value': alloc['position_value'],
                'position_percentage': alloc['position_percentage'],
                'risk_amount': alloc['risk_amount'],
                'risk_percentage': alloc['risk_percentage']
            })
            
            sized_positions.append(enhanced)
        
        # Display summary
        summary = allocation_result['summary']
        print(f"\n  Total Positions: {summary['total_positions']}")
        print(f"  Capital Allocated: ${summary['total_allocated']:,.2f} ({summary['total_allocated_pct']:.1f}%)")
        print(f"  Portfolio Risk: {summary['total_risk_pct']:.1f}%")
        print(f"  Cash Remaining: ${summary['cash_remaining']:,.2f} ({summary['cash_remaining_pct']:.1f}%)")
        
        return sized_positions
    
    def _setup_trailing_stops(self, positions: List[Dict], market_data: Dict, 
                            market_regime: Dict) -> List[Dict]:
        """Set up trailing stops for all positions"""
        positions_with_stops = []
        
        for position in positions:
            symbol = position['symbol']
            if symbol not in market_data:
                continue
            
            # Calculate optimal trailing stop
            stop_info = self.stop_manager.calculate_optimal_trailing_stop(
                market_data[symbol],
                position['entry_price'],
                'long',  # Assuming long positions
                market_regime['regime']
            )
            
            # Add stop information to position
            position['trailing_stop'] = {
                'strategy': stop_info['optimal_strategy'],
                'current_stop': stop_info['current_stop'],
                'stop_distance': stop_info['stop_distance'],
                'stop_percentage': stop_info['stop_percentage'],
                'recommendation': stop_info['recommendation']
            }
            
            # Update stop loss if trailing stop is higher
            if stop_info['current_stop'] > position['stop_loss']:
                position['stop_loss'] = stop_info['current_stop']
                position['stop_updated'] = True
            
            positions_with_stops.append(position)
            
            print(f"\n  {symbol}: {stop_info['optimal_strategy']}")
            print(f"    Stop: ${stop_info['current_stop']:.2f} ({stop_info['stop_percentage']:.1f}% away)")
        
        return positions_with_stops
    
    def _define_exit_strategies(self, positions: List[Dict], market_data: Dict, 
                              market_regime: Dict) -> List[Dict]:
        """Define complete exit strategies for all positions"""
        complete_positions = []
        
        for position in positions:
            symbol = position['symbol']
            if symbol not in market_data:
                continue
            
            # Create position dict for exit manager
            pos_dict = {
                'symbol': symbol,
                'entry_price': position['entry_price'],
                'entry_date': datetime.now(),  # Assuming new positions
                'stop_loss': position['stop_loss'],
                'position_type': 'long',
                'shares': position['shares']
            }
            
            # Calculate exit strategy
            exit_plan = self.exit_manager.calculate_exit_strategy(
                pos_dict,
                market_data[symbol],
                market_regime['regime']
            )
            
            # Add exit plan to position
            position['exit_plan'] = {
                'profit_levels': exit_plan['profit_levels'],
                'optimal_action': exit_plan['optimal_action'],
                'time_limits': {
                    'max_days': 15,
                    'review_days': 5,
                    'breakeven_days': 3
                }
            }
            
            # Add partial take profit schedule
            position['take_profit_schedule'] = []
            for level in exit_plan['profit_levels']:
                if not level['already_hit']:
                    position['take_profit_schedule'].append({
                        'price': level['target_price'],
                        'shares_to_sell': int(position['shares'] * level['percentage_to_exit'] / 100),
                        'r_multiple': level['r_multiple']
                    })
            
            complete_positions.append(position)
            
            print(f"\n  {symbol}: {exit_plan['optimal_action']['action']}")
            print(f"    Targets: ", end='')
            for i, level in enumerate(exit_plan['profit_levels'][:3]):
                print(f"${level['target_price']:.2f} ", end='')
            print()
        
        return complete_positions
    
    def _generate_portfolio_summary(self, positions: List[Dict]) -> Dict:
        """Generate comprehensive portfolio summary"""
        if not positions:
            return None
        
        total_value = sum(p['position_value'] for p in positions)
        total_risk = sum(p['risk_amount'] for p in positions)
        
        summary = {
            'portfolio_metrics': {
                'total_positions': len(positions),
                'total_value': total_value,
                'total_risk': total_risk,
                'average_position_size': total_value / len(positions),
                'portfolio_utilization': (total_value / self.portfolio_size) * 100,
                'portfolio_heat': (total_risk / self.portfolio_size) * 100
            },
            'position_breakdown': [],
            'risk_analysis': {
                'max_single_risk': max(p['risk_percentage'] for p in positions),
                'min_single_risk': min(p['risk_percentage'] for p in positions),
                'diversification_score': self._calculate_diversification_score(positions)
            }
        }
        
        # Add position details
        for position in positions:
            summary['position_breakdown'].append({
                'symbol': position['symbol'],
                'shares': position['shares'],
                'value': position['position_value'],
                'percentage': position['position_percentage'],
                'risk': position['risk_percentage'],
                'stop_strategy': position['trailing_stop']['strategy'],
                'exit_strategy': position['exit_plan']['optimal_action']['action']
            })
        
        return summary
    
    def _calculate_diversification_score(self, positions: List[Dict]) -> float:
        """Calculate portfolio diversification score (0-100)"""
        if len(positions) <= 1:
            return 0
        
        # Simple diversification based on position count and size distribution
        position_count_score = min(100, len(positions) * 20)  # Max at 5 positions
        
        # Calculate concentration (Herfindahl index)
        total_value = sum(p['position_value'] for p in positions)
        concentrations = [(p['position_value'] / total_value) ** 2 for p in positions]
        herfindahl = sum(concentrations)
        concentration_score = (1 - herfindahl) * 100
        
        # Average the scores
        return (position_count_score + concentration_score) / 2
    
    def _save_results(self, phase1_results: Dict, positions: List[Dict], 
                     portfolio_summary: Dict):
        """Save all results to files"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create output directory
        output_dir = 'outputs/phase2_integrated'
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare complete results
        complete_results = {
            'timestamp': timestamp,
            'portfolio_size': self.portfolio_size,
            'market_regime': phase1_results['regime'],
            'phase1_signals': phase1_results['recommendations'],
            'enhanced_positions': positions,
            'portfolio_summary': portfolio_summary
        }
        
        # Save JSON
        with open(f'{output_dir}/complete_analysis_{timestamp}.json', 'w') as f:
            json.dump(complete_results, f, indent=2, default=str)
        
        # Generate readable report
        self._generate_readable_report(complete_results, output_dir, timestamp)
        
        print(f"\n💾 Results saved to {output_dir}/")
    
    def _generate_readable_report(self, results: Dict, output_dir: str, timestamp: str):
        """Generate human-readable report"""
        report = "="*80 + "\n"
        report += "📊 PHASE 2 COMPLETE TRADING SYSTEM REPORT\n"
        report += "="*80 + "\n"
        report += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        report += f"Portfolio Size: ${self.portfolio_size:,}\n"
        report += f"Market Regime: {results['market_regime']['regime']} "
        report += f"(Confidence: {results['market_regime']['confidence']:.1f}%)\n\n"
        
        if results['enhanced_positions']:
            report += "🎯 TRADING POSITIONS WITH ENHANCED RISK MANAGEMENT:\n"
            report += "-"*80 + "\n"
            
            for i, pos in enumerate(results['enhanced_positions'], 1):
                report += f"\n{i}. {pos['symbol']} - {pos['action']}\n"
                report += f"   Entry: ${pos['entry_price']:.2f}\n"
                report += f"   Shares: {pos['shares']} (${pos['position_value']:,.2f} = {pos['position_percentage']:.1f}%)\n"
                report += f"   Risk: ${pos['risk_amount']:,.2f} ({pos['risk_percentage']:.1f}% of portfolio)\n"
                
                # Trailing stop
                ts = pos['trailing_stop']
                report += f"\n   🛡️ Trailing Stop: {ts['strategy']}\n"
                report += f"      Current Stop: ${ts['current_stop']:.2f} ({ts['stop_percentage']:.1f}% away)\n"
                report += f"      {ts['recommendation']}\n"
                
                # Exit plan
                ep = pos['exit_plan']
                report += f"\n   🎯 Exit Strategy: {ep['optimal_action']['action']}\n"
                report += f"      Take Profit Schedule:\n"
                for j, tp in enumerate(pos['take_profit_schedule'][:3]):
                    report += f"        Level {j+1}: ${tp['price']:.2f} ({tp['r_multiple']:.1f}R) - Sell {tp['shares_to_sell']} shares\n"
                
                report += "\n" + "-"*40
            
            # Portfolio summary
            if results['portfolio_summary']:
                ps = results['portfolio_summary']
                report += f"\n\n📊 PORTFOLIO SUMMARY:\n"
                report += f"Total Positions: {ps['portfolio_metrics']['total_positions']}\n"
                report += f"Total Value: ${ps['portfolio_metrics']['total_value']:,.2f}\n"
                report += f"Total Risk: ${ps['portfolio_metrics']['total_risk']:,.2f}\n"
                report += f"Portfolio Utilization: {ps['portfolio_metrics']['portfolio_utilization']:.1f}%\n"
                report += f"Portfolio Heat: {ps['portfolio_metrics']['portfolio_heat']:.1f}%\n"
                report += f"Diversification Score: {ps['risk_analysis']['diversification_score']:.1f}/100\n"
        else:
            report += "\n❌ No positions generated - market conditions unfavorable\n"
        
        report += "\n" + "="*80 + "\n"
        report += "⚠️ DISCLAIMER: This is for educational purposes only. Not financial advice.\n"
        report += "="*80 + "\n"
        
        # Save report
        with open(f'{output_dir}/report_{timestamp}.txt', 'w') as f:
            f.write(report)
        
        # Print to console
        print(report)


def main():
    """Run the Phase 2 integrated system"""
    # Initialize system with $100k portfolio
    system = Phase2IntegratedSystem(portfolio_size=100000)
    
    # Define symbols to analyze
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 
               'META', 'TSLA', 'BRK-B', 'JPM', 'JNJ']
    
    # Run complete analysis
    results = system.run_complete_analysis(symbols)
    
    print("\n✅ Phase 2 Integrated System Analysis Complete!")
    
    if results['enhanced_positions']:
        print(f"\nGenerated {len(results['enhanced_positions'])} positions with:")
        print("  • Optimal position sizing")
        print("  • Dynamic trailing stops")
        print("  • Partial profit taking schedules")
        print("  • Time-based exit rules")
    else:
        print("\nNo positions generated - review market conditions")


if __name__ == "__main__":
    main()