#!/usr/bin/env python3
"""
Integrated Risk Management System
Combines dynamic stops, position sizing, and profit taking
Complete risk management for 2-15 day swing trading
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os

# Import risk management components
from core.risk_management.stop_loss import DynamicStopLossSystem
from core.risk_management.position_sizing import AdvancedPositionSizing
from core.risk_management.profit_targets import ProfitTakingStrategy

class IntegratedRiskManagement:
    """
    Complete risk management system integrating all components
    """
    
    def __init__(self, account_size=10000):
        self.account_size = account_size
        self.stop_system = DynamicStopLossSystem()
        self.position_sizer = AdvancedPositionSizing(account_size)
        self.profit_strategy = ProfitTakingStrategy()
        
        # Portfolio tracking
        self.portfolio = {
            'positions': [],
            'closed_trades': [],
            'total_value': account_size,
            'cash': account_size,
            'total_risk': 0
        }
        
    def calculate_complete_trade_setup(self, analysis_data, df, current_positions=None):
        """
        Calculate complete trade setup including entry, stops, sizing, and targets
        """
        symbol = analysis_data['symbol']
        signal = analysis_data['signal']
        entry_price = df['Close'].iloc[-1]
        
        if signal not in ['BUY', 'STRONG_BUY', 'SELL', 'STRONG_SELL']:
            return {
                'symbol': symbol,
                'action': 'NO_TRADE',
                'reason': f'Signal is {signal}, not actionable'
            }
        
        position_type = 'long' if signal in ['BUY', 'STRONG_BUY'] else 'short'
        
        # 1. Calculate initial stop loss
        stop_data = self.stop_system.calculate_initial_stop(entry_price, df, position_type)
        
        # 2. Calculate position size
        analysis_data['price_data'] = df
        analysis_data['sector'] = analysis_data.get('sector', 'Unknown')
        
        # Get historical performance if available
        historical_performance = self._get_historical_performance(symbol)
        
        sizing_data = self.position_sizer.calculate_optimal_position_size(
            analysis_data,
            stop_data,
            historical_performance,
            current_positions or []
        )
        
        if not sizing_data['can_take_position']:
            return {
                'symbol': symbol,
                'action': 'NO_TRADE',
                'reason': 'Position sizing constraints',
                'details': sizing_data
            }
        
        # 3. Calculate profit targets
        entry_data = {
            'entry_price': entry_price,
            'stop_loss': stop_data['initial_stop'],
            'position_size': sizing_data['recommended_shares'],
            'atr': stop_data.get('atr', stop_data['stop_distance'] / 2)
        }
        
        profit_targets = self.profit_strategy.calculate_optimal_targets(
            entry_data, df, position_type
        )
        
        # 4. Risk/Reward analysis
        risk_reward_analysis = self._calculate_risk_reward(
            entry_price, stop_data['initial_stop'], 
            profit_targets['primary'], position_type
        )
        
        # 5. Generate complete trade setup
        trade_setup = {
            'symbol': symbol,
            'action': 'ENTER_TRADE',
            'signal': signal,
            'position_type': position_type,
            'entry': {
                'price': entry_price,
                'shares': sizing_data['recommended_shares'],
                'position_value': sizing_data['position_value'],
                'position_pct': sizing_data['position_pct']
            },
            'stop_loss': {
                'initial_stop': stop_data['initial_stop'],
                'stop_distance': stop_data['stop_distance'],
                'stop_distance_pct': stop_data['stop_distance_pct'],
                'stop_type': stop_data['stop_type']
            },
            'risk': {
                'risk_amount': sizing_data['actual_risk_amount'],
                'risk_pct': sizing_data['actual_risk_pct'],
                'portfolio_heat': self._calculate_portfolio_heat(current_positions, sizing_data)
            },
            'profit_targets': profit_targets,
            'risk_reward': risk_reward_analysis,
            'trade_quality': self._assess_trade_quality(analysis_data, risk_reward_analysis),
            'execution_plan': self._generate_execution_plan(sizing_data, stop_data, profit_targets)
        }
        
        return trade_setup
    
    def monitor_existing_position(self, position, current_df):
        """
        Monitor and manage existing position with dynamic adjustments
        """
        symbol = position['symbol']
        current_price = current_df['Close'].iloc[-1]
        
        # 1. Check stop loss updates
        stop_recommendations = self.stop_system.get_stop_recommendations(
            position, current_df
        )
        
        # 2. Monitor profit targets
        profit_monitoring = self.profit_strategy.monitor_profit_targets(
            position, current_price, current_df
        )
        
        # 3. Position health analysis
        position_health = self._analyze_position_health(
            position, current_price, stop_recommendations, profit_monitoring
        )
        
        # 4. Generate management recommendations
        management_plan = {
            'symbol': symbol,
            'current_price': current_price,
            'position_health': position_health,
            'stop_management': {
                'current_stop': position['current_stop'],
                'recommended_stop': stop_recommendations['recommended_stop'],
                'should_update': stop_recommendations['recommended_stop'] != position['current_stop'],
                'update_reason': stop_recommendations['stop_type']
            },
            'profit_management': profit_monitoring,
            'recommended_actions': self._generate_position_actions(
                position_health, stop_recommendations, profit_monitoring
            )
        }
        
        return management_plan
    
    def _calculate_risk_reward(self, entry_price, stop_loss, targets, position_type='long'):
        """Calculate risk/reward metrics"""
        risk = abs(entry_price - stop_loss)
        
        risk_reward_ratios = []
        for target in targets:
            reward = abs(target['price'] - entry_price)
            rr_ratio = reward / risk if risk > 0 else 0
            
            risk_reward_ratios.append({
                'target': target['name'],
                'ratio': rr_ratio,
                'reward': reward,
                'reward_pct': (reward / entry_price) * 100
            })
        
        # Calculate expected value
        win_rate = 0.5  # Default, should be from historical data
        avg_rr = sum(t['ratio'] for t in risk_reward_ratios) / len(risk_reward_ratios) if risk_reward_ratios else 0
        expected_value = (win_rate * avg_rr) - ((1 - win_rate) * 1)
        
        return {
            'risk_amount': risk,
            'risk_pct': (risk / entry_price) * 100,
            'targets': risk_reward_ratios,
            'average_rr': avg_rr,
            'expected_value': expected_value,
            'minimum_win_rate': 1 / (1 + avg_rr) if avg_rr > 0 else 0.5
        }
    
    def _assess_trade_quality(self, analysis_data, risk_reward):
        """Assess overall trade quality"""
        quality_score = 0
        quality_factors = []
        
        # Signal strength
        if analysis_data.get('score', 0) >= 80:
            quality_score += 25
            quality_factors.append("Strong signal")
        elif analysis_data.get('score', 0) >= 70:
            quality_score += 15
            quality_factors.append("Good signal")
        
        # Risk/Reward
        if risk_reward['average_rr'] >= 3:
            quality_score += 25
            quality_factors.append("Excellent R/R")
        elif risk_reward['average_rr'] >= 2:
            quality_score += 15
            quality_factors.append("Good R/R")
        
        # Entry timing
        if analysis_data.get('entry_quality') == 'excellent':
            quality_score += 25
            quality_factors.append("Excellent entry timing")
        
        # Expected value
        if risk_reward['expected_value'] > 0.5:
            quality_score += 25
            quality_factors.append("Positive expectancy")
        
        # Determine grade
        if quality_score >= 80:
            grade = 'A'
        elif quality_score >= 60:
            grade = 'B'
        elif quality_score >= 40:
            grade = 'C'
        else:
            grade = 'D'
        
        return {
            'score': quality_score,
            'grade': grade,
            'factors': quality_factors
        }
    
    def _generate_execution_plan(self, sizing, stop, targets):
        """Generate step-by-step execution plan"""
        plan = {
            'entry_orders': [
                {
                    'type': 'LIMIT',
                    'shares': sizing['recommended_shares'],
                    'price': sizing['current_price'],
                    'time_in_force': 'DAY'
                }
            ],
            'stop_orders': [
                {
                    'type': 'STOP_LOSS',
                    'shares': sizing['recommended_shares'],
                    'stop_price': stop['initial_stop'],
                    'time_in_force': 'GTC'
                }
            ],
            'profit_orders': []
        }
        
        # Add scale-out profit orders
        remaining_shares = sizing['recommended_shares']
        for i, level in enumerate(targets['scale_out_plan']):
            exit_shares = level['exit_shares']
            
            plan['profit_orders'].append({
                'type': 'LIMIT',
                'shares': exit_shares,
                'price': level['target_price'],
                'time_in_force': 'GTC',
                'note': f"{level['level']} - Exit {level['exit_pct']:.0%}"
            })
        
        return plan
    
    def _calculate_portfolio_heat(self, current_positions, new_position):
        """Calculate total portfolio heat including new position"""
        current_heat = sum(pos.get('risk_pct', 0) for pos in (current_positions or []))
        new_heat = new_position['actual_risk_pct']
        total_heat = current_heat + new_heat
        
        return {
            'current_heat': current_heat,
            'new_position_heat': new_heat,
            'total_heat': total_heat,
            'heat_limit': self.position_sizer.config['max_portfolio_risk'],
            'within_limits': total_heat <= self.position_sizer.config['max_portfolio_risk']
        }
    
    def _analyze_position_health(self, position, current_price, stop_rec, profit_mon):
        """Analyze overall position health"""
        entry_price = position['entry_price']
        r_multiple = profit_mon['r_multiple']
        days_held = (datetime.now() - position['entry_date']).days
        
        # Calculate position metrics
        if position['position_type'] == 'long':
            unrealized_pnl = (current_price - entry_price) * position['shares']
            pnl_pct = ((current_price - entry_price) / entry_price) * 100
        else:
            unrealized_pnl = (entry_price - current_price) * position['shares']
            pnl_pct = ((entry_price - current_price) / entry_price) * 100
        
        # Health scoring
        health_score = 50  # Base score
        
        # Profit status
        if r_multiple >= 2:
            health_score += 30
            status = 'excellent'
        elif r_multiple >= 1:
            health_score += 20
            status = 'good'
        elif r_multiple >= 0:
            health_score += 10
            status = 'neutral'
        else:
            health_score -= 20
            status = 'poor'
        
        # Time efficiency
        if days_held <= 5 and r_multiple >= 1:
            health_score += 10  # Quick profit
        elif days_held > 10 and r_multiple < 0.5:
            health_score -= 10  # Stagnant
        
        # Stop distance
        stop_distance_pct = stop_rec['current_risk_pct']
        if stop_distance_pct < 2:
            health_score += 10  # Tight stop (protected)
        
        return {
            'status': status,
            'health_score': health_score,
            'unrealized_pnl': unrealized_pnl,
            'pnl_pct': pnl_pct,
            'r_multiple': r_multiple,
            'days_held': days_held,
            'targets_hit': len(profit_mon['targets_hit']),
            'stop_distance_pct': stop_distance_pct
        }
    
    def _generate_position_actions(self, health, stop_rec, profit_mon):
        """Generate specific actions for position management"""
        actions = []
        
        # Stop loss management
        if stop_rec['action'] == 'RAISE_STOP':
            actions.append({
                'type': 'UPDATE_STOP',
                'new_stop': stop_rec['recommended_stop'],
                'priority': 'high',
                'reason': stop_rec['stop_type']
            })
        
        # Profit taking
        recommendation = profit_mon['recommendation']
        if recommendation['action'] == 'EXIT_NOW':
            actions.append({
                'type': 'EXIT_POSITION',
                'priority': 'immediate',
                'reason': recommendation['reason']
            })
        elif recommendation['action'] == 'TAKE_PARTIAL':
            actions.append({
                'type': 'PARTIAL_EXIT',
                'shares_pct': 0.33,
                'priority': 'high',
                'reason': recommendation['reason']
            })
        
        # Time-based actions
        if health['days_held'] > 15:
            actions.append({
                'type': 'TIME_EXIT',
                'priority': 'medium',
                'reason': 'Maximum holding period reached'
            })
        
        # Health-based actions
        if health['health_score'] < 30:
            actions.append({
                'type': 'REVIEW_POSITION',
                'priority': 'high',
                'reason': 'Poor position health'
            })
        
        return actions
    
    def _get_historical_performance(self, symbol):
        """Get historical performance for Kelly sizing"""
        # This would connect to your trade history database
        # For now, return default values
        return {
            'win_rate': 0.50,
            'avg_win': 2.5,
            'avg_loss': -1.0,
            'sample_size': 100
        }
    
    def generate_risk_report(self, trade_setup):
        """Generate comprehensive risk management report"""
        report = f"\n{'='*60}\n"
        report += f"💼 RISK MANAGEMENT REPORT: {trade_setup['symbol']}\n"
        report += f"{'='*60}\n"
        
        if trade_setup['action'] == 'NO_TRADE':
            report += f"\n❌ Trade Rejected: {trade_setup['reason']}\n"
            return report
        
        # Entry details
        entry = trade_setup['entry']
        report += f"\n📍 ENTRY SETUP:\n"
        report += f"  Signal: {trade_setup['signal']} ({trade_setup['position_type']})\n"
        report += f"  Entry Price: ${entry['price']:.2f}\n"
        report += f"  Position Size: {entry['shares']} shares\n"
        report += f"  Position Value: ${entry['position_value']:,.2f} ({entry['position_pct']:.1%} of account)\n"
        
        # Stop loss
        stop = trade_setup['stop_loss']
        report += f"\n🛡️ STOP LOSS:\n"
        report += f"  Stop Price: ${stop['initial_stop']:.2f}\n"
        report += f"  Stop Distance: ${stop['stop_distance']:.2f} ({stop['stop_distance_pct']:.1%})\n"
        report += f"  Stop Type: {stop['stop_type']}\n"
        
        # Risk analysis
        risk = trade_setup['risk']
        report += f"\n⚠️ RISK ANALYSIS:\n"
        report += f"  Position Risk: ${risk['risk_amount']:.2f} ({risk['risk_pct']:.1%} of account)\n"
        report += f"  Current Portfolio Heat: {risk['portfolio_heat']['current_heat']:.1%}\n"
        report += f"  Total Heat (w/ new): {risk['portfolio_heat']['total_heat']:.1%}\n"
        
        # Profit targets
        targets = trade_setup['profit_targets']['primary']
        report += f"\n🎯 PROFIT TARGETS:\n"
        for target in targets:
            report += f"  {target['name']}: ${target['price']:.2f} "
            report += f"(+{target['distance_pct']:.1%}) - Exit {target['exit_pct']:.0%}\n"
        
        # Risk/Reward
        rr = trade_setup['risk_reward']
        report += f"\n📊 RISK/REWARD:\n"
        report += f"  Average R/R: {rr['average_rr']:.2f}:1\n"
        report += f"  Expected Value: {rr['expected_value']:.2f}\n"
        report += f"  Min Win Rate: {rr['minimum_win_rate']:.1%}\n"
        
        # Trade quality
        quality = trade_setup['trade_quality']
        report += f"\n⭐ TRADE QUALITY: Grade {quality['grade']} ({quality['score']}/100)\n"
        for factor in quality['factors']:
            report += f"  ✓ {factor}\n"
        
        # Execution plan
        report += f"\n📋 EXECUTION PLAN:\n"
        plan = trade_setup['execution_plan']
        report += f"  1. Enter: {plan['entry_orders'][0]['shares']} shares at ${plan['entry_orders'][0]['price']:.2f}\n"
        report += f"  2. Set Stop: ${plan['stop_orders'][0]['stop_price']:.2f}\n"
        report += f"  3. Set Targets:\n"
        for order in plan['profit_orders']:
            report += f"     - {order['note']}\n"
        
        return report


def main():
    """Test integrated risk management system"""
    import yfinance as yf
    
    # Initialize system
    risk_manager = IntegratedRiskManagement(account_size=10000)
    
    # Test stock
    symbol = 'NVDA'
    stock = yf.Ticker(symbol)
    df = stock.history(period='3mo')
    
    if not df.empty:
        print(f"🎯 Testing Integrated Risk Management for {symbol}")
        print(f"Account Size: ${risk_manager.account_size:,}")
        
        # Simulate analysis data
        analysis_data = {
            'symbol': symbol,
            'signal': 'STRONG_BUY',
            'score': 85,
            'entry_quality': 'excellent',
            'sector': 'Technology'
        }
        
        # Calculate complete trade setup
        trade_setup = risk_manager.calculate_complete_trade_setup(
            analysis_data, df, current_positions=[]
        )
        
        # Generate report
        print(risk_manager.generate_risk_report(trade_setup))
        
        # Test position monitoring
        if trade_setup['action'] == 'ENTER_TRADE':
            # Simulate existing position
            position = {
                'symbol': symbol,
                'entry_price': trade_setup['entry']['price'],
                'current_stop': trade_setup['stop_loss']['initial_stop'],
                'shares': trade_setup['entry']['shares'],
                'position_type': trade_setup['position_type'],
                'entry_date': datetime.now() - timedelta(days=5),
                'profit_targets': trade_setup['profit_targets']
            }
            
            # Monitor position
            management_plan = risk_manager.monitor_existing_position(position, df)
            
            print(f"\n📊 POSITION MONITORING:")
            print(f"Position Health: {management_plan['position_health']['status'].upper()}")
            print(f"P&L: ${management_plan['position_health']['unrealized_pnl']:.2f} "
                  f"({management_plan['position_health']['pnl_pct']:.1f}%)")
            print(f"R-Multiple: {management_plan['position_health']['r_multiple']:.2f}R")
            
            print(f"\n🎯 Recommended Actions:")
            for action in management_plan['recommended_actions']:
                print(f"  • {action['type']}: {action['reason']} (Priority: {action['priority']})")

if __name__ == "__main__":
    main()