#!/usr/bin/env python3
"""
Profit Taking Strategy Module
Implements scale-out targets, volatility-adjusted exits, and Fibonacci extensions
Optimized for 2-15 day swing trading
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class ProfitTakingStrategy:
    """
    Advanced profit taking strategies for maximizing gains
    """
    
    def __init__(self):
        self.config = {
            # Scale-out levels
            'scale_out_levels': [
                {'r_multiple': 1.0, 'exit_pct': 0.33, 'name': 'TP1'},
                {'r_multiple': 2.0, 'exit_pct': 0.33, 'name': 'TP2'},
                {'r_multiple': 3.0, 'exit_pct': 0.34, 'name': 'TP3'}
            ],
            
            # Volatility adjustments
            'volatility_expansion_multiplier': 1.5,  # Extend targets in low volatility
            'volatility_contraction_multiplier': 0.8,  # Tighten targets in high volatility
            
            # Fibonacci levels
            'fib_extensions': [1.272, 1.618, 2.618, 4.236],
            
            # Time-based exits
            'momentum_exit_days': 3,  # Exit if momentum stalls for 3 days
            'max_extension_days': 20,  # Maximum trade extension
            
            # Advanced exit triggers
            'rsi_overbought_exit': 80,
            'volume_climax_multiplier': 3.0,
            'divergence_exit': True,
            'gap_exit_pct': 0.03  # Exit on 3% gap in direction
        }
        
    def calculate_atr_targets(self, entry_price, atr, position_type='long'):
        """Calculate ATR-based profit targets"""
        targets = []
        
        for level in self.config['scale_out_levels']:
            r_multiple = level['r_multiple']
            
            if position_type == 'long':
                target_price = entry_price + (atr * 2 * r_multiple)  # 2 ATR = 1R
            else:
                target_price = entry_price - (atr * 2 * r_multiple)
                
            targets.append({
                'name': level['name'],
                'price': target_price,
                'r_multiple': r_multiple,
                'exit_pct': level['exit_pct'],
                'distance': abs(target_price - entry_price),
                'distance_pct': abs(target_price - entry_price) / entry_price * 100
            })
        
        return targets
    
    def calculate_fibonacci_targets(self, df, swing_low, swing_high, position_type='long'):
        """Calculate Fibonacci extension targets"""
        fib_targets = []
        swing_range = swing_high - swing_low
        
        for fib_level in self.config['fib_extensions']:
            if position_type == 'long':
                target = swing_high + (swing_range * (fib_level - 1))
            else:
                target = swing_low - (swing_range * (fib_level - 1))
                
            fib_targets.append({
                'level': fib_level,
                'price': target,
                'name': f'Fib {fib_level}'
            })
        
        return fib_targets
    
    def calculate_structure_targets(self, df, position_type='long', lookback=50):
        """Calculate targets based on market structure (previous highs/lows)"""
        structure_targets = []
        
        if position_type == 'long':
            # Find recent highs as resistance targets
            recent_highs = []
            for i in range(max(0, len(df) - lookback), len(df) - 1):
                if i > 0 and i < len(df) - 1:
                    if df['High'].iloc[i] > df['High'].iloc[i-1] and df['High'].iloc[i] > df['High'].iloc[i+1]:
                        recent_highs.append((df.index[i], df['High'].iloc[i]))
            
            # Sort by price
            recent_highs.sort(key=lambda x: x[1])
            
            # Use top 3 highs as targets
            for i, (date, price) in enumerate(recent_highs[-3:]):
                structure_targets.append({
                    'name': f'Resistance {i+1}',
                    'price': price,
                    'date': date,
                    'type': 'resistance'
                })
        else:
            # Find recent lows as support targets for shorts
            recent_lows = []
            for i in range(max(0, len(df) - lookback), len(df) - 1):
                if i > 0 and i < len(df) - 1:
                    if df['Low'].iloc[i] < df['Low'].iloc[i-1] and df['Low'].iloc[i] < df['Low'].iloc[i+1]:
                        recent_lows.append((df.index[i], df['Low'].iloc[i]))
            
            recent_lows.sort(key=lambda x: x[1], reverse=True)
            
            for i, (date, price) in enumerate(recent_lows[-3:]):
                structure_targets.append({
                    'name': f'Support {i+1}',
                    'price': price,
                    'date': date,
                    'type': 'support'
                })
        
        return structure_targets
    
    def adjust_targets_for_volatility(self, targets, current_volatility, average_volatility):
        """Adjust profit targets based on current vs average volatility"""
        if average_volatility == 0:
            return targets
        
        volatility_ratio = current_volatility / average_volatility
        
        # Determine adjustment
        if volatility_ratio < 0.7:
            # Low volatility - extend targets
            adjustment = self.config['volatility_expansion_multiplier']
            adjustment_reason = "low_volatility_expansion"
        elif volatility_ratio > 1.5:
            # High volatility - tighten targets
            adjustment = self.config['volatility_contraction_multiplier']
            adjustment_reason = "high_volatility_contraction"
        else:
            # Normal volatility - no adjustment
            adjustment = 1.0
            adjustment_reason = "normal_volatility"
        
        # Apply adjustment
        adjusted_targets = []
        for target in targets:
            adjusted_target = target.copy()
            if 'distance' in target:
                adjusted_target['original_price'] = target['price']
                adjusted_target['price'] = target['price'] * adjustment
                adjusted_target['volatility_adjustment'] = adjustment
                adjusted_target['adjustment_reason'] = adjustment_reason
            
            adjusted_targets.append(adjusted_target)
        
        return adjusted_targets
    
    def check_exit_conditions(self, position_data, current_data, df):
        """Check various exit conditions beyond fixed targets"""
        exit_signals = []
        current_price = current_data['price']
        entry_price = position_data['entry_price']
        
        # 1. RSI Overbought/Oversold
        if 'rsi' in current_data:
            if position_data['position_type'] == 'long' and current_data['rsi'] > self.config['rsi_overbought_exit']:
                exit_signals.append({
                    'type': 'rsi_overbought',
                    'strength': 'strong',
                    'message': f"RSI overbought at {current_data['rsi']:.1f}"
                })
            elif position_data['position_type'] == 'short' and current_data['rsi'] < (100 - self.config['rsi_overbought_exit']):
                exit_signals.append({
                    'type': 'rsi_oversold',
                    'strength': 'strong',
                    'message': f"RSI oversold at {current_data['rsi']:.1f}"
                })
        
        # 2. Volume Climax
        if 'volume' in current_data:
            avg_volume = df['Volume'].rolling(20).mean().iloc[-1]
            if current_data['volume'] > avg_volume * self.config['volume_climax_multiplier']:
                exit_signals.append({
                    'type': 'volume_climax',
                    'strength': 'medium',
                    'message': f"Volume climax at {current_data['volume']/avg_volume:.1f}x average"
                })
        
        # 3. Momentum Stall
        days_in_trade = (datetime.now() - position_data['entry_date']).days
        if days_in_trade >= self.config['momentum_exit_days']:
            recent_range = df['High'].tail(self.config['momentum_exit_days']).max() - df['Low'].tail(self.config['momentum_exit_days']).min()
            avg_range = (df['High'] - df['Low']).rolling(20).mean().iloc[-1]
            
            if recent_range < avg_range * 0.5:
                exit_signals.append({
                    'type': 'momentum_stall',
                    'strength': 'medium',
                    'message': f"Momentum stalled for {self.config['momentum_exit_days']} days"
                })
        
        # 4. Gap Exit
        if len(df) > 1:
            gap_pct = (df['Open'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]
            if abs(gap_pct) > self.config['gap_exit_pct']:
                if (position_data['position_type'] == 'long' and gap_pct > 0) or \
                   (position_data['position_type'] == 'short' and gap_pct < 0):
                    exit_signals.append({
                        'type': 'gap_exit',
                        'strength': 'strong',
                        'message': f"Favorable gap of {gap_pct:.1%}"
                    })
        
        # 5. Divergence Detection
        if self.config['divergence_exit'] and len(df) > 20:
            # Simple divergence: price making new highs but RSI not
            if position_data['position_type'] == 'long':
                price_higher = df['High'].iloc[-1] > df['High'].iloc[-20:].max()
                rsi_lower = current_data.get('rsi', 50) < df['RSI'].iloc[-20:].max() if 'RSI' in df.columns else False
                
                if price_higher and rsi_lower:
                    exit_signals.append({
                        'type': 'bearish_divergence',
                        'strength': 'medium',
                        'message': "Bearish divergence detected"
                    })
        
        return exit_signals
    
    def calculate_optimal_targets(self, entry_data, df, position_type='long'):
        """Calculate comprehensive profit targets using multiple methods"""
        entry_price = entry_data['entry_price']
        stop_loss = entry_data['stop_loss']
        atr = entry_data.get('atr', abs(entry_price - stop_loss) / 2)
        
        # 1. ATR-based targets (primary)
        atr_targets = self.calculate_atr_targets(entry_price, atr, position_type)
        
        # 2. Structure-based targets
        structure_targets = self.calculate_structure_targets(df, position_type)
        
        # 3. Fibonacci targets (if swing points available)
        fib_targets = []
        if len(df) > 20:
            swing_low = df['Low'].tail(20).min()
            swing_high = df['High'].tail(20).max()
            fib_targets = self.calculate_fibonacci_targets(df, swing_low, swing_high, position_type)
        
        # 4. Adjust for volatility
        current_vol = df['Close'].pct_change().tail(20).std() * np.sqrt(252)
        avg_vol = df['Close'].pct_change().tail(100).std() * np.sqrt(252)
        
        adjusted_targets = self.adjust_targets_for_volatility(atr_targets, current_vol, avg_vol)
        
        # 5. Combine and prioritize targets
        all_targets = {
            'primary': adjusted_targets,
            'structure': structure_targets,
            'fibonacci': fib_targets,
            'exit_conditions': {
                'rsi_exit': self.config['rsi_overbought_exit'],
                'momentum_days': self.config['momentum_exit_days'],
                'max_days': self.config['max_extension_days']
            }
        }
        
        # 6. Generate scale-out plan
        scale_out_plan = self.generate_scale_out_plan(entry_data, adjusted_targets)
        all_targets['scale_out_plan'] = scale_out_plan
        
        return all_targets
    
    def generate_scale_out_plan(self, entry_data, targets):
        """Generate detailed scale-out execution plan"""
        position_size = entry_data.get('position_size', 100)
        remaining_size = position_size
        
        scale_out_plan = []
        cumulative_exit = 0
        
        for i, target in enumerate(targets):
            exit_size = int(position_size * target['exit_pct'])
            
            # Ensure we don't exceed position size
            if i == len(targets) - 1:
                exit_size = remaining_size
            
            cumulative_exit += exit_size
            remaining_size -= exit_size
            
            scale_out_plan.append({
                'level': target['name'],
                'target_price': target['price'],
                'exit_shares': exit_size,
                'exit_pct': target['exit_pct'],
                'remaining_shares': remaining_size,
                'r_multiple': target['r_multiple'],
                'cumulative_exit_pct': cumulative_exit / position_size * 100
            })
        
        return scale_out_plan
    
    def monitor_profit_targets(self, position_data, current_price, df):
        """Monitor current position against profit targets"""
        entry_price = position_data['entry_price']
        targets = position_data.get('profit_targets', {})
        
        # Calculate current profit
        if position_data['position_type'] == 'long':
            current_profit = current_price - entry_price
            profit_pct = (current_profit / entry_price) * 100
        else:
            current_profit = entry_price - current_price
            profit_pct = (current_profit / entry_price) * 100
        
        # Calculate R-multiple
        initial_risk = abs(entry_price - position_data['stop_loss'])
        r_multiple = current_profit / initial_risk if initial_risk > 0 else 0
        
        # Check which targets have been hit
        targets_hit = []
        next_target = None
        
        for target in targets.get('primary', []):
            if position_data['position_type'] == 'long':
                if current_price >= target['price']:
                    targets_hit.append(target)
                elif next_target is None:
                    next_target = target
            else:
                if current_price <= target['price']:
                    targets_hit.append(target)
                elif next_target is None:
                    next_target = target
        
        # Check exit conditions
        current_data = {
            'price': current_price,
            'rsi': df['RSI'].iloc[-1] if 'RSI' in df.columns else 50,
            'volume': df['Volume'].iloc[-1]
        }
        
        exit_conditions = self.check_exit_conditions(position_data, current_data, df)
        
        monitoring_result = {
            'current_price': current_price,
            'entry_price': entry_price,
            'profit_amount': current_profit,
            'profit_pct': profit_pct,
            'r_multiple': r_multiple,
            'targets_hit': targets_hit,
            'next_target': next_target,
            'exit_signals': exit_conditions,
            'recommendation': self._generate_recommendation(r_multiple, targets_hit, exit_conditions)
        }
        
        return monitoring_result
    
    def _generate_recommendation(self, r_multiple, targets_hit, exit_signals):
        """Generate action recommendation based on current status"""
        strong_exits = [s for s in exit_signals if s['strength'] == 'strong']
        
        if strong_exits:
            return {
                'action': 'EXIT_NOW',
                'reason': strong_exits[0]['message'],
                'urgency': 'high'
            }
        elif len(targets_hit) >= 2 and r_multiple >= 2:
            return {
                'action': 'TRAIL_TIGHT',
                'reason': 'Multiple targets hit, protect profits',
                'urgency': 'medium'
            }
        elif r_multiple >= 1 and not targets_hit:
            return {
                'action': 'TAKE_PARTIAL',
                'reason': 'First target reached',
                'urgency': 'medium'
            }
        else:
            return {
                'action': 'HOLD',
                'reason': 'Targets not yet reached',
                'urgency': 'low'
            }
    
    def generate_profit_report(self, targets, monitoring_result=None):
        """Generate human-readable profit taking report"""
        report = "\n💰 Profit Taking Strategy\n"
        report += "=" * 50 + "\n"
        
        # Primary targets
        report += "\n📊 Primary Targets (ATR-based):\n"
        for target in targets.get('primary', []):
            report += f"  {target['name']}: ${target['price']:.2f} "
            report += f"({target['distance_pct']:.1f}% gain) - "
            report += f"Exit {target['exit_pct']:.0%} of position\n"
        
        # Structure targets
        if targets.get('structure'):
            report += "\n🏗️ Structure-Based Targets:\n"
            for target in targets['structure'][:3]:
                report += f"  {target['name']}: ${target['price']:.2f}\n"
        
        # Scale-out plan
        if 'scale_out_plan' in targets:
            report += "\n📈 Scale-Out Execution Plan:\n"
            for plan in targets['scale_out_plan']:
                report += f"  {plan['level']}: Exit {plan['exit_shares']} shares at ${plan['target_price']:.2f}\n"
        
        # Current monitoring
        if monitoring_result:
            report += f"\n📍 Current Status:\n"
            report += f"  Profit: ${monitoring_result['profit_amount']:.2f} ({monitoring_result['profit_pct']:.1f}%)\n"
            report += f"  R-Multiple: {monitoring_result['r_multiple']:.2f}R\n"
            
            if monitoring_result['targets_hit']:
                report += f"  Targets Hit: {len(monitoring_result['targets_hit'])}\n"
            
            if monitoring_result['next_target']:
                next_t = monitoring_result['next_target']
                distance = abs(next_t['price'] - monitoring_result['current_price'])
                report += f"  Next Target: {next_t['name']} at ${next_t['price']:.2f} (${distance:.2f} away)\n"
            
            # Recommendation
            rec = monitoring_result['recommendation']
            report += f"\n🎯 Recommendation: {rec['action']}\n"
            report += f"  Reason: {rec['reason']}\n"
        
        return report


def main():
    """Test the profit taking strategy"""
    import yfinance as yf
    
    # Initialize strategy
    profit_strategy = ProfitTakingStrategy()
    
    # Test stock
    symbol = 'AAPL'
    stock = yf.Ticker(symbol)
    df = stock.history(period='3mo')
    
    # Add RSI for testing (simplified calculation)
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    if not df.empty:
        current_price = df['Close'].iloc[-1]
        entry_price = df['Close'].iloc[-10]  # Simulate entry 10 days ago
        
        print(f"🎯 Testing Profit Taking Strategy for {symbol}")
        print(f"Entry Price: ${entry_price:.2f}")
        print(f"Current Price: ${current_price:.2f}")
        
        # Calculate targets
        entry_data = {
            'entry_price': entry_price,
            'stop_loss': entry_price * 0.98,  # 2% stop
            'position_size': 100,
            'atr': df['Close'].iloc[-1] * 0.02  # Approximate ATR
        }
        
        targets = profit_strategy.calculate_optimal_targets(entry_data, df)
        print(profit_strategy.generate_profit_report(targets))
        
        # Monitor current position
        position_data = {
            'entry_price': entry_price,
            'stop_loss': entry_data['stop_loss'],
            'position_type': 'long',
            'entry_date': datetime.now() - timedelta(days=10),
            'profit_targets': targets
        }
        
        monitoring = profit_strategy.monitor_profit_targets(position_data, current_price, df)
        print(profit_strategy.generate_profit_report(targets, monitoring))

if __name__ == "__main__":
    main()