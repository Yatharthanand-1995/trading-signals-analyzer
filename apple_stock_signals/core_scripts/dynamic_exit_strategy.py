#!/usr/bin/env python3
"""
Dynamic Exit Strategy System
Implements intelligent exit strategies based on market conditions and trade performance
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from typing import Dict, List, Tuple, Optional

class DynamicExitStrategy:
    """
    Manages dynamic exit strategies including:
    - Partial profit taking at multiple levels
    - Time-based exits
    - Volatility-adjusted targets
    - Support/resistance based exits
    - Pattern-based exits
    - Market regime adaptive exits
    """
    
    def __init__(self):
        self.exit_configs = {
            'partial_take_profits': [
                {'level': 1.0, 'percentage': 0.33},  # Take 33% at 1R
                {'level': 2.0, 'percentage': 0.33},  # Take 33% at 2R
                {'level': 3.0, 'percentage': 0.34}   # Take final 34% at 3R
            ],
            'time_limits': {
                'max_days': 15,          # Maximum holding period
                'review_days': 5,        # Review if no movement
                'breakeven_days': 3      # Days to move stop to breakeven
            },
            'volatility_adjustments': {
                'low': 0.8,      # Tighter targets in low volatility
                'normal': 1.0,   # Standard targets
                'high': 1.5      # Wider targets in high volatility
            }
        }
        
    def calculate_exit_strategy(self, position: Dict, market_data: pd.DataFrame, 
                              market_regime: str = 'NEUTRAL') -> Dict:
        """
        Calculate comprehensive exit strategy for a position
        """
        symbol = position['symbol']
        entry_price = position['entry_price']
        entry_date = position.get('entry_date', datetime.now() - timedelta(days=1))
        stop_loss = position['stop_loss']
        position_type = position.get('position_type', 'long')
        current_price = market_data['Close'].iloc[-1]
        
        # Calculate R-value (risk unit)
        r_value = abs(entry_price - stop_loss)
        
        # Generate exit plan
        exit_plan = {
            'symbol': symbol,
            'current_price': current_price,
            'entry_price': entry_price,
            'position_pnl': self._calculate_pnl(entry_price, current_price, position_type),
            'days_in_trade': (datetime.now() - entry_date).days,
            'exit_levels': {},
            'recommendations': []
        }
        
        # 1. Calculate partial profit levels
        profit_levels = self._calculate_profit_levels(
            entry_price, r_value, market_data, position_type, market_regime
        )
        exit_plan['profit_levels'] = profit_levels
        
        # 2. Check time-based exits
        time_exit = self._check_time_based_exit(
            entry_date, current_price, entry_price, position_type
        )
        exit_plan['time_based_exit'] = time_exit
        
        # 3. Support/Resistance exits
        sr_exits = self._calculate_support_resistance_exits(
            market_data, current_price, position_type
        )
        exit_plan['support_resistance_exits'] = sr_exits
        
        # 4. Pattern-based exits
        pattern_exits = self._detect_exit_patterns(market_data, position_type)
        exit_plan['pattern_exits'] = pattern_exits
        
        # 5. Volatility-based adjustments
        volatility_adjustments = self._calculate_volatility_adjustments(
            market_data, profit_levels, position_type
        )
        exit_plan['volatility_adjusted_targets'] = volatility_adjustments
        
        # 6. Generate exit recommendations
        exit_plan['recommendations'] = self._generate_exit_recommendations(
            exit_plan, position, market_regime
        )
        
        # 7. Calculate optimal exit action
        exit_plan['optimal_action'] = self._determine_optimal_action(exit_plan, position)
        
        return exit_plan
    
    def _calculate_pnl(self, entry_price: float, current_price: float, position_type: str) -> Dict:
        """Calculate position P&L"""
        if position_type == 'long':
            pnl_points = current_price - entry_price
            pnl_percentage = (pnl_points / entry_price) * 100
        else:  # short
            pnl_points = entry_price - current_price
            pnl_percentage = (pnl_points / entry_price) * 100
        
        return {
            'points': pnl_points,
            'percentage': pnl_percentage,
            'status': 'profit' if pnl_percentage > 0 else 'loss'
        }
    
    def _calculate_profit_levels(self, entry_price: float, r_value: float, 
                               market_data: pd.DataFrame, position_type: str,
                               market_regime: str) -> List[Dict]:
        """Calculate partial profit taking levels"""
        profit_levels = []
        
        # Get volatility adjustment
        volatility = self._calculate_current_volatility(market_data)
        vol_adjustment = self._get_volatility_adjustment(volatility)
        
        # Regime adjustments
        regime_multipliers = {
            'STRONG_BULL': 1.2,   # Let winners run
            'BULL': 1.1,
            'NEUTRAL': 1.0,
            'BEAR': 0.9,          # Take profits quicker
            'STRONG_BEAR': 0.8,
            'HIGH_VOLATILITY': 1.3  # Wider targets
        }
        regime_mult = regime_multipliers.get(market_regime, 1.0)
        
        for i, config in enumerate(self.exit_configs['partial_take_profits']):
            r_level = config['level']
            percentage = config['percentage']
            
            # Calculate target price
            adjusted_r = r_level * vol_adjustment * regime_mult
            
            if position_type == 'long':
                target_price = entry_price + (r_value * adjusted_r)
            else:  # short
                target_price = entry_price - (r_value * adjusted_r)
            
            # Check if already hit
            if position_type == 'long':
                hit = market_data['High'].max() >= target_price
            else:
                hit = market_data['Low'].min() <= target_price
            
            profit_levels.append({
                'level': i + 1,
                'r_multiple': adjusted_r,
                'target_price': target_price,
                'percentage_to_exit': percentage * 100,
                'already_hit': hit,
                'distance_from_current': abs(market_data['Close'].iloc[-1] - target_price),
                'distance_percentage': abs(market_data['Close'].iloc[-1] - target_price) / market_data['Close'].iloc[-1] * 100
            })
        
        return profit_levels
    
    def _check_time_based_exit(self, entry_date: datetime, current_price: float, 
                              entry_price: float, position_type: str) -> Dict:
        """Check for time-based exit conditions"""
        days_in_trade = (datetime.now() - entry_date).days
        
        time_exit = {
            'days_in_trade': days_in_trade,
            'max_days_reached': days_in_trade >= self.exit_configs['time_limits']['max_days'],
            'review_needed': days_in_trade >= self.exit_configs['time_limits']['review_days'],
            'move_to_breakeven': days_in_trade >= self.exit_configs['time_limits']['breakeven_days']
        }
        
        # Calculate time decay factor
        if days_in_trade > 10:
            time_decay = 1 - (days_in_trade - 10) / 20  # Decay over 20 days
            time_exit['time_decay_factor'] = max(0.5, time_decay)
        else:
            time_exit['time_decay_factor'] = 1.0
        
        # Check if position is stagnant
        if position_type == 'long':
            price_movement = ((current_price - entry_price) / entry_price) * 100
        else:
            price_movement = ((entry_price - current_price) / entry_price) * 100
        
        if abs(price_movement) < 1 and days_in_trade >= 5:
            time_exit['stagnant'] = True
            time_exit['recommendation'] = "Position stagnant - consider exiting to free capital"
        else:
            time_exit['stagnant'] = False
        
        return time_exit
    
    def _calculate_support_resistance_exits(self, market_data: pd.DataFrame, 
                                          current_price: float, position_type: str) -> Dict:
        """Calculate exits based on support/resistance levels"""
        # Calculate key levels
        lookback = min(50, len(market_data))
        recent_data = market_data.tail(lookback)
        
        # Find swing highs and lows
        swing_highs = []
        swing_lows = []
        
        for i in range(2, len(recent_data) - 2):
            # Swing high: higher than 2 candles on each side
            if (recent_data['High'].iloc[i] > recent_data['High'].iloc[i-1] and
                recent_data['High'].iloc[i] > recent_data['High'].iloc[i-2] and
                recent_data['High'].iloc[i] > recent_data['High'].iloc[i+1] and
                recent_data['High'].iloc[i] > recent_data['High'].iloc[i+2]):
                swing_highs.append(recent_data['High'].iloc[i])
            
            # Swing low: lower than 2 candles on each side
            if (recent_data['Low'].iloc[i] < recent_data['Low'].iloc[i-1] and
                recent_data['Low'].iloc[i] < recent_data['Low'].iloc[i-2] and
                recent_data['Low'].iloc[i] < recent_data['Low'].iloc[i+1] and
                recent_data['Low'].iloc[i] < recent_data['Low'].iloc[i+2]):
                swing_lows.append(recent_data['Low'].iloc[i])
        
        # Get nearest levels
        sr_exits = {
            'swing_highs': sorted(swing_highs)[-3:] if swing_highs else [],
            'swing_lows': sorted(swing_lows)[:3] if swing_lows else [],
            'next_resistance': None,
            'next_support': None
        }
        
        # Find next resistance/support
        if position_type == 'long' and swing_highs:
            resistances = [h for h in swing_highs if h > current_price]
            if resistances:
                sr_exits['next_resistance'] = min(resistances)
                sr_exits['resistance_distance'] = ((sr_exits['next_resistance'] - current_price) / current_price) * 100
        
        if swing_lows:
            supports = [l for l in swing_lows if l < current_price]
            if supports:
                sr_exits['next_support'] = max(supports)
                sr_exits['support_distance'] = ((current_price - sr_exits['next_support']) / current_price) * 100
        
        # Psychological levels
        psychological_levels = self._find_psychological_levels(current_price)
        sr_exits['psychological_levels'] = psychological_levels
        
        return sr_exits
    
    def _detect_exit_patterns(self, market_data: pd.DataFrame, position_type: str) -> Dict:
        """Detect chart patterns that suggest exit"""
        patterns = {
            'detected_patterns': [],
            'exit_signals': []
        }
        
        # Recent price action (last 10 days)
        recent = market_data.tail(10)
        
        # 1. Exhaustion patterns
        if self._detect_exhaustion(recent, position_type):
            patterns['detected_patterns'].append('exhaustion')
            patterns['exit_signals'].append({
                'pattern': 'Exhaustion detected',
                'confidence': 'HIGH',
                'action': 'Consider taking profits'
            })
        
        # 2. Reversal patterns
        reversal = self._detect_reversal_pattern(recent, position_type)
        if reversal:
            patterns['detected_patterns'].append(reversal)
            patterns['exit_signals'].append({
                'pattern': f'{reversal} reversal pattern',
                'confidence': 'MEDIUM',
                'action': 'Monitor closely for exit'
            })
        
        # 3. Momentum divergence
        if self._detect_momentum_divergence(market_data, position_type):
            patterns['detected_patterns'].append('momentum_divergence')
            patterns['exit_signals'].append({
                'pattern': 'Momentum divergence',
                'confidence': 'MEDIUM',
                'action': 'Consider reducing position'
            })
        
        # 4. Volume climax
        if self._detect_volume_climax(market_data, position_type):
            patterns['detected_patterns'].append('volume_climax')
            patterns['exit_signals'].append({
                'pattern': 'Volume climax',
                'confidence': 'HIGH',
                'action': 'Take partial profits'
            })
        
        return patterns
    
    def _calculate_volatility_adjustments(self, market_data: pd.DataFrame, 
                                        profit_levels: List[Dict], 
                                        position_type: str) -> Dict:
        """Adjust targets based on current volatility"""
        current_volatility = self._calculate_current_volatility(market_data)
        historical_volatility = self._calculate_historical_volatility(market_data)
        
        # Compare current to historical
        vol_ratio = current_volatility / historical_volatility if historical_volatility > 0 else 1.0
        
        adjustments = {
            'current_volatility': current_volatility,
            'historical_volatility': historical_volatility,
            'volatility_ratio': vol_ratio,
            'adjustment_factor': 1.0,
            'adjusted_targets': []
        }
        
        # Determine adjustment
        if vol_ratio > 1.5:  # Much higher volatility
            adjustments['adjustment_factor'] = 1.3
            adjustments['recommendation'] = "High volatility - targets widened by 30%"
        elif vol_ratio > 1.2:
            adjustments['adjustment_factor'] = 1.15
            adjustments['recommendation'] = "Elevated volatility - targets widened by 15%"
        elif vol_ratio < 0.8:
            adjustments['adjustment_factor'] = 0.85
            adjustments['recommendation'] = "Low volatility - targets tightened by 15%"
        else:
            adjustments['recommendation'] = "Normal volatility - standard targets"
        
        # Apply adjustments to profit levels
        for level in profit_levels:
            adjusted_target = level['target_price']
            if position_type == 'long':
                adjusted_target = level['target_price'] + (
                    (level['target_price'] - market_data['Close'].iloc[-1]) * 
                    (adjustments['adjustment_factor'] - 1)
                )
            else:
                adjusted_target = level['target_price'] - (
                    (market_data['Close'].iloc[-1] - level['target_price']) * 
                    (adjustments['adjustment_factor'] - 1)
                )
            
            adjustments['adjusted_targets'].append({
                'level': level['level'],
                'original_target': level['target_price'],
                'adjusted_target': adjusted_target,
                'difference': abs(adjusted_target - level['target_price'])
            })
        
        return adjustments
    
    def _generate_exit_recommendations(self, exit_plan: Dict, position: Dict, 
                                     market_regime: str) -> List[Dict]:
        """Generate specific exit recommendations"""
        recommendations = []
        current_pnl = exit_plan['position_pnl']['percentage']
        
        # 1. Time-based recommendations
        if exit_plan['time_based_exit']['max_days_reached']:
            recommendations.append({
                'priority': 'HIGH',
                'action': 'EXIT_FULL',
                'reason': 'Maximum holding period reached',
                'confidence': 'HIGH'
            })
        elif exit_plan['time_based_exit']['stagnant']:
            recommendations.append({
                'priority': 'MEDIUM',
                'action': 'EXIT_PARTIAL',
                'reason': 'Position stagnant - opportunity cost',
                'confidence': 'MEDIUM'
            })
        
        # 2. Profit level recommendations
        for level in exit_plan['profit_levels']:
            if not level['already_hit'] and level['distance_percentage'] < 1:
                recommendations.append({
                    'priority': 'HIGH',
                    'action': f'TAKE_PROFIT_{level["level"]}',
                    'reason': f'Near {level["r_multiple"]:.1f}R target',
                    'percentage': level['percentage_to_exit'],
                    'confidence': 'HIGH'
                })
        
        # 3. Pattern-based recommendations
        for signal in exit_plan['pattern_exits']['exit_signals']:
            recommendations.append({
                'priority': 'MEDIUM' if signal['confidence'] == 'MEDIUM' else 'HIGH',
                'action': 'EXIT_PARTIAL' if signal['confidence'] == 'MEDIUM' else 'EXIT_FULL',
                'reason': signal['pattern'],
                'confidence': signal['confidence']
            })
        
        # 4. Support/Resistance recommendations
        sr = exit_plan['support_resistance_exits']
        if sr.get('next_resistance') and sr.get('resistance_distance', 100) < 1:
            recommendations.append({
                'priority': 'MEDIUM',
                'action': 'TAKE_PROFIT_PARTIAL',
                'reason': f'At resistance level ${sr["next_resistance"]:.2f}',
                'confidence': 'MEDIUM'
            })
        
        # 5. Regime-based recommendations
        if market_regime in ['BEAR', 'STRONG_BEAR'] and current_pnl > 5:
            recommendations.append({
                'priority': 'HIGH',
                'action': 'TAKE_PROFIT_AGGRESSIVE',
                'reason': 'Bear market - secure profits',
                'confidence': 'HIGH'
            })
        elif market_regime == 'HIGH_VOLATILITY' and current_pnl > 10:
            recommendations.append({
                'priority': 'MEDIUM',
                'action': 'REDUCE_POSITION',
                'reason': 'High volatility - reduce exposure',
                'confidence': 'MEDIUM'
            })
        
        # Sort by priority
        priority_order = {'HIGH': 1, 'MEDIUM': 2, 'LOW': 3}
        recommendations.sort(key=lambda x: priority_order.get(x['priority'], 4))
        
        return recommendations
    
    def _determine_optimal_action(self, exit_plan: Dict, position: Dict) -> Dict:
        """Determine the single best action to take"""
        recommendations = exit_plan['recommendations']
        current_pnl = exit_plan['position_pnl']['percentage']
        
        # Default action
        optimal_action = {
            'action': 'HOLD',
            'reason': 'No exit signals triggered',
            'urgency': 'LOW'
        }
        
        # Override based on recommendations
        if recommendations:
            top_rec = recommendations[0]  # Already sorted by priority
            
            action_map = {
                'EXIT_FULL': {'action': 'EXIT_100%', 'urgency': 'HIGH'},
                'EXIT_PARTIAL': {'action': 'EXIT_50%', 'urgency': 'MEDIUM'},
                'TAKE_PROFIT_1': {'action': 'EXIT_33%', 'urgency': 'MEDIUM'},
                'TAKE_PROFIT_2': {'action': 'EXIT_33%', 'urgency': 'MEDIUM'},
                'TAKE_PROFIT_3': {'action': 'EXIT_34%', 'urgency': 'MEDIUM'},
                'TAKE_PROFIT_PARTIAL': {'action': 'EXIT_25%', 'urgency': 'LOW'},
                'REDUCE_POSITION': {'action': 'EXIT_50%', 'urgency': 'MEDIUM'},
                'TAKE_PROFIT_AGGRESSIVE': {'action': 'EXIT_75%', 'urgency': 'HIGH'}
            }
            
            if top_rec['action'] in action_map:
                optimal_action = {
                    'action': action_map[top_rec['action']]['action'],
                    'reason': top_rec['reason'],
                    'urgency': action_map[top_rec['action']]['urgency'],
                    'confidence': top_rec['confidence']
                }
        
        # Special cases
        if current_pnl < -10:  # Large loss
            optimal_action = {
                'action': 'REVIEW_STOP',
                'reason': 'Large unrealized loss - review thesis',
                'urgency': 'HIGH'
            }
        elif current_pnl > 50:  # Huge gain
            optimal_action = {
                'action': 'EXIT_75%',
                'reason': 'Exceptional gains - lock in profits',
                'urgency': 'HIGH'
            }
        
        return optimal_action
    
    # Helper methods
    def _calculate_current_volatility(self, data: pd.DataFrame) -> float:
        """Calculate current volatility"""
        returns = data['Close'].pct_change().tail(20)
        return returns.std() * np.sqrt(252)
    
    def _calculate_historical_volatility(self, data: pd.DataFrame) -> float:
        """Calculate historical average volatility"""
        returns = data['Close'].pct_change()
        return returns.std() * np.sqrt(252)
    
    def _get_volatility_adjustment(self, volatility: float) -> float:
        """Get adjustment factor based on volatility"""
        if volatility < 0.15:
            return self.exit_configs['volatility_adjustments']['low']
        elif volatility < 0.30:
            return self.exit_configs['volatility_adjustments']['normal']
        else:
            return self.exit_configs['volatility_adjustments']['high']
    
    def _find_psychological_levels(self, price: float) -> List[float]:
        """Find nearby psychological price levels"""
        levels = []
        
        # Round numbers
        round_100 = round(price / 100) * 100
        round_50 = round(price / 50) * 50
        round_10 = round(price / 10) * 10
        
        for level in [round_100, round_50, round_10]:
            if 0.95 * price <= level <= 1.05 * price:  # Within 5% of current price
                levels.append(level)
        
        return sorted(list(set(levels)))
    
    def _detect_exhaustion(self, data: pd.DataFrame, position_type: str) -> bool:
        """Detect exhaustion patterns"""
        if len(data) < 5:
            return False
        
        # Long exhaustion: series of smaller bodies at highs
        if position_type == 'long':
            recent_bodies = abs(data['Close'] - data['Open']).tail(3)
            recent_ranges = data['High'] - data['Low']
            body_ratio = recent_bodies / recent_ranges
            
            # Small bodies relative to range
            if (body_ratio < 0.3).all() and data['High'].iloc[-1] == data['High'].max():
                return True
        
        return False
    
    def _detect_reversal_pattern(self, data: pd.DataFrame, position_type: str) -> Optional[str]:
        """Detect reversal candlestick patterns"""
        if len(data) < 3:
            return None
        
        last = data.iloc[-1]
        prev = data.iloc[-2]
        
        # Bearish reversal (for longs)
        if position_type == 'long':
            # Shooting star
            body = abs(last['Close'] - last['Open'])
            upper_shadow = last['High'] - max(last['Close'], last['Open'])
            if upper_shadow > body * 2 and last['High'] > prev['High']:
                return 'shooting_star'
            
            # Bearish engulfing
            if (prev['Close'] > prev['Open'] and 
                last['Open'] > prev['Close'] and 
                last['Close'] < prev['Open']):
                return 'bearish_engulfing'
        
        return None
    
    def _detect_momentum_divergence(self, data: pd.DataFrame, position_type: str) -> bool:
        """Detect price/momentum divergence"""
        if len(data) < 20:
            return False
        
        # Calculate RSI
        close = data['Close']
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # Check for divergence
        if position_type == 'long':
            # Bearish divergence: higher price, lower RSI
            if (close.iloc[-1] > close.iloc[-10] and 
                rsi.iloc[-1] < rsi.iloc[-10]):
                return True
        
        return False
    
    def _detect_volume_climax(self, data: pd.DataFrame, position_type: str) -> bool:
        """Detect volume climax"""
        if len(data) < 20:
            return False
        
        current_volume = data['Volume'].iloc[-1]
        avg_volume = data['Volume'].tail(20).mean()
        
        # Volume spike with reversal
        if current_volume > avg_volume * 2.5:
            if position_type == 'long' and data['Close'].iloc[-1] < data['Open'].iloc[-1]:
                return True  # High volume down day
        
        return False


def main():
    """Test the dynamic exit strategy system"""
    import yfinance as yf
    
    # Initialize exit manager
    exit_manager = DynamicExitStrategy()
    
    print("🎯 DYNAMIC EXIT STRATEGY DEMO")
    print("="*80)
    
    # Test position
    position = {
        'symbol': 'AAPL',
        'entry_price': 210.0,
        'entry_date': datetime.now() - timedelta(days=7),
        'stop_loss': 205.0,
        'position_type': 'long',
        'shares': 100
    }
    
    # Fetch market data
    ticker = yf.Ticker(position['symbol'])
    market_data = ticker.history(period='3mo')
    
    # Test with different market regimes
    for regime in ['STRONG_BULL', 'NEUTRAL', 'HIGH_VOLATILITY']:
        print(f"\n📊 Market Regime: {regime}")
        print("-"*60)
        
        exit_plan = exit_manager.calculate_exit_strategy(
            position, market_data, regime
        )
        
        print(f"Position P&L: {exit_plan['position_pnl']['percentage']:.2f}%")
        print(f"Days in Trade: {exit_plan['days_in_trade']}")
        
        print("\n🎯 Profit Targets:")
        for level in exit_plan['profit_levels']:
            status = "✅ HIT" if level['already_hit'] else f"📍 ${level['target_price']:.2f}"
            print(f"  Level {level['level']} ({level['r_multiple']:.1f}R): {status} - Exit {level['percentage_to_exit']:.0f}%")
        
        print("\n📈 Exit Recommendations:")
        for rec in exit_plan['recommendations'][:3]:  # Top 3
            print(f"  [{rec['priority']}] {rec['reason']}")
            print(f"    Action: {rec['action']} (Confidence: {rec['confidence']})")
        
        print(f"\n✅ Optimal Action: {exit_plan['optimal_action']['action']}")
        print(f"   Reason: {exit_plan['optimal_action']['reason']}")
        print(f"   Urgency: {exit_plan['optimal_action']['urgency']}")
    
    # Save example
    import os
    os.makedirs('outputs/exit_strategies', exist_ok=True)
    
    with open('outputs/exit_strategies/example_exit_plan.json', 'w') as f:
        json.dump(exit_plan, f, indent=2, default=str)
    
    print("\n💾 Exit plan saved to outputs/exit_strategies/example_exit_plan.json")


if __name__ == "__main__":
    main()