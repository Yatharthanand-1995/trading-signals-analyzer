#!/usr/bin/env python3
"""
Dynamic Stop Loss System
Adaptive stop losses for 2-15 day swing trading
Includes ATR-based, structure-based, and time-based stops
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

class DynamicStopLossSystem:
    """
    Advanced stop loss management for swing trading
    """
    
    def __init__(self):
        self.config = {
            'initial_stop_atr_multiplier': 2.0,      # Initial stop = 2 ATR
            'trailing_stop_atr_multiplier': 1.5,    # Trailing stop = 1.5 ATR
            'breakeven_trigger_r': 1.0,             # Move to breakeven at 1R profit
            'tight_stop_trigger_r': 2.0,            # Tighten stop at 2R profit
            'time_stop_days': 10,                   # Exit if no progress in 10 days
            'max_hold_days': 15,                    # Maximum holding period
            'volatility_adjustment': True,          # Adjust stops for market volatility
            'structure_buffer_pct': 0.002           # 0.2% buffer below support
        }
        
    def calculate_atr(self, df, period=14):
        """Calculate Average True Range"""
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    def calculate_initial_stop(self, entry_price, df, position_type='long'):
        """Calculate initial stop loss placement"""
        atr = self.calculate_atr(df).iloc[-1]
        
        # Adjust ATR multiplier based on volatility regime
        volatility_regime = self.detect_volatility_regime(df)
        atr_multiplier = self.config['initial_stop_atr_multiplier']
        
        if volatility_regime == 'high':
            atr_multiplier *= 1.25  # Wider stops in high volatility
        elif volatility_regime == 'low':
            atr_multiplier *= 0.85  # Tighter stops in low volatility
        
        # Calculate ATR-based stop
        atr_stop_distance = atr * atr_multiplier
        
        if position_type == 'long':
            atr_stop = entry_price - atr_stop_distance
            
            # Find structure-based stop (below recent swing low)
            structure_stop = self.find_structure_stop(df, 'long')
            
            # Use the higher stop (less risk)
            initial_stop = max(atr_stop, structure_stop)
        else:
            atr_stop = entry_price + atr_stop_distance
            structure_stop = self.find_structure_stop(df, 'short')
            initial_stop = min(atr_stop, structure_stop)
        
        stop_data = {
            'initial_stop': initial_stop,
            'stop_distance': abs(entry_price - initial_stop),
            'stop_distance_pct': abs(entry_price - initial_stop) / entry_price * 100,
            'atr_stop': atr_stop,
            'structure_stop': structure_stop,
            'stop_type': 'structure' if initial_stop == structure_stop else 'atr',
            'volatility_regime': volatility_regime,
            'atr_multiplier': atr_multiplier
        }
        
        return stop_data
    
    def find_structure_stop(self, df, position_type='long', lookback=20):
        """Find stop based on market structure (swing highs/lows)"""
        if position_type == 'long':
            # Find recent swing lows
            recent_lows = []
            for i in range(len(df) - lookback, len(df) - 2):
                if i > 0 and i < len(df) - 1:
                    if df['Low'].iloc[i] < df['Low'].iloc[i-1] and df['Low'].iloc[i] < df['Low'].iloc[i+1]:
                        recent_lows.append(df['Low'].iloc[i])
            
            if recent_lows:
                # Use the most recent significant low
                structure_level = max(recent_lows)  # Highest of recent lows
            else:
                # Fallback to simple recent low
                structure_level = df['Low'].tail(lookback).min()
            
            # Add small buffer below support
            stop = structure_level * (1 - self.config['structure_buffer_pct'])
            
        else:  # short position
            # Find recent swing highs
            recent_highs = []
            for i in range(len(df) - lookback, len(df) - 2):
                if i > 0 and i < len(df) - 1:
                    if df['High'].iloc[i] > df['High'].iloc[i-1] and df['High'].iloc[i] > df['High'].iloc[i+1]:
                        recent_highs.append(df['High'].iloc[i])
            
            if recent_highs:
                structure_level = min(recent_highs)  # Lowest of recent highs
            else:
                structure_level = df['High'].tail(lookback).max()
            
            # Add small buffer above resistance
            stop = structure_level * (1 + self.config['structure_buffer_pct'])
        
        return stop
    
    def detect_volatility_regime(self, df, period=20):
        """Detect current volatility regime"""
        # Calculate rolling volatility
        returns = df['Close'].pct_change()
        current_vol = returns.tail(period).std() * np.sqrt(252)
        
        # Historical volatility over longer period
        hist_vol = returns.tail(100).std() * np.sqrt(252)
        
        # Volatility percentile
        vol_series = returns.rolling(period).std() * np.sqrt(252)
        vol_percentile = (vol_series < current_vol).sum() / len(vol_series) * 100
        
        if vol_percentile > 80:
            return 'high'
        elif vol_percentile < 20:
            return 'low'
        else:
            return 'normal'
    
    def calculate_trailing_stop(self, current_price, entry_price, current_stop, df, 
                              position_type='long', entry_date=None):
        """Calculate dynamic trailing stop based on profit and market conditions"""
        # Calculate current profit in R multiples
        initial_risk = abs(entry_price - current_stop)
        current_profit = current_price - entry_price if position_type == 'long' else entry_price - current_price
        r_multiple = current_profit / initial_risk if initial_risk > 0 else 0
        
        # Get current ATR
        atr = self.calculate_atr(df).iloc[-1]
        
        # Initialize new stop at current stop
        new_stop = current_stop
        stop_update_reason = "no_change"
        
        # Breakeven stop
        if r_multiple >= self.config['breakeven_trigger_r'] and current_stop < entry_price:
            if position_type == 'long':
                new_stop = max(current_stop, entry_price + (entry_price * 0.001))  # Small buffer above breakeven
                stop_update_reason = "breakeven"
            else:
                new_stop = min(current_stop, entry_price - (entry_price * 0.001))
                stop_update_reason = "breakeven"
        
        # Trailing stop when in profit
        if r_multiple >= self.config['tight_stop_trigger_r']:
            trailing_distance = atr * self.config['trailing_stop_atr_multiplier']
            
            # Tighten trailing for higher profits
            if r_multiple >= 3:
                trailing_distance *= 0.7  # Tighter stop
            elif r_multiple >= 4:
                trailing_distance *= 0.5  # Very tight stop
            
            if position_type == 'long':
                trailing_stop = current_price - trailing_distance
                if trailing_stop > new_stop:
                    new_stop = trailing_stop
                    stop_update_reason = "trailing"
            else:
                trailing_stop = current_price + trailing_distance
                if trailing_stop < new_stop:
                    new_stop = trailing_stop
                    stop_update_reason = "trailing"
        
        # Structure-based trailing
        structure_stop = self.find_structure_stop(df, position_type)
        if position_type == 'long' and structure_stop > new_stop:
            new_stop = structure_stop
            stop_update_reason = "structure"
        elif position_type == 'short' and structure_stop < new_stop:
            new_stop = structure_stop
            stop_update_reason = "structure"
        
        # Time-based stop check
        time_stop_triggered = False
        if entry_date:
            days_in_trade = (datetime.now() - entry_date).days
            
            # No progress stop
            if days_in_trade >= self.config['time_stop_days'] and r_multiple < 0.5:
                time_stop_triggered = True
                stop_update_reason = "time_stop_no_progress"
            
            # Max holding period
            elif days_in_trade >= self.config['max_hold_days']:
                time_stop_triggered = True
                stop_update_reason = "max_hold_time"
        
        return {
            'new_stop': new_stop,
            'stop_moved': new_stop != current_stop,
            'stop_distance': abs(current_price - new_stop),
            'stop_distance_pct': abs(current_price - new_stop) / current_price * 100,
            'r_multiple': r_multiple,
            'stop_update_reason': stop_update_reason,
            'time_stop_triggered': time_stop_triggered,
            'trailing_distance_atr': trailing_distance / atr if r_multiple >= 2 else None
        }
    
    def calculate_chandelier_stop(self, df, period=22, multiplier=3.0, position_type='long'):
        """Calculate Chandelier Exit stop (alternative trailing method)"""
        atr = self.calculate_atr(df, period)
        
        if position_type == 'long':
            highest_high = df['High'].rolling(window=period).max()
            chandelier_stop = highest_high - (multiplier * atr)
        else:
            lowest_low = df['Low'].rolling(window=period).min()
            chandelier_stop = lowest_low + (multiplier * atr)
        
        return chandelier_stop.iloc[-1]
    
    def get_stop_recommendations(self, position_data, current_df):
        """Get comprehensive stop loss recommendations for a position"""
        entry_price = position_data['entry_price']
        current_price = position_data['current_price']
        current_stop = position_data['current_stop']
        position_type = position_data.get('position_type', 'long')
        entry_date = position_data.get('entry_date')
        
        # Calculate trailing stop
        trailing_data = self.calculate_trailing_stop(
            current_price, entry_price, current_stop, current_df, 
            position_type, entry_date
        )
        
        # Calculate alternative stops
        chandelier_stop = self.calculate_chandelier_stop(current_df, position_type=position_type)
        
        # Determine best stop
        if position_type == 'long':
            recommended_stop = max(trailing_data['new_stop'], chandelier_stop)
        else:
            recommended_stop = min(trailing_data['new_stop'], chandelier_stop)
        
        # Risk analysis
        current_risk_pct = abs(current_price - recommended_stop) / current_price * 100
        original_risk_pct = abs(entry_price - position_data['initial_stop']) / entry_price * 100
        risk_reduced_pct = ((original_risk_pct - current_risk_pct) / original_risk_pct * 100) if original_risk_pct > 0 else 0
        
        recommendations = {
            'recommended_stop': recommended_stop,
            'trailing_stop': trailing_data['new_stop'],
            'chandelier_stop': chandelier_stop,
            'stop_type': trailing_data['stop_update_reason'],
            'current_risk_pct': current_risk_pct,
            'risk_reduced_pct': risk_reduced_pct,
            'r_multiple': trailing_data['r_multiple'],
            'time_stop_triggered': trailing_data['time_stop_triggered'],
            'action': self._determine_action(trailing_data, position_data)
        }
        
        return recommendations
    
    def _determine_action(self, trailing_data, position_data):
        """Determine recommended action based on stop analysis"""
        if trailing_data['time_stop_triggered']:
            return 'EXIT_TIME_STOP'
        elif trailing_data['r_multiple'] >= 4:
            return 'CONSIDER_PARTIAL_EXIT'
        elif trailing_data['stop_moved'] and trailing_data['new_stop'] > position_data['current_stop']:
            return 'RAISE_STOP'
        elif trailing_data['r_multiple'] < -1:
            return 'EXIT_STOP_LOSS'
        else:
            return 'HOLD'
    
    def generate_stop_summary(self, stop_data):
        """Generate human-readable stop loss summary"""
        summary = "\n📊 Stop Loss Analysis\n"
        summary += "=" * 40 + "\n"
        
        if 'initial_stop' in stop_data:
            summary += f"\nInitial Stop Setup:\n"
            summary += f"  Stop Price: ${stop_data['initial_stop']:.2f}\n"
            summary += f"  Stop Distance: {stop_data['stop_distance_pct']:.1f}%\n"
            summary += f"  Stop Type: {stop_data['stop_type']}\n"
            summary += f"  Volatility: {stop_data['volatility_regime']}\n"
        
        if 'recommended_stop' in stop_data:
            summary += f"\nCurrent Stop Recommendation:\n"
            summary += f"  Recommended Stop: ${stop_data['recommended_stop']:.2f}\n"
            summary += f"  Current Risk: {stop_data['current_risk_pct']:.1f}%\n"
            summary += f"  Risk Reduced: {stop_data['risk_reduced_pct']:.1f}%\n"
            summary += f"  R-Multiple: {stop_data['r_multiple']:.2f}R\n"
            summary += f"  Action: {stop_data['action']}\n"
        
        return summary


def main():
    """Test the dynamic stop loss system"""
    import yfinance as yf
    
    # Initialize system
    stop_system = DynamicStopLossSystem()
    
    # Test with a stock
    symbol = 'AAPL'
    stock = yf.Ticker(symbol)
    df = stock.history(period='3mo')
    
    if not df.empty:
        current_price = df['Close'].iloc[-1]
        entry_price = df['Close'].iloc[-10]  # Simulate entry 10 days ago
        
        print(f"🎯 Testing Dynamic Stop Loss for {symbol}")
        print(f"Entry Price: ${entry_price:.2f}")
        print(f"Current Price: ${current_price:.2f}")
        
        # Calculate initial stop
        initial_stop_data = stop_system.calculate_initial_stop(entry_price, df)
        print(stop_system.generate_stop_summary(initial_stop_data))
        
        # Simulate position data
        position_data = {
            'entry_price': entry_price,
            'current_price': current_price,
            'current_stop': initial_stop_data['initial_stop'],
            'initial_stop': initial_stop_data['initial_stop'],
            'position_type': 'long',
            'entry_date': datetime.now() - timedelta(days=10)
        }
        
        # Get stop recommendations
        recommendations = stop_system.get_stop_recommendations(position_data, df)
        print(stop_system.generate_stop_summary(recommendations))

if __name__ == "__main__":
    main()