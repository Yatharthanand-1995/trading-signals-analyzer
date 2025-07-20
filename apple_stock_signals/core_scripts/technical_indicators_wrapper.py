#!/usr/bin/env python3
"""
Technical Indicators Wrapper
Provides a clean interface for technical analysis calculations
"""

import pandas as pd
import numpy as np
from datetime import datetime

class TechnicalIndicatorsWrapper:
    """Wrapper for technical indicator calculations"""
    
    def __init__(self):
        self.indicators = {}
    
    def calculate_rsi(self, prices, period=14):
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    def calculate_bollinger_bands(self, prices, period=20, std_dev=2):
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return {
            'upper': upper,
            'middle': sma,
            'lower': lower
        }
    
    def calculate_stochastic(self, high, low, close, period=14, smooth=3):
        """Calculate Stochastic Oscillator"""
        lowest_low = low.rolling(window=period).min()
        highest_high = high.rolling(window=period).max()
        
        k_percent = ((close - lowest_low) / (highest_high - lowest_low)) * 100
        d_percent = k_percent.rolling(window=smooth).mean()
        
        return {
            'k': k_percent,
            'd': d_percent
        }
    
    def calculate_atr(self, high, low, close, period=14):
        """Calculate Average True Range"""
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    def calculate_all_indicators(self, df):
        """Calculate all technical indicators"""
        
        # Ensure we have required columns
        if not all(col in df.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume']):
            return None
        
        try:
            # Price-based indicators
            self.indicators['RSI'] = self.calculate_rsi(df['Close'])
            
            # MACD
            macd_data = self.calculate_macd(df['Close'])
            self.indicators['MACD'] = macd_data['macd']
            self.indicators['MACD_Signal'] = macd_data['signal']
            self.indicators['MACD_Histogram'] = macd_data['histogram']
            
            # Bollinger Bands
            bb_data = self.calculate_bollinger_bands(df['Close'])
            self.indicators['BB_Upper'] = bb_data['upper']
            self.indicators['BB_Middle'] = bb_data['middle']
            self.indicators['BB_Lower'] = bb_data['lower']
            
            # Stochastic
            stoch_data = self.calculate_stochastic(df['High'], df['Low'], df['Close'])
            self.indicators['Stoch_K'] = stoch_data['k']
            self.indicators['Stoch_D'] = stoch_data['d']
            
            # ATR
            self.indicators['ATR'] = self.calculate_atr(df['High'], df['Low'], df['Close'])
            
            # Moving Averages
            self.indicators['SMA_20'] = df['Close'].rolling(window=20).mean()
            self.indicators['SMA_50'] = df['Close'].rolling(window=50).mean()
            self.indicators['EMA_12'] = df['Close'].ewm(span=12).mean()
            self.indicators['EMA_26'] = df['Close'].ewm(span=26).mean()
            
            # Volume indicators
            self.indicators['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
            
            # Add current values
            current_idx = len(df) - 1
            self.indicators['current_price'] = df['Close'].iloc[-1]
            self.indicators['current_volume'] = df['Volume'].iloc[-1]
            
            return self.indicators
            
        except Exception as e:
            print(f"Error calculating indicators: {str(e)}")
            return None
    
    def get_current_values(self):
        """Get current indicator values"""
        if not self.indicators:
            return {}
        
        current_values = {}
        for key, value in self.indicators.items():
            if isinstance(value, pd.Series) and len(value) > 0:
                current_values[key] = value.iloc[-1] if not pd.isna(value.iloc[-1]) else 0
            else:
                current_values[key] = value
        
        return current_values
    
    def get_indicator_signals(self):
        """Generate trading signals from indicators"""
        if not self.indicators:
            return {'signal': 'NEUTRAL', 'strength': 0}
        
        buy_signals = 0
        sell_signals = 0
        
        current = self.get_current_values()
        
        # RSI signals
        if 'RSI' in current:
            if current['RSI'] < 30:
                buy_signals += 2
            elif current['RSI'] > 70:
                sell_signals += 2
        
        # MACD signals
        if 'MACD' in current and 'MACD_Signal' in current:
            if current['MACD'] > current['MACD_Signal']:
                buy_signals += 1
            else:
                sell_signals += 1
        
        # Bollinger Bands
        if all(k in current for k in ['BB_Upper', 'BB_Lower', 'current_price']):
            if current['current_price'] < current['BB_Lower']:
                buy_signals += 1
            elif current['current_price'] > current['BB_Upper']:
                sell_signals += 1
        
        # Stochastic
        if 'Stoch_K' in current and 'Stoch_D' in current:
            if current['Stoch_K'] < 20:
                buy_signals += 1
            elif current['Stoch_K'] > 80:
                sell_signals += 1
        
        # Moving averages
        if all(k in current for k in ['SMA_20', 'SMA_50', 'current_price']):
            if current['current_price'] > current['SMA_20'] > current['SMA_50']:
                buy_signals += 2
            elif current['current_price'] < current['SMA_20'] < current['SMA_50']:
                sell_signals += 2
        
        # Determine overall signal
        total_signals = buy_signals + sell_signals
        if total_signals == 0:
            return {'signal': 'NEUTRAL', 'strength': 0}
        
        if buy_signals > sell_signals:
            strength = (buy_signals / total_signals) * 100
            signal = 'BUY' if strength >= 60 else 'WEAK_BUY'
        else:
            strength = (sell_signals / total_signals) * 100
            signal = 'SELL' if strength >= 60 else 'WEAK_SELL'
        
        return {
            'signal': signal,
            'strength': strength,
            'buy_signals': buy_signals,
            'sell_signals': sell_signals
        }


# Compatibility class for AppleTechnicalAnalyzer
class AppleTechnicalAnalyzer(TechnicalIndicatorsWrapper):
    """Compatibility wrapper for legacy code"""
    
    def __init__(self):
        super().__init__()
    
    def calculate_all_indicators(self, df):
        """Override to match expected interface"""
        indicators = super().calculate_all_indicators(df)
        
        if indicators is None:
            # Return a safe default
            return {
                'historical_data': df,
                'current_price': df['Close'].iloc[-1] if len(df) > 0 else 0,
                'indicators': {},
                'signal': 'NEUTRAL'
            }
        
        # Format for compatibility
        return {
            'historical_data': df,
            'current_price': indicators.get('current_price', 0),
            'indicators': indicators,
            'signal': self.get_indicator_signals()['signal']
        }