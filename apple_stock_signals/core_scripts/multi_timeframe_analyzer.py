#!/usr/bin/env python3
"""
Multi-Timeframe Analysis Module
Analyzes stocks across multiple timeframes for better entry/exit timing
Optimized for 2-15 day swing trading
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class MultiTimeframeAnalyzer:
    """
    Analyzes stocks across daily, 4-hour, and weekly timeframes
    """
    
    def __init__(self):
        self.timeframes = {
            'weekly': {'period': '6mo', 'interval': '1wk'},
            'daily': {'period': '3mo', 'interval': '1d'},
            '4hour': {'period': '1mo', 'interval': '1h'}  # Using 1h as proxy for 4h
        }
        
    def fetch_multi_timeframe_data(self, symbol):
        """Fetch data for all timeframes"""
        data = {}
        ticker = yf.Ticker(symbol)
        
        for tf_name, tf_params in self.timeframes.items():
            try:
                df = ticker.history(period=tf_params['period'], interval=tf_params['interval'])
                if not df.empty:
                    # For 4-hour, resample hourly to 4-hour
                    if tf_name == '4hour':
                        df = df.resample('4H').agg({
                            'Open': 'first',
                            'High': 'max',
                            'Low': 'min',
                            'Close': 'last',
                            'Volume': 'sum'
                        }).dropna()
                    data[tf_name] = df
            except Exception as e:
                print(f"Error fetching {tf_name} data for {symbol}: {e}")
                
        return data
    
    def calculate_trend_strength(self, df, period=20):
        """Calculate trend strength using ADX"""
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        # Calculate directional movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        
        pos_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0), index=df.index)
        neg_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0), index=df.index)
        
        pos_di = 100 * (pos_dm.rolling(window=period).mean() / atr)
        neg_di = 100 * (neg_dm.rolling(window=period).mean() / atr)
        
        dx = 100 * abs(pos_di - neg_di) / (pos_di + neg_di)
        adx = dx.rolling(window=period).mean()
        
        return {
            'adx': adx.iloc[-1] if len(adx) > 0 else 0,
            'trend_direction': 'bullish' if pos_di.iloc[-1] > neg_di.iloc[-1] else 'bearish',
            'trend_strength': 'strong' if adx.iloc[-1] > 25 else 'weak'
        }
    
    def identify_support_resistance(self, df, lookback=20):
        """Identify key support and resistance levels"""
        highs = df['High'].rolling(window=lookback).max()
        lows = df['Low'].rolling(window=lookback).min()
        
        # Find pivot points
        pivot = (df['High'] + df['Low'] + df['Close']) / 3
        
        # Calculate support and resistance levels
        resistance_1 = 2 * pivot - df['Low']
        support_1 = 2 * pivot - df['High']
        resistance_2 = pivot + (df['High'] - df['Low'])
        support_2 = pivot - (df['High'] - df['Low'])
        
        current_price = df['Close'].iloc[-1]
        
        return {
            'current_price': current_price,
            'resistance_1': resistance_1.iloc[-1],
            'resistance_2': resistance_2.iloc[-1],
            'support_1': support_1.iloc[-1],
            'support_2': support_2.iloc[-1],
            'nearest_resistance': min([r for r in [resistance_1.iloc[-1], resistance_2.iloc[-1]] if r > current_price], default=resistance_2.iloc[-1]),
            'nearest_support': max([s for s in [support_1.iloc[-1], support_2.iloc[-1]] if s < current_price], default=support_2.iloc[-1])
        }
    
    def calculate_momentum_indicators(self, df):
        """Calculate momentum indicators for each timeframe"""
        close = df['Close']
        
        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = close.ewm(span=12, adjust=False).mean()
        exp2 = close.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        histogram = macd - signal
        
        # Stochastic
        low_14 = df['Low'].rolling(window=14).min()
        high_14 = df['High'].rolling(window=14).max()
        k_percent = 100 * ((close - low_14) / (high_14 - low_14))
        d_percent = k_percent.rolling(window=3).mean()
        
        # Moving averages
        sma_20 = close.rolling(window=20).mean()
        sma_50 = close.rolling(window=50).mean() if len(close) >= 50 else sma_20
        ema_9 = close.ewm(span=9, adjust=False).mean()
        
        return {
            'rsi': rsi.iloc[-1] if len(rsi) > 0 else 50,
            'macd': macd.iloc[-1] if len(macd) > 0 else 0,
            'macd_signal': signal.iloc[-1] if len(signal) > 0 else 0,
            'macd_histogram': histogram.iloc[-1] if len(histogram) > 0 else 0,
            'stoch_k': k_percent.iloc[-1] if len(k_percent) > 0 else 50,
            'stoch_d': d_percent.iloc[-1] if len(d_percent) > 0 else 50,
            'sma_20': sma_20.iloc[-1] if len(sma_20) > 0 else close.iloc[-1],
            'sma_50': sma_50.iloc[-1] if len(sma_50) > 0 else close.iloc[-1],
            'ema_9': ema_9.iloc[-1] if len(ema_9) > 0 else close.iloc[-1],
            'price_vs_sma20': ((close.iloc[-1] - sma_20.iloc[-1]) / sma_20.iloc[-1] * 100) if len(sma_20) > 0 else 0,
            'price_vs_sma50': ((close.iloc[-1] - sma_50.iloc[-1]) / sma_50.iloc[-1] * 100) if len(sma_50) > 0 else 0
        }
    
    def analyze_timeframe_alignment(self, mtf_data):
        """Check if all timeframes are aligned for a trade"""
        alignment_score = 0
        signals = {}
        
        for timeframe, data in mtf_data.items():
            if timeframe == 'indicators':
                continue
                
            indicators = data['indicators']
            trend = data['trend']
            
            # Trend alignment
            if trend['trend_direction'] == 'bullish':
                alignment_score += 1
                signals[f'{timeframe}_trend'] = 'bullish'
            else:
                signals[f'{timeframe}_trend'] = 'bearish'
            
            # Momentum alignment
            if indicators['rsi'] > 50 and indicators['macd'] > indicators['macd_signal']:
                alignment_score += 1
                signals[f'{timeframe}_momentum'] = 'bullish'
            elif indicators['rsi'] < 50 and indicators['macd'] < indicators['macd_signal']:
                signals[f'{timeframe}_momentum'] = 'bearish'
            else:
                signals[f'{timeframe}_momentum'] = 'neutral'
            
            # Price vs moving averages
            if indicators['price_vs_sma20'] > 0 and indicators['price_vs_sma50'] > 0:
                alignment_score += 1
                signals[f'{timeframe}_ma'] = 'above'
            else:
                signals[f'{timeframe}_ma'] = 'below'
        
        # Calculate final alignment
        max_score = len(mtf_data) * 3  # 3 checks per timeframe
        alignment_percentage = (alignment_score / max_score) * 100
        
        return {
            'alignment_score': alignment_score,
            'alignment_percentage': alignment_percentage,
            'is_aligned': alignment_percentage >= 70,  # 70% threshold
            'signals': signals,
            'recommendation': self._get_recommendation(alignment_percentage, signals)
        }
    
    def _get_recommendation(self, alignment_percentage, signals):
        """Generate trading recommendation based on alignment"""
        if alignment_percentage >= 80:
            return 'STRONG_BUY'
        elif alignment_percentage >= 70:
            return 'BUY'
        elif alignment_percentage <= 20:
            return 'STRONG_SELL'
        elif alignment_percentage <= 30:
            return 'SELL'
        else:
            return 'HOLD'
    
    def find_optimal_entry(self, data_4h, data_daily):
        """Find optimal entry point using 4-hour chart within daily trend"""
        daily_trend = self.calculate_trend_strength(data_daily)
        hourly_momentum = self.calculate_momentum_indicators(data_4h)
        
        entry_signals = {
            'daily_trend': daily_trend['trend_direction'],
            'daily_strength': daily_trend['trend_strength'],
            '4h_rsi': hourly_momentum['rsi'],
            '4h_stoch': hourly_momentum['stoch_k'],
            'entry_quality': 'poor'
        }
        
        # Best entry conditions
        if daily_trend['trend_direction'] == 'bullish' and daily_trend['trend_strength'] == 'strong':
            if 30 < hourly_momentum['rsi'] < 50:  # Pullback in uptrend
                entry_signals['entry_quality'] = 'excellent'
            elif 50 < hourly_momentum['rsi'] < 70:
                entry_signals['entry_quality'] = 'good'
            elif hourly_momentum['rsi'] > 70:
                entry_signals['entry_quality'] = 'overbought_wait'
                
        elif daily_trend['trend_direction'] == 'bearish' and daily_trend['trend_strength'] == 'strong':
            if 50 < hourly_momentum['rsi'] < 70:  # Rally in downtrend
                entry_signals['entry_quality'] = 'excellent_short'
            elif 30 < hourly_momentum['rsi'] < 50:
                entry_signals['entry_quality'] = 'good_short'
            elif hourly_momentum['rsi'] < 30:
                entry_signals['entry_quality'] = 'oversold_wait'
        
        return entry_signals
    
    def analyze_stock(self, symbol):
        """Complete multi-timeframe analysis for a stock"""
        print(f"\n🔍 Analyzing {symbol} across multiple timeframes...")
        
        # Fetch data
        data = self.fetch_multi_timeframe_data(symbol)
        if not data:
            return None
        
        # Analyze each timeframe
        analysis = {}
        
        for timeframe, df in data.items():
            if len(df) < 20:  # Need minimum data
                continue
                
            analysis[timeframe] = {
                'trend': self.calculate_trend_strength(df),
                'support_resistance': self.identify_support_resistance(df),
                'indicators': self.calculate_momentum_indicators(df)
            }
        
        # Check timeframe alignment
        if len(analysis) >= 2:  # Need at least 2 timeframes
            analysis['alignment'] = self.analyze_timeframe_alignment(analysis)
            
            # Find optimal entry if we have 4h and daily data
            if '4hour' in data and 'daily' in data:
                analysis['entry_timing'] = self.find_optimal_entry(data['4hour'], data['daily'])
        
        analysis['symbol'] = symbol
        analysis['analysis_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        return analysis
    
    def generate_summary(self, analysis):
        """Generate a summary of the multi-timeframe analysis"""
        if not analysis:
            return "No analysis available"
        
        summary = f"\n📊 Multi-Timeframe Analysis for {analysis['symbol']}\n"
        summary += "=" * 50 + "\n"
        
        # Alignment summary
        if 'alignment' in analysis:
            align = analysis['alignment']
            summary += f"\n🎯 Timeframe Alignment: {align['alignment_percentage']:.1f}%\n"
            summary += f"📈 Recommendation: {align['recommendation']}\n"
            
            # Trend summary
            for tf in ['weekly', 'daily', '4hour']:
                if tf in analysis:
                    trend = analysis[tf]['trend']
                    summary += f"\n{tf.title()} Trend: {trend['trend_direction']} ({trend['trend_strength']})"
        
        # Entry timing
        if 'entry_timing' in analysis:
            entry = analysis['entry_timing']
            summary += f"\n\n⏰ Entry Quality: {entry['entry_quality']}"
            summary += f"\n4H RSI: {entry['4h_rsi']:.1f}"
        
        # Support/Resistance
        if 'daily' in analysis:
            sr = analysis['daily']['support_resistance']
            summary += f"\n\n📊 Key Levels (Daily):"
            summary += f"\nResistance: ${sr['nearest_resistance']:.2f}"
            summary += f"\nCurrent: ${sr['current_price']:.2f}"
            summary += f"\nSupport: ${sr['nearest_support']:.2f}"
        
        return summary


def main():
    """Test the multi-timeframe analyzer"""
    analyzer = MultiTimeframeAnalyzer()
    
    # Test with a few stocks
    test_symbols = ['AAPL', 'TSLA', 'NVDA']
    
    for symbol in test_symbols:
        analysis = analyzer.analyze_stock(symbol)
        if analysis:
            print(analyzer.generate_summary(analysis))
            
            # Show detailed alignment
            if 'alignment' in analysis:
                print(f"\nDetailed Signals:")
                for signal, value in analysis['alignment']['signals'].items():
                    print(f"  {signal}: {value}")

if __name__ == "__main__":
    main()