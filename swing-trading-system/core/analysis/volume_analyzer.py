#!/usr/bin/env python3
"""
Volume Analysis Module
Advanced volume indicators for confirming price movements
Optimized for 2-15 day swing trading
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class VolumeAnalyzer:
    """
    Analyzes volume patterns to confirm price movements and detect institutional activity
    """
    
    def __init__(self):
        self.min_volume_threshold = 100000  # Minimum daily volume
        self.relative_volume_threshold = 1.5  # 150% of average
        
    def calculate_relative_volume(self, df, period=20):
        """Calculate relative volume compared to average"""
        avg_volume = df['Volume'].rolling(window=period).mean()
        relative_volume = df['Volume'] / avg_volume
        
        return {
            'current_volume': df['Volume'].iloc[-1],
            'avg_volume': avg_volume.iloc[-1],
            'relative_volume': relative_volume.iloc[-1],
            'is_high_volume': relative_volume.iloc[-1] > self.relative_volume_threshold,
            'volume_rank': self._calculate_volume_rank(df['Volume'], period)
        }
    
    def _calculate_volume_rank(self, volume, period):
        """Calculate where current volume ranks in recent history"""
        recent_volumes = volume.tail(period)
        current = volume.iloc[-1]
        rank = (recent_volumes < current).sum() / period * 100
        return rank
    
    def calculate_on_balance_volume(self, df):
        """Calculate On Balance Volume (OBV) and its trend"""
        obv = [0]
        for i in range(1, len(df)):
            if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
                obv.append(obv[-1] + df['Volume'].iloc[i])
            elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
                obv.append(obv[-1] - df['Volume'].iloc[i])
            else:
                obv.append(obv[-1])
        
        df['OBV'] = obv
        obv_sma = df['OBV'].rolling(window=10).mean()
        
        # Detect OBV divergence
        price_trend = 'up' if df['Close'].iloc[-1] > df['Close'].iloc[-20] else 'down'
        obv_trend = 'up' if df['OBV'].iloc[-1] > df['OBV'].iloc[-20] else 'down'
        
        divergence = None
        if price_trend == 'up' and obv_trend == 'down':
            divergence = 'bearish_divergence'
        elif price_trend == 'down' and obv_trend == 'up':
            divergence = 'bullish_divergence'
        
        return {
            'obv': df['OBV'].iloc[-1],
            'obv_sma': obv_sma.iloc[-1] if len(obv_sma) > 0 else df['OBV'].iloc[-1],
            'obv_trend': obv_trend,
            'price_trend': price_trend,
            'divergence': divergence,
            'obv_signal': 'bullish' if df['OBV'].iloc[-1] > obv_sma.iloc[-1] else 'bearish'
        }
    
    def calculate_volume_weighted_average_price(self, df, period=20):
        """Calculate VWAP and price position relative to VWAP"""
        df['VWAP'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()
        
        # Calculate standard deviation bands
        df['VWAP_std'] = np.sqrt((df['Volume'] * ((df['High'] + df['Low'] + df['Close']) / 3 - df['VWAP'])**2).rolling(window=period).sum() / df['Volume'].rolling(window=period).sum())
        
        current_price = df['Close'].iloc[-1]
        vwap = df['VWAP'].iloc[-1]
        vwap_std = df['VWAP_std'].iloc[-1] if 'VWAP_std' in df else 0
        
        # Price position relative to VWAP
        price_vs_vwap = ((current_price - vwap) / vwap) * 100
        
        return {
            'vwap': vwap,
            'vwap_upper': vwap + (2 * vwap_std),
            'vwap_lower': vwap - (2 * vwap_std),
            'price_vs_vwap': price_vs_vwap,
            'vwap_signal': self._get_vwap_signal(current_price, vwap, vwap_std)
        }
    
    def _get_vwap_signal(self, price, vwap, vwap_std):
        """Determine VWAP-based signal"""
        if price > vwap + (2 * vwap_std):
            return 'overbought'
        elif price > vwap:
            return 'bullish'
        elif price < vwap - (2 * vwap_std):
            return 'oversold'
        else:
            return 'bearish'
    
    def calculate_money_flow_index(self, df, period=14):
        """Calculate Money Flow Index (MFI) - volume-weighted RSI"""
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        money_flow = typical_price * df['Volume']
        
        # Calculate positive and negative money flow
        positive_flow = pd.Series(np.where(typical_price > typical_price.shift(1), money_flow, 0), index=df.index)
        negative_flow = pd.Series(np.where(typical_price < typical_price.shift(1), money_flow, 0), index=df.index)
        
        positive_mf = positive_flow.rolling(window=period).sum()
        negative_mf = negative_flow.rolling(window=period).sum()
        
        mfi = 100 - (100 / (1 + positive_mf / negative_mf))
        
        return {
            'mfi': mfi.iloc[-1] if len(mfi) > 0 else 50,
            'mfi_signal': self._get_mfi_signal(mfi.iloc[-1] if len(mfi) > 0 else 50)
        }
    
    def _get_mfi_signal(self, mfi):
        """Determine MFI-based signal"""
        if mfi > 80:
            return 'overbought'
        elif mfi > 50:
            return 'bullish'
        elif mfi < 20:
            return 'oversold'
        else:
            return 'bearish'
    
    def detect_volume_patterns(self, df, lookback=10):
        """Detect specific volume patterns"""
        patterns = []
        
        # Volume spike detection
        recent_volume = df['Volume'].tail(lookback)
        avg_volume = recent_volume.mean()
        
        if df['Volume'].iloc[-1] > avg_volume * 2:
            if df['Close'].iloc[-1] > df['Open'].iloc[-1]:
                patterns.append('bullish_volume_spike')
            else:
                patterns.append('bearish_volume_spike')
        
        # Volume dry-up detection
        if df['Volume'].iloc[-1] < avg_volume * 0.5:
            patterns.append('volume_dryup')
        
        # Increasing volume trend
        volume_trend = recent_volume.rolling(window=3).mean()
        if len(volume_trend) >= 2 and volume_trend.iloc[-1] > volume_trend.iloc[-2] * 1.2:
            patterns.append('increasing_volume_trend')
        
        # Climax volume
        if df['Volume'].iloc[-1] == recent_volume.max():
            if abs(df['Close'].iloc[-1] - df['Open'].iloc[-1]) / df['Open'].iloc[-1] > 0.03:
                patterns.append('climax_volume')
        
        return patterns
    
    def calculate_accumulation_distribution(self, df):
        """Calculate Accumulation/Distribution Line"""
        clv = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
        clv = clv.fillna(0)  # Handle division by zero
        
        ad = (clv * df['Volume']).cumsum()
        ad_sma = ad.rolling(window=10).mean()
        
        return {
            'ad_line': ad.iloc[-1],
            'ad_sma': ad_sma.iloc[-1] if len(ad_sma) > 0 else ad.iloc[-1],
            'ad_trend': 'accumulation' if ad.iloc[-1] > ad_sma.iloc[-1] else 'distribution',
            'ad_divergence': self._check_ad_divergence(df['Close'], ad)
        }
    
    def _check_ad_divergence(self, price, ad_line):
        """Check for divergence between price and A/D line"""
        if len(price) < 20:
            return None
            
        price_change = (price.iloc[-1] - price.iloc[-20]) / price.iloc[-20]
        ad_change = (ad_line.iloc[-1] - ad_line.iloc[-20]) / abs(ad_line.iloc[-20]) if ad_line.iloc[-20] != 0 else 0
        
        if price_change > 0.02 and ad_change < -0.02:
            return 'bearish_divergence'
        elif price_change < -0.02 and ad_change > 0.02:
            return 'bullish_divergence'
        
        return None
    
    def get_volume_confirmation_score(self, volume_data):
        """Calculate overall volume confirmation score"""
        score = 50  # Base score
        
        # Relative volume
        if volume_data['relative_volume']['is_high_volume']:
            score += 10
        
        # OBV confirmation
        if volume_data['obv']['obv_signal'] == 'bullish':
            score += 15
        elif volume_data['obv']['obv_signal'] == 'bearish':
            score -= 15
        
        # VWAP position
        if volume_data['vwap']['vwap_signal'] == 'bullish':
            score += 10
        elif volume_data['vwap']['vwap_signal'] == 'bearish':
            score -= 10
        
        # MFI
        mfi = volume_data['mfi']['mfi']
        if 50 < mfi < 80:
            score += 10
        elif mfi > 80:
            score -= 5  # Overbought
        elif mfi < 20:
            score += 5   # Oversold bounce potential
        
        # Volume patterns
        patterns = volume_data['patterns']
        if 'bullish_volume_spike' in patterns:
            score += 15
        elif 'bearish_volume_spike' in patterns:
            score -= 15
        
        if 'volume_dryup' in patterns:
            score -= 5  # Low conviction
        
        # A/D Line
        if volume_data['ad']['ad_trend'] == 'accumulation':
            score += 10
        else:
            score -= 10
        
        # Divergences (strong signals)
        if volume_data['obv']['divergence'] == 'bullish_divergence':
            score += 20
        elif volume_data['obv']['divergence'] == 'bearish_divergence':
            score -= 20
        
        return max(0, min(100, score))
    
    def analyze_volume(self, df):
        """Complete volume analysis"""
        if len(df) < 20:
            return None
        
        analysis = {
            'relative_volume': self.calculate_relative_volume(df),
            'obv': self.calculate_on_balance_volume(df.copy()),
            'vwap': self.calculate_volume_weighted_average_price(df.copy()),
            'mfi': self.calculate_money_flow_index(df),
            'patterns': self.detect_volume_patterns(df),
            'ad': self.calculate_accumulation_distribution(df)
        }
        
        # Calculate overall score
        analysis['volume_score'] = self.get_volume_confirmation_score(analysis)
        analysis['volume_signal'] = self._get_volume_signal(analysis['volume_score'])
        
        return analysis
    
    def _get_volume_signal(self, score):
        """Convert volume score to signal"""
        if score >= 70:
            return 'strong_bullish'
        elif score >= 60:
            return 'bullish'
        elif score <= 30:
            return 'strong_bearish'
        elif score <= 40:
            return 'bearish'
        else:
            return 'neutral'
    
    def generate_volume_summary(self, analysis):
        """Generate human-readable volume analysis summary"""
        if not analysis:
            return "No volume analysis available"
        
        summary = "\n📊 Volume Analysis Summary\n"
        summary += "=" * 40 + "\n"
        
        # Overall signal
        summary += f"Volume Signal: {analysis['volume_signal']} (Score: {analysis['volume_score']})\n"
        
        # Key metrics
        rv = analysis['relative_volume']
        summary += f"\nRelative Volume: {rv['relative_volume']:.2f}x average"
        if rv['is_high_volume']:
            summary += " ⚡ HIGH VOLUME"
        
        # OBV
        obv = analysis['obv']
        summary += f"\nOBV Trend: {obv['obv_signal']}"
        if obv['divergence']:
            summary += f" ⚠️ {obv['divergence']}"
        
        # VWAP
        vwap = analysis['vwap']
        summary += f"\nPrice vs VWAP: {vwap['price_vs_vwap']:+.2f}% ({vwap['vwap_signal']})"
        
        # MFI
        mfi = analysis['mfi']
        summary += f"\nMoney Flow Index: {mfi['mfi']:.1f} ({mfi['mfi_signal']})"
        
        # Patterns
        if analysis['patterns']:
            summary += f"\nVolume Patterns: {', '.join(analysis['patterns'])}"
        
        # A/D
        ad = analysis['ad']
        summary += f"\nA/D Line: {ad['ad_trend']}"
        if ad['ad_divergence']:
            summary += f" ⚠️ {ad['ad_divergence']}"
        
        return summary


def main():
    """Test the volume analyzer"""
    import yfinance as yf
    
    analyzer = VolumeAnalyzer()
    
    # Test with a stock
    symbol = 'AAPL'
    stock = yf.Ticker(symbol)
    df = stock.history(period='3mo')
    
    if not df.empty:
        analysis = analyzer.analyze_volume(df)
        if analysis:
            print(f"\n🔍 Volume Analysis for {symbol}")
            print(analyzer.generate_volume_summary(analysis))
            
            # Show detailed metrics
            print("\n📈 Detailed Metrics:")
            print(f"Relative Volume: {analysis['relative_volume']['relative_volume']:.2f}x")
            print(f"Volume Rank: {analysis['relative_volume']['volume_rank']:.1f}%")
            print(f"OBV Signal: {analysis['obv']['obv_signal']}")
            print(f"VWAP: ${analysis['vwap']['vwap']:.2f}")
            print(f"MFI: {analysis['mfi']['mfi']:.1f}")

if __name__ == "__main__":
    main()