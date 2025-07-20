#!/usr/bin/env python3
"""
Volume Breakout Analyzer
Identifies and confirms breakouts using volume analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from typing import Dict, List, Tuple
import json

class VolumeBreakoutAnalyzer:
    """
    Analyzes volume patterns to confirm breakouts and identify high-probability entries
    """
    
    def __init__(self):
        self.breakout_criteria = {
            'volume_surge_min': 1.5,      # Minimum volume ratio for breakout
            'volume_surge_strong': 2.0,    # Strong breakout volume
            'volume_surge_extreme': 3.0,   # Extreme volume (climax)
            'lookback_periods': 20,        # Days to calculate average volume
            'consolidation_periods': 10,   # Days to check for consolidation
            'breakout_confirm_days': 2     # Days to confirm breakout
        }
        
    def analyze_volume_breakout(self, symbol: str, period: str = '3mo') -> Dict:
        """
        Comprehensive volume breakout analysis for a symbol
        """
        print(f"\n🔍 Analyzing volume breakout for {symbol}...")
        
        # Fetch data
        data = self._fetch_data(symbol, period)
        if data.empty:
            return None
        
        # Calculate volume indicators
        volume_analysis = self._analyze_volume_patterns(data)
        
        # Detect breakout patterns
        breakout_analysis = self._detect_breakout_patterns(data, volume_analysis)
        
        # Calculate breakout quality score
        breakout_score = self._calculate_breakout_score(breakout_analysis, volume_analysis)
        
        # Identify entry points
        entry_analysis = self._identify_entry_points(data, breakout_analysis)
        
        return {
            'symbol': symbol,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'current_price': data['Close'].iloc[-1],
            'volume_analysis': volume_analysis,
            'breakout_analysis': breakout_analysis,
            'breakout_score': breakout_score,
            'entry_analysis': entry_analysis,
            'recommendation': self._generate_recommendation(breakout_score, breakout_analysis)
        }
    
    def _fetch_data(self, symbol: str, period: str) -> pd.DataFrame:
        """Fetch historical data"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            return data
        except Exception as e:
            print(f"Error fetching data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def _analyze_volume_patterns(self, data: pd.DataFrame) -> Dict:
        """Analyze various volume patterns"""
        # Basic volume metrics
        current_volume = data['Volume'].iloc[-1]
        avg_volume = data['Volume'].rolling(self.breakout_criteria['lookback_periods']).mean()
        volume_sma = avg_volume.iloc[-1]
        
        # Volume ratio
        volume_ratio = current_volume / volume_sma if volume_sma > 0 else 0
        
        # Recent volume trend (5, 10, 20 days)
        vol_trend_5 = data['Volume'].tail(5).mean() / volume_sma
        vol_trend_10 = data['Volume'].tail(10).mean() / volume_sma
        vol_trend_20 = data['Volume'].tail(20).mean() / volume_sma
        
        # Volume momentum (is volume increasing?)
        vol_momentum = self._calculate_volume_momentum(data)
        
        # On-Balance Volume (OBV)
        obv = self._calculate_obv(data)
        obv_trend = self._calculate_obv_trend(obv)
        
        # Volume-Price Trend (VPT)
        vpt = self._calculate_vpt(data)
        
        # Accumulation/Distribution Line
        adl = self._calculate_adl(data)
        adl_trend = self._calculate_adl_trend(adl)
        
        # Volume climax detection
        volume_climax = self._detect_volume_climax(data)
        
        return {
            'current_volume': current_volume,
            'average_volume': volume_sma,
            'volume_ratio': volume_ratio,
            'volume_trend': {
                '5_day': vol_trend_5,
                '10_day': vol_trend_10,
                '20_day': vol_trend_20
            },
            'volume_momentum': vol_momentum,
            'obv_trend': obv_trend,
            'adl_trend': adl_trend,
            'volume_climax': volume_climax,
            'volume_classification': self._classify_volume(volume_ratio)
        }
    
    def _detect_breakout_patterns(self, data: pd.DataFrame, volume_analysis: Dict) -> Dict:
        """Detect various breakout patterns"""
        # Price breakout detection
        price_breakout = self._detect_price_breakout(data)
        
        # Consolidation breakout
        consolidation_breakout = self._detect_consolidation_breakout(data)
        
        # Range breakout
        range_breakout = self._detect_range_breakout(data)
        
        # Volume-confirmed breakout
        volume_confirmed = (
            volume_analysis['volume_ratio'] >= self.breakout_criteria['volume_surge_min'] and
            (price_breakout['is_breakout'] or consolidation_breakout['is_breakout'] or range_breakout['is_breakout'])
        )
        
        # Breakout type
        breakout_type = 'none'
        if volume_confirmed:
            if price_breakout['is_breakout']:
                breakout_type = price_breakout['type']
            elif consolidation_breakout['is_breakout']:
                breakout_type = 'consolidation'
            elif range_breakout['is_breakout']:
                breakout_type = 'range'
        
        # False breakout detection
        false_breakout_risk = self._assess_false_breakout_risk(data, volume_analysis)
        
        return {
            'price_breakout': price_breakout,
            'consolidation_breakout': consolidation_breakout,
            'range_breakout': range_breakout,
            'volume_confirmed': volume_confirmed,
            'breakout_type': breakout_type,
            'false_breakout_risk': false_breakout_risk,
            'breakout_strength': self._calculate_breakout_strength(data, volume_analysis)
        }
    
    def _calculate_breakout_score(self, breakout_analysis: Dict, volume_analysis: Dict) -> float:
        """Calculate overall breakout quality score (0-100)"""
        score = 0
        
        # Volume ratio contribution (0-40 points)
        vol_ratio = volume_analysis['volume_ratio']
        if vol_ratio >= self.breakout_criteria['volume_surge_extreme']:
            score += 40
        elif vol_ratio >= self.breakout_criteria['volume_surge_strong']:
            score += 35
        elif vol_ratio >= self.breakout_criteria['volume_surge_min']:
            score += 25
        else:
            score += max(0, vol_ratio * 20)  # Partial credit
        
        # Breakout confirmation (0-20 points)
        if breakout_analysis['volume_confirmed']:
            score += 20
        
        # Volume trend (0-15 points)
        vol_trends = volume_analysis['volume_trend']
        if vol_trends['5_day'] > 1.2 and vol_trends['10_day'] > 1.1:
            score += 15
        elif vol_trends['5_day'] > 1.1:
            score += 10
        elif vol_trends['5_day'] > 1.0:
            score += 5
        
        # OBV/ADL confirmation (0-15 points)
        if volume_analysis['obv_trend'] == 'bullish' and volume_analysis['adl_trend'] == 'bullish':
            score += 15
        elif volume_analysis['obv_trend'] == 'bullish' or volume_analysis['adl_trend'] == 'bullish':
            score += 8
        
        # Breakout strength (0-10 points)
        score += min(10, breakout_analysis['breakout_strength'] * 10)
        
        # Penalty for false breakout risk
        score = score * (1 - breakout_analysis['false_breakout_risk'])
        
        return min(100, max(0, score))
    
    def _identify_entry_points(self, data: pd.DataFrame, breakout_analysis: Dict) -> Dict:
        """Identify optimal entry points based on breakout analysis"""
        current_price = data['Close'].iloc[-1]
        recent_high = data['High'].tail(20).max()
        recent_low = data['Low'].tail(20).min()
        
        entry_points = {
            'current_price': current_price,
            'immediate_entry': None,
            'pullback_entry': None,
            'breakout_retest_entry': None,
            'risk_reward_favorable': False
        }
        
        # Immediate entry if strong breakout
        if breakout_analysis['volume_confirmed'] and breakout_analysis['breakout_strength'] > 0.7:
            entry_points['immediate_entry'] = {
                'price': current_price,
                'stop_loss': recent_low,
                'target_1': current_price + (current_price - recent_low) * 1.5,
                'target_2': current_price + (current_price - recent_low) * 3.0,
                'confidence': 'high'
            }
        
        # Pullback entry levels
        if breakout_analysis['breakout_type'] != 'none':
            pullback_level = recent_high - (recent_high - recent_low) * 0.382  # 38.2% Fibonacci
            entry_points['pullback_entry'] = {
                'price': pullback_level,
                'stop_loss': recent_low,
                'target_1': recent_high + (recent_high - recent_low) * 0.618,
                'target_2': recent_high + (recent_high - recent_low) * 1.0,
                'confidence': 'medium'
            }
        
        # Breakout retest entry
        if breakout_analysis['price_breakout']['is_breakout']:
            retest_level = breakout_analysis['price_breakout']['breakout_level']
            entry_points['breakout_retest_entry'] = {
                'price': retest_level,
                'stop_loss': retest_level * 0.98,  # 2% below breakout
                'target_1': retest_level * 1.05,
                'target_2': retest_level * 1.10,
                'confidence': 'high' if breakout_analysis['false_breakout_risk'] < 0.3 else 'low'
            }
        
        # Calculate risk-reward
        if entry_points['immediate_entry']:
            risk = current_price - entry_points['immediate_entry']['stop_loss']
            reward = entry_points['immediate_entry']['target_1'] - current_price
            entry_points['risk_reward_favorable'] = (reward / risk) >= 2.0
        
        return entry_points
    
    def _calculate_volume_momentum(self, data: pd.DataFrame) -> str:
        """Calculate if volume is increasing, decreasing, or stable"""
        recent_volumes = data['Volume'].tail(10)
        
        # Linear regression slope
        x = np.arange(len(recent_volumes))
        slope = np.polyfit(x, recent_volumes.values, 1)[0]
        
        avg_volume = recent_volumes.mean()
        slope_percentage = (slope / avg_volume) * 100
        
        if slope_percentage > 10:
            return 'increasing'
        elif slope_percentage < -10:
            return 'decreasing'
        else:
            return 'stable'
    
    def _calculate_obv(self, data: pd.DataFrame) -> pd.Series:
        """Calculate On-Balance Volume"""
        obv = pd.Series(index=data.index, dtype=float)
        obv.iloc[0] = 0
        
        for i in range(1, len(data)):
            if data['Close'].iloc[i] > data['Close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + data['Volume'].iloc[i]
            elif data['Close'].iloc[i] < data['Close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - data['Volume'].iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv
    
    def _calculate_obv_trend(self, obv: pd.Series) -> str:
        """Determine OBV trend"""
        if len(obv) < 20:
            return 'neutral'
        
        obv_sma = obv.rolling(20).mean()
        current_obv = obv.iloc[-1]
        sma_obv = obv_sma.iloc[-1]
        
        # Also check short-term trend
        recent_obv_slope = np.polyfit(range(10), obv.tail(10).values, 1)[0]
        
        if current_obv > sma_obv and recent_obv_slope > 0:
            return 'bullish'
        elif current_obv < sma_obv and recent_obv_slope < 0:
            return 'bearish'
        else:
            return 'neutral'
    
    def _calculate_vpt(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Volume Price Trend"""
        vpt = pd.Series(index=data.index, dtype=float)
        vpt.iloc[0] = data['Volume'].iloc[0]
        
        for i in range(1, len(data)):
            price_change = (data['Close'].iloc[i] - data['Close'].iloc[i-1]) / data['Close'].iloc[i-1]
            vpt.iloc[i] = vpt.iloc[i-1] + (price_change * data['Volume'].iloc[i])
        
        return vpt
    
    def _calculate_adl(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Accumulation/Distribution Line"""
        mfm = ((data['Close'] - data['Low']) - (data['High'] - data['Close'])) / (data['High'] - data['Low'])
        mfm = mfm.fillna(0)  # Handle division by zero
        mfv = mfm * data['Volume']
        adl = mfv.cumsum()
        
        return adl
    
    def _calculate_adl_trend(self, adl: pd.Series) -> str:
        """Determine ADL trend"""
        if len(adl) < 20:
            return 'neutral'
        
        adl_sma = adl.rolling(20).mean()
        current_adl = adl.iloc[-1]
        sma_adl = adl_sma.iloc[-1]
        
        # Check divergence with price
        recent_adl_slope = np.polyfit(range(10), adl.tail(10).values, 1)[0]
        
        if current_adl > sma_adl and recent_adl_slope > 0:
            return 'bullish'
        elif current_adl < sma_adl and recent_adl_slope < 0:
            return 'bearish'
        else:
            return 'neutral'
    
    def _detect_volume_climax(self, data: pd.DataFrame) -> Dict:
        """Detect volume climax conditions"""
        recent_volumes = data['Volume'].tail(60)
        current_volume = data['Volume'].iloc[-1]
        
        # Calculate volume percentile
        volume_percentile = (recent_volumes < current_volume).sum() / len(recent_volumes) * 100
        
        # Check for extreme volume
        is_climax = volume_percentile >= 95
        
        # Determine climax type
        climax_type = 'none'
        if is_climax:
            price_change = (data['Close'].iloc[-1] - data['Close'].iloc[-2]) / data['Close'].iloc[-2]
            if price_change > 0.02:  # 2% up move
                climax_type = 'buying_climax'
            elif price_change < -0.02:  # 2% down move
                climax_type = 'selling_climax'
            else:
                climax_type = 'distribution'
        
        return {
            'is_climax': is_climax,
            'climax_type': climax_type,
            'volume_percentile': volume_percentile
        }
    
    def _classify_volume(self, volume_ratio: float) -> str:
        """Classify volume level"""
        if volume_ratio >= self.breakout_criteria['volume_surge_extreme']:
            return 'extreme'
        elif volume_ratio >= self.breakout_criteria['volume_surge_strong']:
            return 'strong'
        elif volume_ratio >= self.breakout_criteria['volume_surge_min']:
            return 'above_average'
        elif volume_ratio >= 0.8:
            return 'average'
        else:
            return 'below_average'
    
    def _detect_price_breakout(self, data: pd.DataFrame) -> Dict:
        """Detect price breakout from recent highs/lows"""
        lookback = 20
        recent_high = data['High'].tail(lookback).max()
        recent_low = data['Low'].tail(lookback).min()
        current_close = data['Close'].iloc[-1]
        prev_close = data['Close'].iloc[-2]
        
        # Breakout detection
        is_breakout = False
        breakout_type = 'none'
        breakout_level = None
        
        if current_close > recent_high and prev_close <= recent_high:
            is_breakout = True
            breakout_type = 'resistance'
            breakout_level = recent_high
        elif current_close < recent_low and prev_close >= recent_low:
            is_breakout = True
            breakout_type = 'support'
            breakout_level = recent_low
        
        # Calculate breakout magnitude
        breakout_magnitude = 0
        if breakout_type == 'resistance':
            breakout_magnitude = (current_close - recent_high) / recent_high
        elif breakout_type == 'support':
            breakout_magnitude = (recent_low - current_close) / recent_low
        
        return {
            'is_breakout': is_breakout,
            'type': breakout_type,
            'breakout_level': breakout_level,
            'magnitude': breakout_magnitude
        }
    
    def _detect_consolidation_breakout(self, data: pd.DataFrame) -> Dict:
        """Detect breakout from consolidation pattern"""
        lookback = self.breakout_criteria['consolidation_periods']
        recent_data = data.tail(lookback)
        
        # Calculate price range
        high_range = recent_data['High'].max() - recent_data['High'].min()
        low_range = recent_data['Low'].max() - recent_data['Low'].min()
        avg_range = (high_range + low_range) / 2
        
        # Calculate average true range
        atr = self._calculate_atr(data).iloc[-lookback:].mean()
        
        # Consolidation detection (tight range)
        is_consolidating = avg_range < atr * 2
        
        # Breakout from consolidation
        current_close = data['Close'].iloc[-1]
        consolidation_high = recent_data['High'].max()
        consolidation_low = recent_data['Low'].min()
        
        is_breakout = False
        if is_consolidating:
            if current_close > consolidation_high:
                is_breakout = True
            elif current_close < consolidation_low:
                is_breakout = True
        
        return {
            'is_consolidating': is_consolidating,
            'is_breakout': is_breakout,
            'consolidation_high': consolidation_high,
            'consolidation_low': consolidation_low,
            'range_percentage': (avg_range / current_close) * 100
        }
    
    def _detect_range_breakout(self, data: pd.DataFrame) -> Dict:
        """Detect breakout from trading range"""
        # Look for range over last 50 days
        lookback = 50
        recent_data = data.tail(lookback)
        
        # Find significant levels (tested multiple times)
        highs = recent_data['High']
        lows = recent_data['Low']
        
        # Find resistance (multiple tests of similar high)
        resistance_levels = []
        for i in range(len(highs) - 5):
            window = highs.iloc[i:i+5]
            if window.std() / window.mean() < 0.01:  # 1% tolerance
                resistance_levels.append(window.mean())
        
        # Find support (multiple tests of similar low)
        support_levels = []
        for i in range(len(lows) - 5):
            window = lows.iloc[i:i+5]
            if window.std() / window.mean() < 0.01:  # 1% tolerance
                support_levels.append(window.mean())
        
        # Current price
        current_close = data['Close'].iloc[-1]
        
        # Check for breakout
        is_breakout = False
        breakout_direction = 'none'
        
        if resistance_levels:
            max_resistance = max(resistance_levels)
            if current_close > max_resistance * 1.01:  # 1% above resistance
                is_breakout = True
                breakout_direction = 'upward'
        
        if support_levels and not is_breakout:
            min_support = min(support_levels)
            if current_close < min_support * 0.99:  # 1% below support
                is_breakout = True
                breakout_direction = 'downward'
        
        return {
            'is_breakout': is_breakout,
            'direction': breakout_direction,
            'resistance_levels': resistance_levels[-3:] if resistance_levels else [],
            'support_levels': support_levels[-3:] if support_levels else []
        }
    
    def _assess_false_breakout_risk(self, data: pd.DataFrame, volume_analysis: Dict) -> float:
        """Assess risk of false breakout (0-1, higher = more risk)"""
        risk_score = 0
        
        # Low volume increases false breakout risk
        if volume_analysis['volume_ratio'] < self.breakout_criteria['volume_surge_min']:
            risk_score += 0.3
        
        # Volume climax might indicate exhaustion
        if volume_analysis['volume_climax']['is_climax']:
            if volume_analysis['volume_climax']['climax_type'] in ['buying_climax', 'selling_climax']:
                risk_score += 0.2
        
        # Declining volume trend
        if volume_analysis['volume_momentum'] == 'decreasing':
            risk_score += 0.2
        
        # OBV/ADL divergence
        obv_trend = volume_analysis['obv_trend']
        adl_trend = volume_analysis['adl_trend']
        price_trend = 'bullish' if data['Close'].iloc[-1] > data['Close'].iloc[-20] else 'bearish'
        
        if (obv_trend != price_trend) or (adl_trend != price_trend):
            risk_score += 0.3
        
        return min(1.0, risk_score)
    
    def _calculate_breakout_strength(self, data: pd.DataFrame, volume_analysis: Dict) -> float:
        """Calculate breakout strength (0-1)"""
        strength = 0
        
        # Volume contribution
        vol_ratio = volume_analysis['volume_ratio']
        if vol_ratio >= self.breakout_criteria['volume_surge_extreme']:
            strength += 0.4
        elif vol_ratio >= self.breakout_criteria['volume_surge_strong']:
            strength += 0.3
        elif vol_ratio >= self.breakout_criteria['volume_surge_min']:
            strength += 0.2
        
        # Price movement contribution
        price_change = abs((data['Close'].iloc[-1] - data['Close'].iloc[-2]) / data['Close'].iloc[-2])
        strength += min(0.3, price_change * 10)  # Max 0.3 for 3% move
        
        # Volume trend contribution
        if volume_analysis['volume_momentum'] == 'increasing':
            strength += 0.15
        
        # OBV/ADL confirmation
        if volume_analysis['obv_trend'] == 'bullish' and volume_analysis['adl_trend'] == 'bullish':
            strength += 0.15
        
        return min(1.0, strength)
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high = data['High']
        low = data['Low']
        close = data['Close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        return tr.rolling(period).mean()
    
    def _generate_recommendation(self, breakout_score: float, breakout_analysis: Dict) -> str:
        """Generate trading recommendation based on breakout analysis"""
        if breakout_score >= 80 and breakout_analysis['volume_confirmed']:
            return 'STRONG_BUY - High quality volume breakout detected'
        elif breakout_score >= 60 and breakout_analysis['volume_confirmed']:
            return 'BUY - Volume-confirmed breakout'
        elif breakout_score >= 40:
            return 'WATCH - Potential breakout forming, wait for confirmation'
        elif breakout_analysis['false_breakout_risk'] > 0.6:
            return 'AVOID - High false breakout risk'
        else:
            return 'HOLD - No significant breakout pattern'
    
    def analyze_multiple_symbols(self, symbols: List[str]) -> List[Dict]:
        """Analyze multiple symbols and rank by breakout quality"""
        results = []
        
        for symbol in symbols:
            print(f"\nAnalyzing {symbol}...")
            analysis = self.analyze_volume_breakout(symbol)
            if analysis:
                results.append(analysis)
        
        # Sort by breakout score
        results.sort(key=lambda x: x['breakout_score'], reverse=True)
        
        return results
    
    def generate_report(self, analyses: List[Dict]) -> str:
        """Generate a summary report of breakout analyses"""
        report = "\n📊 VOLUME BREAKOUT ANALYSIS REPORT\n"
        report += "="*60 + "\n"
        report += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        report += f"Symbols Analyzed: {len(analyses)}\n\n"
        
        # Top breakouts
        report += "🎯 TOP BREAKOUT CANDIDATES:\n"
        report += "-"*60 + "\n"
        
        for i, analysis in enumerate(analyses[:5]):  # Top 5
            symbol = analysis['symbol']
            score = analysis['breakout_score']
            vol_ratio = analysis['volume_analysis']['volume_ratio']
            breakout_type = analysis['breakout_analysis']['breakout_type']
            recommendation = analysis['recommendation']
            
            report += f"\n{i+1}. {symbol}\n"
            report += f"   Score: {score:.1f}/100\n"
            report += f"   Volume Ratio: {vol_ratio:.2f}x average\n"
            report += f"   Breakout Type: {breakout_type}\n"
            report += f"   Recommendation: {recommendation}\n"
            
            # Entry points if available
            if analysis['entry_analysis']['immediate_entry']:
                entry = analysis['entry_analysis']['immediate_entry']
                report += f"   Entry: ${entry['price']:.2f}, Stop: ${entry['stop_loss']:.2f}, Target: ${entry['target_1']:.2f}\n"
        
        return report


def main():
    """Test the volume breakout analyzer"""
    analyzer = VolumeBreakoutAnalyzer()
    
    # Test with top stocks
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA']
    
    print("🚀 VOLUME BREAKOUT ANALYSIS")
    print("="*60)
    
    # Analyze all symbols
    analyses = analyzer.analyze_multiple_symbols(symbols)
    
    # Generate report
    report = analyzer.generate_report(analyses)
    print(report)
    
    # Save detailed results
    import os
    os.makedirs('outputs/volume_breakouts', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    with open(f'outputs/volume_breakouts/analysis_{timestamp}.json', 'w') as f:
        json.dump(analyses, f, indent=2)
    
    print(f"\n💾 Detailed results saved to outputs/volume_breakouts/")


if __name__ == "__main__":
    main()