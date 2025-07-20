#!/usr/bin/env python3
"""
Adaptive Signal Generator
Generates trading signals that adapt to current market regime
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from typing import Dict, List, Tuple
import json
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from .market_regime_detector import MarketRegimeDetector
except ImportError:
    from market_regime_detector import MarketRegimeDetector

from data_modules.technical_analyzer import AppleTechnicalAnalyzer

class AdaptiveSignalGenerator:
    """
    Generates trading signals that adapt based on:
    - Current market regime
    - Multi-timeframe confirmation
    - Volume breakout patterns
    - Momentum acceleration
    """
    
    def __init__(self):
        self.regime_detector = MarketRegimeDetector()
        self.technical_analyzer = AppleTechnicalAnalyzer()
        self.current_regime = None
        self.regime_params = None
        
    def generate_adaptive_signals(self, symbols: List[str]) -> Dict:
        """
        Generate signals for multiple symbols with market regime adaptation
        """
        # First, detect market regime
        self.current_regime = self.regime_detector.detect_regime()
        self.regime_params = self.regime_detector.get_strategy_parameters(self.current_regime['regime'])
        
        print(f"\n🎯 Generating Adaptive Signals")
        print(f"Market Regime: {self.current_regime['regime']} (Confidence: {self.current_regime['confidence']:.1f}%)")
        print(f"Strategy: {self.current_regime['strategy']}")
        print("="*60)
        
        signals = []
        
        for symbol in symbols:
            print(f"\nAnalyzing {symbol}...", end='', flush=True)
            try:
                signal = self._analyze_symbol(symbol)
                if signal:
                    signals.append(signal)
                    print(f" ✅ {signal['action']} (Score: {signal['score']})")
                else:
                    print(f" ❌ No valid signal")
            except Exception as e:
                print(f" ❌ Error: {str(e)}")
        
        # Sort signals by score
        signals.sort(key=lambda x: x['score'], reverse=True)
        
        # Apply position limits based on regime
        max_positions = self.regime_params['max_positions']
        filtered_signals = signals[:max_positions]
        
        return {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'market_regime': self.current_regime,
            'total_symbols_analyzed': len(symbols),
            'valid_signals': len(signals),
            'filtered_signals': len(filtered_signals),
            'signals': filtered_signals
        }
    
    def _analyze_symbol(self, symbol: str) -> Dict:
        """
        Analyze individual symbol with adaptive parameters
        """
        # Fetch data for multiple timeframes
        daily_data = self._fetch_data(symbol, '1d', 100)
        hourly_data = self._fetch_data(symbol, '1h', 100)
        
        if daily_data.empty or hourly_data.empty:
            return None
        
        # Calculate technical indicators
        daily_indicators = self._calculate_indicators(daily_data)
        hourly_indicators = self._calculate_indicators(hourly_data)
        
        # Multi-timeframe confirmation
        daily_signal = self._get_timeframe_signal(daily_indicators, 'daily')
        hourly_signal = self._get_timeframe_signal(hourly_indicators, 'hourly')
        
        # Volume analysis
        volume_score = self._analyze_volume(daily_data)
        
        # Momentum acceleration
        momentum_score = self._analyze_momentum_acceleration(daily_data)
        
        # Calculate composite score
        signal_score = self._calculate_composite_score(
            daily_signal, hourly_signal, volume_score, momentum_score
        )
        
        # Determine action based on score and regime
        action = self._determine_action(signal_score, daily_indicators)
        
        if action == 'HOLD':
            return None  # Don't return HOLD signals
        
        # Calculate position sizing and risk parameters
        position_params = self._calculate_position_parameters(
            symbol, daily_data, daily_indicators, signal_score
        )
        
        return {
            'symbol': symbol,
            'action': action,
            'score': signal_score,
            'regime_adjusted': True,
            'current_price': daily_data['Close'].iloc[-1],
            'indicators': {
                'rsi': daily_indicators['rsi'],
                'macd_signal': daily_indicators['macd_signal'],
                'volume_ratio': daily_indicators['volume_ratio'],
                'atr': daily_indicators['atr']
            },
            'multi_timeframe': {
                'daily': daily_signal,
                'hourly': hourly_signal
            },
            'volume_breakout': volume_score > 70,
            'momentum_accelerating': momentum_score > 60,
            **position_params
        }
    
    def _fetch_data(self, symbol: str, interval: str, period: int) -> pd.DataFrame:
        """Fetch data for specific timeframe"""
        try:
            ticker = yf.Ticker(symbol)
            
            if interval == '1d':
                end_date = datetime.now()
                start_date = end_date - timedelta(days=period * 2)
                data = ticker.history(start=start_date, end=end_date, interval=interval)
            else:
                # For intraday data
                data = ticker.history(period=f"{period}d", interval=interval)
            
            return data
        except:
            return pd.DataFrame()
    
    def _calculate_indicators(self, data: pd.DataFrame) -> Dict:
        """Calculate technical indicators"""
        if len(data) < 50:
            return {}
        
        indicators = {}
        
        # Price and volume
        indicators['close'] = data['Close'].iloc[-1]
        indicators['volume'] = data['Volume'].iloc[-1]
        
        # Moving averages
        indicators['sma_20'] = data['Close'].rolling(20).mean().iloc[-1]
        indicators['sma_50'] = data['Close'].rolling(50).mean().iloc[-1]
        
        # RSI with regime-adjusted levels
        indicators['rsi'] = self._calculate_rsi(data['Close'])
        indicators['rsi_oversold'] = self.regime_params['rsi_oversold']
        indicators['rsi_overbought'] = self.regime_params['rsi_overbought']
        
        # MACD
        exp1 = data['Close'].ewm(span=12, adjust=False).mean()
        exp2 = data['Close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        indicators['macd'] = macd.iloc[-1]
        indicators['macd_signal'] = signal.iloc[-1]
        indicators['macd_histogram'] = indicators['macd'] - indicators['macd_signal']
        
        # ATR for volatility
        indicators['atr'] = self._calculate_atr(data).iloc[-1]
        
        # Volume ratio
        indicators['volume_ratio'] = data['Volume'].iloc[-1] / data['Volume'].rolling(20).mean().iloc[-1]
        
        # Bollinger Bands
        bb_sma = data['Close'].rolling(20).mean()
        bb_std = data['Close'].rolling(20).std()
        indicators['bb_upper'] = (bb_sma + 2 * bb_std).iloc[-1]
        indicators['bb_lower'] = (bb_sma - 2 * bb_std).iloc[-1]
        indicators['bb_position'] = (data['Close'].iloc[-1] - indicators['bb_lower']) / (indicators['bb_upper'] - indicators['bb_lower'])
        
        return indicators
    
    def _get_timeframe_signal(self, indicators: Dict, timeframe: str) -> float:
        """
        Get signal strength for specific timeframe (0-100)
        """
        if not indicators:
            return 50
        
        score = 50  # Neutral baseline
        
        # Price vs moving averages
        if indicators['close'] > indicators['sma_20']:
            score += 10
        else:
            score -= 10
            
        if indicators['close'] > indicators['sma_50']:
            score += 15
        else:
            score -= 15
        
        # RSI with regime-adjusted levels
        if indicators['rsi'] < indicators['rsi_oversold']:
            score += 20  # Oversold bounce opportunity
        elif indicators['rsi'] > indicators['rsi_overbought']:
            score -= 20  # Overbought warning
        
        # MACD
        if indicators['macd_histogram'] > 0:
            score += 15
            # Momentum acceleration
            if indicators['macd'] > indicators['macd_signal']:
                score += 10
        else:
            score -= 15
        
        # Bollinger Bands position
        if indicators['bb_position'] < 0.2:
            score += 10  # Near lower band
        elif indicators['bb_position'] > 0.8:
            score -= 10  # Near upper band
        
        # Adjust weight based on timeframe
        if timeframe == 'hourly':
            score = score * 0.7  # Lower weight for shorter timeframe
        
        return max(0, min(100, score))
    
    def _analyze_volume(self, data: pd.DataFrame) -> float:
        """
        Analyze volume patterns for breakout confirmation (0-100)
        """
        if len(data) < 20:
            return 50
        
        # Current vs average volume
        current_volume = data['Volume'].iloc[-1]
        avg_volume = data['Volume'].rolling(20).mean().iloc[-1]
        volume_ratio = current_volume / avg_volume
        
        # Recent volume trend (last 5 days)
        recent_volume = data['Volume'].tail(5).mean()
        recent_ratio = recent_volume / avg_volume
        
        # Calculate score
        score = 50
        
        # Current volume spike
        if volume_ratio > 2.0:
            score += 30
        elif volume_ratio > 1.5:
            score += 20
        elif volume_ratio > self.regime_params['min_volume_ratio']:
            score += 10
        else:
            score -= 20
        
        # Recent volume trend
        if recent_ratio > 1.3:
            score += 20
        elif recent_ratio > 1.1:
            score += 10
        
        # Volume with price action
        price_change = (data['Close'].iloc[-1] / data['Close'].iloc[-2] - 1) * 100
        if price_change > 1 and volume_ratio > 1.5:
            score += 20  # Strong volume on up day
        elif price_change < -1 and volume_ratio > 1.5:
            score -= 20  # High volume on down day
        
        return max(0, min(100, score))
    
    def _analyze_momentum_acceleration(self, data: pd.DataFrame) -> float:
        """
        Analyze if momentum is accelerating (0-100)
        """
        if len(data) < 20:
            return 50
        
        # Calculate rate of change over different periods
        roc_5 = (data['Close'].iloc[-1] / data['Close'].iloc[-5] - 1) * 100
        roc_10 = (data['Close'].iloc[-1] / data['Close'].iloc[-10] - 1) * 100
        roc_20 = (data['Close'].iloc[-1] / data['Close'].iloc[-20] - 1) * 100
        
        # Check if momentum is accelerating
        score = 50
        
        # Positive and accelerating
        if roc_5 > roc_10 > roc_20 and roc_5 > 0:
            score += 30
            if roc_5 > 5:  # Strong momentum
                score += 20
        # Negative but decelerating (potential reversal)
        elif roc_5 > roc_10 > roc_20 and roc_20 < -5:
            score += 20
        # Negative and accelerating downward
        elif roc_5 < roc_10 < roc_20 and roc_5 < 0:
            score -= 30
        
        # Check MACD acceleration
        if len(data) >= 30:
            exp1 = data['Close'].ewm(span=12, adjust=False).mean()
            exp2 = data['Close'].ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            
            # MACD slope (acceleration)
            macd_current = macd.iloc[-1]
            macd_prev = macd.iloc[-5]
            
            if macd_current > macd_prev and macd_current > 0:
                score += 10
            elif macd_current < macd_prev and macd_current < 0:
                score -= 10
        
        return max(0, min(100, score))
    
    def _calculate_composite_score(self, daily_signal: float, hourly_signal: float, 
                                 volume_score: float, momentum_score: float) -> float:
        """
        Calculate composite score with regime-based weighting
        """
        # Base weights
        weights = {
            'daily': 0.4,
            'hourly': 0.2,
            'volume': 0.2,
            'momentum': 0.2
        }
        
        # Adjust weights based on regime
        if self.current_regime['regime'] in ['STRONG_BULL', 'BULL']:
            # In bull market, momentum is more important
            weights['momentum'] = 0.3
            weights['daily'] = 0.35
            weights['hourly'] = 0.15
            weights['volume'] = 0.2
        elif self.current_regime['regime'] in ['BEAR', 'STRONG_BEAR']:
            # In bear market, volume confirmation is crucial
            weights['volume'] = 0.3
            weights['daily'] = 0.4
            weights['hourly'] = 0.15
            weights['momentum'] = 0.15
        elif self.current_regime['regime'] == 'HIGH_VOLATILITY':
            # In high volatility, short-term signals matter more
            weights['hourly'] = 0.3
            weights['daily'] = 0.3
            weights['volume'] = 0.25
            weights['momentum'] = 0.15
        
        # Calculate weighted score
        composite = (
            daily_signal * weights['daily'] +
            hourly_signal * weights['hourly'] +
            volume_score * weights['volume'] +
            momentum_score * weights['momentum']
        )
        
        # Apply regime confidence multiplier
        confidence_multiplier = 0.8 + (self.current_regime['confidence'] / 500)  # 0.8 to 1.0
        composite = composite * confidence_multiplier
        
        return round(composite, 1)
    
    def _determine_action(self, score: float, indicators: Dict) -> str:
        """
        Determine trading action based on score and regime
        """
        # Adjust thresholds based on regime
        if self.current_regime['regime'] in ['STRONG_BULL', 'BULL']:
            buy_threshold = 55  # Lower threshold in bull market
            sell_threshold = 30
        elif self.current_regime['regime'] in ['BEAR', 'STRONG_BEAR']:
            buy_threshold = 70  # Higher threshold in bear market
            sell_threshold = 45
        else:
            buy_threshold = 60
            sell_threshold = 40
        
        # Additional filters based on regime
        if score >= buy_threshold:
            # Extra confirmation in bear market
            if self.current_regime['regime'] in ['BEAR', 'STRONG_BEAR']:
                if indicators.get('volume_ratio', 0) < self.regime_params['min_volume_ratio']:
                    return 'HOLD'  # Need volume confirmation
            return 'BUY'
        elif score <= sell_threshold:
            return 'SELL'
        else:
            return 'HOLD'
    
    def _calculate_position_parameters(self, symbol: str, data: pd.DataFrame, 
                                     indicators: Dict, score: float) -> Dict:
        """
        Calculate position sizing and risk parameters adapted to regime
        """
        current_price = data['Close'].iloc[-1]
        atr = indicators['atr']
        
        # Base position size (will be adjusted by regime)
        base_position_size = 1000 / current_price  # $1000 base
        
        # Apply regime multipliers
        position_size = base_position_size * self.regime_params['position_size_multiplier']
        
        # Adjust by signal strength
        if score > 80:
            position_size *= 1.2
        elif score < 60:
            position_size *= 0.8
        
        # Stop loss calculation
        stop_loss_distance = atr * 2.0 * self.regime_params['stop_loss_multiplier']
        stop_loss = current_price - stop_loss_distance
        
        # Take profit targets
        tp_multiplier = self.regime_params['take_profit_multiplier']
        take_profit_1 = current_price + (stop_loss_distance * 1.5 * tp_multiplier)
        take_profit_2 = current_price + (stop_loss_distance * 3.0 * tp_multiplier)
        take_profit_3 = current_price + (stop_loss_distance * 5.0 * tp_multiplier)
        
        return {
            'position_size': round(position_size),
            'stop_loss': round(stop_loss, 2),
            'take_profit_1': round(take_profit_1, 2),
            'take_profit_2': round(take_profit_2, 2),
            'take_profit_3': round(take_profit_3, 2),
            'risk_amount': round(position_size * stop_loss_distance, 2),
            'regime_multiplier': self.regime_params['position_size_multiplier']
        }
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = -delta.where(delta < 0, 0).rolling(period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.iloc[-1] if not rsi.empty else 50
    
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


def main():
    """Test the adaptive signal generator"""
    # Test with top 10 stocks
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 
               'META', 'TSLA', 'BRK-B', 'JPM', 'JNJ']
    
    generator = AdaptiveSignalGenerator()
    
    print("🚀 ADAPTIVE SIGNAL GENERATION")
    print("="*60)
    
    # Generate signals
    results = generator.generate_adaptive_signals(symbols)
    
    print(f"\n📊 Summary:")
    print(f"Market Regime: {results['market_regime']['regime']}")
    print(f"Symbols Analyzed: {results['total_symbols_analyzed']}")
    print(f"Valid Signals: {results['valid_signals']}")
    print(f"Filtered Signals: {results['filtered_signals']}")
    
    print(f"\n🎯 Top Trading Signals:")
    print(f"{'Symbol':<8} {'Action':<8} {'Score':<8} {'Price':<10} {'Stop Loss':<10} {'Target 1':<10}")
    print("-"*66)
    
    for signal in results['signals']:
        print(f"{signal['symbol']:<8} {signal['action']:<8} {signal['score']:<8.1f} "
              f"${signal['current_price']:<9.2f} ${signal['stop_loss']:<9.2f} "
              f"${signal['take_profit_1']:<9.2f}")
    
    # Save results
    os.makedirs('outputs/adaptive_signals', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    with open(f'outputs/adaptive_signals/signals_{timestamp}.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to outputs/adaptive_signals/")


if __name__ == "__main__":
    main()