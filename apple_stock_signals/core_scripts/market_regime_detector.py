#!/usr/bin/env python3
"""
Market Regime Detection System
Identifies current market conditions to adapt trading strategies dynamically
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from typing import Dict, Tuple, List
import json

class MarketRegimeDetector:
    """
    Detects market regime using multiple indicators:
    - Trend strength (ADX, Moving averages)
    - Volatility (VIX, ATR)
    - Market breadth (Advance/Decline)
    - Momentum indicators
    """
    
    def __init__(self):
        self.regimes = {
            'STRONG_BULL': {'risk_multiplier': 1.5, 'strategy': 'momentum'},
            'BULL': {'risk_multiplier': 1.2, 'strategy': 'trend_following'},
            'NEUTRAL': {'risk_multiplier': 1.0, 'strategy': 'mixed'},
            'BEAR': {'risk_multiplier': 0.7, 'strategy': 'mean_reversion'},
            'STRONG_BEAR': {'risk_multiplier': 0.5, 'strategy': 'defensive'},
            'HIGH_VOLATILITY': {'risk_multiplier': 0.6, 'strategy': 'volatility_adjusted'}
        }
        
    def detect_market_regime(self, market_data: pd.DataFrame = None, symbol='SPY', lookback_days=50) -> Dict:
        """
        Main method to detect current market regime
        Can accept pre-fetched market data or fetch it
        """
        if market_data is not None:
            # Use provided data
            spy_data = market_data
            vix_data = pd.DataFrame()  # Empty VIX data for tests
        else:
            # Fetch market data
            spy_data = self._fetch_market_data(symbol, lookback_days)
            vix_data = self._fetch_vix_data(lookback_days)
        
        # Calculate regime indicators
        trend_score = self._calculate_trend_score(spy_data)
        volatility_score = self._calculate_volatility_score(spy_data, vix_data)
        breadth_score = self._calculate_breadth_score(spy_data)
        momentum_score = self._calculate_momentum_score(spy_data)
        
        # Combine scores to determine regime
        regime = self._determine_regime(trend_score, volatility_score, breadth_score, momentum_score)
        
        return {
            'regime': regime,
            'confidence': self._calculate_confidence(trend_score, volatility_score, breadth_score, momentum_score),
            'scores': {
                'trend': trend_score,
                'volatility': volatility_score,
                'breadth': breadth_score,
                'momentum': momentum_score
            },
            'strategy': self.regimes[regime]['strategy'],
            'risk_multiplier': self.regimes[regime]['risk_multiplier'],
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def detect_regime(self, symbol='SPY', lookback_days=50) -> Dict:
        """
        Main method to detect current market regime
        """
        try:
            # Fetch market data
            spy_data = self._fetch_market_data(symbol, lookback_days)
            vix_data = self._fetch_vix_data(lookback_days)
            
            # Calculate regime indicators
            trend_score = self._calculate_trend_score(spy_data)
            volatility_score = self._calculate_volatility_score(spy_data, vix_data)
            breadth_score = self._calculate_breadth_score(spy_data)
            momentum_score = self._calculate_momentum_score(spy_data)
            
            # Combine scores to determine regime
            regime = self._determine_regime(trend_score, volatility_score, breadth_score, momentum_score)
            
            return {
                'regime': regime,
                'confidence': self._calculate_confidence(trend_score, volatility_score, breadth_score, momentum_score),
                'scores': {
                    'trend': trend_score,
                    'volatility': volatility_score,
                    'breadth': breadth_score,
                    'momentum': momentum_score
                },
                'strategy': self.regimes[regime]['strategy'],
                'risk_multiplier': self.regimes[regime]['risk_multiplier'],
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
        except Exception as e:
            print(f"Error detecting market regime: {str(e)}")
            return self._default_regime()
    
    def _fetch_market_data(self, symbol: str, lookback_days: int) -> pd.DataFrame:
        """Fetch market data for analysis"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days * 2)  # Extra data for indicators
        
        ticker = yf.Ticker(symbol)
        data = ticker.history(start=start_date, end=end_date)
        
        return data
    
    def _fetch_vix_data(self, lookback_days: int) -> pd.DataFrame:
        """Fetch VIX data for volatility analysis"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days * 2)
            
            vix = yf.Ticker('^VIX')
            vix_data = vix.history(start=start_date, end=end_date)
            
            return vix_data
        except:
            # Return empty dataframe if VIX data unavailable
            return pd.DataFrame()
    
    def _calculate_trend_score(self, data: pd.DataFrame) -> float:
        """
        Calculate trend strength score (0-100)
        Based on moving averages, ADX, and price position
        """
        if len(data) < 50:
            return 50.0
            
        # Moving averages
        data['SMA_20'] = data['Close'].rolling(20).mean()
        data['SMA_50'] = data['Close'].rolling(50).mean()
        data['SMA_200'] = data['Close'].rolling(200).mean() if len(data) >= 200 else data['SMA_50']
        
        # Current price position
        current_price = data['Close'].iloc[-1]
        sma_20 = data['SMA_20'].iloc[-1]
        sma_50 = data['SMA_50'].iloc[-1]
        sma_200 = data['SMA_200'].iloc[-1]
        
        # Calculate trend score components
        score = 50.0  # Neutral baseline
        
        # Price above/below moving averages
        if current_price > sma_20:
            score += 10
        else:
            score -= 10
            
        if current_price > sma_50:
            score += 15
        else:
            score -= 15
            
        if current_price > sma_200:
            score += 20
        else:
            score -= 20
            
        # Moving average alignment
        if sma_20 > sma_50 > sma_200:
            score += 15  # Bullish alignment
        elif sma_20 < sma_50 < sma_200:
            score -= 15  # Bearish alignment
            
        # ADX for trend strength
        adx = self._calculate_adx(data)
        if adx > 25:
            # Strong trend, amplify the score
            score = score + (score - 50) * 0.3
            
        return max(0, min(100, score))
    
    def _calculate_volatility_score(self, spy_data: pd.DataFrame, vix_data: pd.DataFrame = None) -> float:
        """
        Calculate volatility score (0-100)
        Higher score = Higher volatility = More caution needed
        """
        score = 0.0
        
        # VIX level scoring
        if vix_data is not None and not vix_data.empty:
            current_vix = vix_data['Close'].iloc[-1]
            vix_sma = vix_data['Close'].rolling(20).mean().iloc[-1]
            
            # VIX levels
            if current_vix < 15:
                score += 10  # Low volatility
            elif current_vix < 20:
                score += 30  # Normal volatility
            elif current_vix < 30:
                score += 60  # Elevated volatility
            else:
                score += 90  # High volatility
                
            # VIX trend
            if current_vix > vix_sma:
                score += 10  # Increasing volatility
        
        # SPY ATR-based volatility
        spy_data['ATR'] = self._calculate_atr(spy_data)
        current_atr = spy_data['ATR'].iloc[-1]
        avg_atr = spy_data['ATR'].rolling(20).mean().iloc[-1]
        
        if current_atr > avg_atr * 1.5:
            score += 20  # High relative volatility
        elif current_atr > avg_atr * 1.2:
            score += 10
            
        return min(100, score)
    
    def _calculate_breadth_score(self, data: pd.DataFrame) -> float:
        """
        Calculate market breadth score (0-100)
        Using price action and volume as proxy
        """
        # Calculate up/down days
        data['Daily_Return'] = data['Close'].pct_change()
        data['Up_Day'] = (data['Daily_Return'] > 0).astype(int)
        
        # Recent breadth (last 20 days)
        recent_up_days = data['Up_Day'].tail(20).sum()
        breadth_score = (recent_up_days / 20) * 100
        
        # Volume confirmation
        data['Volume_SMA'] = data['Volume'].rolling(20).mean()
        recent_volume = data['Volume'].tail(5).mean()
        avg_volume = data['Volume_SMA'].iloc[-1]
        
        if recent_volume > avg_volume * 1.2:
            # High volume confirms the breadth
            if breadth_score > 50:
                breadth_score = min(100, breadth_score + 10)
            else:
                breadth_score = max(0, breadth_score - 10)
                
        return breadth_score
    
    def _calculate_momentum_score(self, data: pd.DataFrame) -> float:
        """
        Calculate momentum score using RSI and rate of change
        """
        # RSI
        rsi = self._calculate_rsi(data['Close'])
        
        # Rate of Change (20-day)
        roc_20 = ((data['Close'].iloc[-1] / data['Close'].iloc[-20] - 1) * 100) if len(data) >= 20 else 0
        
        # MACD
        exp1 = data['Close'].ewm(span=12, adjust=False).mean()
        exp2 = data['Close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        macd_histogram = macd - signal
        
        # Combine into momentum score
        score = 50.0  # Neutral baseline
        
        # RSI contribution
        if rsi > 70:
            score += 20  # Overbought, strong momentum
        elif rsi > 50:
            score += 10
        elif rsi < 30:
            score -= 20  # Oversold, weak momentum
        else:
            score -= 10
            
        # ROC contribution
        if roc_20 > 10:
            score += 20
        elif roc_20 > 5:
            score += 10
        elif roc_20 < -10:
            score -= 20
        elif roc_20 < -5:
            score -= 10
            
        # MACD contribution
        if macd_histogram.iloc[-1] > 0 and macd_histogram.iloc[-2] < macd_histogram.iloc[-1]:
            score += 10  # Positive and increasing
        elif macd_histogram.iloc[-1] < 0 and macd_histogram.iloc[-2] > macd_histogram.iloc[-1]:
            score -= 10  # Negative and decreasing
            
        return max(0, min(100, score))
    
    def _determine_regime(self, trend: float, volatility: float, breadth: float, momentum: float) -> str:
        """
        Determine market regime based on all scores
        """
        # High volatility overrides other signals
        if volatility > 70:
            return 'HIGH_VOLATILITY'
        
        # Calculate composite score
        # Weights: Trend 40%, Momentum 30%, Breadth 30%
        composite = (trend * 0.4 + momentum * 0.3 + breadth * 0.3)
        
        # Adjust for volatility
        if volatility > 50:
            composite = composite * 0.8  # Reduce bullishness in high volatility
        
        # Determine regime
        if composite >= 75:
            return 'STRONG_BULL'
        elif composite >= 60:
            return 'BULL'
        elif composite >= 40:
            return 'NEUTRAL'
        elif composite >= 25:
            return 'BEAR'
        else:
            return 'STRONG_BEAR'
    
    def _calculate_confidence(self, trend: float, volatility: float, breadth: float, momentum: float) -> float:
        """
        Calculate confidence in regime detection (0-100)
        """
        # Higher confidence when indicators agree
        scores = [trend, breadth, momentum]
        std_dev = np.std(scores)
        
        # Lower confidence in high volatility
        volatility_penalty = min(20, volatility / 5)
        
        # Base confidence on agreement between indicators
        confidence = 100 - std_dev - volatility_penalty
        
        return max(0, min(100, confidence))
    
    def _calculate_adx(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average Directional Index"""
        high = data['High']
        low = data['Low']
        close = data['Close']
        
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        
        # Directional movements
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        
        pos_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
        neg_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)
        
        pos_di = 100 * (pos_dm.rolling(period).mean() / atr)
        neg_di = 100 * (neg_dm.rolling(period).mean() / atr)
        
        dx = 100 * abs(pos_di - neg_di) / (pos_di + neg_di)
        adx = dx.rolling(period).mean()
        
        return adx.iloc[-1] if not adx.empty else 0
    
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
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = -delta.where(delta < 0, 0).rolling(period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.iloc[-1] if not rsi.empty else 50
    
    def _default_regime(self) -> Dict:
        """Return default neutral regime if detection fails"""
        return {
            'regime': 'NEUTRAL',
            'confidence': 0,
            'scores': {
                'trend': 50,
                'volatility': 50,
                'breadth': 50,
                'momentum': 50
            },
            'strategy': 'mixed',
            'risk_multiplier': 1.0,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def get_strategy_parameters(self, regime: str) -> Dict:
        """
        Get specific strategy parameters for each regime
        """
        params = {
            'STRONG_BULL': {
                'rsi_oversold': 40,  # Higher threshold in bull market
                'rsi_overbought': 80,
                'position_size_multiplier': 1.5,
                'stop_loss_multiplier': 1.5,  # Wider stops in trending market
                'take_profit_multiplier': 1.2,  # Let winners run
                'max_positions': 8,
                'preferred_sectors': ['Technology', 'Consumer Discretionary'],
                'min_volume_ratio': 1.2
            },
            'BULL': {
                'rsi_oversold': 35,
                'rsi_overbought': 75,
                'position_size_multiplier': 1.2,
                'stop_loss_multiplier': 1.3,
                'take_profit_multiplier': 1.1,
                'max_positions': 6,
                'preferred_sectors': ['Technology', 'Financials'],
                'min_volume_ratio': 1.0
            },
            'NEUTRAL': {
                'rsi_oversold': 30,
                'rsi_overbought': 70,
                'position_size_multiplier': 1.0,
                'stop_loss_multiplier': 1.0,
                'take_profit_multiplier': 1.0,
                'max_positions': 5,
                'preferred_sectors': [],
                'min_volume_ratio': 1.0
            },
            'BEAR': {
                'rsi_oversold': 25,
                'rsi_overbought': 65,
                'position_size_multiplier': 0.7,
                'stop_loss_multiplier': 0.8,  # Tighter stops
                'take_profit_multiplier': 0.9,  # Take profits quickly
                'max_positions': 3,
                'preferred_sectors': ['Consumer Staples', 'Utilities'],
                'min_volume_ratio': 1.5  # Need stronger confirmation
            },
            'STRONG_BEAR': {
                'rsi_oversold': 20,
                'rsi_overbought': 60,
                'position_size_multiplier': 0.5,
                'stop_loss_multiplier': 0.7,
                'take_profit_multiplier': 0.8,
                'max_positions': 2,
                'preferred_sectors': ['Consumer Staples', 'Healthcare'],
                'min_volume_ratio': 2.0
            },
            'HIGH_VOLATILITY': {
                'rsi_oversold': 20,
                'rsi_overbought': 80,
                'position_size_multiplier': 0.6,
                'stop_loss_multiplier': 1.5,  # Wider stops for volatility
                'take_profit_multiplier': 0.8,  # But take profits quickly
                'max_positions': 4,
                'preferred_sectors': [],
                'min_volume_ratio': 1.5
            }
        }
        
        return params.get(regime, params['NEUTRAL'])


def main():
    """Test the market regime detector"""
    detector = MarketRegimeDetector()
    
    print("🔍 MARKET REGIME DETECTION")
    print("="*60)
    
    # Detect current regime
    regime_info = detector.detect_regime()
    
    print(f"\n📊 Current Market Regime: {regime_info['regime']}")
    print(f"Confidence: {regime_info['confidence']:.1f}%")
    print(f"\nScores:")
    for key, value in regime_info['scores'].items():
        print(f"  {key.capitalize()}: {value:.1f}")
    
    print(f"\n📈 Recommended Strategy: {regime_info['strategy']}")
    print(f"Risk Multiplier: {regime_info['risk_multiplier']}x")
    
    # Get strategy parameters
    params = detector.get_strategy_parameters(regime_info['regime'])
    print(f"\n⚙️ Strategy Parameters:")
    print(f"  RSI Oversold: {params['rsi_oversold']}")
    print(f"  RSI Overbought: {params['rsi_overbought']}")
    print(f"  Max Positions: {params['max_positions']}")
    print(f"  Position Size Multiplier: {params['position_size_multiplier']}x")
    
    # Save regime info
    with open('outputs/market_regime.json', 'w') as f:
        json.dump(regime_info, f, indent=2)
    
    print(f"\n💾 Regime info saved to outputs/market_regime.json")


if __name__ == "__main__":
    main()