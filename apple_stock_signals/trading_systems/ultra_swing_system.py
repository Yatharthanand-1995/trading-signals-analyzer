#!/usr/bin/env python3
"""
Ultra-Enhanced Swing Trading System
Advanced multi-factor model with machine learning elements
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

@dataclass
class UltraTrade:
    """Enhanced trade structure with more details."""
    entry_date: datetime
    exit_date: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    stop_loss: float
    initial_stop: float
    take_profit_1: float
    take_profit_2: float
    take_profit_3: float
    position_size: int
    position_value: float
    signal_strength: float
    setup_quality: str
    market_regime: str
    volatility_regime: str
    holding_days: Optional[int]
    exit_reason: Optional[str]
    profit_loss: Optional[float]
    profit_loss_pct: Optional[float]
    risk_reward_actual: Optional[float]
    scaled_entry: bool = False
    partial_exit: bool = False

class UltraSwingSystem:
    def __init__(self, data_dir: str = "historical_data", 
                 initial_capital: float = 10000,
                 risk_per_trade: float = 0.02,
                 max_position_pct: float = 0.33,
                 min_holding_days: int = 2,
                 max_holding_days: int = 15):
        """Initialize ultra swing trading system."""
        self.data_dir = data_dir
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.max_position_pct = max_position_pct
        self.min_holding_days = min_holding_days
        self.max_holding_days = max_holding_days
        self.trades = []
        self.equity_curve = [initial_capital]
        
        # Pattern memory for learning
        self.successful_patterns = []
        self.failed_patterns = []
        
    def load_data(self, symbol: str) -> pd.DataFrame:
        """Load historical data for a symbol."""
        filename = os.path.join(self.data_dir, f"{symbol}_historical_data.csv")
        if os.path.exists(filename):
            df = pd.read_csv(filename, index_col=0)
            df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
            return df
        return pd.DataFrame()
    
    def calculate_market_internals(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Calculate market breadth and internals."""
        # Simulate market breadth
        df['Market_Trend'] = df['Close'].rolling(window=50).mean()
        df['Market_Strength'] = (df['Close'] - df['Market_Trend']) / df['Market_Trend']
        
        # Relative strength vs market
        df['RS_Ratio'] = df['Close'] / df['SMA_50']
        df['RS_Rank'] = df['RS_Ratio'].rolling(window=20).rank(pct=True)
        
        # Sector strength simulation (would use real sector data in production)
        if symbol in ['AAPL', 'MSFT', 'GOOGL']:
            df['Sector_Strength'] = 1.1  # Tech sector bonus
        else:
            df['Sector_Strength'] = 1.0
        
        return df
    
    def calculate_advanced_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate advanced technical indicators."""
        # Basic indicators first
        df['EMA_8'] = df['Close'].ewm(span=8, adjust=False).mean()
        df['EMA_13'] = df['Close'].ewm(span=13, adjust=False).mean()
        df['EMA_21'] = df['Close'].ewm(span=21, adjust=False).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        
        # VWAP (Volume Weighted Average Price)
        df['VWAP'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()
        df['VWAP_Distance'] = (df['Close'] - df['VWAP']) / df['VWAP']
        
        # ATR with multiple periods
        for period in [10, 14, 20]:
            high_low = df['High'] - df['Low']
            high_close = abs(df['High'] - df['Close'].shift())
            low_close = abs(df['Low'] - df['Close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            df[f'ATR_{period}'] = true_range.rolling(period).mean()
        
        # Keltner Channels
        df['KC_Middle'] = df['EMA_21']
        df['KC_Upper'] = df['KC_Middle'] + (2 * df['ATR_10'])
        df['KC_Lower'] = df['KC_Middle'] - (2 * df['ATR_10'])
        df['KC_Position'] = (df['Close'] - df['KC_Lower']) / (df['KC_Upper'] - df['KC_Lower'])
        
        # RSI with multiple periods
        for period in [5, 10, 14]:
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            df[f'RSI_{period}'] = 100 - (100 / (1 + rs))
        
        # Stochastic RSI
        rsi = df['RSI_14']
        df['StochRSI'] = (rsi - rsi.rolling(14).min()) / (rsi.rolling(14).max() - rsi.rolling(14).min())
        
        # MACD with histogram analysis
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        df['MACD_Histogram_Delta'] = df['MACD_Histogram'].diff()
        
        # Volume analysis
        df['Volume_SMA_20'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_20']
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).cumsum()
        df['OBV_EMA'] = df['OBV'].ewm(span=20, adjust=False).mean()
        
        # Price patterns
        df['Hammer'] = ((df['Close'] - df['Low']) > 2 * (df['High'] - df['Close'])) & \
                       ((df['High'] - df['Low']) > 3 * abs(df['Close'] - df['Open']))
        
        df['Bullish_Engulfing'] = (df['Close'] > df['Open']) & \
                                   (df['Close'].shift() < df['Open'].shift()) & \
                                   (df['Open'] < df['Close'].shift()) & \
                                   (df['Close'] > df['Open'].shift())
        
        # Support/Resistance with Fibonacci
        rolling_high = df['High'].rolling(window=20).max()
        rolling_low = df['Low'].rolling(window=20).min()
        df['Fib_0'] = rolling_low
        df['Fib_236'] = rolling_low + 0.236 * (rolling_high - rolling_low)
        df['Fib_382'] = rolling_low + 0.382 * (rolling_high - rolling_low)
        df['Fib_50'] = rolling_low + 0.5 * (rolling_high - rolling_low)
        df['Fib_618'] = rolling_low + 0.618 * (rolling_high - rolling_low)
        df['Fib_100'] = rolling_high
        
        # Market structure
        df['Higher_High'] = (df['High'] > df['High'].shift(1)) & (df['High'].shift(1) > df['High'].shift(2))
        df['Higher_Low'] = (df['Low'] > df['Low'].shift(1)) & (df['Low'].shift(1) > df['Low'].shift(2))
        df['Lower_High'] = (df['High'] < df['High'].shift(1)) & (df['High'].shift(1) < df['High'].shift(2))
        df['Lower_Low'] = (df['Low'] < df['Low'].shift(1)) & (df['Low'].shift(1) < df['Low'].shift(2))
        
        # Volatility regimes
        df['Volatility'] = df['Close'].pct_change().rolling(20).std() * np.sqrt(252)
        df['Volatility_Regime'] = pd.qcut(df['Volatility'].rolling(60).mean(), q=3, labels=['Low', 'Medium', 'High'])
        
        # Momentum indicators
        df['ROC_5'] = (df['Close'] - df['Close'].shift(5)) / df['Close'].shift(5) * 100
        df['ROC_10'] = (df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10) * 100
        df['ROC_20'] = (df['Close'] - df['Close'].shift(20)) / df['Close'].shift(20) * 100
        
        # ADX for trend strength
        df['DMI_Plus'] = np.where(
            (df['High'] - df['High'].shift(1)) > (df['Low'].shift(1) - df['Low']),
            np.maximum(df['High'] - df['High'].shift(1), 0), 0
        )
        df['DMI_Minus'] = np.where(
            (df['Low'].shift(1) - df['Low']) > (df['High'] - df['High'].shift(1)),
            np.maximum(df['Low'].shift(1) - df['Low'], 0), 0
        )
        
        df['DI_Plus'] = 100 * (df['DMI_Plus'].rolling(14).mean() / df['ATR_14'])
        df['DI_Minus'] = 100 * (df['DMI_Minus'].rolling(14).mean() / df['ATR_14'])
        df['ADX'] = 100 * abs(df['DI_Plus'] - df['DI_Minus']) / (df['DI_Plus'] + df['DI_Minus'])
        df['ADX'] = df['ADX'].rolling(14).mean()
        
        return df
    
    def detect_market_regime(self, df: pd.DataFrame, i: int) -> str:
        """Detect current market regime."""
        # Trend detection
        sma50 = df['SMA_50'].iloc[i]
        sma200 = df['SMA_200'].iloc[i]
        price = df['Close'].iloc[i]
        
        # ADX for trend strength
        adx = df['ADX'].iloc[i] if not pd.isna(df['ADX'].iloc[i]) else 20
        
        if sma50 > sma200 and price > sma50:
            if adx > 25:
                return 'strong_uptrend'
            else:
                return 'uptrend'
        elif sma50 < sma200 and price < sma50:
            if adx > 25:
                return 'strong_downtrend'
            else:
                return 'downtrend'
        else:
            return 'sideways'
    
    def calculate_setup_quality(self, df: pd.DataFrame, i: int) -> Tuple[float, List[str]]:
        """Calculate comprehensive setup quality score."""
        score = 0
        factors = []
        
        # 1. Trend alignment (25 points)
        if df['EMA_8'].iloc[i] > df['EMA_13'].iloc[i] > df['EMA_21'].iloc[i] > df['SMA_50'].iloc[i]:
            score += 25
            factors.append('perfect_ema_alignment')
        elif df['EMA_8'].iloc[i] > df['EMA_21'].iloc[i] > df['SMA_50'].iloc[i]:
            score += 15
            factors.append('good_ema_alignment')
        
        # 2. Pullback quality (20 points)
        close = df['Close'].iloc[i]
        ema21 = df['EMA_21'].iloc[i]
        distance = (close - ema21) / ema21
        
        if -0.02 < distance < 0.01:  # Near EMA21
            score += 20
            factors.append('perfect_pullback')
        elif -0.03 < distance < 0.02:
            score += 10
            factors.append('good_pullback')
        
        # 3. Support confluence (15 points)
        fib_382 = df['Fib_382'].iloc[i]
        if abs(close - fib_382) / close < 0.01:  # Near Fibonacci level
            score += 15
            factors.append('fib_support')
        
        # 4. RSI conditions (15 points)
        rsi = df['RSI_14'].iloc[i]
        rsi_5 = df['RSI_5'].iloc[i]
        
        if 35 < rsi < 55 and rsi_5 > df['RSI_5'].iloc[i-1]:  # Momentum turning up
            score += 15
            factors.append('rsi_momentum')
        elif 30 < rsi < 45:
            score += 10
            factors.append('oversold_bounce')
        
        # 5. Volume surge (10 points)
        if df['Volume_Ratio'].iloc[i] > 1.5:
            score += 10
            factors.append('volume_surge')
        
        # 6. MACD conditions (10 points)
        if df['MACD_Histogram'].iloc[i] > 0 and df['MACD_Histogram_Delta'].iloc[i] > 0:
            score += 10
            factors.append('macd_accelerating')
        
        # 7. Price patterns (10 points)
        if df['Hammer'].iloc[i] or df['Bullish_Engulfing'].iloc[i]:
            score += 10
            factors.append('bullish_pattern')
        
        # 8. Market structure (10 points)
        if df['Higher_Low'].iloc[i-1:i+1].any():
            score += 10
            factors.append('higher_low')
        
        # 9. Relative strength (10 points)
        if df['RS_Rank'].iloc[i] > 0.7:
            score += 10
            factors.append('strong_rs')
        
        # 10. Low volatility bonus (5 points)
        if df['Volatility_Regime'].iloc[i] == 'Low':
            score += 5
            factors.append('low_volatility')
        
        return score, factors
    
    def generate_ultra_signals(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Generate ultra-enhanced trading signals."""
        # Calculate all indicators
        df = self.calculate_advanced_indicators(df)
        df = self.calculate_market_internals(df, symbol)
        
        df['Signal'] = 0
        df['Signal_Strength'] = 0
        df['Setup_Factors'] = ''
        df['Market_Regime'] = ''
        
        for i in range(200, len(df)):
            # Check market regime first
            regime = self.detect_market_regime(df, i)
            df.loc[df.index[i], 'Market_Regime'] = regime
            
            # Only trade in favorable regimes (long only)
            if regime not in ['uptrend', 'strong_uptrend']:
                continue
            
            # Calculate setup quality
            score, factors = self.calculate_setup_quality(df, i)
            
            # Additional filters
            # Don't enter if already extended
            if df['Close'].iloc[i] > df['EMA_8'].iloc[i] * 1.03:  # 3% above EMA8
                continue
            
            # Don't enter in extreme volatility
            if df['Volatility'].iloc[i] > df['Volatility'].rolling(60).mean().iloc[i] * 2:
                continue
            
            # Store signal data
            df.loc[df.index[i], 'Signal_Strength'] = score
            df.loc[df.index[i], 'Setup_Factors'] = ','.join(factors)
            
            # Ultra-strict entry criteria
            if score >= 75 and len(factors) >= 4:  # High score and multiple confirmations
                df.loc[df.index[i], 'Signal'] = 1
        
        return df
    
    def calculate_dynamic_stops(self, df: pd.DataFrame, i: int, entry_price: float) -> Tuple[float, float, float, float]:
        """Calculate dynamic stop loss and take profit levels."""
        atr = df['ATR_14'].iloc[i]
        volatility_regime = df['Volatility_Regime'].iloc[i]
        
        # Adjust multipliers based on volatility
        if volatility_regime == 'Low':
            stop_mult = 1.2
            tp_mults = [1.5, 2.5, 4.0]
        elif volatility_regime == 'High':
            stop_mult = 2.0
            tp_mults = [1.2, 2.0, 3.0]
        else:  # Medium
            stop_mult = 1.5
            tp_mults = [1.5, 2.5, 3.5]
        
        # Calculate levels
        stop_loss = entry_price - (stop_mult * atr)
        
        # Use structure-based stops
        recent_low = df['Low'].iloc[i-10:i].min()
        structure_stop = recent_low * 0.995  # 0.5% below recent low
        
        # Use the higher stop (less risk)
        stop_loss = max(stop_loss, structure_stop)
        
        # Calculate take profits
        risk = entry_price - stop_loss
        take_profit_1 = entry_price + (tp_mults[0] * risk)
        take_profit_2 = entry_price + (tp_mults[1] * risk)
        take_profit_3 = entry_price + (tp_mults[2] * risk)
        
        return stop_loss, take_profit_1, take_profit_2, take_profit_3
    
    def manage_position(self, trade: UltraTrade, df: pd.DataFrame, i: int) -> Tuple[bool, str, float]:
        """Advanced position management with trailing stops and partial exits."""
        current_price = df['Close'].iloc[i]
        high_price = df['High'].iloc[i]
        low_price = df['Low'].iloc[i]
        days_held = (df.index[i] - trade.entry_date).days
        
        # Check stop loss first
        if low_price <= trade.stop_loss:
            return True, 'stop_loss', trade.stop_loss
        
        # Check take profits
        if high_price >= trade.take_profit_3 and not trade.partial_exit:
            return True, 'take_profit_3', trade.take_profit_3
        elif high_price >= trade.take_profit_2 and not trade.partial_exit:
            # Could implement partial exit here
            return True, 'take_profit_2', trade.take_profit_2
        elif high_price >= trade.take_profit_1 and not trade.partial_exit:
            # Move stop to breakeven
            trade.stop_loss = trade.entry_price
            trade.partial_exit = True
            # Continue holding
        
        # Time-based management
        if days_held >= self.min_holding_days:
            # Profit management
            profit_pct = (current_price - trade.entry_price) / trade.entry_price
            
            if profit_pct > 0.05:  # 5% profit
                # Aggressive trailing stop
                atr = df['ATR_10'].iloc[i]
                trailing_stop = current_price - (1.0 * atr)
                trade.stop_loss = max(trade.stop_loss, trailing_stop)
            elif profit_pct > 0.03:  # 3% profit
                # Moderate trailing stop
                atr = df['ATR_14'].iloc[i]
                trailing_stop = current_price - (1.5 * atr)
                trade.stop_loss = max(trade.stop_loss, trailing_stop)
            
            # Exit on technical weakness
            if df['EMA_8'].iloc[i] < df['EMA_13'].iloc[i]:
                if profit_pct < 0:  # Only exit losers on weakness
                    return True, 'technical_weakness', current_price
            
            # Exit if momentum dies
            if df['MACD_Histogram'].iloc[i] < 0 and df['RSI_14'].iloc[i] < 45:
                return True, 'momentum_loss', current_price
            
            # Maximum holding period
            if days_held >= self.max_holding_days:
                return True, 'max_days', current_price
        
        return False, None, None
    
    def backtest(self, symbol: str, df: pd.DataFrame) -> Dict:
        """Run ultra-enhanced backtest."""
        # Generate signals
        df = self.generate_ultra_signals(df, symbol)
        
        # Reset state
        self.trades = []
        self.current_capital = self.initial_capital
        self.equity_curve = [self.initial_capital]
        open_position = None
        
        # Track consecutive losses for risk reduction
        consecutive_losses = 0
        
        for i in range(200, len(df)):
            current_date = df.index[i]
            current_price = df['Close'].iloc[i]
            
            # Manage open position
            if open_position:
                exit_triggered, exit_reason, exit_price = self.manage_position(
                    open_position, df, i
                )
                
                if exit_triggered:
                    # Close position
                    trade = open_position
                    trade.exit_date = current_date
                    trade.exit_price = exit_price if exit_price else current_price
                    trade.exit_reason = exit_reason
                    trade.holding_days = (current_date - trade.entry_date).days
                    
                    # Calculate P&L
                    trade.profit_loss = (trade.exit_price - trade.entry_price) * trade.position_size
                    trade.profit_loss_pct = (trade.exit_price - trade.entry_price) / trade.entry_price
                    
                    # Risk/Reward
                    risk = abs(trade.entry_price - trade.initial_stop)
                    actual_reward = trade.exit_price - trade.entry_price
                    trade.risk_reward_actual = actual_reward / risk if risk > 0 else 0
                    
                    # Update capital
                    self.current_capital += trade.profit_loss
                    self.equity_curve.append(self.current_capital)
                    
                    # Track consecutive losses
                    if trade.profit_loss < 0:
                        consecutive_losses += 1
                    else:
                        consecutive_losses = 0
                    
                    # Store pattern success/failure
                    pattern_data = {
                        'factors': trade.setup_quality,
                        'score': trade.signal_strength,
                        'regime': trade.market_regime,
                        'volatility': trade.volatility_regime,
                        'profit_pct': trade.profit_loss_pct
                    }
                    
                    if trade.profit_loss > 0:
                        self.successful_patterns.append(pattern_data)
                    else:
                        self.failed_patterns.append(pattern_data)
                    
                    self.trades.append(trade)
                    open_position = None
            
            # Check for new entry
            elif df['Signal'].iloc[i] == 1 and consecutive_losses < 3:  # Risk reduction after losses
                # Additional confirmation checks
                if i > 0 and df['Signal'].iloc[i-1] == 1:  # Skip if signal yesterday
                    continue
                
                # Get dynamic stops
                stop_loss, tp1, tp2, tp3 = self.calculate_dynamic_stops(df, i, current_price)
                
                # Position sizing with Kelly Criterion element
                win_rate = len([t for t in self.trades[-20:] if t.profit_loss > 0]) / min(len(self.trades), 20) if self.trades else 0.4
                
                # Adjust risk based on recent performance
                if win_rate > 0.5 and consecutive_losses == 0:
                    risk_multiplier = 1.2
                elif consecutive_losses >= 2:
                    risk_multiplier = 0.5
                else:
                    risk_multiplier = 1.0
                
                adjusted_risk = self.risk_per_trade * risk_multiplier
                risk_amount = self.current_capital * adjusted_risk
                
                risk_per_share = abs(current_price - stop_loss)
                if risk_per_share > 0:
                    shares = int(risk_amount / risk_per_share)
                    
                    # Apply position limits
                    max_shares = int(self.current_capital * self.max_position_pct / current_price)
                    shares = min(shares, max_shares)
                    
                    position_value = shares * current_price
                    
                    if shares > 0 and position_value <= self.current_capital * 0.95:
                        # Create trade
                        trade = UltraTrade(
                            entry_date=current_date,
                            exit_date=None,
                            entry_price=current_price,
                            exit_price=None,
                            stop_loss=stop_loss,
                            initial_stop=stop_loss,
                            take_profit_1=tp1,
                            take_profit_2=tp2,
                            take_profit_3=tp3,
                            position_size=shares,
                            position_value=position_value,
                            signal_strength=df['Signal_Strength'].iloc[i],
                            setup_quality=df['Setup_Factors'].iloc[i],
                            market_regime=df['Market_Regime'].iloc[i],
                            volatility_regime=str(df['Volatility_Regime'].iloc[i]),
                            holding_days=None,
                            exit_reason=None,
                            profit_loss=None,
                            profit_loss_pct=None,
                            risk_reward_actual=None
                        )
                        
                        open_position = trade
        
        # Close any remaining position
        if open_position:
            trade = open_position
            trade.exit_date = df.index[-1]
            trade.exit_price = df['Close'].iloc[-1]
            trade.exit_reason = 'end_of_data'
            trade.holding_days = (trade.exit_date - trade.entry_date).days
            trade.profit_loss = (trade.exit_price - trade.entry_price) * trade.position_size
            trade.profit_loss_pct = (trade.exit_price - trade.entry_price) / trade.entry_price
            
            risk = abs(trade.entry_price - trade.initial_stop)
            actual_reward = trade.exit_price - trade.entry_price
            trade.risk_reward_actual = actual_reward / risk if risk > 0 else 0
            
            self.current_capital += trade.profit_loss
            self.equity_curve.append(self.current_capital)
            self.trades.append(trade)
        
        return self.calculate_performance_metrics(symbol)
    
    def calculate_performance_metrics(self, symbol: str) -> Dict:
        """Calculate comprehensive performance metrics."""
        if not self.trades:
            return {'symbol': symbol, 'total_trades': 0, 'message': 'No trades executed'}
        
        # Basic metrics
        total_trades = len(self.trades)
        winning_trades = [t for t in self.trades if t.profit_loss > 0]
        losing_trades = [t for t in self.trades if t.profit_loss <= 0]
        
        win_rate = len(winning_trades) / total_trades * 100
        
        # Financial metrics
        total_profit = sum(t.profit_loss for t in self.trades)
        total_return = (self.current_capital - self.initial_capital) / self.initial_capital * 100
        
        # Average metrics
        avg_win = sum(t.profit_loss for t in winning_trades) / len(winning_trades) if winning_trades else 0
        avg_loss = sum(t.profit_loss for t in losing_trades) / len(losing_trades) if losing_trades else 0
        avg_win_pct = sum(t.profit_loss_pct for t in winning_trades) / len(winning_trades) * 100 if winning_trades else 0
        avg_loss_pct = sum(t.profit_loss_pct for t in losing_trades) / len(losing_trades) * 100 if losing_trades else 0
        
        # Profit factor
        gross_profit = sum(t.profit_loss for t in winning_trades) if winning_trades else 0
        gross_loss = abs(sum(t.profit_loss for t in losing_trades)) if losing_trades else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Risk/Reward analysis
        winning_rr = []
        for trade in winning_trades:
            if trade.risk_reward_actual is not None and trade.risk_reward_actual > 0:
                winning_rr.append(trade.risk_reward_actual)
        
        avg_winning_rr = np.mean(winning_rr) if winning_rr else 0
        
        # Holding period analysis
        holding_days = [t.holding_days for t in self.trades if t.holding_days is not None]
        avg_holding = np.mean(holding_days) if holding_days else 0
        
        # Exit analysis
        exit_reasons = {}
        for trade in self.trades:
            reason = trade.exit_reason or 'unknown'
            exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
        
        # Maximum drawdown
        peak = self.equity_curve[0]
        max_dd = 0
        
        for value in self.equity_curve:
            if value > peak:
                peak = value
            dd = (peak - value) / peak * 100
            max_dd = max(max_dd, dd)
        
        # Sharpe ratio
        if len(self.equity_curve) > 1:
            returns = np.diff(self.equity_curve) / self.equity_curve[:-1]
            sharpe = np.sqrt(252) * np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        else:
            sharpe = 0
        
        # Pattern analysis
        successful_factors = {}
        if self.successful_patterns:
            for pattern in self.successful_patterns:
                for factor in pattern['factors'].split(','):
                    if factor:
                        successful_factors[factor] = successful_factors.get(factor, 0) + 1
        
        return {
            'symbol': symbol,
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': round(win_rate, 2),
            'profit_factor': round(profit_factor, 2),
            'total_profit': round(total_profit, 2),
            'total_return_pct': round(total_return, 2),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'avg_win_pct': round(avg_win_pct, 2),
            'avg_loss_pct': round(avg_loss_pct, 2),
            'avg_holding_days': round(avg_holding, 1),
            'avg_winning_rr': round(avg_winning_rr, 2),
            'max_drawdown_pct': round(max_dd, 2),
            'sharpe_ratio': round(sharpe, 2),
            'exit_reasons': exit_reasons,
            'successful_patterns': dict(sorted(successful_factors.items(), key=lambda x: x[1], reverse=True)[:5]),
            'initial_capital': self.initial_capital,
            'final_capital': round(self.current_capital, 2)
        }


def main():
    """Run ultra swing trading system."""
    symbols = ['AAPL', 'GOOGL', 'TSLA', 'MSFT', 'UNH']
    
    print("🚀 ULTRA Swing Trading System v2.0")
    print("="*60)
    print("Advanced Features:")
    print("• Multi-factor signal generation")
    print("• Dynamic position sizing")
    print("• Pattern learning")
    print("• Market regime detection")
    print("• Advanced risk management")
    print("="*60)
    
    all_results = {}
    
    for symbol in symbols:
        print(f"\nAnalyzing {symbol}...")
        
        system = UltraSwingSystem(
            initial_capital=10000,
            risk_per_trade=0.02,
            max_position_pct=0.33
        )
        
        df = system.load_data(symbol)
        if not df.empty:
            results = system.backtest(symbol, df)
            all_results[symbol] = results
    
    # Display results
    print("\n" + "="*60)
    print("📊 ULTRA SYSTEM PERFORMANCE")
    print("="*60)
    
    # Sort by total return
    sorted_results = sorted(all_results.items(), key=lambda x: x[1]['total_return_pct'], reverse=True)
    
    for symbol, metrics in sorted_results:
        print(f"\n{symbol}:")
        print(f"  Trades: {metrics['total_trades']}")
        print(f"  Win Rate: {metrics['win_rate']}%")
        print(f"  Return: {metrics['total_return_pct']}%")
        print(f"  Profit Factor: {metrics['profit_factor']}")
        print(f"  Sharpe: {metrics['sharpe_ratio']}")
        print(f"  Max DD: {metrics['max_drawdown_pct']}%")
        print(f"  Avg Win: {metrics['avg_win_pct']}%")
        print(f"  Avg Loss: {metrics['avg_loss_pct']}%")
        print(f"  Win R/R: {metrics['avg_winning_rr']}")
        
        if metrics.get('successful_patterns'):
            print(f"  Best Patterns: {list(metrics['successful_patterns'].keys())[:3]}")
    
    # Portfolio summary
    total_return = sum(r['total_return_pct'] for r in all_results.values()) / len(all_results)
    avg_win_rate = np.mean([r['win_rate'] for r in all_results.values()])
    avg_sharpe = np.mean([r['sharpe_ratio'] for r in all_results.values()])
    
    print("\n" + "="*60)
    print("💎 PORTFOLIO PERFORMANCE")
    print("="*60)
    print(f"Average Return: {total_return:.2f}%")
    print(f"Average Win Rate: {avg_win_rate:.2f}%")
    print(f"Average Sharpe: {avg_sharpe:.2f}")
    
    # Save results
    filename = f"ultra_swing_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {filename}")


if __name__ == "__main__":
    main()