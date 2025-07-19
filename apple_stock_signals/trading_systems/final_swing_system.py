#!/usr/bin/env python3
"""
Final Optimized Swing Trading System
Balanced approach for 2-15 day trades with improved win rate
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

@dataclass
class FinalTrade:
    """Trade structure for final system."""
    entry_date: datetime
    exit_date: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    position_size: int
    position_value: float
    signal_strength: float
    trend_strength: float
    momentum_score: float
    holding_days: Optional[int]
    exit_reason: Optional[str]
    profit_loss: Optional[float]
    profit_loss_pct: Optional[float]
    risk_reward_actual: Optional[float]
    max_profit_pct: Optional[float]

class FinalSwingSystem:
    def __init__(self, data_dir: str = "historical_data", 
                 initial_capital: float = 10000,
                 risk_per_trade: float = 0.02,
                 max_position_pct: float = 0.33,
                 min_holding_days: int = 2,
                 max_holding_days: int = 15):
        """Initialize final optimized swing trading system."""
        self.data_dir = data_dir
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.max_position_pct = max_position_pct
        self.min_holding_days = min_holding_days
        self.max_holding_days = max_holding_days
        self.trades = []
        self.equity_curve = [initial_capital]
        
    def load_data(self, symbol: str) -> pd.DataFrame:
        """Load historical data for a symbol."""
        filename = os.path.join(self.data_dir, f"{symbol}_historical_data.csv")
        if os.path.exists(filename):
            df = pd.read_csv(filename, index_col=0)
            df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
            return df
        return pd.DataFrame()
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate optimized indicators for swing trading."""
        # Core EMAs for trend
        df['EMA_5'] = df['Close'].ewm(span=5, adjust=False).mean()
        df['EMA_8'] = df['Close'].ewm(span=8, adjust=False).mean()
        df['EMA_13'] = df['Close'].ewm(span=13, adjust=False).mean()
        df['EMA_21'] = df['Close'].ewm(span=21, adjust=False).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        
        # ATR for stops
        high_low = df['High'] - df['Low']
        high_close = abs(df['High'] - df['Close'].shift())
        low_close = abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['ATR'] = true_range.rolling(14).mean()
        df['ATR_10'] = true_range.rolling(10).mean()
        
        # RSI with divergence detection
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=9).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=9).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        df['RSI_Slope'] = df['RSI'].diff(3)
        
        # MACD for momentum
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        df['MACD_Slope'] = df['MACD_Histogram'].diff()
        
        # Volume analysis
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        df['Dollar_Volume'] = df['Close'] * df['Volume']
        df['Dollar_Volume_MA'] = df['Dollar_Volume'].rolling(window=20).mean()
        
        # Price momentum
        df['ROC_5'] = (df['Close'] - df['Close'].shift(5)) / df['Close'].shift(5) * 100
        df['ROC_10'] = (df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10) * 100
        df['Momentum_Score'] = (df['ROC_5'] + df['ROC_10']) / 2
        
        # Support/Resistance zones
        df['Resistance_20'] = df['High'].rolling(window=20).max()
        df['Support_20'] = df['Low'].rolling(window=20).min()
        df['Mid_Point'] = (df['Resistance_20'] + df['Support_20']) / 2
        df['Range_Position'] = (df['Close'] - df['Support_20']) / (df['Resistance_20'] - df['Support_20'])
        
        # Trend strength using ADX
        df['DM_Plus'] = np.where(
            (df['High'] - df['High'].shift(1)) > (df['Low'].shift(1) - df['Low']),
            np.maximum(df['High'] - df['High'].shift(1), 0), 0
        )
        df['DM_Minus'] = np.where(
            (df['Low'].shift(1) - df['Low']) > (df['High'] - df['High'].shift(1)),
            np.maximum(df['Low'].shift(1) - df['Low'], 0), 0
        )
        df['DI_Plus'] = 100 * (df['DM_Plus'].rolling(14).mean() / df['ATR'])
        df['DI_Minus'] = 100 * (df['DM_Minus'].rolling(14).mean() / df['ATR'])
        df['DX'] = 100 * abs(df['DI_Plus'] - df['DI_Minus']) / (df['DI_Plus'] + df['DI_Minus'])
        df['ADX'] = df['DX'].rolling(14).mean()
        
        # Bollinger Bands for volatility
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (2 * bb_std)
        df['BB_Lower'] = df['BB_Middle'] - (2 * bb_std)
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # Volatility percentile
        df['Volatility'] = df['Close'].pct_change().rolling(20).std() * np.sqrt(252)
        df['Vol_Percentile'] = df['Volatility'].rolling(100).rank(pct=True)
        
        # Trend detection
        df['Uptrend'] = ((df['SMA_50'] > df['SMA_200']) & 
                         (df['Close'] > df['SMA_50'])).astype(int)
        
        # Price patterns
        df['Inside_Bar'] = ((df['High'] <= df['High'].shift(1)) & 
                           (df['Low'] >= df['Low'].shift(1))).astype(int)
        
        df['Bull_Flag'] = ((df['Close'].shift(5) < df['Close'].shift(1)) & 
                          (df['Close'] > df['Close'].shift(1)) &
                          (df['Volume'] < df['Volume_SMA'])).astype(int)
        
        return df
    
    def calculate_trend_strength(self, df: pd.DataFrame, i: int) -> float:
        """Calculate trend strength score."""
        score = 0
        
        # Moving average alignment
        if df['EMA_5'].iloc[i] > df['EMA_8'].iloc[i] > df['EMA_13'].iloc[i] > df['EMA_21'].iloc[i]:
            score += 30
        elif df['EMA_8'].iloc[i] > df['EMA_13'].iloc[i] > df['EMA_21'].iloc[i]:
            score += 20
        elif df['EMA_13'].iloc[i] > df['EMA_21'].iloc[i]:
            score += 10
        
        # Price vs MAs
        if df['Close'].iloc[i] > df['EMA_21'].iloc[i]:
            score += 10
        if df['Close'].iloc[i] > df['SMA_50'].iloc[i]:
            score += 10
        
        # ADX trend strength
        if df['ADX'].iloc[i] > 25:
            score += 20
        elif df['ADX'].iloc[i] > 20:
            score += 10
        
        # Trend consistency
        if df['DI_Plus'].iloc[i] > df['DI_Minus'].iloc[i]:
            score += 10
        
        # Market structure
        recent_lows = df['Low'].iloc[i-10:i]
        if len(recent_lows) >= 2 and recent_lows.iloc[-1] > recent_lows.iloc[-5]:
            score += 10  # Higher lows
        
        return score
    
    def calculate_momentum_score(self, df: pd.DataFrame, i: int) -> float:
        """Calculate momentum score."""
        score = 0
        
        # RSI momentum
        rsi = df['RSI'].iloc[i]
        if 40 < rsi < 60:
            score += 20  # Healthy momentum
            if df['RSI_Slope'].iloc[i] > 0:
                score += 10  # Increasing momentum
        elif 30 < rsi < 40 and df['RSI_Slope'].iloc[i] > 0:
            score += 25  # Oversold bounce
        
        # MACD momentum
        if df['MACD'].iloc[i] > df['MACD_Signal'].iloc[i]:
            score += 15
            if df['MACD_Slope'].iloc[i] > 0:
                score += 10  # Accelerating
        
        # Price momentum
        if df['ROC_5'].iloc[i] > 0:
            score += 10
        if df['ROC_10'].iloc[i] > 0:
            score += 10
        
        # Volume momentum
        if df['Volume_Ratio'].iloc[i] > 1.2:
            score += 15
        
        return score
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate balanced trading signals."""
        df = self.calculate_indicators(df)
        
        df['Signal'] = 0
        df['Signal_Strength'] = 0
        df['Trend_Strength'] = 0
        df['Momentum_Score'] = 0
        
        for i in range(200, len(df)):
            # Skip if not in uptrend
            if not df['Uptrend'].iloc[i]:
                continue
            
            # Calculate component scores
            trend_score = self.calculate_trend_strength(df, i)
            momentum_score = self.calculate_momentum_score(df, i)
            
            # Setup conditions
            setup_score = 0
            
            # 1. Pullback to support
            range_pos = df['Range_Position'].iloc[i]
            if 0.2 < range_pos < 0.5:  # Lower half of range
                setup_score += 20
            
            # 2. Near moving average support
            ema21_distance = abs(df['Close'].iloc[i] - df['EMA_21'].iloc[i]) / df['EMA_21'].iloc[i]
            if ema21_distance < 0.02:  # Within 2% of EMA21
                setup_score += 15
            
            # 3. Bollinger Band setup
            if df['BB_Position'].iloc[i] < 0.3:  # Near lower band
                setup_score += 15
            
            # 4. Pattern bonus
            if df['Inside_Bar'].iloc[i]:
                setup_score += 10
            if df['Bull_Flag'].iloc[i]:
                setup_score += 10
            
            # 5. Low volatility bonus (easier to manage risk)
            if df['Vol_Percentile'].iloc[i] < 0.5:
                setup_score += 10
            
            # Total signal strength
            total_score = trend_score + momentum_score + setup_score
            
            # Entry filters
            entry_valid = True
            
            # Don't enter if too extended
            if df['Close'].iloc[i] > df['EMA_8'].iloc[i] * 1.04:  # 4% above EMA8
                entry_valid = False
            
            # Don't enter if RSI extreme
            if df['RSI'].iloc[i] > 70 or df['RSI'].iloc[i] < 20:
                entry_valid = False
            
            # Store scores
            df.loc[df.index[i], 'Signal_Strength'] = total_score
            df.loc[df.index[i], 'Trend_Strength'] = trend_score
            df.loc[df.index[i], 'Momentum_Score'] = momentum_score
            
            # Generate signal with balanced threshold
            if total_score >= 65 and entry_valid:  # Moderate threshold
                # Additional confirmation: price action
                if df['Close'].iloc[i] > df['Open'].iloc[i]:  # Bullish candle
                    df.loc[df.index[i], 'Signal'] = 1
        
        return df
    
    def calculate_position_size(self, entry_price: float, stop_loss: float, 
                              current_capital: float, volatility_percentile: float) -> Tuple[int, float]:
        """Calculate position size with volatility adjustment."""
        # Base risk amount
        risk_amount = current_capital * self.risk_per_trade
        
        # Adjust for volatility
        if volatility_percentile < 0.3:  # Low volatility
            risk_amount *= 1.2
        elif volatility_percentile > 0.7:  # High volatility
            risk_amount *= 0.8
        
        # Calculate shares
        risk_per_share = abs(entry_price - stop_loss)
        if risk_per_share == 0:
            return 0, 0
        
        shares = int(risk_amount / risk_per_share)
        
        # Position limits
        max_position_value = current_capital * self.max_position_pct
        max_shares = int(max_position_value / entry_price)
        shares = min(shares, max_shares)
        
        position_value = shares * entry_price
        
        return shares, position_value
    
    def manage_trade(self, trade: FinalTrade, df: pd.DataFrame, i: int) -> Tuple[bool, str, float]:
        """Manage open trade with optimized exits."""
        current_price = df['Close'].iloc[i]
        high_price = df['High'].iloc[i]
        low_price = df['Low'].iloc[i]
        days_held = (df.index[i] - trade.entry_date).days
        
        # Track maximum profit
        profit_pct = (high_price - trade.entry_price) / trade.entry_price
        if trade.max_profit_pct is None or profit_pct > trade.max_profit_pct:
            trade.max_profit_pct = profit_pct
        
        # Check stop loss
        if low_price <= trade.stop_loss:
            return True, 'stop_loss', trade.stop_loss
        
        # Check take profits
        if high_price >= trade.take_profit_2:
            return True, 'take_profit_2', trade.take_profit_2
        elif high_price >= trade.take_profit_1:
            # Move stop to breakeven plus small profit
            trade.stop_loss = trade.entry_price * 1.005  # 0.5% profit locked
            # Don't exit yet, aim for TP2
        
        # Time-based management after minimum hold
        if days_held >= self.min_holding_days:
            current_profit_pct = (current_price - trade.entry_price) / trade.entry_price
            
            # Trailing stop for winners
            if current_profit_pct > 0.03:  # 3% profit
                # Tight trailing stop
                trailing_stop = current_price - (1.2 * df['ATR_10'].iloc[i])
                trade.stop_loss = max(trade.stop_loss, trailing_stop)
            elif current_profit_pct > 0.015:  # 1.5% profit
                # Breakeven stop
                trade.stop_loss = max(trade.stop_loss, trade.entry_price)
            
            # Exit conditions
            # 1. Trend weakness
            if df['EMA_5'].iloc[i] < df['EMA_13'].iloc[i]:
                if current_profit_pct < 0.01:  # Less than 1% profit
                    return True, 'trend_weakness', current_price
            
            # 2. Momentum loss
            if df['MACD'].iloc[i] < df['MACD_Signal'].iloc[i] and df['RSI'].iloc[i] < 40:
                return True, 'momentum_loss', current_price
            
            # 3. Give back too much profit
            if trade.max_profit_pct and trade.max_profit_pct > 0.05:  # Had 5%+ profit
                if current_profit_pct < trade.max_profit_pct * 0.5:  # Lost half
                    return True, 'profit_give_back', current_price
            
            # 4. Maximum holding period
            if days_held >= self.max_holding_days:
                return True, 'max_hold', current_price
        
        return False, None, None
    
    def backtest(self, symbol: str, df: pd.DataFrame) -> Dict:
        """Run backtest with optimized strategy."""
        # Generate signals
        df = self.generate_signals(df)
        
        # Reset state
        self.trades = []
        self.current_capital = self.initial_capital
        self.equity_curve = [self.initial_capital]
        open_position = None
        last_exit_date = None
        
        for i in range(200, len(df)):
            current_date = df.index[i]
            current_price = df['Close'].iloc[i]
            
            # Manage open position
            if open_position:
                exit_triggered, exit_reason, exit_price = self.manage_trade(
                    open_position, df, i
                )
                
                if exit_triggered:
                    # Close position
                    trade = open_position
                    trade.exit_date = current_date
                    trade.exit_price = exit_price
                    trade.exit_reason = exit_reason
                    trade.holding_days = (current_date - trade.entry_date).days
                    
                    # Calculate P&L
                    trade.profit_loss = (trade.exit_price - trade.entry_price) * trade.position_size
                    trade.profit_loss_pct = (trade.exit_price - trade.entry_price) / trade.entry_price
                    
                    # Risk/Reward
                    risk = abs(trade.entry_price - trade.stop_loss)
                    actual_reward = trade.exit_price - trade.entry_price
                    trade.risk_reward_actual = actual_reward / risk if risk > 0 else 0
                    
                    # Update capital
                    self.current_capital += trade.profit_loss
                    self.equity_curve.append(self.current_capital)
                    
                    self.trades.append(trade)
                    open_position = None
                    last_exit_date = current_date
            
            # Check for new entry
            elif df['Signal'].iloc[i] == 1:
                # Avoid re-entry too quickly
                if last_exit_date and (current_date - last_exit_date).days < 2:
                    continue
                
                # Calculate stops and targets
                atr = df['ATR'].iloc[i]
                
                # Dynamic stop based on volatility and support
                atr_stop = current_price - (1.5 * atr)
                support_stop = df['Support_20'].iloc[i] * 0.995
                stop_loss = max(atr_stop, support_stop)
                
                # Ensure minimum risk/reward
                min_risk = current_price * 0.01  # At least 1% risk
                if current_price - stop_loss < min_risk:
                    stop_loss = current_price - min_risk
                
                # Calculate targets
                risk = current_price - stop_loss
                take_profit_1 = current_price + (1.8 * risk)  # 1.8:1 R/R
                take_profit_2 = current_price + (3.0 * risk)  # 3.0:1 R/R
                
                # Position sizing
                vol_percentile = df['Vol_Percentile'].iloc[i]
                position_size, position_value = self.calculate_position_size(
                    current_price, stop_loss, self.current_capital, vol_percentile
                )
                
                if position_size > 0 and position_value <= self.current_capital * 0.95:
                    # Create trade
                    trade = FinalTrade(
                        entry_date=current_date,
                        exit_date=None,
                        entry_price=current_price,
                        exit_price=None,
                        stop_loss=stop_loss,
                        take_profit_1=take_profit_1,
                        take_profit_2=take_profit_2,
                        position_size=position_size,
                        position_value=position_value,
                        signal_strength=df['Signal_Strength'].iloc[i],
                        trend_strength=df['Trend_Strength'].iloc[i],
                        momentum_score=df['Momentum_Score'].iloc[i],
                        holding_days=None,
                        exit_reason=None,
                        profit_loss=None,
                        profit_loss_pct=None,
                        risk_reward_actual=None,
                        max_profit_pct=None
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
            
            risk = abs(trade.entry_price - trade.stop_loss)
            actual_reward = trade.exit_price - trade.entry_price
            trade.risk_reward_actual = actual_reward / risk if risk > 0 else 0
            
            self.current_capital += trade.profit_loss
            self.equity_curve.append(self.current_capital)
            self.trades.append(trade)
        
        return self.calculate_metrics(symbol)
    
    def calculate_metrics(self, symbol: str) -> Dict:
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
        
        # Risk/Reward
        winning_rr = []
        for trade in winning_trades:
            if trade.risk_reward_actual is not None and trade.risk_reward_actual > 0:
                winning_rr.append(trade.risk_reward_actual)
        
        avg_winning_rr = np.mean(winning_rr) if winning_rr else 0
        
        # Holding periods
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
        
        # Calculate annual return (approximation)
        years = 3  # We're using 3 years of data
        annual_return = ((self.current_capital / self.initial_capital) ** (1/years) - 1) * 100
        
        # Sharpe ratio (simplified)
        if len(self.equity_curve) > 1:
            returns = np.diff(self.equity_curve) / self.equity_curve[:-1]
            if np.std(returns) > 0:
                sharpe = np.sqrt(252/20) * np.mean(returns) / np.std(returns)  # Assuming ~20 trades/year
            else:
                sharpe = 0
        else:
            sharpe = 0
        
        # Win streaks
        max_wins = 0
        max_losses = 0
        current_wins = 0
        current_losses = 0
        
        for trade in self.trades:
            if trade.profit_loss > 0:
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)
        
        return {
            'symbol': symbol,
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': round(win_rate, 2),
            'profit_factor': round(profit_factor, 2),
            'total_profit': round(total_profit, 2),
            'total_return_pct': round(total_return, 2),
            'annual_return_pct': round(annual_return, 2),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'avg_win_pct': round(avg_win_pct, 2),
            'avg_loss_pct': round(avg_loss_pct, 2),
            'avg_holding_days': round(avg_holding, 1),
            'avg_winning_rr': round(avg_winning_rr, 2),
            'max_consecutive_wins': max_wins,
            'max_consecutive_losses': max_losses,
            'max_drawdown_pct': round(max_dd, 2),
            'sharpe_ratio': round(sharpe, 2),
            'exit_reasons': exit_reasons,
            'trades_per_year': round(total_trades / years, 1),
            'initial_capital': self.initial_capital,
            'final_capital': round(self.current_capital, 2)
        }


def main():
    """Run final optimized swing trading system."""
    symbols = ['AAPL', 'GOOGL', 'TSLA', 'MSFT', 'UNH']
    
    print("🎯 Final Optimized Swing Trading System")
    print("="*60)
    print("Strategy Features:")
    print("• Balanced signal generation")
    print("• Dynamic position sizing")
    print("• Optimized entry/exit rules")
    print("• 2-15 day holding period")
    print("• Risk-adjusted for volatility")
    print("="*60)
    
    all_results = {}
    
    for symbol in symbols:
        print(f"\nAnalyzing {symbol}...")
        
        system = FinalSwingSystem(
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
    print("📊 FINAL SYSTEM PERFORMANCE RESULTS")
    print("="*60)
    
    # Sort by total return
    sorted_results = sorted(all_results.items(), key=lambda x: x[1]['total_return_pct'], reverse=True)
    
    for symbol, metrics in sorted_results:
        print(f"\n{symbol}:")
        print(f"  Total Return: {metrics['total_return_pct']}% (Annual: {metrics['annual_return_pct']}%)")
        print(f"  Win Rate: {metrics['win_rate']}%")
        print(f"  Profit Factor: {metrics['profit_factor']}")
        print(f"  Trades: {metrics['total_trades']} ({metrics['trades_per_year']}/year)")
        print(f"  Avg Win: {metrics['avg_win_pct']}% | Avg Loss: {metrics['avg_loss_pct']}%")
        print(f"  Win R/R: {metrics['avg_winning_rr']}")
        print(f"  Max Drawdown: {metrics['max_drawdown_pct']}%")
        print(f"  Sharpe Ratio: {metrics['sharpe_ratio']}")
        print(f"  Exit Breakdown: {metrics['exit_reasons']}")
    
    # Portfolio summary
    print("\n" + "="*60)
    print("💎 PORTFOLIO PERFORMANCE SUMMARY")
    print("="*60)
    
    avg_return = np.mean([r['total_return_pct'] for r in all_results.values()])
    avg_annual = np.mean([r['annual_return_pct'] for r in all_results.values()])
    avg_win_rate = np.mean([r['win_rate'] for r in all_results.values()])
    avg_profit_factor = np.mean([r['profit_factor'] for r in all_results.values()])
    total_trades = sum(r['total_trades'] for r in all_results.values())
    
    print(f"Average Total Return: {avg_return:.2f}%")
    print(f"Average Annual Return: {avg_annual:.2f}%")
    print(f"Average Win Rate: {avg_win_rate:.2f}%")
    print(f"Average Profit Factor: {avg_profit_factor:.2f}")
    print(f"Total Trades: {total_trades}")
    
    # Best and worst
    best = max(sorted_results, key=lambda x: x[1]['total_return_pct'])
    worst = min(sorted_results, key=lambda x: x[1]['total_return_pct'])
    print(f"\nBest Performer: {best[0]} ({best[1]['total_return_pct']}%)")
    print(f"Worst Performer: {worst[0]} ({worst[1]['total_return_pct']}%)")
    
    # Save results
    filename = f"final_swing_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nDetailed results saved to {filename}")


if __name__ == "__main__":
    main()