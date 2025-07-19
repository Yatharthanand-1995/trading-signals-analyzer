#!/usr/bin/env python3
"""
Improved Swing Trading System with Better Risk Management
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
class Trade:
    """Represents a single trade."""
    entry_date: datetime
    exit_date: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    stop_loss: float
    take_profit: float
    position_size: int
    position_value: float
    trade_type: str
    signal_strength: float
    exit_reason: Optional[str]
    profit_loss: Optional[float]
    profit_loss_pct: Optional[float]
    risk_reward_actual: Optional[float]

class ImprovedSwingSystem:
    def __init__(self, data_dir: str = "historical_data", 
                 initial_capital: float = 10000,
                 risk_per_trade: float = 0.02,
                 max_position_pct: float = 0.25):  # Max 25% per position
        """Initialize the improved swing trading system."""
        self.data_dir = data_dir
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.max_position_pct = max_position_pct
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
        """Calculate swing trading indicators."""
        # Price action
        df['HL2'] = (df['High'] + df['Low']) / 2
        df['HLC3'] = (df['High'] + df['Low'] + df['Close']) / 3
        
        # ATR for volatility
        if 'ATR' not in df.columns:
            high_low = df['High'] - df['Low']
            high_close = abs(df['High'] - df['Close'].shift())
            low_close = abs(df['Low'] - df['Close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            df['ATR'] = true_range.rolling(14).mean()
        
        # Moving averages
        df['EMA_8'] = df['Close'].ewm(span=8, adjust=False).mean()
        df['EMA_21'] = df['Close'].ewm(span=21, adjust=False).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        
        # RSI
        if 'RSI' not in df.columns:
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        if 'MACD' not in df.columns:
            exp1 = df['Close'].ewm(span=12, adjust=False).mean()
            exp2 = df['Close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = exp1 - exp2
            df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
            df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # Volume analysis
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        # Support and Resistance
        df['Resistance'] = df['High'].rolling(window=20).max()
        df['Support'] = df['Low'].rolling(window=20).min()
        df['SR_Range'] = df['Resistance'] - df['Support']
        
        # Price position within range
        df['Price_Position'] = (df['Close'] - df['Support']) / df['SR_Range']
        
        return df
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate improved swing trading signals."""
        df = self.calculate_indicators(df)
        
        df['Signal'] = 0
        df['Signal_Strength'] = 0
        
        for i in range(50, len(df)):
            score = 0
            
            # 1. Trend alignment (30 points)
            if df['EMA_8'].iloc[i] > df['EMA_21'].iloc[i] > df['SMA_50'].iloc[i]:
                score += 30
            elif df['EMA_8'].iloc[i] < df['EMA_21'].iloc[i] < df['SMA_50'].iloc[i]:
                score -= 30
            
            # 2. RSI conditions (20 points)
            rsi = df['RSI'].iloc[i]
            if 40 < rsi < 60:  # Neutral zone, trend continuation
                if df['EMA_8'].iloc[i] > df['EMA_21'].iloc[i]:
                    score += 10
                else:
                    score -= 10
            elif rsi < 30:  # Oversold
                score += 20
            elif rsi > 70:  # Overbought
                score -= 20
            
            # 3. MACD momentum (20 points)
            if df['MACD'].iloc[i] > df['MACD_Signal'].iloc[i]:
                score += 10
                # Histogram increasing
                if df['MACD_Histogram'].iloc[i] > df['MACD_Histogram'].iloc[i-1]:
                    score += 10
            else:
                score -= 10
                if df['MACD_Histogram'].iloc[i] < df['MACD_Histogram'].iloc[i-1]:
                    score -= 10
            
            # 4. Volume confirmation (15 points)
            if df['Volume_Ratio'].iloc[i] > 1.2:
                if score > 0:
                    score += 15  # Volume confirms bullish signal
                else:
                    score -= 15  # Volume confirms bearish signal
            
            # 5. Support/Resistance (15 points)
            price_pos = df['Price_Position'].iloc[i]
            if price_pos < 0.25:  # Near support
                score += 15
            elif price_pos > 0.75:  # Near resistance
                score -= 15
            
            # 6. Price action patterns (bonus points)
            # Bullish: Higher lows
            if df['Low'].iloc[i] > df['Low'].iloc[i-5] and df['Low'].iloc[i-5] > df['Low'].iloc[i-10]:
                score += 10
            # Bearish: Lower highs
            elif df['High'].iloc[i] < df['High'].iloc[i-5] and df['High'].iloc[i-5] < df['High'].iloc[i-10]:
                score -= 10
            
            df.loc[df.index[i], 'Signal_Strength'] = score
            
            # Generate signals with threshold
            if score >= 60:
                df.loc[df.index[i], 'Signal'] = 1  # Strong Buy
            elif score <= -60:
                df.loc[df.index[i], 'Signal'] = -1  # Strong Sell
        
        return df
    
    def calculate_position_size(self, entry_price: float, stop_loss: float, 
                              current_capital: float) -> Tuple[int, float]:
        """Calculate position size with improved risk management."""
        # Risk amount
        risk_amount = current_capital * self.risk_per_trade
        
        # Risk per share
        risk_per_share = abs(entry_price - stop_loss)
        if risk_per_share == 0:
            return 0, 0
        
        # Position size based on risk
        shares_by_risk = int(risk_amount / risk_per_share)
        
        # Maximum position size (% of capital)
        max_position_value = current_capital * self.max_position_pct
        shares_by_capital = int(max_position_value / entry_price)
        
        # Use the smaller of the two
        shares = min(shares_by_risk, shares_by_capital)
        position_value = shares * entry_price
        
        return shares, position_value
    
    def backtest(self, symbol: str, df: pd.DataFrame) -> Dict:
        """Run improved backtest."""
        # Generate signals
        df = self.generate_signals(df)
        
        # Reset state
        self.trades = []
        self.current_capital = self.initial_capital
        self.equity_curve = [self.initial_capital]
        open_position = None
        
        for i in range(50, len(df)):
            current_date = df.index[i]
            current_price = df['Close'].iloc[i]
            
            # Check if we have an open position
            if open_position:
                trade = open_position
                
                # Exit conditions
                exit_triggered = False
                exit_price = current_price
                exit_reason = None
                
                # Stop loss
                if df['Low'].iloc[i] <= trade.stop_loss:
                    exit_triggered = True
                    exit_price = trade.stop_loss
                    exit_reason = 'stop_loss'
                
                # Take profit
                elif df['High'].iloc[i] >= trade.take_profit:
                    exit_triggered = True
                    exit_price = trade.take_profit
                    exit_reason = 'take_profit'
                
                # Trailing stop (if in profit)
                elif current_price > trade.entry_price * 1.02:  # 2% profit
                    trailing_stop = current_price - (1.5 * df['ATR'].iloc[i])
                    if current_price <= trailing_stop:
                        exit_triggered = True
                        exit_reason = 'trailing_stop'
                
                # Signal reversal
                elif df['Signal'].iloc[i] == -1:
                    exit_triggered = True
                    exit_reason = 'signal_reversal'
                
                # Time stop (20 days)
                elif (current_date - trade.entry_date).days >= 20:
                    exit_triggered = True
                    exit_reason = 'time_stop'
                
                if exit_triggered:
                    # Close position
                    trade.exit_date = current_date
                    trade.exit_price = exit_price
                    trade.exit_reason = exit_reason
                    
                    # Calculate P&L
                    trade.profit_loss = (exit_price - trade.entry_price) * trade.position_size
                    trade.profit_loss_pct = (exit_price - trade.entry_price) / trade.entry_price
                    
                    # Risk/Reward
                    risk = abs(trade.entry_price - trade.stop_loss)
                    actual_reward = exit_price - trade.entry_price
                    trade.risk_reward_actual = actual_reward / risk if risk > 0 else 0
                    
                    # Update capital
                    self.current_capital += trade.profit_loss
                    self.equity_curve.append(self.current_capital)
                    
                    self.trades.append(trade)
                    open_position = None
            
            # Check for new entry (only if no position)
            elif df['Signal'].iloc[i] == 1 and df['Signal_Strength'].iloc[i] >= 60:
                # Calculate stop loss and take profit
                atr = df['ATR'].iloc[i]
                
                # Dynamic stop loss based on ATR
                stop_loss = current_price - (1.5 * atr)
                
                # Multiple take profit levels
                take_profit_1 = current_price + (2 * atr)
                take_profit_2 = current_price + (3 * atr)
                take_profit_3 = current_price + (4 * atr)
                
                # Use first target for now
                take_profit = take_profit_1
                
                # Calculate position size
                position_size, position_value = self.calculate_position_size(
                    current_price, stop_loss, self.current_capital
                )
                
                if position_size > 0 and position_value <= self.current_capital * 0.95:
                    # Create trade
                    trade = Trade(
                        entry_date=current_date,
                        exit_date=None,
                        entry_price=current_price,
                        exit_price=None,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        position_size=position_size,
                        position_value=position_value,
                        trade_type='LONG',
                        signal_strength=df['Signal_Strength'].iloc[i],
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
        
        # Basic counts
        total_trades = len(self.trades)
        winning_trades = [t for t in self.trades if t.profit_loss > 0]
        losing_trades = [t for t in self.trades if t.profit_loss <= 0]
        
        # Win rate
        win_rate = len(winning_trades) / total_trades * 100
        
        # Financial metrics
        total_profit = sum(t.profit_loss for t in self.trades)
        total_return = (self.current_capital - self.initial_capital) / self.initial_capital * 100
        
        # Average trade metrics
        avg_win = sum(t.profit_loss for t in winning_trades) / len(winning_trades) if winning_trades else 0
        avg_loss = sum(t.profit_loss for t in losing_trades) / len(losing_trades) if losing_trades else 0
        avg_win_pct = sum(t.profit_loss_pct for t in winning_trades) / len(winning_trades) * 100 if winning_trades else 0
        avg_loss_pct = sum(t.profit_loss_pct for t in losing_trades) / len(losing_trades) * 100 if losing_trades else 0
        
        # Profit factor
        gross_profit = sum(t.profit_loss for t in winning_trades) if winning_trades else 0
        gross_loss = abs(sum(t.profit_loss for t in losing_trades)) if losing_trades else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Risk/Reward
        rr_planned = []
        rr_actual = []
        
        for trade in self.trades:
            # Planned R/R
            risk = abs(trade.entry_price - trade.stop_loss)
            reward = abs(trade.take_profit - trade.entry_price)
            planned_rr = reward / risk if risk > 0 else 0
            rr_planned.append(planned_rr)
            
            # Actual R/R
            if trade.risk_reward_actual is not None:
                rr_actual.append(trade.risk_reward_actual)
        
        avg_planned_rr = np.mean(rr_planned) if rr_planned else 0
        avg_actual_rr = np.mean(rr_actual) if rr_actual else 0
        
        # Win/Loss streaks
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        current_wins = 0
        current_losses = 0
        
        for trade in self.trades:
            if trade.profit_loss > 0:
                current_wins += 1
                current_losses = 0
                max_consecutive_wins = max(max_consecutive_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, current_losses)
        
        # Maximum drawdown
        peak = self.equity_curve[0]
        max_drawdown = 0
        max_drawdown_pct = 0
        
        for value in self.equity_curve:
            if value > peak:
                peak = value
            drawdown = peak - value
            drawdown_pct = drawdown / peak * 100 if peak > 0 else 0
            if drawdown_pct > max_drawdown_pct:
                max_drawdown = drawdown
                max_drawdown_pct = drawdown_pct
        
        # Exit reasons
        exit_reasons = {}
        for trade in self.trades:
            reason = trade.exit_reason or 'unknown'
            exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
        
        # Sharpe ratio (simplified)
        returns = []
        for i in range(1, len(self.equity_curve)):
            ret = (self.equity_curve[i] - self.equity_curve[i-1]) / self.equity_curve[i-1]
            returns.append(ret)
        
        if returns:
            avg_return = np.mean(returns)
            std_return = np.std(returns)
            sharpe_ratio = (avg_return * np.sqrt(252)) / (std_return * np.sqrt(252)) if std_return > 0 else 0
        else:
            sharpe_ratio = 0
        
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
            'max_consecutive_wins': max_consecutive_wins,
            'max_consecutive_losses': max_consecutive_losses,
            'max_drawdown': round(max_drawdown, 2),
            'max_drawdown_pct': round(max_drawdown_pct, 2),
            'sharpe_ratio': round(sharpe_ratio, 2),
            'risk_reward_ratios': {
                'planned_avg': round(avg_planned_rr, 2),
                'actual_avg': round(avg_actual_rr, 2)
            },
            'exit_reasons': exit_reasons,
            'initial_capital': self.initial_capital,
            'final_capital': round(self.current_capital, 2)
        }


def main():
    """Run improved swing trading backtest."""
    symbols = ['AAPL', 'GOOGL', 'TSLA', 'MSFT', 'UNH']
    
    print("🚀 Improved Swing Trading System")
    print("="*60)
    print("Strategy Features:")
    print("• Multi-indicator confirmation")
    print("• Dynamic position sizing")
    print("• Trailing stops for winners")
    print("• Maximum 25% per position")
    print("• 2% risk per trade")
    print("="*60)
    
    all_results = {}
    
    for symbol in symbols:
        print(f"\nTesting {symbol}...")
        
        system = ImprovedSwingSystem(
            initial_capital=10000,
            risk_per_trade=0.02,
            max_position_pct=0.25
        )
        
        df = system.load_data(symbol)
        if not df.empty:
            results = system.backtest(symbol, df)
            all_results[symbol] = results
    
    # Display results
    print("\n" + "="*60)
    print("📊 SWING TRADING PERFORMANCE SUMMARY")
    print("="*60)
    
    best_performer = None
    best_return = -float('inf')
    
    for symbol, metrics in all_results.items():
        print(f"\n{symbol}:")
        print(f"  Win Rate: {metrics['win_rate']}%")
        print(f"  Total Return: {metrics['total_return_pct']}%")
        print(f"  Risk/Reward (Actual): {metrics['risk_reward_ratios']['actual_avg']}")
        print(f"  Profit Factor: {metrics['profit_factor']}")
        print(f"  Max Drawdown: {metrics['max_drawdown_pct']}%")
        print(f"  Sharpe Ratio: {metrics['sharpe_ratio']}")
        
        if metrics['total_return_pct'] > best_return:
            best_return = metrics['total_return_pct']
            best_performer = symbol
    
    # Portfolio summary
    print("\n" + "="*60)
    print("📈 PORTFOLIO SUMMARY")
    print("="*60)
    
    avg_win_rate = np.mean([r['win_rate'] for r in all_results.values()])
    avg_return = np.mean([r['total_return_pct'] for r in all_results.values()])
    avg_profit_factor = np.mean([r['profit_factor'] for r in all_results.values()])
    
    print(f"Average Win Rate: {avg_win_rate:.2f}%")
    print(f"Average Return: {avg_return:.2f}%")
    print(f"Average Profit Factor: {avg_profit_factor:.2f}")
    print(f"Best Performer: {best_performer} ({best_return:.2f}%)")
    
    # Save detailed results
    filename = f"improved_swing_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nDetailed results saved to {filename}")


if __name__ == "__main__":
    main()