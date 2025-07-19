#!/usr/bin/env python3
"""
Optimized Swing Trading System
Long-only positions with 2-15 day holding period
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
class SwingTrade:
    """Represents a swing trade."""
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
    holding_days: Optional[int]
    exit_reason: Optional[str]
    profit_loss: Optional[float]
    profit_loss_pct: Optional[float]
    risk_reward_actual: Optional[float]

class OptimizedSwingSystem:
    def __init__(self, data_dir: str = "historical_data", 
                 initial_capital: float = 10000,
                 risk_per_trade: float = 0.02,
                 max_position_pct: float = 0.30,  # 30% max per position
                 min_holding_days: int = 2,
                 max_holding_days: int = 15):
        """Initialize optimized swing trading system."""
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
    
    def calculate_swing_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate indicators optimized for 2-15 day swings."""
        # Short-term EMAs for quick entries
        df['EMA_5'] = df['Close'].ewm(span=5, adjust=False).mean()
        df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
        df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
        
        # Trend filter
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        
        # ATR for volatility-based stops
        if 'ATR' not in df.columns:
            high_low = df['High'] - df['Low']
            high_close = abs(df['High'] - df['Close'].shift())
            low_close = abs(df['Low'] - df['Close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            df['ATR'] = true_range.rolling(10).mean()  # Faster ATR for swing trades
        
        # RSI with shorter period for swing trades
        if 'RSI' not in df.columns:
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=10).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=10).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD for momentum
        if 'MACD' not in df.columns:
            exp1 = df['Close'].ewm(span=12, adjust=False).mean()
            exp2 = df['Close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = exp1 - exp2
            df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
            df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # Volume surge detection
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        # Price momentum
        df['ROC_5'] = (df['Close'] - df['Close'].shift(5)) / df['Close'].shift(5) * 100
        df['ROC_10'] = (df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10) * 100
        
        # Swing highs and lows
        df['Swing_High'] = df['High'].rolling(window=5, center=True).max() == df['High']
        df['Swing_Low'] = df['Low'].rolling(window=5, center=True).min() == df['Low']
        
        # Support and Resistance (10-day for swing trades)
        df['Resistance_10'] = df['High'].rolling(window=10).max()
        df['Support_10'] = df['Low'].rolling(window=10).min()
        df['SR_Position'] = (df['Close'] - df['Support_10']) / (df['Resistance_10'] - df['Support_10'])
        
        # Bollinger Bands for mean reversion
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        df['BB_Std'] = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (2 * df['BB_Std'])
        df['BB_Lower'] = df['BB_Middle'] - (2 * df['BB_Std'])
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        return df
    
    def generate_swing_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate signals optimized for 2-15 day holding period."""
        df = self.calculate_swing_indicators(df)
        
        df['Signal'] = 0
        df['Signal_Strength'] = 0
        df['Setup_Quality'] = ''
        
        for i in range(200, len(df)):  # Need 200 days for SMA
            score = 0
            setup_notes = []
            
            # 1. Primary trend filter (LONG ONLY - must be in uptrend)
            if df['SMA_50'].iloc[i] > df['SMA_200'].iloc[i]:
                score += 20
                setup_notes.append('uptrend')
            else:
                # Skip if not in uptrend
                continue
            
            # 2. Price above key MAs
            if df['Close'].iloc[i] > df['SMA_50'].iloc[i]:
                score += 10
            
            # 3. Pullback to support (key for swing trades)
            sr_pos = df['SR_Position'].iloc[i]
            if 0.2 < sr_pos < 0.4:  # Near support but not too low
                score += 20
                setup_notes.append('pullback_support')
            
            # 4. RSI conditions for swing trades
            rsi = df['RSI'].iloc[i]
            rsi_prev = df['RSI'].iloc[i-1]
            
            # Oversold bounce
            if 30 < rsi < 50 and rsi > rsi_prev:
                score += 15
                setup_notes.append('rsi_bounce')
            # Mid-range momentum
            elif 50 < rsi < 65:
                score += 10
            
            # 5. EMA alignment for entry
            if (df['EMA_5'].iloc[i] > df['EMA_10'].iloc[i] and 
                df['EMA_5'].iloc[i-1] <= df['EMA_10'].iloc[i-1]):  # Crossover
                score += 15
                setup_notes.append('ema_cross')
            
            # 6. MACD momentum
            if df['MACD'].iloc[i] > df['MACD_Signal'].iloc[i]:
                score += 10
                # Histogram increasing
                if df['MACD_Histogram'].iloc[i] > df['MACD_Histogram'].iloc[i-1]:
                    score += 5
                    setup_notes.append('macd_momentum')
            
            # 7. Volume confirmation
            if df['Volume_Ratio'].iloc[i] > 1.3:  # 30% above average
                score += 10
                setup_notes.append('volume_surge')
            
            # 8. Price momentum (not too extended)
            roc_5 = df['ROC_5'].iloc[i]
            if -3 < roc_5 < 2:  # Slight pullback to flat
                score += 10
                setup_notes.append('good_entry')
            
            # 9. Bollinger Band position
            bb_pos = df['BB_Position'].iloc[i]
            if 0.2 < bb_pos < 0.5:  # Lower half of bands
                score += 5
            
            # Store results
            df.loc[df.index[i], 'Signal_Strength'] = score
            df.loc[df.index[i], 'Setup_Quality'] = ','.join(setup_notes)
            
            # Generate signal with higher threshold for quality
            if score >= 70:
                df.loc[df.index[i], 'Signal'] = 1
        
        return df
    
    def calculate_position_size(self, entry_price: float, stop_loss: float, 
                              current_capital: float) -> Tuple[int, float]:
        """Calculate position size with proper risk management."""
        # Risk amount
        risk_amount = current_capital * self.risk_per_trade
        
        # Risk per share
        risk_per_share = abs(entry_price - stop_loss)
        if risk_per_share == 0:
            return 0, 0
        
        # Position size based on risk
        shares_by_risk = int(risk_amount / risk_per_share)
        
        # Maximum position size constraint
        max_position_value = current_capital * self.max_position_pct
        shares_by_capital = int(max_position_value / entry_price)
        
        # Use the smaller of the two
        shares = min(shares_by_risk, shares_by_capital)
        position_value = shares * entry_price
        
        # Ensure we have enough capital
        if position_value > current_capital * 0.95:
            shares = int(current_capital * 0.95 / entry_price)
            position_value = shares * entry_price
        
        return shares, position_value
    
    def backtest(self, symbol: str, df: pd.DataFrame) -> Dict:
        """Run backtest optimized for swing trading."""
        # Generate signals
        df = self.generate_swing_signals(df)
        
        # Reset state
        self.trades = []
        self.current_capital = self.initial_capital
        self.equity_curve = [self.initial_capital]
        open_position = None
        
        for i in range(200, len(df)):
            current_date = df.index[i]
            current_price = df['Close'].iloc[i]
            
            # Check open position
            if open_position:
                trade = open_position
                days_held = (current_date - trade.entry_date).days
                
                # Exit conditions
                exit_triggered = False
                exit_price = current_price
                exit_reason = None
                
                # Check stop loss
                if df['Low'].iloc[i] <= trade.stop_loss:
                    exit_triggered = True
                    exit_price = trade.stop_loss
                    exit_reason = 'stop_loss'
                
                # Check take profit levels
                elif df['High'].iloc[i] >= trade.take_profit_2:
                    exit_triggered = True
                    exit_price = trade.take_profit_2
                    exit_reason = 'take_profit_2'
                elif df['High'].iloc[i] >= trade.take_profit_1:
                    # Partial exit logic - for now, full exit
                    exit_triggered = True
                    exit_price = trade.take_profit_1
                    exit_reason = 'take_profit_1'
                
                # Time-based exits
                elif days_held >= self.max_holding_days:
                    exit_triggered = True
                    exit_reason = 'max_days'
                
                # Minimum holding period check
                elif days_held >= self.min_holding_days:
                    # Trailing stop for winners
                    if current_price > trade.entry_price * 1.03:  # 3% profit
                        trailing_stop = current_price - (1.5 * df['ATR'].iloc[i])
                        if current_price <= trailing_stop:
                            exit_triggered = True
                            exit_reason = 'trailing_stop'
                    
                    # Exit on weakness
                    if df['EMA_5'].iloc[i] < df['EMA_10'].iloc[i]:
                        if current_price < trade.entry_price:  # Only if losing
                            exit_triggered = True
                            exit_reason = 'weakness_exit'
                
                if exit_triggered:
                    # Close position
                    trade.exit_date = current_date
                    trade.exit_price = exit_price
                    trade.exit_reason = exit_reason
                    trade.holding_days = days_held
                    
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
            elif df['Signal'].iloc[i] == 1:
                # Additional entry filters
                # Don't enter if RSI > 70 (overbought)
                if df['RSI'].iloc[i] > 70:
                    continue
                
                # Don't enter if too extended from 20 EMA
                distance_from_ema20 = (current_price - df['EMA_20'].iloc[i]) / df['EMA_20'].iloc[i]
                if distance_from_ema20 > 0.05:  # More than 5% above
                    continue
                
                # Calculate dynamic stop loss
                atr = df['ATR'].iloc[i]
                
                # Tighter stop for swing trades
                stop_loss = current_price - (1.5 * atr)
                
                # Support-based stop
                recent_support = df['Support_10'].iloc[i]
                support_stop = recent_support - (0.002 * recent_support)  # 0.2% below support
                
                # Use the higher stop (less risk)
                stop_loss = max(stop_loss, support_stop)
                
                # Take profit levels for swing trades
                risk = current_price - stop_loss
                take_profit_1 = current_price + (1.5 * risk)  # 1.5:1 R/R
                take_profit_2 = current_price + (2.5 * risk)  # 2.5:1 R/R
                
                # Calculate position size
                position_size, position_value = self.calculate_position_size(
                    current_price, stop_loss, self.current_capital
                )
                
                if position_size > 0:
                    # Create trade
                    trade = SwingTrade(
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
            
            risk = abs(trade.entry_price - trade.stop_loss)
            actual_reward = trade.exit_price - trade.entry_price
            trade.risk_reward_actual = actual_reward / risk if risk > 0 else 0
            
            self.current_capital += trade.profit_loss
            self.equity_curve.append(self.current_capital)
            self.trades.append(trade)
        
        return self.calculate_metrics(symbol)
    
    def calculate_metrics(self, symbol: str) -> Dict:
        """Calculate comprehensive metrics for swing trading."""
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
        
        # Trade statistics
        avg_win = sum(t.profit_loss for t in winning_trades) / len(winning_trades) if winning_trades else 0
        avg_loss = sum(t.profit_loss for t in losing_trades) / len(losing_trades) if losing_trades else 0
        avg_win_pct = sum(t.profit_loss_pct for t in winning_trades) / len(winning_trades) * 100 if winning_trades else 0
        avg_loss_pct = sum(t.profit_loss_pct for t in losing_trades) / len(losing_trades) * 100 if losing_trades else 0
        
        # Holding period analysis
        holding_days = [t.holding_days for t in self.trades if t.holding_days is not None]
        avg_holding_days = np.mean(holding_days) if holding_days else 0
        
        winning_days = [t.holding_days for t in winning_trades if t.holding_days is not None]
        losing_days = [t.holding_days for t in losing_trades if t.holding_days is not None]
        avg_win_days = np.mean(winning_days) if winning_days else 0
        avg_loss_days = np.mean(losing_days) if losing_days else 0
        
        # Risk metrics
        gross_profit = sum(t.profit_loss for t in winning_trades) if winning_trades else 0
        gross_loss = abs(sum(t.profit_loss for t in losing_trades)) if losing_trades else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Risk/Reward
        rr_actual = []
        for trade in self.trades:
            if trade.risk_reward_actual is not None and trade.profit_loss > 0:
                rr_actual.append(trade.risk_reward_actual)
        
        avg_winning_rr = np.mean(rr_actual) if rr_actual else 0
        
        # Exit analysis
        exit_reasons = {}
        for trade in self.trades:
            reason = trade.exit_reason or 'unknown'
            exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
        
        # Expectancy
        expectancy = (win_rate/100 * avg_win) + ((100-win_rate)/100 * avg_loss)
        
        # Maximum drawdown
        peak = self.equity_curve[0]
        max_drawdown = 0
        max_drawdown_pct = 0
        
        for value in self.equity_curve:
            if value > peak:
                peak = value
            drawdown = peak - value
            drawdown_pct = drawdown / peak * 100 if peak > 0 else 0
            max_drawdown_pct = max(max_drawdown_pct, drawdown_pct)
        
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
            'expectancy': round(expectancy, 2),
            'avg_holding_days': round(avg_holding_days, 1),
            'avg_win_days': round(avg_win_days, 1),
            'avg_loss_days': round(avg_loss_days, 1),
            'avg_winning_rr': round(avg_winning_rr, 2),
            'max_drawdown_pct': round(max_drawdown_pct, 2),
            'exit_reasons': exit_reasons,
            'initial_capital': self.initial_capital,
            'final_capital': round(self.current_capital, 2)
        }


def main():
    """Run optimized swing trading system."""
    symbols = ['AAPL', 'GOOGL', 'TSLA', 'MSFT', 'UNH']
    
    print("🎯 Optimized Swing Trading System (2-15 Days)")
    print("="*60)
    print("Strategy Rules:")
    print("• Long-only positions")
    print("• Hold 2-15 days")
    print("• Enter on pullbacks in uptrends")
    print("• 2% risk per trade")
    print("• Multiple profit targets")
    print("="*60)
    
    all_results = {}
    
    for symbol in symbols:
        print(f"\nTesting {symbol}...")
        
        system = OptimizedSwingSystem(
            initial_capital=10000,
            risk_per_trade=0.02,
            max_position_pct=0.30,
            min_holding_days=2,
            max_holding_days=15
        )
        
        df = system.load_data(symbol)
        if not df.empty:
            results = system.backtest(symbol, df)
            all_results[symbol] = results
            
            # Save trade details
            if system.trades:
                trades_data = []
                for t in system.trades:
                    trades_data.append({
                        'entry_date': t.entry_date.strftime('%Y-%m-%d'),
                        'exit_date': t.exit_date.strftime('%Y-%m-%d') if t.exit_date else None,
                        'holding_days': t.holding_days,
                        'entry_price': round(t.entry_price, 2),
                        'exit_price': round(t.exit_price, 2) if t.exit_price else None,
                        'profit_pct': round(t.profit_loss_pct * 100, 2) if t.profit_loss_pct else None,
                        'exit_reason': t.exit_reason
                    })
                
                filename = f"swing_trades_{symbol}_{datetime.now().strftime('%Y%m%d')}.json"
                with open(filename, 'w') as f:
                    json.dump(trades_data, f, indent=2)
    
    # Display results
    print("\n" + "="*60)
    print("📊 SWING TRADING RESULTS (2-15 DAY HOLDING)")
    print("="*60)
    
    # Sort by return
    sorted_results = sorted(all_results.items(), key=lambda x: x[1]['total_return_pct'], reverse=True)
    
    for symbol, metrics in sorted_results:
        print(f"\n{symbol}:")
        print(f"  Total Trades: {metrics['total_trades']}")
        print(f"  Win Rate: {metrics['win_rate']}%")
        print(f"  Total Return: {metrics['total_return_pct']}%")
        print(f"  Profit Factor: {metrics['profit_factor']}")
        print(f"  Avg Holding: {metrics['avg_holding_days']} days")
        print(f"  Avg Win: {metrics['avg_win_pct']}% in {metrics['avg_win_days']} days")
        print(f"  Avg Loss: {metrics['avg_loss_pct']}% in {metrics['avg_loss_days']} days")
        print(f"  Win R/R: {metrics['avg_winning_rr']}")
        print(f"  Expectancy: ${metrics['expectancy']}")
        print(f"  Exit Reasons: {metrics['exit_reasons']}")
    
    # Portfolio summary
    total_trades = sum(r['total_trades'] for r in all_results.values())
    avg_win_rate = np.mean([r['win_rate'] for r in all_results.values()])
    avg_return = np.mean([r['total_return_pct'] for r in all_results.values()])
    
    print("\n" + "="*60)
    print("📈 PORTFOLIO SUMMARY")
    print("="*60)
    print(f"Total Trades: {total_trades}")
    print(f"Average Win Rate: {avg_win_rate:.2f}%")
    print(f"Average Return: {avg_return:.2f}%")
    
    # Save summary
    summary_file = f"swing_trading_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {summary_file}")


if __name__ == "__main__":
    main()