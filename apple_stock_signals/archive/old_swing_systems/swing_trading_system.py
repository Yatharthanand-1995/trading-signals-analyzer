#!/usr/bin/env python3
"""
Enhanced Swing Trading System
Uses historical data to optimize swing trading strategies with backtesting
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
    trade_type: str  # 'LONG' or 'SHORT'
    signal_strength: float
    exit_reason: Optional[str]  # 'stop_loss', 'take_profit', 'signal', 'time_stop'
    profit_loss: Optional[float]
    profit_loss_pct: Optional[float]
    risk_reward_actual: Optional[float]

class SwingTradingSystem:
    def __init__(self, data_dir: str = "historical_data", 
                 initial_capital: float = 10000,
                 risk_per_trade: float = 0.02,  # 2% risk per trade
                 max_positions: int = 3):
        """Initialize the swing trading system."""
        self.data_dir = data_dir
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.max_positions = max_positions
        self.trades = []
        self.open_positions = []
        
    def load_data(self, symbol: str) -> pd.DataFrame:
        """Load historical data for a symbol."""
        filename = os.path.join(self.data_dir, f"{symbol}_historical_data.csv")
        if os.path.exists(filename):
            df = pd.read_csv(filename, index_col=0)
            df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
            return df
        return pd.DataFrame()
    
    def calculate_swing_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate swing trading signals based on multiple timeframes."""
        # Ensure we have required columns
        if 'RSI' not in df.columns:
            df['RSI'] = self.calculate_rsi(df['Close'])
        
        if 'MACD' not in df.columns:
            exp1 = df['Close'].ewm(span=12, adjust=False).mean()
            exp2 = df['Close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = exp1 - exp2
            df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
            df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # Calculate additional indicators for swing trading
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_30'] = df['Close'].rolling(window=30).mean()
        df['EMA_9'] = df['Close'].ewm(span=9, adjust=False).mean()
        df['EMA_21'] = df['Close'].ewm(span=21, adjust=False).mean()
        
        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        # Price patterns
        df['Higher_High'] = (df['High'] > df['High'].shift(1)) & (df['High'].shift(1) > df['High'].shift(2))
        df['Lower_Low'] = (df['Low'] < df['Low'].shift(1)) & (df['Low'].shift(1) < df['Low'].shift(2))
        
        # Swing High/Low identification
        df['Swing_High'] = df['High'].rolling(window=5, center=True).max() == df['High']
        df['Swing_Low'] = df['Low'].rolling(window=5, center=True).min() == df['Low']
        
        # Generate signals
        df['Signal'] = 0
        df['Signal_Strength'] = 0
        
        for i in range(50, len(df)):
            signal_score = 0
            
            # 1. Trend following signals
            if df['SMA_10'].iloc[i] > df['SMA_30'].iloc[i]:
                signal_score += 20
            else:
                signal_score -= 20
            
            # 2. Momentum signals
            if 30 < df['RSI'].iloc[i] < 70:
                if df['RSI'].iloc[i] > df['RSI'].iloc[i-1]:
                    signal_score += 15
                else:
                    signal_score -= 15
            elif df['RSI'].iloc[i] <= 30:
                signal_score += 25  # Oversold bounce
            elif df['RSI'].iloc[i] >= 70:
                signal_score -= 25  # Overbought reversal
            
            # 3. MACD signals
            if df['MACD'].iloc[i] > df['MACD_Signal'].iloc[i]:
                signal_score += 20
                if df['MACD'].iloc[i-1] <= df['MACD_Signal'].iloc[i-1]:
                    signal_score += 10  # Crossover bonus
            else:
                signal_score -= 20
            
            # 4. Volume confirmation
            if df['Volume_Ratio'].iloc[i] > 1.2:
                signal_score = signal_score * 1.2  # Amplify signal with volume
            
            # 5. Support/Resistance levels
            recent_lows = df['Low'].iloc[i-20:i].min()
            recent_highs = df['High'].iloc[i-20:i].max()
            price_position = (df['Close'].iloc[i] - recent_lows) / (recent_highs - recent_lows)
            
            if price_position < 0.3:  # Near support
                signal_score += 15
            elif price_position > 0.7:  # Near resistance
                signal_score -= 15
            
            # 6. Swing pattern recognition
            if i >= 10:
                # Bullish swing: Higher lows and higher highs
                if (df['Low'].iloc[i-5:i].min() > df['Low'].iloc[i-10:i-5].min() and
                    df['High'].iloc[i-5:i].max() > df['High'].iloc[i-10:i-5].max()):
                    signal_score += 10
                # Bearish swing: Lower highs and lower lows
                elif (df['High'].iloc[i-5:i].max() < df['High'].iloc[i-10:i-5].max() and
                      df['Low'].iloc[i-5:i].min() < df['Low'].iloc[i-10:i-5].min()):
                    signal_score -= 10
            
            # Convert score to signal
            df.loc[df.index[i], 'Signal_Strength'] = signal_score
            
            if signal_score >= 50:
                df.loc[df.index[i], 'Signal'] = 1  # Buy
            elif signal_score <= -50:
                df.loc[df.index[i], 'Signal'] = -1  # Sell/Short
            else:
                df.loc[df.index[i], 'Signal'] = 0  # Hold
        
        return df
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_position_size(self, entry_price: float, stop_loss: float) -> int:
        """Calculate position size based on risk management."""
        risk_amount = self.current_capital * self.risk_per_trade
        risk_per_share = abs(entry_price - stop_loss)
        
        if risk_per_share == 0:
            return 0
        
        shares = int(risk_amount / risk_per_share)
        
        # Ensure we don't exceed capital
        max_shares = int(self.current_capital * 0.95 / entry_price)  # Use 95% of capital max
        shares = min(shares, max_shares)
        
        return shares
    
    def backtest(self, symbol: str, df: pd.DataFrame) -> Dict:
        """Run backtest on historical data."""
        # Calculate signals
        df = self.calculate_swing_signals(df)
        
        # Reset for backtest
        self.trades = []
        self.open_positions = []
        self.current_capital = self.initial_capital
        
        # Backtest parameters
        max_holding_days = 20  # Maximum days to hold a position
        
        for i in range(50, len(df)):
            current_date = df.index[i]
            current_price = df['Close'].iloc[i]
            
            # Check exit conditions for open positions
            positions_to_close = []
            for position in self.open_positions:
                trade = position['trade']
                days_held = (current_date - trade.entry_date).days
                
                # Exit conditions
                exit_triggered = False
                exit_reason = None
                exit_price = current_price
                
                if trade.trade_type == 'LONG':
                    # Check stop loss
                    if df['Low'].iloc[i] <= trade.stop_loss:
                        exit_triggered = True
                        exit_reason = 'stop_loss'
                        exit_price = trade.stop_loss
                    # Check take profit
                    elif df['High'].iloc[i] >= trade.take_profit:
                        exit_triggered = True
                        exit_reason = 'take_profit'
                        exit_price = trade.take_profit
                    # Check signal reversal
                    elif df['Signal'].iloc[i] == -1:
                        exit_triggered = True
                        exit_reason = 'signal_reversal'
                    # Time stop
                    elif days_held >= max_holding_days:
                        exit_triggered = True
                        exit_reason = 'time_stop'
                
                if exit_triggered:
                    # Close position
                    trade.exit_date = current_date
                    trade.exit_price = exit_price
                    trade.exit_reason = exit_reason
                    
                    # Calculate P&L
                    if trade.trade_type == 'LONG':
                        trade.profit_loss = (exit_price - trade.entry_price) * trade.position_size
                        trade.profit_loss_pct = (exit_price - trade.entry_price) / trade.entry_price
                    
                    # Calculate actual risk/reward
                    risk = abs(trade.entry_price - trade.stop_loss)
                    reward = abs(exit_price - trade.entry_price)
                    trade.risk_reward_actual = reward / risk if risk > 0 else 0
                    
                    # Update capital
                    self.current_capital += trade.profit_loss
                    
                    positions_to_close.append(position)
                    self.trades.append(trade)
            
            # Remove closed positions
            for position in positions_to_close:
                self.open_positions.remove(position)
            
            # Check entry conditions (only if we have capacity)
            if len(self.open_positions) < self.max_positions:
                signal = df['Signal'].iloc[i]
                signal_strength = df['Signal_Strength'].iloc[i]
                
                if signal == 1 and signal_strength >= 50:  # Buy signal
                    # Calculate stop loss and take profit
                    atr = df['ATR'].iloc[i] if 'ATR' in df.columns else current_price * 0.02
                    
                    stop_loss = current_price - (2 * atr)
                    take_profit_1 = current_price + (3 * atr)
                    take_profit_2 = current_price + (5 * atr)
                    
                    # Use first take profit for swing trades
                    take_profit = take_profit_1
                    
                    # Calculate position size
                    position_size = self.calculate_position_size(current_price, stop_loss)
                    
                    if position_size > 0 and (position_size * current_price) <= self.current_capital * 0.95:
                        # Create trade
                        trade = Trade(
                            entry_date=current_date,
                            exit_date=None,
                            entry_price=current_price,
                            exit_price=None,
                            stop_loss=stop_loss,
                            take_profit=take_profit,
                            position_size=position_size,
                            trade_type='LONG',
                            signal_strength=signal_strength,
                            exit_reason=None,
                            profit_loss=None,
                            profit_loss_pct=None,
                            risk_reward_actual=None
                        )
                        
                        # Update capital (reserve for position)
                        self.current_capital -= position_size * current_price
                        
                        # Add to open positions
                        self.open_positions.append({
                            'trade': trade,
                            'symbol': symbol
                        })
        
        # Close any remaining positions at end
        for position in self.open_positions:
            trade = position['trade']
            trade.exit_date = df.index[-1]
            trade.exit_price = df['Close'].iloc[-1]
            trade.exit_reason = 'end_of_data'
            
            if trade.trade_type == 'LONG':
                trade.profit_loss = (trade.exit_price - trade.entry_price) * trade.position_size
                trade.profit_loss_pct = (trade.exit_price - trade.entry_price) / trade.entry_price
            
            risk = abs(trade.entry_price - trade.stop_loss)
            reward = abs(trade.exit_price - trade.entry_price)
            trade.risk_reward_actual = reward / risk if risk > 0 else 0
            
            self.current_capital += trade.position_size * trade.exit_price
            self.trades.append(trade)
        
        # Calculate performance metrics
        return self.calculate_performance_metrics(symbol)
    
    def calculate_performance_metrics(self, symbol: str) -> Dict:
        """Calculate comprehensive performance metrics."""
        if not self.trades:
            return {
                'symbol': symbol,
                'total_trades': 0,
                'win_rate': 0,
                'message': 'No trades executed'
            }
        
        # Basic metrics
        total_trades = len(self.trades)
        winning_trades = [t for t in self.trades if t.profit_loss > 0]
        losing_trades = [t for t in self.trades if t.profit_loss <= 0]
        
        win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0
        
        # Profit/Loss metrics
        total_profit = sum(t.profit_loss for t in self.trades)
        total_return = (self.current_capital - self.initial_capital) / self.initial_capital * 100
        
        # Average metrics
        avg_win = sum(t.profit_loss for t in winning_trades) / len(winning_trades) if winning_trades else 0
        avg_loss = sum(t.profit_loss for t in losing_trades) / len(losing_trades) if losing_trades else 0
        
        avg_win_pct = sum(t.profit_loss_pct for t in winning_trades) / len(winning_trades) * 100 if winning_trades else 0
        avg_loss_pct = sum(t.profit_loss_pct for t in losing_trades) / len(losing_trades) * 100 if losing_trades else 0
        
        # Risk/Reward metrics
        profit_factor = abs(sum(t.profit_loss for t in winning_trades) / sum(t.profit_loss for t in losing_trades)) if losing_trades and sum(t.profit_loss for t in losing_trades) != 0 else 0
        
        # Expected value per trade
        expected_value = (win_rate/100 * avg_win) + ((100-win_rate)/100 * avg_loss)
        
        # Risk/Reward ratios
        planned_rr_ratios = []
        actual_rr_ratios = []
        
        for trade in self.trades:
            # Planned R/R (take profit vs stop loss)
            risk = abs(trade.entry_price - trade.stop_loss)
            reward = abs(trade.take_profit - trade.entry_price)
            planned_rr = reward / risk if risk > 0 else 0
            planned_rr_ratios.append(planned_rr)
            
            # Actual R/R
            if trade.risk_reward_actual is not None:
                actual_rr_ratios.append(trade.risk_reward_actual)
        
        avg_planned_rr = sum(planned_rr_ratios) / len(planned_rr_ratios) if planned_rr_ratios else 0
        avg_actual_rr = sum(actual_rr_ratios) / len(actual_rr_ratios) if actual_rr_ratios else 0
        
        # Exit reason analysis
        exit_reasons = {}
        for trade in self.trades:
            reason = trade.exit_reason or 'unknown'
            exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
        
        # Maximum drawdown
        equity_curve = [self.initial_capital]
        running_capital = self.initial_capital
        
        for trade in sorted(self.trades, key=lambda x: x.exit_date or datetime.now()):
            if trade.profit_loss is not None:
                running_capital += trade.profit_loss
                equity_curve.append(running_capital)
        
        max_drawdown = 0
        peak = equity_curve[0]
        for value in equity_curve:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak * 100
            max_drawdown = max(max_drawdown, drawdown)
        
        # Compile results
        metrics = {
            'symbol': symbol,
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': round(win_rate, 2),
            'total_profit': round(total_profit, 2),
            'total_return_pct': round(total_return, 2),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'avg_win_pct': round(avg_win_pct, 2),
            'avg_loss_pct': round(avg_loss_pct, 2),
            'profit_factor': round(profit_factor, 2),
            'expected_value': round(expected_value, 2),
            'max_drawdown_pct': round(max_drawdown, 2),
            'risk_reward_ratios': {
                'avg_planned': round(avg_planned_rr, 2),
                'avg_actual': round(avg_actual_rr, 2)
            },
            'exit_reasons': exit_reasons,
            'initial_capital': self.initial_capital,
            'final_capital': round(self.current_capital, 2)
        }
        
        return metrics
    
    def generate_trade_log(self, symbol: str, save_to_file: bool = True) -> pd.DataFrame:
        """Generate detailed trade log."""
        trade_data = []
        
        for trade in self.trades:
            trade_data.append({
                'Symbol': symbol,
                'Entry Date': trade.entry_date.strftime('%Y-%m-%d'),
                'Exit Date': trade.exit_date.strftime('%Y-%m-%d') if trade.exit_date else 'Open',
                'Entry Price': round(trade.entry_price, 2),
                'Exit Price': round(trade.exit_price, 2) if trade.exit_price else None,
                'Stop Loss': round(trade.stop_loss, 2),
                'Take Profit': round(trade.take_profit, 2),
                'Position Size': trade.position_size,
                'P&L': round(trade.profit_loss, 2) if trade.profit_loss else None,
                'P&L %': round(trade.profit_loss_pct * 100, 2) if trade.profit_loss_pct else None,
                'Exit Reason': trade.exit_reason,
                'Signal Strength': round(trade.signal_strength, 2),
                'Actual R/R': round(trade.risk_reward_actual, 2) if trade.risk_reward_actual else None
            })
        
        df_trades = pd.DataFrame(trade_data)
        
        if save_to_file and not df_trades.empty:
            filename = f"swing_trades_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df_trades.to_csv(filename, index=False)
            print(f"Trade log saved to {filename}")
        
        return df_trades
    
    def run_multi_symbol_backtest(self, symbols: List[str]) -> Dict:
        """Run backtest on multiple symbols."""
        all_results = {}
        
        for symbol in symbols:
            print(f"\nBacktesting {symbol}...")
            df = self.load_data(symbol)
            
            if not df.empty:
                metrics = self.backtest(symbol, df)
                all_results[symbol] = metrics
                
                # Generate trade log
                self.generate_trade_log(symbol, save_to_file=True)
            else:
                print(f"No data available for {symbol}")
        
        return all_results


def main():
    """Main function to run swing trading backtest."""
    symbols = ['AAPL', 'GOOGL', 'TSLA', 'MSFT', 'UNH']
    
    # Initialize system
    system = SwingTradingSystem(
        initial_capital=10000,
        risk_per_trade=0.02,  # 2% risk
        max_positions=3
    )
    
    print("🚀 Enhanced Swing Trading System Backtest")
    print("="*60)
    print(f"Initial Capital: $10,000")
    print(f"Risk per Trade: 2%")
    print(f"Strategy: Multi-timeframe swing trading")
    print(f"Backtesting Period: 3 years")
    print("="*60)
    
    # Run backtest
    results = system.run_multi_symbol_backtest(symbols)
    
    # Display results
    print("\n📊 BACKTEST RESULTS SUMMARY")
    print("="*60)
    
    for symbol, metrics in results.items():
        print(f"\n{symbol} Performance:")
        print("-"*40)
        print(f"Total Trades: {metrics['total_trades']}")
        print(f"Win Rate: {metrics['win_rate']}%")
        print(f"Winning Trades: {metrics['winning_trades']} / Losing Trades: {metrics['losing_trades']}")
        print(f"Total Return: {metrics['total_return_pct']}%")
        print(f"Final Capital: ${metrics['final_capital']:,.2f}")
        print(f"Profit Factor: {metrics['profit_factor']}")
        print(f"Max Drawdown: {metrics['max_drawdown_pct']}%")
        print(f"\nRisk/Reward Ratios:")
        print(f"  Planned R/R: {metrics['risk_reward_ratios']['avg_planned']}")
        print(f"  Actual R/R: {metrics['risk_reward_ratios']['avg_actual']}")
        print(f"\nAverage Win: ${metrics['avg_win']:.2f} ({metrics['avg_win_pct']}%)")
        print(f"Average Loss: ${metrics['avg_loss']:.2f} ({metrics['avg_loss_pct']}%)")
        print(f"Expected Value per Trade: ${metrics['expected_value']:.2f}")
        print(f"\nExit Reasons: {metrics['exit_reasons']}")
    
    # Overall portfolio metrics
    print("\n"*2)
    print("="*60)
    print("📈 PORTFOLIO SUMMARY")
    print("="*60)
    
    total_trades = sum(r['total_trades'] for r in results.values())
    total_wins = sum(r['winning_trades'] for r in results.values())
    overall_win_rate = total_wins / total_trades * 100 if total_trades > 0 else 0
    
    print(f"Total Trades Across All Symbols: {total_trades}")
    print(f"Overall Win Rate: {overall_win_rate:.2f}%")
    print(f"Average Profit Factor: {sum(r['profit_factor'] for r in results.values()) / len(results):.2f}")
    
    # Save summary
    summary_file = f"swing_trading_backtest_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to {summary_file}")
    
    print("\n✅ Backtest complete!")


if __name__ == "__main__":
    main()