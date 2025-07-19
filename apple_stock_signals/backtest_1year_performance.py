#!/usr/bin/env python3
"""
Backtest Analysis: $1000 invested 1 year ago using our trading signals
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core_scripts.enhanced_trading_analyzer import EnhancedTradingAnalyzer
from data_modules.technical_analyzer import TechnicalAnalyzer

class BacktestAnalyzer:
    def __init__(self, initial_capital=1000):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = {}
        self.trades = []
        self.analyzer = EnhancedTradingAnalyzer()
        self.tech_analyzer = TechnicalAnalyzer()
        
    def calculate_indicators(self, df):
        """Calculate technical indicators for backtesting"""
        # RSI
        df['RSI'] = self.tech_analyzer.calculate_rsi(df['Close'])
        
        # MACD
        macd_data = self.tech_analyzer.calculate_macd(df['Close'])
        df['MACD'] = macd_data['macd']
        df['MACD_Signal'] = macd_data['signal']
        df['MACD_Histogram'] = macd_data['histogram']
        
        # Moving Averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        
        # Bollinger Bands
        bb_data = self.tech_analyzer.calculate_bollinger_bands(df['Close'])
        df['BB_Upper'] = bb_data['upper']
        df['BB_Lower'] = bb_data['lower']
        
        # Stochastic
        stoch_data = self.tech_analyzer.calculate_stochastic(df)
        df['Stoch_K'] = stoch_data['K']
        df['Stoch_D'] = stoch_data['D']
        
        return df
    
    def generate_signal(self, row, symbol):
        """Generate trading signal based on our logic"""
        score = 50  # Base score
        reasons = []
        
        # RSI Analysis
        if pd.notna(row['RSI']):
            if row['RSI'] > 70:
                score += 10
                reasons.append("RSI overbought")
            elif row['RSI'] < 30:
                score -= 10
                reasons.append("RSI oversold - potential buy")
                
        # MACD Analysis
        if pd.notna(row['MACD']) and pd.notna(row['MACD_Signal']):
            if row['MACD'] > row['MACD_Signal']:
                score += 15
                reasons.append("MACD bullish crossover")
            else:
                score -= 15
                reasons.append("MACD bearish")
                
        # Moving Average Analysis
        if pd.notna(row['SMA_20']) and pd.notna(row['SMA_50']):
            if row['Close'] > row['SMA_20'] and row['Close'] > row['SMA_50']:
                score += 10
                reasons.append("Price above key MAs")
            elif row['Close'] < row['SMA_20'] and row['Close'] < row['SMA_50']:
                score -= 10
                reasons.append("Price below key MAs")
                
        # Determine signal
        if score >= 65:
            return 'BUY', score, reasons
        elif score <= 35:
            return 'SELL', score, reasons
        else:
            return 'HOLD', score, reasons
    
    def backtest_strategy(self):
        """Run backtest for all stocks over 1 year"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        symbols = ['AAPL', 'GOOGL', 'TSLA', 'MSFT', 'UNH']
        
        print(f"📊 BACKTESTING STRATEGY")
        print(f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        print(f"Initial Capital: ${self.initial_capital}")
        print("="*80)
        
        # Allocate capital equally among stocks
        capital_per_stock = self.initial_capital / len(symbols)
        
        all_results = {}
        
        for symbol in symbols:
            print(f"\n🔍 Backtesting {symbol}...")
            
            # Download historical data
            stock = yf.Ticker(symbol)
            df = stock.history(start=start_date, end=end_date)
            
            if df.empty:
                print(f"❌ No data available for {symbol}")
                continue
                
            # Calculate indicators
            df = self.calculate_indicators(df)
            
            # Initialize tracking for this stock
            position = None
            trades_list = []
            
            # Simulate trading
            for date, row in df.iterrows():
                signal, score, reasons = self.generate_signal(row, symbol)
                
                # Entry logic
                if signal == 'BUY' and position is None and score >= 65:
                    # Buy signal
                    shares = capital_per_stock / row['Close']
                    position = {
                        'entry_date': date,
                        'entry_price': row['Close'],
                        'shares': shares,
                        'stop_loss': row['Close'] * 0.98,  # 2% stop loss
                        'take_profit': row['Close'] * 1.05  # 5% take profit
                    }
                    trades_list.append({
                        'date': date.strftime('%Y-%m-%d'),
                        'action': 'BUY',
                        'price': row['Close'],
                        'shares': shares,
                        'value': capital_per_stock,
                        'signal_score': score,
                        'reasons': ', '.join(reasons)
                    })
                    
                # Exit logic
                elif position is not None:
                    exit_trade = False
                    exit_reason = ""
                    
                    # Check stop loss
                    if row['Low'] <= position['stop_loss']:
                        exit_price = position['stop_loss']
                        exit_reason = "Stop loss hit"
                        exit_trade = True
                    # Check take profit
                    elif row['High'] >= position['take_profit']:
                        exit_price = position['take_profit']
                        exit_reason = "Take profit hit"
                        exit_trade = True
                    # Check sell signal
                    elif signal == 'SELL' and score <= 35:
                        exit_price = row['Close']
                        exit_reason = f"Sell signal (score: {score})"
                        exit_trade = True
                        
                    if exit_trade:
                        # Calculate profit/loss
                        pnl = (exit_price - position['entry_price']) * position['shares']
                        pnl_pct = ((exit_price - position['entry_price']) / position['entry_price']) * 100
                        
                        trades_list.append({
                            'date': date.strftime('%Y-%m-%d'),
                            'action': 'SELL',
                            'price': exit_price,
                            'shares': position['shares'],
                            'value': exit_price * position['shares'],
                            'pnl': pnl,
                            'pnl_pct': pnl_pct,
                            'exit_reason': exit_reason,
                            'holding_days': (date - position['entry_date']).days
                        })
                        position = None
            
            # Close any open positions at end
            if position is not None:
                exit_price = df.iloc[-1]['Close']
                pnl = (exit_price - position['entry_price']) * position['shares']
                pnl_pct = ((exit_price - position['entry_price']) / position['entry_price']) * 100
                
                trades_list.append({
                    'date': df.index[-1].strftime('%Y-%m-%d'),
                    'action': 'SELL',
                    'price': exit_price,
                    'shares': position['shares'],
                    'value': exit_price * position['shares'],
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'exit_reason': 'End of backtest period',
                    'holding_days': (df.index[-1] - position['entry_date']).days
                })
            
            # Calculate final value for this stock
            if trades_list:
                final_value = capital_per_stock
                total_pnl = 0
                winning_trades = 0
                losing_trades = 0
                
                for trade in trades_list:
                    if trade['action'] == 'SELL':
                        final_value += trade.get('pnl', 0)
                        total_pnl += trade.get('pnl', 0)
                        if trade.get('pnl', 0) > 0:
                            winning_trades += 1
                        else:
                            losing_trades += 1
                
                all_results[symbol] = {
                    'initial_investment': capital_per_stock,
                    'final_value': final_value,
                    'total_return': final_value - capital_per_stock,
                    'return_pct': ((final_value - capital_per_stock) / capital_per_stock) * 100,
                    'trades': trades_list,
                    'winning_trades': winning_trades,
                    'losing_trades': losing_trades,
                    'win_rate': (winning_trades / (winning_trades + losing_trades) * 100) if (winning_trades + losing_trades) > 0 else 0
                }
                
                print(f"✅ {symbol}: ${capital_per_stock:.2f} → ${final_value:.2f} ({all_results[symbol]['return_pct']:.1f}%)")
                print(f"   Trades: {len([t for t in trades_list if t['action'] == 'BUY'])} | Win Rate: {all_results[symbol]['win_rate']:.1f}%")
            else:
                all_results[symbol] = {
                    'initial_investment': capital_per_stock,
                    'final_value': capital_per_stock,
                    'total_return': 0,
                    'return_pct': 0,
                    'trades': [],
                    'note': 'No trades executed'
                }
                print(f"❌ {symbol}: No trades executed")
        
        return all_results
    
    def generate_report(self, results):
        """Generate detailed performance report"""
        print("\n" + "="*80)
        print("📊 PORTFOLIO PERFORMANCE SUMMARY")
        print("="*80)
        
        total_initial = sum(r['initial_investment'] for r in results.values())
        total_final = sum(r['final_value'] for r in results.values())
        total_return = total_final - total_initial
        total_return_pct = (total_return / total_initial) * 100
        
        print(f"\n💰 OVERALL RESULTS:")
        print(f"Initial Investment: ${total_initial:.2f}")
        print(f"Final Portfolio Value: ${total_final:.2f}")
        print(f"Total Return: ${total_return:.2f} ({total_return_pct:.1f}%)")
        
        # Compare with buy-and-hold
        print(f"\n📈 COMPARISON WITH BUY-AND-HOLD:")
        
        for symbol, data in results.items():
            # Get buy-and-hold return
            stock = yf.Ticker(symbol)
            hist = stock.history(period="1y")
            if not hist.empty:
                bh_return = ((hist['Close'].iloc[-1] - hist['Close'].iloc[0]) / hist['Close'].iloc[0]) * 100
                print(f"\n{symbol}:")
                print(f"  Strategy Return: {data['return_pct']:.1f}%")
                print(f"  Buy-and-Hold Return: {bh_return:.1f}%")
                print(f"  Outperformance: {data['return_pct'] - bh_return:.1f}%")
        
        # Trade statistics
        print(f"\n📊 TRADING STATISTICS:")
        all_trades = []
        for symbol, data in results.items():
            if 'trades' in data:
                for trade in data['trades']:
                    if trade['action'] == 'SELL' and 'pnl' in trade:
                        all_trades.append({
                            'symbol': symbol,
                            'pnl': trade['pnl'],
                            'pnl_pct': trade['pnl_pct'],
                            'holding_days': trade.get('holding_days', 0)
                        })
        
        if all_trades:
            winning = [t for t in all_trades if t['pnl'] > 0]
            losing = [t for t in all_trades if t['pnl'] <= 0]
            
            print(f"Total Trades: {len(all_trades)}")
            print(f"Winning Trades: {len(winning)} ({len(winning)/len(all_trades)*100:.1f}%)")
            print(f"Losing Trades: {len(losing)} ({len(losing)/len(all_trades)*100:.1f}%)")
            
            if winning:
                avg_win = sum(t['pnl'] for t in winning) / len(winning)
                avg_win_pct = sum(t['pnl_pct'] for t in winning) / len(winning)
                print(f"Average Win: ${avg_win:.2f} ({avg_win_pct:.1f}%)")
            
            if losing:
                avg_loss = sum(t['pnl'] for t in losing) / len(losing)
                avg_loss_pct = sum(t['pnl_pct'] for t in losing) / len(losing)
                print(f"Average Loss: ${avg_loss:.2f} ({avg_loss_pct:.1f}%)")
            
            avg_holding = sum(t['holding_days'] for t in all_trades) / len(all_trades)
            print(f"Average Holding Period: {avg_holding:.1f} days")
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"backtest_results/backtest_1year_{timestamp}.json"
        os.makedirs("backtest_results", exist_ok=True)
        
        with open(filename, 'w') as f:
            json.dump({
                'summary': {
                    'initial_capital': total_initial,
                    'final_value': total_final,
                    'total_return': total_return,
                    'return_pct': total_return_pct,
                    'period': '1 year',
                    'end_date': datetime.now().strftime('%Y-%m-%d')
                },
                'details': results
            }, f, indent=2, default=str)
        
        print(f"\n💾 Detailed results saved to: {filename}")
        
        return total_final

if __name__ == "__main__":
    print("🚀 Starting 1-Year Backtest Analysis")
    print("="*80)
    
    backtester = BacktestAnalyzer(initial_capital=1000)
    results = backtester.backtest_strategy()
    final_value = backtester.generate_report(results)
    
    print("\n" + "="*80)
    print("✅ BACKTEST COMPLETE")
    print("="*80)