#!/usr/bin/env python3
"""
Apple Trading Signal Backtesting Script
Tests the accuracy of our trading signals against actual price movements
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os

class AppleSignalBacktester:
    def __init__(self):
        self.symbol = 'AAPL'
        self.results = []
        self.test_period = 30  # Test last 30 days
        
    def calculate_technical_indicators(self, data, date):
        """Calculate indicators for a specific date"""
        try:
            # Get data up to the specified date
            mask = data.index <= date
            historical_data = data[mask]
            
            if len(historical_data) < 50:  # Need enough data for indicators
                return None
            
            close_prices = historical_data['Close']
            
            # RSI
            delta = close_prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            # Moving averages
            sma_20 = close_prices.rolling(window=20).mean()
            sma_50 = close_prices.rolling(window=50).mean()
            sma_200 = close_prices.rolling(window=200).mean()
            
            # MACD
            ema_12 = close_prices.ewm(span=12).mean()
            ema_26 = close_prices.ewm(span=26).mean()
            macd = ema_12 - ema_26
            macd_signal = macd.ewm(span=9).mean()
            
            # Bollinger Bands
            bb_sma = close_prices.rolling(window=20).mean()
            bb_std = close_prices.rolling(window=20).std()
            bb_upper = bb_sma + (bb_std * 2)
            bb_lower = bb_sma - (bb_std * 2)
            
            # ATR
            high_low = historical_data['High'] - historical_data['Low']
            high_close = np.abs(historical_data['High'] - historical_data['Close'].shift())
            low_close = np.abs(historical_data['Low'] - historical_data['Close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            atr = true_range.rolling(14).mean()
            
            return {
                'RSI': rsi.iloc[-1],
                'MACD': macd.iloc[-1],
                'MACD_signal': macd_signal.iloc[-1],
                'SMA_20': sma_20.iloc[-1],
                'SMA_50': sma_50.iloc[-1],
                'SMA_200': sma_200.iloc[-1] if not pd.isna(sma_200.iloc[-1]) else None,
                'BB_upper': bb_upper.iloc[-1],
                'BB_lower': bb_lower.iloc[-1],
                'ATR': atr.iloc[-1],
                'close_price': close_prices.iloc[-1],
                'volume': historical_data['Volume'].iloc[-1],
                'avg_volume': historical_data['Volume'].rolling(20).mean().iloc[-1]
            }
        except Exception as e:
            print(f"Error calculating indicators: {e}")
            return None
    
    def generate_signal(self, indicators):
        """Generate trading signal based on indicators"""
        if not indicators:
            return 'HOLD', 50, []
        
        score = 50
        reasons = []
        
        # RSI Analysis
        rsi = indicators['RSI']
        if rsi < 30:
            score += 20
            reasons.append(f"RSI oversold ({rsi:.1f})")
        elif rsi > 70:
            score -= 20
            reasons.append(f"RSI overbought ({rsi:.1f})")
        elif rsi < 40:
            score += 10
            reasons.append(f"RSI approaching oversold ({rsi:.1f})")
        elif rsi > 60:
            score -= 10
            reasons.append(f"RSI approaching overbought ({rsi:.1f})")
        
        # MACD Analysis
        if indicators['MACD'] > indicators['MACD_signal']:
            score += 15
            reasons.append("MACD bullish crossover")
        else:
            score -= 15
            reasons.append("MACD bearish")
        
        # Moving Average Analysis
        current_price = indicators['close_price']
        if indicators['SMA_20'] and indicators['SMA_50']:
            if current_price > indicators['SMA_20'] > indicators['SMA_50']:
                score += 20
                reasons.append("Price above key moving averages")
            elif current_price < indicators['SMA_20'] < indicators['SMA_50']:
                score -= 20
                reasons.append("Price below key moving averages")
        
        # Bollinger Bands
        if current_price <= indicators['BB_lower']:
            score += 10
            reasons.append("Price at lower Bollinger Band")
        elif current_price >= indicators['BB_upper']:
            score -= 10
            reasons.append("Price at upper Bollinger Band")
        
        # Volume analysis
        if indicators['volume'] > indicators['avg_volume'] * 1.5:
            score += 5
            reasons.append("High volume confirmation")
        
        # Determine action
        if score >= 70:
            action = 'STRONG_BUY'
        elif score >= 60:
            action = 'BUY'
        elif score <= 30:
            action = 'STRONG_SELL'
        elif score <= 40:
            action = 'SELL'
        else:
            action = 'HOLD'
        
        return action, score, reasons
    
    def calculate_price_targets(self, indicators, signal):
        """Calculate price targets based on ATR"""
        if not indicators or signal == 'HOLD':
            return None
        
        current_price = indicators['close_price']
        atr = indicators['ATR']
        
        if signal in ['BUY', 'STRONG_BUY']:
            return {
                'entry': current_price,
                'stop_loss': current_price - (atr * 2),
                'take_profit_1': current_price + (atr * 2),
                'take_profit_2': current_price + (atr * 4)
            }
        elif signal in ['SELL', 'STRONG_SELL']:
            return {
                'entry': current_price,
                'stop_loss': current_price + (atr * 2),
                'take_profit_1': current_price - (atr * 2),
                'take_profit_2': current_price - (atr * 4)
            }
        
        return None
    
    def check_signal_outcome(self, entry_date, signal, targets, future_data):
        """Check if the signal was successful"""
        if not targets or signal == 'HOLD':
            return {
                'success': None,
                'outcome': 'No trade signal',
                'days_to_target': None,
                'max_profit_percent': 0,
                'max_loss_percent': 0
            }
        
        # Get future price movements (up to 10 trading days)
        future_prices = future_data['Close'][:10] if len(future_data) >= 10 else future_data['Close']
        
        entry_price = targets['entry']
        stop_loss = targets['stop_loss']
        take_profit_1 = targets['take_profit_1']
        
        # For BUY signals
        if signal in ['BUY', 'STRONG_BUY']:
            # Check if stop loss hit
            min_price = future_prices.min()
            if min_price <= stop_loss:
                days_to_stop = (future_prices <= stop_loss).idxmax()
                days_to_target = (days_to_stop - entry_date).days if days_to_stop != future_prices.index[0] else 1
                return {
                    'success': False,
                    'outcome': 'Stop loss hit',
                    'days_to_target': days_to_target,
                    'exit_price': stop_loss,
                    'profit_percent': ((stop_loss - entry_price) / entry_price) * 100,
                    'max_profit_percent': ((future_prices.max() - entry_price) / entry_price) * 100,
                    'max_loss_percent': ((min_price - entry_price) / entry_price) * 100
                }
            
            # Check if take profit hit
            max_price = future_prices.max()
            if max_price >= take_profit_1:
                days_to_profit = (future_prices >= take_profit_1).idxmax()
                days_to_target = (days_to_profit - entry_date).days if days_to_profit != future_prices.index[0] else 1
                return {
                    'success': True,
                    'outcome': 'Take profit 1 hit',
                    'days_to_target': days_to_target,
                    'exit_price': take_profit_1,
                    'profit_percent': ((take_profit_1 - entry_price) / entry_price) * 100,
                    'max_profit_percent': ((max_price - entry_price) / entry_price) * 100,
                    'max_loss_percent': ((min_price - entry_price) / entry_price) * 100
                }
            
            # Neither hit - check final performance
            final_price = future_prices.iloc[-1]
            profit_percent = ((final_price - entry_price) / entry_price) * 100
            return {
                'success': profit_percent > 0,
                'outcome': 'Time expired',
                'days_to_target': len(future_prices),
                'exit_price': final_price,
                'profit_percent': profit_percent,
                'max_profit_percent': ((max_price - entry_price) / entry_price) * 100,
                'max_loss_percent': ((min_price - entry_price) / entry_price) * 100
            }
        
        # For SELL signals (short positions)
        elif signal in ['SELL', 'STRONG_SELL']:
            # Check if stop loss hit (price went up)
            max_price = future_prices.max()
            if max_price >= stop_loss:
                days_to_stop = (future_prices >= stop_loss).idxmax()
                days_to_target = (days_to_stop - entry_date).days if days_to_stop != future_prices.index[0] else 1
                return {
                    'success': False,
                    'outcome': 'Stop loss hit',
                    'days_to_target': days_to_target,
                    'exit_price': stop_loss,
                    'profit_percent': ((entry_price - stop_loss) / entry_price) * 100,
                    'max_profit_percent': ((entry_price - future_prices.min()) / entry_price) * 100,
                    'max_loss_percent': ((entry_price - max_price) / entry_price) * 100
                }
            
            # Check if take profit hit (price went down)
            min_price = future_prices.min()
            if min_price <= take_profit_1:
                days_to_profit = (future_prices <= take_profit_1).idxmax()
                days_to_target = (days_to_profit - entry_date).days if days_to_profit != future_prices.index[0] else 1
                return {
                    'success': True,
                    'outcome': 'Take profit 1 hit',
                    'days_to_target': days_to_target,
                    'exit_price': take_profit_1,
                    'profit_percent': ((entry_price - take_profit_1) / entry_price) * 100,
                    'max_profit_percent': ((entry_price - min_price) / entry_price) * 100,
                    'max_loss_percent': ((entry_price - max_price) / entry_price) * 100
                }
            
            # Neither hit - check final performance
            final_price = future_prices.iloc[-1]
            profit_percent = ((entry_price - final_price) / entry_price) * 100
            return {
                'success': profit_percent > 0,
                'outcome': 'Time expired',
                'days_to_target': len(future_prices),
                'exit_price': final_price,
                'profit_percent': profit_percent,
                'max_profit_percent': ((entry_price - min_price) / entry_price) * 100,
                'max_loss_percent': ((entry_price - max_price) / entry_price) * 100
            }
    
    def run_backtest(self):
        """Run the backtest for the specified period"""
        print(f"\n🔄 Starting backtest for {self.symbol}...")
        print(f"Testing period: Last {self.test_period} days")
        print("="*80)
        
        # Fetch historical data
        ticker = yf.Ticker(self.symbol)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)  # Get full year for indicators
        
        print(f"Fetching data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...")
        data = ticker.history(start=start_date, end=end_date)
        
        if data.empty:
            print("❌ Failed to fetch historical data")
            return
        
        # Get test dates (last 30 trading days)
        test_dates = data.index[-self.test_period:]
        
        results = []
        
        for i, test_date in enumerate(test_dates[:-10]):  # Leave 10 days for outcome checking
            print(f"\n📅 Testing {test_date.strftime('%Y-%m-%d')}...")
            
            # Calculate indicators for this date
            indicators = self.calculate_technical_indicators(data, test_date)
            
            if not indicators:
                print("  ⚠️ Insufficient data for indicators")
                continue
            
            # Generate signal
            signal, score, reasons = self.generate_signal(indicators)
            targets = self.calculate_price_targets(indicators, signal)
            
            # Get future data for outcome checking
            future_start_idx = data.index.get_loc(test_date) + 1
            future_data = data.iloc[future_start_idx:future_start_idx + 10]
            
            # Check outcome
            outcome = self.check_signal_outcome(test_date, signal, targets, future_data)
            
            # Store result
            result = {
                'date': test_date.strftime('%Y-%m-%d'),
                'signal': signal,
                'score': score,
                'entry_price': indicators['close_price'],
                'reasons': reasons,
                'outcome': outcome,
                'targets': targets
            }
            results.append(result)
            
            # Display result
            print(f"  📊 Signal: {signal} (Score: {score})")
            print(f"  💰 Entry: ${indicators['close_price']:.2f}")
            if outcome['success'] is not None:
                success_emoji = "✅" if outcome['success'] else "❌"
                print(f"  {success_emoji} Result: {outcome['outcome']}")
                print(f"  📈 P/L: {outcome['profit_percent']:.2f}%")
        
        self.results = results
        return results
    
    def calculate_performance_metrics(self):
        """Calculate overall performance metrics"""
        if not self.results:
            return None
        
        # Filter out HOLD signals
        trade_results = [r for r in self.results if r['outcome']['success'] is not None]
        
        if not trade_results:
            return None
        
        # Calculate metrics
        total_trades = len(trade_results)
        successful_trades = sum(1 for r in trade_results if r['outcome']['success'])
        failed_trades = total_trades - successful_trades
        
        success_rate = (successful_trades / total_trades) * 100 if total_trades > 0 else 0
        
        # Calculate profit/loss
        total_profit = sum(r['outcome']['profit_percent'] for r in trade_results)
        avg_profit = total_profit / total_trades if total_trades > 0 else 0
        
        # Separate by signal type
        buy_signals = [r for r in trade_results if r['signal'] in ['BUY', 'STRONG_BUY']]
        sell_signals = [r for r in trade_results if r['signal'] in ['SELL', 'STRONG_SELL']]
        
        buy_success = sum(1 for r in buy_signals if r['outcome']['success']) / len(buy_signals) * 100 if buy_signals else 0
        sell_success = sum(1 for r in sell_signals if r['outcome']['success']) / len(sell_signals) * 100 if sell_signals else 0
        
        # Calculate max drawdown
        cumulative_returns = []
        cumulative = 0
        for r in trade_results:
            cumulative += r['outcome']['profit_percent']
            cumulative_returns.append(cumulative)
        
        if cumulative_returns:
            peak = cumulative_returns[0]
            max_drawdown = 0
            for ret in cumulative_returns:
                if ret > peak:
                    peak = ret
                drawdown = (peak - ret)
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
        else:
            max_drawdown = 0
        
        return {
            'total_trades': total_trades,
            'successful_trades': successful_trades,
            'failed_trades': failed_trades,
            'success_rate': success_rate,
            'total_return': total_profit,
            'avg_return_per_trade': avg_profit,
            'buy_signal_success': buy_success,
            'sell_signal_success': sell_success,
            'max_drawdown': max_drawdown,
            'buy_signals_count': len(buy_signals),
            'sell_signals_count': len(sell_signals)
        }
    
    def display_performance_report(self):
        """Display detailed performance report"""
        metrics = self.calculate_performance_metrics()
        
        if not metrics:
            print("\n❌ No trading signals generated during test period")
            return
        
        print("\n" + "="*80)
        print("📊 BACKTEST PERFORMANCE REPORT - APPLE (AAPL)")
        print("="*80)
        
        print(f"\n📈 OVERALL PERFORMANCE:")
        print(f"Total Trades: {metrics['total_trades']}")
        print(f"Successful: {metrics['successful_trades']} ({metrics['success_rate']:.1f}% conversion rate)")
        print(f"Failed: {metrics['failed_trades']}")
        print(f"Total Return: {metrics['total_return']:.2f}%")
        print(f"Average Return per Trade: {metrics['avg_return_per_trade']:.2f}%")
        print(f"Max Drawdown: {metrics['max_drawdown']:.2f}%")
        
        print(f"\n📊 SIGNAL BREAKDOWN:")
        print(f"BUY Signals: {metrics['buy_signals_count']} (Success: {metrics['buy_signal_success']:.1f}%)")
        print(f"SELL Signals: {metrics['sell_signals_count']} (Success: {metrics['sell_signal_success']:.1f}%)")
        
        # Grade the performance
        print(f"\n🎯 PERFORMANCE GRADE:")
        if metrics['success_rate'] >= 70:
            grade = "A - Excellent"
            emoji = "🌟"
        elif metrics['success_rate'] >= 60:
            grade = "B - Good"
            emoji = "✅"
        elif metrics['success_rate'] >= 50:
            grade = "C - Average"
            emoji = "📊"
        elif metrics['success_rate'] >= 40:
            grade = "D - Below Average"
            emoji = "⚠️"
        else:
            grade = "F - Poor"
            emoji = "❌"
        
        print(f"{emoji} Grade: {grade}")
        print(f"Success Rate: {metrics['success_rate']:.1f}%")
        
        # Detailed trade log
        print(f"\n📋 DETAILED TRADE LOG:")
        print("-"*80)
        print(f"{'Date':<12} {'Signal':<12} {'Entry':<8} {'Outcome':<20} {'P/L %':<8} {'Days':<6}")
        print("-"*80)
        
        for result in self.results:
            if result['outcome']['success'] is not None:
                success_icon = "✅" if result['outcome']['success'] else "❌"
                print(f"{result['date']:<12} {result['signal']:<12} ${result['entry_price']:<7.2f} "
                      f"{success_icon} {result['outcome']['outcome']:<17} "
                      f"{result['outcome']['profit_percent']:>6.2f}% "
                      f"{result['outcome']['days_to_target'] or 0:>4}")
        
        # Save results
        self.save_results(metrics)
    
    def save_results(self, metrics):
        """Save backtest results to JSON"""
        output = {
            'symbol': self.symbol,
            'test_period_days': self.test_period,
            'test_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'performance_metrics': metrics,
            'detailed_results': self.results
        }
        
        try:
            os.makedirs('outputs', exist_ok=True)
            filename = f"outputs/backtest_apple_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(output, f, indent=2, default=str)
            print(f"\n💾 Backtest results saved to {filename}")
        except Exception as e:
            print(f"\n⚠️ Error saving results: {e}")

def main():
    """Main function"""
    print("\n🍎 Apple Trading Signal Backtester")
    print("="*60)
    print("Testing our trading model's accuracy on historical data")
    print("="*60)
    
    backtester = AppleSignalBacktester()
    
    # Run backtest
    results = backtester.run_backtest()
    
    # Display performance report
    backtester.display_performance_report()
    
    print("\n" + "="*80)
    print("⚠️ IMPORTANT NOTES:")
    print("="*80)
    print("• Past performance does not guarantee future results")
    print("• This backtest uses simplified logic and may not reflect real trading")
    print("• Always use proper risk management and stop losses")
    print("• Results are based on closing prices and perfect execution")
    print("="*80)

if __name__ == "__main__":
    main()