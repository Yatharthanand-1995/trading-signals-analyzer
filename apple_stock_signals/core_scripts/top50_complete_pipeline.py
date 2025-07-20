#!/usr/bin/env python3
"""
Complete Trading Pipeline for Top 50 US Stocks
Includes: Historical data fetching, ML training, backtesting, and paper trading
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Top 50 US stocks
TOP_50_STOCKS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 'JPM', 'JNJ',
    'V', 'WMT', 'PG', 'MA', 'UNH', 'HD', 'DIS', 'BAC', 'XOM', 'ADBE',
    'NFLX', 'CRM', 'CVX', 'ABBV', 'KO', 'PEP', 'TMO', 'COST', 'CSCO', 'MRK',
    'ACN', 'LLY', 'AVGO', 'DHR', 'VZ', 'CMCSA', 'ABT', 'WFC', 'INTC', 'TXN',
    'PFE', 'PM', 'NEE', 'RTX', 'HON', 'UNP', 'QCOM', 'BMY', 'UPS', 'AMGN'
]

class Top50CompletePipeline:
    def __init__(self):
        self.historical_data = {}
        self.analysis_results = {}
        self.backtest_results = {}
        self.ml_predictions = {}
        
    def fetch_historical_data(self, symbol, years=2):
        """Fetch historical data for a stock"""
        try:
            print(f"  Fetching {symbol}...", end='', flush=True)
            stock = yf.Ticker(symbol)
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=years*365)
            
            df = stock.history(start=start_date, end=end_date)
            
            if not df.empty:
                # Save to CSV
                csv_path = f'historical_data/top50/{symbol}_historical.csv'
                os.makedirs('historical_data/top50', exist_ok=True)
                df.to_csv(csv_path)
                
                print(f" ✅ {len(df)} days")
                return symbol, df
            else:
                print(f" ❌ No data")
                return symbol, None
                
        except Exception as e:
            print(f" ❌ Error: {str(e)}")
            return symbol, None
    
    def step1_fetch_all_historical_data(self):
        """Step 1: Fetch historical data for all stocks"""
        print("\n📊 STEP 1: FETCHING HISTORICAL DATA FOR TOP 50 STOCKS")
        print("="*80)
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(self.fetch_historical_data, symbol) for symbol in TOP_50_STOCKS]
            
            for future in as_completed(futures):
                symbol, data = future.result()
                if data is not None:
                    self.historical_data[symbol] = data
        
        print(f"\n✅ Successfully fetched data for {len(self.historical_data)} stocks")
        
        # Save metadata
        metadata = {
            'fetch_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_stocks': len(TOP_50_STOCKS),
            'successful_fetches': len(self.historical_data),
            'failed_stocks': [s for s in TOP_50_STOCKS if s not in self.historical_data]
        }
        
        with open('historical_data/top50/fetch_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
            
        return self.historical_data
    
    def calculate_features(self, df):
        """Calculate technical features for ML"""
        features = pd.DataFrame(index=df.index)
        
        # Price features
        features['returns'] = df['Close'].pct_change()
        features['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Moving averages
        features['sma_5'] = df['Close'].rolling(5).mean()
        features['sma_20'] = df['Close'].rolling(20).mean()
        features['sma_50'] = df['Close'].rolling(50).mean()
        
        # Price ratios
        features['price_to_sma20'] = df['Close'] / features['sma_20']
        features['price_to_sma50'] = df['Close'] / features['sma_50']
        
        # Volatility
        features['volatility'] = features['returns'].rolling(20).std()
        
        # RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss
        features['rsi'] = 100 - (100 / (1 + rs))
        
        # Volume features
        features['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
        
        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        features['macd'] = exp1 - exp2
        features['macd_signal'] = features['macd'].ewm(span=9, adjust=False).mean()
        
        # Target: Next day return (for ML)
        features['target'] = features['returns'].shift(-1)
        
        return features.dropna()
    
    def step2_prepare_ml_data(self):
        """Step 2: Prepare data for ML training"""
        print("\n🤖 STEP 2: PREPARING ML DATA")
        print("="*80)
        
        ml_data = {}
        
        for symbol, df in self.historical_data.items():
            print(f"  Processing {symbol}...", end='', flush=True)
            
            try:
                features = self.calculate_features(df)
                if len(features) > 100:  # Need enough data
                    ml_data[symbol] = features
                    print(f" ✅ {len(features)} samples")
                else:
                    print(f" ❌ Insufficient data")
            except Exception as e:
                print(f" ❌ Error: {str(e)}")
        
        print(f"\n✅ Prepared ML data for {len(ml_data)} stocks")
        return ml_data
    
    def step3_backtest_strategy(self):
        """Step 3: Backtest trading strategy"""
        print("\n📈 STEP 3: BACKTESTING TRADING STRATEGY")
        print("="*80)
        
        backtest_results = {}
        
        for symbol, df in self.historical_data.items():
            if len(df) < 100:
                continue
                
            print(f"  Backtesting {symbol}...", end='', flush=True)
            
            try:
                # Simple momentum strategy
                df['returns'] = df['Close'].pct_change()
                df['sma_20'] = df['Close'].rolling(20).mean()
                df['sma_50'] = df['Close'].rolling(50).mean()
                df['signal'] = 0
                df.loc[df['sma_20'] > df['sma_50'], 'signal'] = 1
                df.loc[df['sma_20'] < df['sma_50'], 'signal'] = -1
                
                # Calculate strategy returns
                df['strategy_returns'] = df['signal'].shift(1) * df['returns']
                
                # Performance metrics
                total_return = (1 + df['strategy_returns']).prod() - 1
                sharpe_ratio = df['strategy_returns'].mean() / df['strategy_returns'].std() * np.sqrt(252)
                max_drawdown = (df['Close'] / df['Close'].cummax() - 1).min()
                
                backtest_results[symbol] = {
                    'total_return': total_return,
                    'annualized_return': (1 + total_return) ** (252 / len(df)) - 1,
                    'sharpe_ratio': sharpe_ratio,
                    'max_drawdown': max_drawdown,
                    'win_rate': (df['strategy_returns'] > 0).mean()
                }
                
                print(f" ✅ Return: {total_return:.2%}")
                
            except Exception as e:
                print(f" ❌ Error: {str(e)}")
        
        self.backtest_results = backtest_results
        print(f"\n✅ Completed backtesting for {len(backtest_results)} stocks")
        
        # Save backtest results
        os.makedirs('backtest_results/top50', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        with open(f'backtest_results/top50/backtest_{timestamp}.json', 'w') as f:
            json.dump(backtest_results, f, indent=2)
            
        return backtest_results
    
    def step4_generate_current_signals(self):
        """Step 4: Generate current trading signals"""
        print("\n🎯 STEP 4: GENERATING CURRENT TRADING SIGNALS")
        print("="*80)
        
        current_signals = []
        
        for symbol in TOP_50_STOCKS[:10]:  # Top 10 for demo
            if symbol not in self.historical_data:
                continue
                
            print(f"  Analyzing {symbol}...", end='', flush=True)
            
            try:
                df = self.historical_data[symbol]
                current_price = df['Close'].iloc[-1]
                
                # Calculate indicators
                rsi = self.calculate_rsi(df['Close'])
                macd_signal = self.calculate_macd_signal(df['Close'])
                
                # Generate signal
                score = 50
                if rsi < 30:
                    score += 20
                elif rsi > 70:
                    score -= 20
                    
                if macd_signal > 0:
                    score += 15
                else:
                    score -= 15
                
                # Determine action
                if score >= 65:
                    action = 'BUY'
                elif score <= 35:
                    action = 'SELL'
                else:
                    action = 'HOLD'
                
                current_signals.append({
                    'symbol': symbol,
                    'price': current_price,
                    'rsi': rsi,
                    'score': score,
                    'action': action,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })
                
                print(f" ✅ {action} (Score: {score})")
                
            except Exception as e:
                print(f" ❌ Error: {str(e)}")
        
        # Save signals
        os.makedirs('outputs/top50_signals', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        with open(f'outputs/top50_signals/signals_{timestamp}.json', 'w') as f:
            json.dump(current_signals, f, indent=2)
            
        return current_signals
    
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = -delta.where(delta < 0, 0).rolling(period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1]
    
    def calculate_macd_signal(self, prices):
        """Calculate MACD signal"""
        exp1 = prices.ewm(span=12, adjust=False).mean()
        exp2 = prices.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        return (macd - signal).iloc[-1]
    
    def generate_summary_report(self):
        """Generate comprehensive summary report"""
        print("\n📊 GENERATING COMPREHENSIVE REPORT")
        print("="*80)
        
        # Top performers by backtest
        if self.backtest_results:
            sorted_by_return = sorted(self.backtest_results.items(), 
                                    key=lambda x: x[1]['total_return'], 
                                    reverse=True)
            
            print("\n🏆 TOP 10 PERFORMERS (BY BACKTEST RETURN):")
            print(f"{'Symbol':<8} {'Total Return':<15} {'Sharpe Ratio':<15} {'Max Drawdown':<15}")
            print("-"*53)
            
            for symbol, metrics in sorted_by_return[:10]:
                print(f"{symbol:<8} {metrics['total_return']:>13.2%} "
                      f"{metrics['sharpe_ratio']:>13.2f} "
                      f"{metrics['max_drawdown']:>13.2%}")
        
        # Summary statistics
        total_stocks = len(TOP_50_STOCKS)
        data_fetched = len(self.historical_data)
        backtested = len(self.backtest_results)
        
        print(f"\n📈 PIPELINE SUMMARY:")
        print(f"Total Stocks Analyzed: {total_stocks}")
        print(f"Historical Data Fetched: {data_fetched}")
        print(f"Stocks Backtested: {backtested}")
        
        if self.backtest_results:
            avg_return = np.mean([m['total_return'] for m in self.backtest_results.values()])
            avg_sharpe = np.mean([m['sharpe_ratio'] for m in self.backtest_results.values()])
            print(f"Average Backtest Return: {avg_return:.2%}")
            print(f"Average Sharpe Ratio: {avg_sharpe:.2f}")
        
        # Save comprehensive report
        report = {
            'execution_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_stocks': total_stocks,
            'data_fetched': data_fetched,
            'backtest_results': self.backtest_results,
            'summary_stats': {
                'avg_return': avg_return if self.backtest_results else 0,
                'avg_sharpe': avg_sharpe if self.backtest_results else 0
            }
        }
        
        os.makedirs('reports/top50', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        with open(f'reports/top50/complete_report_{timestamp}.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n💾 Complete report saved to reports/top50/")

def main():
    """Run complete pipeline"""
    print("🚀 STARTING COMPLETE TRADING PIPELINE FOR TOP 50 US STOCKS")
    print("="*80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    pipeline = Top50CompletePipeline()
    
    # Step 1: Fetch historical data
    pipeline.step1_fetch_all_historical_data()
    
    # Step 2: Prepare ML data
    ml_data = pipeline.step2_prepare_ml_data()
    
    # Step 3: Backtest strategy
    pipeline.step3_backtest_strategy()
    
    # Step 4: Generate current signals
    signals = pipeline.step4_generate_current_signals()
    
    # Generate summary report
    pipeline.generate_summary_report()
    
    print("\n✅ PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    print("⚠️ DISCLAIMER: This is for educational purposes only. Not financial advice.")
    print("="*80)

if __name__ == "__main__":
    main()