#!/usr/bin/env python3
"""
Live Swing Trading Signals
Combines real-time data with swing trading system for today's signals
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
from typing import Dict, List, Tuple

class LiveSwingSignals:
    def __init__(self):
        self.symbols = ['AAPL', 'GOOGL', 'TSLA', 'MSFT', 'UNH']
        self.current_date = datetime.now()
        
    def fetch_live_data(self, symbol: str, days: int = 250) -> pd.DataFrame:
        """Fetch recent data including today's live prices."""
        try:
            ticker = yf.Ticker(symbol)
            
            # Get historical data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            df = ticker.history(start=start_date, end=end_date, interval='1d')
            
            if df.empty:
                print(f"No data available for {symbol}")
                return pd.DataFrame()
            
            return df
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all swing trading indicators."""
        # EMAs
        df['EMA_5'] = df['Close'].ewm(span=5, adjust=False).mean()
        df['EMA_8'] = df['Close'].ewm(span=8, adjust=False).mean()
        df['EMA_13'] = df['Close'].ewm(span=13, adjust=False).mean()
        df['EMA_21'] = df['Close'].ewm(span=21, adjust=False).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        
        # ATR
        high_low = df['High'] - df['Low']
        high_close = abs(df['High'] - df['Close'].shift())
        low_close = abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['ATR'] = true_range.rolling(14).mean()
        df['ATR_10'] = true_range.rolling(10).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=9).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=9).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        df['RSI_Slope'] = df['RSI'].diff(3)
        
        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        df['MACD_Slope'] = df['MACD_Histogram'].diff()
        
        # Volume
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        # Momentum
        df['ROC_5'] = (df['Close'] - df['Close'].shift(5)) / df['Close'].shift(5) * 100
        df['ROC_10'] = (df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10) * 100
        
        # Support/Resistance
        df['Resistance_20'] = df['High'].rolling(window=20).max()
        df['Support_20'] = df['Low'].rolling(window=20).min()
        df['Range_Position'] = (df['Close'] - df['Support_20']) / (df['Resistance_20'] - df['Support_20'])
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (2 * bb_std)
        df['BB_Lower'] = df['BB_Middle'] - (2 * bb_std)
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # Volatility
        df['Volatility'] = df['Close'].pct_change().rolling(20).std() * np.sqrt(252)
        df['Vol_Percentile'] = df['Volatility'].rolling(100).rank(pct=True)
        
        # Trend
        df['Uptrend'] = ((df['SMA_50'] > df['SMA_200']) & (df['Close'] > df['SMA_50'])).astype(int)
        
        return df
    
    def calculate_signal_strength(self, df: pd.DataFrame, i: int) -> Tuple[float, float, float, List[str]]:
        """Calculate detailed signal strength and components."""
        trend_score = 0
        momentum_score = 0
        setup_score = 0
        factors = []
        
        # Trend Analysis (0-100 points)
        if df['EMA_5'].iloc[i] > df['EMA_8'].iloc[i] > df['EMA_13'].iloc[i] > df['EMA_21'].iloc[i]:
            trend_score += 30
            factors.append("Perfect EMA alignment")
        elif df['EMA_8'].iloc[i] > df['EMA_13'].iloc[i] > df['EMA_21'].iloc[i]:
            trend_score += 20
            factors.append("Good EMA alignment")
        elif df['EMA_13'].iloc[i] > df['EMA_21'].iloc[i]:
            trend_score += 10
            factors.append("Basic uptrend")
        
        if df['Close'].iloc[i] > df['SMA_50'].iloc[i]:
            trend_score += 10
            factors.append("Above SMA50")
        
        if df['SMA_50'].iloc[i] > df['SMA_200'].iloc[i]:
            trend_score += 10
            factors.append("50>200 Golden Cross")
        
        # Momentum Analysis (0-80 points)
        rsi = df['RSI'].iloc[i]
        if 40 < rsi < 60:
            momentum_score += 20
            factors.append(f"RSI healthy ({rsi:.1f})")
        elif 30 < rsi < 40 and df['RSI_Slope'].iloc[i] > 0:
            momentum_score += 25
            factors.append(f"RSI oversold bounce ({rsi:.1f})")
        
        if df['MACD'].iloc[i] > df['MACD_Signal'].iloc[i]:
            momentum_score += 15
            factors.append("MACD bullish")
            if df['MACD_Slope'].iloc[i] > 0:
                momentum_score += 10
                factors.append("MACD accelerating")
        
        if df['ROC_5'].iloc[i] > 0:
            momentum_score += 10
            factors.append(f"5-day momentum +{df['ROC_5'].iloc[i]:.1f}%")
        
        if df['Volume_Ratio'].iloc[i] > 1.2:
            momentum_score += 15
            factors.append(f"Volume surge {df['Volume_Ratio'].iloc[i]:.1f}x")
        
        # Setup Quality (0-80 points)
        range_pos = df['Range_Position'].iloc[i]
        if 0.2 < range_pos < 0.5:
            setup_score += 20
            factors.append(f"Near support ({range_pos:.0%} of range)")
        
        ema21_distance = abs(df['Close'].iloc[i] - df['EMA_21'].iloc[i]) / df['EMA_21'].iloc[i]
        if ema21_distance < 0.02:
            setup_score += 15
            factors.append("At EMA21 support")
        
        if df['BB_Position'].iloc[i] < 0.3:
            setup_score += 15
            factors.append("Near Bollinger lower band")
        
        if df['Vol_Percentile'].iloc[i] < 0.5:
            setup_score += 10
            factors.append("Low volatility environment")
        
        total_score = trend_score + momentum_score + setup_score
        
        return total_score, trend_score, momentum_score, factors
    
    def generate_live_signals(self, symbol: str) -> Dict:
        """Generate live trading signals for a symbol."""
        # Fetch data
        df = self.fetch_live_data(symbol)
        if df.empty:
            return {'symbol': symbol, 'error': 'No data available'}
        
        # Calculate indicators
        df = self.calculate_indicators(df)
        
        # Get latest values
        latest_idx = -1
        current_price = df['Close'].iloc[latest_idx]
        
        # Check if in uptrend
        in_uptrend = df['Uptrend'].iloc[latest_idx]
        
        # Calculate signal strength
        total_score, trend_score, momentum_score, factors = self.calculate_signal_strength(df, latest_idx)
        
        # Determine signal
        if not in_uptrend:
            signal = "NO TRADE - Not in uptrend"
            action = "WAIT"
        elif total_score >= 65:
            signal = "BUY SIGNAL"
            action = "BUY"
        else:
            signal = "NO SIGNAL"
            action = "WAIT"
        
        # Calculate entry, stop, and targets
        atr = df['ATR'].iloc[latest_idx]
        
        # Dynamic stop
        atr_stop = current_price - (1.5 * atr)
        support_stop = df['Support_20'].iloc[latest_idx] * 0.995
        stop_loss = max(atr_stop, support_stop)
        
        # Ensure minimum risk
        min_risk = current_price * 0.01
        if current_price - stop_loss < min_risk:
            stop_loss = current_price - min_risk
        
        # Targets
        risk = current_price - stop_loss
        take_profit_1 = current_price + (1.8 * risk)
        take_profit_2 = current_price + (3.0 * risk)
        
        # Risk metrics
        risk_pct = (current_price - stop_loss) / current_price * 100
        reward_pct_1 = (take_profit_1 - current_price) / current_price * 100
        reward_pct_2 = (take_profit_2 - current_price) / current_price * 100
        
        # Position sizing (2% risk on $10,000)
        risk_amount = 200  # $200 risk
        risk_per_share = current_price - stop_loss
        shares = int(risk_amount / risk_per_share) if risk_per_share > 0 else 0
        position_value = shares * current_price
        
        # Get current market data
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        result = {
            'symbol': symbol,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'signal': signal,
            'action': action,
            'signal_strength': {
                'total_score': total_score,
                'trend_score': trend_score,
                'momentum_score': momentum_score,
                'minimum_required': 65
            },
            'current_data': {
                'price': round(current_price, 2),
                'volume': int(df['Volume'].iloc[latest_idx]),
                'volume_ratio': round(df['Volume_Ratio'].iloc[latest_idx], 2),
                'day_change': round(current_price - df['Close'].iloc[-2], 2),
                'day_change_pct': round((current_price - df['Close'].iloc[-2]) / df['Close'].iloc[-2] * 100, 2)
            },
            'technical_indicators': {
                'rsi': round(df['RSI'].iloc[latest_idx], 2),
                'macd': round(df['MACD'].iloc[latest_idx], 3),
                'macd_signal': round(df['MACD_Signal'].iloc[latest_idx], 3),
                'ema_21': round(df['EMA_21'].iloc[latest_idx], 2),
                'sma_50': round(df['SMA_50'].iloc[latest_idx], 2),
                'sma_200': round(df['SMA_200'].iloc[latest_idx], 2),
                'atr': round(atr, 2),
                'bb_position': round(df['BB_Position'].iloc[latest_idx], 2),
                'range_position': round(df['Range_Position'].iloc[latest_idx], 2)
            },
            'trade_setup': {
                'entry_price': round(current_price, 2),
                'stop_loss': round(stop_loss, 2),
                'take_profit_1': round(take_profit_1, 2),
                'take_profit_2': round(take_profit_2, 2),
                'risk_pct': round(risk_pct, 2),
                'reward_pct_1': round(reward_pct_1, 2),
                'reward_pct_2': round(reward_pct_2, 2),
                'risk_reward_1': round(reward_pct_1 / risk_pct, 2) if risk_pct > 0 else 0,
                'risk_reward_2': round(reward_pct_2 / risk_pct, 2) if risk_pct > 0 else 0
            },
            'position_sizing': {
                'shares_to_buy': shares,
                'position_value': round(position_value, 2),
                'risk_amount': risk_amount,
                'account_size': 10000,
                'position_pct': round(position_value / 10000 * 100, 2)
            },
            'signal_factors': factors,
            'market_info': {
                'in_uptrend': bool(in_uptrend),
                'volatility_percentile': round(df['Vol_Percentile'].iloc[latest_idx] * 100, 1),
                '52_week_high': info.get('fiftyTwoWeekHigh', 'N/A'),
                '52_week_low': info.get('fiftyTwoWeekLow', 'N/A'),
                'market_cap': info.get('marketCap', 'N/A')
            }
        }
        
        return result
    
    def analyze_all_symbols(self):
        """Analyze all symbols and generate report."""
        print("🎯 LIVE SWING TRADING SIGNALS")
        print("="*80)
        print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Strategy: 2-15 Day Swing Trading System")
        print(f"Risk: 2% per trade | Account: $10,000")
        print("="*80)
        
        all_signals = {}
        buy_signals = []
        
        for symbol in self.symbols:
            print(f"\nAnalyzing {symbol}...")
            signal_data = self.generate_live_signals(symbol)
            all_signals[symbol] = signal_data
            
            if signal_data.get('action') == 'BUY':
                buy_signals.append(symbol)
        
        # Display detailed results
        print("\n" + "="*80)
        print("📊 DETAILED SIGNAL ANALYSIS")
        print("="*80)
        
        for symbol, data in all_signals.items():
            if 'error' in data:
                print(f"\n{symbol}: ERROR - {data['error']}")
                continue
            
            print(f"\n{'🟢' if data['action'] == 'BUY' else '🔴'} {symbol} - {data['signal']}")
            print("-"*60)
            
            # Current data
            print(f"Current Price: ${data['current_data']['price']}")
            print(f"Day Change: ${data['current_data']['day_change']} ({data['current_data']['day_change_pct']}%)")
            print(f"Volume Ratio: {data['current_data']['volume_ratio']}x average")
            
            # Signal strength
            print(f"\nSignal Strength: {data['signal_strength']['total_score']}/260")
            print(f"  • Trend: {data['signal_strength']['trend_score']}/100")
            print(f"  • Momentum: {data['signal_strength']['momentum_score']}/80")
            print(f"  • Required: {data['signal_strength']['minimum_required']}")
            
            # Technical indicators
            tech = data['technical_indicators']
            print(f"\nTechnical Indicators:")
            print(f"  • RSI: {tech['rsi']}")
            print(f"  • MACD: {tech['macd']} (Signal: {tech['macd_signal']})")
            print(f"  • Price vs EMA21: ${data['current_data']['price']} vs ${tech['ema_21']}")
            print(f"  • Range Position: {tech['range_position']:.0%}")
            
            if data['action'] == 'BUY':
                # Trade setup
                setup = data['trade_setup']
                print(f"\n💰 TRADE SETUP:")
                print(f"  Entry: ${setup['entry_price']}")
                print(f"  Stop Loss: ${setup['stop_loss']} (-{setup['risk_pct']}%)")
                print(f"  Target 1: ${setup['take_profit_1']} (+{setup['reward_pct_1']}%) [R:R {setup['risk_reward_1']}]")
                print(f"  Target 2: ${setup['take_profit_2']} (+{setup['reward_pct_2']}%) [R:R {setup['risk_reward_2']}]")
                
                # Position sizing
                pos = data['position_sizing']
                print(f"\n📊 POSITION SIZING:")
                print(f"  Shares: {pos['shares_to_buy']}")
                print(f"  Position Value: ${pos['position_value']} ({pos['position_pct']}% of account)")
                print(f"  Risk Amount: ${pos['risk_amount']}")
            
            # Signal factors
            if data['signal_factors']:
                print(f"\n✅ Positive Factors:")
                for factor in data['signal_factors']:
                    print(f"  • {factor}")
        
        # Summary
        print("\n" + "="*80)
        print("📈 SUMMARY")
        print("="*80)
        
        if buy_signals:
            print(f"\n🟢 BUY SIGNALS TODAY: {', '.join(buy_signals)}")
            print("\nRECOMMENDED ACTIONS:")
            for symbol in buy_signals:
                setup = all_signals[symbol]['trade_setup']
                pos = all_signals[symbol]['position_sizing']
                print(f"\n{symbol}:")
                print(f"  1. Place BUY order for {pos['shares_to_buy']} shares at ${setup['entry_price']}")
                print(f"  2. Set STOP LOSS at ${setup['stop_loss']}")
                print(f"  3. Set TAKE PROFIT orders at ${setup['take_profit_1']} and ${setup['take_profit_2']}")
        else:
            print("\n⚠️ NO BUY SIGNALS TODAY")
            print("Market conditions do not meet swing trading criteria.")
            print("Continue monitoring for better setups.")
        
        # Risk reminder
        print("\n" + "="*80)
        print("⚠️ RISK MANAGEMENT REMINDER")
        print("="*80)
        print("• Never risk more than 2% per trade")
        print("• Use stop losses on ALL positions")
        print("• Start with small positions to test the system")
        print("• Monitor positions daily for exit signals")
        print("• This is not financial advice - trade at your own risk")
        
        # Save results
        filename = f"live_signals_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(all_signals, f, indent=2)
        print(f"\n💾 Detailed signals saved to {filename}")
        
        return all_signals


def main():
    """Run live signal analysis."""
    analyzer = LiveSwingSignals()
    signals = analyzer.analyze_all_symbols()
    
    # Additional analysis for positions
    print("\n" + "="*80)
    print("📊 MARKET CONTEXT")
    print("="*80)
    
    for symbol, data in signals.items():
        if 'error' not in data:
            market = data['market_info']
            current = data['current_data']['price']
            
            if isinstance(market['52_week_high'], (int, float)):
                pct_from_high = (current - market['52_week_high']) / market['52_week_high'] * 100
                pct_from_low = (current - market['52_week_low']) / market['52_week_low'] * 100
                
                print(f"\n{symbol}:")
                print(f"  52-Week Range: ${market['52_week_low']:.2f} - ${market['52_week_high']:.2f}")
                print(f"  Position in Range: {pct_from_low:.1f}% from low, {pct_from_high:.1f}% from high")
                print(f"  Volatility Percentile: {market['volatility_percentile']:.0f}%")


if __name__ == "__main__":
    main()