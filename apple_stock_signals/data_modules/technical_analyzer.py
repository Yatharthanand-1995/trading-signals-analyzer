import pandas as pd
import numpy as np
import talib
from core_scripts.config import TECHNICAL_SETTINGS

class AppleTechnicalAnalyzer:
    def __init__(self):
        self.indicators = {}
        
    def calculate_all_indicators(self, stock_data):
        """Calculate all technical indicators and display them"""
        print("\n🔧 Calculating Technical Indicators...")
        
        # Handle both DataFrame and dict input
        if isinstance(stock_data, dict) and 'historical_data' in stock_data:
            df = stock_data['historical_data']
        elif isinstance(stock_data, pd.DataFrame):
            df = stock_data
        else:
            df = stock_data
        close = df['Close'].values.astype(np.float64)
        high = df['High'].values.astype(np.float64)
        low = df['Low'].values.astype(np.float64)
        volume = df['Volume'].values.astype(np.float64)
        
        # Calculate indicators
        indicators = {}
        
        # Moving Averages
        indicators['SMA_20'] = talib.SMA(close, timeperiod=TECHNICAL_SETTINGS['SMA_PERIODS'][0])
        indicators['SMA_50'] = talib.SMA(close, timeperiod=TECHNICAL_SETTINGS['SMA_PERIODS'][1])
        indicators['SMA_200'] = talib.SMA(close, timeperiod=TECHNICAL_SETTINGS['SMA_PERIODS'][2])
        indicators['EMA_12'] = talib.EMA(close, timeperiod=TECHNICAL_SETTINGS['MACD_FAST'])
        indicators['EMA_26'] = talib.EMA(close, timeperiod=TECHNICAL_SETTINGS['MACD_SLOW'])
        
        # MACD
        indicators['MACD'], indicators['MACD_signal'], indicators['MACD_hist'] = talib.MACD(
            close, 
            fastperiod=TECHNICAL_SETTINGS['MACD_FAST'],
            slowperiod=TECHNICAL_SETTINGS['MACD_SLOW'],
            signalperiod=TECHNICAL_SETTINGS['MACD_SIGNAL']
        )
        
        # RSI
        indicators['RSI'] = talib.RSI(close, timeperiod=TECHNICAL_SETTINGS['RSI_PERIOD'])
        
        # Bollinger Bands
        indicators['BB_upper'], indicators['BB_middle'], indicators['BB_lower'] = talib.BBANDS(
            close,
            timeperiod=TECHNICAL_SETTINGS['BOLLINGER_PERIOD'],
            nbdevup=TECHNICAL_SETTINGS['BOLLINGER_STD'],
            nbdevdn=TECHNICAL_SETTINGS['BOLLINGER_STD']
        )
        
        # Stochastic
        indicators['STOCH_K'], indicators['STOCH_D'] = talib.STOCH(high, low, close)
        
        # ATR
        indicators['ATR'] = talib.ATR(high, low, close, timeperiod=TECHNICAL_SETTINGS['ATR_PERIOD'])
        
        # Williams %R
        indicators['WILLIAMS_R'] = talib.WILLR(high, low, close, timeperiod=14)
        
        # Volume indicators
        indicators['OBV'] = talib.OBV(close, volume)
        
        # ADX
        indicators['ADX'] = talib.ADX(high, low, close, timeperiod=14)
        
        # CCI
        indicators['CCI'] = talib.CCI(high, low, close, timeperiod=20)
        
        # MFI
        indicators['MFI'] = talib.MFI(high, low, close, volume, timeperiod=14)
        
        # Add current price and volume data
        indicators['current_price'] = close[-1]
        # Calculate volume ratio if not provided
        if isinstance(stock_data, dict) and 'volume_ratio' in stock_data:
            indicators['volume_ratio'] = stock_data['volume_ratio']
        else:
            # Calculate it from the data
            volume_avg = df['Volume'].rolling(window=20).mean().iloc[-1]
            current_volume = df['Volume'].iloc[-1]
            indicators['volume_ratio'] = current_volume / volume_avg if volume_avg > 0 else 1.0
        
        # Display calculated indicators
        self.display_technical_indicators(indicators)
        
        return indicators
    
    def display_technical_indicators(self, indicators):
        """Display all technical indicators with current values"""
        print("\n📈 TECHNICAL INDICATORS:")
        
        current_price = indicators['current_price']
        
        # Moving Averages
        print(f"\n🔄 Moving Averages:")
        print(f"Current Price: ${current_price:.2f}")
        print(f"SMA 20: ${indicators['SMA_20'][-1]:.2f}")
        print(f"SMA 50: ${indicators['SMA_50'][-1]:.2f}")
        print(f"SMA 200: ${indicators['SMA_200'][-1]:.2f}")
        print(f"EMA 12: ${indicators['EMA_12'][-1]:.2f}")
        print(f"EMA 26: ${indicators['EMA_26'][-1]:.2f}")
        
        # Trend Analysis
        sma_20 = indicators['SMA_20'][-1]
        sma_50 = indicators['SMA_50'][-1]
        sma_200 = indicators['SMA_200'][-1]
        
        if sma_20 > sma_50 > sma_200:
            trend = "🟢 Strong Uptrend"
        elif sma_20 > sma_50:
            trend = "🟡 Short-term Uptrend"
        elif sma_20 < sma_50 < sma_200:
            trend = "🔴 Strong Downtrend"
        elif sma_20 < sma_50:
            trend = "🟡 Short-term Downtrend"
        else:
            trend = "🟡 Sideways/Mixed"
        
        print(f"Overall Trend: {trend}")
        
        # Momentum Indicators
        print(f"\n⚡ Momentum Indicators:")
        rsi = indicators['RSI'][-1]
        print(f"RSI (14): {rsi:.2f}")
        
        if rsi < 30:
            rsi_signal = "🟢 Oversold (Bullish)"
        elif rsi > 70:
            rsi_signal = "🔴 Overbought (Bearish)"
        elif rsi < 40:
            rsi_signal = "🟡 Approaching Oversold"
        elif rsi > 60:
            rsi_signal = "🟡 Approaching Overbought"
        else:
            rsi_signal = "🟡 Neutral"
        
        print(f"RSI Signal: {rsi_signal}")
        
        # MACD
        macd = indicators['MACD'][-1]
        macd_signal = indicators['MACD_signal'][-1]
        macd_hist = indicators['MACD_hist'][-1]
        
        print(f"MACD: {macd:.4f}")
        print(f"MACD Signal: {macd_signal:.4f}")
        print(f"MACD Histogram: {macd_hist:.4f}")
        
        if macd > macd_signal:
            macd_signal_text = "🟢 Bullish (MACD > Signal)"
        else:
            macd_signal_text = "🔴 Bearish (MACD < Signal)"
        
        print(f"MACD Signal: {macd_signal_text}")
        
        # Stochastic
        stoch_k = indicators['STOCH_K'][-1]
        stoch_d = indicators['STOCH_D'][-1]
        print(f"Stochastic %K: {stoch_k:.2f}")
        print(f"Stochastic %D: {stoch_d:.2f}")
        
        if stoch_k < 20:
            stoch_signal = "🟢 Oversold"
        elif stoch_k > 80:
            stoch_signal = "🔴 Overbought"
        else:
            stoch_signal = "🟡 Neutral"
        
        print(f"Stochastic Signal: {stoch_signal}")
        
        # Bollinger Bands
        print(f"\n📊 Bollinger Bands:")
        bb_upper = indicators['BB_upper'][-1]
        bb_middle = indicators['BB_middle'][-1]
        bb_lower = indicators['BB_lower'][-1]
        
        print(f"Upper Band: ${bb_upper:.2f}")
        print(f"Middle Band: ${bb_middle:.2f}")
        print(f"Lower Band: ${bb_lower:.2f}")
        
        # BB Position
        if current_price > bb_upper:
            bb_position = "🔴 Above Upper Band (Overbought)"
        elif current_price < bb_lower:
            bb_position = "🟢 Below Lower Band (Oversold)"
        elif current_price > bb_middle:
            bb_position = "🟡 Above Middle (Bullish Bias)"
        else:
            bb_position = "🟡 Below Middle (Bearish Bias)"
        
        print(f"Price Position: {bb_position}")
        
        # Volume Analysis
        print(f"\n📊 Volume Analysis:")
        print(f"Volume Ratio: {indicators['volume_ratio']:.2f}x")
        
        if indicators['volume_ratio'] > 2.0:
            volume_signal = "🟢 Very High Volume"
        elif indicators['volume_ratio'] > 1.5:
            volume_signal = "🟡 High Volume"
        elif indicators['volume_ratio'] < 0.5:
            volume_signal = "🔴 Low Volume"
        else:
            volume_signal = "🟡 Normal Volume"
        
        print(f"Volume Signal: {volume_signal}")
        
        # ATR (Volatility)
        print(f"\n📈 Volatility:")
        atr = indicators['ATR'][-1]
        print(f"ATR (14): ${atr:.2f}")
        
        atr_percent = (atr / current_price) * 100
        print(f"ATR as % of Price: {atr_percent:.2f}%")
        
        if atr_percent > 3:
            volatility_signal = "🔴 High Volatility"
        elif atr_percent < 1:
            volatility_signal = "🟢 Low Volatility"
        else:
            volatility_signal = "🟡 Normal Volatility"
        
        print(f"Volatility Signal: {volatility_signal}")
        
        # Additional Indicators
        print(f"\n📊 Additional Indicators:")
        print(f"ADX (14): {indicators['ADX'][-1]:.2f} - ", end='')
        if indicators['ADX'][-1] > 25:
            print("Strong Trend")
        else:
            print("Weak/No Trend")
        
        print(f"CCI (20): {indicators['CCI'][-1]:.2f} - ", end='')
        if indicators['CCI'][-1] > 100:
            print("Overbought")
        elif indicators['CCI'][-1] < -100:
            print("Oversold")
        else:
            print("Neutral")
        
        print(f"MFI (14): {indicators['MFI'][-1]:.2f} - ", end='')
        if indicators['MFI'][-1] > 80:
            print("Overbought")
        elif indicators['MFI'][-1] < 20:
            print("Oversold")
        else:
            print("Neutral")