#!/usr/bin/env python3
"""
Enhanced Feature Engineering for ML Trading Models
Implements advanced features including market microstructure, sentiment, and intermarket relationships
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class EnhancedFeatureEngineer:
    """
    Creates sophisticated features for ML models:
    - Market microstructure features
    - Intermarket relationships
    - Sentiment indicators
    - Seasonality features
    - Advanced technical features
    """
    
    def __init__(self):
        self.feature_groups = {
            'price_action': self._create_price_action_features,
            'volume_profile': self._create_volume_profile_features,
            'market_microstructure': self._create_microstructure_features,
            'intermarket': self._create_intermarket_features,
            'sentiment': self._create_sentiment_features,
            'seasonality': self._create_seasonality_features,
            'advanced_technical': self._create_advanced_technical_features,
            'pattern_recognition': self._create_pattern_features
        }
        
        # Cache for intermarket data
        self.intermarket_cache = {}
        
    def engineer_features(self, data: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
        """
        Create all feature groups for the given data
        """
        print(f"\n🔧 Engineering features for {symbol if symbol else 'data'}...")
        
        # Initialize features dataframe
        features = pd.DataFrame(index=data.index)
        
        # Add OHLCV data
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in data.columns:
                features[col] = data[col]
        
        # Generate each feature group
        for group_name, feature_func in self.feature_groups.items():
            try:
                print(f"  Creating {group_name} features...", end='', flush=True)
                group_features = feature_func(data, symbol)
                
                # Add features to main dataframe
                for col in group_features.columns:
                    features[f"{group_name}_{col}"] = group_features[col]
                
                print(f" ✅ ({len(group_features.columns)} features)")
            except Exception as e:
                print(f" ❌ Error: {str(e)}")
        
        # Create target variables
        features = self._create_target_variables(features)
        
        # Remove any remaining NaN values
        features = features.dropna()
        
        print(f"\n📊 Total features created: {len(features.columns)}")
        print(f"📈 Data points after cleaning: {len(features)}")
        
        return features
    
    def _create_price_action_features(self, data: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
        """
        Advanced price action features
        """
        features = pd.DataFrame(index=data.index)
        
        # Returns at multiple horizons
        for period in [1, 2, 3, 5, 10, 20]:
            features[f'returns_{period}d'] = data['Close'].pct_change(period)
            features[f'log_returns_{period}d'] = np.log(data['Close'] / data['Close'].shift(period))
        
        # Price relative to various levels
        features['price_to_high_20d'] = data['Close'] / data['High'].rolling(20).max()
        features['price_to_low_20d'] = data['Close'] / data['Low'].rolling(20).min()
        features['price_to_high_52w'] = data['Close'] / data['High'].rolling(252).max()
        
        # Volatility features
        returns = data['Close'].pct_change()
        for period in [5, 10, 20, 60]:
            features[f'volatility_{period}d'] = returns.rolling(period).std() * np.sqrt(252)
            features[f'volatility_ratio_{period}d'] = (
                returns.rolling(period).std() / returns.rolling(period * 2).std()
            )
        
        # Range features
        features['true_range'] = pd.concat([
            data['High'] - data['Low'],
            abs(data['High'] - data['Close'].shift(1)),
            abs(data['Low'] - data['Close'].shift(1))
        ], axis=1).max(axis=1)
        
        features['avg_true_range_14'] = features['true_range'].rolling(14).mean()
        features['range_position'] = (data['Close'] - data['Low']) / (data['High'] - data['Low'])
        
        # Gap features
        features['gap_open'] = (data['Open'] - data['Close'].shift(1)) / data['Close'].shift(1)
        features['gap_filled'] = (
            ((data['Open'] > data['Close'].shift(1)) & (data['Low'] <= data['Close'].shift(1))) |
            ((data['Open'] < data['Close'].shift(1)) & (data['High'] >= data['Close'].shift(1)))
        ).astype(int)
        
        # Trend strength
        features['trend_strength_20d'] = self._calculate_trend_strength(data['Close'], 20)
        features['trend_strength_50d'] = self._calculate_trend_strength(data['Close'], 50)
        
        return features
    
    def _create_volume_profile_features(self, data: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
        """
        Volume profile and volume-based features
        """
        features = pd.DataFrame(index=data.index)
        
        # Volume moving averages and ratios
        for period in [5, 10, 20, 50]:
            features[f'volume_ma_{period}'] = data['Volume'].rolling(period).mean()
            features[f'volume_ratio_{period}'] = data['Volume'] / features[f'volume_ma_{period}']
        
        # Volume-weighted average price (VWAP)
        features['vwap'] = (data['Volume'] * (data['High'] + data['Low'] + data['Close']) / 3).cumsum() / data['Volume'].cumsum()
        features['price_to_vwap'] = data['Close'] / features['vwap']
        
        # On-Balance Volume (OBV)
        obv = pd.Series(index=data.index, dtype=float)
        obv.iloc[0] = 0
        for i in range(1, len(data)):
            if data['Close'].iloc[i] > data['Close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + data['Volume'].iloc[i]
            elif data['Close'].iloc[i] < data['Close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - data['Volume'].iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        features['obv'] = obv
        features['obv_ema_21'] = obv.ewm(span=21).mean()
        features['obv_trend'] = features['obv'] - features['obv_ema_21']
        
        # Accumulation/Distribution Line
        mfm = ((data['Close'] - data['Low']) - (data['High'] - data['Close'])) / (data['High'] - data['Low'])
        mfm = mfm.fillna(0)
        mfv = mfm * data['Volume']
        features['adl'] = mfv.cumsum()
        features['adl_trend'] = features['adl'] - features['adl'].rolling(21).mean()
        
        # Volume price trend (VPT)
        features['vpt'] = (data['Volume'] * data['Close'].pct_change()).cumsum()
        
        # Money Flow Index (MFI)
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        money_flow = typical_price * data['Volume']
        
        positive_flow = pd.Series(0, index=data.index)
        negative_flow = pd.Series(0, index=data.index)
        
        for i in range(1, len(data)):
            if typical_price.iloc[i] > typical_price.iloc[i-1]:
                positive_flow.iloc[i] = money_flow.iloc[i]
            else:
                negative_flow.iloc[i] = money_flow.iloc[i]
        
        positive_mf = positive_flow.rolling(14).sum()
        negative_mf = negative_flow.rolling(14).sum()
        
        mfi = 100 - (100 / (1 + positive_mf / negative_mf))
        features['mfi'] = mfi
        
        # Volume climax detection
        features['volume_spike'] = (data['Volume'] > data['Volume'].rolling(20).mean() * 2).astype(int)
        features['volume_dry_up'] = (data['Volume'] < data['Volume'].rolling(20).mean() * 0.5).astype(int)
        
        return features
    
    def _create_microstructure_features(self, data: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
        """
        Market microstructure features (bid-ask spread, order flow, etc.)
        """
        features = pd.DataFrame(index=data.index)
        
        # Approximate bid-ask spread using high-low
        features['hl_spread'] = (data['High'] - data['Low']) / data['Close']
        features['hl_spread_ma'] = features['hl_spread'].rolling(20).mean()
        features['relative_spread'] = features['hl_spread'] / features['hl_spread_ma']
        
        # Kyle's lambda (price impact)
        returns = data['Close'].pct_change()
        volume_change = data['Volume'].pct_change()
        features['kyle_lambda'] = returns.rolling(20).cov(volume_change) / volume_change.rolling(20).var()
        
        # Amihud illiquidity measure
        features['amihud_illiquidity'] = abs(returns) / (data['Volume'] * data['Close'])
        features['amihud_ma'] = features['amihud_illiquidity'].rolling(20).mean()
        
        # Microstructure noise (Hasbrouck's measure)
        features['microstructure_noise'] = returns.rolling(5).std() / returns.rolling(20).std()
        
        # Order flow imbalance (approximation)
        features['close_to_high'] = (data['Close'] - data['Low']) / (data['High'] - data['Low'])
        features['order_flow_imbalance'] = features['close_to_high'].rolling(10).mean()
        
        # Effective spread proxy
        mid_price = (data['High'] + data['Low']) / 2
        features['effective_spread'] = 2 * abs(data['Close'] - mid_price) / mid_price
        
        # Roll's implied spread
        returns_diff = returns.diff()
        cov_returns = returns_diff.rolling(20).cov(returns_diff.shift(1))
        features['roll_spread'] = 2 * np.sqrt(-cov_returns.clip(upper=0))
        
        # PIN (Probability of Informed Trading) proxy
        volume_imbalance = abs(data['Volume'] - data['Volume'].rolling(20).mean())
        features['pin_proxy'] = volume_imbalance / data['Volume'].rolling(20).sum()
        
        return features
    
    def _create_intermarket_features(self, data: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
        """
        Intermarket relationships (bonds, dollar, commodities, etc.)
        """
        features = pd.DataFrame(index=data.index)
        
        # Define intermarket symbols
        intermarket_symbols = {
            'spy': 'SPY',      # S&P 500
            'qqq': 'QQQ',      # Nasdaq 100
            'iwm': 'IWM',      # Russell 2000
            'tlt': 'TLT',      # 20+ Year Treasury
            'dxy': 'UUP',      # Dollar Index
            'gld': 'GLD',      # Gold
            'uso': 'USO',      # Oil
            'vix': '^VIX'      # Volatility Index
        }
        
        # Fetch intermarket data
        self._fetch_intermarket_data(intermarket_symbols, data.index[0], data.index[-1])
        
        # Calculate correlations and relationships
        if symbol and 'Close' in data.columns:
            symbol_returns = data['Close'].pct_change()
            
            for market_name, market_data in self.intermarket_cache.items():
                if not market_data.empty:
                    # Align data
                    aligned_data = market_data.reindex(data.index, method='ffill')
                    market_returns = aligned_data['Close'].pct_change()
                    
                    # Rolling correlation
                    features[f'corr_{market_name}_20d'] = symbol_returns.rolling(20).corr(market_returns)
                    features[f'corr_{market_name}_60d'] = symbol_returns.rolling(60).corr(market_returns)
                    
                    # Beta to market
                    if market_name == 'spy':
                        features['beta_20d'] = (
                            symbol_returns.rolling(20).cov(market_returns) / 
                            market_returns.rolling(20).var()
                        )
                    
                    # Relative strength
                    features[f'rel_strength_{market_name}'] = (
                        data['Close'] / data['Close'].shift(20) / 
                        (aligned_data['Close'] / aligned_data['Close'].shift(20))
                    )
        
        # Special indicators
        if 'vix' in self.intermarket_cache and not self.intermarket_cache['vix'].empty:
            vix_data = self.intermarket_cache['vix'].reindex(data.index, method='ffill')
            features['vix_level'] = vix_data['Close']
            features['vix_percentile'] = vix_data['Close'].rolling(252).rank(pct=True)
            features['vix_term_structure'] = vix_data['Close'] / vix_data['Close'].rolling(20).mean()
        
        # Bond-stock relationship
        if 'tlt' in self.intermarket_cache and 'spy' in self.intermarket_cache:
            tlt_data = self.intermarket_cache['tlt'].reindex(data.index, method='ffill')
            spy_data = self.intermarket_cache['spy'].reindex(data.index, method='ffill')
            
            features['stocks_bonds_ratio'] = spy_data['Close'] / tlt_data['Close']
            features['stocks_bonds_trend'] = (
                features['stocks_bonds_ratio'] / 
                features['stocks_bonds_ratio'].rolling(20).mean()
            )
        
        return features
    
    def _create_sentiment_features(self, data: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
        """
        Sentiment and market psychology features
        """
        features = pd.DataFrame(index=data.index)
        
        # Put/Call ratio proxy (using price action)
        returns = data['Close'].pct_change()
        down_days = (returns < 0).rolling(20).sum()
        up_days = (returns > 0).rolling(20).sum()
        features['up_down_ratio'] = up_days / (down_days + 1)
        
        # Fear and Greed components
        # 1. Momentum
        features['momentum_1m'] = data['Close'] / data['Close'].shift(20) - 1
        features['momentum_3m'] = data['Close'] / data['Close'].shift(60) - 1
        
        # 2. Strength vs Volume
        features['strength_volume'] = (
            returns.rolling(20).mean() * data['Volume'].rolling(20).mean() / 
            data['Volume'].rolling(60).mean()
        )
        
        # 3. High-Low spread
        features['high_low_spread'] = (
            (data['High'].rolling(20).max() - data['Low'].rolling(20).min()) / 
            data['Close'].rolling(20).mean()
        )
        
        # 4. Volatility regime
        features['volatility_regime'] = returns.rolling(20).std() / returns.rolling(252).std()
        
        # Market breadth proxy
        features['new_highs'] = (data['Close'] == data['Close'].rolling(252).max()).astype(int)
        features['new_lows'] = (data['Close'] == data['Close'].rolling(252).min()).astype(int)
        features['highs_lows_ratio'] = (
            features['new_highs'].rolling(20).sum() / 
            (features['new_lows'].rolling(20).sum() + 1)
        )
        
        # Advance/Decline proxy
        features['advance_decline'] = (returns > 0).rolling(20).mean()
        
        # Smart money indicators
        # Morning vs afternoon performance
        features['open_to_noon'] = (data['High'] + data['Low']) / 2 / data['Open'] - 1
        features['noon_to_close'] = data['Close'] / ((data['High'] + data['Low']) / 2) - 1
        features['smart_money_flow'] = features['noon_to_close'] - features['open_to_noon']
        
        return features
    
    def _create_seasonality_features(self, data: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
        """
        Seasonality and calendar features
        """
        features = pd.DataFrame(index=data.index)
        
        # Extract datetime components
        features['day_of_week'] = data.index.dayofweek
        features['day_of_month'] = data.index.day
        features['week_of_year'] = data.index.isocalendar().week
        features['month'] = data.index.month
        features['quarter'] = data.index.quarter
        
        # Cyclical encoding for temporal features
        features['day_of_week_sin'] = np.sin(2 * np.pi * features['day_of_week'] / 7)
        features['day_of_week_cos'] = np.cos(2 * np.pi * features['day_of_week'] / 7)
        
        features['day_of_month_sin'] = np.sin(2 * np.pi * features['day_of_month'] / 31)
        features['day_of_month_cos'] = np.cos(2 * np.pi * features['day_of_month'] / 31)
        
        features['month_sin'] = np.sin(2 * np.pi * features['month'] / 12)
        features['month_cos'] = np.cos(2 * np.pi * features['month'] / 12)
        
        # Trading day features
        features['is_monday'] = (features['day_of_week'] == 0).astype(int)
        features['is_friday'] = (features['day_of_week'] == 4).astype(int)
        features['is_month_start'] = (features['day_of_month'] <= 5).astype(int)
        features['is_month_end'] = (features['day_of_month'] >= 25).astype(int)
        features['is_quarter_end'] = (features['month'].isin([3, 6, 9, 12]) & features['is_month_end']).astype(int)
        
        # Options expiration week (third Friday)
        features['is_opex_week'] = self._is_options_expiration_week(data.index)
        
        # Holiday effects (simplified)
        features['days_to_next_holiday'] = self._days_to_next_holiday(data.index)
        features['days_from_last_holiday'] = self._days_from_last_holiday(data.index)
        
        # Earnings season proxy
        features['is_earnings_season'] = (
            features['month'].isin([1, 4, 7, 10]) & 
            (features['day_of_month'] >= 10) & 
            (features['day_of_month'] <= 25)
        ).astype(int)
        
        # Year-end effects
        features['is_december'] = (features['month'] == 12).astype(int)
        features['is_january'] = (features['month'] == 1).astype(int)
        features['is_tax_loss_season'] = (features['month'].isin([11, 12])).astype(int)
        
        return features
    
    def _create_advanced_technical_features(self, data: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
        """
        Advanced technical indicators and combinations
        """
        features = pd.DataFrame(index=data.index)
        
        # Ichimoku Cloud components
        high_9 = data['High'].rolling(9).max()
        low_9 = data['Low'].rolling(9).min()
        features['tenkan_sen'] = (high_9 + low_9) / 2
        
        high_26 = data['High'].rolling(26).max()
        low_26 = data['Low'].rolling(26).min()
        features['kijun_sen'] = (high_26 + low_26) / 2
        
        features['senkou_span_a'] = ((features['tenkan_sen'] + features['kijun_sen']) / 2).shift(26)
        
        high_52 = data['High'].rolling(52).max()
        low_52 = data['Low'].rolling(52).min()
        features['senkou_span_b'] = ((high_52 + low_52) / 2).shift(26)
        
        features['chikou_span'] = data['Close'].shift(-26)
        
        # Price relative to cloud
        features['price_above_cloud'] = (
            (data['Close'] > features['senkou_span_a']) & 
            (data['Close'] > features['senkou_span_b'])
        ).astype(int)
        
        # Keltner Channels
        ema_20 = data['Close'].ewm(span=20).mean()
        atr = features.get('avg_true_range_14', pd.Series(0, index=data.index))
        
        features['keltner_upper'] = ema_20 + (2 * atr)
        features['keltner_lower'] = ema_20 - (2 * atr)
        features['keltner_position'] = (data['Close'] - features['keltner_lower']) / (features['keltner_upper'] - features['keltner_lower'])
        
        # Donchian Channels
        features['donchian_upper'] = data['High'].rolling(20).max()
        features['donchian_lower'] = data['Low'].rolling(20).min()
        features['donchian_middle'] = (features['donchian_upper'] + features['donchian_lower']) / 2
        features['donchian_position'] = (data['Close'] - features['donchian_lower']) / (features['donchian_upper'] - features['donchian_lower'])
        
        # Average Directional Index (ADX)
        features['adx'] = self._calculate_adx(data)
        
        # Commodity Channel Index (CCI)
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        sma_tp = typical_price.rolling(20).mean()
        mad = (typical_price - sma_tp).abs().rolling(20).mean()
        features['cci'] = (typical_price - sma_tp) / (0.015 * mad)
        
        # Williams %R
        features['williams_r'] = -100 * (data['High'].rolling(14).max() - data['Close']) / (data['High'].rolling(14).max() - data['Low'].rolling(14).min())
        
        # Ultimate Oscillator
        features['ultimate_oscillator'] = self._calculate_ultimate_oscillator(data)
        
        # Stochastic RSI
        rsi = self._calculate_rsi(data['Close'])
        rsi_min = rsi.rolling(14).min()
        rsi_max = rsi.rolling(14).max()
        features['stoch_rsi'] = (rsi - rsi_min) / (rsi_max - rsi_min)
        
        # Choppiness Index
        features['choppiness'] = self._calculate_choppiness_index(data)
        
        return features
    
    def _create_pattern_features(self, data: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
        """
        Pattern recognition features
        """
        features = pd.DataFrame(index=data.index)
        
        # Candlestick patterns
        features['doji'] = self._detect_doji(data)
        features['hammer'] = self._detect_hammer(data)
        features['shooting_star'] = self._detect_shooting_star(data)
        features['engulfing'] = self._detect_engulfing(data)
        
        # Chart patterns (simplified)
        # Head and Shoulders
        features['potential_head_shoulders'] = self._detect_head_shoulders(data)
        
        # Triangle patterns
        features['triangle_forming'] = self._detect_triangle(data)
        
        # Support/Resistance levels
        features['near_resistance'] = self._near_resistance(data)
        features['near_support'] = self._near_support(data)
        
        # Breakout patterns
        features['breakout_up'] = (
            (data['Close'] > data['High'].shift(1).rolling(20).max()) & 
            (data['Volume'] > data['Volume'].rolling(20).mean() * 1.5)
        ).astype(int)
        
        features['breakout_down'] = (
            (data['Close'] < data['Low'].shift(1).rolling(20).min()) & 
            (data['Volume'] > data['Volume'].rolling(20).mean() * 1.5)
        ).astype(int)
        
        # Consolidation patterns
        features['is_consolidating'] = self._detect_consolidation(data)
        
        # Trend line breaks
        features['trendline_break'] = self._detect_trendline_break(data)
        
        return features
    
    def _create_target_variables(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Create various target variables for different ML tasks
        """
        # Future returns at different horizons
        for days in [1, 2, 3, 5, 10]:
            features[f'target_return_{days}d'] = features['Close'].pct_change(days).shift(-days)
            features[f'target_direction_{days}d'] = (features[f'target_return_{days}d'] > 0).astype(int)
        
        # Multi-class targets
        # Define return buckets
        features['target_class_5d'] = pd.cut(
            features['target_return_5d'],
            bins=[-np.inf, -0.02, -0.01, 0.01, 0.02, np.inf],
            labels=['strong_down', 'down', 'neutral', 'up', 'strong_up']
        )
        
        # Volatility targets
        future_volatility = features['Close'].pct_change().rolling(5).std().shift(-5)
        features['target_high_volatility'] = (future_volatility > future_volatility.rolling(20).mean() * 1.5).astype(int)
        
        # Drawdown targets
        future_drawdown = (features['Close'].shift(-10) / features['Close'].shift(-1).rolling(10).max() - 1).clip(upper=0)
        features['target_significant_drawdown'] = (future_drawdown < -0.05).astype(int)
        
        return features
    
    # Helper methods
    def _fetch_intermarket_data(self, symbols: Dict[str, str], start_date, end_date):
        """Fetch and cache intermarket data"""
        for name, symbol in symbols.items():
            if name not in self.intermarket_cache:
                try:
                    ticker = yf.Ticker(symbol)
                    data = ticker.history(start=start_date, end=end_date)
                    self.intermarket_cache[name] = data
                except:
                    self.intermarket_cache[name] = pd.DataFrame()
    
    def _calculate_trend_strength(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate trend strength using linear regression"""
        def trend_strength(window):
            if len(window) < period:
                return np.nan
            x = np.arange(len(window))
            y = window.values
            slope, intercept = np.polyfit(x, y, 1)
            r_squared = np.corrcoef(x, y)[0, 1] ** 2
            return slope * r_squared
        
        return prices.rolling(period).apply(trend_strength)
    
    def _calculate_adx(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average Directional Index"""
        high = data['High']
        low = data['Low']
        close = data['Close']
        
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        
        # Directional movements
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        
        pos_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
        neg_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)
        
        pos_di = 100 * (pos_dm.rolling(period).mean() / atr)
        neg_di = 100 * (neg_dm.rolling(period).mean() / atr)
        
        dx = 100 * abs(pos_di - neg_di) / (pos_di + neg_di)
        adx = dx.rolling(period).mean()
        
        return adx
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = -delta.where(delta < 0, 0).rolling(period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_ultimate_oscillator(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Ultimate Oscillator"""
        high = data['High']
        low = data['Low']
        close = data['Close']
        
        bp = close - pd.concat([low, close.shift(1)], axis=1).min(axis=1)
        tr = pd.concat([
            high - low,
            abs(high - close.shift(1)),
            abs(low - close.shift(1))
        ], axis=1).max(axis=1)
        
        avg7 = bp.rolling(7).sum() / tr.rolling(7).sum()
        avg14 = bp.rolling(14).sum() / tr.rolling(14).sum()
        avg28 = bp.rolling(28).sum() / tr.rolling(28).sum()
        
        uo = 100 * ((4 * avg7) + (2 * avg14) + avg28) / 7
        return uo
    
    def _calculate_choppiness_index(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Choppiness Index"""
        high = data['High']
        low = data['Low']
        close = data['Close']
        
        tr = pd.concat([
            high - low,
            abs(high - close.shift(1)),
            abs(low - close.shift(1))
        ], axis=1).max(axis=1)
        
        atr_sum = tr.rolling(period).sum()
        high_low_range = high.rolling(period).max() - low.rolling(period).min()
        
        choppiness = 100 * np.log10(atr_sum / high_low_range) / np.log10(period)
        return choppiness
    
    def _is_options_expiration_week(self, dates: pd.DatetimeIndex) -> pd.Series:
        """Detect options expiration week (third Friday)"""
        result = pd.Series(0, index=dates)
        
        for i, date in enumerate(dates):
            # Find third Friday of the month
            first_day = date.replace(day=1)
            first_friday = first_day + pd.Timedelta(days=(4 - first_day.weekday()) % 7)
            third_friday = first_friday + pd.Timedelta(weeks=2)
            
            # Check if current week contains third Friday
            week_start = date - pd.Timedelta(days=date.weekday())
            week_end = week_start + pd.Timedelta(days=6)
            
            if week_start <= third_friday <= week_end:
                result.iloc[i] = 1
        
        return result
    
    def _days_to_next_holiday(self, dates: pd.DatetimeIndex) -> pd.Series:
        """Calculate days to next major market holiday (simplified)"""
        # Simplified - would use actual holiday calendar in production
        result = pd.Series(30, index=dates)  # Default 30 days
        
        for i, date in enumerate(dates):
            # Check for major holidays (simplified)
            if date.month == 12 and date.day >= 20:  # Christmas week
                result.iloc[i] = 25 - date.day
            elif date.month == 7 and date.day <= 4:  # July 4th
                result.iloc[i] = 4 - date.day
            elif date.month == 11 and date.day >= 20 and date.day <= 27:  # Thanksgiving
                result.iloc[i] = 27 - date.day
        
        return result
    
    def _days_from_last_holiday(self, dates: pd.DatetimeIndex) -> pd.Series:
        """Calculate days from last major market holiday (simplified)"""
        # Inverse of days to next holiday logic
        result = pd.Series(30, index=dates)  # Default 30 days
        
        for i, date in enumerate(dates):
            if date.month == 1 and date.day <= 10:  # After New Year
                result.iloc[i] = date.day
            elif date.month == 7 and date.day >= 5 and date.day <= 15:  # After July 4th
                result.iloc[i] = date.day - 4
            elif date.month == 12 and date.day >= 26:  # After Christmas
                result.iloc[i] = date.day - 25
        
        return result
    
    # Pattern detection methods (simplified implementations)
    def _detect_doji(self, data: pd.DataFrame) -> pd.Series:
        """Detect doji candlestick pattern"""
        body = abs(data['Close'] - data['Open'])
        range_ = data['High'] - data['Low']
        return (body / range_ < 0.1).astype(int)
    
    def _detect_hammer(self, data: pd.DataFrame) -> pd.Series:
        """Detect hammer pattern"""
        body = abs(data['Close'] - data['Open'])
        lower_shadow = pd.concat([data['Open'], data['Close']], axis=1).min(axis=1) - data['Low']
        upper_shadow = data['High'] - pd.concat([data['Open'], data['Close']], axis=1).max(axis=1)
        
        return ((lower_shadow > body * 2) & (upper_shadow < body * 0.5)).astype(int)
    
    def _detect_shooting_star(self, data: pd.DataFrame) -> pd.Series:
        """Detect shooting star pattern"""
        body = abs(data['Close'] - data['Open'])
        lower_shadow = pd.concat([data['Open'], data['Close']], axis=1).min(axis=1) - data['Low']
        upper_shadow = data['High'] - pd.concat([data['Open'], data['Close']], axis=1).max(axis=1)
        
        return ((upper_shadow > body * 2) & (lower_shadow < body * 0.5)).astype(int)
    
    def _detect_engulfing(self, data: pd.DataFrame) -> pd.Series:
        """Detect engulfing pattern"""
        curr_body = data['Close'] - data['Open']
        prev_body = data['Close'].shift(1) - data['Open'].shift(1)
        
        bullish_engulfing = (
            (prev_body < 0) & 
            (curr_body > 0) & 
            (data['Open'] < data['Close'].shift(1)) & 
            (data['Close'] > data['Open'].shift(1))
        )
        
        bearish_engulfing = (
            (prev_body > 0) & 
            (curr_body < 0) & 
            (data['Open'] > data['Close'].shift(1)) & 
            (data['Close'] < data['Open'].shift(1))
        )
        
        return (bullish_engulfing.astype(int) - bearish_engulfing.astype(int))
    
    def _detect_head_shoulders(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """Detect potential head and shoulders pattern"""
        # Simplified detection
        result = pd.Series(0, index=data.index)
        
        for i in range(window * 2, len(data)):
            window_data = data['High'].iloc[i-window*2:i]
            
            # Find three peaks
            peaks = []
            for j in range(1, len(window_data) - 1):
                if window_data.iloc[j] > window_data.iloc[j-1] and window_data.iloc[j] > window_data.iloc[j+1]:
                    peaks.append((j, window_data.iloc[j]))
            
            # Check if pattern resembles head and shoulders
            if len(peaks) >= 3:
                # Middle peak should be highest
                peaks.sort(key=lambda x: x[1], reverse=True)
                if peaks[0][0] > peaks[1][0] and peaks[0][0] < peaks[2][0]:
                    result.iloc[i] = 1
        
        return result
    
    def _detect_triangle(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """Detect triangle consolidation pattern"""
        high_slope = (data['High'].rolling(window).max() - data['High'].rolling(window).max().shift(window)) / window
        low_slope = (data['Low'].rolling(window).min() - data['Low'].rolling(window).min().shift(window)) / window
        
        # Converging lines indicate triangle
        return ((high_slope < 0) & (low_slope > 0)).astype(int)
    
    def _near_resistance(self, data: pd.DataFrame, threshold: float = 0.02) -> pd.Series:
        """Check if price is near resistance"""
        resistance = data['High'].rolling(20).max()
        return ((resistance - data['Close']) / data['Close'] < threshold).astype(int)
    
    def _near_support(self, data: pd.DataFrame, threshold: float = 0.02) -> pd.Series:
        """Check if price is near support"""
        support = data['Low'].rolling(20).min()
        return ((data['Close'] - support) / data['Close'] < threshold).astype(int)
    
    def _detect_consolidation(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """Detect consolidation periods"""
        atr = self._calculate_atr(data)
        avg_atr = atr.rolling(window * 2).mean()
        return (atr < avg_atr * 0.7).astype(int)
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ATR"""
        tr = pd.concat([
            data['High'] - data['Low'],
            abs(data['High'] - data['Close'].shift(1)),
            abs(data['Low'] - data['Close'].shift(1))
        ], axis=1).max(axis=1)
        
        return tr.rolling(period).mean()
    
    def _detect_trendline_break(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """Detect trendline breaks"""
        # Simplified - calculate linear regression line and check for breaks
        def check_break(prices):
            if len(prices) < window:
                return 0
            x = np.arange(len(prices))
            slope, intercept = np.polyfit(x, prices.values, 1)
            trendline = slope * x + intercept
            
            # Check if last price breaks the trendline
            if slope > 0 and prices.iloc[-1] < trendline[-1] * 0.98:  # Break below uptrend
                return -1
            elif slope < 0 and prices.iloc[-1] > trendline[-1] * 1.02:  # Break above downtrend
                return 1
            return 0
        
        return data['Close'].rolling(window).apply(check_break)
    
    def get_feature_importance_analysis(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze feature statistics and distributions
        """
        analysis = pd.DataFrame({
            'mean': features.mean(),
            'std': features.std(),
            'min': features.min(),
            'max': features.max(),
            'skew': features.skew(),
            'kurtosis': features.kurtosis(),
            'missing_pct': (features.isna().sum() / len(features)) * 100
        })
        
        return analysis.sort_values('std', ascending=False)


def main():
    """Test the enhanced feature engineering"""
    print("🧪 ENHANCED FEATURE ENGINEERING DEMO")
    print("="*80)
    
    # Test with sample data
    symbol = 'AAPL'
    ticker = yf.Ticker(symbol)
    data = ticker.history(period='1y')
    
    # Initialize feature engineer
    engineer = EnhancedFeatureEngineer()
    
    # Engineer features
    features = engineer.engineer_features(data, symbol)
    
    # Display feature groups
    print("\n📊 FEATURE GROUPS CREATED:")
    print("-"*80)
    
    feature_groups = {}
    for col in features.columns:
        group = col.split('_')[0]
        if group not in feature_groups:
            feature_groups[group] = []
        feature_groups[group].append(col)
    
    for group, cols in feature_groups.items():
        print(f"\n{group}: {len(cols)} features")
        print(f"  Sample features: {', '.join(cols[:5])}")
    
    # Feature importance analysis
    print("\n📈 FEATURE STATISTICS:")
    print("-"*80)
    
    importance = engineer.get_feature_importance_analysis(features)
    print(importance.head(10))
    
    # Save features
    import os
    os.makedirs('outputs/ml_features', exist_ok=True)
    
    features.to_csv(f'outputs/ml_features/{symbol}_enhanced_features.csv')
    importance.to_csv(f'outputs/ml_features/{symbol}_feature_analysis.csv')
    
    print(f"\n💾 Features saved to outputs/ml_features/")
    print(f"📊 Total features: {len(features.columns)}")
    print(f"📈 Total samples: {len(features)}")


if __name__ == "__main__":
    main()