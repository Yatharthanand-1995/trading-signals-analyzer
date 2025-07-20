#!/usr/bin/env python3
"""
Market Microstructure Feature Integration
Advanced features capturing market dynamics and order flow
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class MarketMicrostructureAnalyzer:
    """
    Extracts advanced market microstructure features:
    - Order flow imbalance
    - Price impact measures
    - Liquidity indicators
    - Market depth proxies
    - Execution quality metrics
    - Information asymmetry indicators
    """
    
    def __init__(self):
        self.feature_groups = {
            'order_flow': self._calculate_order_flow_features,
            'liquidity': self._calculate_liquidity_features,
            'price_impact': self._calculate_price_impact_features,
            'market_quality': self._calculate_market_quality_features,
            'information_flow': self._calculate_information_flow_features,
            'execution_metrics': self._calculate_execution_metrics
        }
    
    def calculate_all_features(self, data: pd.DataFrame, 
                             intraday_data: pd.DataFrame = None) -> pd.DataFrame:
        """
        Calculate all microstructure features
        """
        features = pd.DataFrame(index=data.index)
        
        # Calculate each feature group
        for group_name, feature_func in self.feature_groups.items():
            try:
                if group_name in ['order_flow', 'execution_metrics'] and intraday_data is not None:
                    group_features = feature_func(data, intraday_data)
                else:
                    group_features = feature_func(data)
                
                # Add group features
                for col in group_features.columns:
                    features[f'{group_name}_{col}'] = group_features[col]
            except Exception as e:
                print(f"Warning: Failed to calculate {group_name} features: {str(e)}")
        
        return features
    
    def _calculate_order_flow_features(self, data: pd.DataFrame, 
                                     intraday_data: pd.DataFrame = None) -> pd.DataFrame:
        """
        Calculate order flow imbalance and toxicity measures
        """
        features = pd.DataFrame(index=data.index)
        
        # Volume-weighted average price (VWAP)
        features['vwap'] = (data['Close'] * data['Volume']).cumsum() / data['Volume'].cumsum()
        features['vwap_distance'] = (data['Close'] - features['vwap']) / data['Close']
        
        # Order flow imbalance proxy
        # Using price changes and volume as proxy
        price_change = data['Close'].pct_change()
        features['order_flow_imbalance'] = np.where(
            price_change > 0,
            data['Volume'],  # Buy pressure
            -data['Volume']  # Sell pressure
        ).rolling(10).sum() / data['Volume'].rolling(10).sum()
        
        # Kyle's Lambda (price impact coefficient)
        # Approximation using returns and volume
        returns = data['Close'].pct_change()
        signed_volume = data['Volume'] * np.sign(returns)
        
        features['kyle_lambda'] = returns.rolling(20).apply(
            lambda x: np.abs(x).mean() if len(x) > 0 else 0
        ) / (signed_volume.rolling(20).std() + 1e-8)
        
        # Trade intensity
        features['trade_intensity'] = data['Volume'] / data['Volume'].rolling(20).mean()
        
        # Buy/Sell pressure using High/Low
        features['buy_pressure'] = (data['Close'] - data['Low']) / (data['High'] - data['Low'] + 1e-8)
        features['sell_pressure'] = (data['High'] - data['Close']) / (data['High'] - data['Low'] + 1e-8)
        
        # Order flow persistence
        features['flow_persistence'] = features['order_flow_imbalance'].rolling(5).apply(
            lambda x: np.corrcoef(x, np.arange(len(x)))[0, 1] if len(x) > 1 else 0
        )
        
        # Toxic order flow (large trades moving price)
        volume_percentile = data['Volume'].rolling(50).apply(lambda x: np.percentile(x, 90))
        large_trades = data['Volume'] > volume_percentile
        features['toxic_flow'] = (large_trades * np.abs(returns)).rolling(10).mean()
        
        return features
    
    def _calculate_liquidity_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate market liquidity indicators
        """
        features = pd.DataFrame(index=data.index)
        
        # Amihud illiquidity measure
        returns = data['Close'].pct_change()
        features['amihud_illiquidity'] = (np.abs(returns) / (data['Volume'] * data['Close'] + 1e-8)).rolling(20).mean()
        
        # Roll's implied spread estimator
        # Based on serial covariance of returns
        features['roll_spread'] = returns.rolling(20).apply(
            lambda x: 2 * np.sqrt(-np.cov(x[:-1], x[1:])[0, 1]) if len(x) > 1 and np.cov(x[:-1], x[1:])[0, 1] < 0 else 0
        )
        
        # Effective spread proxy
        features['effective_spread'] = 2 * np.abs(data['Close'] - data['Close'].shift(1)) / data['Close']
        features['avg_effective_spread'] = features['effective_spread'].rolling(20).mean()
        
        # Liquidity ratio (volume to volatility)
        volatility = returns.rolling(20).std()
        features['liquidity_ratio'] = np.log1p(data['Volume']) / (volatility + 1e-8)
        
        # Market depth proxy (using volume and price range)
        price_range = (data['High'] - data['Low']) / data['Close']
        features['depth_proxy'] = data['Volume'] / (price_range + 1e-8)
        
        # Turnover rate
        avg_volume = data['Volume'].rolling(50).mean()
        features['turnover_rate'] = data['Volume'] / avg_volume
        
        # Price resilience (mean reversion speed)
        features['price_resilience'] = returns.rolling(20).apply(
            lambda x: -np.corrcoef(x[:-1], x[1:])[0, 1] if len(x) > 1 else 0
        )
        
        return features
    
    def _calculate_price_impact_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate price impact and market impact measures
        """
        features = pd.DataFrame(index=data.index)
        
        returns = data['Close'].pct_change()
        
        # Temporary price impact
        # Correlation between volume and subsequent return reversal
        features['temp_price_impact'] = data['Volume'].rolling(20).apply(
            lambda v, r=returns: np.corrcoef(v[:-1], -r[1:])[0, 1] if len(v) > 1 else 0
        )
        
        # Permanent price impact
        # Correlation between signed volume and returns
        signed_volume = data['Volume'] * np.sign(returns)
        features['perm_price_impact'] = signed_volume.rolling(20).apply(
            lambda v, r=returns: np.corrcoef(v, r[-len(v):])[0, 1] if len(v) > 1 else 0
        )
        
        # Volume-synchronized probability of informed trading (VPIN)
        # Simplified version
        volume_buckets = data['Volume'].rolling(10).sum()
        buy_volume = data['Volume'] * features.get('buy_pressure', 0.5)
        sell_volume = data['Volume'] * features.get('sell_pressure', 0.5)
        
        features['vpin'] = np.abs(buy_volume.rolling(10).sum() - sell_volume.rolling(10).sum()) / volume_buckets
        
        # Market impact asymmetry
        pos_returns = returns[returns > 0]
        neg_returns = returns[returns < 0]
        pos_volume = data['Volume'][returns > 0]
        neg_volume = data['Volume'][returns < 0]
        
        features['impact_asymmetry'] = (
            pos_returns.rolling(20).mean() / pos_volume.rolling(20).mean() - 
            np.abs(neg_returns.rolling(20).mean()) / neg_volume.rolling(20).mean()
        )
        
        # Price discovery metric
        # How much of the daily move happens early
        daily_return = data['Close'] / data['Open'] - 1
        first_hour_return = (data['High'].rolling(3).max() - data['Open']) / data['Open']
        features['price_discovery'] = first_hour_return / (daily_return + 1e-8)
        
        return features
    
    def _calculate_market_quality_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate market quality and efficiency measures
        """
        features = pd.DataFrame(index=data.index)
        
        returns = data['Close'].pct_change()
        
        # Variance ratio test for market efficiency
        def variance_ratio(returns, k):
            if len(returns) < k:
                return 1
            var_1 = returns.var()
            var_k = returns.rolling(k).sum().var() / k
            return var_k / var_1 if var_1 > 0 else 1
        
        features['variance_ratio_5'] = returns.rolling(50).apply(lambda x: variance_ratio(x, 5))
        features['variance_ratio_10'] = returns.rolling(100).apply(lambda x: variance_ratio(x, 10))
        
        # Hurst exponent (persistence measure)
        def hurst_exponent(ts, max_lag=20):
            if len(ts) < max_lag:
                return 0.5
            lags = range(2, min(max_lag, len(ts)//2))
            tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return poly[0] * 2
        
        features['hurst_exponent'] = data['Close'].rolling(50).apply(hurst_exponent)
        
        # Market efficiency coefficient
        # Based on predictability of returns
        features['efficiency_coef'] = 1 - np.abs(returns.rolling(20).apply(
            lambda x: np.corrcoef(x[:-1], x[1:])[0, 1] if len(x) > 1 else 0
        ))
        
        # Quote stuffing indicator (abnormal volume spikes)
        volume_zscore = (data['Volume'] - data['Volume'].rolling(20).mean()) / data['Volume'].rolling(20).std()
        features['quote_stuffing'] = (volume_zscore > 3).astype(int)
        
        # Market stress indicator
        # Combination of volatility, spread, and volume
        volatility = returns.rolling(20).std()
        spread_proxy = (data['High'] - data['Low']) / data['Close']
        volume_ratio = data['Volume'] / data['Volume'].rolling(20).mean()
        
        features['market_stress'] = (
            volatility / volatility.rolling(50).mean() + 
            spread_proxy / spread_proxy.rolling(50).mean() + 
            1 / (volume_ratio + 0.1)
        ) / 3
        
        return features
    
    def _calculate_information_flow_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate information flow and price discovery features
        """
        features = pd.DataFrame(index=data.index)
        
        returns = data['Close'].pct_change()
        
        # Information share (Hasbrouck)
        # Simplified version using return autocorrelation
        features['info_share'] = returns.rolling(20).apply(
            lambda x: 1 - np.abs(np.corrcoef(x[:-1], x[1:])[0, 1]) if len(x) > 1 else 0.5
        )
        
        # PIN (Probability of Informed Trading) proxy
        # Using volume imbalance and price moves
        volume_imbalance = np.abs(features.get('order_flow_imbalance', 0))
        price_move = np.abs(returns)
        features['pin_proxy'] = (volume_imbalance * price_move).rolling(20).mean()
        
        # Information arrival rate
        # Measured by return volatility jumps
        vol_changes = volatility.pct_change()
        features['info_arrival'] = (vol_changes > vol_changes.rolling(50).std() * 2).astype(int)
        
        # Price contribution
        # How much this period contributes to longer-term price moves
        future_returns = returns.shift(-10).rolling(10).sum()
        features['price_contribution'] = returns.rolling(20).apply(
            lambda x, f=future_returns: np.corrcoef(x, f[-len(x):])[0, 1] if len(x) > 1 else 0
        )
        
        # Information asymmetry
        # Difference between large and small trade impacts
        large_volume = data['Volume'] > data['Volume'].rolling(50).quantile(0.8)
        small_volume = data['Volume'] < data['Volume'].rolling(50).quantile(0.2)
        
        features['info_asymmetry'] = (
            (large_volume * np.abs(returns)).rolling(20).mean() - 
            (small_volume * np.abs(returns)).rolling(20).mean()
        )
        
        # News impact curve asymmetry
        pos_news = returns > returns.rolling(50).std()
        neg_news = returns < -returns.rolling(50).std()
        
        features['news_asymmetry'] = (
            (pos_news * data['Volume']).rolling(20).sum() / 
            (neg_news * data['Volume']).rolling(20).sum().replace(0, 1)
        )
        
        return features
    
    def _calculate_execution_metrics(self, data: pd.DataFrame, 
                                   intraday_data: pd.DataFrame = None) -> pd.DataFrame:
        """
        Calculate execution quality metrics
        """
        features = pd.DataFrame(index=data.index)
        
        # Implementation shortfall proxy
        # Difference between close and VWAP
        vwap = (data['Close'] * data['Volume']).rolling(20).sum() / data['Volume'].rolling(20).sum()
        features['impl_shortfall'] = (data['Close'] - vwap) / data['Close']
        
        # Arrival price impact
        # Price move from open to close
        features['arrival_impact'] = (data['Close'] - data['Open']) / data['Open']
        
        # Realized spread
        # 2 * (execution price - midpoint after execution)
        midpoint = (data['High'] + data['Low']) / 2
        features['realized_spread'] = 2 * np.abs(data['Close'] - midpoint.shift(-1)) / data['Close']
        
        # Execution probability
        # Probability of executing at favorable prices
        favorable_exec = (data['Close'] < midpoint) & (data['Close'].pct_change() > 0)
        features['exec_probability'] = favorable_exec.rolling(20).mean()
        
        # Market timing ability
        # Correlation between trades and future price moves
        volume_weighted_price = data['Close'] * data['Volume']
        future_return = data['Close'].shift(-5) / data['Close'] - 1
        
        features['timing_ability'] = volume_weighted_price.rolling(20).apply(
            lambda x, f=future_return: np.corrcoef(x, f[-len(x):])[0, 1] if len(x) > 1 else 0
        )
        
        # Opportunity cost
        # Best price vs execution price
        daily_best = data['Low']
        daily_worst = data['High']
        features['opportunity_cost'] = np.where(
            data['Close'].pct_change() > 0,
            (data['Close'] - daily_best) / data['Close'],
            (daily_worst - data['Close']) / data['Close']
        )
        
        return features
    
    def get_feature_importance_hints(self) -> Dict[str, str]:
        """
        Provide hints about feature importance and usage
        """
        return {
            'order_flow_imbalance': 'Key predictor of short-term price direction',
            'kyle_lambda': 'Measures price impact - higher values indicate less liquid markets',
            'amihud_illiquidity': 'Critical for position sizing - avoid large positions in illiquid stocks',
            'vpin': 'Probability of informed trading - high values suggest information asymmetry',
            'market_stress': 'Risk indicator - reduce positions when elevated',
            'info_share': 'Price discovery metric - higher values mean more informative prices',
            'timing_ability': 'Execution quality - positive values indicate good market timing'
        }


def demonstrate_microstructure_analysis():
    """
    Demonstrate market microstructure analysis
    """
    print("🎯 MARKET MICROSTRUCTURE ANALYSIS DEMO")
    print("="*80)
    
    # Fetch sample data
    symbol = 'AAPL'
    print(f"\nFetching data for {symbol}...")
    
    ticker = yf.Ticker(symbol)
    data = ticker.history(period='6mo', interval='1d')
    
    # Initialize analyzer
    analyzer = MarketMicrostructureAnalyzer()
    
    # Calculate features
    print("\n📈 Calculating microstructure features...")
    features = analyzer.calculate_all_features(data)
    
    # Display sample features
    print("\n📊 Sample Microstructure Features (last 5 days):")
    print("="*60)
    
    key_features = [
        'order_flow_order_flow_imbalance',
        'liquidity_amihud_illiquidity', 
        'price_impact_vpin',
        'market_quality_market_stress',
        'information_flow_info_share'
    ]
    
    for feature in key_features:
        if feature in features.columns:
            print(f"\n{feature}:")
            print(features[feature].tail(5).round(4))
    
    # Feature statistics
    print("\n📊 Feature Statistics:")
    print("="*60)
    
    feature_stats = features.describe().T
    print(feature_stats[['mean', 'std', 'min', 'max']].head(10).round(4))
    
    # Feature importance hints
    print("\n💡 Feature Importance Hints:")
    print("="*60)
    
    hints = analyzer.get_feature_importance_hints()
    for feature, hint in list(hints.items())[:5]:
        print(f"\n{feature}:")
        print(f"  → {hint}")
    
    # Current market conditions
    print("\n🎯 Current Market Conditions:")
    print("="*60)
    
    latest = features.iloc[-1]
    
    # Order flow analysis
    if 'order_flow_order_flow_imbalance' in latest:
        imbalance = latest['order_flow_order_flow_imbalance']
        if imbalance > 0.2:
            print("✔️ Strong BUY pressure detected")
        elif imbalance < -0.2:
            print("✖️ Strong SELL pressure detected")
        else:
            print("⚖️ Balanced order flow")
    
    # Liquidity analysis
    if 'liquidity_amihud_illiquidity' in latest:
        illiquidity = latest['liquidity_amihud_illiquidity']
        if illiquidity > features['liquidity_amihud_illiquidity'].quantile(0.8):
            print("⚠️ Low liquidity - use smaller position sizes")
        else:
            print("✔️ Good liquidity conditions")
    
    # Market stress
    if 'market_quality_market_stress' in latest:
        stress = latest['market_quality_market_stress']
        if stress > 1.5:
            print("🚨 High market stress - increased risk")
        elif stress < 0.8:
            print("✔️ Low market stress - favorable conditions")
        else:
            print("⚖️ Normal market conditions")
    
    # Save results
    import os
    os.makedirs('outputs/microstructure', exist_ok=True)
    
    features.tail(50).to_csv('outputs/microstructure/sample_features.csv')
    print("\n💾 Features saved to outputs/microstructure/sample_features.csv")
    
    print("\n✅ Market Microstructure Analysis Complete!")


if __name__ == "__main__":
    demonstrate_microstructure_analysis()