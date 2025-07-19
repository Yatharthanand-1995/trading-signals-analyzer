#!/usr/bin/env python3
"""
Historical Data Analyzer
Analyzes historical patterns and generates insights from 3 years of data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json
from typing import Dict, List, Tuple, Optional
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    
from scipy import stats

class HistoricalAnalyzer:
    def __init__(self, data_dir: str = "historical_data"):
        """Initialize the historical analyzer."""
        self.data_dir = data_dir
        self.data = {}
        
    def load_stock_data(self, symbol: str) -> pd.DataFrame:
        """Load historical data for a stock."""
        filename = os.path.join(self.data_dir, f"{symbol}_historical_data.csv")
        if os.path.exists(filename):
            df = pd.read_csv(filename, index_col=0)
            # Handle timezone-aware datetime
            df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
            self.data[symbol] = df
            return df
        else:
            print(f"No data found for {symbol}")
            return pd.DataFrame()
    
    def analyze_price_patterns(self, symbol: str) -> Dict:
        """Analyze historical price patterns."""
        df = self.data.get(symbol, self.load_stock_data(symbol))
        if df.empty:
            return {}
        
        analysis = {
            'symbol': symbol,
            'current_price': df['Close'].iloc[-1],
            'patterns': {}
        }
        
        # 1. Support and Resistance Levels (Historical)
        analysis['patterns']['historical_support_resistance'] = self.find_support_resistance(df)
        
        # 2. Seasonal Patterns
        analysis['patterns']['seasonal'] = self.analyze_seasonality(df)
        
        # 3. Price Distribution
        analysis['patterns']['price_distribution'] = self.analyze_price_distribution(df)
        
        # 4. Trend Analysis
        analysis['patterns']['trend'] = self.analyze_trends(df)
        
        # 5. Volatility Patterns
        analysis['patterns']['volatility'] = self.analyze_volatility(df)
        
        # 6. Technical Pattern Recognition
        analysis['patterns']['technical'] = self.find_technical_patterns(df)
        
        return analysis
    
    def find_support_resistance(self, df: pd.DataFrame) -> Dict:
        """Find historical support and resistance levels."""
        # Use price pivots
        highs = df['High'].rolling(window=20, center=True).max() == df['High']
        lows = df['Low'].rolling(window=20, center=True).min() == df['Low']
        
        resistance_levels = df[highs]['High'].values
        support_levels = df[lows]['Low'].values
        
        # Cluster nearby levels
        def cluster_levels(levels, threshold=0.02):
            if len(levels) == 0:
                return []
            
            sorted_levels = np.sort(levels)
            clusters = []
            current_cluster = [sorted_levels[0]]
            
            for level in sorted_levels[1:]:
                if (level - current_cluster[-1]) / current_cluster[-1] < threshold:
                    current_cluster.append(level)
                else:
                    clusters.append(np.mean(current_cluster))
                    current_cluster = [level]
            
            clusters.append(np.mean(current_cluster))
            return clusters
        
        resistance_clusters = cluster_levels(resistance_levels)
        support_clusters = cluster_levels(support_levels)
        
        # Find most tested levels
        current_price = df['Close'].iloc[-1]
        
        return {
            'major_resistance': [r for r in resistance_clusters if r > current_price][:3],
            'major_support': [s for s in support_clusters if s < current_price][-3:],
            'all_resistance': resistance_clusters,
            'all_support': support_clusters
        }
    
    def analyze_seasonality(self, df: pd.DataFrame) -> Dict:
        """Analyze seasonal patterns in the data."""
        df['Month'] = df.index.month
        df['Quarter'] = df.index.quarter
        df['DayOfWeek'] = df.index.dayofweek
        df['MonthReturn'] = df['Close'].pct_change()
        
        # Monthly performance
        monthly_returns = df.groupby('Month')['MonthReturn'].agg(['mean', 'std', 'count'])
        monthly_returns['sharpe'] = monthly_returns['mean'] / monthly_returns['std']
        
        # Best and worst months
        best_months = monthly_returns.nlargest(3, 'mean')
        worst_months = monthly_returns.nsmallest(3, 'mean')
        
        # Day of week analysis
        dow_returns = df.groupby('DayOfWeek')['MonthReturn'].mean()
        
        return {
            'best_months': best_months.to_dict(),
            'worst_months': worst_months.to_dict(),
            'day_of_week_returns': dow_returns.to_dict(),
            'current_month_historical': monthly_returns.loc[datetime.now().month].to_dict()
        }
    
    def analyze_price_distribution(self, df: pd.DataFrame) -> Dict:
        """Analyze price distribution and probabilities."""
        returns = df['Daily_Return'].dropna()
        
        # Calculate percentiles
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        price_percentiles = {}
        for p in percentiles:
            price_percentiles[f'p{p}'] = np.percentile(df['Close'], p)
        
        # Current price percentile
        current_price = df['Close'].iloc[-1]
        current_percentile = stats.percentileofscore(df['Close'], current_price)
        
        # Return distribution
        return {
            'price_percentiles': price_percentiles,
            'current_price_percentile': current_percentile,
            'return_stats': {
                'mean': returns.mean(),
                'std': returns.std(),
                'skew': returns.skew(),
                'kurtosis': returns.kurtosis(),
                'var_95': np.percentile(returns, 5),  # Value at Risk
                'cvar_95': returns[returns <= np.percentile(returns, 5)].mean()  # Conditional VaR
            }
        }
    
    def analyze_trends(self, df: pd.DataFrame) -> Dict:
        """Analyze long-term trends."""
        # Calculate trend strength over different periods
        periods = [20, 50, 100, 200]
        trends = {}
        
        for period in periods:
            if len(df) >= period:
                # Linear regression
                x = np.arange(period)
                y = df['Close'].iloc[-period:].values
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                
                # Trend strength (R-squared)
                trend_strength = r_value ** 2
                
                # Annualized trend
                daily_return = slope / y[0]
                annualized_trend = (1 + daily_return) ** 252 - 1
                
                trends[f'trend_{period}d'] = {
                    'slope': slope,
                    'strength': trend_strength,
                    'annualized_return': annualized_trend,
                    'direction': 'up' if slope > 0 else 'down'
                }
        
        return trends
    
    def analyze_volatility(self, df: pd.DataFrame) -> Dict:
        """Analyze volatility patterns."""
        # Rolling volatility
        df['Volatility_20'] = df['Daily_Return'].rolling(20).std() * np.sqrt(252)
        df['Volatility_60'] = df['Daily_Return'].rolling(60).std() * np.sqrt(252)
        
        # Current vs historical volatility
        current_vol = df['Volatility_20'].iloc[-1]
        vol_percentile = stats.percentileofscore(df['Volatility_20'].dropna(), current_vol)
        
        # Volatility regime
        vol_mean = df['Volatility_20'].mean()
        vol_std = df['Volatility_20'].std()
        
        if current_vol < vol_mean - vol_std:
            regime = 'low'
        elif current_vol > vol_mean + vol_std:
            regime = 'high'
        else:
            regime = 'normal'
        
        return {
            'current_volatility': current_vol,
            'volatility_percentile': vol_percentile,
            'volatility_regime': regime,
            'historical_avg_volatility': vol_mean,
            'volatility_trend': 'increasing' if df['Volatility_20'].iloc[-5:].mean() > df['Volatility_20'].iloc[-20:-5].mean() else 'decreasing'
        }
    
    def find_technical_patterns(self, df: pd.DataFrame) -> Dict:
        """Find technical chart patterns."""
        patterns = {}
        
        # 1. Golden/Death Cross
        if 'SMA_50' in df.columns and 'SMA_200' in df.columns:
            recent_cross = self.find_ma_crossovers(df)
            patterns['ma_crossovers'] = recent_cross
        
        # 2. RSI Divergence
        if 'RSI' in df.columns:
            patterns['rsi_divergence'] = self.find_rsi_divergence(df)
        
        # 3. Bollinger Band Squeeze
        if 'BB_Upper' in df.columns:
            patterns['bb_squeeze'] = self.find_bb_squeeze(df)
        
        # 4. Price breakouts
        patterns['breakouts'] = self.find_breakouts(df)
        
        return patterns
    
    def find_ma_crossovers(self, df: pd.DataFrame) -> Dict:
        """Find moving average crossovers."""
        # Golden cross: 50 > 200
        # Death cross: 50 < 200
        
        df['Signal'] = 0
        df.loc[df['SMA_50'] > df['SMA_200'], 'Signal'] = 1
        df.loc[df['SMA_50'] < df['SMA_200'], 'Signal'] = -1
        
        # Find recent crossovers
        df['SignalChange'] = df['Signal'].diff()
        
        recent_golden = df[df['SignalChange'] == 2].tail(1)
        recent_death = df[df['SignalChange'] == -2].tail(1)
        
        result = {
            'current_position': 'above' if df['Signal'].iloc[-1] == 1 else 'below',
            'last_golden_cross': recent_golden.index[0].strftime('%Y-%m-%d') if not recent_golden.empty else None,
            'last_death_cross': recent_death.index[0].strftime('%Y-%m-%d') if not recent_death.empty else None
        }
        
        return result
    
    def find_rsi_divergence(self, df: pd.DataFrame, lookback: int = 50) -> Dict:
        """Find RSI divergences."""
        recent_df = df.tail(lookback)
        
        # Find price peaks and troughs
        price_peaks = recent_df['High'].rolling(window=5, center=True).max() == recent_df['High']
        price_troughs = recent_df['Low'].rolling(window=5, center=True).min() == recent_df['Low']
        
        # Find RSI peaks and troughs
        rsi_peaks = recent_df['RSI'].rolling(window=5, center=True).max() == recent_df['RSI']
        rsi_troughs = recent_df['RSI'].rolling(window=5, center=True).min() == recent_df['RSI']
        
        divergence = {
            'bullish_divergence': False,
            'bearish_divergence': False
        }
        
        # Check for divergences in last 20 days
        if len(recent_df) > 20:
            # Bullish divergence: price makes lower low, RSI makes higher low
            price_lows = recent_df[price_troughs]['Low'].tail(2)
            rsi_lows = recent_df[rsi_troughs]['RSI'].tail(2)
            
            if len(price_lows) >= 2 and len(rsi_lows) >= 2:
                if price_lows.iloc[-1] < price_lows.iloc[-2] and rsi_lows.iloc[-1] > rsi_lows.iloc[-2]:
                    divergence['bullish_divergence'] = True
            
            # Bearish divergence: price makes higher high, RSI makes lower high
            price_highs = recent_df[price_peaks]['High'].tail(2)
            rsi_highs = recent_df[rsi_peaks]['RSI'].tail(2)
            
            if len(price_highs) >= 2 and len(rsi_highs) >= 2:
                if price_highs.iloc[-1] > price_highs.iloc[-2] and rsi_highs.iloc[-1] < rsi_highs.iloc[-2]:
                    divergence['bearish_divergence'] = True
        
        return divergence
    
    def find_bb_squeeze(self, df: pd.DataFrame) -> Dict:
        """Find Bollinger Band squeeze patterns."""
        # BB width
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        
        # Find squeeze (low volatility)
        bb_width_percentile = stats.percentileofscore(df['BB_Width'].dropna(), df['BB_Width'].iloc[-1])
        
        squeeze = {
            'in_squeeze': bb_width_percentile < 20,
            'bb_width_percentile': bb_width_percentile,
            'potential_breakout': bb_width_percentile < 10
        }
        
        return squeeze
    
    def find_breakouts(self, df: pd.DataFrame, lookback: int = 20) -> Dict:
        """Find price breakouts."""
        # Recent high/low
        recent_high = df['High'].iloc[-lookback:].max()
        recent_low = df['Low'].iloc[-lookback:].min()
        
        # 52-week high/low
        if len(df) >= 252:
            yearly_high = df['High'].iloc[-252:].max()
            yearly_low = df['Low'].iloc[-252:].min()
        else:
            yearly_high = df['High'].max()
            yearly_low = df['Low'].min()
        
        current_price = df['Close'].iloc[-1]
        
        breakouts = {
            'near_resistance': (recent_high - current_price) / current_price < 0.02,
            'near_support': (current_price - recent_low) / current_price < 0.02,
            'at_52w_high': (yearly_high - current_price) / current_price < 0.01,
            'at_52w_low': (current_price - yearly_low) / current_price < 0.01,
            'breakout_potential': 'high' if (recent_high - current_price) / current_price < 0.01 else 'low'
        }
        
        return breakouts
    
    def generate_trading_insights(self, symbol: str) -> Dict:
        """Generate actionable trading insights from historical analysis."""
        analysis = self.analyze_price_patterns(symbol)
        if not analysis:
            return {}
        
        insights = {
            'symbol': symbol,
            'current_price': analysis['current_price'],
            'recommendations': []
        }
        
        patterns = analysis['patterns']
        
        # 1. Support/Resistance insights
        if patterns.get('historical_support_resistance'):
            sr = patterns['historical_support_resistance']
            if sr['major_support']:
                nearest_support = sr['major_support'][-1]
                support_distance = (analysis['current_price'] - nearest_support) / analysis['current_price']
                if support_distance < 0.03:
                    insights['recommendations'].append({
                        'type': 'support_nearby',
                        'message': f"Price near major support at ${nearest_support:.2f} ({support_distance*100:.1f}% away)",
                        'action': 'consider_buy'
                    })
        
        # 2. Seasonal insights
        if patterns.get('seasonal'):
            current_month = datetime.now().month
            month_stats = patterns['seasonal']['current_month_historical']
            if month_stats['mean'] > 0.02:  # 2% average return
                insights['recommendations'].append({
                    'type': 'seasonal_positive',
                    'message': f"Historically strong month with {month_stats['mean']*100:.1f}% average return",
                    'action': 'seasonal_buy'
                })
        
        # 3. Volatility insights
        if patterns.get('volatility'):
            vol = patterns['volatility']
            if vol['volatility_regime'] == 'low' and vol['volatility_percentile'] < 20:
                insights['recommendations'].append({
                    'type': 'low_volatility',
                    'message': "Volatility at historical lows - potential for breakout",
                    'action': 'prepare_for_move'
                })
        
        # 4. Technical pattern insights
        if patterns.get('technical'):
            tech = patterns['technical']
            
            # MA crossover
            if tech.get('ma_crossovers'):
                if tech['ma_crossovers']['current_position'] == 'above':
                    insights['recommendations'].append({
                        'type': 'golden_cross_active',
                        'message': "50-day MA above 200-day MA (bullish)",
                        'action': 'trend_following_buy'
                    })
            
            # RSI divergence
            if tech.get('rsi_divergence'):
                if tech['rsi_divergence']['bullish_divergence']:
                    insights['recommendations'].append({
                        'type': 'bullish_divergence',
                        'message': "Bullish RSI divergence detected",
                        'action': 'divergence_buy'
                    })
            
            # Breakout potential
            if tech.get('breakouts'):
                if tech['breakouts']['breakout_potential'] == 'high':
                    insights['recommendations'].append({
                        'type': 'breakout_imminent',
                        'message': "Price near resistance with high breakout potential",
                        'action': 'breakout_watch'
                    })
        
        # 5. Risk assessment
        risk_score = self.calculate_risk_score(analysis)
        insights['risk_score'] = risk_score
        insights['risk_level'] = 'high' if risk_score > 70 else 'medium' if risk_score > 40 else 'low'
        
        return insights
    
    def calculate_risk_score(self, analysis: Dict) -> float:
        """Calculate a risk score from 0-100."""
        risk_score = 0
        patterns = analysis['patterns']
        
        # Volatility risk
        if patterns.get('volatility'):
            vol_percentile = patterns['volatility']['volatility_percentile']
            risk_score += vol_percentile * 0.3  # 30% weight
        
        # Price distribution risk
        if patterns.get('price_distribution'):
            price_percentile = patterns['price_distribution']['current_price_percentile']
            # High percentile = overbought risk
            if price_percentile > 80:
                risk_score += (price_percentile - 50) * 0.3
            # Low percentile = oversold opportunity (lower risk)
            elif price_percentile < 20:
                risk_score += (50 - price_percentile) * 0.1
        
        # Trend risk
        if patterns.get('trend'):
            # Negative trends increase risk
            for period, trend_data in patterns['trend'].items():
                if trend_data['direction'] == 'down':
                    risk_score += 10
        
        return min(risk_score, 100)
    
    def create_analysis_report(self, symbols: List[str]):
        """Create a comprehensive analysis report for all symbols."""
        report = {
            'report_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'symbols_analyzed': symbols,
            'analyses': {}
        }
        
        for symbol in symbols:
            print(f"\nAnalyzing {symbol}...")
            
            # Load data
            df = self.load_stock_data(symbol)
            if df.empty:
                continue
            
            # Perform analysis
            analysis = self.analyze_price_patterns(symbol)
            insights = self.generate_trading_insights(symbol)
            
            report['analyses'][symbol] = {
                'analysis': analysis,
                'insights': insights
            }
        
        # Save report
        report_filename = os.path.join(self.data_dir, 'historical_analysis_report.json')
        with open(report_filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nSaved analysis report to {report_filename}")
        
        return report


def main():
    """Main function to run historical analysis."""
    symbols = ['AAPL', 'GOOGL', 'TSLA', 'MSFT', 'UNH']
    
    analyzer = HistoricalAnalyzer()
    
    print("🔍 Starting Historical Analysis")
    print("="*50)
    
    # Create comprehensive report
    report = analyzer.create_analysis_report(symbols)
    
    # Print summary of insights
    print("\n📊 TRADING INSIGHTS SUMMARY")
    print("="*50)
    
    for symbol, data in report['analyses'].items():
        insights = data['insights']
        print(f"\n{symbol}:")
        print(f"Current Price: ${insights['current_price']:.2f}")
        print(f"Risk Level: {insights['risk_level']} ({insights['risk_score']:.0f}/100)")
        
        if insights['recommendations']:
            print("Recommendations:")
            for rec in insights['recommendations']:
                print(f"  • {rec['message']}")
                print(f"    Action: {rec['action']}")
        else:
            print("  No specific recommendations at this time")
    
    print("\n✅ Historical analysis complete!")


if __name__ == "__main__":
    main()