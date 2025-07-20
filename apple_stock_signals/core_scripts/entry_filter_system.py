#!/usr/bin/env python3
"""
Entry Filter System
Filters out low-probability trades using multiple criteria
Optimized for 2-15 day swing trading
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

class EntryFilterSystem:
    """
    Comprehensive filtering system to ensure only high-probability trades
    """
    
    def __init__(self):
        # Filter thresholds
        self.min_volume = 500000  # Minimum average daily volume
        self.min_price = 5.0      # Minimum stock price
        self.max_spread_pct = 0.5  # Maximum bid-ask spread %
        self.min_atr = 0.5        # Minimum ATR for volatility
        self.min_market_cap = 1e9  # $1B minimum market cap
        
        # Technical thresholds
        self.trend_alignment_threshold = 0.7  # 70% alignment required
        self.relative_strength_threshold = 1.05  # 5% outperformance
        
    def check_liquidity_filter(self, df, info=None):
        """Check if stock meets liquidity requirements"""
        avg_volume = df['Volume'].rolling(window=20).mean().iloc[-1]
        current_price = df['Close'].iloc[-1]
        
        # Calculate average dollar volume
        avg_dollar_volume = avg_volume * current_price
        
        liquidity_checks = {
            'avg_volume': avg_volume,
            'avg_dollar_volume': avg_dollar_volume,
            'current_price': current_price,
            'passes_volume': avg_volume >= self.min_volume,
            'passes_price': current_price >= self.min_price,
            'passes_dollar_volume': avg_dollar_volume >= 10_000_000,  # $10M daily
            'liquidity_score': 0
        }
        
        # Calculate liquidity score
        if liquidity_checks['passes_volume']:
            liquidity_checks['liquidity_score'] += 33
        if liquidity_checks['passes_price']:
            liquidity_checks['liquidity_score'] += 33
        if liquidity_checks['passes_dollar_volume']:
            liquidity_checks['liquidity_score'] += 34
            
        liquidity_checks['passes_filter'] = liquidity_checks['liquidity_score'] >= 66
        
        return liquidity_checks
    
    def check_trend_alignment(self, df, market_df=None):
        """Check if stock trend aligns with market trend"""
        # Stock trend
        sma_20 = df['Close'].rolling(window=20).mean()
        sma_50 = df['Close'].rolling(window=50).mean()
        sma_200 = df['Close'].rolling(window=200).mean() if len(df) >= 200 else sma_50
        
        current_price = df['Close'].iloc[-1]
        
        # Calculate trend scores
        trend_score = 0
        trend_checks = {}
        
        # Price vs moving averages
        if current_price > sma_20.iloc[-1]:
            trend_score += 25
            trend_checks['above_sma20'] = True
        else:
            trend_checks['above_sma20'] = False
            
        if current_price > sma_50.iloc[-1]:
            trend_score += 25
            trend_checks['above_sma50'] = True
        else:
            trend_checks['above_sma50'] = False
            
        if current_price > sma_200.iloc[-1]:
            trend_score += 25
            trend_checks['above_sma200'] = True
        else:
            trend_checks['above_sma200'] = False
        
        # Moving average alignment
        if sma_20.iloc[-1] > sma_50.iloc[-1]:
            trend_score += 25
            trend_checks['ma_aligned'] = True
        else:
            trend_checks['ma_aligned'] = False
        
        # Recent performance
        performance_5d = (current_price - df['Close'].iloc[-6]) / df['Close'].iloc[-6]
        performance_20d = (current_price - df['Close'].iloc[-21]) / df['Close'].iloc[-21] if len(df) > 21 else 0
        
        trend_checks.update({
            'trend_score': trend_score,
            'performance_5d': performance_5d * 100,
            'performance_20d': performance_20d * 100,
            'trend_strength': 'strong' if trend_score >= 75 else 'moderate' if trend_score >= 50 else 'weak',
            'passes_filter': trend_score >= 50  # At least moderate trend
        })
        
        return trend_checks
    
    def calculate_relative_strength(self, df, market_df, sector_df=None):
        """Calculate relative strength vs market and sector"""
        if len(df) < 20 or market_df is None or len(market_df) < 20:
            return {
                'rs_vs_market': 1.0,
                'rs_vs_sector': 1.0,
                'outperforming_market': False,
                'outperforming_sector': False,
                'passes_filter': False
            }
        
        # Calculate returns
        stock_return_20d = (df['Close'].iloc[-1] - df['Close'].iloc[-21]) / df['Close'].iloc[-21]
        market_return_20d = (market_df['Close'].iloc[-1] - market_df['Close'].iloc[-21]) / market_df['Close'].iloc[-21]
        
        rs_vs_market = (1 + stock_return_20d) / (1 + market_return_20d)
        
        rs_data = {
            'stock_return_20d': stock_return_20d * 100,
            'market_return_20d': market_return_20d * 100,
            'rs_vs_market': rs_vs_market,
            'outperforming_market': rs_vs_market >= self.relative_strength_threshold,
        }
        
        # Sector comparison if available
        if sector_df is not None and len(sector_df) >= 20:
            sector_return_20d = (sector_df['Close'].iloc[-1] - sector_df['Close'].iloc[-21]) / sector_df['Close'].iloc[-21]
            rs_vs_sector = (1 + stock_return_20d) / (1 + sector_return_20d)
            
            rs_data.update({
                'sector_return_20d': sector_return_20d * 100,
                'rs_vs_sector': rs_vs_sector,
                'outperforming_sector': rs_vs_sector >= 1.0
            })
        else:
            rs_data.update({
                'rs_vs_sector': 1.0,
                'outperforming_sector': True
            })
        
        rs_data['passes_filter'] = rs_data['outperforming_market']
        
        return rs_data
    
    def check_volatility_filter(self, df):
        """Check if volatility is within acceptable range"""
        # Calculate ATR
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=14).mean().iloc[-1]
        
        # Calculate volatility metrics
        atr_pct = (atr / close.iloc[-1]) * 100
        daily_returns = close.pct_change()
        volatility = daily_returns.std() * np.sqrt(252) * 100  # Annualized
        
        volatility_data = {
            'atr': atr,
            'atr_pct': atr_pct,
            'annualized_volatility': volatility,
            'volatility_suitable': 1.0 <= atr_pct <= 5.0,  # 1-5% daily range
            'passes_filter': atr >= self.min_atr and atr_pct >= 1.0
        }
        
        return volatility_data
    
    def check_momentum_filter(self, df):
        """Check momentum indicators for entry timing"""
        close = df['Close']
        
        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # Rate of Change
        roc_10 = ((close.iloc[-1] - close.iloc[-11]) / close.iloc[-11]) * 100 if len(close) > 11 else 0
        
        # Price position in range
        high_20 = df['High'].rolling(window=20).max().iloc[-1]
        low_20 = df['Low'].rolling(window=20).min().iloc[-1]
        price_position = (close.iloc[-1] - low_20) / (high_20 - low_20) if high_20 > low_20 else 0.5
        
        momentum_data = {
            'rsi': rsi.iloc[-1],
            'roc_10': roc_10,
            'price_position': price_position * 100,
            'rsi_status': 'oversold' if rsi.iloc[-1] < 30 else 'overbought' if rsi.iloc[-1] > 70 else 'neutral',
            'momentum_favorable': 30 <= rsi.iloc[-1] <= 70 and roc_10 > -10,
            'passes_filter': 25 <= rsi.iloc[-1] <= 75  # Not extreme
        }
        
        return momentum_data
    
    def check_pattern_filter(self, df):
        """Check for favorable price patterns"""
        patterns = []
        
        # Check for higher lows (uptrend)
        lows = df['Low'].tail(20)
        if len(lows) >= 3:
            recent_lows = [lows.iloc[-1], lows.iloc[-10], lows.iloc[-20]]
            if recent_lows[0] > recent_lows[1] > recent_lows[2]:
                patterns.append('higher_lows')
        
        # Check for consolidation breakout
        high_10 = df['High'].tail(10).max()
        low_10 = df['Low'].tail(10).min()
        range_pct = ((high_10 - low_10) / low_10) * 100
        
        if range_pct < 5 and df['Close'].iloc[-1] > high_10 * 0.99:
            patterns.append('consolidation_breakout')
        
        # Check for pullback in uptrend
        sma_20 = df['Close'].rolling(window=20).mean()
        if (df['Close'].iloc[-1] > sma_20.iloc[-1] and 
            df['Low'].iloc[-1] <= sma_20.iloc[-1] * 1.02):
            patterns.append('pullback_to_support')
        
        # Flag pattern detection (simple version)
        if len(df) >= 30:
            recent_high = df['High'].iloc[-30:-10].max()
            recent_low = df['Low'].iloc[-30:-10].min()
            current_range = df['High'].iloc[-10:].max() - df['Low'].iloc[-10:].min()
            prior_range = recent_high - recent_low
            
            if current_range < prior_range * 0.5:
                patterns.append('flag_pattern')
        
        pattern_data = {
            'patterns_detected': patterns,
            'pattern_count': len(patterns),
            'has_bullish_pattern': len(patterns) > 0,
            'passes_filter': len(patterns) > 0 or True  # Don't require patterns
        }
        
        return pattern_data
    
    def calculate_filter_score(self, all_filters):
        """Calculate overall filter score"""
        score = 0
        max_score = 0
        
        # Liquidity (critical)
        if all_filters['liquidity']['passes_filter']:
            score += 20
        max_score += 20
        
        # Trend alignment (important)
        if all_filters['trend']['passes_filter']:
            score += 20
        max_score += 20
        
        # Relative strength (important)
        if all_filters['relative_strength']['passes_filter']:
            score += 20
        max_score += 20
        
        # Volatility (moderate)
        if all_filters['volatility']['passes_filter']:
            score += 15
        max_score += 15
        
        # Momentum (moderate)
        if all_filters['momentum']['passes_filter']:
            score += 15
        max_score += 15
        
        # Patterns (bonus)
        if all_filters['patterns']['has_bullish_pattern']:
            score += 10
        max_score += 10
        
        filter_percentage = (score / max_score) * 100
        
        return {
            'filter_score': score,
            'max_score': max_score,
            'filter_percentage': filter_percentage,
            'passes_all_critical': (
                all_filters['liquidity']['passes_filter'] and
                all_filters['trend']['passes_filter']
            ),
            'recommendation': self._get_filter_recommendation(filter_percentage)
        }
    
    def _get_filter_recommendation(self, percentage):
        """Convert filter score to recommendation"""
        if percentage >= 80:
            return 'EXCELLENT'
        elif percentage >= 70:
            return 'GOOD'
        elif percentage >= 60:
            return 'FAIR'
        else:
            return 'POOR'
    
    def apply_filters(self, symbol, df, market_df=None, info=None):
        """Apply all filters to a stock"""
        print(f"\n🔍 Applying entry filters for {symbol}...")
        
        filters = {
            'symbol': symbol,
            'liquidity': self.check_liquidity_filter(df, info),
            'trend': self.check_trend_alignment(df, market_df),
            'relative_strength': self.calculate_relative_strength(df, market_df),
            'volatility': self.check_volatility_filter(df),
            'momentum': self.check_momentum_filter(df),
            'patterns': self.check_pattern_filter(df)
        }
        
        # Calculate overall score
        filters['overall'] = self.calculate_filter_score(filters)
        filters['passes_filters'] = filters['overall']['filter_percentage'] >= 60
        
        return filters
    
    def generate_filter_summary(self, filters):
        """Generate human-readable filter summary"""
        summary = f"\n📋 Entry Filter Analysis for {filters['symbol']}\n"
        summary += "=" * 50 + "\n"
        
        # Overall result
        overall = filters['overall']
        summary += f"\n🎯 Overall Score: {overall['filter_percentage']:.1f}% - {overall['recommendation']}\n"
        
        # Individual filters
        summary += "\n📊 Filter Results:\n"
        
        # Liquidity
        liq = filters['liquidity']
        status = "✅" if liq['passes_filter'] else "❌"
        summary += f"{status} Liquidity: ${liq['avg_dollar_volume']/1e6:.1f}M daily volume\n"
        
        # Trend
        trend = filters['trend']
        status = "✅" if trend['passes_filter'] else "❌"
        summary += f"{status} Trend: {trend['trend_strength']} (Score: {trend['trend_score']})\n"
        
        # Relative Strength
        rs = filters['relative_strength']
        status = "✅" if rs['passes_filter'] else "❌"
        summary += f"{status} Relative Strength: {rs['rs_vs_market']:.2f}x market\n"
        
        # Volatility
        vol = filters['volatility']
        status = "✅" if vol['passes_filter'] else "❌"
        summary += f"{status} Volatility: {vol['atr_pct']:.1f}% ATR\n"
        
        # Momentum
        mom = filters['momentum']
        status = "✅" if mom['passes_filter'] else "❌"
        summary += f"{status} Momentum: RSI {mom['rsi']:.1f} ({mom['rsi_status']})\n"
        
        # Patterns
        pat = filters['patterns']
        if pat['patterns_detected']:
            summary += f"✅ Patterns: {', '.join(pat['patterns_detected'])}\n"
        
        # Final recommendation
        if filters['passes_filters']:
            summary += "\n✅ PASSES ENTRY FILTERS - Proceed with analysis"
        else:
            summary += "\n❌ FAILS ENTRY FILTERS - Skip this trade"
        
        return summary


def main():
    """Test the entry filter system"""
    filter_system = EntryFilterSystem()
    
    # Test with a stock
    symbol = 'AAPL'
    stock = yf.Ticker(symbol)
    df = stock.history(period='6mo')
    
    # Get market data (SPY as proxy)
    spy = yf.Ticker('SPY')
    market_df = spy.history(period='6mo')
    
    if not df.empty:
        filters = filter_system.apply_filters(symbol, df, market_df, stock.info)
        print(filter_system.generate_filter_summary(filters))
        
        # Show detailed scores
        print("\n📊 Detailed Scores:")
        for filter_name, filter_data in filters.items():
            if filter_name not in ['symbol', 'overall', 'passes_filters']:
                print(f"\n{filter_name.title()}:")
                for key, value in filter_data.items():
                    if key != 'passes_filter':
                        print(f"  {key}: {value}")

if __name__ == "__main__":
    main()