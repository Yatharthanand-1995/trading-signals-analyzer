#!/usr/bin/env python3
"""
Volatility-Adjusted Position Sizing System
Implements Kelly Criterion and advanced position sizing methods
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class VolatilityPositionSizer:
    """
    Advanced position sizing that considers:
    - Kelly Criterion for optimal sizing
    - Volatility-based adjustments
    - Correlation-based portfolio sizing
    - Maximum portfolio heat limits
    - Dynamic leverage optimization
    """
    
    def __init__(self, portfolio_size: float = 10000):
        self.portfolio_size = portfolio_size
        self.risk_config = self.risk_parameters = {
            'max_risk_per_trade': 0.02,      # 2% max risk per trade
            'max_portfolio_heat': 0.06,      # 6% max portfolio risk
            'max_correlated_exposure': 0.15,  # 15% max in correlated positions
            'min_position_size': 0.01,        # 1% minimum position
            'max_position_size': 0.25,        # 25% maximum position
            'volatility_lookback': 20,        # Days for volatility calculation
            'confidence_threshold': 0.55      # Minimum win rate for Kelly
        }
        
    def calculate_position_size(self, entry_price: float, stop_loss: float, confidence: float = 0.5,
                              take_profit: float = None, market_data: pd.DataFrame = None,
                              historical_performance: Dict = None, existing_positions: List[Dict] = None) -> Dict:
        """
        Calculate position size - simplified interface for tests
        """
        if take_profit is None:
            take_profit = entry_price * 1.05
        
        if historical_performance is None:
            historical_performance = {
                'win_rate': 0.5,
                'avg_win': 0.02,
                'avg_loss': 0.01
            }
        
        if market_data is None:
            # Create mock market data
            dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
            market_data = pd.DataFrame({
                'Close': np.random.normal(entry_price, entry_price * 0.02, 100),
                'High': np.random.normal(entry_price * 1.01, entry_price * 0.02, 100),
                'Low': np.random.normal(entry_price * 0.99, entry_price * 0.02, 100),
                'Volume': np.random.randint(1000000, 5000000, 100)
            }, index=dates)
        
        trade_setup = {
            'symbol': 'TEST',
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit_1': take_profit,
            'confidence': 'HIGH' if confidence > 0.7 else 'MEDIUM' if confidence > 0.5 else 'LOW'
        }
        
        result = self.calculate_optimal_position_size(trade_setup, historical_performance, market_data, existing_positions)
        
        # Convert to expected format
        return {
            'shares': result['optimal_shares'],
            'position_value': result['position_value'],
            'risk_amount': result['risk_amount'],
            'risk_percentage': result['risk_percentage'],
            'position_percentage': result['position_percentage']
        }
    
    def calculate_optimal_position_size(self, trade_setup: Dict, historical_performance: Dict, 
                                       market_data: pd.DataFrame, existing_positions: List[Dict] = None) -> Dict:
        """
        Calculate optimal position size using multiple methods
        """
        symbol = trade_setup['symbol']
        entry_price = trade_setup['entry_price']
        stop_loss = trade_setup['stop_loss']
        take_profit = trade_setup.get('take_profit_1', entry_price * 1.05)
        confidence = trade_setup.get('confidence', 'MEDIUM')
        
        # Calculate various position sizes
        kelly_size = self._calculate_kelly_position(historical_performance, entry_price, stop_loss, take_profit)
        volatility_size = self._calculate_volatility_adjusted_size(market_data, entry_price, stop_loss)
        risk_based_size = self._calculate_risk_based_size(entry_price, stop_loss)
        
        # Adjust for correlation if existing positions
        correlation_adjustment = 1.0
        if existing_positions:
            correlation_adjustment = self._calculate_correlation_adjustment(symbol, market_data, existing_positions)
        
        # Combine methods with weights
        weights = self._get_sizing_weights(confidence, market_data)
        
        optimal_size = (
            kelly_size * weights['kelly'] +
            volatility_size * weights['volatility'] +
            risk_based_size * weights['risk_based']
        ) * correlation_adjustment
        
        # Apply portfolio heat limits
        if existing_positions:
            optimal_size = self._apply_portfolio_heat_limit(optimal_size, entry_price, stop_loss, existing_positions)
        
        # Apply min/max constraints
        min_shares = max(1, int(self.portfolio_size * self.risk_parameters['min_position_size'] / entry_price))
        max_shares = int(self.portfolio_size * self.risk_parameters['max_position_size'] / entry_price)
        
        final_shares = int(np.clip(optimal_size, min_shares, max_shares))
        
        # Calculate position metrics
        position_value = final_shares * entry_price
        position_percentage = (position_value / self.portfolio_size) * 100
        risk_amount = final_shares * abs(entry_price - stop_loss)
        risk_percentage = (risk_amount / self.portfolio_size) * 100
        
        return {
            'symbol': symbol,
            'optimal_shares': final_shares,
            'position_value': position_value,
            'position_percentage': position_percentage,
            'risk_amount': risk_amount,
            'risk_percentage': risk_percentage,
            'sizing_methods': {
                'kelly': kelly_size,
                'volatility_adjusted': volatility_size,
                'risk_based': risk_based_size
            },
            'adjustments': {
                'correlation_factor': correlation_adjustment,
                'confidence_weight': weights
            },
            'constraints_applied': {
                'min_shares': min_shares,
                'max_shares': max_shares,
                'portfolio_heat_limited': existing_positions is not None
            },
            'recommendation': self._generate_sizing_recommendation(
                final_shares, position_percentage, risk_percentage, confidence
            )
        }
    
    def _calculate_kelly_criterion(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """
        Calculate Kelly Criterion fraction
        """
        if avg_loss == 0:
            return 0
        
        b = avg_win / avg_loss
        p = win_rate
        q = 1 - p
        
        if p * b > q:
            kelly = (p * b - q) / b
            return max(0, min(1, kelly))  # Bound between 0 and 1
        else:
            return 0
    
    def _calculate_kelly_position(self, historical_performance: Dict, entry_price: float, 
                                 stop_loss: float, take_profit: float) -> int:
        """
        Calculate position size using Kelly Criterion
        Kelly % = (p * b - q) / b
        where p = win probability, q = loss probability, b = win/loss ratio
        """
        # Extract performance metrics
        win_rate = historical_performance.get('win_rate', 0.45)
        avg_win = historical_performance.get('avg_win', 0.02)  # 2% average win
        avg_loss = historical_performance.get('avg_loss', 0.01)  # 1% average loss
        
        # Calculate for this specific trade
        potential_win = (take_profit - entry_price) / entry_price
        potential_loss = (entry_price - stop_loss) / entry_price
        
        # Use historical ratios if available, otherwise use trade setup
        if avg_loss > 0:
            b = avg_win / avg_loss
        else:
            b = potential_win / potential_loss if potential_loss > 0 else 2.0
        
        # Kelly formula
        p = win_rate
        q = 1 - p
        
        # Only use Kelly if we have positive expectancy
        if p * b > q:
            kelly_percentage = (p * b - q) / b
            
            # Apply Kelly fraction (typically 0.25 to 0.5 of full Kelly)
            kelly_fraction = 0.25  # Conservative
            adjusted_kelly = kelly_percentage * kelly_fraction
            
            # Convert to shares
            position_value = self.portfolio_size * adjusted_kelly
            shares = int(position_value / entry_price)
            
            return max(0, shares)
        else:
            # Negative expectancy - minimum position
            return int(self.portfolio_size * 0.01 / entry_price)
    
    def _calculate_volatility_adjusted_size(self, market_data: pd.DataFrame, 
                                          entry_price: float, stop_loss: float) -> int:
        """
        Adjust position size based on volatility
        Lower volatility = larger position, Higher volatility = smaller position
        """
        # Calculate recent volatility
        returns = market_data['Close'].pct_change()
        volatility = returns.tail(self.risk_parameters['volatility_lookback']).std()
        annualized_volatility = volatility * np.sqrt(252)
        
        # Calculate ATR-based volatility
        atr = self._calculate_atr(market_data)
        atr_percentage = (atr.iloc[-1] / market_data['Close'].iloc[-1]) * 100
        
        # Volatility buckets and position size multipliers
        if annualized_volatility < 0.15:  # Low volatility (<15%)
            volatility_multiplier = 1.5
        elif annualized_volatility < 0.25:  # Normal volatility (15-25%)
            volatility_multiplier = 1.0
        elif annualized_volatility < 0.40:  # High volatility (25-40%)
            volatility_multiplier = 0.7
        else:  # Very high volatility (>40%)
            volatility_multiplier = 0.5
        
        # ATR adjustment
        if atr_percentage < 1:  # Very tight ATR
            atr_multiplier = 1.2
        elif atr_percentage < 2:  # Normal ATR
            atr_multiplier = 1.0
        elif atr_percentage < 3:  # Wide ATR
            atr_multiplier = 0.8
        else:  # Very wide ATR
            atr_multiplier = 0.6
        
        # Base position from risk
        risk_amount = self.portfolio_size * self.risk_parameters['max_risk_per_trade']
        stop_distance = abs(entry_price - stop_loss)
        if stop_distance > 0:
            base_shares = risk_amount / stop_distance
        else:
            base_shares = 0
        
        # Apply volatility adjustments
        adjusted_shares = base_shares * volatility_multiplier * atr_multiplier
        
        return int(adjusted_shares)
    
    def _calculate_risk_based_size(self, entry_price: float, stop_loss: float) -> int:
        """
        Traditional fixed-risk position sizing
        Position Size = (Account Risk $) / (Stop Distance $)
        """
        risk_amount = self.portfolio_size * self.risk_parameters['max_risk_per_trade']
        stop_distance = abs(entry_price - stop_loss)
        
        if stop_distance > 0:
            shares = risk_amount / stop_distance
            return int(shares)
        else:
            return 0
    
    def _calculate_correlation_adjustment(self, symbol: str, market_data: pd.DataFrame, 
                                        existing_positions: List[Dict]) -> float:
        """
        Reduce position size if highly correlated with existing positions
        """
        if not existing_positions:
            return 1.0
        
        # Get returns for correlation calculation
        symbol_returns = market_data['Close'].pct_change().tail(60)
        
        max_correlation = 0
        total_correlated_exposure = 0
        
        for position in existing_positions:
            # Skip if same symbol
            if position['symbol'] == symbol:
                continue
            
            # Calculate correlation (would need other symbol's data in real implementation)
            # For now, use sector-based correlation estimates
            correlation = self._estimate_correlation(symbol, position['symbol'])
            
            if correlation > 0.7:  # High correlation threshold
                max_correlation = max(max_correlation, correlation)
                total_correlated_exposure += position.get('position_percentage', 0)
        
        # Reduce size based on correlation
        if max_correlation > 0.9:
            adjustment = 0.5  # Halve position for very high correlation
        elif max_correlation > 0.8:
            adjustment = 0.7
        elif max_correlation > 0.7:
            adjustment = 0.85
        else:
            adjustment = 1.0
        
        # Further reduce if total correlated exposure is high
        if total_correlated_exposure > 10:  # >10% in correlated positions
            adjustment *= 0.8
        
        return adjustment
    
    def _apply_portfolio_heat_limit(self, proposed_shares: int, entry_price: float, 
                                   stop_loss: float, existing_positions: List[Dict]) -> int:
        """
        Ensure total portfolio risk doesn't exceed heat limit
        """
        # Calculate current portfolio heat
        current_heat = 0
        for position in existing_positions:
            current_heat += position.get('risk_percentage', 0) / 100
        
        # Calculate proposed position risk
        proposed_risk = (proposed_shares * abs(entry_price - stop_loss)) / self.portfolio_size
        
        # Check if we exceed limit
        total_heat = current_heat + proposed_risk
        
        if total_heat > self.risk_parameters['max_portfolio_heat']:
            # Reduce position to fit within limit
            available_heat = self.risk_parameters['max_portfolio_heat'] - current_heat
            if available_heat > 0:
                max_risk_amount = self.portfolio_size * available_heat
                reduced_shares = int(max_risk_amount / abs(entry_price - stop_loss))
                return max(1, reduced_shares)
            else:
                return 0  # No room for new positions
        
        return int(proposed_shares)
    
    def _get_sizing_weights(self, confidence: str, market_data: pd.DataFrame) -> Dict:
        """
        Determine weights for different sizing methods based on confidence
        """
        # Base weights
        weights = {
            'kelly': 0.3,
            'volatility': 0.4,
            'risk_based': 0.3
        }
        
        # Adjust based on confidence
        if confidence == 'VERY_HIGH':
            weights['kelly'] = 0.5  # More weight to Kelly with high confidence
            weights['volatility'] = 0.3
            weights['risk_based'] = 0.2
        elif confidence == 'HIGH':
            weights['kelly'] = 0.4
            weights['volatility'] = 0.35
            weights['risk_based'] = 0.25
        elif confidence == 'LOW':
            weights['kelly'] = 0.1  # Less Kelly with low confidence
            weights['volatility'] = 0.4
            weights['risk_based'] = 0.5
        
        # Adjust for market conditions
        recent_volatility = market_data['Close'].pct_change().tail(20).std()
        if recent_volatility > 0.03:  # High volatility
            weights['volatility'] += 0.1
            weights['kelly'] -= 0.1
        
        # Normalize weights
        total = sum(weights.values())
        weights = {k: v/total for k, v in weights.items()}
        
        return weights
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high = data['High']
        low = data['Low']
        close = data['Close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        return tr.rolling(period).mean()
    
    def _estimate_correlation(self, symbol1: str, symbol2: str) -> float:
        """
        Estimate correlation between symbols based on sector/industry
        In real implementation, would calculate actual correlation
        """
        # Define sector groups
        tech_stocks = ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'AMZN']
        financial_stocks = ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C']
        healthcare_stocks = ['JNJ', 'UNH', 'PFE', 'ABBV', 'MRK', 'TMO']
        
        # Same symbol
        if symbol1 == symbol2:
            return 1.0
        
        # Check sector correlation
        for sector in [tech_stocks, financial_stocks, healthcare_stocks]:
            if symbol1 in sector and symbol2 in sector:
                return 0.8  # High correlation within sector
        
        # Different sectors - lower correlation
        return 0.3
    
    def _generate_sizing_recommendation(self, shares: int, position_pct: float, 
                                      risk_pct: float, confidence: str) -> str:
        """
        Generate recommendation text for position sizing
        """
        rec = f"Position Size: {shares} shares ({position_pct:.1f}% of portfolio)\n"
        rec += f"Risk: {risk_pct:.1f}% of portfolio\n"
        
        # Size assessment
        if position_pct > 20:
            rec += "⚠️ Large position - consider reducing for diversification\n"
        elif position_pct > 15:
            rec += "Significant position - monitor closely\n"
        elif position_pct < 2:
            rec += "Small position - limited impact on portfolio\n"
        
        # Risk assessment
        if risk_pct > 1.5:
            rec += "Above normal risk - ensure stop loss is appropriate\n"
        elif risk_pct < 0.5:
            rec += "Conservative risk - could increase if high confidence\n"
        
        # Confidence-based advice
        if confidence == 'VERY_HIGH':
            rec += "High confidence setup - position sized appropriately"
        elif confidence == 'LOW':
            rec += "Low confidence - position reduced for safety"
        
        return rec
    
    def calculate_portfolio_allocation(self, trade_signals: List[Dict], 
                                     historical_performance: Dict,
                                     market_data: Dict[str, pd.DataFrame]) -> Dict:
        """
        Allocate capital across multiple signals optimally
        """
        if not trade_signals:
            return {'allocations': [], 'summary': 'No signals to allocate'}
        
        # Sort by score/confidence
        sorted_signals = sorted(trade_signals, key=lambda x: x.get('combined_score', 0), reverse=True)
        
        allocations = []
        total_allocated = 0
        total_risk = 0
        existing_positions = []
        
        for signal in sorted_signals:
            # Check if we have room for more positions
            if total_risk >= self.risk_parameters['max_portfolio_heat']:
                signal['allocation_status'] = 'Skipped - Portfolio heat limit reached'
                continue
            
            if total_allocated >= 0.9 * self.portfolio_size:
                signal['allocation_status'] = 'Skipped - Capital fully allocated'
                continue
            
            # Get market data for symbol
            if signal['symbol'] not in market_data:
                signal['allocation_status'] = 'Skipped - No market data'
                continue
            
            # Calculate position size
            position_info = self.calculate_optimal_position_size(
                signal,
                historical_performance,
                market_data[signal['symbol']],
                existing_positions
            )
            
            # Check if position fits within constraints
            if position_info['position_value'] + total_allocated > self.portfolio_size:
                # Reduce position to fit
                available_capital = self.portfolio_size - total_allocated
                reduced_shares = int(available_capital / signal['entry_price'])
                if reduced_shares > 0:
                    position_info['optimal_shares'] = reduced_shares
                    position_info['position_value'] = reduced_shares * signal['entry_price']
                else:
                    signal['allocation_status'] = 'Skipped - Insufficient capital'
                    continue
            
            # Add to allocations
            allocation = {
                'symbol': signal['symbol'],
                'shares': position_info['optimal_shares'],
                'position_value': position_info['position_value'],
                'position_percentage': position_info['position_percentage'],
                'risk_amount': position_info['risk_amount'],
                'risk_percentage': position_info['risk_percentage'],
                'entry_price': signal['entry_price'],
                'stop_loss': signal['stop_loss'],
                'confidence': signal.get('confidence', 'MEDIUM'),
                'allocation_status': 'Allocated'
            }
            
            allocations.append(allocation)
            existing_positions.append(allocation)
            total_allocated += position_info['position_value']
            total_risk += position_info['risk_percentage'] / 100
        
        # Generate summary
        summary = {
            'total_positions': len(allocations),
            'total_allocated': total_allocated,
            'total_allocated_pct': (total_allocated / self.portfolio_size) * 100,
            'total_risk_pct': total_risk * 100,
            'cash_remaining': self.portfolio_size - total_allocated,
            'cash_remaining_pct': ((self.portfolio_size - total_allocated) / self.portfolio_size) * 100
        }
        
        return {
            'allocations': allocations,
            'summary': summary,
            'skipped_signals': [s for s in sorted_signals if s.get('allocation_status', '').startswith('Skipped')]
        }


def main():
    """Test the position sizing system"""
    import yfinance as yf
    
    # Initialize sizer
    sizer = VolatilityPositionSizer(portfolio_size=100000)
    
    print("📊 VOLATILITY-ADJUSTED POSITION SIZING DEMO")
    print("="*80)
    print(f"Portfolio Size: ${sizer.portfolio_size:,}")
    print(f"Max Risk Per Trade: {sizer.risk_parameters['max_risk_per_trade']*100}%")
    print(f"Max Portfolio Heat: {sizer.risk_parameters['max_portfolio_heat']*100}%")
    print("\n")
    
    # Test signals
    test_signals = [
        {
            'symbol': 'AAPL',
            'entry_price': 220.0,
            'stop_loss': 215.0,
            'take_profit_1': 230.0,
            'confidence': 'HIGH',
            'combined_score': 75
        },
        {
            'symbol': 'NVDA',
            'entry_price': 500.0,
            'stop_loss': 480.0,
            'take_profit_1': 540.0,
            'confidence': 'VERY_HIGH',
            'combined_score': 85
        },
        {
            'symbol': 'TSLA',
            'entry_price': 300.0,
            'stop_loss': 285.0,
            'take_profit_1': 330.0,
            'confidence': 'MEDIUM',
            'combined_score': 65
        }
    ]
    
    # Historical performance (mock data)
    historical_performance = {
        'win_rate': 0.55,
        'avg_win': 0.03,  # 3% average win
        'avg_loss': 0.015  # 1.5% average loss
    }
    
    # Fetch market data
    market_data = {}
    for signal in test_signals:
        ticker = yf.Ticker(signal['symbol'])
        market_data[signal['symbol']] = ticker.history(period='3mo')
    
    # Calculate individual position sizes
    print("📈 INDIVIDUAL POSITION ANALYSIS")
    print("-"*80)
    
    for signal in test_signals:
        result = sizer.calculate_optimal_position_size(
            signal,
            historical_performance,
            market_data[signal['symbol']]
        )
        
        print(f"\n{signal['symbol']}:")
        print(f"  Shares: {result['optimal_shares']}")
        print(f"  Position Value: ${result['position_value']:,.2f} ({result['position_percentage']:.1f}%)")
        print(f"  Risk Amount: ${result['risk_amount']:,.2f} ({result['risk_percentage']:.1f}%)")
        print(f"  Sizing Methods:")
        print(f"    - Kelly: {result['sizing_methods']['kelly']} shares")
        print(f"    - Volatility: {result['sizing_methods']['volatility_adjusted']} shares")
        print(f"    - Risk-based: {result['sizing_methods']['risk_based']} shares")
    
    # Calculate portfolio allocation
    print("\n\n📊 PORTFOLIO ALLOCATION")
    print("-"*80)
    
    allocation_result = sizer.calculate_portfolio_allocation(
        test_signals,
        historical_performance,
        market_data
    )
    
    print("\nAllocated Positions:")
    for alloc in allocation_result['allocations']:
        print(f"  {alloc['symbol']}: {alloc['shares']} shares @ ${alloc['entry_price']:.2f}")
        print(f"    Value: ${alloc['position_value']:,.2f} ({alloc['position_percentage']:.1f}%)")
        print(f"    Risk: ${alloc['risk_amount']:,.2f} ({alloc['risk_percentage']:.1f}%)")
    
    summary = allocation_result['summary']
    print(f"\nPortfolio Summary:")
    print(f"  Total Positions: {summary['total_positions']}")
    print(f"  Total Allocated: ${summary['total_allocated']:,.2f} ({summary['total_allocated_pct']:.1f}%)")
    print(f"  Total Risk: {summary['total_risk_pct']:.1f}%")
    print(f"  Cash Remaining: ${summary['cash_remaining']:,.2f} ({summary['cash_remaining_pct']:.1f}%)")
    
    # Save results
    import os
    os.makedirs('outputs/position_sizing', exist_ok=True)
    
    output = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'portfolio_size': sizer.portfolio_size,
        'test_signals': test_signals,
        'individual_sizing': [],
        'portfolio_allocation': allocation_result
    }
    
    for signal in test_signals:
        sizing = sizer.calculate_optimal_position_size(
            signal,
            historical_performance,
            market_data[signal['symbol']]
        )
        output['individual_sizing'].append(sizing)
    
    with open('outputs/position_sizing/example_allocation.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print("\n💾 Results saved to outputs/position_sizing/example_allocation.json")


if __name__ == "__main__":
    main()