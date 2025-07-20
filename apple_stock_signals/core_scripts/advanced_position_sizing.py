#!/usr/bin/env python3
"""
Advanced Position Sizing Module
Implements Kelly Criterion, volatility-based, and correlation-adjusted sizing
Optimized for 2-15 day swing trading
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

class AdvancedPositionSizing:
    """
    Sophisticated position sizing for optimal risk-adjusted returns
    """
    
    def __init__(self, account_size=10000):
        self.account_size = account_size
        self.config = {
            'max_risk_per_trade': 0.02,          # 2% max risk per trade
            'max_portfolio_risk': 0.06,          # 6% max total portfolio risk
            'max_position_size_pct': 0.25,       # 25% max position size
            'max_correlated_exposure': 0.40,     # 40% max in correlated positions
            'min_position_size_pct': 0.02,       # 2% minimum position
            'kelly_fraction': 0.25,              # Use 25% of Kelly recommendation
            'max_positions': 5,                  # Maximum concurrent positions
            'volatility_scaling': True,          # Scale position by volatility
            'correlation_threshold': 0.7         # High correlation threshold
        }
        
        # Track current positions for portfolio management
        self.current_positions = []
        
    def calculate_kelly_criterion(self, win_rate, avg_win, avg_loss):
        """
        Calculate position size using Kelly Criterion
        Kelly % = (p * b - q) / b
        where p = win probability, q = loss probability, b = win/loss ratio
        """
        if avg_loss == 0 or win_rate <= 0 or win_rate >= 1:
            return 0
        
        loss_rate = 1 - win_rate
        win_loss_ratio = abs(avg_win / avg_loss)
        
        # Kelly formula
        kelly_pct = (win_rate * win_loss_ratio - loss_rate) / win_loss_ratio
        
        # Apply Kelly fraction (never use full Kelly)
        adjusted_kelly = kelly_pct * self.config['kelly_fraction']
        
        # Cap at maximum position size
        kelly_size = min(max(0, adjusted_kelly), self.config['max_position_size_pct'])
        
        return {
            'kelly_full': kelly_pct,
            'kelly_adjusted': adjusted_kelly,
            'kelly_final': kelly_size,
            'win_loss_ratio': win_loss_ratio
        }
    
    def calculate_volatility_adjusted_size(self, df, target_volatility=0.15):
        """
        Adjust position size based on stock volatility vs target volatility
        Lower position size for more volatile stocks
        """
        # Calculate annualized volatility
        returns = df['Close'].pct_change().dropna()
        stock_volatility = returns.std() * np.sqrt(252)
        
        # Calculate volatility scalar
        if stock_volatility > 0:
            volatility_scalar = target_volatility / stock_volatility
            volatility_scalar = min(max(volatility_scalar, 0.5), 1.5)  # Cap between 0.5x and 1.5x
        else:
            volatility_scalar = 1.0
        
        # Volatility categories
        if stock_volatility < 0.20:
            volatility_category = 'low'
        elif stock_volatility < 0.40:
            volatility_category = 'medium'
        else:
            volatility_category = 'high'
        
        return {
            'stock_volatility': stock_volatility,
            'volatility_scalar': volatility_scalar,
            'volatility_category': volatility_category,
            'target_volatility': target_volatility
        }
    
    def calculate_atr_based_size(self, df, stop_distance):
        """
        Calculate position size based on ATR and stop distance
        Ensures consistent dollar risk across different volatility stocks
        """
        current_price = df['Close'].iloc[-1]
        
        # Calculate shares based on risk amount
        risk_amount = self.account_size * self.config['max_risk_per_trade']
        shares = int(risk_amount / stop_distance)
        
        # Calculate position value
        position_value = shares * current_price
        position_pct = position_value / self.account_size
        
        # Apply position size limits
        if position_pct > self.config['max_position_size_pct']:
            # Reduce shares to fit within max position size
            max_position_value = self.account_size * self.config['max_position_size_pct']
            shares = int(max_position_value / current_price)
            position_value = shares * current_price
            position_pct = position_value / self.account_size
            actual_risk = (shares * stop_distance) / self.account_size
        else:
            actual_risk = self.config['max_risk_per_trade']
        
        return {
            'shares': shares,
            'position_value': position_value,
            'position_pct': position_pct,
            'risk_amount': shares * stop_distance,
            'actual_risk_pct': actual_risk,
            'stop_distance': stop_distance
        }
    
    def check_correlation_limits(self, new_symbol, new_sector, current_positions):
        """
        Check if adding new position would exceed correlation limits
        """
        if not current_positions:
            return {'can_add': True, 'correlated_exposure': 0}
        
        # Count exposure by sector
        sector_exposure = {}
        total_exposure = 0
        
        for pos in current_positions:
            sector = pos.get('sector', 'Unknown')
            exposure = pos.get('position_pct', 0)
            
            sector_exposure[sector] = sector_exposure.get(sector, 0) + exposure
            total_exposure += exposure
        
        # Check if adding new position would exceed limits
        current_sector_exposure = sector_exposure.get(new_sector, 0)
        
        correlation_check = {
            'can_add': True,
            'correlated_exposure': current_sector_exposure,
            'sector_exposure': sector_exposure,
            'total_exposure': total_exposure,
            'reasons': []
        }
        
        # Check sector concentration
        if current_sector_exposure >= self.config['max_correlated_exposure']:
            correlation_check['can_add'] = False
            correlation_check['reasons'].append(f"Sector exposure limit reached ({current_sector_exposure:.1%})")
        
        # Check total exposure
        if total_exposure >= 0.9:  # 90% invested
            correlation_check['can_add'] = False
            correlation_check['reasons'].append(f"Portfolio nearly fully invested ({total_exposure:.1%})")
        
        # Check number of positions
        if len(current_positions) >= self.config['max_positions']:
            correlation_check['can_add'] = False
            correlation_check['reasons'].append(f"Maximum positions reached ({len(current_positions)})")
        
        return correlation_check
    
    def calculate_risk_parity_size(self, positions_data):
        """
        Calculate position sizes to equalize risk contribution across positions
        """
        if not positions_data:
            return {}
        
        # Calculate risk contribution for each position
        total_risk = 0
        position_risks = []
        
        for pos in positions_data:
            volatility = pos.get('volatility', 0.20)
            position_size = pos.get('position_pct', 0)
            risk_contribution = position_size * volatility
            
            position_risks.append({
                'symbol': pos['symbol'],
                'current_size': position_size,
                'volatility': volatility,
                'risk_contribution': risk_contribution
            })
            
            total_risk += risk_contribution
        
        # Calculate target risk per position
        target_risk = total_risk / len(positions_data)
        
        # Calculate adjusted sizes
        risk_parity_sizes = {}
        
        for pos in position_risks:
            if pos['volatility'] > 0:
                target_size = target_risk / pos['volatility']
                risk_parity_sizes[pos['symbol']] = {
                    'current_size': pos['current_size'],
                    'target_size': target_size,
                    'adjustment': target_size - pos['current_size']
                }
        
        return risk_parity_sizes
    
    def calculate_optimal_position_size(self, analysis_data, stop_loss_data, 
                                      historical_performance=None, current_positions=None):
        """
        Calculate optimal position size considering all factors
        """
        symbol = analysis_data['symbol']
        df = analysis_data.get('price_data')
        stop_distance = stop_loss_data['stop_distance']
        
        # Initialize sizing components
        sizing_components = {}
        
        # 1. ATR-based sizing (base calculation)
        atr_sizing = self.calculate_atr_based_size(df, stop_distance)
        sizing_components['atr_based'] = atr_sizing
        
        # 2. Volatility adjustment
        if self.config['volatility_scaling']:
            vol_adjustment = self.calculate_volatility_adjusted_size(df)
            adjusted_shares = int(atr_sizing['shares'] * vol_adjustment['volatility_scalar'])
            sizing_components['volatility_adjusted'] = {
                'shares': adjusted_shares,
                'adjustment_factor': vol_adjustment['volatility_scalar'],
                'volatility': vol_adjustment['stock_volatility']
            }
        else:
            adjusted_shares = atr_sizing['shares']
        
        # 3. Kelly Criterion (if historical performance available)
        if historical_performance:
            kelly_data = self.calculate_kelly_criterion(
                historical_performance.get('win_rate', 0.5),
                historical_performance.get('avg_win', 2.0),
                historical_performance.get('avg_loss', -1.0)
            )
            
            kelly_position_value = self.account_size * kelly_data['kelly_final']
            kelly_shares = int(kelly_position_value / df['Close'].iloc[-1])
            
            sizing_components['kelly'] = {
                'shares': kelly_shares,
                'position_pct': kelly_data['kelly_final']
            }
            
            # Use minimum of ATR-based and Kelly
            final_shares = min(adjusted_shares, kelly_shares)
        else:
            final_shares = adjusted_shares
        
        # 4. Check correlation and portfolio limits
        current_positions = current_positions or []
        correlation_check = self.check_correlation_limits(
            symbol, 
            analysis_data.get('sector', 'Unknown'),
            current_positions
        )
        
        if not correlation_check['can_add']:
            final_shares = 0
            sizing_components['correlation_limit'] = correlation_check
        
        # 5. Calculate final position metrics
        current_price = df['Close'].iloc[-1]
        position_value = final_shares * current_price
        position_pct = position_value / self.account_size
        
        # Ensure minimum position size if taking position
        if final_shares > 0 and position_pct < self.config['min_position_size_pct']:
            min_value = self.account_size * self.config['min_position_size_pct']
            final_shares = int(min_value / current_price)
            position_value = final_shares * current_price
            position_pct = position_value / self.account_size
        
        # Calculate actual risk
        actual_risk_amount = final_shares * stop_distance
        actual_risk_pct = actual_risk_amount / self.account_size
        
        # Risk/Reward metrics
        risk_reward_ratio = analysis_data.get('risk_reward_ratio', 2.0)
        expected_return = (actual_risk_pct * risk_reward_ratio * 
                          historical_performance.get('win_rate', 0.5) - 
                          actual_risk_pct * (1 - historical_performance.get('win_rate', 0.5))) \
                          if historical_performance else actual_risk_pct
        
        return {
            'symbol': symbol,
            'recommended_shares': final_shares,
            'position_value': position_value,
            'position_pct': position_pct,
            'actual_risk_amount': actual_risk_amount,
            'actual_risk_pct': actual_risk_pct,
            'stop_distance': stop_distance,
            'current_price': current_price,
            'sizing_components': sizing_components,
            'can_take_position': final_shares > 0,
            'expected_return': expected_return,
            'risk_reward_ratio': risk_reward_ratio,
            'sizing_method': self._determine_sizing_method(sizing_components)
        }
    
    def _determine_sizing_method(self, components):
        """Determine which sizing method was most restrictive"""
        if 'correlation_limit' in components and components['correlation_limit']['can_add'] == False:
            return 'correlation_limited'
        elif 'kelly' in components:
            return 'kelly_criterion'
        elif 'volatility_adjusted' in components:
            return 'volatility_adjusted'
        else:
            return 'atr_based'
    
    def calculate_portfolio_heat(self, current_positions):
        """
        Calculate total portfolio heat (risk)
        Heat = sum of all position risks
        """
        total_heat = 0
        position_details = []
        
        for pos in current_positions:
            risk_pct = pos.get('risk_pct', 0)
            total_heat += risk_pct
            
            position_details.append({
                'symbol': pos['symbol'],
                'risk_pct': risk_pct,
                'position_pct': pos.get('position_pct', 0)
            })
        
        remaining_risk_capacity = self.config['max_portfolio_risk'] - total_heat
        
        return {
            'total_heat': total_heat,
            'heat_pct': (total_heat / self.config['max_portfolio_risk']) * 100,
            'remaining_risk_capacity': max(0, remaining_risk_capacity),
            'can_add_position': total_heat < self.config['max_portfolio_risk'],
            'position_count': len(current_positions),
            'positions': position_details
        }
    
    def generate_sizing_report(self, sizing_data):
        """Generate human-readable position sizing report"""
        report = "\n💰 Position Sizing Analysis\n"
        report += "=" * 50 + "\n"
        
        report += f"\nRecommended Position:"
        report += f"\n  Shares: {sizing_data['recommended_shares']}"
        report += f"\n  Position Value: ${sizing_data['position_value']:,.2f}"
        report += f"\n  Position Size: {sizing_data['position_pct']:.1%} of account"
        report += f"\n  Risk Amount: ${sizing_data['actual_risk_amount']:.2f}"
        report += f"\n  Risk Percent: {sizing_data['actual_risk_pct']:.2%}"
        report += f"\n  Sizing Method: {sizing_data['sizing_method']}"
        
        if not sizing_data['can_take_position']:
            report += f"\n\n⚠️  WARNING: Cannot take position"
            if 'correlation_limit' in sizing_data['sizing_components']:
                reasons = sizing_data['sizing_components']['correlation_limit']['reasons']
                for reason in reasons:
                    report += f"\n  - {reason}"
        
        # Component breakdown
        report += f"\n\nSizing Components:"
        
        if 'atr_based' in sizing_data['sizing_components']:
            atr = sizing_data['sizing_components']['atr_based']
            report += f"\n  ATR-Based: {atr['shares']} shares ({atr['position_pct']:.1%})"
        
        if 'volatility_adjusted' in sizing_data['sizing_components']:
            vol = sizing_data['sizing_components']['volatility_adjusted']
            report += f"\n  Vol-Adjusted: {vol['shares']} shares (×{vol['adjustment_factor']:.2f})"
        
        if 'kelly' in sizing_data['sizing_components']:
            kelly = sizing_data['sizing_components']['kelly']
            report += f"\n  Kelly Criterion: {kelly['shares']} shares ({kelly['position_pct']:.1%})"
        
        return report


def main():
    """Test the position sizing module"""
    import yfinance as yf
    
    # Initialize with $10,000 account
    sizer = AdvancedPositionSizing(account_size=10000)
    
    # Test stock
    symbol = 'AAPL'
    stock = yf.Ticker(symbol)
    df = stock.history(period='3mo')
    
    if not df.empty:
        print(f"🎯 Testing Position Sizing for {symbol}")
        print(f"Account Size: ${sizer.account_size:,}")
        print(f"Current Price: ${df['Close'].iloc[-1]:.2f}")
        
        # Simulate analysis data
        analysis_data = {
            'symbol': symbol,
            'price_data': df,
            'sector': 'Technology',
            'risk_reward_ratio': 2.5
        }
        
        # Simulate stop loss data
        stop_loss_data = {
            'stop_distance': df['Close'].iloc[-1] * 0.02  # 2% stop
        }
        
        # Simulate historical performance
        historical_performance = {
            'win_rate': 0.55,
            'avg_win': 2.5,
            'avg_loss': -1.0
        }
        
        # Simulate current positions
        current_positions = [
            {'symbol': 'MSFT', 'sector': 'Technology', 'position_pct': 0.15, 'risk_pct': 0.02},
            {'symbol': 'GOOGL', 'sector': 'Technology', 'position_pct': 0.10, 'risk_pct': 0.015}
        ]
        
        # Calculate optimal position size
        sizing = sizer.calculate_optimal_position_size(
            analysis_data, 
            stop_loss_data,
            historical_performance,
            current_positions
        )
        
        # Generate report
        print(sizer.generate_sizing_report(sizing))
        
        # Portfolio heat check
        heat = sizer.calculate_portfolio_heat(current_positions)
        print(f"\n🔥 Portfolio Heat Analysis:")
        print(f"  Total Heat: {heat['total_heat']:.2%}")
        print(f"  Heat Percentage: {heat['heat_pct']:.1f}%")
        print(f"  Remaining Capacity: {heat['remaining_risk_capacity']:.2%}")

if __name__ == "__main__":
    main()