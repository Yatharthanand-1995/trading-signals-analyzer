#!/usr/bin/env python3
"""
Advanced Risk Management Dashboard
Portfolio risk analysis, correlation monitoring, and stress testing
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from typing import Dict, List, Tuple, Optional
import json
import logging
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RiskManagementDashboard:
    def __init__(self, portfolio_value: float = 100000):
        self.portfolio_value = portfolio_value
        self.risk_free_rate = 0.045  # 4.5% annual risk-free rate
        self.confidence_levels = [0.95, 0.99]  # For VaR calculations
        
        # Risk thresholds
        self.risk_thresholds = {
            'max_portfolio_risk': 0.06,  # 6% max portfolio risk
            'max_position_risk': 0.02,   # 2% max per position
            'max_sector_exposure': 0.30,  # 30% max sector exposure
            'max_correlation': 0.80,      # Max correlation between positions
            'max_drawdown': 0.15,         # 15% max drawdown threshold
            'min_sharpe_ratio': 1.0       # Minimum acceptable Sharpe ratio
        }
        
        # Sector mapping for tech stocks
        self.sector_mapping = {
            'AAPL': 'Technology - Hardware',
            'GOOGL': 'Technology - Services',
            'MSFT': 'Technology - Software',
            'TSLA': 'Consumer Discretionary - Auto',
            'UNH': 'Healthcare'
        }
    
    def calculate_portfolio_var(self, positions: Dict[str, float], 
                              lookback_days: int = 252) -> Dict[str, float]:
        """Calculate Value at Risk for the portfolio"""
        symbols = list(positions.keys())
        weights = np.array(list(positions.values()))
        weights = weights / weights.sum()  # Normalize weights
        
        # Fetch historical data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days * 1.5)
        
        try:
            data = yf.download(symbols, start=start_date, end=end_date)['Adj Close']
            returns = data.pct_change().dropna()
            
            # Portfolio returns
            portfolio_returns = (returns * weights).sum(axis=1)
            
            # Calculate VaR
            var_results = {}
            for confidence in self.confidence_levels:
                var_percentile = (1 - confidence) * 100
                var_value = np.percentile(portfolio_returns, var_percentile)
                var_dollar = self.portfolio_value * abs(var_value)
                var_results[f'var_{int(confidence*100)}'] = {
                    'percentage': abs(var_value) * 100,
                    'dollar_amount': var_dollar
                }
            
            # Calculate CVaR (Conditional VaR / Expected Shortfall)
            var_95_threshold = np.percentile(portfolio_returns, 5)
            cvar_95 = portfolio_returns[portfolio_returns <= var_95_threshold].mean()
            var_results['cvar_95'] = {
                'percentage': abs(cvar_95) * 100,
                'dollar_amount': self.portfolio_value * abs(cvar_95)
            }
            
            return var_results
            
        except Exception as e:
            logger.error(f"Error calculating VaR: {e}")
            return {}
    
    def calculate_correlation_matrix(self, symbols: List[str], 
                                   lookback_days: int = 60) -> pd.DataFrame:
        """Calculate correlation matrix for given symbols"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days * 1.5)
        
        try:
            data = yf.download(symbols, start=start_date, end=end_date)['Adj Close']
            returns = data.pct_change().dropna()
            correlation_matrix = returns.corr()
            return correlation_matrix
            
        except Exception as e:
            logger.error(f"Error calculating correlations: {e}")
            return pd.DataFrame()
    
    def analyze_portfolio_risk(self, positions: Dict[str, Dict]) -> Dict:
        """Comprehensive portfolio risk analysis"""
        # positions format: {'AAPL': {'shares': 100, 'entry_price': 210.50}, ...}
        
        risk_analysis = {
            'portfolio_metrics': {},
            'position_risks': {},
            'correlation_risk': {},
            'sector_exposure': {},
            'risk_scores': {},
            'warnings': [],
            'recommendations': []
        }
        
        # Calculate current values
        total_value = 0
        position_values = {}
        position_weights = {}
        
        for symbol, pos in positions.items():
            current_price = self._get_current_price(symbol)
            position_value = pos['shares'] * current_price
            position_values[symbol] = position_value
            total_value += position_value
        
        # Calculate weights
        for symbol, value in position_values.items():
            position_weights[symbol] = value / total_value
        
        # Portfolio-level metrics
        risk_analysis['portfolio_metrics'] = {
            'total_value': total_value,
            'position_count': len(positions),
            'largest_position': max(position_weights.items(), key=lambda x: x[1]),
            'portfolio_concentration': sum([w**2 for w in position_weights.values()])  # HHI
        }
        
        # Position-level risk analysis
        for symbol, pos in positions.items():
            current_price = self._get_current_price(symbol)
            position_risk = {
                'current_price': current_price,
                'position_value': position_values[symbol],
                'position_weight': position_weights[symbol] * 100,
                'unrealized_pnl': (current_price - pos['entry_price']) * pos['shares'],
                'unrealized_pnl_pct': ((current_price / pos['entry_price']) - 1) * 100,
                'risk_per_share': current_price - pos.get('stop_loss', current_price * 0.95),
                'total_risk': (current_price - pos.get('stop_loss', current_price * 0.95)) * pos['shares']
            }
            risk_analysis['position_risks'][symbol] = position_risk
        
        # Correlation risk
        if len(positions) > 1:
            corr_matrix = self.calculate_correlation_matrix(list(positions.keys()))
            if not corr_matrix.empty:
                # Find high correlations
                high_corr_pairs = []
                for i in range(len(corr_matrix)):
                    for j in range(i+1, len(corr_matrix)):
                        corr = corr_matrix.iloc[i, j]
                        if abs(corr) > self.risk_thresholds['max_correlation']:
                            high_corr_pairs.append({
                                'pair': f"{corr_matrix.index[i]}-{corr_matrix.columns[j]}",
                                'correlation': corr
                            })
                
                risk_analysis['correlation_risk'] = {
                    'average_correlation': corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean(),
                    'max_correlation': corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].max(),
                    'high_correlation_pairs': high_corr_pairs
                }
        
        # Sector exposure
        sector_exposure = {}
        for symbol, weight in position_weights.items():
            sector = self.sector_mapping.get(symbol, 'Unknown')
            sector_exposure[sector] = sector_exposure.get(sector, 0) + weight
        
        risk_analysis['sector_exposure'] = {
            sector: exposure * 100 for sector, exposure in sector_exposure.items()
        }
        
        # Calculate risk scores
        risk_scores = self._calculate_risk_scores(risk_analysis)
        risk_analysis['risk_scores'] = risk_scores
        
        # Generate warnings and recommendations
        risk_analysis['warnings'] = self._generate_warnings(risk_analysis)
        risk_analysis['recommendations'] = self._generate_recommendations(risk_analysis)
        
        return risk_analysis
    
    def stress_test_portfolio(self, positions: Dict[str, Dict], 
                            scenarios: Optional[Dict] = None) -> Dict[str, Dict]:
        """Run stress tests on portfolio"""
        if scenarios is None:
            scenarios = {
                'market_crash': {'market_change': -0.20, 'vix_spike': 2.5},
                'tech_selloff': {'tech_change': -0.15, 'other_change': -0.05},
                'rate_hike': {'growth_change': -0.10, 'value_change': 0.02},
                'black_swan': {'market_change': -0.35, 'correlation': 0.95}
            }
        
        stress_results = {}
        current_value = sum([pos['shares'] * self._get_current_price(sym) 
                           for sym, pos in positions.items()])
        
        for scenario_name, scenario_params in scenarios.items():
            scenario_result = {
                'portfolio_impact': 0,
                'position_impacts': {},
                'new_portfolio_value': 0,
                'loss_amount': 0,
                'loss_percentage': 0
            }
            
            new_value = current_value
            
            # Apply scenario
            if 'market_change' in scenario_params:
                # General market decline
                for symbol, pos in positions.items():
                    position_value = pos['shares'] * self._get_current_price(symbol)
                    beta = self._get_beta(symbol)
                    impact = scenario_params['market_change'] * beta
                    new_position_value = position_value * (1 + impact)
                    scenario_result['position_impacts'][symbol] = {
                        'change_pct': impact * 100,
                        'change_dollar': new_position_value - position_value
                    }
                    new_value += (new_position_value - position_value)
            
            elif 'tech_change' in scenario_params:
                # Sector-specific scenario
                for symbol, pos in positions.items():
                    position_value = pos['shares'] * self._get_current_price(symbol)
                    if 'Technology' in self.sector_mapping.get(symbol, ''):
                        impact = scenario_params['tech_change']
                    else:
                        impact = scenario_params.get('other_change', 0)
                    
                    new_position_value = position_value * (1 + impact)
                    scenario_result['position_impacts'][symbol] = {
                        'change_pct': impact * 100,
                        'change_dollar': new_position_value - position_value
                    }
                    new_value += (new_position_value - position_value)
            
            scenario_result['new_portfolio_value'] = new_value
            scenario_result['loss_amount'] = new_value - current_value
            scenario_result['loss_percentage'] = ((new_value / current_value) - 1) * 100
            
            stress_results[scenario_name] = scenario_result
        
        return stress_results
    
    def calculate_portfolio_heat_map(self, positions: Dict[str, Dict]) -> Dict[str, float]:
        """Create a risk heat map for portfolio positions"""
        heat_map = {}
        
        for symbol, pos in positions.items():
            # Calculate multiple risk factors
            current_price = self._get_current_price(symbol)
            position_value = pos['shares'] * current_price
            
            # Risk factors (0-100 scale, higher = riskier)
            size_risk = min((position_value / self.portfolio_value) * 100 / 0.25, 100)  # >25% = 100
            
            volatility = self._get_volatility(symbol)
            volatility_risk = min(volatility * 100 / 0.40, 100)  # >40% vol = 100
            
            pnl_pct = ((current_price / pos['entry_price']) - 1) * 100
            pnl_risk = 0
            if pnl_pct < -10:
                pnl_risk = min(abs(pnl_pct) * 2, 100)  # Large losses increase risk
            
            stop_distance = ((current_price - pos.get('stop_loss', current_price * 0.95)) / current_price) * 100
            stop_risk = min(stop_distance * 5, 100)  # >20% to stop = 100
            
            # Combined risk score
            risk_score = (size_risk * 0.3 + volatility_risk * 0.3 + 
                         pnl_risk * 0.2 + stop_risk * 0.2)
            
            heat_map[symbol] = {
                'overall_risk': risk_score,
                'size_risk': size_risk,
                'volatility_risk': volatility_risk,
                'pnl_risk': pnl_risk,
                'stop_risk': stop_risk,
                'risk_level': self._get_risk_level(risk_score)
            }
        
        return heat_map
    
    def _get_current_price(self, symbol: str) -> float:
        """Get current price for symbol"""
        try:
            ticker = yf.Ticker(symbol)
            return ticker.info.get('currentPrice', ticker.info.get('regularMarketPrice', 0))
        except:
            return 0
    
    def _get_volatility(self, symbol: str, days: int = 30) -> float:
        """Calculate historical volatility"""
        try:
            data = yf.download(symbol, period=f'{days}d', progress=False)
            returns = data['Adj Close'].pct_change().dropna()
            return returns.std() * np.sqrt(252)  # Annualized
        except:
            return 0.25  # Default 25% volatility
    
    def _get_beta(self, symbol: str) -> float:
        """Get beta for symbol"""
        try:
            ticker = yf.Ticker(symbol)
            return ticker.info.get('beta', 1.0)
        except:
            return 1.0
    
    def _calculate_risk_scores(self, analysis: Dict) -> Dict[str, float]:
        """Calculate overall risk scores"""
        scores = {}
        
        # Concentration risk
        hhi = analysis['portfolio_metrics']['portfolio_concentration']
        scores['concentration_risk'] = min(hhi * 100, 100)
        
        # Correlation risk
        if 'correlation_risk' in analysis and analysis['correlation_risk']:
            avg_corr = analysis['correlation_risk']['average_correlation']
            scores['correlation_risk'] = min(abs(avg_corr) * 100, 100)
        else:
            scores['correlation_risk'] = 0
        
        # Sector risk
        max_sector = max(analysis['sector_exposure'].values()) if analysis['sector_exposure'] else 0
        scores['sector_risk'] = min(max_sector / 30, 100)  # 30% threshold
        
        # Overall risk score
        scores['overall_risk'] = (scores['concentration_risk'] * 0.3 +
                                 scores['correlation_risk'] * 0.3 +
                                 scores['sector_risk'] * 0.4)
        
        return scores
    
    def _generate_warnings(self, analysis: Dict) -> List[str]:
        """Generate risk warnings"""
        warnings = []
        
        # Position concentration warnings
        for symbol, risk in analysis['position_risks'].items():
            if risk['position_weight'] > 25:
                warnings.append(f"⚠️ {symbol} exceeds 25% of portfolio ({risk['position_weight']:.1f}%)")
        
        # Correlation warnings
        if 'correlation_risk' in analysis and analysis['correlation_risk'].get('high_correlation_pairs'):
            for pair in analysis['correlation_risk']['high_correlation_pairs']:
                warnings.append(f"⚠️ High correlation: {pair['pair']} ({pair['correlation']:.2f})")
        
        # Sector concentration warnings
        for sector, exposure in analysis['sector_exposure'].items():
            if exposure > 30:
                warnings.append(f"⚠️ {sector} sector exceeds 30% ({exposure:.1f}%)")
        
        # Large loss warnings
        for symbol, risk in analysis['position_risks'].items():
            if risk['unrealized_pnl_pct'] < -10:
                warnings.append(f"⚠️ {symbol} down {abs(risk['unrealized_pnl_pct']):.1f}% from entry")
        
        return warnings
    
    def _generate_recommendations(self, analysis: Dict) -> List[str]:
        """Generate risk management recommendations"""
        recommendations = []
        
        # Based on risk scores
        if analysis['risk_scores']['concentration_risk'] > 70:
            recommendations.append("📌 Reduce position sizes in largest holdings")
        
        if analysis['risk_scores']['correlation_risk'] > 70:
            recommendations.append("📌 Diversify into uncorrelated assets")
        
        if analysis['risk_scores']['sector_risk'] > 70:
            recommendations.append("📌 Rebalance sector allocations")
        
        # Based on individual positions
        for symbol, risk in analysis['position_risks'].items():
            if risk['unrealized_pnl_pct'] < -8:
                recommendations.append(f"📌 Review stop loss for {symbol}")
            elif risk['unrealized_pnl_pct'] > 20:
                recommendations.append(f"📌 Consider taking profits on {symbol}")
        
        return recommendations
    
    def _get_risk_level(self, score: float) -> str:
        """Convert risk score to risk level"""
        if score < 25:
            return "LOW"
        elif score < 50:
            return "MEDIUM"
        elif score < 75:
            return "HIGH"
        else:
            return "CRITICAL"
    
    def generate_risk_report(self, positions: Dict[str, Dict]) -> str:
        """Generate comprehensive risk management report"""
        report = []
        report.append("="*60)
        report.append("⚠️ RISK MANAGEMENT DASHBOARD")
        report.append("="*60)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Portfolio risk analysis
        analysis = self.analyze_portfolio_risk(positions)
        
        # Portfolio overview
        report.append("📊 PORTFOLIO OVERVIEW")
        report.append("-"*40)
        report.append(f"Total Value: ${analysis['portfolio_metrics']['total_value']:,.2f}")
        report.append(f"Position Count: {analysis['portfolio_metrics']['position_count']}")
        report.append(f"Largest Position: {analysis['portfolio_metrics']['largest_position'][0]} "
                     f"({analysis['portfolio_metrics']['largest_position'][1]*100:.1f}%)")
        
        # Risk scores
        report.append("\n🎯 RISK SCORES (0-100)")
        report.append("-"*40)
        for risk_type, score in analysis['risk_scores'].items():
            level = self._get_risk_level(score)
            report.append(f"{risk_type.replace('_', ' ').title()}: {score:.1f} [{level}]")
        
        # Position risks
        report.append("\n📈 POSITION RISK ANALYSIS")
        report.append("-"*40)
        for symbol, risk in analysis['position_risks'].items():
            report.append(f"\n{symbol}:")
            report.append(f"  Position Size: ${risk['position_value']:,.2f} ({risk['position_weight']:.1f}%)")
            report.append(f"  Unrealized P&L: ${risk['unrealized_pnl']:,.2f} ({risk['unrealized_pnl_pct']:+.1f}%)")
            report.append(f"  Risk Amount: ${risk['total_risk']:,.2f}")
        
        # VaR calculation
        var_results = self.calculate_portfolio_var(
            {sym: 1/len(positions) for sym in positions.keys()}
        )
        if var_results:
            report.append("\n💰 VALUE AT RISK (VaR)")
            report.append("-"*40)
            for var_type, var_data in var_results.items():
                report.append(f"{var_type.upper()}: {var_data['percentage']:.2f}% "
                            f"(${var_data['dollar_amount']:,.2f})")
        
        # Stress test results
        report.append("\n🔥 STRESS TEST RESULTS")
        report.append("-"*40)
        stress_results = self.stress_test_portfolio(positions)
        for scenario, result in stress_results.items():
            report.append(f"\n{scenario.replace('_', ' ').title()}:")
            report.append(f"  Portfolio Impact: {result['loss_percentage']:+.1f}%")
            report.append(f"  Dollar Impact: ${result['loss_amount']:,.2f}")
        
        # Heat map
        heat_map = self.calculate_portfolio_heat_map(positions)
        report.append("\n🔥 RISK HEAT MAP")
        report.append("-"*40)
        sorted_positions = sorted(heat_map.items(), key=lambda x: x[1]['overall_risk'], reverse=True)
        for symbol, heat in sorted_positions:
            risk_bar = "█" * int(heat['overall_risk'] / 10)
            report.append(f"{symbol}: {risk_bar} {heat['overall_risk']:.0f} [{heat['risk_level']}]")
        
        # Warnings
        if analysis['warnings']:
            report.append("\n⚠️ WARNINGS")
            report.append("-"*40)
            for warning in analysis['warnings']:
                report.append(warning)
        
        # Recommendations
        if analysis['recommendations']:
            report.append("\n💡 RECOMMENDATIONS")
            report.append("-"*40)
            for rec in analysis['recommendations']:
                report.append(rec)
        
        return "\n".join(report)


def main():
    """Test the risk management dashboard"""
    # Example portfolio
    test_positions = {
        'AAPL': {'shares': 100, 'entry_price': 205.00, 'stop_loss': 195.00},
        'GOOGL': {'shares': 50, 'entry_price': 180.00, 'stop_loss': 171.00},
        'MSFT': {'shares': 40, 'entry_price': 500.00, 'stop_loss': 475.00},
        'TSLA': {'shares': 30, 'entry_price': 320.00, 'stop_loss': 290.00}
    }
    
    dashboard = RiskManagementDashboard(portfolio_value=100000)
    report = dashboard.generate_risk_report(test_positions)
    print(report)
    
    # Save report
    with open("advanced_features/risk_management/risk_report.txt", "w") as f:
        f.write(report)
    
    print("\n✅ Risk management report saved")


if __name__ == "__main__":
    main()