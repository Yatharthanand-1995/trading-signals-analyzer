#!/usr/bin/env python3
"""
Integrated Advanced Trading System
Combines Economic Calendar, Trade Journal, and Risk Management with existing trading signals
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
import json
import pandas as pd
from typing import Dict, List, Optional, Tuple

# Import existing modules
from core_scripts.config import STOCKS, RISK_PER_TRADE
from data_modules.signal_generator import AppleSignalGenerator as SignalGenerator
from data_modules.technical_analyzer import AppleTechnicalAnalyzer as TechnicalAnalyzer

# Import advanced features
from advanced_features.economic_calendar.economic_events import EconomicCalendar
from advanced_features.trade_journal.trade_journal import TradeJournal, Trade
from advanced_features.risk_management.risk_dashboard import RiskManagementDashboard

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IntegratedTradingSystem:
    def __init__(self, portfolio_value: float = 100000):
        """Initialize the integrated trading system"""
        self.portfolio_value = portfolio_value
        
        # Initialize components
        self.signal_generator = SignalGenerator()
        self.technical_analyzer = TechnicalAnalyzer()
        self.economic_calendar = EconomicCalendar()
        self.trade_journal = TradeJournal()
        self.risk_dashboard = RiskManagementDashboard(portfolio_value)
        
        # Track current positions
        self.current_positions = {}
        self.load_positions()
    
    def load_positions(self):
        """Load current positions from file"""
        positions_file = "advanced_features/current_positions.json"
        if os.path.exists(positions_file):
            with open(positions_file, 'r') as f:
                self.current_positions = json.load(f)
        else:
            self.current_positions = {}
    
    def save_positions(self):
        """Save current positions to file"""
        positions_file = "advanced_features/current_positions.json"
        os.makedirs(os.path.dirname(positions_file), exist_ok=True)
        with open(positions_file, 'w') as f:
            json.dump(self.current_positions, f, indent=2)
    
    def analyze_trading_opportunity(self, symbol: str) -> Dict:
        """Comprehensive analysis combining all systems"""
        analysis = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'signal_analysis': {},
            'economic_impact': {},
            'risk_assessment': {},
            'position_sizing': {},
            'final_recommendation': {}
        }
        
        # 1. Get base trading signal
        try:
            signal_data = self.signal_generator.generate_signal(symbol)
            analysis['signal_analysis'] = {
                'signal': signal_data['signal'],
                'score': signal_data['score'],
                'entry_price': signal_data.get('entry_price', 0),
                'stop_loss': signal_data.get('stop_loss', 0),
                'take_profit': signal_data.get('take_profit', 0)
            }
        except Exception as e:
            logger.error(f"Signal generation failed for {symbol}: {e}")
            analysis['signal_analysis'] = {'error': str(e)}
            return analysis
        
        # 2. Check economic calendar impact
        base_position_size = self.portfolio_value * RISK_PER_TRADE
        adjusted_size, event_reason = self.economic_calendar.calculate_event_adjusted_position_size(
            base_position_size, symbol
        )
        
        analysis['economic_impact'] = {
            'adjusted_position_size': adjusted_size,
            'size_reduction': (1 - adjusted_size / base_position_size) * 100,
            'reason': event_reason,
            'upcoming_events': self.economic_calendar.get_upcoming_events(7)
        }
        
        # 3. Risk assessment
        # Include current positions in risk calculation
        test_positions = self.current_positions.copy()
        if signal_data['signal'] in ['BUY', 'STRONG_BUY']:
            # Simulate adding this position
            shares = int(adjusted_size / signal_data['entry_price'])
            test_positions[symbol] = {
                'shares': shares,
                'entry_price': signal_data['entry_price'],
                'stop_loss': signal_data['stop_loss']
            }
        
        portfolio_risk = self.risk_dashboard.analyze_portfolio_risk(test_positions)
        
        analysis['risk_assessment'] = {
            'portfolio_risk_score': portfolio_risk['risk_scores']['overall_risk'],
            'position_correlation': portfolio_risk['correlation_risk'].get('average_correlation', 0) if 'correlation_risk' in portfolio_risk else 0,
            'sector_exposure': portfolio_risk['sector_exposure'],
            'warnings': portfolio_risk['warnings']
        }
        
        # 4. Final position sizing
        risk_multiplier = 1.0
        if portfolio_risk['risk_scores']['overall_risk'] > 70:
            risk_multiplier *= 0.7  # Reduce size if portfolio risk is high
        
        final_position_size = adjusted_size * risk_multiplier
        shares_to_buy = int(final_position_size / signal_data['entry_price']) if signal_data['entry_price'] > 0 else 0
        
        analysis['position_sizing'] = {
            'base_size': base_position_size,
            'event_adjusted_size': adjusted_size,
            'risk_adjusted_size': final_position_size,
            'shares_to_buy': shares_to_buy,
            'position_value': shares_to_buy * signal_data['entry_price'] if signal_data['entry_price'] > 0 else 0
        }
        
        # 5. Generate final recommendation
        should_trade = (
            signal_data['signal'] in ['BUY', 'STRONG_BUY'] and
            signal_data['score'] >= 60 and
            portfolio_risk['risk_scores']['overall_risk'] < 80 and
            shares_to_buy > 0
        )
        
        analysis['final_recommendation'] = {
            'action': 'EXECUTE' if should_trade else 'PASS',
            'confidence': signal_data['score'] * (1 - portfolio_risk['risk_scores']['overall_risk'] / 100),
            'reasons': []
        }
        
        # Add reasoning
        if should_trade:
            analysis['final_recommendation']['reasons'].append(f"Strong signal: {signal_data['signal']} ({signal_data['score']})")
            if event_reason != "No major events":
                analysis['final_recommendation']['reasons'].append(f"Position sized for: {event_reason}")
        else:
            if signal_data['signal'] not in ['BUY', 'STRONG_BUY']:
                analysis['final_recommendation']['reasons'].append(f"Weak signal: {signal_data['signal']}")
            if portfolio_risk['risk_scores']['overall_risk'] >= 80:
                analysis['final_recommendation']['reasons'].append("Portfolio risk too high")
            if shares_to_buy == 0:
                analysis['final_recommendation']['reasons'].append("Position size too small")
        
        return analysis
    
    def execute_trade(self, analysis: Dict) -> Optional[str]:
        """Execute trade and log to journal"""
        if analysis['final_recommendation']['action'] != 'EXECUTE':
            return None
        
        symbol = analysis['symbol']
        signal_data = analysis['signal_analysis']
        position_data = analysis['position_sizing']
        
        # Create trade entry
        trade = Trade(
            trade_id=f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            symbol=symbol,
            entry_date=datetime.now().strftime('%Y-%m-%d'),
            entry_price=signal_data['entry_price'],
            exit_date=None,
            exit_price=None,
            position_size=position_data['shares_to_buy'],
            trade_type='long',
            entry_signal=signal_data['signal'],
            entry_score=signal_data['score'],
            exit_reason=None,
            stop_loss=signal_data['stop_loss'],
            take_profit=signal_data['take_profit'],
            commission=position_data['shares_to_buy'] * signal_data['entry_price'] * 0.001,  # 0.1% commission
            market_conditions={
                'economic_events': analysis['economic_impact']['reason'],
                'portfolio_risk': analysis['risk_assessment']['portfolio_risk_score']
            },
            notes=f"Automated entry. {'; '.join(analysis['final_recommendation']['reasons'])}"
        )
        
        # Log to journal
        trade_id = self.trade_journal.log_trade_entry(trade)
        
        # Update positions
        self.current_positions[symbol] = {
            'trade_id': trade_id,
            'shares': position_data['shares_to_buy'],
            'entry_price': signal_data['entry_price'],
            'stop_loss': signal_data['stop_loss'],
            'take_profit': signal_data['take_profit'],
            'entry_date': datetime.now().strftime('%Y-%m-%d')
        }
        self.save_positions()
        
        logger.info(f"Trade executed: {trade_id}")
        return trade_id
    
    def check_exit_conditions(self) -> List[Dict]:
        """Check all positions for exit conditions"""
        exit_signals = []
        
        for symbol, position in self.current_positions.items():
            try:
                # Get current price
                current_price = self._get_current_price(symbol)
                
                # Check exit conditions
                exit_reason = None
                should_exit = False
                
                # Stop loss check
                if current_price <= position['stop_loss']:
                    exit_reason = "stop_loss_hit"
                    should_exit = True
                
                # Take profit check
                elif current_price >= position['take_profit']:
                    exit_reason = "take_profit_hit"
                    should_exit = True
                
                # Check for signal reversal
                else:
                    current_signal = self.signal_generator.generate_signal(symbol)
                    if current_signal['signal'] in ['SELL', 'STRONG_SELL']:
                        exit_reason = "signal_reversal"
                        should_exit = True
                
                # Time-based exit (optional)
                entry_date = datetime.strptime(position['entry_date'], '%Y-%m-%d')
                days_held = (datetime.now() - entry_date).days
                if days_held > 15 and not should_exit:  # Max 15 days for swing trades
                    exit_reason = "max_holding_period"
                    should_exit = True
                
                if should_exit:
                    exit_signals.append({
                        'symbol': symbol,
                        'trade_id': position['trade_id'],
                        'exit_price': current_price,
                        'exit_reason': exit_reason,
                        'days_held': days_held
                    })
                
            except Exception as e:
                logger.error(f"Error checking exit for {symbol}: {e}")
        
        return exit_signals
    
    def close_position(self, exit_signal: Dict) -> Dict:
        """Close a position and update journal"""
        symbol = exit_signal['symbol']
        position = self.current_positions[symbol]
        
        # Update trade journal
        result = self.trade_journal.update_trade_exit(
            trade_id=position['trade_id'],
            exit_date=datetime.now().strftime('%Y-%m-%d'),
            exit_price=exit_signal['exit_price'],
            exit_reason=exit_signal['exit_reason']
        )
        
        # Remove from current positions
        del self.current_positions[symbol]
        self.save_positions()
        
        logger.info(f"Position closed: {symbol} - {exit_signal['exit_reason']}")
        return result
    
    def _get_current_price(self, symbol: str) -> float:
        """Get current price for symbol"""
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            return ticker.info.get('currentPrice', ticker.info.get('regularMarketPrice', 0))
        except:
            return 0
    
    def generate_daily_report(self) -> str:
        """Generate comprehensive daily trading report"""
        report = []
        report.append("="*80)
        report.append("📊 INTEGRATED TRADING SYSTEM - DAILY REPORT")
        report.append("="*80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Current positions summary
        report.append("📈 CURRENT POSITIONS")
        report.append("-"*60)
        if self.current_positions:
            total_value = 0
            for symbol, pos in self.current_positions.items():
                current_price = self._get_current_price(symbol)
                position_value = pos['shares'] * current_price
                pnl = (current_price - pos['entry_price']) * pos['shares']
                pnl_pct = ((current_price / pos['entry_price']) - 1) * 100
                total_value += position_value
                
                report.append(f"\n{symbol}:")
                report.append(f"  Shares: {pos['shares']} @ ${pos['entry_price']:.2f}")
                report.append(f"  Current: ${current_price:.2f}")
                report.append(f"  P&L: ${pnl:,.2f} ({pnl_pct:+.1f}%)")
                report.append(f"  Value: ${position_value:,.2f}")
            
            report.append(f"\nTotal Portfolio Value: ${total_value:,.2f}")
        else:
            report.append("No open positions")
        
        # Trading opportunities
        report.append("\n\n🎯 TODAY'S TRADING OPPORTUNITIES")
        report.append("-"*60)
        
        for symbol in STOCKS:
            if symbol not in self.current_positions:  # Only analyze if not already in position
                analysis = self.analyze_trading_opportunity(symbol)
                
                report.append(f"\n{symbol}: {analysis['signal_analysis'].get('signal', 'N/A')} "
                            f"(Score: {analysis['signal_analysis'].get('score', 0):.1f})")
                report.append(f"  Recommendation: {analysis['final_recommendation']['action']}")
                
                if analysis['final_recommendation']['reasons']:
                    report.append(f"  Reasons: {'; '.join(analysis['final_recommendation']['reasons'])}")
                
                if analysis['final_recommendation']['action'] == 'EXECUTE':
                    report.append(f"  Position Size: ${analysis['position_sizing']['position_value']:,.2f} "
                                f"({analysis['position_sizing']['shares_to_buy']} shares)")
        
        # Exit signals
        report.append("\n\n🚪 EXIT SIGNALS")
        report.append("-"*60)
        exit_signals = self.check_exit_conditions()
        if exit_signals:
            for signal in exit_signals:
                report.append(f"{signal['symbol']}: EXIT - {signal['exit_reason']} @ ${signal['exit_price']:.2f}")
        else:
            report.append("No exit signals")
        
        # Economic calendar
        report.append("\n\n📅 ECONOMIC CALENDAR (Next 7 Days)")
        report.append("-"*60)
        events = self.economic_calendar.get_upcoming_events(7)
        if events:
            for event in events[:5]:  # Top 5 events
                report.append(f"{event['date']} ({event['days_until']}d): {event['event']} [{event['impact'].upper()}]")
        else:
            report.append("No major events")
        
        # Risk summary from risk dashboard
        if self.current_positions:
            risk_analysis = self.risk_dashboard.analyze_portfolio_risk(self.current_positions)
            report.append("\n\n⚠️ RISK SUMMARY")
            report.append("-"*60)
            report.append(f"Portfolio Risk Score: {risk_analysis['risk_scores']['overall_risk']:.1f}/100")
            
            if risk_analysis['warnings']:
                report.append("\nWarnings:")
                for warning in risk_analysis['warnings'][:3]:  # Top 3 warnings
                    report.append(f"  {warning}")
        
        # Performance summary from journal
        perf_metrics = self.trade_journal.calculate_performance_metrics(30)
        if 'error' not in perf_metrics:
            report.append("\n\n📊 30-DAY PERFORMANCE")
            report.append("-"*60)
            report.append(f"Win Rate: {perf_metrics['win_rate']:.1f}%")
            report.append(f"Profit Factor: {perf_metrics['profit_factor']:.2f}")
            report.append(f"Total P&L: ${perf_metrics['total_pnl']:,.2f}")
            report.append(f"Expectancy: ${perf_metrics['expectancy']:.2f}")
        
        report.append("\n" + "="*80)
        
        return "\n".join(report)


def main():
    """Run the integrated trading system"""
    system = IntegratedTradingSystem(portfolio_value=100000)
    
    # Generate daily report
    report = system.generate_daily_report()
    print(report)
    
    # Save report
    os.makedirs("advanced_features/daily_reports", exist_ok=True)
    report_file = f"advanced_features/daily_reports/integrated_report_{datetime.now().strftime('%Y%m%d')}.txt"
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"\n✅ Integrated trading report saved to {report_file}")
    
    # Check for trading opportunities and execute if appropriate
    print("\n🔍 Analyzing trading opportunities...")
    for symbol in STOCKS:
        if symbol not in system.current_positions:
            analysis = system.analyze_trading_opportunity(symbol)
            if analysis['final_recommendation']['action'] == 'EXECUTE':
                print(f"\n✅ Executing trade for {symbol}")
                trade_id = system.execute_trade(analysis)
                print(f"Trade ID: {trade_id}")
    
    # Check for exit signals
    print("\n🔍 Checking exit conditions...")
    exit_signals = system.check_exit_conditions()
    for signal in exit_signals:
        print(f"\n🚪 Closing position: {signal['symbol']} - {signal['exit_reason']}")
        result = system.close_position(signal)
        print(f"P&L: ${result['pnl']:,.2f} ({result['pnl_percent']:.1f}%)")


if __name__ == "__main__":
    main()