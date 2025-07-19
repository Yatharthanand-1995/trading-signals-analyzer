#!/usr/bin/env python3
"""
Economic Calendar Integration
Tracks major economic events and their potential market impact
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import json
import yfinance as yf
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EconomicCalendar:
    def __init__(self):
        self.events_file = "advanced_features/economic_calendar/economic_events.json"
        self.impact_weights = {
            'high': 1.0,
            'medium': 0.6,
            'low': 0.3
        }
        
        # Major economic events and their typical market impact
        self.event_types = {
            'fomc_meeting': {
                'name': 'Federal Reserve FOMC Meeting',
                'impact': 'high',
                'frequency': 'monthly',
                'market_effect': 'volatility_spike',
                'typical_move': 0.02  # 2% typical market move
            },
            'cpi_release': {
                'name': 'Consumer Price Index',
                'impact': 'high',
                'frequency': 'monthly',
                'market_effect': 'trend_catalyst',
                'typical_move': 0.015
            },
            'nfp_release': {
                'name': 'Non-Farm Payrolls',
                'impact': 'high',
                'frequency': 'monthly',
                'market_effect': 'volatility_spike',
                'typical_move': 0.012
            },
            'gdp_release': {
                'name': 'GDP Report',
                'impact': 'medium',
                'frequency': 'quarterly',
                'market_effect': 'trend_confirmation',
                'typical_move': 0.01
            },
            'earnings_season': {
                'name': 'Earnings Season',
                'impact': 'high',
                'frequency': 'quarterly',
                'market_effect': 'sector_rotation',
                'typical_move': 0.02
            }
        }
        
        # 2024-2025 Economic Calendar (example dates)
        self.scheduled_events = {
            'fomc_meetings': [
                '2024-12-18', '2025-01-29', '2025-03-19', '2025-05-07',
                '2025-06-18', '2025-07-30', '2025-09-17', '2025-11-05', '2025-12-17'
            ],
            'cpi_releases': [
                '2024-12-11', '2025-01-14', '2025-02-13', '2025-03-12',
                '2025-04-10', '2025-05-13', '2025-06-11', '2025-07-11',
                '2025-08-13', '2025-09-11', '2025-10-10', '2025-11-13', '2025-12-10'
            ],
            'nfp_releases': [
                '2024-12-06', '2025-01-10', '2025-02-07', '2025-03-07',
                '2025-04-04', '2025-05-02', '2025-06-06', '2025-07-03',
                '2025-08-01', '2025-09-05', '2025-10-03', '2025-11-07', '2025-12-05'
            ],
            'gdp_releases': [
                '2024-12-19', '2025-01-30', '2025-04-25', '2025-07-25', '2025-10-30'
            ]
        }
    
    def get_upcoming_events(self, days_ahead: int = 7) -> List[Dict]:
        """Get economic events in the next N days"""
        upcoming_events = []
        today = datetime.now().date()
        end_date = today + timedelta(days=days_ahead)
        
        for event_type, dates in self.scheduled_events.items():
            event_info = None
            if 'fomc' in event_type:
                event_info = self.event_types['fomc_meeting']
            elif 'cpi' in event_type:
                event_info = self.event_types['cpi_release']
            elif 'nfp' in event_type:
                event_info = self.event_types['nfp_release']
            elif 'gdp' in event_type:
                event_info = self.event_types['gdp_release']
            
            if event_info:
                for date_str in dates:
                    event_date = datetime.strptime(date_str, '%Y-%m-%d').date()
                    if today <= event_date <= end_date:
                        days_until = (event_date - today).days
                        upcoming_events.append({
                            'date': date_str,
                            'days_until': days_until,
                            'event': event_info['name'],
                            'impact': event_info['impact'],
                            'typical_move': event_info['typical_move'],
                            'market_effect': event_info['market_effect']
                        })
        
        return sorted(upcoming_events, key=lambda x: x['days_until'])
    
    def get_earnings_calendar(self, symbols: List[str]) -> Dict[str, Dict]:
        """Fetch earnings dates for given symbols"""
        earnings_data = {}
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                
                # Get next earnings date if available
                if 'earningsDate' in info and info['earningsDate']:
                    earnings_date = datetime.fromtimestamp(info['earningsDate'][0])
                    days_until = (earnings_date.date() - datetime.now().date()).days
                    
                    earnings_data[symbol] = {
                        'date': earnings_date.strftime('%Y-%m-%d'),
                        'days_until': days_until,
                        'previous_eps': info.get('trailingEps', 'N/A'),
                        'expected_eps': info.get('forwardEps', 'N/A'),
                        'impact': 'high' if days_until <= 7 else 'medium'
                    }
                else:
                    earnings_data[symbol] = {
                        'date': 'Not scheduled',
                        'days_until': None,
                        'impact': 'low'
                    }
                    
            except Exception as e:
                logger.warning(f"Could not fetch earnings data for {symbol}: {e}")
                earnings_data[symbol] = {'date': 'Error', 'impact': 'unknown'}
        
        return earnings_data
    
    def calculate_event_adjusted_position_size(self, base_position_size: float, 
                                             symbol: str, days_ahead: int = 7) -> Tuple[float, str]:
        """Adjust position size based on upcoming events"""
        # Get upcoming economic events
        upcoming_events = self.get_upcoming_events(days_ahead)
        
        # Get earnings calendar
        earnings = self.get_earnings_calendar([symbol])
        
        # Calculate risk multiplier based on events
        risk_multiplier = 1.0
        reasons = []
        
        # Check for economic events
        for event in upcoming_events:
            if event['days_until'] <= 3:  # Event within 3 days
                impact_weight = self.impact_weights[event['impact']]
                risk_reduction = 0.5 * impact_weight  # Reduce position by up to 50%
                risk_multiplier *= (1 - risk_reduction)
                reasons.append(f"{event['event']} in {event['days_until']} days")
        
        # Check for earnings
        if symbol in earnings and earnings[symbol]['days_until'] is not None:
            days_to_earnings = earnings[symbol]['days_until']
            if 0 <= days_to_earnings <= 5:  # Earnings within 5 days
                risk_multiplier *= 0.5  # Cut position in half
                reasons.append(f"Earnings in {days_to_earnings} days")
            elif 6 <= days_to_earnings <= 10:
                risk_multiplier *= 0.75
                reasons.append(f"Earnings approaching ({days_to_earnings} days)")
        
        adjusted_position = base_position_size * risk_multiplier
        reason_str = " | ".join(reasons) if reasons else "No major events"
        
        return adjusted_position, reason_str
    
    def get_market_regime(self) -> Dict[str, any]:
        """Determine current market regime based on economic calendar"""
        upcoming_events = self.get_upcoming_events(30)
        
        # Count high-impact events
        high_impact_count = sum(1 for event in upcoming_events if event['impact'] == 'high')
        
        # Determine regime
        if high_impact_count >= 3:
            regime = 'high_event_density'
            volatility_expectation = 'elevated'
            recommended_stance = 'defensive'
        elif high_impact_count >= 1:
            regime = 'moderate_event_density'
            volatility_expectation = 'normal'
            recommended_stance = 'balanced'
        else:
            regime = 'low_event_density'
            volatility_expectation = 'subdued'
            recommended_stance = 'opportunistic'
        
        return {
            'regime': regime,
            'volatility_expectation': volatility_expectation,
            'recommended_stance': recommended_stance,
            'high_impact_events_30d': high_impact_count,
            'next_major_event': upcoming_events[0] if upcoming_events else None
        }
    
    def generate_event_report(self) -> str:
        """Generate a comprehensive economic event report"""
        report = []
        report.append("="*60)
        report.append("📅 ECONOMIC CALENDAR REPORT")
        report.append("="*60)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Market regime
        regime = self.get_market_regime()
        report.append("📊 MARKET REGIME ANALYSIS")
        report.append("-"*40)
        report.append(f"Current Regime: {regime['regime'].replace('_', ' ').title()}")
        report.append(f"Volatility Expectation: {regime['volatility_expectation'].title()}")
        report.append(f"Recommended Stance: {regime['recommended_stance'].title()}")
        report.append(f"High Impact Events (30d): {regime['high_impact_events_30d']}\n")
        
        # Upcoming events
        events_7d = self.get_upcoming_events(7)
        events_30d = self.get_upcoming_events(30)
        
        report.append("📌 UPCOMING EVENTS (Next 7 Days)")
        report.append("-"*40)
        if events_7d:
            for event in events_7d:
                report.append(f"• {event['date']} ({event['days_until']}d): {event['event']}")
                report.append(f"  Impact: {event['impact'].upper()} | "
                            f"Expected Move: {event['typical_move']*100:.1f}%")
        else:
            report.append("No major events in the next 7 days")
        
        report.append("\n📈 POSITION SIZING RECOMMENDATIONS")
        report.append("-"*40)
        
        # Example position adjustments
        test_symbols = ['AAPL', 'GOOGL', 'TSLA']
        base_size = 10000
        
        for symbol in test_symbols:
            adjusted_size, reason = self.calculate_event_adjusted_position_size(base_size, symbol)
            adjustment_pct = ((adjusted_size / base_size) - 1) * 100
            report.append(f"{symbol}: ${adjusted_size:,.0f} ({adjustment_pct:+.0f}%)")
            report.append(f"  Reason: {reason}")
        
        report.append("\n📊 TRADING GUIDELINES BY EVENT")
        report.append("-"*40)
        report.append("FOMC Meetings: Reduce positions 50% three days prior")
        report.append("CPI/NFP Release: Reduce positions 30% two days prior")
        report.append("Earnings: Reduce positions 50-75% based on proximity")
        report.append("GDP Release: Monitor for trend confirmation signals")
        
        return "\n".join(report)


def main():
    """Test the economic calendar integration"""
    calendar = EconomicCalendar()
    
    # Generate and print report
    report = calendar.generate_event_report()
    print(report)
    
    # Save report
    with open("advanced_features/economic_calendar/event_report.txt", "w") as f:
        f.write(report)
    
    print("\n✅ Economic calendar report saved to event_report.txt")


if __name__ == "__main__":
    main()