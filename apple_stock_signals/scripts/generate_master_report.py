#!/usr/bin/env python3
"""
Generate Master Report Script
Creates comprehensive trading analysis reports
"""

import sys
import os
import json
import pandas as pd
from datetime import datetime, timedelta
import glob

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

def load_latest_results():
    """Load the most recent analysis results"""
    results = {
        'analysis': None,
        'backtest': None,
        'signals': None,
        'timestamp': None
    }
    
    # Find latest analysis output
    analysis_files = glob.glob(os.path.join(parent_dir, 'outputs', '*.json'))
    if analysis_files:
        latest_analysis = max(analysis_files, key=os.path.getctime)
        with open(latest_analysis, 'r') as f:
            results['analysis'] = json.load(f)
    
    # Find latest backtest results
    backtest_dirs = glob.glob(os.path.join(parent_dir, 'backtest_results', '*'))
    if backtest_dirs:
        latest_dir = max(backtest_dirs, key=os.path.getctime)
        backtest_files = glob.glob(os.path.join(latest_dir, '*.json'))
        if backtest_files:
            latest_backtest = max(backtest_files, key=os.path.getctime)
            with open(latest_backtest, 'r') as f:
                results['backtest'] = json.load(f)
    
    results['timestamp'] = datetime.now()
    return results

def generate_summary_report(results):
    """Generate a summary report from all results"""
    report = {
        'timestamp': datetime.now().isoformat(),
        'summary': {
            'total_stocks_analyzed': 0,
            'buy_signals': 0,
            'sell_signals': 0,
            'neutral_signals': 0,
            'average_score': 0,
            'top_opportunities': [],
            'risk_warnings': []
        },
        'details': {},
        'performance': {},
        'recommendations': []
    }
    
    # Process analysis results
    if results['analysis']:
        if isinstance(results['analysis'], list):
            stocks = results['analysis']
        elif 'stocks' in results['analysis']:
            stocks = results['analysis']['stocks']
        else:
            stocks = [results['analysis']]
        
        total_score = 0
        for stock in stocks:
            if isinstance(stock, dict) and 'symbol' in stock:
                symbol = stock['symbol']
                report['details'][symbol] = {
                    'signal': stock.get('signal', 'N/A'),
                    'score': stock.get('score', 0),
                    'current_price': stock.get('current_price', 0)
                }
                
                # Count signals
                signal = stock.get('signal', 'NEUTRAL')
                if 'BUY' in signal:
                    report['summary']['buy_signals'] += 1
                elif 'SELL' in signal:
                    report['summary']['sell_signals'] += 1
                else:
                    report['summary']['neutral_signals'] += 1
                
                total_score += stock.get('score', 0)
                report['summary']['total_stocks_analyzed'] += 1
                
                # Top opportunities
                if stock.get('score', 0) >= 70 and 'BUY' in signal:
                    report['summary']['top_opportunities'].append({
                        'symbol': symbol,
                        'score': stock.get('score', 0),
                        'signal': signal
                    })
        
        # Calculate average score
        if report['summary']['total_stocks_analyzed'] > 0:
            report['summary']['average_score'] = total_score / report['summary']['total_stocks_analyzed']
        
        # Sort top opportunities
        report['summary']['top_opportunities'].sort(key=lambda x: x['score'], reverse=True)
        report['summary']['top_opportunities'] = report['summary']['top_opportunities'][:5]
    
    # Process backtest results
    if results['backtest']:
        if 'summary' in results['backtest']:
            report['performance'] = results['backtest']['summary']
        elif 'results' in results['backtest']:
            # Calculate performance metrics
            total_return = 0
            winning_trades = 0
            total_trades = 0
            
            for symbol, data in results['backtest']['results'].items():
                if 'return' in data:
                    total_return += data['return']
                    total_trades += 1
                    if data['return'] > 0:
                        winning_trades += 1
            
            if total_trades > 0:
                report['performance'] = {
                    'average_return': total_return / total_trades,
                    'win_rate': (winning_trades / total_trades) * 100,
                    'total_trades': total_trades
                }
    
    # Generate recommendations
    if report['summary']['buy_signals'] > 0:
        report['recommendations'].append(
            f"Found {report['summary']['buy_signals']} buy opportunities. "
            f"Top pick: {report['summary']['top_opportunities'][0]['symbol'] if report['summary']['top_opportunities'] else 'N/A'}"
        )
    
    if report['summary']['average_score'] < 50:
        report['recommendations'].append(
            "Market conditions appear weak. Consider reducing position sizes or waiting for better setups."
        )
    
    return report

def save_report(report):
    """Save the report to file"""
    # Create reports directory if it doesn't exist
    reports_dir = os.path.join(parent_dir, 'reports')
    os.makedirs(reports_dir, exist_ok=True)
    
    # Save as JSON
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    json_path = os.path.join(reports_dir, f'master_report_{timestamp}.json')
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Create human-readable summary
    summary_path = os.path.join(reports_dir, f'summary_{timestamp}.txt')
    with open(summary_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("📊 TRADING ANALYSIS MASTER REPORT\n")
        f.write("="*60 + "\n")
        f.write(f"Generated: {report['timestamp']}\n")
        f.write("="*60 + "\n\n")
        
        # Summary section
        f.write("📈 SUMMARY\n")
        f.write("-"*40 + "\n")
        f.write(f"Stocks Analyzed: {report['summary']['total_stocks_analyzed']}\n")
        f.write(f"Buy Signals: {report['summary']['buy_signals']}\n")
        f.write(f"Sell Signals: {report['summary']['sell_signals']}\n")
        f.write(f"Neutral: {report['summary']['neutral_signals']}\n")
        f.write(f"Average Score: {report['summary']['average_score']:.1f}/100\n\n")
        
        # Top opportunities
        if report['summary']['top_opportunities']:
            f.write("🎯 TOP OPPORTUNITIES\n")
            f.write("-"*40 + "\n")
            for i, opp in enumerate(report['summary']['top_opportunities'], 1):
                f.write(f"{i}. {opp['symbol']} - Score: {opp['score']} ({opp['signal']})\n")
            f.write("\n")
        
        # Performance
        if report['performance']:
            f.write("📊 BACKTEST PERFORMANCE\n")
            f.write("-"*40 + "\n")
            for key, value in report['performance'].items():
                f.write(f"{key.replace('_', ' ').title()}: {value}\n")
            f.write("\n")
        
        # Recommendations
        if report['recommendations']:
            f.write("💡 RECOMMENDATIONS\n")
            f.write("-"*40 + "\n")
            for rec in report['recommendations']:
                f.write(f"• {rec}\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write("⚠️ Disclaimer: For educational purposes only. Not financial advice.\n")
        f.write("="*60 + "\n")
    
    print(f"✅ Report saved to:")
    print(f"   JSON: {json_path}")
    print(f"   Summary: {summary_path}")
    
    return json_path, summary_path

def main():
    """Main function"""
    print("📊 Generating Master Report...")
    print("="*60)
    
    # Load results
    results = load_latest_results()
    
    if not results['analysis'] and not results['backtest']:
        print("❌ No recent analysis results found!")
        print("Please run analysis first using: ./run_analysis.sh")
        return
    
    # Generate report
    report = generate_summary_report(results)
    
    # Save report
    json_path, summary_path = save_report(report)
    
    # Display summary
    print("\n📈 Report Summary:")
    print(f"   Total Stocks: {report['summary']['total_stocks_analyzed']}")
    print(f"   Buy Signals: {report['summary']['buy_signals']}")
    print(f"   Average Score: {report['summary']['average_score']:.1f}/100")
    
    if report['summary']['top_opportunities']:
        print("\n🎯 Top Opportunities:")
        for opp in report['summary']['top_opportunities'][:3]:
            print(f"   • {opp['symbol']} (Score: {opp['score']})")
    
    print("\n✅ Report generation complete!")

if __name__ == "__main__":
    main()