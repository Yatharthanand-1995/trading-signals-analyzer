#!/usr/bin/env python3
"""
Trade Journal & Analytics System
Automated trade logging, pattern analysis, and performance tracking
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
from typing import Dict, List, Optional, Tuple
import sqlite3
from dataclasses import dataclass, asdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Trade:
    """Trade data structure"""
    trade_id: str
    symbol: str
    entry_date: str
    entry_price: float
    exit_date: Optional[str]
    exit_price: Optional[float]
    position_size: int
    trade_type: str  # 'long' or 'short'
    entry_signal: str
    entry_score: float
    exit_reason: Optional[str]
    stop_loss: float
    take_profit: float
    commission: float = 0.0
    pnl: Optional[float] = None
    pnl_percent: Optional[float] = None
    holding_days: Optional[int] = None
    max_drawdown: Optional[float] = None
    notes: Optional[str] = None
    market_conditions: Optional[Dict] = None
    mistakes: Optional[List[str]] = None
    lessons: Optional[List[str]] = None

class TradeJournal:
    def __init__(self, db_path: str = "advanced_features/trade_journal/trades.db"):
        self.db_path = db_path
        self.init_database()
        
        # Common trading mistakes to track
        self.mistake_patterns = {
            'early_exit': "Exited position too early before target",
            'no_stop_loss': "Failed to set or honor stop loss",
            'revenge_trade': "Entered trade emotionally after loss",
            'oversize_position': "Position size too large for account",
            'fomo_entry': "Entered due to fear of missing out",
            'ignored_signal': "Traded against system signal",
            'held_too_long': "Held losing position hoping for reversal"
        }
        
        # Performance metrics to track
        self.performance_metrics = [
            'win_rate', 'avg_win', 'avg_loss', 'profit_factor',
            'sharpe_ratio', 'max_drawdown', 'expectancy', 'avg_hold_time'
        ]
    
    def init_database(self):
        """Initialize SQLite database for trade storage"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create trades table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                trade_id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                entry_date TEXT NOT NULL,
                entry_price REAL NOT NULL,
                exit_date TEXT,
                exit_price REAL,
                position_size INTEGER NOT NULL,
                trade_type TEXT NOT NULL,
                entry_signal TEXT NOT NULL,
                entry_score REAL NOT NULL,
                exit_reason TEXT,
                stop_loss REAL NOT NULL,
                take_profit REAL NOT NULL,
                commission REAL DEFAULT 0,
                pnl REAL,
                pnl_percent REAL,
                holding_days INTEGER,
                max_drawdown REAL,
                notes TEXT,
                market_conditions TEXT,
                mistakes TEXT,
                lessons TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create performance metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS daily_performance (
                date TEXT PRIMARY KEY,
                total_pnl REAL,
                win_rate REAL,
                trades_count INTEGER,
                winning_trades INTEGER,
                losing_trades INTEGER,
                largest_win REAL,
                largest_loss REAL,
                notes TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def log_trade_entry(self, trade: Trade) -> str:
        """Log a new trade entry"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        trade_dict = asdict(trade)
        trade_dict['market_conditions'] = json.dumps(trade_dict.get('market_conditions', {}))
        trade_dict['mistakes'] = json.dumps(trade_dict.get('mistakes', []))
        trade_dict['lessons'] = json.dumps(trade_dict.get('lessons', []))
        
        columns = ', '.join(trade_dict.keys())
        placeholders = ', '.join(['?' for _ in trade_dict])
        
        cursor.execute(f'''
            INSERT INTO trades ({columns})
            VALUES ({placeholders})
        ''', list(trade_dict.values()))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Trade entry logged: {trade.trade_id}")
        return trade.trade_id
    
    def update_trade_exit(self, trade_id: str, exit_date: str, exit_price: float, 
                         exit_reason: str, notes: Optional[str] = None) -> Dict:
        """Update trade with exit information"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get trade info
        cursor.execute('SELECT * FROM trades WHERE trade_id = ?', (trade_id,))
        trade_data = cursor.fetchone()
        
        if not trade_data:
            raise ValueError(f"Trade {trade_id} not found")
        
        # Calculate P&L
        entry_price = trade_data[3]
        position_size = trade_data[6]
        trade_type = trade_data[7]
        commission = trade_data[12] or 0
        
        if trade_type == 'long':
            pnl = (exit_price - entry_price) * position_size - commission
        else:  # short
            pnl = (entry_price - exit_price) * position_size - commission
        
        pnl_percent = (pnl / (entry_price * position_size)) * 100
        
        # Calculate holding days
        entry_date = datetime.strptime(trade_data[2], '%Y-%m-%d')
        exit_date_obj = datetime.strptime(exit_date, '%Y-%m-%d')
        holding_days = (exit_date_obj - entry_date).days
        
        # Update trade
        cursor.execute('''
            UPDATE trades 
            SET exit_date = ?, exit_price = ?, exit_reason = ?, 
                pnl = ?, pnl_percent = ?, holding_days = ?, notes = ?
            WHERE trade_id = ?
        ''', (exit_date, exit_price, exit_reason, pnl, pnl_percent, 
              holding_days, notes, trade_id))
        
        conn.commit()
        conn.close()
        
        return {
            'trade_id': trade_id,
            'pnl': pnl,
            'pnl_percent': pnl_percent,
            'holding_days': holding_days
        }
    
    def analyze_win_loss_patterns(self) -> Dict:
        """Analyze patterns in winning vs losing trades"""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query('''
            SELECT * FROM trades 
            WHERE exit_date IS NOT NULL
        ''', conn)
        conn.close()
        
        if df.empty:
            return {"error": "No completed trades to analyze"}
        
        # Separate winners and losers
        winners = df[df['pnl'] > 0]
        losers = df[df['pnl'] <= 0]
        
        patterns = {
            'overall_stats': {
                'total_trades': len(df),
                'winning_trades': len(winners),
                'losing_trades': len(losers),
                'win_rate': len(winners) / len(df) * 100,
                'avg_win': winners['pnl'].mean() if not winners.empty else 0,
                'avg_loss': losers['pnl'].mean() if not losers.empty else 0,
                'profit_factor': abs(winners['pnl'].sum() / losers['pnl'].sum()) if not losers.empty and losers['pnl'].sum() != 0 else 0
            },
            'winning_patterns': {},
            'losing_patterns': {},
            'time_analysis': {},
            'signal_analysis': {}
        }
        
        # Analyze by signal type
        for signal in df['entry_signal'].unique():
            signal_trades = df[df['entry_signal'] == signal]
            signal_winners = signal_trades[signal_trades['pnl'] > 0]
            patterns['signal_analysis'][signal] = {
                'total': len(signal_trades),
                'win_rate': len(signal_winners) / len(signal_trades) * 100 if len(signal_trades) > 0 else 0,
                'avg_pnl': signal_trades['pnl'].mean()
            }
        
        # Analyze by holding period
        if not winners.empty:
            patterns['winning_patterns']['avg_hold_days'] = winners['holding_days'].mean()
            patterns['winning_patterns']['most_common_exit'] = winners['exit_reason'].mode()[0] if not winners['exit_reason'].mode().empty else 'N/A'
        
        if not losers.empty:
            patterns['losing_patterns']['avg_hold_days'] = losers['holding_days'].mean()
            patterns['losing_patterns']['most_common_exit'] = losers['exit_reason'].mode()[0] if not losers['exit_reason'].mode().empty else 'N/A'
        
        # Day of week analysis
        df['entry_dow'] = pd.to_datetime(df['entry_date']).dt.day_name()
        dow_performance = df.groupby('entry_dow').agg({
            'pnl': ['count', 'sum', 'mean'],
            'pnl_percent': 'mean'
        }).round(2)
        
        patterns['time_analysis']['day_of_week'] = dow_performance.to_dict()
        
        return patterns
    
    def identify_mistake_patterns(self) -> Dict[str, int]:
        """Identify common mistakes from trade notes"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT mistakes FROM trades WHERE mistakes IS NOT NULL')
        all_mistakes = cursor.fetchall()
        conn.close()
        
        mistake_counts = {}
        for row in all_mistakes:
            if row[0]:
                mistakes = json.loads(row[0])
                for mistake in mistakes:
                    mistake_counts[mistake] = mistake_counts.get(mistake, 0) + 1
        
        return dict(sorted(mistake_counts.items(), key=lambda x: x[1], reverse=True))
    
    def calculate_performance_metrics(self, period_days: int = 30) -> Dict:
        """Calculate comprehensive performance metrics"""
        conn = sqlite3.connect(self.db_path)
        
        # Get trades from period
        cutoff_date = (datetime.now() - timedelta(days=period_days)).strftime('%Y-%m-%d')
        df = pd.read_sql_query(f'''
            SELECT * FROM trades 
            WHERE exit_date >= '{cutoff_date}' AND exit_date IS NOT NULL
        ''', conn)
        conn.close()
        
        if df.empty:
            return {"error": f"No trades in the last {period_days} days"}
        
        # Calculate metrics
        metrics = {}
        
        # Basic metrics
        metrics['total_trades'] = len(df)
        metrics['winning_trades'] = len(df[df['pnl'] > 0])
        metrics['losing_trades'] = len(df[df['pnl'] <= 0])
        metrics['win_rate'] = (metrics['winning_trades'] / metrics['total_trades'] * 100) if metrics['total_trades'] > 0 else 0
        
        # P&L metrics
        metrics['total_pnl'] = df['pnl'].sum()
        metrics['avg_win'] = df[df['pnl'] > 0]['pnl'].mean() if metrics['winning_trades'] > 0 else 0
        metrics['avg_loss'] = df[df['pnl'] <= 0]['pnl'].mean() if metrics['losing_trades'] > 0 else 0
        metrics['largest_win'] = df['pnl'].max()
        metrics['largest_loss'] = df['pnl'].min()
        
        # Risk metrics
        if metrics['losing_trades'] > 0 and df[df['pnl'] <= 0]['pnl'].sum() != 0:
            metrics['profit_factor'] = abs(df[df['pnl'] > 0]['pnl'].sum() / df[df['pnl'] <= 0]['pnl'].sum())
        else:
            metrics['profit_factor'] = float('inf') if metrics['winning_trades'] > 0 else 0
        
        # Expectancy
        win_prob = metrics['win_rate'] / 100
        metrics['expectancy'] = (win_prob * metrics['avg_win']) + ((1 - win_prob) * metrics['avg_loss'])
        
        # Calculate daily returns for Sharpe ratio
        df['exit_date'] = pd.to_datetime(df['exit_date'])
        daily_pnl = df.groupby('exit_date')['pnl'].sum()
        
        if len(daily_pnl) > 1:
            daily_returns = daily_pnl.pct_change().dropna()
            if daily_returns.std() != 0:
                metrics['sharpe_ratio'] = np.sqrt(252) * (daily_returns.mean() / daily_returns.std())
            else:
                metrics['sharpe_ratio'] = 0
        else:
            metrics['sharpe_ratio'] = 0
        
        # Max drawdown calculation
        cumulative_pnl = df.sort_values('exit_date')['pnl'].cumsum()
        running_max = cumulative_pnl.expanding().max()
        drawdown = (cumulative_pnl - running_max)
        metrics['max_drawdown'] = drawdown.min()
        metrics['max_drawdown_percent'] = (metrics['max_drawdown'] / running_max.max() * 100) if running_max.max() != 0 else 0
        
        # Time metrics
        metrics['avg_holding_days'] = df['holding_days'].mean()
        metrics['total_trading_days'] = (df['exit_date'].max() - df['exit_date'].min()).days
        
        return metrics
    
    def generate_journal_report(self) -> str:
        """Generate comprehensive trade journal report"""
        report = []
        report.append("="*60)
        report.append("📔 TRADE JOURNAL ANALYTICS REPORT")
        report.append("="*60)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # 30-day performance metrics
        metrics_30d = self.calculate_performance_metrics(30)
        if 'error' not in metrics_30d:
            report.append("📊 30-DAY PERFORMANCE METRICS")
            report.append("-"*40)
            report.append(f"Total Trades: {metrics_30d['total_trades']}")
            report.append(f"Win Rate: {metrics_30d['win_rate']:.1f}%")
            report.append(f"Total P&L: ${metrics_30d['total_pnl']:,.2f}")
            report.append(f"Average Win: ${metrics_30d['avg_win']:,.2f}")
            report.append(f"Average Loss: ${metrics_30d['avg_loss']:,.2f}")
            report.append(f"Profit Factor: {metrics_30d['profit_factor']:.2f}")
            report.append(f"Expectancy: ${metrics_30d['expectancy']:,.2f}")
            report.append(f"Sharpe Ratio: {metrics_30d['sharpe_ratio']:.2f}")
            report.append(f"Max Drawdown: ${metrics_30d['max_drawdown']:,.2f} ({metrics_30d['max_drawdown_percent']:.1f}%)")
            report.append(f"Avg Hold Time: {metrics_30d['avg_holding_days']:.1f} days\n")
        
        # Win/Loss pattern analysis
        patterns = self.analyze_win_loss_patterns()
        if 'error' not in patterns:
            report.append("🎯 WIN/LOSS PATTERN ANALYSIS")
            report.append("-"*40)
            
            # Signal performance
            report.append("\nPerformance by Signal Type:")
            for signal, stats in patterns['signal_analysis'].items():
                report.append(f"  {signal}: {stats['win_rate']:.1f}% win rate, "
                            f"${stats['avg_pnl']:.2f} avg P&L ({stats['total']} trades)")
            
            # Winning vs Losing patterns
            if patterns['winning_patterns']:
                report.append(f"\nWinning Trades: Avg hold {patterns['winning_patterns'].get('avg_hold_days', 0):.1f} days")
                report.append(f"  Most common exit: {patterns['winning_patterns'].get('most_common_exit', 'N/A')}")
            
            if patterns['losing_patterns']:
                report.append(f"\nLosing Trades: Avg hold {patterns['losing_patterns'].get('avg_hold_days', 0):.1f} days")
                report.append(f"  Most common exit: {patterns['losing_patterns'].get('most_common_exit', 'N/A')}")
        
        # Common mistakes
        mistakes = self.identify_mistake_patterns()
        if mistakes:
            report.append("\n❌ COMMON MISTAKES")
            report.append("-"*40)
            for mistake, count in list(mistakes.items())[:5]:  # Top 5
                report.append(f"  {mistake}: {count} occurrences")
        
        # Trading recommendations
        report.append("\n💡 RECOMMENDATIONS")
        report.append("-"*40)
        
        if 'error' not in metrics_30d and metrics_30d['win_rate'] < 40:
            report.append("• Win rate below 40% - Review entry criteria")
        if 'error' not in metrics_30d and metrics_30d['profit_factor'] < 1.5:
            report.append("• Profit factor below 1.5 - Improve risk/reward ratios")
        if mistakes and 'early_exit' in mistakes and mistakes['early_exit'] > 3:
            report.append("• Frequent early exits - Let winners run longer")
        
        return "\n".join(report)


def main():
    """Test the trade journal system"""
    journal = TradeJournal()
    
    # Example: Log a new trade
    example_trade = Trade(
        trade_id=f"AAPL_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        symbol="AAPL",
        entry_date=datetime.now().strftime('%Y-%m-%d'),
        entry_price=210.50,
        exit_date=None,
        exit_price=None,
        position_size=100,
        trade_type="long",
        entry_signal="STRONG_BUY",
        entry_score=85.5,
        exit_reason=None,
        stop_loss=206.50,
        take_profit=218.50,
        commission=2.00,
        market_conditions={"vix": 15.2, "trend": "bullish"},
        notes="FOMC meeting next week, reduced position size"
    )
    
    # Generate report
    report = journal.generate_journal_report()
    print(report)
    
    # Save report
    with open("advanced_features/trade_journal/journal_report.txt", "w") as f:
        f.write(report)
    
    print("\n✅ Trade journal report saved")


if __name__ == "__main__":
    main()