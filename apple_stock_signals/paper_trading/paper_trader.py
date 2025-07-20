#!/usr/bin/env python3
"""
Paper Trading Mode - Test strategies without real money
Simulates real trading with virtual portfolio
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from typing import Dict, List, Optional, Tuple
import sqlite3
import sys
import logging

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.validators import Validators, ValidationError, validate_inputs

logger = logging.getLogger(__name__)

class PaperTradingAccount:
    @validate_inputs(
        initial_balance=lambda x: Validators.validate_positive_number(x, "initial_balance", max_value=10000000),
        account_name=lambda x: str(x).strip() if x else "default"
    )
    def __init__(self, initial_balance=10000, account_name="default"):
        # Sanitize account name to prevent directory traversal
        self.account_name = account_name.replace("/", "_").replace("\\", "_").replace("..", "_")
        self.initial_balance = initial_balance
        self.cash_balance = initial_balance
        self.positions = {}  # {symbol: {'shares': x, 'avg_price': y}}
        self.trades = []
        self.daily_values = []
        self.db_path = f"paper_trading/paper_trading_{self.account_name}.db"
        
        # Create directory
        os.makedirs("paper_trading", exist_ok=True)
        os.makedirs("paper_trading/trades", exist_ok=True)
        os.makedirs("paper_trading/performance", exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        # Load existing account if exists
        self._load_account()
    
    def _init_database(self):
        """Initialize SQLite database for paper trading"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS account_info (
                account_name TEXT PRIMARY KEY,
                initial_balance REAL,
                created_date TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                trade_id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                symbol TEXT,
                action TEXT,
                quantity INTEGER,
                price REAL,
                commission REAL,
                total_value REAL,
                notes TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS positions (
                symbol TEXT PRIMARY KEY,
                shares INTEGER,
                avg_price REAL,
                last_updated TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS daily_performance (
                date TEXT PRIMARY KEY,
                total_value REAL,
                cash_balance REAL,
                positions_value REAL,
                daily_return REAL,
                cumulative_return REAL
            )
        """)
        
        conn.commit()
        conn.close()
    
    def _load_account(self):
        """Load existing account data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check if account exists
        cursor.execute("SELECT * FROM account_info WHERE account_name = ?", (self.account_name,))
        account = cursor.fetchone()
        
        if account:
            self.initial_balance = account[1]
            
            # Load cash balance from latest daily performance
            cursor.execute("SELECT cash_balance FROM daily_performance ORDER BY date DESC LIMIT 1")
            cash = cursor.fetchone()
            if cash:
                self.cash_balance = cash[0]
            
            # Load positions
            cursor.execute("SELECT * FROM positions")
            for row in cursor.fetchall():
                self.positions[row[0]] = {
                    'shares': row[1],
                    'avg_price': row[2]
                }
        else:
            # Create new account
            cursor.execute("""
                INSERT INTO account_info (account_name, initial_balance, created_date)
                VALUES (?, ?, ?)
            """, (self.account_name, self.initial_balance, datetime.now().isoformat()))
            conn.commit()
        
        conn.close()
    
    @validate_inputs(
        symbol=lambda x: Validators.validate_stock_symbol(x, allow_lowercase=True),
        action=lambda x: Validators.validate_trading_action(x),
        quantity=lambda x: Validators.validate_integer(x, min_value=1, max_value=1000000, name="quantity"),
        price=lambda x: Validators.validate_positive_number(x, "price", max_value=1000000),
        commission=lambda x: Validators.validate_positive_number(x, "commission", allow_zero=True, max_value=1000)
    )
    def execute_trade(self, symbol: str, action: str, quantity: int, price: float, 
                     signal_data: Dict = None, commission: float = 0):
        """Execute a paper trade with validated inputs"""
        timestamp = datetime.now()
        
        if action == 'BUY':  # Already validated and uppercase
            total_cost = quantity * price + commission
            
            if total_cost > self.cash_balance:
                print(f"❌ Insufficient funds. Need ${total_cost:.2f}, have ${self.cash_balance:.2f}")
                return False
            
            # Update cash
            self.cash_balance -= total_cost
            
            # Update positions
            if symbol in self.positions:
                # Average down/up
                current_shares = self.positions[symbol]['shares']
                current_avg = self.positions[symbol]['avg_price']
                new_shares = current_shares + quantity
                new_avg = ((current_shares * current_avg) + (quantity * price)) / new_shares
                self.positions[symbol] = {'shares': new_shares, 'avg_price': new_avg}
            else:
                self.positions[symbol] = {'shares': quantity, 'avg_price': price}
            
            trade_type = 'BUY'
            
        elif action == 'SELL':  # Already validated and uppercase
            if symbol not in self.positions or self.positions[symbol]['shares'] < quantity:
                print(f"❌ Insufficient shares. Trying to sell {quantity}, have {self.positions.get(symbol, {}).get('shares', 0)}")
                return False
            
            # Calculate profit/loss
            sell_value = quantity * price - commission
            cost_basis = quantity * self.positions[symbol]['avg_price']
            profit = sell_value - cost_basis
            
            # Update cash
            self.cash_balance += sell_value
            
            # Update positions
            self.positions[symbol]['shares'] -= quantity
            if self.positions[symbol]['shares'] == 0:
                del self.positions[symbol]
            
            trade_type = 'SELL'
        
        # No else needed - action is already validated
        
        # Record trade
        trade_record = {
            'timestamp': timestamp.isoformat(),
            'symbol': symbol,
            'action': trade_type,
            'quantity': quantity,
            'price': price,
            'commission': commission,
            'total_value': quantity * price,
            'signal_data': signal_data or {},
            'cash_after': self.cash_balance,
            'portfolio_value': self.get_portfolio_value(current_prices={symbol: price})
        }
        
        self.trades.append(trade_record)
        
        # Save to database
        self._save_trade(trade_record)
        
        # Save daily snapshot
        self._save_daily_snapshot({symbol: price})
        
        print(f"✅ Paper Trade Executed: {trade_type} {quantity} {symbol} @ ${price:.2f}")
        print(f"   Cash Balance: ${self.cash_balance:.2f}")
        
        return True
    
    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """Calculate total portfolio value"""
        positions_value = 0
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                positions_value += position['shares'] * current_prices[symbol]
            else:
                # Use last known price (avg_price as fallback)
                positions_value += position['shares'] * position['avg_price']
        
        return self.cash_balance + positions_value
    
    def get_positions_summary(self, current_prices: Dict[str, float]) -> Dict:
        """Get detailed positions summary"""
        summary = {
            'positions': {},
            'total_positions_value': 0,
            'cash_balance': self.cash_balance,
            'total_portfolio_value': self.cash_balance,
            'initial_balance': self.initial_balance,
            'total_return': 0,
            'total_return_pct': 0
        }
        
        for symbol, position in self.positions.items():
            current_price = current_prices.get(symbol, position['avg_price'])
            position_value = position['shares'] * current_price
            cost_basis = position['shares'] * position['avg_price']
            
            summary['positions'][symbol] = {
                'shares': position['shares'],
                'avg_price': position['avg_price'],
                'current_price': current_price,
                'position_value': position_value,
                'cost_basis': cost_basis,
                'unrealized_pnl': position_value - cost_basis,
                'unrealized_pnl_pct': ((position_value - cost_basis) / cost_basis * 100) if cost_basis > 0 else 0
            }
            
            summary['total_positions_value'] += position_value
        
        summary['total_portfolio_value'] = summary['cash_balance'] + summary['total_positions_value']
        summary['total_return'] = summary['total_portfolio_value'] - self.initial_balance
        summary['total_return_pct'] = (summary['total_return'] / self.initial_balance * 100) if self.initial_balance > 0 else 0
        
        return summary
    
    def get_performance_metrics(self) -> Dict:
        """Calculate performance metrics"""
        if not self.trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'total_pnl': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0
            }
        
        # Analyze closed trades
        closed_trades = []
        trade_pairs = {}  # Track buy/sell pairs
        
        for trade in self.trades:
            if trade['action'] == 'BUY':
                symbol = trade['symbol']
                if symbol not in trade_pairs:
                    trade_pairs[symbol] = []
                trade_pairs[symbol].append(trade)
            elif trade['action'] == 'SELL':
                symbol = trade['symbol']
                if symbol in trade_pairs and trade_pairs[symbol]:
                    # Match with earliest buy
                    buy_trade = trade_pairs[symbol].pop(0)
                    pnl = (trade['price'] - buy_trade['price']) * trade['quantity']
                    closed_trades.append({
                        'symbol': symbol,
                        'buy_date': buy_trade['timestamp'],
                        'sell_date': trade['timestamp'],
                        'quantity': trade['quantity'],
                        'buy_price': buy_trade['price'],
                        'sell_price': trade['price'],
                        'pnl': pnl,
                        'pnl_pct': (pnl / (buy_trade['price'] * trade['quantity'])) * 100
                    })
        
        if not closed_trades:
            # Return basic metrics if no closed trades
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'total_pnl': 0,
                'best_trade': 0,
                'worst_trade': 0,
                'avg_trade_duration': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0
            }
        
        # Calculate metrics
        df = pd.DataFrame(closed_trades)
        winning_trades = df[df['pnl'] > 0]
        losing_trades = df[df['pnl'] < 0]
        
        metrics = {
            'total_trades': len(closed_trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': (len(winning_trades) / len(closed_trades) * 100) if closed_trades else 0,
            'avg_win': winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0,
            'avg_loss': losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0,
            'profit_factor': abs(winning_trades['pnl'].sum() / losing_trades['pnl'].sum()) if len(losing_trades) > 0 and losing_trades['pnl'].sum() != 0 else 0,
            'total_pnl': df['pnl'].sum(),
            'best_trade': df['pnl'].max(),
            'worst_trade': df['pnl'].min(),
            'avg_trade_duration': self._calculate_avg_duration(df),
            'sharpe_ratio': self._calculate_sharpe_ratio(),
            'max_drawdown': self._calculate_max_drawdown()
        }
        
        return metrics
    
    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio from daily returns"""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("SELECT * FROM daily_performance ORDER BY date", conn)
        conn.close()
        
        if len(df) < 2:
            return 0
        
        # Calculate daily returns
        df['daily_return'] = df['total_value'].pct_change()
        
        # Annualized Sharpe ratio (252 trading days)
        if df['daily_return'].std() > 0:
            sharpe = (df['daily_return'].mean() * 252) / (df['daily_return'].std() * np.sqrt(252))
            return sharpe
        return 0
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown"""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("SELECT * FROM daily_performance ORDER BY date", conn)
        conn.close()
        
        if len(df) < 2:
            return 0
        
        # Calculate running maximum
        df['running_max'] = df['total_value'].cummax()
        df['drawdown'] = (df['total_value'] - df['running_max']) / df['running_max']
        
        return abs(df['drawdown'].min() * 100)
    
    def _save_trade(self, trade_record):
        """Save trade to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO trades (timestamp, symbol, action, quantity, price, commission, total_value, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trade_record['timestamp'],
            trade_record['symbol'],
            trade_record['action'],
            trade_record['quantity'],
            trade_record['price'],
            trade_record['commission'],
            trade_record['total_value'],
            json.dumps(trade_record.get('signal_data', {}))
        ))
        
        # Update positions table
        if trade_record['symbol'] in self.positions:
            cursor.execute("""
                INSERT OR REPLACE INTO positions (symbol, shares, avg_price, last_updated)
                VALUES (?, ?, ?, ?)
            """, (
                trade_record['symbol'],
                self.positions[trade_record['symbol']]['shares'],
                self.positions[trade_record['symbol']]['avg_price'],
                trade_record['timestamp']
            ))
        else:
            cursor.execute("DELETE FROM positions WHERE symbol = ?", (trade_record['symbol'],))
        
        conn.commit()
        conn.close()
        
        # Also save to JSON for easy viewing
        trade_file = f"paper_trading/trades/trades_{datetime.now().strftime('%Y%m%d')}.json"
        if os.path.exists(trade_file):
            with open(trade_file, 'r') as f:
                trades = json.load(f)
        else:
            trades = []
        
        trades.append(trade_record)
        
        with open(trade_file, 'w') as f:
            json.dump(trades, f, indent=2)
    
    def _save_daily_snapshot(self, current_prices: Dict[str, float]):
        """Save daily performance snapshot"""
        date = datetime.now().strftime('%Y-%m-%d')
        total_value = self.get_portfolio_value(current_prices)
        positions_value = total_value - self.cash_balance
        
        # Calculate returns
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get previous day's value
        cursor.execute("SELECT total_value FROM daily_performance ORDER BY date DESC LIMIT 1")
        prev = cursor.fetchone()
        
        if prev:
            daily_return = (total_value - prev[0]) / prev[0]
        else:
            daily_return = 0
        
        cumulative_return = (total_value - self.initial_balance) / self.initial_balance
        
        # Save snapshot
        cursor.execute("""
            INSERT OR REPLACE INTO daily_performance 
            (date, total_value, cash_balance, positions_value, daily_return, cumulative_return)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (date, total_value, self.cash_balance, positions_value, daily_return, cumulative_return))
        
        conn.commit()
        conn.close()
    
    def generate_performance_report(self) -> Dict:
        """Generate comprehensive performance report"""
        report = {
            'account_name': self.account_name,
            'report_date': datetime.now().isoformat(),
            'account_summary': {
                'initial_balance': self.initial_balance,
                'current_cash': self.cash_balance,
                'positions_count': len(self.positions)
            },
            'performance_metrics': self.get_performance_metrics(),
            'current_positions': self.get_positions_summary(self._get_latest_prices()),
            'recent_trades': self.trades[-10:] if self.trades else []
        }
        
        # Save report
        report_file = f"paper_trading/performance/performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report
    
    def _get_latest_prices(self) -> Dict[str, float]:
        """Get latest prices for all positions"""
        prices = {}
        for symbol in self.positions:
            # In real implementation, fetch from market data
            # For now, use last trade price or avg price
            for trade in reversed(self.trades):
                if trade['symbol'] == symbol:
                    prices[symbol] = trade['price']
                    break
            if symbol not in prices:
                prices[symbol] = self.positions[symbol]['avg_price']
        
        return prices
    
    def reset_account(self):
        """Reset paper trading account"""
        if input(f"Are you sure you want to reset {self.account_name} account? (yes/no): ").lower() == 'yes':
            self.cash_balance = self.initial_balance
            self.positions = {}
            self.trades = []
            
            # Clear database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("DELETE FROM trades")
            cursor.execute("DELETE FROM positions")
            cursor.execute("DELETE FROM daily_performance")
            conn.commit()
            conn.close()
            
            print(f"✅ Account {self.account_name} has been reset")


def create_paper_trading_report(account_name='default'):
    """Generate paper trading performance report"""
    account = PaperTradingAccount(account_name=account_name)
    report = account.generate_performance_report()
    
    print("\n📊 PAPER TRADING PERFORMANCE REPORT")
    print("=" * 60)
    print(f"Account: {account_name}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 60)
    
    # Account Summary
    summary = report['account_summary']
    current_value = report['current_positions']['total_portfolio_value']
    total_return = current_value - summary['initial_balance']
    total_return_pct = (total_return / summary['initial_balance']) * 100
    
    print(f"\n💰 Account Summary:")
    print(f"   Initial Balance: ${summary['initial_balance']:,.2f}")
    print(f"   Current Value: ${current_value:,.2f}")
    print(f"   Total Return: ${total_return:,.2f} ({total_return_pct:+.2f}%)")
    print(f"   Cash Balance: ${summary['current_cash']:,.2f}")
    print(f"   Positions: {summary['positions_count']}")
    
    # Performance Metrics
    metrics = report['performance_metrics']
    print(f"\n📈 Performance Metrics:")
    print(f"   Total Trades: {metrics['total_trades']}")
    print(f"   Win Rate: {metrics['win_rate']:.1f}%")
    print(f"   Profit Factor: {metrics['profit_factor']:.2f}")
    print(f"   Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"   Max Drawdown: {metrics['max_drawdown']:.1f}%")
    
    # Current Positions
    if report['current_positions']['positions']:
        print(f"\n📊 Current Positions:")
        for symbol, pos in report['current_positions']['positions'].items():
            print(f"   {symbol}: {pos['shares']} shares @ ${pos['avg_price']:.2f}")
            print(f"      Current: ${pos['current_price']:.2f} ({pos['unrealized_pnl_pct']:+.1f}%)")
            print(f"      P&L: ${pos['unrealized_pnl']:+,.2f}")
    
    print("\n" + "=" * 60)
    
    return report


if __name__ == "__main__":
    # Example usage
    print("📄 Paper Trading Mode - Example Usage")
    print("=" * 50)
    
    # Create account
    account = PaperTradingAccount(initial_balance=10000, account_name="test_strategy")
    
    # Example trades
    account.execute_trade("AAPL", "BUY", 10, 210.50, {"signal": "STRONG_BUY", "confidence": 85})
    account.execute_trade("GOOGL", "BUY", 5, 185.25, {"signal": "BUY", "confidence": 70})
    
    # Generate report
    create_paper_trading_report("test_strategy")