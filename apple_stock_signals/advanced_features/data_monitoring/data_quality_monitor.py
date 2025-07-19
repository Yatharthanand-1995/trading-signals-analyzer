#!/usr/bin/env python3
"""
Data Quality Monitoring System
Real-time monitoring of data quality, anomaly detection, and system health
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import json
import os
import sys
from typing import Dict, List, Tuple, Optional
import logging
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataQualityMonitor:
    def __init__(self):
        self.stocks = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'UNH']
        self.data_dir = "historical_data"
        self.monitoring_log = "advanced_features/data_monitoring/monitoring_log.json"
        
        # Data quality thresholds
        self.quality_thresholds = {
            'max_price_change': 0.20,      # 20% max daily change
            'min_volume': 1000000,          # Minimum 1M volume
            'max_spread_pct': 0.02,         # 2% max bid-ask spread
            'max_missing_days': 5,          # Max missing trading days
            'data_staleness_hours': 24,     # Max hours since last update
            'anomaly_z_score': 3.0          # Z-score for anomaly detection
        }
        
        # System health metrics
        self.health_checks = {
            'api_availability': True,
            'data_freshness': True,
            'disk_space': True,
            'memory_usage': True,
            'error_rate': 0.0
        }
    
    def check_data_quality(self, symbol: str) -> Dict:
        """Comprehensive data quality check for a symbol"""
        quality_report = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'quality_score': 100.0,
            'issues': [],
            'warnings': [],
            'data_stats': {}
        }
        
        try:
            # Fetch recent data
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1mo")
            
            if hist.empty:
                quality_report['quality_score'] = 0
                quality_report['issues'].append("No data available")
                return quality_report
            
            # 1. Check data freshness
            last_date = hist.index[-1]
            # Handle timezone-aware datetime comparison
            if hasattr(last_date, 'tz') and last_date.tz is not None:
                # If last_date is timezone-aware, convert to naive
                last_date = last_date.tz_localize(None)
            days_old = (datetime.now() - last_date).days
            if days_old > 1 and datetime.now().weekday() < 5:  # Weekday check
                quality_report['warnings'].append(f"Data is {days_old} days old")
                quality_report['quality_score'] -= 10
            
            # 2. Check for price anomalies
            daily_returns = hist['Close'].pct_change().dropna()
            anomalies = self._detect_anomalies(daily_returns)
            if anomalies:
                quality_report['warnings'].append(f"Found {len(anomalies)} price anomalies")
                quality_report['quality_score'] -= 5 * len(anomalies)
            
            # 3. Check volume consistency
            avg_volume = hist['Volume'].mean()
            low_volume_days = hist[hist['Volume'] < self.quality_thresholds['min_volume']]
            if len(low_volume_days) > 0:
                quality_report['warnings'].append(f"{len(low_volume_days)} days with low volume")
                quality_report['quality_score'] -= 2 * len(low_volume_days)
            
            # 4. Check for data gaps
            expected_days = pd.bdate_range(start=hist.index[0], end=hist.index[-1])
            missing_days = expected_days.difference(hist.index)
            if len(missing_days) > self.quality_thresholds['max_missing_days']:
                quality_report['issues'].append(f"{len(missing_days)} missing trading days")
                quality_report['quality_score'] -= 15
            
            # 5. Check bid-ask spread (using high-low as proxy)
            hist['spread_pct'] = (hist['High'] - hist['Low']) / hist['Close']
            high_spread_days = hist[hist['spread_pct'] > self.quality_thresholds['max_spread_pct']]
            if len(high_spread_days) > 0:
                quality_report['warnings'].append(f"{len(high_spread_days)} days with high spread")
                quality_report['quality_score'] -= len(high_spread_days)
            
            # 6. Data statistics
            quality_report['data_stats'] = {
                'last_update': last_date.strftime('%Y-%m-%d'),
                'days_of_data': len(hist),
                'avg_volume': f"{avg_volume:,.0f}",
                'avg_daily_range': f"{hist['spread_pct'].mean() * 100:.2f}%",
                'volatility': f"{daily_returns.std() * np.sqrt(252) * 100:.1f}%",
                'missing_days': len(missing_days),
                'anomaly_count': len(anomalies)
            }
            
            # Ensure quality score doesn't go below 0
            quality_report['quality_score'] = max(0, quality_report['quality_score'])
            
        except Exception as e:
            logger.error(f"Error checking data quality for {symbol}: {e}")
            quality_report['quality_score'] = 0
            quality_report['issues'].append(f"Error: {str(e)}")
        
        return quality_report
    
    def _detect_anomalies(self, returns: pd.Series) -> List[Tuple[str, float]]:
        """Detect anomalous returns using z-score"""
        anomalies = []
        
        if len(returns) < 5:
            return anomalies
        
        mean = returns.mean()
        std = returns.std()
        
        for date, ret in returns.items():
            z_score = abs((ret - mean) / std) if std > 0 else 0
            if z_score > self.quality_thresholds['anomaly_z_score']:
                # Handle timezone-aware dates
                if hasattr(date, 'tz') and date.tz is not None:
                    date = date.tz_localize(None)
                anomalies.append((date.strftime('%Y-%m-%d'), ret))
        
        return anomalies
    
    def monitor_system_health(self) -> Dict:
        """Monitor overall system health"""
        health_report = {
            'timestamp': datetime.now().isoformat(),
            'overall_health': 'HEALTHY',
            'health_score': 100,
            'components': {},
            'alerts': []
        }
        
        # 1. Check API availability
        try:
            test_ticker = yf.Ticker('SPY')
            test_data = test_ticker.info
            health_report['components']['api_status'] = 'ONLINE'
        except:
            health_report['components']['api_status'] = 'OFFLINE'
            health_report['health_score'] -= 30
            health_report['alerts'].append("Yahoo Finance API unavailable")
        
        # 2. Check data freshness
        stale_count = 0
        for symbol in self.stocks:
            filepath = os.path.join(self.data_dir, f"{symbol}_historical_data.csv")
            if os.path.exists(filepath):
                mod_time = os.path.getmtime(filepath)
                hours_old = (datetime.now().timestamp() - mod_time) / 3600
                if hours_old > self.quality_thresholds['data_staleness_hours']:
                    stale_count += 1
        
        if stale_count > 0:
            health_report['components']['data_freshness'] = f"{stale_count} stale files"
            health_report['health_score'] -= 5 * stale_count
            health_report['alerts'].append(f"{stale_count} data files need updating")
        else:
            health_report['components']['data_freshness'] = 'CURRENT'
        
        # 3. Check disk space
        try:
            import shutil
            total, used, free = shutil.disk_usage("/")
            free_gb = free // (2**30)
            free_pct = (free / total) * 100
            
            health_report['components']['disk_space'] = f"{free_gb}GB free ({free_pct:.1f}%)"
            if free_pct < 10:
                health_report['health_score'] -= 20
                health_report['alerts'].append(f"Low disk space: {free_pct:.1f}% free")
        except:
            health_report['components']['disk_space'] = 'UNKNOWN'
        
        # 4. Check memory usage
        try:
            import psutil
            memory = psutil.virtual_memory()
            health_report['components']['memory_usage'] = f"{memory.percent:.1f}% used"
            if memory.percent > 90:
                health_report['health_score'] -= 15
                health_report['alerts'].append(f"High memory usage: {memory.percent:.1f}%")
        except:
            health_report['components']['memory_usage'] = 'UNKNOWN'
        
        # 5. Check error logs
        error_count = self._count_recent_errors()
        health_report['components']['error_rate'] = f"{error_count} errors/hour"
        if error_count > 10:
            health_report['health_score'] -= 10
            health_report['alerts'].append(f"High error rate: {error_count} errors/hour")
        
        # Determine overall health status
        if health_report['health_score'] >= 90:
            health_report['overall_health'] = 'HEALTHY'
        elif health_report['health_score'] >= 70:
            health_report['overall_health'] = 'WARNING'
        else:
            health_report['overall_health'] = 'CRITICAL'
        
        return health_report
    
    def _count_recent_errors(self) -> int:
        """Count errors in recent log entries"""
        # This is a placeholder - in production, you'd parse actual log files
        return 0
    
    def compare_data_sources(self) -> Dict:
        """Compare data from multiple sources for validation"""
        comparison_report = {
            'timestamp': datetime.now().isoformat(),
            'discrepancies': [],
            'source_status': {}
        }
        
        for symbol in self.stocks[:2]:  # Check first 2 symbols to save time
            try:
                # Get data from yfinance
                yf_ticker = yf.Ticker(symbol)
                yf_price = yf_ticker.info.get('currentPrice', yf_ticker.info.get('regularMarketPrice', 0))
                
                # Get data from historical file
                filepath = os.path.join(self.data_dir, f"{symbol}_historical_data.csv")
                if os.path.exists(filepath):
                    df = pd.read_csv(filepath, index_col='Date', parse_dates=True)
                    if not df.empty:
                        file_price = df['Close'].iloc[-1]
                        
                        # Compare prices
                        if yf_price > 0 and file_price > 0:
                            diff_pct = abs((yf_price - file_price) / file_price) * 100
                            if diff_pct > 5:  # More than 5% difference
                                comparison_report['discrepancies'].append({
                                    'symbol': symbol,
                                    'yfinance_price': yf_price,
                                    'file_price': file_price,
                                    'difference_pct': diff_pct
                                })
                
                comparison_report['source_status']['yfinance'] = 'ONLINE'
                
            except Exception as e:
                logger.error(f"Error comparing sources for {symbol}: {e}")
                comparison_report['source_status']['yfinance'] = 'ERROR'
        
        return comparison_report
    
    def generate_monitoring_dashboard(self) -> str:
        """Generate comprehensive monitoring dashboard"""
        dashboard = []
        dashboard.append("="*80)
        dashboard.append("📊 DATA QUALITY MONITORING DASHBOARD")
        dashboard.append("="*80)
        dashboard.append(f"Report Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # System health check
        health = self.monitor_system_health()
        health_emoji = "🟢" if health['overall_health'] == 'HEALTHY' else "🟡" if health['overall_health'] == 'WARNING' else "🔴"
        
        dashboard.append(f"{health_emoji} SYSTEM HEALTH: {health['overall_health']} (Score: {health['health_score']}/100)")
        dashboard.append("-"*60)
        
        for component, status in health['components'].items():
            dashboard.append(f"  {component}: {status}")
        
        if health['alerts']:
            dashboard.append("\n⚠️ ALERTS:")
            for alert in health['alerts']:
                dashboard.append(f"  • {alert}")
        
        # Data quality by symbol
        dashboard.append("\n\n📈 DATA QUALITY BY SYMBOL")
        dashboard.append("-"*60)
        
        overall_quality = 0
        quality_reports = []
        
        for symbol in self.stocks:
            quality = self.check_data_quality(symbol)
            quality_reports.append(quality)
            overall_quality += quality['quality_score']
            
            # Quality emoji
            if quality['quality_score'] >= 90:
                emoji = "🟢"
            elif quality['quality_score'] >= 70:
                emoji = "🟡"
            else:
                emoji = "🔴"
            
            dashboard.append(f"\n{emoji} {symbol}: {quality['quality_score']:.0f}/100")
            
            if quality['data_stats']:
                stats = quality['data_stats']
                dashboard.append(f"  Last Update: {stats['last_update']}")
                dashboard.append(f"  Volatility: {stats['volatility']}")
                dashboard.append(f"  Avg Volume: {stats['avg_volume']}")
            
            if quality['issues']:
                dashboard.append("  Issues:")
                for issue in quality['issues']:
                    dashboard.append(f"    ❌ {issue}")
            
            if quality['warnings']:
                dashboard.append("  Warnings:")
                for warning in quality['warnings']:
                    dashboard.append(f"    ⚠️ {warning}")
        
        # Overall data quality score
        avg_quality = overall_quality / len(self.stocks)
        dashboard.append(f"\n\n📊 OVERALL DATA QUALITY: {avg_quality:.1f}/100")
        
        # Source comparison
        dashboard.append("\n\n🔄 DATA SOURCE VALIDATION")
        dashboard.append("-"*60)
        comparison = self.compare_data_sources()
        
        if comparison['discrepancies']:
            dashboard.append("⚠️ Price Discrepancies Found:")
            for disc in comparison['discrepancies']:
                dashboard.append(f"  {disc['symbol']}: YF=${disc['yfinance_price']:.2f} vs "
                              f"File=${disc['file_price']:.2f} ({disc['difference_pct']:.1f}% diff)")
        else:
            dashboard.append("✅ All data sources aligned")
        
        # Recommendations
        dashboard.append("\n\n💡 RECOMMENDATIONS")
        dashboard.append("-"*60)
        
        if avg_quality < 80:
            dashboard.append("• Run data update to refresh historical files")
        if health['health_score'] < 80:
            dashboard.append("• Address system health alerts")
        if any(q['quality_score'] < 70 for q in quality_reports):
            dashboard.append("• Investigate low-quality data symbols")
        if comparison['discrepancies']:
            dashboard.append("• Reconcile data source discrepancies")
        
        if avg_quality >= 90 and health['health_score'] >= 90:
            dashboard.append("✅ All systems operational - ready for trading")
        
        dashboard.append("\n" + "="*80)
        
        # Save monitoring log
        self._save_monitoring_log({
            'timestamp': datetime.now().isoformat(),
            'system_health': health,
            'data_quality': quality_reports,
            'source_comparison': comparison,
            'overall_quality_score': avg_quality
        })
        
        return "\n".join(dashboard)
    
    def _save_monitoring_log(self, log_data: Dict):
        """Save monitoring data to log file"""
        os.makedirs(os.path.dirname(self.monitoring_log), exist_ok=True)
        
        # Load existing logs
        logs = []
        if os.path.exists(self.monitoring_log):
            try:
                with open(self.monitoring_log, 'r') as f:
                    logs = json.load(f)
            except:
                logs = []
        
        # Add new log entry
        logs.append(log_data)
        
        # Keep only last 100 entries
        logs = logs[-100:]
        
        # Save updated logs
        with open(self.monitoring_log, 'w') as f:
            json.dump(logs, f, indent=2)
    
    def get_monitoring_trends(self, hours: int = 24) -> Dict:
        """Analyze monitoring trends over time"""
        if not os.path.exists(self.monitoring_log):
            return {'error': 'No monitoring history available'}
        
        with open(self.monitoring_log, 'r') as f:
            logs = json.load(f)
        
        # Filter logs by time
        cutoff = datetime.now() - timedelta(hours=hours)
        recent_logs = []
        for log in logs:
            try:
                log_time = datetime.fromisoformat(log['timestamp'])
                # Handle timezone-aware datetime if present
                if hasattr(log_time, 'tz') and log_time.tz is not None:
                    log_time = log_time.tz_localize(None)
                if log_time > cutoff:
                    recent_logs.append(log)
            except:
                # Skip malformed timestamps
                continue
        
        if not recent_logs:
            return {'error': f'No logs in the last {hours} hours'}
        
        # Analyze trends
        trends = {
            'period_hours': hours,
            'log_count': len(recent_logs),
            'avg_quality_score': np.mean([log['overall_quality_score'] for log in recent_logs]),
            'min_quality_score': min([log['overall_quality_score'] for log in recent_logs]),
            'max_quality_score': max([log['overall_quality_score'] for log in recent_logs]),
            'health_scores': [log['system_health']['health_score'] for log in recent_logs],
            'alert_count': sum(len(log['system_health']['alerts']) for log in recent_logs)
        }
        
        return trends


def main():
    """Run data quality monitoring"""
    monitor = DataQualityMonitor()
    
    # Generate monitoring dashboard
    dashboard = monitor.generate_monitoring_dashboard()
    print(dashboard)
    
    # Save dashboard report
    os.makedirs("advanced_features/data_monitoring", exist_ok=True)
    with open("advanced_features/data_monitoring/monitoring_report.txt", "w") as f:
        f.write(dashboard)
    
    # Check trends
    trends = monitor.get_monitoring_trends(24)
    if 'error' not in trends:
        print(f"\n📈 24-Hour Trends:")
        print(f"  Average Quality: {trends['avg_quality_score']:.1f}/100")
        print(f"  Total Alerts: {trends['alert_count']}")
    
    print("\n✅ Data monitoring report saved")


if __name__ == "__main__":
    main()