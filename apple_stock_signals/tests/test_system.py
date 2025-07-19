#!/usr/bin/env python3
"""
Comprehensive System Test
Tests all components of the trading system to ensure everything is working
"""

import os
import sys
import json
import importlib
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SystemTester:
    def __init__(self):
        self.test_results = {
            'timestamp': datetime.now().isoformat(),
            'passed': 0,
            'failed': 0,
            'warnings': 0,
            'tests': []
        }
        
    def add_test_result(self, component: str, status: str, message: str = ""):
        """Add a test result"""
        self.test_results['tests'].append({
            'component': component,
            'status': status,
            'message': message
        })
        
        if status == 'PASS':
            self.test_results['passed'] += 1
        elif status == 'FAIL':
            self.test_results['failed'] += 1
        elif status == 'WARNING':
            self.test_results['warnings'] += 1
    
    def test_core_modules(self):
        """Test core trading modules"""
        print("\n🔍 Testing Core Modules...")
        print("-" * 50)
        
        core_modules = [
            ('core_scripts.config', 'Configuration'),
            ('data_modules.data_fetcher', 'Data Fetcher'),
            ('data_modules.technical_analyzer', 'Technical Analyzer'),
            ('data_modules.sentiment_analyzer', 'Sentiment Analyzer'),
            ('data_modules.signal_generator', 'Signal Generator'),
            ('data_modules.historical_data_updater', 'Historical Data Updater')
        ]
        
        for module_path, module_name in core_modules:
            try:
                module = importlib.import_module(module_path)
                self.add_test_result(module_name, 'PASS', 'Module loaded successfully')
                print(f"✅ {module_name}: PASS")
            except Exception as e:
                self.add_test_result(module_name, 'FAIL', str(e))
                print(f"❌ {module_name}: FAIL - {e}")
    
    def test_advanced_features(self):
        """Test advanced feature modules"""
        print("\n🔍 Testing Advanced Features...")
        print("-" * 50)
        
        # Add parent directory to path
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        
        advanced_modules = [
            ('advanced_features.economic_calendar.economic_events', 'Economic Calendar'),
            ('advanced_features.trade_journal.trade_journal', 'Trade Journal'),
            ('advanced_features.risk_management.risk_dashboard', 'Risk Dashboard'),
            ('advanced_features.data_monitoring.data_quality_monitor', 'Data Monitor'),
            ('advanced_features.integrated_trading_system', 'Integrated System')
        ]
        
        for module_path, module_name in advanced_modules:
            try:
                module = importlib.import_module(module_path)
                self.add_test_result(module_name, 'PASS', 'Module loaded successfully')
                print(f"✅ {module_name}: PASS")
            except Exception as e:
                self.add_test_result(module_name, 'FAIL', str(e))
                print(f"❌ {module_name}: FAIL - {e}")
    
    def test_data_integrity(self):
        """Test data files and directories"""
        print("\n🔍 Testing Data Integrity...")
        print("-" * 50)
        
        # Check directories
        required_dirs = [
            'historical_data',
            'outputs',
            'backtest_results',
            'advanced_features',
            'core_scripts',
            'data_modules',
            'trading_systems'
        ]
        
        for dir_name in required_dirs:
            if os.path.exists(dir_name):
                self.add_test_result(f"Directory: {dir_name}", 'PASS')
                print(f"✅ Directory {dir_name}: EXISTS")
            else:
                self.add_test_result(f"Directory: {dir_name}", 'WARNING', 'Directory missing')
                print(f"⚠️  Directory {dir_name}: MISSING")
        
        # Check historical data files
        stocks = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'UNH']
        for symbol in stocks:
            filepath = f"historical_data/{symbol}_historical_data.csv"
            if os.path.exists(filepath):
                self.add_test_result(f"Data: {symbol}", 'PASS')
                print(f"✅ {symbol} data: EXISTS")
            else:
                self.add_test_result(f"Data: {symbol}", 'WARNING', 'Data file missing')
                print(f"⚠️  {symbol} data: MISSING")
    
    def test_dependencies(self):
        """Test Python package dependencies"""
        print("\n🔍 Testing Dependencies...")
        print("-" * 50)
        
        required_packages = [
            'pandas',
            'numpy',
            'yfinance',
            'scipy',
            'ta',
            'requests'
        ]
        
        for package in required_packages:
            try:
                importlib.import_module(package)
                self.add_test_result(f"Package: {package}", 'PASS')
                print(f"✅ {package}: INSTALLED")
            except ImportError:
                self.add_test_result(f"Package: {package}", 'FAIL', 'Package not installed')
                print(f"❌ {package}: NOT INSTALLED")
    
    def test_api_connectivity(self):
        """Test API connectivity"""
        print("\n🔍 Testing API Connectivity...")
        print("-" * 50)
        
        try:
            import yfinance as yf
            ticker = yf.Ticker('SPY')
            info = ticker.info
            if info:
                self.add_test_result('Yahoo Finance API', 'PASS')
                print("✅ Yahoo Finance API: CONNECTED")
            else:
                self.add_test_result('Yahoo Finance API', 'WARNING', 'No data returned')
                print("⚠️  Yahoo Finance API: NO DATA")
        except Exception as e:
            self.add_test_result('Yahoo Finance API', 'FAIL', str(e))
            print(f"❌ Yahoo Finance API: FAIL - {e}")
    
    def run_sample_analysis(self):
        """Run a sample analysis to test the system"""
        print("\n🔍 Running Sample Analysis...")
        print("-" * 50)
        
        try:
            # Test signal generation
            from data_modules.signal_generator import AppleSignalGenerator as SignalGenerator
            sg = SignalGenerator()
            signal = sg.generate_signal('AAPL')
            
            if signal and 'signal' in signal:
                self.add_test_result('Signal Generation', 'PASS', f"Generated: {signal['signal']}")
                print(f"✅ Signal Generation: SUCCESS - {signal['signal']}")
            else:
                self.add_test_result('Signal Generation', 'WARNING', 'No signal generated')
                print("⚠️  Signal Generation: NO SIGNAL")
                
        except Exception as e:
            self.add_test_result('Signal Generation', 'FAIL', str(e))
            print(f"❌ Signal Generation: FAIL - {e}")
    
    def generate_test_report(self) -> str:
        """Generate comprehensive test report"""
        report = []
        report.append("="*70)
        report.append("🧪 TRADING SYSTEM TEST REPORT")
        report.append("="*70)
        report.append(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Summary
        total_tests = self.test_results['passed'] + self.test_results['failed'] + self.test_results['warnings']
        success_rate = (self.test_results['passed'] / total_tests * 100) if total_tests > 0 else 0
        
        report.append("📊 TEST SUMMARY")
        report.append("-"*50)
        report.append(f"Total Tests: {total_tests}")
        report.append(f"Passed: {self.test_results['passed']} ✅")
        report.append(f"Failed: {self.test_results['failed']} ❌")
        report.append(f"Warnings: {self.test_results['warnings']} ⚠️")
        report.append(f"Success Rate: {success_rate:.1f}%\n")
        
        # Overall status
        if self.test_results['failed'] == 0:
            if self.test_results['warnings'] == 0:
                report.append("🟢 SYSTEM STATUS: FULLY OPERATIONAL")
            else:
                report.append("🟡 SYSTEM STATUS: OPERATIONAL WITH WARNINGS")
        else:
            report.append("🔴 SYSTEM STATUS: CRITICAL ISSUES DETECTED")
        
        # Detailed results
        report.append("\n\n📋 DETAILED TEST RESULTS")
        report.append("-"*50)
        
        for test in self.test_results['tests']:
            status_emoji = "✅" if test['status'] == 'PASS' else "❌" if test['status'] == 'FAIL' else "⚠️"
            report.append(f"{status_emoji} {test['component']}: {test['status']}")
            if test['message']:
                report.append(f"   → {test['message']}")
        
        # Recommendations
        report.append("\n\n💡 RECOMMENDATIONS")
        report.append("-"*50)
        
        if self.test_results['failed'] > 0:
            report.append("• Fix critical failures before running the system")
            report.append("• Check error messages for specific issues")
            report.append("• Run 'pip install -r requirements.txt' to install missing packages")
        
        if self.test_results['warnings'] > 0:
            report.append("• Address warnings for optimal performance")
            report.append("• Run 'trade-update' to update missing data files")
            report.append("• Create missing directories with proper structure")
        
        if self.test_results['failed'] == 0 and self.test_results['warnings'] == 0:
            report.append("✅ System is fully operational and ready for trading!")
        
        report.append("\n" + "="*70)
        
        return "\n".join(report)
    
    def run_all_tests(self):
        """Run all system tests"""
        print("\n🧪 COMPREHENSIVE SYSTEM TEST")
        print("="*70)
        
        # Run all test suites
        self.test_dependencies()
        self.test_core_modules()
        self.test_advanced_features()
        self.test_data_integrity()
        self.test_api_connectivity()
        self.run_sample_analysis()
        
        # Generate and display report
        report = self.generate_test_report()
        print("\n" + report)
        
        # Save report
        os.makedirs("test_reports", exist_ok=True)
        report_file = f"test_reports/system_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"\n📄 Test report saved to: {report_file}")
        
        # Return status code
        return 0 if self.test_results['failed'] == 0 else 1


def main():
    """Main test runner"""
    tester = SystemTester()
    return tester.run_all_tests()


if __name__ == "__main__":
    sys.exit(main())