#!/usr/bin/env python3
"""
Master Trading Pipeline
Automatically runs complete analysis when stock list changes
"""

import json
import os
import sys
import hashlib
import subprocess
from datetime import datetime
import time
import pandas as pd

class MasterTradingPipeline:
    def __init__(self):
        self.config_path = 'config/stocks.json'
        self.state_path = 'config/pipeline_state.json'
        self.config = self.load_config()
        self.state = self.load_state()
        
    def load_config(self):
        """Load stock configuration"""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                return json.load(f)
        else:
            print(f"❌ Config file not found: {self.config_path}")
            sys.exit(1)
            
    def load_state(self):
        """Load pipeline state"""
        if os.path.exists(self.state_path):
            with open(self.state_path, 'r') as f:
                return json.load(f)
        else:
            return {
                'last_run': None,
                'stock_list_hash': None,
                'completed_steps': []
            }
    
    def save_state(self):
        """Save pipeline state"""
        os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
        with open(self.state_path, 'w') as f:
            json.dump(self.state, f, indent=2)
    
    def get_active_stocks(self):
        """Get currently active stock list"""
        for list_name, list_data in self.config['stock_lists'].items():
            if list_data['active']:
                return list_name, list_data['symbols']
        return None, []
    
    def calculate_stock_hash(self, stocks):
        """Calculate hash of stock list to detect changes"""
        return hashlib.md5(','.join(sorted(stocks)).encode()).hexdigest()
    
    def has_stock_list_changed(self, stocks):
        """Check if stock list has changed since last run"""
        current_hash = self.calculate_stock_hash(stocks)
        return current_hash != self.state.get('stock_list_hash')
    
    def run_step(self, step_name, command, timeout=600):
        """Run a pipeline step"""
        print(f"\n{'='*80}")
        print(f"🔄 Running: {step_name}")
        print(f"{'='*80}")
        
        start_time = time.time()
        
        try:
            # Run command with timeout
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            elapsed = time.time() - start_time
            
            if result.returncode == 0:
                print(f"✅ {step_name} completed in {elapsed:.1f} seconds")
                return True
            else:
                print(f"❌ {step_name} failed with error:")
                print(result.stderr)
                return False
                
        except subprocess.TimeoutExpired:
            print(f"❌ {step_name} timed out after {timeout} seconds")
            return False
        except Exception as e:
            print(f"❌ {step_name} failed with exception: {e}")
            return False
    
    def run_complete_pipeline(self):
        """Run the complete trading pipeline"""
        list_name, stocks = self.get_active_stocks()
        
        if not stocks:
            print("❌ No active stock list found in configuration")
            return
        
        print(f"\n{'='*80}")
        print(f"🚀 MASTER TRADING PIPELINE")
        print(f"{'='*80}")
        print(f"Active List: {list_name}")
        print(f"Total Stocks: {len(stocks)}")
        print(f"Stocks: {', '.join(stocks[:10])}{'...' if len(stocks) > 10 else ''}")
        
        # Check if stock list has changed
        if self.has_stock_list_changed(stocks):
            print(f"\n⚠️  Stock list has changed! Running complete pipeline...")
            force_full_run = True
        else:
            print(f"\n✅ Stock list unchanged since last run")
            force_full_run = False
        
        # Save current stock list
        with open('config/active_stocks.txt', 'w') as f:
            f.write('\n'.join(stocks))
        
        # Pipeline steps
        steps = [
            {
                'name': 'Update Historical Data',
                'command': 'python3 data_modules/historical_data_updater.py',
                'required': True,
                'timeout': 300
            },
            {
                'name': 'Technical Analysis',
                'command': 'python3 core/enhanced_trading_analyzer.py',
                'required': True,
                'timeout': 300
            },
            {
                'name': 'ML Model Training',
                'command': 'python3 scripts/train_ml_model.py',
                'required': self.config['pipeline_settings']['run_ml_training'],
                'timeout': 600
            },
            {
                'name': 'Backtesting',
                'command': 'python3 trading_systems/final_swing_system.py',
                'required': self.config['pipeline_settings']['run_backtesting'],
                'timeout': 600
            },
            {
                'name': 'Paper Trading',
                'command': 'python3 paper_trading/paper_trader.py',
                'required': self.config['pipeline_settings']['run_paper_trading'],
                'timeout': 300
            },
            {
                'name': 'Generate Reports',
                'command': 'python3 scripts/generate_master_report.py',
                'required': True,
                'timeout': 300
            }
        ]
        
        # Run pipeline steps
        successful_steps = []
        failed_steps = []
        
        for step in steps:
            if not step['required']:
                print(f"\n⏭️  Skipping: {step['name']} (disabled in config)")
                continue
                
            if self.run_step(step['name'], step['command'], step['timeout']):
                successful_steps.append(step['name'])
            else:
                failed_steps.append(step['name'])
                if force_full_run:
                    print(f"⚠️  Continuing despite failure...")
                else:
                    print(f"❌ Stopping pipeline due to failure")
                    break
        
        # Update state
        self.state['last_run'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.state['stock_list_hash'] = self.calculate_stock_hash(stocks)
        self.state['completed_steps'] = successful_steps
        self.state['failed_steps'] = failed_steps
        self.state['total_stocks'] = len(stocks)
        self.save_state()
        
        # Generate summary
        self.generate_summary(successful_steps, failed_steps, stocks)
    
    def generate_summary(self, successful_steps, failed_steps, stocks):
        """Generate pipeline execution summary"""
        print(f"\n{'='*80}")
        print(f"📊 PIPELINE EXECUTION SUMMARY")
        print(f"{'='*80}")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Stocks Analyzed: {len(stocks)}")
        print(f"Successful Steps: {len(successful_steps)}")
        print(f"Failed Steps: {len(failed_steps)}")
        
        if successful_steps:
            print(f"\n✅ Completed Steps:")
            for step in successful_steps:
                print(f"  • {step}")
        
        if failed_steps:
            print(f"\n❌ Failed Steps:")
            for step in failed_steps:
                print(f"  • {step}")
        
        # Check for recent results
        if os.path.exists('data/analysis_results/enhanced_analysis.json'):
            with open('data/analysis_results/enhanced_analysis.json', 'r') as f:
                analysis = json.load(f)
                
            print(f"\n📈 Latest Analysis Highlights:")
            signals = {}
            for stock in analysis.get('results', [])[:10]:
                signal = stock.get('signal', 'UNKNOWN')
                if signal not in signals:
                    signals[signal] = 0
                signals[signal] += 1
            
            for signal, count in signals.items():
                print(f"  • {signal}: {count} stocks")
        
        print(f"\n💾 Results saved to:")
        print(f"  • data/analysis_results/")
        print(f"  • data/backtest_results/")
        print(f"  • data/reports/")
        
        print(f"\n{'='*80}")
        print(f"✅ PIPELINE COMPLETE")
        print(f"{'='*80}")

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Master Trading Pipeline')
    parser.add_argument('--list', help='Activate specific stock list', choices=['top_5', 'top_10', 'top_50', 'tech_sector', 'healthcare_sector', 'custom'])
    parser.add_argument('--add-stock', help='Add stock to custom list')
    parser.add_argument('--remove-stock', help='Remove stock from custom list')
    parser.add_argument('--show-config', action='store_true', help='Show current configuration')
    parser.add_argument('--force', action='store_true', help='Force full pipeline run')
    
    args = parser.parse_args()
    
    pipeline = MasterTradingPipeline()
    
    # Handle configuration changes
    config_changed = False
    
    if args.list:
        # Activate specified list
        for list_name in pipeline.config['stock_lists']:
            pipeline.config['stock_lists'][list_name]['active'] = (list_name == args.list)
        config_changed = True
        print(f"✅ Activated stock list: {args.list}")
    
    if args.add_stock:
        # Add stock to custom list
        custom_stocks = pipeline.config['stock_lists']['custom']['symbols']
        if args.add_stock not in custom_stocks:
            custom_stocks.append(args.add_stock)
            config_changed = True
            print(f"✅ Added {args.add_stock} to custom list")
    
    if args.remove_stock:
        # Remove stock from custom list
        custom_stocks = pipeline.config['stock_lists']['custom']['symbols']
        if args.remove_stock in custom_stocks:
            custom_stocks.remove(args.remove_stock)
            config_changed = True
            print(f"✅ Removed {args.remove_stock} from custom list")
    
    if args.show_config:
        # Show current configuration
        print(f"\n📋 CURRENT CONFIGURATION:")
        for list_name, list_data in pipeline.config['stock_lists'].items():
            status = "🟢 ACTIVE" if list_data['active'] else "⚪ inactive"
            print(f"\n{status} {list_name}: {list_data['name']}")
            print(f"   Stocks ({len(list_data['symbols'])}): {', '.join(list_data['symbols'][:5])}{'...' if len(list_data['symbols']) > 5 else ''}")
        return
    
    # Save configuration if changed
    if config_changed:
        with open(pipeline.config_path, 'w') as f:
            json.dump(pipeline.config, f, indent=2)
        print(f"💾 Configuration saved")
    
    # Force full run if requested
    if args.force:
        pipeline.state['stock_list_hash'] = None
    
    # Run pipeline
    pipeline.run_complete_pipeline()

if __name__ == "__main__":
    main()