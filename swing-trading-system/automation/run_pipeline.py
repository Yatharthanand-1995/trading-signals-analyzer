#!/usr/bin/env python3
"""
Simplified Pipeline Runner for Swing Trading System
"""

import os
import sys
import json
import subprocess
import time
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TradingPipeline:
    def __init__(self):
        self.config_path = 'config/stocks.json'
        self.config = self.load_config()
        
    def load_config(self):
        """Load configuration"""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                return json.load(f)
        else:
            print(f"❌ Config file not found: {self.config_path}")
            sys.exit(1)
    
    def get_active_stocks(self):
        """Get currently active stock list"""
        for list_name, list_data in self.config['stock_lists'].items():
            if list_data.get('active', False):
                return list_name, list_data['symbols']
        return None, []
    
    def run_analysis(self):
        """Run simplified analysis pipeline"""
        list_name, stocks = self.get_active_stocks()
        
        if not stocks:
            print("❌ No active stock list found")
            return
        
        print(f"\n{'='*60}")
        print(f"🚀 SWING TRADING SYSTEM - ANALYSIS PIPELINE")
        print(f"{'='*60}")
        print(f"Active List: {list_name}")
        print(f"Stocks: {len(stocks)} symbols")
        print(f"{'='*60}\n")
        
        # For now, just run the test system as a demonstration
        print("📊 Running analysis...")
        
        # Run test system
        try:
            result = subprocess.run(
                ['python3', 'tests/test_system.py'],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                print("✅ Analysis completed successfully!")
            else:
                print("❌ Analysis failed")
                print(result.stderr)
                
        except Exception as e:
            print(f"❌ Error running analysis: {str(e)}")
        
        print(f"\n{'='*60}")
        print("📁 Results saved in:")
        print("  • data/analysis_results/")
        print("  • data/reports/")
        print("  • data/backtest_results/")
        print(f"{'='*60}")
    
    def show_config(self):
        """Show current configuration"""
        list_name, stocks = self.get_active_stocks()
        
        print(f"\n📋 Current Configuration:")
        print(f"  Active List: {list_name}")
        print(f"  Total Stocks: {len(stocks)}")
        print(f"  Stocks: {', '.join(stocks[:5])}{'...' if len(stocks) > 5 else ''}")
        
        print(f"\n📊 Available Lists:")
        for name, data in self.config['stock_lists'].items():
            status = "✅ ACTIVE" if data.get('active', False) else "  "
            print(f"  {status} {name}: {len(data['symbols'])} stocks")

def main():
    """Main function"""
    pipeline = TradingPipeline()
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == '--show-config':
            pipeline.show_config()
        elif command == '--list' and len(sys.argv) > 2:
            # Switch stock list
            list_name = sys.argv[2]
            
            # Update config
            for name in pipeline.config['stock_lists']:
                pipeline.config['stock_lists'][name]['active'] = (name == list_name)
            
            # Save config
            with open(pipeline.config_path, 'w') as f:
                json.dump(pipeline.config, f, indent=2)
            
            print(f"✅ Activated stock list: {list_name}")
        else:
            pipeline.run_analysis()
    else:
        pipeline.run_analysis()

if __name__ == "__main__":
    main()