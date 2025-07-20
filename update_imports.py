#!/usr/bin/env python3
"""
Update all import paths in the reorganized project
"""

import os
import re

# Define import mappings
IMPORT_MAPPINGS = {
    # Old module paths -> New module paths
    'from multi_timeframe_analyzer import': 'from core.analysis.multi_timeframe import',
    'from volume_analyzer import': 'from core.analysis.volume_analyzer import',
    'from entry_filter_system import': 'from core.analysis.entry_filters import',
    'from swing_trading_analyzer import': 'from core.analysis.swing_analyzer import',
    'from dynamic_stop_loss_system import': 'from core.risk_management.stop_loss import',
    'from advanced_position_sizing import': 'from core.risk_management.position_sizing import',
    'from profit_taking_strategy import': 'from core.risk_management.profit_targets import',
    'from integrated_risk_management import': 'from core.risk_management.risk_manager import',
    'from technical_indicators_wrapper import': 'from core.indicators.technical_wrapper import',
    'from enhanced_trading_analyzer import': 'from core.indicators.enhanced_analyzer import',
    
    # Core scripts imports
    'from core_scripts.': 'from core.',
    'import core_scripts.': 'import core.',
    
    # Data modules
    'from data_modules.': 'from core.utils.',
    
    # Path updates
    'apple_stock_signals': 'swing-trading-system',
    'core_scripts': 'core',
    'outputs/': 'data/analysis_results/',
    'historical_data/': 'data/historical/',
    'backtest_results/': 'data/backtest_results/',
    'reports/': 'data/reports/',
    'config/stocks_config.json': 'config/stocks.json',
}

def update_file_imports(filepath):
    """Update imports in a single file"""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        original_content = content
        
        # Apply all mappings
        for old_import, new_import in IMPORT_MAPPINGS.items():
            content = content.replace(old_import, new_import)
        
        # Write back if changed
        if content != original_content:
            with open(filepath, 'w') as f:
                f.write(content)
            print(f"✅ Updated: {filepath}")
            return True
        return False
    except Exception as e:
        print(f"❌ Error updating {filepath}: {str(e)}")
        return False

def update_all_imports(base_dir):
    """Update imports in all Python files"""
    updated_count = 0
    
    for root, dirs, files in os.walk(base_dir):
        # Skip archive and old directories
        if 'archive' in root or 'apple_stock_signals' in root:
            continue
            
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                if update_file_imports(filepath):
                    updated_count += 1
    
    return updated_count

def update_shell_scripts(base_dir):
    """Update paths in shell scripts"""
    shell_files = [
        os.path.join(base_dir, 'automation', 'run_analysis.sh'),
    ]
    
    for filepath in shell_files:
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    content = f.read()
                
                # Update paths
                content = content.replace('apple_stock_signals', 'swing-trading-system')
                content = content.replace('core_scripts', 'core')
                content = content.replace('outputs/', 'data/analysis_results/')
                content = content.replace('historical_data/', 'data/historical/')
                content = content.replace('backtest_results/', 'data/backtest_results/')
                content = content.replace('reports/', 'data/reports/')
                content = content.replace('stocks_config.json', 'stocks.json')
                
                with open(filepath, 'w') as f:
                    f.write(content)
                
                print(f"✅ Updated shell script: {filepath}")
            except Exception as e:
                print(f"❌ Error updating {filepath}: {str(e)}")

def main():
    base_dir = '/Users/yatharthanand/genai-assistant-vercel/trading-script/swing-trading-system'
    
    print("🔄 Updating import paths...")
    print("="*50)
    
    # Update Python imports
    updated = update_all_imports(base_dir)
    print(f"\n✅ Updated {updated} Python files")
    
    # Update shell scripts
    print("\n🔄 Updating shell scripts...")
    update_shell_scripts(base_dir)
    
    print("\n✅ Import path updates complete!")

if __name__ == "__main__":
    main()