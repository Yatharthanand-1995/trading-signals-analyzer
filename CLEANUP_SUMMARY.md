# Trading Script Cleanup Summary

## Date: July 20, 2025

### Files and Folders Removed:

#### 1. Test Reports (all from July 19)
- `test_reports/` directory containing 8 old test report files

#### 2. Duplicate Documentation Files in apple_stock_signals root
- ADVANCED_FEATURES_GUIDE.md
- ANALYSIS_PARAMETERS_SUMMARY.md
- AUTOMATED_PIPELINE_GUIDE.md
- COMPLETE_SYSTEM_GUIDE.md
- COMPLETE_SYSTEM_REPORT.md
- COMPREHENSIVE_PROJECT_DOCUMENTATION.md
- FIXES_COMPLETED.md
- FOLDER_STRUCTURE.md
- HOW_TO_RUN_GUIDE.md
- IMPLEMENTATION_STATUS.md
- IMPROVEMENT_AREAS_ANALYSIS.md
- MASTER_DOCUMENTATION.md
- ML_PAPER_TRADING_GUIDE.md
- PRIORITY_FEATURES_IMPLEMENTATION.md
- README_ORGANIZED.md
- SYSTEM_ENHANCEMENT_PLAN.md
- SYSTEM_STATUS_REPORT.md
- TALIB_INSTALLATION_GUIDE.md

#### 3. Old Backtest Results
- All backtest results from July 19 in `backtest_results/` directory
- Kept only latest results from July 20 in `backtest_results/top50/`

#### 4. Duplicate Folders
- `historical_data/` (duplicate of data in swing-trading-system)
- `outputs/` (duplicate outputs)
- `reports/` (duplicate reports)
- `archived_outputs/`
- `archive/old_docs/`
- `archive/old_swing_systems/`

#### 5. Redundant ML Model Files
- `ml_models/analysis_results/`
- `ml_models/performance/`
- `ml_models/predictions/`
- `ml_models/saved_models/`
- `ml_models/training_reports/`

#### 6. Redundant Scripts in Root
- `backtest_1year_performance.py`
- `master_intelligent_system.py`
- `master_pipeline.py`
- `update_imports.py`
- Multiple `final_swing_results_*.json` files

#### 7. Old Paper Trading Files
- Performance reports from July 19
- Trade files from July 19

#### 8. Empty Directories
- `analytics/`
- `scripts/outputs/`

### Current Structure Summary:

The cleaned-up `apple_stock_signals` folder now contains:
- **Core modules**: `core_scripts/`, `data_modules/`, `trading_systems/`
- **Advanced features**: `advanced_features/` with monitoring, risk management, and trade journal
- **ML components**: `ml_models/` with only essential files
- **Testing**: `tests/` directory
- **Configuration**: `config/` with active configuration files
- **Documentation**: `documentation/` with essential docs
- **Scripts**: `scripts/` with utility and automation scripts
- **Paper Trading**: `paper_trading/` with latest results
- **Phase 3**: `phase3/` with future development modules
- **Archive**: `archive/` with essential historical files
- **Utilities**: `utils/` with helper scripts

### Note:
There appears to be a parallel `swing-trading-system` folder at the parent level that contains a reorganized version of the project with better structure. Consider using that as the primary working directory going forward.