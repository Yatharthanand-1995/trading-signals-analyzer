#!/usr/bin/env python3
"""
Validated Configuration Loader
Loads and validates all configuration files with proper error handling
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.validators import (
    Validators, 
    ValidationError, 
    validate_config_schema,
    SCHEMAS
)

logger = logging.getLogger(__name__)

class ConfigLoader:
    """Handles loading and validation of all configuration files"""
    
    def __init__(self, base_path: Optional[Path] = None):
        if base_path is None:
            self.base_path = Path(__file__).parent.parent
        else:
            self.base_path = Path(base_path)
        
        self.config_dir = self.base_path / 'config'
    
    def load_stocks_config(self) -> Dict[str, Any]:
        """Load and validate stocks configuration"""
        config_path = self.config_dir / 'stocks_config.json'
        
        try:
            # Load JSON file
            config = Validators.validate_json_file(config_path)
            
            # Validate stock lists
            if 'stocks' in config:
                for category, symbols in config['stocks'].items():
                    config['stocks'][category] = Validators.validate_stock_list(symbols)
            
            # Validate analysis settings
            if 'analysis_settings' in config:
                settings = config['analysis_settings']
                validated_settings = validate_config_schema(settings, SCHEMAS['stocks_config'])
                config['analysis_settings'] = validated_settings
            
            # Validate position sizing
            if 'position_sizing' in config:
                sizing = config['position_sizing']
                sizing['max_position_pct'] = Validators.validate_percentage(
                    sizing.get('max_position_pct', 10),
                    name='max_position_pct'
                )
                sizing['risk_per_trade_pct'] = Validators.validate_percentage(
                    sizing.get('risk_per_trade_pct', 2),
                    name='risk_per_trade_pct',
                    max_value=10
                )
                sizing['sl_pct'] = Validators.validate_percentage(
                    sizing.get('sl_pct', 5),
                    name='sl_pct',
                    max_value=50
                )
            
            logger.info(f"Successfully loaded and validated stocks config")
            return config
            
        except ValidationError as e:
            logger.error(f"Validation error in stocks_config.json: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading stocks_config.json: {e}")
            raise
    
    def load_active_stocks(self) -> List[str]:
        """Load and validate active stocks list"""
        active_stocks_path = self.config_dir / 'active_stocks.txt'
        
        try:
            if not active_stocks_path.exists():
                logger.warning("active_stocks.txt not found, using empty list")
                return []
            
            with open(active_stocks_path, 'r') as f:
                lines = f.readlines()
            
            # Extract stock symbols, ignoring comments and empty lines
            symbols = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#'):
                    symbols.append(line)
            
            # Validate all symbols
            validated_symbols = Validators.validate_stock_list(symbols)
            
            logger.info(f"Loaded {len(validated_symbols)} active stocks")
            return validated_symbols
            
        except ValidationError as e:
            logger.error(f"Validation error in active_stocks.txt: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading active_stocks.txt: {e}")
            raise
    
    def validate_api_keys(self) -> Dict[str, bool]:
        """Validate all API keys in environment"""
        from config.env_config import config as env_config
        
        results = {}
        
        # Finnhub API key
        try:
            key = Validators.validate_api_key(
                env_config.FINNHUB_API_KEY,
                'Finnhub',
                required=not env_config.IS_DEVELOPMENT
            )
            results['finnhub'] = key is not None
        except ValidationError as e:
            logger.warning(f"Finnhub API key validation: {e}")
            results['finnhub'] = False
        
        # Add other API keys as needed
        # Example: NewsAPI, Alpha Vantage, etc.
        
        return results
    
    def load_trading_params(self, params_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Validate trading parameters"""
        try:
            validated = {}
            
            # Validate symbol
            if 'symbol' in params_dict:
                validated['symbol'] = Validators.validate_stock_symbol(
                    params_dict['symbol'],
                    allow_lowercase=True
                )
            
            # Validate quantity
            if 'quantity' in params_dict:
                validated['quantity'] = Validators.validate_integer(
                    params_dict['quantity'],
                    min_value=1,
                    max_value=1000000,
                    name='quantity'
                )
            
            # Validate price
            if 'price' in params_dict:
                validated['price'] = Validators.validate_positive_number(
                    params_dict['price'],
                    name='price',
                    max_value=1000000
                )
            
            # Validate action
            if 'action' in params_dict:
                validated['action'] = Validators.validate_trading_action(
                    params_dict['action']
                )
            
            # Validate stop loss
            if 'stop_loss' in params_dict:
                validated['stop_loss'] = Validators.validate_positive_number(
                    params_dict['stop_loss'],
                    name='stop_loss'
                )
            
            # Validate take profit
            if 'take_profit' in params_dict:
                validated['take_profit'] = Validators.validate_positive_number(
                    params_dict['take_profit'],
                    name='take_profit'
                )
            
            return validated
            
        except ValidationError as e:
            logger.error(f"Trading parameter validation error: {e}")
            raise
    
    def validate_backtest_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate backtesting parameters"""
        validated = {}
        
        try:
            # Validate date range
            if 'start_date' in params and 'end_date' in params:
                start, end = Validators.validate_date_range(
                    params['start_date'],
                    params['end_date'],
                    max_days=3650  # 10 years max
                )
                validated['start_date'] = start
                validated['end_date'] = end
            
            # Validate initial capital
            if 'initial_capital' in params:
                validated['initial_capital'] = Validators.validate_positive_number(
                    params['initial_capital'],
                    name='initial_capital',
                    max_value=10000000  # 10M max
                )
            
            # Validate commission
            if 'commission' in params:
                validated['commission'] = Validators.validate_percentage(
                    params['commission'],
                    name='commission',
                    max_value=5  # 5% max commission
                )
            
            # Validate slippage
            if 'slippage' in params:
                validated['slippage'] = Validators.validate_percentage(
                    params['slippage'],
                    name='slippage',
                    max_value=5  # 5% max slippage
                )
            
            return validated
            
        except ValidationError as e:
            logger.error(f"Backtest parameter validation error: {e}")
            raise

# Singleton instance
_config_loader = None

def get_config_loader() -> ConfigLoader:
    """Get singleton ConfigLoader instance"""
    global _config_loader
    if _config_loader is None:
        _config_loader = ConfigLoader()
    return _config_loader

# Convenience functions
def load_validated_stocks_config() -> Dict[str, Any]:
    """Load and validate stocks configuration"""
    return get_config_loader().load_stocks_config()

def load_validated_active_stocks() -> List[str]:
    """Load and validate active stocks list"""
    return get_config_loader().load_active_stocks()

def validate_trading_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Validate trading parameters"""
    return get_config_loader().load_trading_params(params)

def validate_api_keys() -> Dict[str, bool]:
    """Check all API keys"""
    return get_config_loader().validate_api_keys()