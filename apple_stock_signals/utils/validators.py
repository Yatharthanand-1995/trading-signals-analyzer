#!/usr/bin/env python3
"""
Input Validation Utilities
Provides comprehensive validation for all user inputs in the trading system
"""

import re
import os
import json
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime, date
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class ValidationError(Exception):
    """Custom exception for validation errors"""
    pass

class Validators:
    """Collection of validation functions"""
    
    # Stock symbol pattern (1-5 uppercase letters)
    STOCK_SYMBOL_PATTERN = re.compile(r'^[A-Z]{1,5}$')
    
    # Allowed stock exchanges
    VALID_EXCHANGES = {'NYSE', 'NASDAQ', 'AMEX'}
    
    # Common invalid symbols to block
    BLOCKED_SYMBOLS = {'TEST', 'DEMO', 'NULL', 'NONE'}
    
    # API key patterns (examples, adjust based on actual APIs)
    API_KEY_PATTERNS = {
        'alphavantage': re.compile(r'^[A-Z0-9]{16}$'),
        'finnhub': re.compile(r'^[a-z0-9]{20,}$'),
        'newsapi': re.compile(r'^[a-f0-9]{32}$')
    }
    
    @staticmethod
    def validate_stock_symbol(symbol: str, allow_lowercase: bool = False) -> str:
        """
        Validate stock symbol
        
        Args:
            symbol: Stock symbol to validate
            allow_lowercase: Whether to allow lowercase (will convert to uppercase)
            
        Returns:
            Validated and normalized symbol
            
        Raises:
            ValidationError: If symbol is invalid
        """
        if not symbol:
            raise ValidationError("Stock symbol cannot be empty")
        
        # Normalize to uppercase
        if allow_lowercase:
            symbol = symbol.upper()
        
        # Check pattern
        if not Validators.STOCK_SYMBOL_PATTERN.match(symbol):
            raise ValidationError(
                f"Invalid stock symbol '{symbol}'. "
                "Must be 1-5 uppercase letters"
            )
        
        # Check blocked list
        if symbol in Validators.BLOCKED_SYMBOLS:
            raise ValidationError(f"Symbol '{symbol}' is not allowed")
        
        return symbol
    
    @staticmethod
    def validate_stock_list(symbols: List[str]) -> List[str]:
        """Validate a list of stock symbols"""
        if not symbols:
            raise ValidationError("Stock list cannot be empty")
        
        validated = []
        seen = set()
        
        for symbol in symbols:
            validated_symbol = Validators.validate_stock_symbol(symbol, allow_lowercase=True)
            
            if validated_symbol in seen:
                logger.warning(f"Duplicate symbol '{validated_symbol}' removed")
                continue
                
            seen.add(validated_symbol)
            validated.append(validated_symbol)
        
        return validated
    
    @staticmethod
    def validate_percentage(
        value: Union[float, int, str],
        min_value: float = 0.0,
        max_value: float = 100.0,
        name: str = "percentage"
    ) -> float:
        """
        Validate percentage value
        
        Args:
            value: Value to validate
            min_value: Minimum allowed value
            max_value: Maximum allowed value
            name: Name for error messages
            
        Returns:
            Validated percentage as float
        """
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise ValidationError(f"{name} must be a number, got {type(value).__name__}")
        
        if not min_value <= value <= max_value:
            raise ValidationError(
                f"{name} must be between {min_value} and {max_value}, got {value}"
            )
        
        return value
    
    @staticmethod
    def validate_positive_number(
        value: Union[float, int, str],
        name: str = "value",
        allow_zero: bool = False,
        max_value: Optional[float] = None
    ) -> float:
        """Validate positive number"""
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise ValidationError(f"{name} must be a number, got {type(value).__name__}")
        
        min_value = 0 if allow_zero else 0.0001
        if value < min_value:
            raise ValidationError(f"{name} must be {'non-negative' if allow_zero else 'positive'}, got {value}")
        
        if max_value is not None and value > max_value:
            raise ValidationError(f"{name} must not exceed {max_value}, got {value}")
        
        return value
    
    @staticmethod
    def validate_integer(
        value: Union[int, str],
        min_value: Optional[int] = None,
        max_value: Optional[int] = None,
        name: str = "value"
    ) -> int:
        """Validate integer value"""
        # Check if it's a float (not an integer)
        if isinstance(value, float) and not value.is_integer():
            raise ValidationError(f"{name} must be an integer, got float {value}")
        
        try:
            # Convert string to int, or float to int if it's a whole number
            if isinstance(value, float) and value.is_integer():
                value = int(value)
            else:
                value = int(value)
        except (TypeError, ValueError):
            raise ValidationError(f"{name} must be an integer, got {type(value).__name__}")
        
        if min_value is not None and value < min_value:
            raise ValidationError(f"{name} must be at least {min_value}, got {value}")
        
        if max_value is not None and value > max_value:
            raise ValidationError(f"{name} must not exceed {max_value}, got {value}")
        
        return value
    
    @staticmethod
    def validate_date(
        date_value: Union[str, date, datetime],
        date_format: str = "%Y-%m-%d",
        name: str = "date"
    ) -> date:
        """Validate date"""
        if isinstance(date_value, datetime):
            return date_value.date()
        
        if isinstance(date_value, date):
            return date_value
        
        try:
            parsed_date = datetime.strptime(date_value, date_format).date()
            return parsed_date
        except (TypeError, ValueError):
            raise ValidationError(
                f"{name} must be a valid date in format {date_format}, got '{date_value}'"
            )
    
    @staticmethod
    def validate_date_range(
        start_date: Union[str, date],
        end_date: Union[str, date],
        max_days: Optional[int] = None
    ) -> tuple[date, date]:
        """Validate date range"""
        start = Validators.validate_date(start_date, name="start_date")
        end = Validators.validate_date(end_date, name="end_date")
        
        if start > end:
            raise ValidationError(f"Start date {start} cannot be after end date {end}")
        
        if max_days:
            days_diff = (end - start).days
            if days_diff > max_days:
                raise ValidationError(
                    f"Date range cannot exceed {max_days} days, got {days_diff} days"
                )
        
        return start, end
    
    @staticmethod
    def validate_file_path(
        file_path: Union[str, Path],
        must_exist: bool = True,
        allowed_extensions: Optional[List[str]] = None,
        base_directory: Optional[Union[str, Path]] = None
    ) -> Path:
        """
        Validate file path
        
        Args:
            file_path: Path to validate
            must_exist: Whether file must exist
            allowed_extensions: List of allowed extensions (e.g., ['.json', '.txt'])
            base_directory: Base directory to prevent traversal attacks
            
        Returns:
            Validated Path object
        """
        try:
            path = Path(file_path).resolve()
        except Exception as e:
            raise ValidationError(f"Invalid file path: {e}")
        
        # Check if file exists
        if must_exist and not path.exists():
            raise ValidationError(f"File does not exist: {path}")
        
        # Check extension
        if allowed_extensions and path.suffix not in allowed_extensions:
            raise ValidationError(
                f"File extension {path.suffix} not allowed. "
                f"Allowed: {', '.join(allowed_extensions)}"
            )
        
        # Prevent directory traversal
        if base_directory:
            base = Path(base_directory).resolve()
            try:
                path.relative_to(base)
            except ValueError:
                raise ValidationError(
                    f"File path must be within {base_directory}"
                )
        
        return path
    
    @staticmethod
    def validate_json_file(file_path: Union[str, Path]) -> Dict[str, Any]:
        """Validate and load JSON file"""
        path = Validators.validate_file_path(
            file_path,
            must_exist=True,
            allowed_extensions=['.json']
        )
        
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            return data
        except json.JSONDecodeError as e:
            raise ValidationError(f"Invalid JSON in {path}: {e}")
        except Exception as e:
            raise ValidationError(f"Error reading {path}: {e}")
    
    @staticmethod
    def validate_api_key(
        key: Optional[str],
        api_name: str,
        required: bool = True
    ) -> Optional[str]:
        """Validate API key format"""
        if not key:
            if required:
                raise ValidationError(f"{api_name} API key is required")
            return None
        
        # Remove whitespace
        key = key.strip()
        
        # Check for placeholder values
        if key.lower() in ['your_api_key_here', 'demo', 'test', 'xxx']:
            if required:
                raise ValidationError(f"Invalid {api_name} API key (placeholder detected)")
            return None
        
        # Check pattern if available
        if api_name.lower() in Validators.API_KEY_PATTERNS:
            pattern = Validators.API_KEY_PATTERNS[api_name.lower()]
            if not pattern.match(key):
                raise ValidationError(
                    f"Invalid {api_name} API key format"
                )
        
        return key
    
    @staticmethod
    def validate_email(email: str) -> str:
        """Validate email address"""
        email_pattern = re.compile(
            r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        )
        
        email = email.strip().lower()
        
        if not email_pattern.match(email):
            raise ValidationError(f"Invalid email address: {email}")
        
        return email
    
    @staticmethod
    def validate_url(url: str, require_https: bool = False) -> str:
        """Validate URL"""
        url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE
        )
        
        if not url_pattern.match(url):
            raise ValidationError(f"Invalid URL: {url}")
        
        if require_https and not url.startswith('https://'):
            raise ValidationError("URL must use HTTPS")
        
        return url
    
    @staticmethod
    def validate_trading_action(action: str) -> str:
        """Validate trading action"""
        valid_actions = {'BUY', 'SELL', 'HOLD'}
        
        action = action.upper().strip()
        
        if action not in valid_actions:
            raise ValidationError(
                f"Invalid trading action '{action}'. "
                f"Must be one of: {', '.join(valid_actions)}"
            )
        
        return action
    
    @staticmethod
    def validate_environment(env: str) -> str:
        """Validate environment name"""
        valid_environments = {'development', 'staging', 'production', 'test'}
        
        env = env.lower().strip()
        
        if env not in valid_environments:
            raise ValidationError(
                f"Invalid environment '{env}'. "
                f"Must be one of: {', '.join(valid_environments)}"
            )
        
        return env

def validate_config_schema(config: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate configuration against a schema
    
    Args:
        config: Configuration dictionary to validate
        schema: Schema definition
        
    Returns:
        Validated configuration
        
    Example schema:
        {
            'portfolio_size': {'type': 'float', 'min': 0, 'required': True},
            'symbols': {'type': 'list', 'items': 'stock_symbol', 'required': True},
            'risk_percentage': {'type': 'percentage', 'min': 0, 'max': 100}
        }
    """
    validated = {}
    
    for key, rules in schema.items():
        # Check if required
        if rules.get('required', False) and key not in config:
            raise ValidationError(f"Missing required field: {key}")
        
        if key not in config:
            if 'default' in rules:
                validated[key] = rules['default']
            continue
        
        value = config[key]
        field_type = rules.get('type', 'string')
        
        # Validate based on type
        if field_type == 'float':
            validated[key] = Validators.validate_positive_number(
                value,
                name=key,
                allow_zero=rules.get('allow_zero', False),
                max_value=rules.get('max')
            )
        
        elif field_type == 'integer':
            validated[key] = Validators.validate_integer(
                value,
                name=key,
                min_value=rules.get('min'),
                max_value=rules.get('max')
            )
        
        elif field_type == 'percentage':
            validated[key] = Validators.validate_percentage(
                value,
                name=key,
                min_value=rules.get('min', 0),
                max_value=rules.get('max', 100)
            )
        
        elif field_type == 'stock_symbol':
            validated[key] = Validators.validate_stock_symbol(value)
        
        elif field_type == 'list':
            if rules.get('items') == 'stock_symbol':
                validated[key] = Validators.validate_stock_list(value)
            else:
                validated[key] = list(value)
        
        elif field_type == 'string':
            validated[key] = str(value).strip()
        
        elif field_type == 'boolean':
            validated[key] = bool(value)
        
        else:
            validated[key] = value
    
    return validated

# Decorator for input validation
def validate_inputs(**validators):
    """
    Decorator to validate function inputs
    
    Usage:
        @validate_inputs(
            symbol=Validators.validate_stock_symbol,
            quantity=lambda x: Validators.validate_positive_number(x, "quantity")
        )
        def place_order(symbol, quantity):
            pass
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Get function signature
            import inspect
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Validate each argument
            for param_name, validator in validators.items():
                if param_name in bound_args.arguments:
                    try:
                        bound_args.arguments[param_name] = validator(
                            bound_args.arguments[param_name]
                        )
                    except ValidationError as e:
                        raise ValidationError(f"Invalid {param_name}: {e}")
            
            return func(*bound_args.args, **bound_args.kwargs)
        
        return wrapper
    return decorator

# Predefined schemas for common configurations
SCHEMAS = {
    'stocks_config': {
        'portfolio_size': {'type': 'float', 'min': 0, 'required': True},
        'max_position_pct': {'type': 'percentage', 'min': 0, 'max': 100, 'required': True},
        'risk_per_trade_pct': {'type': 'percentage', 'min': 0, 'max': 10, 'required': True},
        'sl_pct': {'type': 'percentage', 'min': 0, 'max': 50, 'required': True},
        'historical_data_years': {'type': 'integer', 'min': 1, 'max': 10, 'default': 2},
        'workers': {'type': 'integer', 'min': 1, 'max': 20, 'default': 5}
    },
    
    'trade_params': {
        'symbol': {'type': 'stock_symbol', 'required': True},
        'quantity': {'type': 'integer', 'min': 1, 'required': True},
        'price': {'type': 'float', 'min': 0, 'required': True},
        'action': {'type': 'string', 'required': True}  # Will use validate_trading_action
    }
}