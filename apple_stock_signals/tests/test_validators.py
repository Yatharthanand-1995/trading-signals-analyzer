#!/usr/bin/env python3
"""
Unit tests for Input Validators
"""

import pytest
from datetime import date, datetime
from pathlib import Path
import tempfile
import json

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.validators import (
    Validators,
    ValidationError,
    validate_config_schema,
    validate_inputs
)

class TestStockSymbolValidation:
    """Test stock symbol validation"""
    
    def test_valid_symbols(self):
        """Test valid stock symbols"""
        valid_symbols = ['AAPL', 'GOOGL', 'TSLA', 'MSFT', 'A', 'AA', 'AAAAA']
        
        for symbol in valid_symbols:
            result = Validators.validate_stock_symbol(symbol)
            assert result == symbol
    
    def test_lowercase_conversion(self):
        """Test lowercase symbol conversion"""
        assert Validators.validate_stock_symbol('aapl', allow_lowercase=True) == 'AAPL'
        assert Validators.validate_stock_symbol('googl', allow_lowercase=True) == 'GOOGL'
    
    def test_invalid_symbols(self):
        """Test invalid stock symbols"""
        invalid_symbols = [
            '',  # Empty
            'AAAAAA',  # Too long
            '123',  # Numbers
            'AAPL1',  # Mixed
            'AA-PL',  # Special chars
            'aa pl',  # Spaces
            None  # None
        ]
        
        for symbol in invalid_symbols:
            with pytest.raises(ValidationError):
                Validators.validate_stock_symbol(symbol or '')
    
    def test_blocked_symbols(self):
        """Test blocked symbols"""
        blocked = ['TEST', 'DEMO', 'NULL', 'NONE']
        
        for symbol in blocked:
            with pytest.raises(ValidationError) as exc_info:
                Validators.validate_stock_symbol(symbol)
            assert "not allowed" in str(exc_info.value)
    
    def test_stock_list_validation(self):
        """Test stock list validation"""
        # Valid list
        symbols = ['AAPL', 'googl', 'TSLA', 'MSFT']
        result = Validators.validate_stock_list(symbols)
        assert result == ['AAPL', 'GOOGL', 'TSLA', 'MSFT']
        
        # Duplicates removed
        symbols_with_dupes = ['AAPL', 'aapl', 'GOOGL', 'AAPL']
        result = Validators.validate_stock_list(symbols_with_dupes)
        assert result == ['AAPL', 'GOOGL']
        assert len(result) == 2
        
        # Empty list
        with pytest.raises(ValidationError):
            Validators.validate_stock_list([])

class TestNumberValidation:
    """Test number validation"""
    
    def test_percentage_validation(self):
        """Test percentage validation"""
        # Valid percentages
        assert Validators.validate_percentage(50) == 50.0
        assert Validators.validate_percentage('25.5') == 25.5
        assert Validators.validate_percentage(0) == 0.0
        assert Validators.validate_percentage(100) == 100.0
        
        # Custom range
        assert Validators.validate_percentage(5, min_value=0, max_value=10) == 5.0
        
        # Invalid percentages
        with pytest.raises(ValidationError):
            Validators.validate_percentage(101)
        
        with pytest.raises(ValidationError):
            Validators.validate_percentage(-1)
        
        with pytest.raises(ValidationError):
            Validators.validate_percentage('abc')
    
    def test_positive_number_validation(self):
        """Test positive number validation"""
        # Valid positive numbers
        assert Validators.validate_positive_number(100) == 100.0
        assert Validators.validate_positive_number('50.5') == 50.5
        assert Validators.validate_positive_number(0.1) == 0.1
        
        # Allow zero
        assert Validators.validate_positive_number(0, allow_zero=True) == 0.0
        
        # Max value
        assert Validators.validate_positive_number(100, max_value=1000) == 100.0
        
        # Invalid numbers
        with pytest.raises(ValidationError):
            Validators.validate_positive_number(-1)
        
        with pytest.raises(ValidationError):
            Validators.validate_positive_number(0, allow_zero=False)
        
        with pytest.raises(ValidationError):
            Validators.validate_positive_number(1001, max_value=1000)
    
    def test_integer_validation(self):
        """Test integer validation"""
        # Valid integers
        assert Validators.validate_integer(10) == 10
        assert Validators.validate_integer('100') == 100
        
        # With range
        assert Validators.validate_integer(5, min_value=1, max_value=10) == 5
        
        # Invalid integers
        with pytest.raises(ValidationError):
            Validators.validate_integer(10.5)
        
        with pytest.raises(ValidationError):
            Validators.validate_integer('abc')
        
        with pytest.raises(ValidationError):
            Validators.validate_integer(0, min_value=1)

class TestDateValidation:
    """Test date validation"""
    
    def test_date_validation(self):
        """Test date validation"""
        # String date
        result = Validators.validate_date('2024-01-15')
        assert result == date(2024, 1, 15)
        
        # Date object
        today = date.today()
        assert Validators.validate_date(today) == today
        
        # DateTime object
        now = datetime.now()
        assert Validators.validate_date(now) == now.date()
        
        # Invalid date
        with pytest.raises(ValidationError):
            Validators.validate_date('2024-13-01')  # Invalid month
        
        with pytest.raises(ValidationError):
            Validators.validate_date('not-a-date')
    
    def test_date_range_validation(self):
        """Test date range validation"""
        # Valid range
        start, end = Validators.validate_date_range('2024-01-01', '2024-12-31')
        assert start == date(2024, 1, 1)
        assert end == date(2024, 12, 31)
        
        # Invalid range (start after end)
        with pytest.raises(ValidationError) as exc_info:
            Validators.validate_date_range('2024-12-31', '2024-01-01')
        assert "cannot be after" in str(exc_info.value)
        
        # Max days constraint
        with pytest.raises(ValidationError) as exc_info:
            Validators.validate_date_range('2024-01-01', '2024-02-01', max_days=30)
        assert "cannot exceed 30 days" in str(exc_info.value)

class TestFileValidation:
    """Test file path validation"""
    
    def test_file_path_validation(self):
        """Test file path validation"""
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
            tmp.write(b'{"test": true}')
            tmp_path = tmp.name
        
        try:
            # Valid file
            result = Validators.validate_file_path(tmp_path)
            assert isinstance(result, Path)
            assert result.exists()
            
            # Must exist check
            with pytest.raises(ValidationError):
                Validators.validate_file_path('/nonexistent/file.txt', must_exist=True)
            
            # Extension check
            with pytest.raises(ValidationError):
                Validators.validate_file_path(tmp_path, allowed_extensions=['.txt'])
            
        finally:
            os.unlink(tmp_path)
    
    def test_json_file_validation(self):
        """Test JSON file validation"""
        # Valid JSON
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
            json.dump({'test': 'data'}, tmp)
            tmp_path = tmp.name
        
        try:
            data = Validators.validate_json_file(tmp_path)
            assert data == {'test': 'data'}
        finally:
            os.unlink(tmp_path)
        
        # Invalid JSON
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
            tmp.write('invalid json')
            tmp_path = tmp.name
        
        try:
            with pytest.raises(ValidationError) as exc_info:
                Validators.validate_json_file(tmp_path)
            assert "Invalid JSON" in str(exc_info.value)
        finally:
            os.unlink(tmp_path)

class TestAPIKeyValidation:
    """Test API key validation"""
    
    def test_api_key_validation(self):
        """Test API key validation"""
        # Valid key
        key = Validators.validate_api_key('abc123def456', 'TestAPI', required=False)
        assert key == 'abc123def456'
        
        # Missing required key
        with pytest.raises(ValidationError):
            Validators.validate_api_key(None, 'TestAPI', required=True)
        
        # Placeholder detection
        placeholders = ['your_api_key_here', 'demo', 'test', 'xxx']
        for placeholder in placeholders:
            with pytest.raises(ValidationError) as exc_info:
                Validators.validate_api_key(placeholder, 'TestAPI', required=True)
            assert "placeholder detected" in str(exc_info.value)

class TestTradingValidation:
    """Test trading-specific validation"""
    
    def test_trading_action_validation(self):
        """Test trading action validation"""
        # Valid actions
        assert Validators.validate_trading_action('buy') == 'BUY'
        assert Validators.validate_trading_action('SELL') == 'SELL'
        assert Validators.validate_trading_action('HoLd') == 'HOLD'
        
        # Invalid actions
        with pytest.raises(ValidationError):
            Validators.validate_trading_action('INVALID')
        
        with pytest.raises(ValidationError):
            Validators.validate_trading_action('')
    
    def test_environment_validation(self):
        """Test environment validation"""
        # Valid environments
        assert Validators.validate_environment('development') == 'development'
        assert Validators.validate_environment('PRODUCTION') == 'production'
        assert Validators.validate_environment('Test') == 'test'
        
        # Invalid environment
        with pytest.raises(ValidationError):
            Validators.validate_environment('invalid_env')

class TestConfigSchemaValidation:
    """Test configuration schema validation"""
    
    def test_schema_validation(self):
        """Test schema-based validation"""
        schema = {
            'portfolio_size': {'type': 'float', 'min': 0, 'required': True},
            'max_position_pct': {'type': 'percentage', 'min': 0, 'max': 100},
            'symbols': {'type': 'list', 'items': 'stock_symbol'},
            'workers': {'type': 'integer', 'min': 1, 'max': 10, 'default': 5}
        }
        
        # Valid config
        config = {
            'portfolio_size': 10000,
            'max_position_pct': 10,
            'symbols': ['AAPL', 'GOOGL']
        }
        
        result = validate_config_schema(config, schema)
        assert result['portfolio_size'] == 10000.0
        assert result['max_position_pct'] == 10.0
        assert result['symbols'] == ['AAPL', 'GOOGL']
        assert result['workers'] == 5  # Default value
        
        # Missing required field
        invalid_config = {
            'max_position_pct': 10
        }
        
        with pytest.raises(ValidationError) as exc_info:
            validate_config_schema(invalid_config, schema)
        assert "Missing required field: portfolio_size" in str(exc_info.value)

class TestValidationDecorator:
    """Test validation decorator"""
    
    def test_validate_inputs_decorator(self):
        """Test input validation decorator"""
        @validate_inputs(
            symbol=lambda x: Validators.validate_stock_symbol(x, allow_lowercase=True),
            quantity=lambda x: Validators.validate_integer(x, min_value=1, name="quantity")
        )
        def place_order(symbol, quantity):
            return f"Order: {quantity} shares of {symbol}"
        
        # Valid inputs
        result = place_order('aapl', 100)
        assert result == "Order: 100 shares of AAPL"
        
        # Invalid symbol
        with pytest.raises(ValidationError) as exc_info:
            place_order('INVALID123', 100)
        assert "Invalid symbol" in str(exc_info.value)
        
        # Invalid quantity
        with pytest.raises(ValidationError) as exc_info:
            place_order('AAPL', 0)
        assert "Invalid quantity" in str(exc_info.value)