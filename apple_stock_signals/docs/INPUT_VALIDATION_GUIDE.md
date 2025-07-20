# Input Validation Guide

## Overview

This guide documents the comprehensive input validation system that ensures data integrity and security throughout the trading system.

## Validation Components

### 1. Core Validators

The `utils/validators.py` module provides validation functions for all input types:

#### Stock Symbol Validation
```python
from utils.validators import Validators, ValidationError

# Validate single symbol
try:
    symbol = Validators.validate_stock_symbol('AAPL')  # Returns 'AAPL'
    symbol = Validators.validate_stock_symbol('aapl', allow_lowercase=True)  # Returns 'AAPL'
except ValidationError as e:
    print(f"Invalid symbol: {e}")

# Validate list of symbols
symbols = Validators.validate_stock_list(['AAPL', 'googl', 'TSLA'])  # Returns ['AAPL', 'GOOGL', 'TSLA']
```

**Rules:**
- 1-5 uppercase letters only
- Converts lowercase if allowed
- Removes duplicates from lists
- Blocks test symbols (TEST, DEMO, NULL, NONE)

#### Numeric Validation
```python
# Percentages (0-100 by default)
risk_pct = Validators.validate_percentage(2.5, name="risk_percentage")

# Positive numbers
amount = Validators.validate_positive_number(10000, name="portfolio_size", max_value=1000000)

# Integers with range
quantity = Validators.validate_integer(100, min_value=1, max_value=10000, name="quantity")
```

#### Date Validation
```python
# Single date
trade_date = Validators.validate_date('2024-01-15')

# Date range with constraints
start, end = Validators.validate_date_range('2024-01-01', '2024-12-31', max_days=365)
```

#### File Path Validation
```python
# Validate file path with security checks
config_path = Validators.validate_file_path(
    'config/settings.json',
    must_exist=True,
    allowed_extensions=['.json'],
    base_directory='./config'  # Prevents directory traversal
)

# Load and validate JSON
config = Validators.validate_json_file('config/stocks_config.json')
```

### 2. Validation Decorator

Use the `@validate_inputs` decorator for automatic parameter validation:

```python
from utils.validators import validate_inputs, Validators

@validate_inputs(
    symbol=lambda x: Validators.validate_stock_symbol(x, allow_lowercase=True),
    quantity=lambda x: Validators.validate_integer(x, min_value=1, name="quantity"),
    price=lambda x: Validators.validate_positive_number(x, name="price")
)
def place_order(symbol: str, quantity: int, price: float):
    # All inputs are validated before function executes
    return f"Placing order: {quantity} shares of {symbol} at ${price}"

# Usage
place_order('aapl', 100, 150.50)  # Symbol converted to 'AAPL'
place_order('AAPL', 0, 150.50)    # Raises ValidationError: quantity must be at least 1
```

### 3. Configuration Schema Validation

Define schemas for complex configuration validation:

```python
from utils.validators import validate_config_schema, SCHEMAS

# Define schema
schema = {
    'portfolio_size': {'type': 'float', 'min': 0, 'required': True},
    'max_position_pct': {'type': 'percentage', 'min': 0, 'max': 100, 'required': True},
    'risk_per_trade_pct': {'type': 'percentage', 'min': 0, 'max': 10, 'required': True},
    'symbols': {'type': 'list', 'items': 'stock_symbol', 'required': True},
    'workers': {'type': 'integer', 'min': 1, 'max': 20, 'default': 5}
}

# Validate configuration
config = {
    'portfolio_size': 10000,
    'max_position_pct': 10,
    'risk_per_trade_pct': 2,
    'symbols': ['AAPL', 'GOOGL', 'TSLA']
}

validated_config = validate_config_schema(config, schema)
```

### 4. Configuration Loader

The `utils/config_loader.py` provides validated configuration loading:

```python
from utils.config_loader import load_validated_stocks_config, load_validated_active_stocks

# Load and validate stocks configuration
config = load_validated_stocks_config()

# Load and validate active stocks list
active_stocks = load_validated_active_stocks()

# Validate trading parameters
from utils.config_loader import validate_trading_params

params = validate_trading_params({
    'symbol': 'aapl',
    'quantity': 100,
    'price': 150.50,
    'action': 'buy'
})
```

## Implementation Examples

### 1. Paper Trading with Validation

```python
from paper_trading.paper_trader import PaperTradingAccount

# Constructor validates inputs automatically
account = PaperTradingAccount(
    initial_balance=10000,  # Must be positive, max 10M
    account_name="test_account"  # Sanitized for security
)

# Execute trade with validation
account.execute_trade(
    symbol='AAPL',      # Validated stock symbol
    action='BUY',       # Must be BUY/SELL/HOLD
    quantity=100,       # Must be positive integer
    price=150.50,       # Must be positive number
    commission=5.00     # Must be non-negative
)
```

### 2. Command-Line Validation

```python
# In ml_enhanced_analyzer.py
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', type=str)
    args = parser.parse_args()
    
    # Validate symbol if provided
    if args.symbol:
        try:
            args.symbol = Validators.validate_stock_symbol(args.symbol, allow_lowercase=True)
        except ValidationError as e:
            print(f"Error: {e}")
            sys.exit(1)
```

### 3. Shell Script Validation

For `run_analysis.sh`, use the validation script:

```bash
# Validate command
result=$(python3 utils/validate_shell_input.py "$1" "$2")
if [[ $result == ERROR:* ]]; then
    echo "Invalid input: ${result#ERROR:}"
    exit 1
fi
```

## Validation Rules Summary

### Stock Symbols
- Format: 1-5 uppercase letters (A-Z)
- No numbers or special characters
- Automatic uppercase conversion available
- Blocked symbols: TEST, DEMO, NULL, NONE

### Numeric Values
- **Percentages**: 0-100 by default, customizable range
- **Positive Numbers**: > 0 by default, allow_zero option
- **Integers**: Whole numbers with optional min/max
- **Prices**: Positive decimals, reasonable maximums

### Dates
- Format: YYYY-MM-DD
- Range validation with max_days constraint
- Supports date, datetime, and string inputs

### File Paths
- Existence checking
- Extension validation
- Directory traversal prevention
- Base directory constraints

### Trading Parameters
- **Actions**: BUY, SELL, HOLD only
- **Quantities**: Positive integers
- **Prices**: Positive decimals
- **Commission**: Non-negative decimals

### API Keys
- Format validation per API
- Placeholder detection
- Required/optional handling
- Whitespace trimming

## Error Handling

All validation errors raise `ValidationError` with descriptive messages:

```python
try:
    symbol = Validators.validate_stock_symbol('AAPL123')
except ValidationError as e:
    # Error: Invalid stock symbol 'AAPL123'. Must be 1-5 uppercase letters
    logger.error(f"Validation failed: {e}")
    # Handle error appropriately
```

## Security Considerations

1. **Path Traversal Prevention**: File validation prevents directory traversal attacks
2. **SQL Injection Prevention**: All database inputs are parameterized
3. **Command Injection Prevention**: Shell inputs are validated before execution
4. **Account Name Sanitization**: Special characters removed from account names
5. **API Key Security**: Placeholder values detected and rejected

## Best Practices

1. **Validate Early**: Validate inputs as soon as they enter the system
2. **Use Decorators**: Apply `@validate_inputs` to functions accepting user input
3. **Consistent Error Messages**: Use ValidationError for all validation failures
4. **Log Validation Failures**: Record validation errors for security monitoring
5. **Default Values**: Provide sensible defaults in schemas
6. **Type Hints**: Combine with Python type hints for clarity

## Testing Validation

Run validation tests:
```bash
python3 -m pytest tests/test_validators.py -v
```

Test coverage includes:
- Stock symbol validation (valid, invalid, blocked)
- Numeric validation (percentages, positive numbers, integers)
- Date validation (single dates, ranges)
- File path validation (security, extensions)
- API key validation (format, placeholders)
- Trading action validation
- Configuration schema validation
- Decorator functionality

## Extending Validation

To add new validators:

1. Add validation function to `Validators` class
2. Create corresponding tests in `test_validators.py`
3. Update schema definitions if needed
4. Document validation rules

Example:
```python
@staticmethod
def validate_email(email: str) -> str:
    """Validate email address"""
    email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    email = email.strip().lower()
    
    if not email_pattern.match(email):
        raise ValidationError(f"Invalid email address: {email}")
    
    return email
```

## Summary

The input validation system provides:
- **Data Integrity**: Ensures all inputs meet expected formats
- **Security**: Prevents injection attacks and path traversal
- **User Experience**: Clear error messages for invalid inputs
- **Maintainability**: Centralized validation logic
- **Testability**: Comprehensive test coverage

Always validate user inputs to maintain system reliability and security.