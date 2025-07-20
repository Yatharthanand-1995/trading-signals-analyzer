# Testing Guide

## Overview

This guide explains the testing framework for the trading system and how to write and run tests.

## Test Structure

```
tests/
├── __init__.py              # Test package marker
├── conftest.py              # Pytest configuration and fixtures
├── test_market_regime_detector.py
├── test_volatility_position_sizing.py
├── test_trailing_stop_manager.py
└── test_env_config.py       # Configuration tests
```

## Running Tests

### Run All Tests
```bash
python3 run_tests.py
```

### Run Specific Test File
```bash
python3 run_tests.py tests/test_env_config.py
```

### Run with Pytest Directly
```bash
# Basic run
pytest

# Verbose output
pytest -v

# Run specific test
pytest tests/test_market_regime_detector.py::TestMarketRegimeDetector::test_initialization

# Run with coverage
pytest --cov=core_scripts --cov-report=html
```

## Test Coverage

### Current Coverage Status

| Module | Coverage | Status |
|--------|----------|--------|
| config.env_config | 90%+ | ✅ Good |
| core_scripts.market_regime_detector | 0% | 🔴 Needs work |
| core_scripts.volatility_position_sizing | 0% | 🔴 Needs work |
| core_scripts.trailing_stop_manager | 0% | 🔴 Needs work |

### View Coverage Report
```bash
# After running tests
open coverage_report/index.html
```

## Writing Tests

### Basic Test Structure

```python
import pytest
from core_scripts.your_module import YourClass

class TestYourClass:
    def setup_method(self):
        """Set up test fixtures"""
        self.instance = YourClass()
    
    def test_basic_functionality(self):
        """Test basic functionality"""
        result = self.instance.method()
        assert result is not None
        assert isinstance(result, dict)
    
    def test_edge_case(self):
        """Test edge cases"""
        with pytest.raises(ValueError):
            self.instance.method(invalid_input)
```

### Using Fixtures

Fixtures are defined in `conftest.py`:

```python
@pytest.fixture
def sample_stock_data():
    """Returns sample stock data for testing"""
    # Returns DataFrame with OHLCV data
    
def test_with_fixture(sample_stock_data):
    """Use fixture in test"""
    assert len(sample_stock_data) > 0
```

### Available Fixtures

- `sample_stock_data`: 100 days of OHLCV data
- `sample_portfolio`: Portfolio configuration
- `sample_market_regime`: Market regime data
- `sample_signals`: Trading signals
- `mock_api_response`: Mock API response

## Test Categories

### Unit Tests
Test individual components in isolation:
- Single function/method behavior
- Edge cases and error handling
- Input validation

### Integration Tests
Test component interactions:
- Data flow between modules
- System configuration
- API integrations

### Performance Tests
Test system performance:
- Execution time
- Memory usage
- Scalability

## Common Test Patterns

### Testing Exceptions
```python
def test_invalid_input():
    with pytest.raises(ValueError) as exc_info:
        function_that_should_fail()
    assert "specific error message" in str(exc_info.value)
```

### Testing Numeric Values
```python
def test_calculations():
    result = calculate_something()
    assert abs(result - expected) < 0.001  # Float comparison
    assert 0 <= result <= 1  # Range check
```

### Mocking External Dependencies
```python
from unittest.mock import patch

@patch('yfinance.Ticker')
def test_with_mock(mock_ticker):
    mock_ticker.return_value.history.return_value = sample_data
    result = function_using_yfinance()
    assert result is not None
```

## Debugging Failed Tests

### Common Issues

1. **Import Errors**
   - Check PYTHONPATH
   - Ensure __init__.py files exist
   - Use absolute imports in tests

2. **Fixture Not Found**
   - Check fixture name spelling
   - Ensure conftest.py is in tests/
   - Fixture scope issues

3. **Assertion Failures**
   - Print actual vs expected values
   - Check data types
   - Consider floating point precision

### Debug Commands

```bash
# Show print statements
pytest -s

# Stop on first failure
pytest -x

# Run last failed tests
pytest --lf

# Detailed failure info
pytest -vv
```

## Continuous Integration

### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
    - name: Run tests
      run: |
        pytest --cov=core_scripts --cov-report=xml
    - name: Upload coverage
      uses: codecov/codecov-action@v1
```

## Best Practices

1. **Test Naming**
   - Use descriptive names: `test_calculate_position_size_with_high_volatility`
   - Group related tests in classes
   - Follow pattern: `test_<what>_<condition>_<expected>`

2. **Test Independence**
   - Each test should be independent
   - Use setup/teardown methods
   - Don't rely on test execution order

3. **Test Data**
   - Use fixtures for common data
   - Keep test data minimal but realistic
   - Consider edge cases

4. **Assertions**
   - One logical assertion per test
   - Use descriptive assertion messages
   - Test both success and failure cases

5. **Performance**
   - Keep tests fast (<1 second each)
   - Mock external dependencies
   - Use smaller datasets for tests

## Adding New Tests

1. Create test file: `test_<module_name>.py`
2. Import module to test
3. Create test class: `Test<ClassName>`
4. Write test methods: `test_<functionality>`
5. Run tests to verify
6. Check coverage report

## Test Maintenance

- Run tests before committing
- Update tests when changing functionality
- Remove obsolete tests
- Keep test coverage above 80%
- Review and refactor tests periodically

## Troubleshooting

### Tests Not Found
```bash
# Ensure you're in the project root
cd /path/to/apple_stock_signals

# Check Python path
python3 -c "import sys; print(sys.path)"
```

### Coverage Not Working
```bash
# Reinstall pytest-cov
pip install --upgrade pytest-cov

# Run with explicit coverage
python3 -m pytest --cov
```

### Slow Tests
- Profile with `pytest --durations=10`
- Mock external API calls
- Use smaller test datasets
- Run tests in parallel: `pytest -n auto`

## Summary

A comprehensive test suite:
- Ensures code reliability
- Documents expected behavior
- Enables confident refactoring
- Catches bugs early
- Improves code quality

Aim for 80%+ test coverage and run tests frequently during development.