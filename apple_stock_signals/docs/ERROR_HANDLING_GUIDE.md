# Error Handling Guide

## Overview

This guide explains the comprehensive error handling system implemented in the trading system, including retry logic, circuit breakers, and rate limiting.

## Error Handling Components

### 1. Retry Logic

The system implements exponential backoff with configurable retry policies:

```python
from utils.error_handler import retry_with_backoff, RetryConfig

# Basic usage with default settings
@retry_with_backoff()
def fetch_data():
    return requests.get('https://api.example.com/data')

# Custom configuration
custom_config = RetryConfig(
    max_retries=5,
    initial_delay=2.0,
    max_delay=120.0,
    exponential_base=3.0,
    jitter=True
)

@retry_with_backoff(custom_config)
def fetch_critical_data():
    return requests.get('https://api.example.com/critical')
```

### 2. Rate Limiting

Prevents API rate limit violations:

```python
from utils.error_handler import rate_limited

# Limit to 10 calls per second
@rate_limited(calls_per_second=10)
def api_call():
    return requests.get('https://api.example.com/data')
```

### 3. Circuit Breaker Pattern

Prevents cascading failures:

```python
from utils.error_handler import CircuitBreaker

breaker = CircuitBreaker(
    failure_threshold=5,
    recovery_timeout=60,
    expected_exception=RequestException
)

@retry_with_backoff(circuit_breaker=breaker)
def protected_call():
    return external_service.call()
```

### 4. Error Categories

The system classifies errors into categories:

- **Transient Errors**: Automatically retried
  - ConnectionError
  - Timeout
  - ReadTimeout
  - ConnectTimeout
  
- **Rate Limit Errors**: Special handling with backoff
  - HTTP 429 (Too Many Requests)
  - API-specific rate limit responses
  
- **Non-Recoverable Errors**: No retry
  - ValueError
  - TypeError
  - KeyError
  - AttributeError

## Implementation Examples

### Data Fetcher with Error Handling

```python
class DataFetcher:
    @retry_with_backoff(RetryConfig(max_retries=3))
    @rate_limited(calls_per_second=2.0)
    def fetch_stock_data(self, symbol):
        """Fetch stock data with automatic retry and rate limiting"""
        response = requests.get(f"https://api.example.com/stock/{symbol}")
        response.raise_for_status()
        return response.json()
```

### Safe Request Wrapper

```python
from utils.error_handler import safe_request

# Make HTTP request with built-in retry logic
response = safe_request(
    method='GET',
    url='https://api.example.com/data',
    timeout=30.0,
    retry_config=RetryConfig(max_retries=5)
)

if response:
    data = response.json()
else:
    # Handle failure after all retries
    data = get_fallback_data()
```

## Configuration

### RetryConfig Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| max_retries | 3 | Maximum number of retry attempts |
| initial_delay | 1.0 | Initial delay between retries (seconds) |
| max_delay | 60.0 | Maximum delay between retries (seconds) |
| exponential_base | 2.0 | Base for exponential backoff calculation |
| jitter | True | Add randomness to prevent thundering herd |

### Circuit Breaker Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| failure_threshold | 5 | Number of failures before opening circuit |
| recovery_timeout | 60 | Seconds before attempting to close circuit |
| expected_exception | Exception | Exception type to monitor |

## Best Practices

### 1. Choose Appropriate Retry Strategies

```python
# For critical data with potential network issues
@retry_with_backoff(RetryConfig(
    max_retries=5,
    initial_delay=2.0,
    max_delay=300.0  # 5 minutes
))
def fetch_critical_market_data():
    pass

# For rate-limited APIs
@retry_with_backoff(RetryConfig(
    max_retries=3,
    initial_delay=1.0,
    exponential_base=2.0
))
@rate_limited(calls_per_second=1.0)
def fetch_rate_limited_data():
    pass
```

### 2. Handle Specific Error Types

```python
@retry_with_backoff(RetryConfig(
    retry_on=(ConnectionError, Timeout, HTTPError),
    dont_retry_on=(ValueError, KeyError)
))
def fetch_with_specific_retry():
    pass
```

### 3. Implement Fallback Mechanisms

```python
def get_stock_price(symbol):
    # Try primary source
    try:
        return fetch_from_primary_api(symbol)
    except Exception:
        pass
    
    # Try secondary source
    try:
        return fetch_from_secondary_api(symbol)
    except Exception:
        pass
    
    # Return cached/default value
    return get_cached_price(symbol)
```

### 4. Monitor and Log Errors

```python
import logging

logger = logging.getLogger(__name__)

@retry_with_backoff()
def monitored_function():
    try:
        result = external_call()
        logger.info(f"Successful call: {result}")
        return result
    except Exception as e:
        logger.error(f"Failed after retries: {e}")
        raise
```

## Common Patterns

### 1. Cascading Retry with Different Strategies

```python
# Fast retry for transient errors
@retry_with_backoff(RetryConfig(
    max_retries=3,
    initial_delay=0.5,
    max_delay=5.0
))
def quick_retry_fetch():
    return api_call()

# Slower retry for the wrapper
@retry_with_backoff(RetryConfig(
    max_retries=2,
    initial_delay=10.0,
    max_delay=60.0
))
def robust_fetch():
    try:
        return quick_retry_fetch()
    except Exception:
        # Try alternative method
        return alternative_fetch()
```

### 2. Handling Rate Limit Headers

```python
def handle_rate_limit_response(response):
    if response.status_code == 429:
        retry_after = response.headers.get('Retry-After', 60)
        time.sleep(int(retry_after))
        raise requests.exceptions.HTTPError("Rate limited")
```

### 3. Progressive Degradation

```python
class DataService:
    def __init__(self):
        self.circuit_breaker = CircuitBreaker()
    
    def get_data(self):
        # Try real-time data
        if self.circuit_breaker.state != 'open':
            try:
                return self.fetch_realtime_data()
            except Exception:
                pass
        
        # Fall back to cached data
        return self.get_cached_data()
```

## Testing Error Handling

### Unit Test Example

```python
def test_retry_on_connection_error():
    call_count = 0
    
    @retry_with_backoff(RetryConfig(max_retries=2, initial_delay=0.01))
    def flaky_function():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ConnectionError()
        return "success"
    
    result = flaky_function()
    assert result == "success"
    assert call_count == 3
```

### Integration Test Example

```python
@patch('requests.get')
def test_api_retry_integration(mock_get):
    # Simulate network failure then success
    mock_get.side_effect = [
        ConnectionError(),
        Mock(status_code=200, json=lambda: {"price": 100})
    ]
    
    fetcher = DataFetcher()
    result = fetcher.fetch_stock_data("AAPL")
    
    assert result["price"] == 100
    assert mock_get.call_count == 2
```

## Monitoring and Alerts

### Metrics to Track

1. **Retry Metrics**
   - Total retry attempts
   - Success rate after retries
   - Average retry count per request

2. **Circuit Breaker Metrics**
   - Circuit state changes
   - Time spent in open state
   - Recovery success rate

3. **Rate Limit Metrics**
   - Requests blocked by rate limiter
   - Average wait time
   - Peak request rates

### Example Monitoring Integration

```python
from utils.error_handler import retry_with_backoff
import prometheus_client

retry_counter = prometheus_client.Counter(
    'api_retries_total',
    'Total number of API retry attempts',
    ['endpoint', 'result']
)

@retry_with_backoff()
def monitored_api_call(endpoint):
    try:
        result = make_api_call(endpoint)
        retry_counter.labels(endpoint=endpoint, result='success').inc()
        return result
    except Exception as e:
        retry_counter.labels(endpoint=endpoint, result='failure').inc()
        raise
```

## Troubleshooting

### Common Issues

1. **Too Many Retries**
   - Check if error is actually transient
   - Verify retry configuration
   - Consider circuit breaker

2. **Rate Limit Violations**
   - Verify rate limiter configuration
   - Check for parallel requests
   - Monitor actual request rate

3. **Circuit Breaker Won't Close**
   - Check recovery timeout
   - Verify service health
   - Look for persistent errors

### Debug Mode

Enable detailed logging:

```python
import logging

# Enable debug logging for error handler
logging.getLogger('utils.error_handler').setLevel(logging.DEBUG)

# This will show:
# - Retry attempts and delays
# - Circuit breaker state changes
# - Rate limiter wait times
```

## Summary

The error handling system provides:

1. **Reliability**: Automatic retry for transient failures
2. **Stability**: Circuit breakers prevent cascade failures
3. **Compliance**: Rate limiting prevents API violations
4. **Flexibility**: Configurable strategies for different scenarios
5. **Observability**: Built-in logging and monitoring hooks

Always consider the specific requirements of each external service when configuring error handling strategies.