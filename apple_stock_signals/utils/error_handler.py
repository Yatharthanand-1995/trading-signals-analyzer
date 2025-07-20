#!/usr/bin/env python3
"""
Advanced Error Handling and Retry Utilities
Provides decorators and utilities for robust error handling with retries
"""

import time
import logging
import functools
import random
from typing import Callable, Any, Dict, List, Optional, Type, Union, Tuple
from datetime import datetime, timedelta
import requests
from requests.exceptions import RequestException, Timeout, ConnectionError, HTTPError

# Configure logging
logger = logging.getLogger(__name__)

# Error categories
TRANSIENT_ERRORS = (
    ConnectionError,
    Timeout,
    requests.exceptions.ReadTimeout,
    requests.exceptions.ConnectTimeout,
)

RATE_LIMIT_ERRORS = (
    requests.exceptions.HTTPError,
)

NON_RECOVERABLE_ERRORS = (
    ValueError,
    TypeError,
    KeyError,
    AttributeError,
)

class RetryConfig:
    """Configuration for retry behavior"""
    def __init__(
        self,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        retry_on: Optional[Tuple[Type[Exception], ...]] = None,
        dont_retry_on: Optional[Tuple[Type[Exception], ...]] = None,
    ):
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.retry_on = retry_on or TRANSIENT_ERRORS
        self.dont_retry_on = dont_retry_on or NON_RECOVERABLE_ERRORS

class CircuitBreaker:
    """Circuit breaker pattern implementation"""
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: Type[Exception] = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'closed'  # closed, open, half-open
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        if self.state == 'open':
            if self._should_attempt_reset():
                self.state = 'half-open'
            else:
                raise Exception(f"Circuit breaker is open. Service unavailable.")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset"""
        return (
            self.last_failure_time and
            datetime.now() - self.last_failure_time > timedelta(seconds=self.recovery_timeout)
        )
    
    def _on_success(self):
        """Reset circuit breaker on success"""
        self.failure_count = 0
        self.state = 'closed'
        self.last_failure_time = None
    
    def _on_failure(self):
        """Handle failure"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        if self.failure_count >= self.failure_threshold:
            self.state = 'open'

def calculate_backoff_delay(
    attempt: int,
    config: RetryConfig
) -> float:
    """Calculate exponential backoff delay with optional jitter"""
    delay = min(
        config.initial_delay * (config.exponential_base ** attempt),
        config.max_delay
    )
    
    if config.jitter:
        # Add random jitter (±25% of delay)
        jitter_range = delay * 0.25
        delay += random.uniform(-jitter_range, jitter_range)
    
    return max(0, delay)

def is_retryable_error(
    error: Exception,
    config: RetryConfig
) -> bool:
    """Determine if an error is retryable"""
    # Check if it's explicitly non-retryable
    if isinstance(error, config.dont_retry_on):
        return False
    
    # Check if it's explicitly retryable
    if isinstance(error, config.retry_on):
        return True
    
    # Check for HTTP rate limit errors (429)
    if isinstance(error, HTTPError):
        if error.response and error.response.status_code == 429:
            return True
        # 5xx errors are typically retryable
        if error.response and 500 <= error.response.status_code < 600:
            return True
    
    return False

def retry_with_backoff(
    config: Optional[RetryConfig] = None,
    circuit_breaker: Optional[CircuitBreaker] = None
) -> Callable:
    """
    Decorator for retrying functions with exponential backoff
    
    Usage:
        @retry_with_backoff()
        def fetch_data():
            return requests.get('https://api.example.com/data')
        
        @retry_with_backoff(RetryConfig(max_retries=5, initial_delay=2.0))
        def fetch_important_data():
            return requests.get('https://api.example.com/important')
    """
    if config is None:
        config = RetryConfig()
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_error = None
            
            for attempt in range(config.max_retries + 1):
                try:
                    # Use circuit breaker if provided
                    if circuit_breaker:
                        return circuit_breaker.call(func, *args, **kwargs)
                    else:
                        return func(*args, **kwargs)
                    
                except Exception as e:
                    last_error = e
                    
                    # Check if we should retry
                    if attempt >= config.max_retries:
                        logger.error(
                            f"Function {func.__name__} failed after {config.max_retries + 1} attempts: {str(e)}"
                        )
                        raise
                    
                    if not is_retryable_error(e, config):
                        logger.error(
                            f"Non-retryable error in {func.__name__}: {str(e)}"
                        )
                        raise
                    
                    # Calculate delay
                    delay = calculate_backoff_delay(attempt, config)
                    
                    # Check for Retry-After header in HTTP errors
                    if isinstance(e, HTTPError) and e.response:
                        retry_after = e.response.headers.get('Retry-After')
                        if retry_after:
                            try:
                                delay = float(retry_after)
                            except ValueError:
                                pass
                    
                    logger.warning(
                        f"Retrying {func.__name__} after {delay:.2f}s "
                        f"(attempt {attempt + 1}/{config.max_retries + 1}): {str(e)}"
                    )
                    
                    time.sleep(delay)
            
            # Should never reach here, but just in case
            raise last_error
        
        return wrapper
    return decorator

def handle_api_errors(
    default_return: Any = None,
    log_errors: bool = True
) -> Callable:
    """
    Decorator for handling API errors gracefully
    
    Usage:
        @handle_api_errors(default_return={})
        def fetch_data():
            return requests.get('https://api.example.com/data').json()
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except requests.exceptions.HTTPError as e:
                if log_errors:
                    logger.error(f"HTTP error in {func.__name__}: {e.response.status_code} - {str(e)}")
                return default_return
            except requests.exceptions.RequestException as e:
                if log_errors:
                    logger.error(f"Request error in {func.__name__}: {str(e)}")
                return default_return
            except Exception as e:
                if log_errors:
                    logger.error(f"Unexpected error in {func.__name__}: {str(e)}")
                return default_return
        
        return wrapper
    return decorator

class RateLimiter:
    """Simple rate limiter implementation"""
    def __init__(self, calls_per_second: float = 1.0):
        self.calls_per_second = calls_per_second
        self.min_interval = 1.0 / calls_per_second
        self.last_call_time = 0
    
    def wait_if_needed(self):
        """Wait if necessary to respect rate limit"""
        current_time = time.time()
        time_since_last_call = current_time - self.last_call_time
        
        if time_since_last_call < self.min_interval:
            sleep_time = self.min_interval - time_since_last_call
            time.sleep(sleep_time)
        
        self.last_call_time = time.time()

def rate_limited(calls_per_second: float = 1.0) -> Callable:
    """
    Decorator for rate limiting function calls
    
    Usage:
        @rate_limited(calls_per_second=10)
        def api_call():
            return requests.get('https://api.example.com/data')
    """
    limiter = RateLimiter(calls_per_second)
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            limiter.wait_if_needed()
            return func(*args, **kwargs)
        
        return wrapper
    return decorator

def validate_response(
    response: requests.Response,
    expected_status: Union[int, List[int]] = 200
) -> None:
    """
    Validate HTTP response
    
    Args:
        response: Response object to validate
        expected_status: Expected status code(s)
    
    Raises:
        HTTPError: If status code doesn't match expected
    """
    if isinstance(expected_status, int):
        expected_status = [expected_status]
    
    if response.status_code not in expected_status:
        response.raise_for_status()

def safe_request(
    method: str,
    url: str,
    retry_config: Optional[RetryConfig] = None,
    timeout: float = 30.0,
    **kwargs
) -> Optional[requests.Response]:
    """
    Make HTTP request with built-in retry logic
    
    Args:
        method: HTTP method (GET, POST, etc.)
        url: URL to request
        retry_config: Retry configuration
        timeout: Request timeout
        **kwargs: Additional arguments for requests
    
    Returns:
        Response object or None if all retries failed
    """
    if retry_config is None:
        retry_config = RetryConfig()
    
    @retry_with_backoff(retry_config)
    def make_request():
        response = requests.request(
            method=method,
            url=url,
            timeout=timeout,
            **kwargs
        )
        response.raise_for_status()
        return response
    
    try:
        return make_request()
    except Exception as e:
        logger.error(f"Request failed after retries: {url} - {str(e)}")
        return None

# Example usage patterns
if __name__ == "__main__":
    # Example 1: Basic retry with default settings
    @retry_with_backoff()
    def fetch_stock_data(symbol: str):
        response = requests.get(f"https://api.example.com/stock/{symbol}")
        response.raise_for_status()
        return response.json()
    
    # Example 2: Custom retry configuration
    custom_config = RetryConfig(
        max_retries=5,
        initial_delay=2.0,
        max_delay=120.0,
        exponential_base=3.0
    )
    
    @retry_with_backoff(custom_config)
    def fetch_critical_data():
        response = requests.get("https://api.example.com/critical")
        response.raise_for_status()
        return response.json()
    
    # Example 3: Rate limited API calls
    @rate_limited(calls_per_second=10)
    @retry_with_backoff()
    def fetch_rate_limited_data(item_id: int):
        response = requests.get(f"https://api.example.com/items/{item_id}")
        response.raise_for_status()
        return response.json()
    
    # Example 4: Circuit breaker pattern
    api_circuit_breaker = CircuitBreaker(
        failure_threshold=3,
        recovery_timeout=30,
        expected_exception=RequestException
    )
    
    @retry_with_backoff(circuit_breaker=api_circuit_breaker)
    def fetch_with_circuit_breaker():
        response = requests.get("https://api.example.com/data")
        response.raise_for_status()
        return response.json()