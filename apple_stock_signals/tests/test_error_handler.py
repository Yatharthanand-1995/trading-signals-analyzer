#!/usr/bin/env python3
"""
Unit tests for Error Handler utilities
"""

import pytest
import time
import requests
from unittest.mock import Mock, patch, MagicMock
from requests.exceptions import ConnectionError, Timeout, HTTPError

# Import the error handler module
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.error_handler import (
    RetryConfig,
    CircuitBreaker,
    calculate_backoff_delay,
    is_retryable_error,
    retry_with_backoff,
    handle_api_errors,
    rate_limited,
    RateLimiter,
    safe_request
)

class TestRetryConfig:
    """Test suite for RetryConfig"""
    
    def test_default_config(self):
        """Test default retry configuration"""
        config = RetryConfig()
        
        assert config.max_retries == 3
        assert config.initial_delay == 1.0
        assert config.max_delay == 60.0
        assert config.exponential_base == 2.0
        assert config.jitter is True
    
    def test_custom_config(self):
        """Test custom retry configuration"""
        config = RetryConfig(
            max_retries=5,
            initial_delay=2.0,
            max_delay=120.0,
            exponential_base=3.0,
            jitter=False
        )
        
        assert config.max_retries == 5
        assert config.initial_delay == 2.0
        assert config.max_delay == 120.0
        assert config.exponential_base == 3.0
        assert config.jitter is False

class TestBackoffCalculation:
    """Test suite for backoff delay calculation"""
    
    def test_exponential_backoff_without_jitter(self):
        """Test exponential backoff calculation without jitter"""
        config = RetryConfig(initial_delay=1.0, exponential_base=2.0, jitter=False)
        
        # Test exponential growth
        assert calculate_backoff_delay(0, config) == 1.0  # 1 * 2^0
        assert calculate_backoff_delay(1, config) == 2.0  # 1 * 2^1
        assert calculate_backoff_delay(2, config) == 4.0  # 1 * 2^2
        assert calculate_backoff_delay(3, config) == 8.0  # 1 * 2^3
    
    def test_max_delay_cap(self):
        """Test that delay is capped at max_delay"""
        config = RetryConfig(
            initial_delay=10.0,
            exponential_base=2.0,
            max_delay=20.0,
            jitter=False
        )
        
        # Should be capped at 20
        assert calculate_backoff_delay(3, config) == 20.0  # Would be 80 without cap
    
    def test_jitter_adds_randomness(self):
        """Test that jitter adds randomness to delay"""
        config = RetryConfig(initial_delay=10.0, jitter=True)
        
        delays = [calculate_backoff_delay(1, config) for _ in range(10)]
        
        # With jitter, not all delays should be the same
        assert len(set(delays)) > 1
        
        # All delays should be within expected range (±25%)
        base_delay = 20.0  # 10 * 2^1
        for delay in delays:
            assert base_delay * 0.75 <= delay <= base_delay * 1.25

class TestRetryableError:
    """Test suite for retryable error detection"""
    
    def test_transient_errors_are_retryable(self):
        """Test that transient errors are retryable"""
        config = RetryConfig()
        
        assert is_retryable_error(ConnectionError(), config) is True
        assert is_retryable_error(Timeout(), config) is True
    
    def test_non_recoverable_errors_not_retryable(self):
        """Test that non-recoverable errors are not retryable"""
        config = RetryConfig()
        
        assert is_retryable_error(ValueError(), config) is False
        assert is_retryable_error(TypeError(), config) is False
        assert is_retryable_error(KeyError(), config) is False
    
    def test_http_429_is_retryable(self):
        """Test that HTTP 429 (rate limit) is retryable"""
        config = RetryConfig()
        
        response = Mock()
        response.status_code = 429
        error = HTTPError(response=response)
        
        assert is_retryable_error(error, config) is True
    
    def test_http_5xx_is_retryable(self):
        """Test that HTTP 5xx errors are retryable"""
        config = RetryConfig()
        
        for status_code in [500, 502, 503, 504]:
            response = Mock()
            response.status_code = status_code
            error = HTTPError(response=response)
            
            assert is_retryable_error(error, config) is True
    
    def test_http_4xx_not_retryable(self):
        """Test that HTTP 4xx errors (except 429) are not retryable"""
        config = RetryConfig()
        
        for status_code in [400, 401, 403, 404]:
            response = Mock()
            response.status_code = status_code
            error = HTTPError(response=response)
            
            assert is_retryable_error(error, config) is False

class TestRetryDecorator:
    """Test suite for retry decorator"""
    
    def test_successful_call_no_retry(self):
        """Test that successful calls don't retry"""
        call_count = 0
        
        @retry_with_backoff(RetryConfig(max_retries=3))
        def successful_function():
            nonlocal call_count
            call_count += 1
            return "success"
        
        result = successful_function()
        
        assert result == "success"
        assert call_count == 1
    
    def test_retries_on_transient_error(self):
        """Test that function retries on transient errors"""
        call_count = 0
        
        @retry_with_backoff(RetryConfig(max_retries=3, initial_delay=0.01))
        def failing_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Connection failed")
            return "success"
        
        result = failing_function()
        
        assert result == "success"
        assert call_count == 3
    
    def test_fails_after_max_retries(self):
        """Test that function fails after max retries"""
        call_count = 0
        
        @retry_with_backoff(RetryConfig(max_retries=2, initial_delay=0.01))
        def always_failing_function():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("Connection failed")
        
        with pytest.raises(ConnectionError):
            always_failing_function()
        
        assert call_count == 3  # Initial + 2 retries
    
    def test_no_retry_on_non_retryable_error(self):
        """Test that non-retryable errors don't trigger retries"""
        call_count = 0
        
        @retry_with_backoff(RetryConfig(max_retries=3))
        def value_error_function():
            nonlocal call_count
            call_count += 1
            raise ValueError("Invalid value")
        
        with pytest.raises(ValueError):
            value_error_function()
        
        assert call_count == 1  # No retries

class TestRateLimiter:
    """Test suite for rate limiter"""
    
    def test_rate_limiting(self):
        """Test that rate limiter enforces call rate"""
        limiter = RateLimiter(calls_per_second=10.0)
        
        start_time = time.time()
        
        # Make 5 calls
        for _ in range(5):
            limiter.wait_if_needed()
        
        elapsed = time.time() - start_time
        
        # Should take at least 0.4 seconds (5 calls at 10/sec = 0.5 sec)
        # Using 0.35 to account for timing variations
        assert elapsed >= 0.35
    
    def test_rate_limited_decorator(self):
        """Test rate limited decorator"""
        call_times = []
        
        @rate_limited(calls_per_second=10.0)
        def rate_limited_function():
            call_times.append(time.time())
            return "done"
        
        # Make 3 calls
        for _ in range(3):
            rate_limited_function()
        
        # Check spacing between calls
        if len(call_times) >= 2:
            for i in range(1, len(call_times)):
                time_diff = call_times[i] - call_times[i-1]
                assert time_diff >= 0.09  # Should be ~0.1 seconds

class TestCircuitBreaker:
    """Test suite for circuit breaker"""
    
    def test_circuit_breaker_closes_after_threshold(self):
        """Test that circuit breaker opens after failure threshold"""
        breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=1)
        
        def failing_function():
            raise Exception("Failed")
        
        # First failure
        with pytest.raises(Exception):
            breaker.call(failing_function)
        
        # Second failure - should open circuit
        with pytest.raises(Exception):
            breaker.call(failing_function)
        
        assert breaker.state == 'open'
        
        # Next call should fail immediately
        with pytest.raises(Exception) as exc_info:
            breaker.call(failing_function)
        
        assert "Circuit breaker is open" in str(exc_info.value)
    
    def test_circuit_breaker_recovers(self):
        """Test that circuit breaker recovers after timeout"""
        breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=0.1)
        
        def sometimes_working_function():
            if breaker.state == 'half-open':
                return "success"
            raise Exception("Failed")
        
        # Open the circuit
        with pytest.raises(Exception):
            breaker.call(sometimes_working_function)
        
        assert breaker.state == 'open'
        
        # Wait for recovery timeout
        time.sleep(0.15)
        
        # Should work now
        result = breaker.call(sometimes_working_function)
        assert result == "success"
        assert breaker.state == 'closed'

class TestHandleApiErrors:
    """Test suite for handle_api_errors decorator"""
    
    def test_returns_default_on_error(self):
        """Test that decorator returns default value on error"""
        @handle_api_errors(default_return="default", log_errors=False)
        def failing_function():
            raise requests.exceptions.ConnectionError()
        
        result = failing_function()
        assert result == "default"
    
    def test_successful_call_returns_result(self):
        """Test that successful calls return their result"""
        @handle_api_errors(default_return="default")
        def successful_function():
            return "success"
        
        result = successful_function()
        assert result == "success"

class TestSafeRequest:
    """Test suite for safe_request function"""
    
    @patch('requests.request')
    def test_successful_request(self, mock_request):
        """Test successful request"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": "test"}
        mock_request.return_value = mock_response
        
        response = safe_request('GET', 'http://example.com')
        
        assert response is not None
        assert response.status_code == 200
    
    @patch('requests.request')
    def test_retries_on_failure(self, mock_request):
        """Test that safe_request retries on failure"""
        # First two calls fail, third succeeds
        mock_request.side_effect = [
            ConnectionError(),
            ConnectionError(),
            Mock(status_code=200)
        ]
        
        response = safe_request(
            'GET', 
            'http://example.com',
            retry_config=RetryConfig(max_retries=2, initial_delay=0.01)
        )
        
        assert response is not None
        assert mock_request.call_count == 3
    
    @patch('requests.request')
    def test_returns_none_after_all_retries_fail(self, mock_request):
        """Test that safe_request returns None when all retries fail"""
        mock_request.side_effect = ConnectionError()
        
        response = safe_request(
            'GET',
            'http://example.com',
            retry_config=RetryConfig(max_retries=2, initial_delay=0.01)
        )
        
        assert response is None
        assert mock_request.call_count == 3  # Initial + 2 retries