#!/usr/bin/env python3
"""
Cache Manager for Performance Optimization
Provides in-memory and disk-based caching with TTL support
"""

import json
import pickle
import time
import hashlib
import os
from pathlib import Path
from typing import Any, Optional, Union, Callable, Dict
from datetime import datetime, timedelta
import logging
from functools import wraps
import threading
import sqlite3
from collections import OrderedDict

logger = logging.getLogger(__name__)

class CacheManager:
    """Manages caching for API responses and computed data"""
    
    def __init__(
        self,
        cache_dir: str = "cache",
        max_memory_items: int = 1000,
        default_ttl: int = 3600,  # 1 hour
        enable_disk_cache: bool = True
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.memory_cache: OrderedDict = OrderedDict()
        self.max_memory_items = max_memory_items
        self.default_ttl = default_ttl
        self.enable_disk_cache = enable_disk_cache
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Initialize disk cache database
        if self.enable_disk_cache:
            self._init_disk_cache()
        
        # Cache statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0
        }
    
    def _init_disk_cache(self):
        """Initialize SQLite database for disk cache"""
        db_path = self.cache_dir / "cache.db"
        self.db_conn = sqlite3.connect(str(db_path), check_same_thread=False)
        
        cursor = self.db_conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                value BLOB,
                expires_at REAL,
                created_at REAL,
                access_count INTEGER DEFAULT 0,
                last_accessed REAL
            )
        """)
        
        # Create index for expiration cleanup
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_expires_at ON cache(expires_at)
        """)
        
        self.db_conn.commit()
    
    def _generate_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments"""
        key_data = {
            'args': args,
            'kwargs': sorted(kwargs.items())
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        with self.lock:
            # Check memory cache first
            if key in self.memory_cache:
                entry = self.memory_cache[key]
                if entry['expires_at'] > time.time():
                    # Move to end (LRU)
                    self.memory_cache.move_to_end(key)
                    self.stats['hits'] += 1
                    return entry['value']
                else:
                    # Expired
                    del self.memory_cache[key]
            
            # Check disk cache
            if self.enable_disk_cache:
                value = self._get_from_disk(key)
                if value is not None:
                    # Promote to memory cache
                    self._set_memory_cache(key, value, self.default_ttl)
                    self.stats['hits'] += 1
                    return value
            
            self.stats['misses'] += 1
            return None
    
    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> None:
        """Set value in cache"""
        if ttl is None:
            ttl = self.default_ttl
        
        with self.lock:
            # Set in memory cache
            self._set_memory_cache(key, value, ttl)
            
            # Set in disk cache
            if self.enable_disk_cache:
                self._set_disk_cache(key, value, ttl)
    
    def _set_memory_cache(self, key: str, value: Any, ttl: int):
        """Set value in memory cache with LRU eviction"""
        expires_at = time.time() + ttl
        
        self.memory_cache[key] = {
            'value': value,
            'expires_at': expires_at
        }
        
        # LRU eviction
        if len(self.memory_cache) > self.max_memory_items:
            # Remove oldest item
            oldest_key = next(iter(self.memory_cache))
            del self.memory_cache[oldest_key]
            self.stats['evictions'] += 1
    
    def _get_from_disk(self, key: str) -> Optional[Any]:
        """Get value from disk cache"""
        cursor = self.db_conn.cursor()
        cursor.execute("""
            SELECT value, expires_at FROM cache
            WHERE key = ? AND expires_at > ?
        """, (key, time.time()))
        
        row = cursor.fetchone()
        if row:
            # Update access statistics
            cursor.execute("""
                UPDATE cache
                SET access_count = access_count + 1,
                    last_accessed = ?
                WHERE key = ?
            """, (time.time(), key))
            self.db_conn.commit()
            
            return pickle.loads(row[0])
        
        return None
    
    def _set_disk_cache(self, key: str, value: Any, ttl: int):
        """Set value in disk cache"""
        expires_at = time.time() + ttl
        created_at = time.time()
        value_blob = pickle.dumps(value)
        
        cursor = self.db_conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO cache
            (key, value, expires_at, created_at, last_accessed)
            VALUES (?, ?, ?, ?, ?)
        """, (key, value_blob, expires_at, created_at, created_at))
        
        self.db_conn.commit()
    
    def delete(self, key: str) -> None:
        """Delete value from cache"""
        with self.lock:
            # Remove from memory cache
            if key in self.memory_cache:
                del self.memory_cache[key]
            
            # Remove from disk cache
            if self.enable_disk_cache:
                cursor = self.db_conn.cursor()
                cursor.execute("DELETE FROM cache WHERE key = ?", (key,))
                self.db_conn.commit()
    
    def clear(self) -> None:
        """Clear all cache"""
        with self.lock:
            self.memory_cache.clear()
            
            if self.enable_disk_cache:
                cursor = self.db_conn.cursor()
                cursor.execute("DELETE FROM cache")
                self.db_conn.commit()
            
            self.stats = {'hits': 0, 'misses': 0, 'evictions': 0}
    
    def cleanup_expired(self) -> int:
        """Remove expired entries from cache"""
        with self.lock:
            # Memory cache cleanup
            expired_keys = []
            current_time = time.time()
            
            for key, entry in self.memory_cache.items():
                if entry['expires_at'] <= current_time:
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self.memory_cache[key]
            
            # Disk cache cleanup
            if self.enable_disk_cache:
                cursor = self.db_conn.cursor()
                cursor.execute("""
                    DELETE FROM cache WHERE expires_at <= ?
                """, (current_time,))
                self.db_conn.commit()
                
                return len(expired_keys) + cursor.rowcount
            
            return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            total_requests = self.stats['hits'] + self.stats['misses']
            hit_rate = (self.stats['hits'] / total_requests * 100) if total_requests > 0 else 0
            
            stats = {
                **self.stats,
                'memory_items': len(self.memory_cache),
                'hit_rate': f"{hit_rate:.1f}%",
                'total_requests': total_requests
            }
            
            if self.enable_disk_cache:
                cursor = self.db_conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM cache WHERE expires_at > ?", (time.time(),))
                stats['disk_items'] = cursor.fetchone()[0]
            
            return stats

# Global cache instance
_cache_manager = None

def get_cache_manager() -> CacheManager:
    """Get global cache manager instance"""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager

def cached(
    ttl: Optional[int] = None,
    key_prefix: Optional[str] = None,
    cache_none: bool = False
) -> Callable:
    """
    Decorator for caching function results
    
    Args:
        ttl: Time to live in seconds
        key_prefix: Optional prefix for cache keys
        cache_none: Whether to cache None results
    
    Usage:
        @cached(ttl=3600)
        def fetch_stock_data(symbol):
            return expensive_api_call(symbol)
    """
    def decorator(func: Callable) -> Callable:
        cache = get_cache_manager()
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = cache._generate_key(*args, **kwargs)
            if key_prefix:
                cache_key = f"{key_prefix}:{cache_key}"
            
            # Check cache
            cached_value = cache.get(cache_key)
            if cached_value is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return cached_value
            
            # Call function
            result = func(*args, **kwargs)
            
            # Cache result
            if result is not None or cache_none:
                cache.set(cache_key, result, ttl)
                logger.debug(f"Cached result for {func.__name__}")
            
            return result
        
        # Add cache control methods
        wrapper.cache_clear = lambda: cache.clear()
        wrapper.cache_delete = lambda *args, **kwargs: cache.delete(
            cache._generate_key(*args, **kwargs)
        )
        
        return wrapper
    return decorator

class TimeBasedCache:
    """Cache with time-based segments for market data"""
    
    def __init__(self, segment_duration: int = 300):  # 5 minutes
        self.segment_duration = segment_duration
        self.cache = get_cache_manager()
    
    def get_segment_key(self, base_key: str, timestamp: Optional[float] = None) -> str:
        """Get cache key for time segment"""
        if timestamp is None:
            timestamp = time.time()
        
        segment = int(timestamp // self.segment_duration)
        return f"{base_key}:segment:{segment}"
    
    def get(self, base_key: str, timestamp: Optional[float] = None) -> Optional[Any]:
        """Get value for current time segment"""
        segment_key = self.get_segment_key(base_key, timestamp)
        return self.cache.get(segment_key)
    
    def set(self, base_key: str, value: Any, timestamp: Optional[float] = None):
        """Set value for current time segment"""
        segment_key = self.get_segment_key(base_key, timestamp)
        # TTL is segment duration + buffer
        ttl = self.segment_duration + 60
        self.cache.set(segment_key, value, ttl)

# Cache configuration for different data types
CACHE_CONFIGS = {
    'stock_quotes': {
        'ttl': 60,  # 1 minute for real-time quotes
        'key_prefix': 'quote'
    },
    'historical_data': {
        'ttl': 3600,  # 1 hour for historical data
        'key_prefix': 'hist'
    },
    'news_data': {
        'ttl': 1800,  # 30 minutes for news
        'key_prefix': 'news'
    },
    'technical_indicators': {
        'ttl': 300,  # 5 minutes for indicators
        'key_prefix': 'tech'
    },
    'ml_predictions': {
        'ttl': 600,  # 10 minutes for ML predictions
        'key_prefix': 'ml'
    },
    'options_data': {
        'ttl': 300,  # 5 minutes for options
        'key_prefix': 'opt'
    }
}

def get_cached_or_fetch(
    cache_type: str,
    fetch_function: Callable,
    *args,
    **kwargs
) -> Any:
    """
    Helper to get cached data or fetch if not available
    
    Args:
        cache_type: Type of cache from CACHE_CONFIGS
        fetch_function: Function to call if cache miss
        *args, **kwargs: Arguments for fetch function
    
    Returns:
        Cached or fetched data
    """
    config = CACHE_CONFIGS.get(cache_type, {})
    ttl = config.get('ttl', 3600)
    prefix = config.get('key_prefix', cache_type)
    
    @cached(ttl=ttl, key_prefix=prefix)
    def cached_fetch(*args, **kwargs):
        return fetch_function(*args, **kwargs)
    
    return cached_fetch(*args, **kwargs)