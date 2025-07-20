#!/usr/bin/env python3
"""
Unit tests for Performance Optimization modules
"""

import pytest
import time
import tempfile
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
import json
from unittest.mock import Mock, patch, MagicMock
import threading

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.cache_manager import (
    CacheManager, cached, TimeBasedCache, get_cached_or_fetch
)
from utils.parallel_processor import (
    ParallelProcessor, AsyncDataFetcher, parallel_stock_fetch,
    chunked_parallel_process, DataPipeline
)
from utils.db_optimizer import (
    ConnectionPool, BatchInsertManager, QueryCache, OptimizedDatabase
)
from utils.data_compressor import (
    DataCompressor, OptimizedStorage, estimate_compression_ratio
)

class TestCacheManager:
    """Test suite for Cache Manager"""
    
    def setup_method(self):
        """Set up test cache"""
        self.temp_dir = tempfile.mkdtemp()
        self.cache = CacheManager(
            cache_dir=self.temp_dir,
            max_memory_items=5,
            default_ttl=2
        )
    
    def teardown_method(self):
        """Clean up"""
        shutil.rmtree(self.temp_dir)
    
    def test_basic_cache_operations(self):
        """Test basic get/set operations"""
        # Set value
        self.cache.set('key1', 'value1')
        
        # Get value
        assert self.cache.get('key1') == 'value1'
        
        # Cache miss
        assert self.cache.get('nonexistent') is None
        
        # Check stats
        stats = self.cache.get_stats()
        assert stats['hits'] == 1
        assert stats['misses'] == 1
    
    def test_cache_expiration(self):
        """Test TTL expiration"""
        # Set with short TTL
        self.cache.set('expire_key', 'value', ttl=1)
        
        # Should exist
        assert self.cache.get('expire_key') == 'value'
        
        # Wait for expiration
        time.sleep(1.5)
        
        # Should be expired
        assert self.cache.get('expire_key') is None
    
    def test_lru_eviction(self):
        """Test LRU eviction"""
        # Fill cache beyond limit
        for i in range(7):
            self.cache.set(f'key{i}', f'value{i}')
        
        # First two should be evicted
        assert self.cache.get('key0') is None
        assert self.cache.get('key1') is None
        
        # Last ones should exist
        assert self.cache.get('key6') == 'value6'
        
        # Check eviction count
        stats = self.cache.get_stats()
        assert stats['evictions'] == 2
    
    def test_cached_decorator(self):
        """Test @cached decorator"""
        call_count = 0
        
        @cached(ttl=2)
        def expensive_function(x):
            nonlocal call_count
            call_count += 1
            return x * 2
        
        # First call
        assert expensive_function(5) == 10
        assert call_count == 1
        
        # Cached call
        assert expensive_function(5) == 10
        assert call_count == 1  # Not called again
        
        # Different argument
        assert expensive_function(3) == 6
        assert call_count == 2

class TestParallelProcessor:
    """Test suite for Parallel Processor"""
    
    def test_parallel_map(self):
        """Test parallel map operation"""
        def square(x):
            return x * x
        
        with ParallelProcessor(max_workers=2) as processor:
            results = processor.map(square, [1, 2, 3, 4, 5])
        
        assert results == [1, 4, 9, 16, 25]
    
    def test_batch_processing(self):
        """Test batch processing"""
        def sum_batch(batch):
            return sum(batch)
        
        items = list(range(10))
        
        with ParallelProcessor(max_workers=2) as processor:
            results = processor.batch_process(
                sum_batch,
                items,
                batch_size=3
            )
        
        # 4 batches: [0,1,2], [3,4,5], [6,7,8], [9]
        assert results == [3, 12, 21, 9]
    
    def test_parallel_fetch(self):
        """Test parallel URL fetching"""
        # Mock URLs
        urls = ['http://api1.com', 'http://api2.com', 'http://api3.com']
        
        with patch('requests.get') as mock_get:
            # Mock responses
            mock_get.return_value.json.return_value = {'data': 'test'}
            mock_get.return_value.raise_for_status = Mock()
            
            with ParallelProcessor(max_workers=3) as processor:
                results = processor.parallel_fetch(urls)
            
            assert len(results) == 3
            assert all(result[1] == {'data': 'test'} for result in results)
    
    def test_data_pipeline(self):
        """Test data processing pipeline"""
        pipeline = DataPipeline(max_workers=2)
        
        # Add stages
        pipeline.add_stage('double', lambda x: x * 2)
        pipeline.add_stage('add_one', lambda x: x + 1)
        
        # Process data
        result = pipeline.process([1, 2, 3])
        
        assert result == [3, 5, 7]  # (1*2)+1, (2*2)+1, (3*2)+1

class TestDatabaseOptimizer:
    """Test suite for Database Optimizer"""
    
    def setup_method(self):
        """Set up test database"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test.db"
        self.db = OptimizedDatabase(str(self.db_path), pool_size=2)
        
        # Create test table
        with self.db.pool.get_connection() as conn:
            conn.execute("""
                CREATE TABLE test_table (
                    id INTEGER PRIMARY KEY,
                    value TEXT
                )
            """)
            conn.commit()
    
    def teardown_method(self):
        """Clean up"""
        self.db.close()
        shutil.rmtree(self.temp_dir)
    
    def test_connection_pool(self):
        """Test connection pooling"""
        # Get connections
        with self.db.pool.get_connection() as conn1:
            with self.db.pool.get_connection() as conn2:
                # Both should be valid
                assert conn1 is not None
                assert conn2 is not None
                
                # Should be different connections
                assert conn1 is not conn2
    
    def test_batch_insert(self):
        """Test batch insert operations"""
        records = [(i, f'value_{i}') for i in range(100)]
        
        self.db.batch_insert('test_table', records)
        
        # Verify data
        results = self.db.execute_query("SELECT COUNT(*) FROM test_table")
        assert results[0][0] == 100
    
    def test_query_cache(self):
        """Test query caching"""
        # Insert test data
        self.db.execute_update("INSERT INTO test_table VALUES (1, 'test')")
        
        # First query (cache miss)
        result1 = self.db.execute_query("SELECT * FROM test_table WHERE id = ?", (1,))
        
        # Second query (cache hit)
        result2 = self.db.execute_query("SELECT * FROM test_table WHERE id = ?", (1,))
        
        assert result1 == result2
        assert result1[0] == (1, 'test')

class TestDataCompressor:
    """Test suite for Data Compressor"""
    
    def test_json_compression(self):
        """Test JSON compression/decompression"""
        data = {'key': 'value', 'numbers': list(range(100))}
        
        # Compress
        compressed = DataCompressor.compress_json(data)
        
        # Should be smaller
        original_size = len(json.dumps(data))
        assert len(compressed) < original_size
        
        # Decompress
        decompressed = DataCompressor.decompress_json(compressed)
        assert decompressed == data
    
    def test_dataframe_compression(self):
        """Test DataFrame compression"""
        df = pd.DataFrame({
            'A': range(100),
            'B': np.random.randn(100),
            'C': ['text'] * 100
        })
        
        # Compress
        compressed = DataCompressor.compress_dataframe(df, format='parquet')
        
        # Decompress
        df_decompressed = DataCompressor.decompress_dataframe(
            compressed, format='parquet'
        )
        
        pd.testing.assert_frame_equal(df, df_decompressed)
    
    def test_numpy_compression(self):
        """Test numpy array compression"""
        arr = np.random.randn(100, 50)
        
        # Compress
        compressed = DataCompressor.compress_numpy_array(arr)
        
        # Should be smaller
        assert len(compressed) < arr.nbytes
        
        # Decompress
        arr_decompressed = DataCompressor.decompress_numpy_array(compressed)
        
        np.testing.assert_array_almost_equal(arr, arr_decompressed)
    
    def test_compression_ratio_estimation(self):
        """Test compression ratio estimation"""
        # Test with DataFrame
        df = pd.DataFrame({'A': range(1000), 'B': ['text'] * 1000})
        ratio = estimate_compression_ratio(df)
        assert ratio > 1  # Should compress
        
        # Test with repetitive data (high compression)
        repetitive_data = {'data': ['same'] * 1000}
        ratio = estimate_compression_ratio(repetitive_data)
        assert ratio > 10  # High compression expected

class TestOptimizedStorage:
    """Test suite for Optimized Storage"""
    
    def setup_method(self):
        """Set up test storage"""
        self.temp_dir = tempfile.mkdtemp()
        self.storage = OptimizedStorage(self.temp_dir)
    
    def teardown_method(self):
        """Clean up"""
        shutil.rmtree(self.temp_dir)
    
    def test_save_load_timeseries(self):
        """Test time series save/load"""
        # Create sample data
        dates = pd.date_range('2024-01-01', periods=100)
        df = pd.DataFrame({
            'open': np.random.randn(100) + 100,
            'high': np.random.randn(100) + 101,
            'low': np.random.randn(100) + 99,
            'close': np.random.randn(100) + 100,
            'volume': np.random.randint(1000000, 5000000, 100)
        }, index=dates)
        
        # Save
        metadata = self.storage.save_time_series('AAPL', df)
        
        # Check metadata
        assert metadata['symbol'] == 'AAPL'
        assert metadata['rows'] == 100
        assert metadata['compressed'] is True
        
        # Load
        loaded_df = self.storage.load_time_series('AAPL')
        
        pd.testing.assert_frame_equal(df, loaded_df)
    
    def test_dataframe_type_optimization(self):
        """Test DataFrame type optimization"""
        df = pd.DataFrame({
            'small_int': range(10),  # Can be uint8
            'large_int': range(100000, 200000),  # Needs larger type
            'float_col': np.random.randn(100000) * 0.01,  # Can be float32
            'category': ['A', 'B', 'C'] * 33334  # Can be category
        })
        
        optimized = self.storage._optimize_dataframe_types(df)
        
        # Check optimizations
        assert optimized['small_int'].dtype == 'uint8'
        assert optimized['float_col'].dtype == 'float32'
        assert optimized['category'].dtype.name == 'category'
        
        # Check data integrity
        assert df['small_int'].equals(optimized['small_int'])

# Performance benchmark tests
class TestPerformanceBenchmarks:
    """Benchmark tests for performance improvements"""
    
    def test_parallel_vs_sequential(self):
        """Compare parallel vs sequential processing"""
        def slow_function(x):
            time.sleep(0.01)  # Simulate work
            return x * x
        
        items = list(range(20))
        
        # Sequential
        start = time.time()
        sequential_results = [slow_function(x) for x in items]
        sequential_time = time.time() - start
        
        # Parallel
        start = time.time()
        with ParallelProcessor(max_workers=4) as processor:
            parallel_results = processor.map(slow_function, items)
        parallel_time = time.time() - start
        
        # Should be faster
        assert parallel_time < sequential_time * 0.5  # At least 2x speedup
        assert sequential_results == parallel_results
    
    def test_cache_performance(self):
        """Test cache performance improvement"""
        call_count = 0
        
        @cached(ttl=10)
        def expensive_calculation(n):
            nonlocal call_count
            call_count += 1
            time.sleep(0.1)  # Simulate expensive operation
            return sum(range(n))
        
        # First call (slow)
        start = time.time()
        result1 = expensive_calculation(1000)
        first_time = time.time() - start
        
        # Cached call (fast)
        start = time.time()
        result2 = expensive_calculation(1000)
        cached_time = time.time() - start
        
        assert result1 == result2
        assert call_count == 1  # Only called once
        assert cached_time < first_time * 0.01  # Much faster