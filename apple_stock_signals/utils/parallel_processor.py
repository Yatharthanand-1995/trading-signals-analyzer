#!/usr/bin/env python3
"""
Parallel Processing Utilities for Performance Optimization
Handles concurrent data fetching and processing
"""

import asyncio
import aiohttp
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import List, Dict, Any, Callable, Optional, Union, Tuple
import logging
import time
from functools import partial
import multiprocessing
import threading
from queue import Queue
import numpy as np

logger = logging.getLogger(__name__)

class ParallelProcessor:
    """Manages parallel processing for data operations"""
    
    def __init__(
        self,
        max_workers: Optional[int] = None,
        use_processes: bool = False,
        timeout: int = 30
    ):
        """
        Initialize parallel processor
        
        Args:
            max_workers: Maximum number of workers (None for CPU count)
            use_processes: Use processes instead of threads
            timeout: Timeout for each task in seconds
        """
        if max_workers is None:
            max_workers = multiprocessing.cpu_count()
        
        self.max_workers = max_workers
        self.use_processes = use_processes
        self.timeout = timeout
        
        # Create executor
        if use_processes:
            self.executor = ProcessPoolExecutor(max_workers=max_workers)
        else:
            self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Performance metrics
        self.metrics = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'total_time': 0
        }
    
    def map(
        self,
        func: Callable,
        items: List[Any],
        chunk_size: Optional[int] = None
    ) -> List[Any]:
        """
        Map function over items in parallel
        
        Args:
            func: Function to apply
            items: List of items to process
            chunk_size: Items per chunk for process pool
            
        Returns:
            List of results in order
        """
        if not items:
            return []
        
        start_time = time.time()
        self.metrics['total_tasks'] += len(items)
        
        try:
            if chunk_size and self.use_processes:
                # Use chunks for process pool
                results = list(self.executor.map(func, items, chunksize=chunk_size))
            else:
                results = list(self.executor.map(func, items))
            
            self.metrics['completed_tasks'] += len(results)
            
        except Exception as e:
            logger.error(f"Parallel map failed: {e}")
            self.metrics['failed_tasks'] += len(items)
            raise
        
        finally:
            self.metrics['total_time'] += time.time() - start_time
        
        return results
    
    def map_async(
        self,
        func: Callable,
        items: List[Any],
        callback: Optional[Callable] = None,
        error_callback: Optional[Callable] = None
    ) -> List[concurrent.futures.Future]:
        """
        Map function over items asynchronously
        
        Returns:
            List of Future objects
        """
        futures = []
        
        for item in items:
            future = self.executor.submit(func, item)
            
            if callback:
                future.add_done_callback(
                    lambda f: callback(f.result()) if not f.exception() else None
                )
            
            if error_callback:
                future.add_done_callback(
                    lambda f: error_callback(f.exception()) if f.exception() else None
                )
            
            futures.append(future)
        
        return futures
    
    def batch_process(
        self,
        func: Callable,
        items: List[Any],
        batch_size: int,
        aggregate_func: Optional[Callable] = None
    ) -> Union[List[Any], Any]:
        """
        Process items in batches
        
        Args:
            func: Function to process each batch
            items: Items to process
            batch_size: Items per batch
            aggregate_func: Optional function to aggregate results
            
        Returns:
            List of batch results or aggregated result
        """
        batches = [items[i:i+batch_size] for i in range(0, len(items), batch_size)]
        results = self.map(func, batches)
        
        if aggregate_func:
            return aggregate_func(results)
        
        return results
    
    def parallel_fetch(
        self,
        urls: List[str],
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None
    ) -> List[Tuple[str, Optional[Dict[str, Any]]]]:
        """
        Fetch multiple URLs in parallel
        
        Returns:
            List of (url, response_data) tuples
        """
        if timeout is None:
            timeout = self.timeout
        
        def fetch_url(url: str) -> Tuple[str, Optional[Dict[str, Any]]]:
            import requests
            try:
                response = requests.get(url, headers=headers, timeout=timeout)
                response.raise_for_status()
                return (url, response.json())
            except Exception as e:
                logger.error(f"Failed to fetch {url}: {e}")
                return (url, None)
        
        return self.map(fetch_url, urls)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        total_tasks = self.metrics['total_tasks']
        if total_tasks > 0:
            success_rate = (self.metrics['completed_tasks'] / total_tasks) * 100
            avg_time = self.metrics['total_time'] / total_tasks
        else:
            success_rate = 0
            avg_time = 0
        
        return {
            **self.metrics,
            'success_rate': f"{success_rate:.1f}%",
            'avg_time_per_task': f"{avg_time:.3f}s"
        }
    
    def shutdown(self, wait: bool = True):
        """Shutdown the executor"""
        self.executor.shutdown(wait=wait)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()

class AsyncDataFetcher:
    """Asynchronous data fetcher for high-performance API calls"""
    
    def __init__(
        self,
        max_concurrent: int = 10,
        timeout: int = 30,
        retry_count: int = 3
    ):
        self.max_concurrent = max_concurrent
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.retry_count = retry_count
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(timeout=self.timeout)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def fetch_one(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """Fetch single URL with retry logic"""
        for attempt in range(self.retry_count):
            try:
                async with self.session.get(url, headers=headers, params=params) as response:
                    response.raise_for_status()
                    return await response.json()
            
            except asyncio.TimeoutError:
                logger.warning(f"Timeout fetching {url} (attempt {attempt + 1})")
                if attempt < self.retry_count - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
            
            except Exception as e:
                logger.error(f"Error fetching {url}: {e}")
                if attempt < self.retry_count - 1:
                    await asyncio.sleep(1)
        
        return None
    
    async def fetch_many(
        self,
        urls: List[str],
        headers: Optional[Dict[str, str]] = None
    ) -> List[Tuple[str, Optional[Dict[str, Any]]]]:
        """Fetch multiple URLs concurrently"""
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def fetch_with_semaphore(url: str):
            async with semaphore:
                result = await self.fetch_one(url, headers)
                return (url, result)
        
        tasks = [fetch_with_semaphore(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to fetch {urls[i]}: {result}")
                processed_results.append((urls[i], None))
            else:
                processed_results.append(result)
        
        return processed_results

def parallel_stock_fetch(
    symbols: List[str],
    fetch_func: Callable,
    max_workers: int = 10
) -> Dict[str, Any]:
    """
    Fetch data for multiple stocks in parallel
    
    Args:
        symbols: List of stock symbols
        fetch_func: Function to fetch data for one symbol
        max_workers: Maximum concurrent workers
        
    Returns:
        Dictionary mapping symbols to their data
    """
    with ParallelProcessor(max_workers=max_workers) as processor:
        # Create tasks
        tasks = [(symbol, fetch_func) for symbol in symbols]
        
        # Process in parallel
        results = processor.map(
            lambda task: (task[0], task[1](task[0])),
            tasks
        )
        
        # Convert to dictionary
        return {symbol: data for symbol, data in results}

def chunked_parallel_process(
    items: List[Any],
    process_func: Callable,
    chunk_size: int = 100,
    max_workers: int = 4,
    aggregate: bool = True
) -> Union[List[Any], Any]:
    """
    Process large datasets in chunks with parallel processing
    
    Args:
        items: Items to process
        process_func: Function to process each chunk
        chunk_size: Items per chunk
        max_workers: Maximum parallel workers
        aggregate: Whether to flatten results
        
    Returns:
        Processed results
    """
    # Split into chunks
    chunks = [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]
    
    # Process chunks in parallel
    with ParallelProcessor(max_workers=max_workers) as processor:
        chunk_results = processor.map(process_func, chunks)
    
    # Aggregate results if requested
    if aggregate and isinstance(chunk_results[0], list):
        return [item for chunk in chunk_results for item in chunk]
    
    return chunk_results

class DataPipeline:
    """Pipeline for parallel data processing with stages"""
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.stages = []
    
    def add_stage(
        self,
        name: str,
        func: Callable,
        parallel: bool = True
    ) -> 'DataPipeline':
        """Add processing stage to pipeline"""
        self.stages.append({
            'name': name,
            'func': func,
            'parallel': parallel
        })
        return self
    
    def process(self, data: Any) -> Any:
        """Process data through all stages"""
        result = data
        
        for stage in self.stages:
            logger.info(f"Processing stage: {stage['name']}")
            
            if stage['parallel'] and isinstance(result, list):
                with ParallelProcessor(max_workers=self.max_workers) as processor:
                    result = processor.map(stage['func'], result)
            else:
                result = stage['func'](result)
        
        return result

# Async helper for running in sync context
def run_async_fetch(
    urls: List[str],
    headers: Optional[Dict[str, str]] = None,
    max_concurrent: int = 10
) -> List[Tuple[str, Optional[Dict[str, Any]]]]:
    """
    Run async fetch in synchronous context
    
    Args:
        urls: List of URLs to fetch
        headers: Optional headers
        max_concurrent: Maximum concurrent requests
        
    Returns:
        List of (url, data) tuples
    """
    async def fetch_all():
        async with AsyncDataFetcher(max_concurrent=max_concurrent) as fetcher:
            return await fetcher.fetch_many(urls, headers)
    
    # Create new event loop if needed
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(fetch_all())

# Performance monitoring decorator
def monitor_performance(func: Callable) -> Callable:
    """Decorator to monitor function performance"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        execution_time = end_time - start_time
        logger.info(f"{func.__name__} executed in {execution_time:.3f}s")
        
        return result
    
    return wrapper