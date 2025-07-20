#!/usr/bin/env python3
"""
Database Optimization Utilities
Provides connection pooling, batch operations, and query optimization
"""

import sqlite3
import threading
from typing import List, Dict, Any, Optional, Tuple, Union
from contextlib import contextmanager
import logging
from queue import Queue
import time
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class ConnectionPool:
    """SQLite connection pool for better performance"""
    
    def __init__(
        self,
        database: str,
        pool_size: int = 5,
        check_same_thread: bool = False
    ):
        self.database = database
        self.pool_size = pool_size
        self.check_same_thread = check_same_thread
        
        # Connection pool
        self._pool = Queue(maxsize=pool_size)
        self._all_connections = []
        
        # Initialize pool
        self._init_pool()
        
        # Lock for thread safety
        self._lock = threading.Lock()
    
    def _init_pool(self):
        """Initialize connection pool"""
        for _ in range(self.pool_size):
            conn = self._create_connection()
            self._pool.put(conn)
            self._all_connections.append(conn)
    
    def _create_connection(self) -> sqlite3.Connection:
        """Create new database connection with optimizations"""
        conn = sqlite3.connect(
            self.database,
            check_same_thread=self.check_same_thread
        )
        
        # Enable optimizations
        conn.execute("PRAGMA journal_mode=WAL")  # Write-Ahead Logging
        conn.execute("PRAGMA synchronous=NORMAL")  # Faster writes
        conn.execute("PRAGMA cache_size=10000")  # Larger cache
        conn.execute("PRAGMA temp_store=MEMORY")  # Memory temp tables
        conn.execute("PRAGMA mmap_size=268435456")  # 256MB memory-mapped I/O
        
        # Enable foreign keys
        conn.execute("PRAGMA foreign_keys=ON")
        
        return conn
    
    @contextmanager
    def get_connection(self):
        """Get connection from pool"""
        conn = self._pool.get()
        try:
            yield conn
        finally:
            self._pool.put(conn)
    
    def close_all(self):
        """Close all connections"""
        with self._lock:
            for conn in self._all_connections:
                conn.close()
            self._all_connections.clear()
            while not self._pool.empty():
                self._pool.get()

class BatchInsertManager:
    """Manages batch inserts for better performance"""
    
    def __init__(
        self,
        connection_pool: ConnectionPool,
        batch_size: int = 1000,
        flush_interval: int = 5
    ):
        self.pool = connection_pool
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        
        # Batch storage
        self._batches: Dict[str, List[Tuple]] = {}
        self._batch_lock = threading.Lock()
        
        # Auto-flush thread
        self._running = True
        self._flush_thread = threading.Thread(target=self._auto_flush)
        self._flush_thread.daemon = True
        self._flush_thread.start()
    
    def add_record(self, table: str, values: Tuple, columns: Optional[List[str]] = None):
        """Add record to batch"""
        with self._batch_lock:
            if table not in self._batches:
                self._batches[table] = []
            
            self._batches[table].append(values)
            
            # Flush if batch is full
            if len(self._batches[table]) >= self.batch_size:
                self._flush_table(table)
    
    def _flush_table(self, table: str):
        """Flush specific table batch"""
        if table not in self._batches or not self._batches[table]:
            return
        
        records = self._batches[table]
        self._batches[table] = []
        
        # Perform batch insert
        with self.pool.get_connection() as conn:
            placeholders = ','.join(['?' for _ in records[0]])
            query = f"INSERT INTO {table} VALUES ({placeholders})"
            
            try:
                conn.executemany(query, records)
                conn.commit()
                logger.debug(f"Flushed {len(records)} records to {table}")
            except Exception as e:
                logger.error(f"Batch insert failed for {table}: {e}")
                conn.rollback()
    
    def flush_all(self):
        """Flush all pending batches"""
        with self._batch_lock:
            tables = list(self._batches.keys())
            for table in tables:
                self._flush_table(table)
    
    def _auto_flush(self):
        """Auto-flush thread"""
        while self._running:
            time.sleep(self.flush_interval)
            self.flush_all()
    
    def close(self):
        """Stop auto-flush and flush remaining"""
        self._running = False
        self._flush_thread.join()
        self.flush_all()

class QueryCache:
    """Cache for frequently used queries"""
    
    def __init__(self, max_size: int = 100, ttl: int = 300):
        self.max_size = max_size
        self.ttl = ttl
        self.cache: Dict[str, Tuple[Any, float]] = {}
        self._lock = threading.Lock()
    
    def get(self, query: str, params: Optional[Tuple] = None) -> Optional[Any]:
        """Get cached query result"""
        key = self._make_key(query, params)
        
        with self._lock:
            if key in self.cache:
                result, timestamp = self.cache[key]
                if time.time() - timestamp < self.ttl:
                    return result
                else:
                    del self.cache[key]
        
        return None
    
    def set(self, query: str, params: Optional[Tuple], result: Any):
        """Cache query result"""
        key = self._make_key(query, params)
        
        with self._lock:
            # LRU eviction
            if len(self.cache) >= self.max_size:
                oldest_key = min(self.cache.keys(), 
                               key=lambda k: self.cache[k][1])
                del self.cache[oldest_key]
            
            self.cache[key] = (result, time.time())
    
    def _make_key(self, query: str, params: Optional[Tuple]) -> str:
        """Generate cache key"""
        return f"{query}:{str(params)}"
    
    def clear(self):
        """Clear cache"""
        with self._lock:
            self.cache.clear()

class OptimizedDatabase:
    """Optimized database interface with pooling and caching"""
    
    def __init__(
        self,
        database: str,
        pool_size: int = 5,
        cache_queries: bool = True,
        batch_size: int = 1000
    ):
        self.database = database
        
        # Connection pool
        self.pool = ConnectionPool(database, pool_size)
        
        # Query cache
        self.query_cache = QueryCache() if cache_queries else None
        
        # Batch manager
        self.batch_manager = BatchInsertManager(self.pool, batch_size)
        
        # Prepared statements cache
        self._prepared_statements: Dict[str, str] = {}
    
    def execute_query(
        self,
        query: str,
        params: Optional[Tuple] = None,
        cache: bool = True
    ) -> List[Tuple]:
        """Execute SELECT query with caching"""
        # Check cache
        if cache and self.query_cache:
            cached_result = self.query_cache.get(query, params)
            if cached_result is not None:
                return cached_result
        
        # Execute query
        with self.pool.get_connection() as conn:
            cursor = conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            results = cursor.fetchall()
        
        # Cache result
        if cache and self.query_cache:
            self.query_cache.set(query, params, results)
        
        return results
    
    def execute_update(
        self,
        query: str,
        params: Optional[Tuple] = None
    ) -> int:
        """Execute INSERT/UPDATE/DELETE query"""
        with self.pool.get_connection() as conn:
            cursor = conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            conn.commit()
            return cursor.rowcount
    
    def batch_insert(
        self,
        table: str,
        records: List[Tuple],
        columns: Optional[List[str]] = None
    ):
        """Perform batch insert"""
        if not records:
            return
        
        with self.pool.get_connection() as conn:
            if columns:
                placeholders = ','.join(['?' for _ in columns])
                cols = ','.join(columns)
                query = f"INSERT INTO {table} ({cols}) VALUES ({placeholders})"
            else:
                placeholders = ','.join(['?' for _ in records[0]])
                query = f"INSERT INTO {table} VALUES ({placeholders})"
            
            conn.executemany(query, records)
            conn.commit()
    
    def create_index(self, table: str, columns: List[str], unique: bool = False):
        """Create database index for faster queries"""
        index_name = f"idx_{table}_{'_'.join(columns)}"
        unique_clause = "UNIQUE" if unique else ""
        
        query = f"""
        CREATE {unique_clause} INDEX IF NOT EXISTS {index_name}
        ON {table} ({','.join(columns)})
        """
        
        with self.pool.get_connection() as conn:
            conn.execute(query)
            conn.commit()
    
    def analyze(self):
        """Update database statistics for query optimizer"""
        with self.pool.get_connection() as conn:
            conn.execute("ANALYZE")
            conn.commit()
    
    def vacuum(self):
        """Vacuum database to reclaim space"""
        with self.pool.get_connection() as conn:
            conn.execute("VACUUM")
            conn.commit()
    
    def get_table_stats(self, table: str) -> Dict[str, Any]:
        """Get table statistics"""
        stats = {}
        
        with self.pool.get_connection() as conn:
            # Row count
            cursor = conn.cursor()
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            stats['row_count'] = cursor.fetchone()[0]
            
            # Table size
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            
            # Get page count
            cursor.execute(f"SELECT COUNT(*) FROM pragma_table_info('{table}')")
            stats['column_count'] = cursor.fetchone()[0]
        
        return stats
    
    def close(self):
        """Close all resources"""
        self.batch_manager.close()
        self.pool.close_all()
        if self.query_cache:
            self.query_cache.clear()

# Helper functions for common operations
def create_optimized_tables(db: OptimizedDatabase):
    """Create optimized table structures"""
    
    # Historical data table with proper indexing
    with db.pool.get_connection() as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS stock_data (
            symbol TEXT NOT NULL,
            date TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            PRIMARY KEY (symbol, date)
        )
        """)
        
        conn.execute("""
        CREATE TABLE IF NOT EXISTS indicators (
            symbol TEXT NOT NULL,
            date TEXT NOT NULL,
            indicator_name TEXT NOT NULL,
            value REAL,
            PRIMARY KEY (symbol, date, indicator_name)
        )
        """)
        
        conn.execute("""
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            signal_type TEXT NOT NULL,
            strength REAL,
            metadata TEXT
        )
        """)
        
        conn.commit()
    
    # Create indexes
    db.create_index('stock_data', ['symbol', 'date'])
    db.create_index('stock_data', ['date'])
    db.create_index('indicators', ['symbol', 'date'])
    db.create_index('signals', ['symbol', 'timestamp'])
    db.create_index('signals', ['signal_type'])

def migrate_to_optimized_db(old_db_path: str, new_db_path: str):
    """Migrate data to optimized database"""
    old_conn = sqlite3.connect(old_db_path)
    new_db = OptimizedDatabase(new_db_path)
    
    # Get all tables
    cursor = old_conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    
    for table_name in tables:
        table = table_name[0]
        
        # Get data
        cursor.execute(f"SELECT * FROM {table}")
        records = cursor.fetchall()
        
        if records:
            # Batch insert into new database
            new_db.batch_insert(table, records)
    
    old_conn.close()
    new_db.close()

# Global instance
_optimized_db = None

def get_optimized_db(database: str = "trading_data.db") -> OptimizedDatabase:
    """Get global optimized database instance"""
    global _optimized_db
    if _optimized_db is None:
        _optimized_db = OptimizedDatabase(database)
        create_optimized_tables(_optimized_db)
    return _optimized_db