#!/usr/bin/env python3
"""
Data Compression and Storage Optimization
Reduces storage requirements and improves I/O performance
"""

import gzip
import pickle
import json
import zlib
import lz4.frame
import numpy as np
import pandas as pd
from typing import Any, Union, Optional, Dict
from pathlib import Path
import logging
import struct
import io

logger = logging.getLogger(__name__)

class DataCompressor:
    """Handles data compression with multiple algorithms"""
    
    COMPRESSION_LEVELS = {
        'fast': {'gzip': 1, 'zlib': 1, 'lz4': 0},
        'balanced': {'gzip': 6, 'zlib': 6, 'lz4': 9},
        'best': {'gzip': 9, 'zlib': 9, 'lz4': 16}
    }
    
    @staticmethod
    def compress_json(
        data: Union[dict, list],
        algorithm: str = 'gzip',
        level: str = 'balanced'
    ) -> bytes:
        """Compress JSON data"""
        json_str = json.dumps(data, separators=(',', ':'))
        json_bytes = json_str.encode('utf-8')
        
        compression_level = DataCompressor.COMPRESSION_LEVELS[level][algorithm]
        
        if algorithm == 'gzip':
            return gzip.compress(json_bytes, compresslevel=compression_level)
        elif algorithm == 'zlib':
            return zlib.compress(json_bytes, level=compression_level)
        elif algorithm == 'lz4':
            return lz4.frame.compress(json_bytes, compression_level=compression_level)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
    
    @staticmethod
    def decompress_json(
        compressed_data: bytes,
        algorithm: str = 'gzip'
    ) -> Union[dict, list]:
        """Decompress JSON data"""
        if algorithm == 'gzip':
            json_bytes = gzip.decompress(compressed_data)
        elif algorithm == 'zlib':
            json_bytes = zlib.decompress(compressed_data)
        elif algorithm == 'lz4':
            json_bytes = lz4.frame.decompress(compressed_data)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        json_str = json_bytes.decode('utf-8')
        return json.loads(json_str)
    
    @staticmethod
    def compress_dataframe(
        df: pd.DataFrame,
        algorithm: str = 'gzip',
        format: str = 'parquet'
    ) -> bytes:
        """Compress pandas DataFrame"""
        buffer = io.BytesIO()
        
        if format == 'parquet':
            df.to_parquet(buffer, compression=algorithm)
        elif format == 'pickle':
            if algorithm == 'gzip':
                with gzip.GzipFile(fileobj=buffer, mode='wb') as gz:
                    pickle.dump(df, gz)
            else:
                pickle.dump(df, buffer)
        elif format == 'csv':
            if algorithm == 'gzip':
                with gzip.GzipFile(fileobj=buffer, mode='wb') as gz:
                    df.to_csv(gz, index=False)
            else:
                df.to_csv(buffer, index=False)
        else:
            raise ValueError(f"Unknown format: {format}")
        
        return buffer.getvalue()
    
    @staticmethod
    def decompress_dataframe(
        compressed_data: bytes,
        algorithm: str = 'gzip',
        format: str = 'parquet'
    ) -> pd.DataFrame:
        """Decompress pandas DataFrame"""
        buffer = io.BytesIO(compressed_data)
        
        if format == 'parquet':
            return pd.read_parquet(buffer)
        elif format == 'pickle':
            if algorithm == 'gzip':
                with gzip.GzipFile(fileobj=buffer, mode='rb') as gz:
                    return pickle.load(gz)
            else:
                return pickle.load(buffer)
        elif format == 'csv':
            if algorithm == 'gzip':
                with gzip.GzipFile(fileobj=buffer, mode='rb') as gz:
                    return pd.read_csv(gz)
            else:
                return pd.read_csv(buffer)
        else:
            raise ValueError(f"Unknown format: {format}")
    
    @staticmethod
    def compress_numpy_array(
        arr: np.ndarray,
        algorithm: str = 'lz4'
    ) -> bytes:
        """Compress numpy array efficiently"""
        # Save array metadata
        metadata = {
            'dtype': str(arr.dtype),
            'shape': arr.shape
        }
        
        # Convert to bytes
        arr_bytes = arr.tobytes()
        
        # Compress
        if algorithm == 'lz4':
            compressed = lz4.frame.compress(arr_bytes)
        elif algorithm == 'gzip':
            compressed = gzip.compress(arr_bytes)
        elif algorithm == 'zlib':
            compressed = zlib.compress(arr_bytes)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        # Combine metadata and compressed data
        metadata_bytes = json.dumps(metadata).encode('utf-8')
        metadata_size = struct.pack('I', len(metadata_bytes))
        
        return metadata_size + metadata_bytes + compressed
    
    @staticmethod
    def decompress_numpy_array(
        compressed_data: bytes,
        algorithm: str = 'lz4'
    ) -> np.ndarray:
        """Decompress numpy array"""
        # Extract metadata size
        metadata_size = struct.unpack('I', compressed_data[:4])[0]
        
        # Extract metadata
        metadata_bytes = compressed_data[4:4+metadata_size]
        metadata = json.loads(metadata_bytes.decode('utf-8'))
        
        # Extract compressed array data
        compressed_array = compressed_data[4+metadata_size:]
        
        # Decompress
        if algorithm == 'lz4':
            arr_bytes = lz4.frame.decompress(compressed_array)
        elif algorithm == 'gzip':
            arr_bytes = gzip.decompress(compressed_array)
        elif algorithm == 'zlib':
            arr_bytes = zlib.decompress(compressed_array)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        # Reconstruct array
        dtype = np.dtype(metadata['dtype'])
        shape = tuple(metadata['shape'])
        return np.frombuffer(arr_bytes, dtype=dtype).reshape(shape)

class OptimizedStorage:
    """Optimized storage for time series data"""
    
    def __init__(self, base_path: str = "optimized_data"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        
        self.compressor = DataCompressor()
    
    def save_time_series(
        self,
        symbol: str,
        data: pd.DataFrame,
        compress: bool = True
    ):
        """Save time series data with optimization"""
        # Optimize data types
        optimized_df = self._optimize_dataframe_types(data)
        
        # Separate numeric and non-numeric columns
        numeric_cols = optimized_df.select_dtypes(include=[np.number]).columns
        non_numeric_cols = optimized_df.select_dtypes(exclude=[np.number]).columns
        
        if compress:
            # Save as compressed parquet
            file_path = self.base_path / f"{symbol}_timeseries.parquet.gz"
            optimized_df.to_parquet(file_path, compression='gzip')
        else:
            # Save as regular parquet
            file_path = self.base_path / f"{symbol}_timeseries.parquet"
            optimized_df.to_parquet(file_path)
        
        # Save metadata
        metadata = {
            'symbol': symbol,
            'rows': len(optimized_df),
            'columns': list(optimized_df.columns),
            'date_range': [
                str(optimized_df.index.min()),
                str(optimized_df.index.max())
            ],
            'compressed': compress,
            'size_bytes': file_path.stat().st_size
        }
        
        metadata_path = self.base_path / f"{symbol}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
        
        return metadata
    
    def load_time_series(
        self,
        symbol: str,
        columns: Optional[list] = None,
        date_range: Optional[tuple] = None
    ) -> pd.DataFrame:
        """Load time series data efficiently"""
        # Check compressed version first
        compressed_path = self.base_path / f"{symbol}_timeseries.parquet.gz"
        regular_path = self.base_path / f"{symbol}_timeseries.parquet"
        
        if compressed_path.exists():
            df = pd.read_parquet(compressed_path, columns=columns)
        elif regular_path.exists():
            df = pd.read_parquet(regular_path, columns=columns)
        else:
            raise FileNotFoundError(f"No data found for {symbol}")
        
        # Filter by date range if specified
        if date_range:
            start_date, end_date = date_range
            df = df.loc[start_date:end_date]
        
        return df
    
    def _optimize_dataframe_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame data types for storage"""
        optimized_df = df.copy()
        
        for col in optimized_df.columns:
            col_type = optimized_df[col].dtype
            
            # Optimize integers
            if col_type in ['int64', 'int32', 'int16']:
                max_val = optimized_df[col].max()
                min_val = optimized_df[col].min()
                
                if min_val >= 0:
                    if max_val < 255:
                        optimized_df[col] = optimized_df[col].astype('uint8')
                    elif max_val < 65535:
                        optimized_df[col] = optimized_df[col].astype('uint16')
                    elif max_val < 4294967295:
                        optimized_df[col] = optimized_df[col].astype('uint32')
                else:
                    if min_val > -128 and max_val < 127:
                        optimized_df[col] = optimized_df[col].astype('int8')
                    elif min_val > -32768 and max_val < 32767:
                        optimized_df[col] = optimized_df[col].astype('int16')
                    elif min_val > -2147483648 and max_val < 2147483647:
                        optimized_df[col] = optimized_df[col].astype('int32')
            
            # Optimize floats
            elif col_type == 'float64':
                # Check if can be converted to float32 without loss
                max_val = optimized_df[col].abs().max()
                if max_val < 3.4e38:
                    optimized_df[col] = optimized_df[col].astype('float32')
            
            # Optimize object columns
            elif col_type == 'object':
                # Try to convert to category if cardinality is low
                unique_count = optimized_df[col].nunique()
                total_count = len(optimized_df[col])
                
                if unique_count / total_count < 0.5:  # Less than 50% unique
                    optimized_df[col] = optimized_df[col].astype('category')
        
        return optimized_df
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        stats = {
            'total_files': 0,
            'total_size_mb': 0,
            'symbols': {},
            'compression_ratio': 0
        }
        
        for file_path in self.base_path.glob('*.parquet*'):
            stats['total_files'] += 1
            size_bytes = file_path.stat().st_size
            stats['total_size_mb'] += size_bytes / (1024 * 1024)
            
            # Extract symbol
            symbol = file_path.stem.split('_')[0]
            if symbol not in stats['symbols']:
                stats['symbols'][symbol] = {
                    'files': 0,
                    'size_mb': 0
                }
            
            stats['symbols'][symbol]['files'] += 1
            stats['symbols'][symbol]['size_mb'] += size_bytes / (1024 * 1024)
        
        return stats

class ChunkedDataWriter:
    """Write large datasets in chunks to avoid memory issues"""
    
    def __init__(
        self,
        file_path: str,
        chunk_size: int = 10000,
        compression: str = 'gzip'
    ):
        self.file_path = Path(file_path)
        self.chunk_size = chunk_size
        self.compression = compression
        self.chunks_written = 0
        
        # Initialize file
        self.file_path.parent.mkdir(exist_ok=True)
    
    def write_chunk(self, df_chunk: pd.DataFrame):
        """Write single chunk"""
        mode = 'w' if self.chunks_written == 0 else 'a'
        
        if self.file_path.suffix == '.parquet':
            # Parquet doesn't support append mode directly
            if self.chunks_written == 0:
                df_chunk.to_parquet(self.file_path, compression=self.compression)
            else:
                # Read existing, append, and rewrite
                existing = pd.read_parquet(self.file_path)
                combined = pd.concat([existing, df_chunk], ignore_index=True)
                combined.to_parquet(self.file_path, compression=self.compression)
        else:
            # CSV supports append
            df_chunk.to_csv(
                self.file_path,
                mode=mode,
                header=(self.chunks_written == 0),
                index=False,
                compression=self.compression
            )
        
        self.chunks_written += 1
        logger.info(f"Written chunk {self.chunks_written} ({len(df_chunk)} rows)")

# Utility functions
def estimate_compression_ratio(data: Any, algorithm: str = 'gzip') -> float:
    """Estimate compression ratio for data"""
    if isinstance(data, pd.DataFrame):
        original_size = data.memory_usage(deep=True).sum()
        compressed = DataCompressor.compress_dataframe(data, algorithm)
        compressed_size = len(compressed)
    elif isinstance(data, np.ndarray):
        original_size = data.nbytes
        compressed = DataCompressor.compress_numpy_array(data, algorithm)
        compressed_size = len(compressed)
    elif isinstance(data, (dict, list)):
        original_size = len(json.dumps(data))
        compressed = DataCompressor.compress_json(data, algorithm)
        compressed_size = len(compressed)
    else:
        raise ValueError(f"Unsupported data type: {type(data)}")
    
    return original_size / compressed_size if compressed_size > 0 else 0

def optimize_storage_format(df: pd.DataFrame) -> str:
    """Recommend optimal storage format for DataFrame"""
    # Check data characteristics
    numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
    total_cols = len(df.columns)
    numeric_ratio = numeric_cols / total_cols if total_cols > 0 else 0
    
    # Check sparsity
    null_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
    
    if numeric_ratio > 0.8 and null_ratio < 0.1:
        return 'parquet'  # Best for numeric data
    elif null_ratio > 0.5:
        return 'pickle'  # Handles sparse data well
    else:
        return 'csv'  # Universal format