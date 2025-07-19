#!/usr/bin/env python3
"""
Historical Data Fetcher
Fetches 3 years of historical data for stocks and saves to CSV files
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import time
from typing import Dict, List, Optional
import json

class HistoricalDataFetcher:
    def __init__(self, data_dir: str = "historical_data"):
        """Initialize the historical data fetcher."""
        self.data_dir = data_dir
        self.ensure_data_directory()
        
    def ensure_data_directory(self):
        """Create data directory if it doesn't exist."""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            print(f"Created directory: {self.data_dir}")
    
    def fetch_historical_data(self, symbol: str, years: int = 3) -> pd.DataFrame:
        """
        Fetch historical data for a given symbol.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            years: Number of years of data to fetch
            
        Returns:
            DataFrame with historical data
        """
        try:
            print(f"Fetching {years} years of data for {symbol}...")
            
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=years*365)
            
            # Create ticker object
            ticker = yf.Ticker(symbol)
            
            # Fetch historical data
            hist_data = ticker.history(
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                interval='1d'
            )
            
            if hist_data.empty:
                print(f"Warning: No data found for {symbol}")
                return pd.DataFrame()
            
            # Add additional calculated columns
            hist_data['Symbol'] = symbol
            hist_data['SMA_20'] = hist_data['Close'].rolling(window=20).mean()
            hist_data['SMA_50'] = hist_data['Close'].rolling(window=50).mean()
            hist_data['SMA_200'] = hist_data['Close'].rolling(window=200).mean()
            
            # Calculate daily returns
            hist_data['Daily_Return'] = hist_data['Close'].pct_change()
            hist_data['Cumulative_Return'] = (1 + hist_data['Daily_Return']).cumprod() - 1
            
            # Calculate RSI
            hist_data['RSI'] = self.calculate_rsi(hist_data['Close'])
            
            # Calculate MACD
            exp1 = hist_data['Close'].ewm(span=12, adjust=False).mean()
            exp2 = hist_data['Close'].ewm(span=26, adjust=False).mean()
            hist_data['MACD'] = exp1 - exp2
            hist_data['MACD_Signal'] = hist_data['MACD'].ewm(span=9, adjust=False).mean()
            hist_data['MACD_Histogram'] = hist_data['MACD'] - hist_data['MACD_Signal']
            
            # Calculate Bollinger Bands
            bb_period = 20
            bb_std = 2
            hist_data['BB_Middle'] = hist_data['Close'].rolling(window=bb_period).mean()
            bb_stddev = hist_data['Close'].rolling(window=bb_period).std()
            hist_data['BB_Upper'] = hist_data['BB_Middle'] + (bb_stddev * bb_std)
            hist_data['BB_Lower'] = hist_data['BB_Middle'] - (bb_stddev * bb_std)
            
            # Calculate ATR
            hist_data['ATR'] = self.calculate_atr(hist_data)
            
            print(f"Successfully fetched {len(hist_data)} days of data for {symbol}")
            return hist_data
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high_low = df['High'] - df['Low']
        high_close = abs(df['High'] - df['Close'].shift())
        low_close = abs(df['Low'] - df['Close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        atr = true_range.rolling(period).mean()
        
        return atr
    
    def save_to_csv(self, df: pd.DataFrame, symbol: str):
        """Save DataFrame to CSV file."""
        if df.empty:
            print(f"Skipping save for {symbol} - no data")
            return
            
        filename = os.path.join(self.data_dir, f"{symbol}_historical_data.csv")
        df.to_csv(filename)
        print(f"Saved {symbol} data to {filename}")
    
    def save_to_parquet(self, df: pd.DataFrame, symbol: str):
        """Save DataFrame to Parquet file for better performance."""
        if df.empty:
            print(f"Skipping save for {symbol} - no data")
            return
            
        try:
            filename = os.path.join(self.data_dir, f"{symbol}_historical_data.parquet")
            df.to_parquet(filename)
            print(f"Saved {symbol} data to {filename} (Parquet format)")
        except ImportError:
            print(f"Parquet format not available - install pyarrow or fastparquet for better performance")
    
    def load_historical_data(self, symbol: str, format: str = 'csv') -> pd.DataFrame:
        """Load historical data from file."""
        if format == 'csv':
            filename = os.path.join(self.data_dir, f"{symbol}_historical_data.csv")
            if os.path.exists(filename):
                return pd.read_csv(filename, index_col=0, parse_dates=True)
        else:
            filename = os.path.join(self.data_dir, f"{symbol}_historical_data.parquet")
            if os.path.exists(filename):
                return pd.read_parquet(filename)
        
        print(f"No saved data found for {symbol}")
        return pd.DataFrame()
    
    def fetch_multiple_stocks(self, symbols: List[str], years: int = 3):
        """Fetch data for multiple stocks."""
        results = {}
        metadata = {
            'fetch_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'years': years,
            'symbols': symbols,
            'status': {}
        }
        
        for symbol in symbols:
            print(f"\n{'='*50}")
            print(f"Processing {symbol}...")
            print(f"{'='*50}")
            
            # Fetch data
            df = self.fetch_historical_data(symbol, years)
            
            if not df.empty:
                # Save to both CSV and Parquet
                self.save_to_csv(df, symbol)
                self.save_to_parquet(df, symbol)
                
                # Store in results
                results[symbol] = df
                
                # Update metadata
                metadata['status'][symbol] = {
                    'success': True,
                    'rows': len(df),
                    'start_date': df.index[0].strftime('%Y-%m-%d'),
                    'end_date': df.index[-1].strftime('%Y-%m-%d'),
                    'file_csv': f"{symbol}_historical_data.csv",
                    'file_parquet': f"{symbol}_historical_data.parquet"
                }
            else:
                metadata['status'][symbol] = {
                    'success': False,
                    'error': 'No data fetched'
                }
            
            # Small delay to avoid rate limiting
            time.sleep(1)
        
        # Save metadata
        self.save_metadata(metadata)
        
        return results
    
    def save_metadata(self, metadata: Dict):
        """Save metadata about the data fetch."""
        filename = os.path.join(self.data_dir, 'fetch_metadata.json')
        with open(filename, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"\nSaved metadata to {filename}")
    
    def generate_summary_report(self, symbols: List[str]):
        """Generate a summary report of all historical data."""
        report = []
        report.append("="*80)
        report.append("HISTORICAL DATA SUMMARY REPORT")
        report.append("="*80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        for symbol in symbols:
            df = self.load_historical_data(symbol)
            if not df.empty:
                report.append(f"\n{symbol} Summary:")
                report.append("-" * 40)
                report.append(f"Total Trading Days: {len(df)}")
                report.append(f"Date Range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
                report.append(f"Starting Price: ${df['Close'].iloc[0]:.2f}")
                report.append(f"Ending Price: ${df['Close'].iloc[-1]:.2f}")
                report.append(f"Total Return: {((df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1) * 100:.2f}%")
                report.append(f"Highest Price: ${df['High'].max():.2f}")
                report.append(f"Lowest Price: ${df['Low'].min():.2f}")
                report.append(f"Average Volume: {df['Volume'].mean():,.0f}")
                report.append(f"Volatility (Std Dev): {df['Daily_Return'].std() * 100:.2f}%")
        
        # Save report
        report_filename = os.path.join(self.data_dir, 'historical_data_summary.txt')
        with open(report_filename, 'w') as f:
            f.write('\n'.join(report))
        
        print(f"\nSaved summary report to {report_filename}")
        print('\n'.join(report))


def main():
    """Main function to fetch historical data."""
    # List of stocks to analyze
    symbols = ['AAPL', 'GOOGL', 'TSLA', 'MSFT', 'UNH']
    
    # Initialize fetcher
    fetcher = HistoricalDataFetcher()
    
    print("🚀 Starting Historical Data Fetch")
    print(f"Fetching 3 years of data for: {', '.join(symbols)}")
    print("This may take a few moments...\n")
    
    # Fetch data for all symbols
    results = fetcher.fetch_multiple_stocks(symbols, years=3)
    
    # Generate summary report
    print("\n📊 Generating Summary Report...")
    fetcher.generate_summary_report(symbols)
    
    print("\n✅ Historical data fetch complete!")
    print(f"Data saved in: {os.path.abspath(fetcher.data_dir)}/")


if __name__ == "__main__":
    main()