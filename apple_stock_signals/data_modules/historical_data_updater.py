#!/usr/bin/env python3
"""
Historical Data Updater
Updates existing historical data files with new data to keep them current
"""

import os
import sys
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import json
import logging
from typing import Dict, Optional, Tuple

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HistoricalDataUpdater:
    def __init__(self, data_dir: str = "historical_data"):
        """Initialize the data updater with the historical data directory"""
        self.data_dir = data_dir
        self.metadata_file = os.path.join(data_dir, "update_metadata.json")
        self.stocks = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'UNH']
        
    def load_metadata(self) -> Dict:
        """Load metadata about last updates"""
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {}
    
    def save_metadata(self, metadata: Dict):
        """Save metadata about updates"""
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def get_last_date_in_file(self, filepath: str) -> Optional[datetime]:
        """Get the last date in the historical data file"""
        try:
            if not os.path.exists(filepath):
                return None
                
            df = pd.read_csv(filepath, index_col='Date', parse_dates=True)
            if df.empty:
                return None
                
            # Sort by date and get the last date
            df.sort_index(inplace=True)
            last_date = df.index[-1]
            return last_date.to_pydatetime()
            
        except Exception as e:
            logger.error(f"Error reading {filepath}: {e}")
            return None
    
    def fetch_incremental_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Fetch data from start_date to end_date"""
        try:
            logger.info(f"Fetching {symbol} data from {start_date.date()} to {end_date.date()}")
            
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, interval='1d')
            
            if df.empty:
                logger.warning(f"No new data available for {symbol}")
                return pd.DataFrame()
            
            # Add symbol column
            df['Symbol'] = symbol
            
            # Calculate additional metrics
            df['Daily_Return'] = df['Close'].pct_change()
            df['MA_20'] = df['Close'].rolling(window=20).mean()
            df['MA_50'] = df['Close'].rolling(window=50).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume'].rolling(window=20).mean()
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    def update_historical_file(self, symbol: str) -> Tuple[bool, str]:
        """Update the historical data file for a given symbol"""
        filepath = os.path.join(self.data_dir, f"{symbol}_historical_data.csv")
        
        try:
            # Get the last date in the file
            last_date = self.get_last_date_in_file(filepath)
            
            if last_date is None:
                # File doesn't exist or is empty, fetch last 3 years
                start_date = datetime.now() - timedelta(days=3*365)
                logger.info(f"No existing data for {symbol}, fetching last 3 years")
                existing_df = pd.DataFrame()
            else:
                # Fetch from the day after last date
                start_date = last_date + timedelta(days=1)
                existing_df = pd.read_csv(filepath, index_col='Date', parse_dates=True)
                
            end_date = datetime.now()
            
            # Check if update is needed
            if start_date.date() >= end_date.date():
                message = f"{symbol}: Already up to date (last date: {last_date.date() if last_date else 'N/A'})"
                logger.info(message)
                return True, message
            
            # Fetch new data
            new_df = self.fetch_incremental_data(symbol, start_date, end_date)
            
            if new_df.empty:
                message = f"{symbol}: No new data available"
                logger.info(message)
                return True, message
            
            # Combine with existing data
            if not existing_df.empty:
                # Remove any duplicate dates (keep the new data)
                combined_df = pd.concat([existing_df, new_df])
                combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
                combined_df.sort_index(inplace=True)
            else:
                combined_df = new_df
            
            # Save the updated file
            combined_df.to_csv(filepath)
            
            # Update metadata
            metadata = self.load_metadata()
            metadata[symbol] = {
                'last_update': datetime.now().isoformat(),
                'last_date': combined_df.index[-1].strftime('%Y-%m-%d'),
                'total_records': len(combined_df),
                'new_records_added': len(new_df)
            }
            self.save_metadata(metadata)
            
            message = f"{symbol}: Added {len(new_df)} new records (up to {combined_df.index[-1].date()})"
            logger.info(message)
            return True, message
            
        except Exception as e:
            message = f"{symbol}: Update failed - {str(e)}"
            logger.error(message)
            return False, message
    
    def update_all_stocks(self) -> Dict[str, str]:
        """Update all stock historical data files"""
        results = {}
        
        logger.info("="*60)
        logger.info("Starting Historical Data Update")
        logger.info("="*60)
        
        for symbol in self.stocks:
            success, message = self.update_historical_file(symbol)
            results[symbol] = {'success': success, 'message': message}
        
        # Create summary report
        self.create_update_report(results)
        
        return results
    
    def create_update_report(self, results: Dict[str, Dict]):
        """Create a summary report of the update process"""
        report_path = os.path.join(self.data_dir, "update_report.txt")
        
        with open(report_path, 'w') as f:
            f.write("Historical Data Update Report\n")
            f.write("="*60 + "\n")
            f.write(f"Update Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for symbol, result in results.items():
                status = "✓" if result['success'] else "✗"
                f.write(f"{status} {result['message']}\n")
            
            f.write("\n" + "="*60 + "\n")
            
            # Add file statistics
            f.write("\nFile Statistics:\n")
            for symbol in self.stocks:
                filepath = os.path.join(self.data_dir, f"{symbol}_historical_data.csv")
                if os.path.exists(filepath):
                    df = pd.read_csv(filepath, index_col='Date', parse_dates=True)
                    f.write(f"\n{symbol}:\n")
                    f.write(f"  - Records: {len(df)}\n")
                    f.write(f"  - Date Range: {df.index[0].date()} to {df.index[-1].date()}\n")
                    f.write(f"  - File Size: {os.path.getsize(filepath) / 1024:.1f} KB\n")
    
    def validate_data_integrity(self, symbol: str) -> bool:
        """Validate the integrity of historical data"""
        filepath = os.path.join(self.data_dir, f"{symbol}_historical_data.csv")
        
        try:
            df = pd.read_csv(filepath, index_col='Date', parse_dates=True)
            
            # Check for duplicate dates
            duplicates = df.index.duplicated().sum()
            if duplicates > 0:
                logger.warning(f"{symbol}: Found {duplicates} duplicate dates")
                # Remove duplicates
                df = df[~df.index.duplicated(keep='last')]
                df.to_csv(filepath)
            
            # Check for missing required columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.error(f"{symbol}: Missing columns: {missing_columns}")
                return False
            
            # Check for data gaps (weekends excluded)
            df.sort_index(inplace=True)
            date_range = pd.date_range(start=df.index[0], end=df.index[-1], freq='B')  # Business days
            missing_dates = date_range.difference(df.index)
            
            if len(missing_dates) > 10:  # Allow some gaps for holidays
                logger.warning(f"{symbol}: {len(missing_dates)} missing business days")
            
            return True
            
        except Exception as e:
            logger.error(f"Validation error for {symbol}: {e}")
            return False


def main():
    """Main function to run the updater"""
    updater = HistoricalDataUpdater()
    
    # Create data directory if it doesn't exist
    os.makedirs(updater.data_dir, exist_ok=True)
    
    # Update all stocks
    results = updater.update_all_stocks()
    
    # Validate data integrity
    print("\nValidating data integrity...")
    for symbol in updater.stocks:
        if updater.validate_data_integrity(symbol):
            print(f"✓ {symbol}: Data integrity verified")
        else:
            print(f"✗ {symbol}: Data integrity issues found")
    
    print("\nUpdate complete! Check historical_data/update_report.txt for details.")


if __name__ == "__main__":
    main()