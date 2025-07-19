#!/usr/bin/env python3
"""
Data Status Viewer
Shows the current status of historical data files
"""

import os
import pandas as pd
from datetime import datetime
import json
from tabulate import tabulate

def check_data_status(data_dir="historical_data"):
    """Check and display the status of all historical data files"""
    
    stocks = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'UNH']
    status_data = []
    
    print("\n📊 Historical Data Status Report")
    print("=" * 80)
    print(f"Report Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    for symbol in stocks:
        filepath = os.path.join(data_dir, f"{symbol}_historical_data.csv")
        
        if os.path.exists(filepath):
            try:
                # Read the CSV file
                df = pd.read_csv(filepath, index_col='Date', parse_dates=True)
                df.sort_index(inplace=True)
                
                # Get file stats
                file_size = os.path.getsize(filepath) / 1024  # KB
                record_count = len(df)
                
                # Get date range
                first_date = df.index[0].strftime('%Y-%m-%d')
                last_date = df.index[-1].strftime('%Y-%m-%d')
                
                # Calculate days behind
                days_behind = (datetime.now() - df.index[-1]).days
                
                # Determine status
                if days_behind == 0:
                    status = "✅ Up to date"
                    status_emoji = "🟢"
                elif days_behind <= 1:
                    status = "✅ Current"
                    status_emoji = "🟢"
                elif days_behind <= 7:
                    status = "⚠️ Needs update"
                    status_emoji = "🟡"
                else:
                    status = "❌ Outdated"
                    status_emoji = "🔴"
                
                # Check for data gaps
                date_range = pd.date_range(start=df.index[0], end=df.index[-1], freq='B')
                missing_days = len(date_range.difference(df.index))
                
                status_data.append([
                    f"{status_emoji} {symbol}",
                    record_count,
                    first_date,
                    last_date,
                    f"{days_behind} days",
                    f"{file_size:.1f} KB",
                    missing_days,
                    status
                ])
                
            except Exception as e:
                status_data.append([
                    f"❌ {symbol}",
                    "Error",
                    "-",
                    "-",
                    "-",
                    "-",
                    "-",
                    f"Error: {str(e)}"
                ])
        else:
            status_data.append([
                f"❌ {symbol}",
                "No file",
                "-",
                "-",
                "-",
                "-",
                "-",
                "File not found"
            ])
    
    # Display table
    headers = ["Stock", "Records", "First Date", "Last Date", "Days Behind", "File Size", "Missing Days", "Status"]
    print("\n" + tabulate(status_data, headers=headers, tablefmt="grid"))
    
    # Check metadata
    metadata_file = os.path.join(data_dir, "update_metadata.json")
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        print("\n📅 Last Update Information:")
        print("-" * 80)
        for symbol, info in metadata.items():
            if isinstance(info, dict) and 'last_update' in info:
                last_update = datetime.fromisoformat(info['last_update'])
                print(f"{symbol}: {last_update.strftime('%Y-%m-%d %H:%M:%S')} - Added {info.get('new_records_added', 0)} records")
    
    # Summary
    print("\n📈 Summary:")
    print("-" * 80)
    up_to_date = sum(1 for row in status_data if "✅" in row[0])
    needs_update = sum(1 for row in status_data if "⚠️" in row[0])
    outdated = sum(1 for row in status_data if "❌" in row[0])
    
    print(f"✅ Up to date: {up_to_date}")
    print(f"⚠️  Needs update: {needs_update}")
    print(f"❌ Outdated/Missing: {outdated}")
    
    # Recommendations
    if needs_update > 0 or outdated > 0:
        print("\n💡 Recommendations:")
        print("-" * 80)
        print("Run 'trade-update' to update all historical data files")
        for row in status_data:
            if "⚠️" in row[0] or "❌" in row[0]:
                symbol = row[0].split()[-1]
                print(f"- {symbol}: {row[7]}")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    check_data_status()