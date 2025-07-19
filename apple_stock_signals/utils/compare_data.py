import pandas as pd
import numpy as np
from datetime import datetime

# Read the new historical data file
new_data = pd.read_csv('HistoricalData_1752903633769.csv')

# Read the existing AAPL data file
existing_data = pd.read_csv('historical_data/AAPL_historical_data.csv')

# Display basic info about both dataframes
print("=== NEW DATA FILE INFO ===")
print(f"Shape: {new_data.shape}")
print(f"Columns: {list(new_data.columns)}")
print(f"Date range: {new_data['Date'].iloc[-1]} to {new_data['Date'].iloc[0]}")
print("\n")

print("=== EXISTING DATA FILE INFO ===")
print(f"Shape: {existing_data.shape}")
print(f"Columns: {list(existing_data.columns)}")
print(f"Date range: {existing_data['Date'].iloc[0]} to {existing_data['Date'].iloc[-1]}")
print("\n")

# Parse dates properly
new_data['Date'] = pd.to_datetime(new_data['Date'], format='%m/%d/%Y')
existing_data['Date'] = pd.to_datetime(existing_data['Date'], utc=True)

# Extract date only for comparison
existing_data['Date_only'] = pd.to_datetime(existing_data['Date']).dt.date
new_data['Date_only'] = new_data['Date'].dt.date

# Get the most recent 20 days from new data
recent_new = new_data.head(20).copy()

# Find matching dates in existing data
print("=== COMPARING RECENT 20 DAYS ===")
print(f"New data date range: {recent_new['Date_only'].iloc[-1]} to {recent_new['Date_only'].iloc[0]}")

# Check if there are any overlapping dates
overlap_dates = set(recent_new['Date_only']) & set(existing_data['Date_only'])
print(f"Number of overlapping dates: {len(overlap_dates)}")

if len(overlap_dates) == 0:
    print("\nNO OVERLAPPING DATES FOUND!")
    print(f"Existing data ends at: {existing_data['Date_only'].iloc[-1]}")
    print(f"New data starts from: {recent_new['Date_only'].iloc[-1]}")
else:
    print(f"\nOverlapping dates found: {sorted(overlap_dates)}")
    
    # Create comparison dataframe
    comparison_results = []
    
    for date in sorted(overlap_dates, reverse=True):
        new_row = recent_new[recent_new['Date_only'] == date].iloc[0]
        existing_row = existing_data[existing_data['Date_only'] == date].iloc[0]
        
        # Extract prices from new data (removing $ sign)
        new_close = float(new_row['Close/Last'].replace('$', ''))
        new_open = float(new_row['Open'].replace('$', ''))
        new_high = float(new_row['High'].replace('$', ''))
        new_low = float(new_row['Low'].replace('$', ''))
        new_volume = new_row['Volume']
        
        # Extract prices from existing data
        existing_close = existing_row['Close']
        existing_open = existing_row['Open']
        existing_high = existing_row['High']
        existing_low = existing_row['Low']
        existing_volume = existing_row['Volume']
        
        # Calculate percentage differences
        close_diff_pct = ((new_close - existing_close) / existing_close) * 100
        open_diff_pct = ((new_open - existing_open) / existing_open) * 100
        high_diff_pct = ((new_high - existing_high) / existing_high) * 100
        low_diff_pct = ((new_low - existing_low) / existing_low) * 100
        volume_diff_pct = ((new_volume - existing_volume) / existing_volume) * 100
        
        comparison_results.append({
            'Date': date,
            'New_Close': new_close,
            'Existing_Close': existing_close,
            'Close_Diff_%': close_diff_pct,
            'New_Open': new_open,
            'Existing_Open': existing_open,
            'Open_Diff_%': open_diff_pct,
            'New_High': new_high,
            'Existing_High': existing_high,
            'High_Diff_%': high_diff_pct,
            'New_Low': new_low,
            'Existing_Low': existing_low,
            'Low_Diff_%': low_diff_pct,
            'New_Volume': new_volume,
            'Existing_Volume': existing_volume,
            'Volume_Diff_%': volume_diff_pct
        })
    
    # Create comparison dataframe
    comp_df = pd.DataFrame(comparison_results)
    
    print("\n=== DETAILED COMPARISON ===")
    for _, row in comp_df.iterrows():
        print(f"\nDate: {row['Date']}")
        print(f"  Close: ${row['New_Close']:.2f} vs ${row['Existing_Close']:.2f} (Diff: {row['Close_Diff_%']:.2f}%)")
        print(f"  Open:  ${row['New_Open']:.2f} vs ${row['Existing_Open']:.2f} (Diff: {row['Open_Diff_%']:.2f}%)")
        print(f"  High:  ${row['New_High']:.2f} vs ${row['Existing_High']:.2f} (Diff: {row['High_Diff_%']:.2f}%)")
        print(f"  Low:   ${row['New_Low']:.2f} vs ${row['Existing_Low']:.2f} (Diff: {row['Low_Diff_%']:.2f}%)")
        print(f"  Volume: {row['New_Volume']:,} vs {row['Existing_Volume']:,} (Diff: {row['Volume_Diff_%']:.2f}%)")
    
    # Summary statistics
    print("\n=== SUMMARY STATISTICS ===")
    print(f"Average Close Price Difference: {comp_df['Close_Diff_%'].mean():.2f}%")
    print(f"Average Open Price Difference: {comp_df['Open_Diff_%'].mean():.2f}%")
    print(f"Average High Price Difference: {comp_df['High_Diff_%'].mean():.2f}%")
    print(f"Average Low Price Difference: {comp_df['Low_Diff_%'].mean():.2f}%")
    print(f"Average Volume Difference: {comp_df['Volume_Diff_%'].mean():.2f}%")
    
    print(f"\nMax Close Price Difference: {comp_df['Close_Diff_%'].abs().max():.2f}%")
    print(f"Max Open Price Difference: {comp_df['Open_Diff_%'].abs().max():.2f}%")
    print(f"Max High Price Difference: {comp_df['High_Diff_%'].abs().max():.2f}%")
    print(f"Max Low Price Difference: {comp_df['Low_Diff_%'].abs().max():.2f}%")
    print(f"Max Volume Difference: {comp_df['Volume_Diff_%'].abs().max():.2f}%")

# Additional analysis - check column mappings
print("\n=== COLUMN MAPPING ===")
print("New Data Columns:")
print("  - Date")
print("  - Close/Last (corresponds to Close)")
print("  - Volume")
print("  - Open")
print("  - High")
print("  - Low")
print("\nExisting Data Columns (Price-related):")
print("  - Date")
print("  - Open")
print("  - High")
print("  - Low")
print("  - Close")
print("  - Volume")
print("  - Plus technical indicators: SMA_20, SMA_50, SMA_200, RSI, MACD, etc.")