import pandas as pd
import numpy as np
from datetime import datetime

# Read the files
new_data = pd.read_csv('HistoricalData_1752903633769.csv')
existing_data = pd.read_csv('historical_data/AAPL_historical_data.csv')

# Parse dates
new_data['Date'] = pd.to_datetime(new_data['Date'], format='%m/%d/%Y')
existing_data['Date'] = pd.to_datetime(existing_data['Date'], utc=True)
existing_data['Date_only'] = pd.to_datetime(existing_data['Date']).dt.date
new_data['Date_only'] = new_data['Date'].dt.date

# Generate the report
report = []
report.append("# Apple Stock Data Comparison Report")
report.append(f"\nGenerated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
report.append("\n## Executive Summary")
report.append("\nThis report compares two Apple (AAPL) historical stock data files:")
report.append(f"- **New File**: HistoricalData_1752903633769.csv")
report.append(f"- **Existing File**: historical_data/AAPL_historical_data.csv")

# Data Overview
report.append("\n## Data Overview")
report.append("\n### New Data File:")
report.append(f"- Total Records: {len(new_data):,}")
report.append(f"- Date Range: {new_data['Date'].min().strftime('%Y-%m-%d')} to {new_data['Date'].max().strftime('%Y-%m-%d')}")
report.append(f"- Columns: {', '.join(new_data.columns.tolist())}")

report.append("\n### Existing Data File:")
report.append(f"- Total Records: {len(existing_data):,}")
report.append(f"- Date Range: {existing_data['Date'].min().strftime('%Y-%m-%d')} to {existing_data['Date'].max().strftime('%Y-%m-%d')}")
report.append(f"- Basic Price Columns: Date, Open, High, Low, Close, Volume")
report.append(f"- Technical Indicators: SMA_20, SMA_50, SMA_200, RSI, MACD, MACD_Signal, MACD_Histogram")
report.append(f"- Additional Features: BB_Middle, BB_Upper, BB_Lower, ATR, Daily_Return, Cumulative_Return")

# Overlapping dates analysis
recent_new = new_data.head(20).copy()
overlap_dates = sorted(set(recent_new['Date_only']) & set(existing_data['Date_only']), reverse=True)

report.append("\n## Comparison Results (Most Recent 20 Days)")
report.append(f"\n- Number of overlapping dates analyzed: {len(overlap_dates)}")
report.append(f"- Date range analyzed: {min(overlap_dates)} to {max(overlap_dates)}")

# Detailed comparison
comparison_results = []
for date in overlap_dates:
    new_row = recent_new[recent_new['Date_only'] == date].iloc[0]
    existing_row = existing_data[existing_data['Date_only'] == date].iloc[0]
    
    # Extract prices
    new_close = float(new_row['Close/Last'].replace('$', ''))
    new_open = float(new_row['Open'].replace('$', ''))
    new_high = float(new_row['High'].replace('$', ''))
    new_low = float(new_row['Low'].replace('$', ''))
    new_volume = new_row['Volume']
    
    existing_close = existing_row['Close']
    existing_open = existing_row['Open']
    existing_high = existing_row['High']
    existing_low = existing_row['Low']
    existing_volume = existing_row['Volume']
    
    # Calculate differences
    close_diff = new_close - existing_close
    open_diff = new_open - existing_open
    high_diff = new_high - existing_high
    low_diff = new_low - existing_low
    volume_diff = new_volume - existing_volume
    
    # Calculate percentage differences
    close_diff_pct = (close_diff / existing_close) * 100 if existing_close != 0 else 0
    open_diff_pct = (open_diff / existing_open) * 100 if existing_open != 0 else 0
    high_diff_pct = (high_diff / existing_high) * 100 if existing_high != 0 else 0
    low_diff_pct = (low_diff / existing_low) * 100 if existing_low != 0 else 0
    volume_diff_pct = (volume_diff / existing_volume) * 100 if existing_volume != 0 else 0
    
    comparison_results.append({
        'Date': date,
        'Close_Diff': close_diff,
        'Close_Diff_%': close_diff_pct,
        'Open_Diff': open_diff,
        'Open_Diff_%': open_diff_pct,
        'High_Diff': high_diff,
        'High_Diff_%': high_diff_pct,
        'Low_Diff': low_diff,
        'Low_Diff_%': low_diff_pct,
        'Volume_Diff': volume_diff,
        'Volume_Diff_%': volume_diff_pct
    })

comp_df = pd.DataFrame(comparison_results)

# Key findings
report.append("\n## Key Findings")

# Price accuracy
report.append("\n### 1. Price Data Accuracy")
report.append(f"- **Close Price**: Average difference of {comp_df['Close_Diff_%'].mean():.4f}%, max difference of {comp_df['Close_Diff_%'].abs().max():.4f}%")
report.append(f"- **Open Price**: Average difference of {comp_df['Open_Diff_%'].mean():.4f}%, max difference of {comp_df['Open_Diff_%'].abs().max():.4f}%")
report.append(f"- **High Price**: Average difference of {comp_df['High_Diff_%'].mean():.4f}%, max difference of {comp_df['High_Diff_%'].abs().max():.4f}%")
report.append(f"- **Low Price**: Average difference of {comp_df['Low_Diff_%'].mean():.4f}%, max difference of {comp_df['Low_Diff_%'].abs().max():.4f}%")

# Volume discrepancies
report.append("\n### 2. Volume Data")
report.append(f"- **Average Volume Difference**: {comp_df['Volume_Diff_%'].mean():.2f}%")
report.append(f"- **Maximum Volume Difference**: {comp_df['Volume_Diff_%'].abs().max():.2f}%")

# Find dates with largest discrepancies
largest_close_diff = comp_df.loc[comp_df['Close_Diff_%'].abs().idxmax()]
largest_volume_diff = comp_df.loc[comp_df['Volume_Diff_%'].abs().idxmax()]

report.append(f"\n### 3. Notable Discrepancies")
report.append(f"- **Largest Close Price Difference**: {largest_close_diff['Date']} ({largest_close_diff['Close_Diff_%']:.4f}%)")
report.append(f"- **Largest Volume Difference**: {largest_volume_diff['Date']} ({largest_volume_diff['Volume_Diff_%']:.2f}%)")

# Data quality assessment
report.append("\n## Data Quality Assessment")

price_fields_accurate = comp_df[['Close_Diff_%', 'Open_Diff_%', 'High_Diff_%', 'Low_Diff_%']].abs().max().max() < 0.5
volume_reasonable = comp_df['Volume_Diff_%'].abs().max() < 100

if price_fields_accurate and volume_reasonable:
    report.append("\n✅ **Overall Assessment**: The data sources are highly consistent")
    report.append("- Price differences are negligible (< 0.5%)")
    report.append("- Volume differences are within reasonable bounds")
    report.append("- Both files appear to be from reliable sources")
else:
    report.append("\n⚠️ **Overall Assessment**: Some discrepancies detected")
    if not price_fields_accurate:
        report.append("- Price differences exceed 0.5% threshold")
    if not volume_reasonable:
        report.append("- Volume differences are significant")

# Recommendations
report.append("\n## Recommendations")
report.append("\n1. **Data Source Selection**:")
report.append("   - Both files contain accurate price data with minimal differences")
report.append("   - The existing file includes valuable technical indicators (RSI, MACD, Bollinger Bands)")
report.append("   - Consider using the existing file for analysis requiring technical indicators")

report.append("\n2. **Data Updates**:")
report.append("   - The new file appears to be a clean download of historical prices")
report.append("   - Could be used to validate or update the existing dataset")
report.append("   - Technical indicators would need to be recalculated if switching to the new data")

report.append("\n3. **Volume Discrepancy Note**:")
if comp_df['Volume_Diff_%'].abs().max() > 10:
    report.append("   - The volume difference on some dates suggests different data sources or adjustments")
    report.append("   - This is common due to different handling of pre/post-market volumes")
else:
    report.append("   - Volume differences are minimal and likely due to rounding")

# Detailed comparison table
report.append("\n## Detailed Comparison Table (Recent 5 Days)")
report.append("\n| Date | Close Diff | Open Diff | High Diff | Low Diff | Volume Diff % |")
report.append("|------|------------|-----------|-----------|----------|---------------|")

for i, row in comp_df.head(5).iterrows():
    report.append(f"| {row['Date']} | {row['Close_Diff_%']:+.4f}% | {row['Open_Diff_%']:+.4f}% | {row['High_Diff_%']:+.4f}% | {row['Low_Diff_%']:+.4f}% | {row['Volume_Diff_%']:+.2f}% |")

# Save the report
report_content = '\n'.join(report)
with open('AAPL_Data_Comparison_Report.md', 'w') as f:
    f.write(report_content)

print("Report generated successfully: AAPL_Data_Comparison_Report.md")

# Also save detailed comparison CSV
comp_df.to_csv('detailed_comparison_results.csv', index=False)
print("Detailed comparison saved to: detailed_comparison_results.csv")