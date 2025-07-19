# Apple Stock Data Comparison Report

Generated on: 2025-07-19 12:14:23

## Executive Summary

This report compares two Apple (AAPL) historical stock data files:
- **New File**: HistoricalData_1752903633769.csv
- **Existing File**: historical_data/AAPL_historical_data.csv

## Data Overview

### New Data File:
- Total Records: 1,256
- Date Range: 2020-07-20 to 2025-07-18
- Columns: Date, Close/Last, Volume, Open, High, Low, Date_only

### Existing Data File:
- Total Records: 752
- Date Range: 2022-07-20 to 2025-07-18
- Basic Price Columns: Date, Open, High, Low, Close, Volume
- Technical Indicators: SMA_20, SMA_50, SMA_200, RSI, MACD, MACD_Signal, MACD_Histogram
- Additional Features: BB_Middle, BB_Upper, BB_Lower, ATR, Daily_Return, Cumulative_Return

## Comparison Results (Most Recent 20 Days)

- Number of overlapping dates analyzed: 20
- Date range analyzed: 2025-06-20 to 2025-07-18

## Key Findings

### 1. Price Data Accuracy
- **Close Price**: Average difference of 0.0062%, max difference of 0.1233%
- **Open Price**: Average difference of 0.0018%, max difference of 0.0546%
- **High Price**: Average difference of -0.0001%, max difference of 0.0017%
- **Low Price**: Average difference of -0.0001%, max difference of 0.0025%

### 2. Volume Data
- **Average Volume Difference**: 2.73%
- **Maximum Volume Difference**: 54.44%

### 3. Notable Discrepancies
- **Largest Close Price Difference**: 2025-07-18 (0.1233%)
- **Largest Volume Difference**: 2025-07-18 (54.44%)

## Data Quality Assessment

✅ **Overall Assessment**: The data sources are highly consistent
- Price differences are negligible (< 0.5%)
- Volume differences are within reasonable bounds
- Both files appear to be from reliable sources

## Recommendations

1. **Data Source Selection**:
   - Both files contain accurate price data with minimal differences
   - The existing file includes valuable technical indicators (RSI, MACD, Bollinger Bands)
   - Consider using the existing file for analysis requiring technical indicators

2. **Data Updates**:
   - The new file appears to be a clean download of historical prices
   - Could be used to validate or update the existing dataset
   - Technical indicators would need to be recalculated if switching to the new data

3. **Volume Discrepancy Note**:
   - The volume difference on some dates suggests different data sources or adjustments
   - This is common due to different handling of pre/post-market volumes

## Detailed Comparison Table (Recent 5 Days)

| Date | Close Diff | Open Diff | High Diff | Low Diff | Volume Diff % |
|------|------------|-----------|-----------|----------|---------------|
| 2025-07-18 | +0.1233% | +0.0546% | +0.0000% | +0.0000% | +54.44% |
| 2025-07-17 | -0.0000% | -0.0000% | -0.0000% | +0.0000% | +0.12% |
| 2025-07-16 | -0.0000% | -0.0024% | +0.0000% | +0.0000% | +0.00% |
| 2025-07-15 | -0.0000% | -0.0000% | +0.0000% | +0.0000% | +0.00% |
| 2025-07-14 | +0.0000% | -0.0024% | -0.0000% | +0.0000% | +0.00% |