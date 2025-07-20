#!/usr/bin/env python3
"""
Performance Demonstration Script
Shows all improvements: caching, parallel processing, validation, error handling
"""

import time
import os
import sys
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our modules
from data_modules.data_fetcher import AppleDataFetcher
from data_modules.technical_analyzer import AppleTechnicalAnalyzer
from data_modules.sentiment_analyzer import SentimentAnalyzer
from data_modules.signal_generator import AppleSignalGenerator
from utils.cache_manager import get_cache_manager
from utils.parallel_processor import ParallelProcessor
from utils.validators import Validators, ValidationError

def print_section(title):
    """Print formatted section header"""
    print(f"\n{'='*80}")
    print(f"🚀 {title}")
    print('='*80)

def demonstrate_input_validation():
    """Show input validation in action"""
    print_section("INPUT VALIDATION DEMONSTRATION")
    
    # Valid inputs
    print("\n✅ Valid Inputs:")
    try:
        symbol = Validators.validate_stock_symbol('aapl', allow_lowercase=True)
        print(f"  Stock symbol 'aapl' → '{symbol}'")
        
        quantity = Validators.validate_integer(100, min_value=1, name="quantity")
        print(f"  Quantity 100 → {quantity}")
        
        risk_pct = Validators.validate_percentage(2.5, name="risk percentage")
        print(f"  Risk 2.5% → {risk_pct}%")
    except ValidationError as e:
        print(f"  ❌ Validation error: {e}")
    
    # Invalid inputs
    print("\n❌ Invalid Inputs (caught by validation):")
    test_cases = [
        ('AAPL123', 'stock symbol'),
        (-10, 'quantity'),
        (150, 'percentage')
    ]
    
    for value, name in test_cases:
        try:
            if name == 'stock symbol':
                Validators.validate_stock_symbol(str(value))
            elif name == 'quantity':
                Validators.validate_integer(value, min_value=1, name=name)
            elif name == 'percentage':
                Validators.validate_percentage(value, name=name)
        except ValidationError as e:
            print(f"  {name} '{value}' → {e}")

def demonstrate_caching():
    """Show caching performance improvement"""
    print_section("CACHING DEMONSTRATION")
    
    cache = get_cache_manager()
    fetcher = AppleDataFetcher()
    
    # Clear cache for fair comparison
    cache.clear()
    
    # First call (no cache)
    print("\n📊 First API call (no cache):")
    start = time.time()
    data1 = fetcher.fetch_stock_data()
    first_time = time.time() - start
    print(f"  Time taken: {first_time:.3f} seconds")
    
    # Second call (cached)
    print("\n💾 Second API call (cached):")
    start = time.time()
    data2 = fetcher.fetch_stock_data()
    cached_time = time.time() - start
    print(f"  Time taken: {cached_time:.3f} seconds")
    
    # Show improvement
    if cached_time > 0:
        speedup = first_time / cached_time
        print(f"\n⚡ Speedup: {speedup:.1f}x faster with cache!")
    
    # Show cache stats
    stats = cache.get_stats()
    print(f"\n📈 Cache Statistics:")
    print(f"  Hit rate: {stats['hit_rate']}")
    print(f"  Total requests: {stats['total_requests']}")
    print(f"  Memory items: {stats['memory_items']}")

def demonstrate_parallel_processing():
    """Show parallel processing improvement"""
    print_section("PARALLEL PROCESSING DEMONSTRATION")
    
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
    
    def fetch_symbol_data(symbol):
        """Simulate API call"""
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        return {'symbol': symbol, 'price': ticker.info.get('currentPrice', 0)}
    
    # Sequential processing
    print(f"\n🐌 Sequential Processing ({len(symbols)} stocks):")
    start = time.time()
    sequential_results = []
    for symbol in symbols:
        result = fetch_symbol_data(symbol)
        sequential_results.append(result)
        print(f"  Fetched {symbol}: ${result['price']:.2f}")
    sequential_time = time.time() - start
    print(f"  Total time: {sequential_time:.3f} seconds")
    
    # Parallel processing
    print(f"\n⚡ Parallel Processing ({len(symbols)} stocks):")
    start = time.time()
    with ParallelProcessor(max_workers=5) as processor:
        parallel_results = processor.map(fetch_symbol_data, symbols)
    parallel_time = time.time() - start
    
    for result in parallel_results:
        print(f"  Fetched {result['symbol']}: ${result['price']:.2f}")
    print(f"  Total time: {parallel_time:.3f} seconds")
    
    # Show improvement
    if parallel_time > 0:
        speedup = sequential_time / parallel_time
        print(f"\n⚡ Speedup: {speedup:.1f}x faster with parallel processing!")

def demonstrate_error_handling():
    """Show error handling and retry logic"""
    print_section("ERROR HANDLING DEMONSTRATION")
    
    from utils.error_handler import retry_with_backoff, RetryConfig
    
    call_count = 0
    
    @retry_with_backoff(RetryConfig(max_retries=3, initial_delay=0.5))
    def flaky_api_call():
        """Simulate API that fails sometimes"""
        nonlocal call_count
        call_count += 1
        
        if call_count < 3:
            print(f"  Attempt {call_count}: Failed (simulated error)")
            raise ConnectionError("Network error")
        
        print(f"  Attempt {call_count}: Success!")
        return {"status": "success", "data": "important data"}
    
    print("\n🔄 Retry Logic Demo (fails 2 times, succeeds on 3rd):")
    try:
        result = flaky_api_call()
        print(f"  Final result: {result}")
    except Exception as e:
        print(f"  Failed after all retries: {e}")

def run_complete_analysis():
    """Run complete analysis with all optimizations"""
    print_section("COMPLETE APPLE STOCK ANALYSIS")
    
    # Initialize components
    fetcher = AppleDataFetcher()
    analyzer = AppleTechnicalAnalyzer()
    sentiment = SentimentAnalyzer()
    signal_gen = AppleSignalGenerator()
    
    print("\n🔍 Fetching all data with optimizations...")
    start_time = time.time()
    
    # Fetch all data (uses parallel processing internally)
    all_data = fetcher.fetch_all_data()
    
    fetch_time = time.time() - start_time
    print(f"✅ Data fetched in {fetch_time:.2f} seconds")
    
    if all_data and all_data.get('stock_data'):
        # Technical analysis
        print("\n📊 Running Technical Analysis...")
        tech_start = time.time()
        
        stock_data = all_data['stock_data']
        hist_data = stock_data['historical_data']
        
        # Calculate indicators
        indicators = analyzer.calculate_all_indicators(hist_data)
        
        # Analyze patterns
        patterns = analyzer.analyze_chart_patterns(hist_data)
        
        tech_time = time.time() - tech_start
        print(f"✅ Technical analysis completed in {tech_time:.2f} seconds")
        
        # Display results
        if indicators:
            current_price = stock_data['current_price']
            print(f"\n💰 Current Price: ${current_price:.2f}")
            
            # Technical indicators
            print(f"\n📈 Technical Indicators:")
            print(f"  RSI: {indicators.get('RSI', 'N/A')}")
            print(f"  MACD: {indicators.get('MACD', {}).get('signal', 'N/A')}")
            print(f"  Bollinger Band: {indicators.get('BB', {}).get('position', 'N/A')}")
            
            # Moving averages
            print(f"\n📊 Moving Averages:")
            print(f"  SMA 20: ${indicators.get('SMA_20', 0):.2f}")
            print(f"  SMA 50: ${indicators.get('SMA_50', 0):.2f}")
            print(f"  EMA 12: ${indicators.get('EMA_12', 0):.2f}")
            
            # Patterns
            if patterns:
                print(f"\n🎯 Chart Patterns Detected:")
                for pattern_type, pattern_data in patterns.items():
                    if pattern_data.get('detected'):
                        print(f"  {pattern_type}: {pattern_data.get('direction', 'N/A')}")
        
        # Sentiment analysis
        if all_data.get('news_data'):
            print(f"\n😊 Sentiment Analysis:")
            news_sentiment = all_data['news_data']['avg_sentiment']
            print(f"  News Sentiment: {news_sentiment:.1f}/100")
            print(f"  Positive Articles: {all_data['news_data']['positive_count']}")
            print(f"  Negative Articles: {all_data['news_data']['negative_count']}")
        
        # Generate signals
        print(f"\n🚦 Trading Signals:")
        signals = signal_gen.generate_all_signals(all_data, indicators, patterns)
        
        if signals:
            print(f"  Primary Signal: {signals.get('primary_signal', 'HOLD')}")
            print(f"  Signal Strength: {signals.get('signal_strength', 0):.1f}/100")
            print(f"  Confidence Level: {signals.get('confidence_level', 'N/A')}")
            
            if signals.get('reasons'):
                print(f"\n  Signal Reasons:")
                for reason in signals['reasons'][:3]:  # Top 3 reasons
                    print(f"    • {reason}")
    
    # Show cache statistics
    cache = get_cache_manager()
    stats = cache.get_stats()
    
    print(f"\n📊 Performance Statistics:")
    print(f"  Total analysis time: {time.time() - start_time:.2f} seconds")
    print(f"  Cache hit rate: {stats['hit_rate']}")
    print(f"  API calls saved: {stats['hits']}")

def main():
    """Run all demonstrations"""
    print("\n" + "="*80)
    print("🎯 TRADING SYSTEM PERFORMANCE DEMONSTRATION")
    print("="*80)
    print("\nThis demo showcases all improvements:")
    print("• Input Validation")
    print("• Caching System")
    print("• Parallel Processing")
    print("• Error Handling")
    print("• Complete Analysis")
    
    # Run demonstrations
    demonstrate_input_validation()
    demonstrate_caching()
    demonstrate_parallel_processing()
    demonstrate_error_handling()
    run_complete_analysis()
    
    print("\n" + "="*80)
    print("✅ DEMONSTRATION COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    main()