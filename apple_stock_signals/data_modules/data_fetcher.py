import yfinance as yf
import pandas as pd
import numpy as np
from newsapi import NewsApiClient
import requests
from datetime import datetime, timedelta
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Optional

from core_scripts.config import API_KEYS, VERIFICATION_SETTINGS
from utils.error_handler import (
    retry_with_backoff, 
    handle_api_errors, 
    rate_limited,
    safe_request,
    RetryConfig,
    RateLimiter
)
from utils.cache_manager import cached, get_cached_or_fetch, CACHE_CONFIGS
from utils.parallel_processor import ParallelProcessor, run_async_fetch

# Configure logging
logger = logging.getLogger(__name__)

class AppleDataFetcher:
    def __init__(self):
        self.news_key = API_KEYS['NEWS_API_KEY']
        self.av_key = API_KEYS.get('ALPHA_VANTAGE_KEY')
        if self.news_key and self.news_key != 'your_news_api_key_here':
            self.news_api = NewsApiClient(api_key=self.news_key)
        else:
            self.news_api = None
        self.symbol = "AAPL"
        
    def fetch_all_data(self):
        """Fetch all AAPL data using parallel processing"""
        print("🔄 Fetching Apple Stock Data...")
        start_time = time.time()
        
        # Use parallel processing for independent data fetches
        with ParallelProcessor(max_workers=5) as processor:
            # Submit all tasks
            futures = {
                processor.executor.submit(self.fetch_stock_data): 'stock_data',
                processor.executor.submit(self.fetch_news_sentiment): 'news_data',
                processor.executor.submit(self.fetch_social_sentiment): 'social_data',
                processor.executor.submit(self.fetch_fundamental_data): 'fundamental_data'
            }
            
            # Collect results
            results = {}
            for future in as_completed(futures):
                data_type = futures[future]
                try:
                    results[data_type] = future.result()
                except Exception as e:
                    logger.error(f"Failed to fetch {data_type}: {e}")
                    results[data_type] = None
        
        # Verify data accuracy (depends on stock_data)
        if results.get('stock_data'):
            verification_results = self.verify_data_accuracy(results['stock_data'])
        else:
            verification_results = None
        
        fetch_time = time.time() - start_time
        print(f"⚡ Data fetched in {fetch_time:.2f} seconds")
        
        # Display all fetched data
        self.display_fetched_data(
            results.get('stock_data'), 
            results.get('news_data'), 
            results.get('social_data'), 
            results.get('fundamental_data')
        )
        
        # Display verification results
        self.display_verification_results(verification_results)
        
        return {
            'stock_data': results.get('stock_data'),
            'news_data': results.get('news_data'),
            'social_data': results.get('social_data'),
            'fundamental_data': results.get('fundamental_data'),
            'verification_results': verification_results
        }
    
    @cached(ttl=CACHE_CONFIGS['stock_quotes']['ttl'], key_prefix='stock')
    @retry_with_backoff(RetryConfig(max_retries=3, initial_delay=1.0))
    @handle_api_errors(default_return=None, log_errors=True)
    def fetch_stock_data(self):
        """Fetch AAPL stock price data with caching and retry logic"""
        ticker = yf.Ticker(self.symbol)
        
        # Get historical data (1 year)
        hist = ticker.history(period="1y")
        
        if hist.empty:
            raise ValueError("No historical data returned from yfinance")
        
        # Get real-time info
        info = ticker.info
        
        # Get current price
        current_price = hist['Close'].iloc[-1]
        
        # Calculate additional metrics
        volume_avg = hist['Volume'].rolling(window=20).mean().iloc[-1]
        volume_current = hist['Volume'].iloc[-1]
        volume_ratio = volume_current / volume_avg if volume_avg > 0 else 1.0
        
        # Get options data
        options_data = self.fetch_options_data(ticker)
        
        return {
            'historical_data': hist,
            'current_price': current_price,
            'volume_ratio': volume_ratio,
            'company_info': info,
            'options_data': options_data
        }
    
    @cached(ttl=CACHE_CONFIGS['news_data']['ttl'], key_prefix='news')
    @retry_with_backoff(RetryConfig(max_retries=2, initial_delay=2.0))
    @rate_limited(calls_per_second=2.0)  # NewsAPI rate limiting
    def fetch_news_sentiment(self):
        """Fetch Apple news and analyze sentiment with caching and retry logic"""
        print("📰 Fetching Apple news...")
        
        if not self.news_api:
            print("⚠️ News API key not configured - using mock data")
            return self.get_mock_news_data()
        
        try:
            # Get news from last 7 days
            news = self.news_api.get_everything(
                q="Apple stock OR AAPL",
                language='en',
                sort_by='publishedAt',
                from_param=(datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'),
                page_size=50
            )
            
            articles = news.get('articles', [])
            
            # Analyze sentiment for each article
            sentiment_scores = []
            for article in articles:
                if article.get('title') and article.get('description'):
                    sentiment = self.analyze_text_sentiment(
                        article['title'] + " " + article['description']
                    )
                    sentiment_scores.append({
                        'title': article['title'],
                        'sentiment_score': sentiment['score'],
                        'sentiment_label': sentiment['label'],
                        'published': article['publishedAt'],
                        'source': article['source']['name']
                    })
            
            # Calculate average sentiment
            if sentiment_scores:
                avg_sentiment = np.mean([s['sentiment_score'] for s in sentiment_scores])
                positive_count = sum(1 for s in sentiment_scores if s['sentiment_score'] > 60)
                negative_count = sum(1 for s in sentiment_scores if s['sentiment_score'] < 40)
            else:
                avg_sentiment = 50
                positive_count = 0
                negative_count = 0
            
            return {
                'articles': sentiment_scores[:5],  # Top 5 articles
                'avg_sentiment': avg_sentiment,
                'total_articles': len(sentiment_scores),
                'positive_count': positive_count,
                'negative_count': negative_count,
                'neutral_count': len(sentiment_scores) - positive_count - negative_count
            }
        except Exception as e:
            logger.warning(f"News API failed, falling back to mock data: {e}")
            return self.get_mock_news_data()
    
    def get_mock_news_data(self):
        """Return mock news data for demo purposes"""
        return {
            'articles': [
                {
                    'title': 'Apple Reports Strong Q4 Earnings, Beats Expectations',
                    'sentiment_score': 75.2,
                    'sentiment_label': 'Positive',
                    'published': datetime.now().isoformat(),
                    'source': 'Mock News'
                },
                {
                    'title': 'Apple iPhone 15 Sales Show Mixed Results in Key Markets',
                    'sentiment_score': 52.8,
                    'sentiment_label': 'Neutral',
                    'published': datetime.now().isoformat(),
                    'source': 'Mock News'
                }
            ],
            'avg_sentiment': 65.5,
            'total_articles': 15,
            'positive_count': 9,
            'negative_count': 3,
            'neutral_count': 3
        }
    
    @cached(ttl=CACHE_CONFIGS['news_data']['ttl'], key_prefix='social')
    def fetch_social_sentiment(self):
        """Fetch Apple social media sentiment with caching"""
        try:
            print("💬 Fetching social media sentiment...")
            
            # Fetch Reddit data with caching
            reddit_data = get_cached_or_fetch(
                'news_data',
                self.fetch_reddit_sentiment
            )
            
            # Twitter sentiment (simplified - would need Twitter API)
            twitter_data = {'avg_sentiment': 50, 'post_count': 0}  # Placeholder
            
            # Combine social sentiments
            if reddit_data['post_count'] > 0:
                combined_sentiment = reddit_data['avg_sentiment']
            else:
                combined_sentiment = 50
            
            return {
                'reddit_data': reddit_data,
                'twitter_data': twitter_data,
                'combined_sentiment': combined_sentiment
            }
            
        except Exception as e:
            print(f"Error fetching social sentiment: {e}")
            return {'combined_sentiment': 50}
    
    @retry_with_backoff(RetryConfig(max_retries=3, initial_delay=1.0, exponential_base=2.0))
    @rate_limited(calls_per_second=0.5)  # Reddit is strict with rate limits
    def fetch_reddit_sentiment(self):
        """Fetch Reddit sentiment for Apple with retry logic"""
        headers = {'User-Agent': 'AppleStockAnalyzer/1.0'}
        subreddits = ['wallstreetbets', 'investing', 'stocks', 'apple']
        all_posts = []
        
        # Use rate limiter for subreddit requests
        reddit_limiter = RateLimiter(calls_per_second=1.0)
        
        for subreddit in subreddits:
            reddit_limiter.wait_if_needed()
            
            url = f"https://www.reddit.com/r/{subreddit}/search.json"
            params = {
                'q': 'Apple OR AAPL',
                'sort': 'new',
                'limit': 25,
                't': 'week'
            }
            
            response = safe_request(
                method='GET',
                url=url,
                headers=headers,
                params=params,
                timeout=10,
                retry_config=RetryConfig(max_retries=2)
            )
            
            if response and response.status_code == 200:
                data = response.json()
                posts = data.get('data', {}).get('children', [])
                all_posts.extend(posts)
        
        # Analyze sentiment
        sentiments = []
        for post in all_posts:
            post_data = post.get('data', {})
            text = post_data.get('title', '') + " " + post_data.get('selftext', '')
            if text.strip():
                sentiment = self.analyze_text_sentiment(text)
                sentiments.append(sentiment['score'])
        
        if sentiments:
            avg_sentiment = np.mean(sentiments)
            bullish_posts = sum(1 for s in sentiments if s > 60)
            bearish_posts = sum(1 for s in sentiments if s < 40)
        else:
            avg_sentiment = 50
            bullish_posts = 0
            bearish_posts = 0
        
        return {
            'avg_sentiment': avg_sentiment,
            'post_count': len(sentiments),
            'bullish_posts': bullish_posts,
            'bearish_posts': bearish_posts
        }
    
    @cached(ttl=CACHE_CONFIGS['historical_data']['ttl'], key_prefix='fundamental')
    def fetch_fundamental_data(self):
        """Fetch fundamental data for Apple with caching"""
        try:
            ticker = yf.Ticker(self.symbol)
            info = ticker.info
            
            # Get key fundamental metrics
            fundamental_data = {
                'pe_ratio': info.get('trailingPE', 0),
                'forward_pe': info.get('forwardPE', 0),
                'peg_ratio': info.get('pegRatio', 0),
                'price_to_book': info.get('priceToBook', 0),
                'revenue_growth': info.get('revenueGrowth', 0),
                'earnings_growth': info.get('earningsGrowth', 0),
                'profit_margin': info.get('profitMargins', 0),
                'debt_to_equity': info.get('debtToEquity', 0),
                'return_on_equity': info.get('returnOnEquity', 0),
                'market_cap': info.get('marketCap', 0),
                'enterprise_value': info.get('enterpriseValue', 0),
                'dividend_yield': info.get('dividendYield', 0),
                'beta': info.get('beta', 1.0)
            }
            
            return fundamental_data
            
        except Exception as e:
            print(f"Error fetching fundamental data: {e}")
            return {}
    
    def fetch_options_data(self, ticker):
        """Fetch options data for Apple"""
        try:
            options_dates = ticker.options
            if not options_dates:
                return None
            
            # Get options for nearest expiration
            options_data = ticker.option_chain(options_dates[0])
            
            # Calculate put/call metrics
            calls_volume = options_data.calls['volume'].sum()
            puts_volume = options_data.puts['volume'].sum()
            
            if calls_volume > 0:
                put_call_ratio = puts_volume / calls_volume
            else:
                put_call_ratio = 1.0
            
            return {
                'put_call_ratio': put_call_ratio,
                'total_call_volume': calls_volume,
                'total_put_volume': puts_volume,
                'expiration_date': options_dates[0]
            }
            
        except Exception as e:
            print(f"Error fetching options data: {e}")
            return None
    
    def analyze_text_sentiment(self, text):
        """Analyze sentiment of text using TextBlob"""
        try:
            from textblob import TextBlob
            
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity  # -1 to 1
            
            # Convert to 0-100 scale
            score = (polarity + 1) * 50
            
            if score > 60:
                label = 'Positive'
            elif score < 40:
                label = 'Negative'
            else:
                label = 'Neutral'
            
            return {
                'score': score,
                'label': label,
                'polarity': polarity
            }
            
        except Exception as e:
            return {'score': 50, 'label': 'Neutral', 'polarity': 0}
    
    def verify_data_accuracy(self, stock_data):
        """Verify the accuracy of fetched stock data"""
        print("🔍 Verifying data accuracy...")
        
        verification_results = {
            'status': 'UNKNOWN',
            'sources_compared': 0,
            'price_discrepancies': [],
            'data_quality_checks': [],
            'anomalies_detected': [],
            'confidence_score': 0
        }
        
        if not stock_data:
            verification_results['status'] = 'FAILED'
            verification_results['data_quality_checks'].append("❌ No stock data available")
            return verification_results
        
        try:
            # 1. Cross-verify with multiple sources
            price_verification = self.cross_verify_prices(stock_data)
            verification_results['sources_compared'] = price_verification['sources_compared']
            verification_results['price_discrepancies'] = price_verification['discrepancies']
            
            # 2. Check for data anomalies
            anomaly_check = self.check_data_anomalies(stock_data)
            verification_results['anomalies_detected'] = anomaly_check['anomalies']
            
            # 3. Validate data ranges
            range_check = self.validate_data_ranges(stock_data)
            verification_results['data_quality_checks'].extend(range_check['checks'])
            
            # 4. Check data freshness
            freshness_check = self.check_data_freshness(stock_data)
            verification_results['data_quality_checks'].extend(freshness_check['checks'])
            
            # 5. Calculate overall confidence score
            confidence_score = self.calculate_data_confidence(price_verification, anomaly_check, range_check, freshness_check)
            verification_results['confidence_score'] = confidence_score
            
            # 6. Determine overall status
            if confidence_score >= 90:
                verification_results['status'] = 'EXCELLENT'
            elif confidence_score >= 75:
                verification_results['status'] = 'GOOD'
            elif confidence_score >= 50:
                verification_results['status'] = 'FAIR'
            else:
                verification_results['status'] = 'POOR'
            
            return verification_results
            
        except Exception as e:
            verification_results['status'] = 'ERROR'
            verification_results['data_quality_checks'].append(f"❌ Verification error: {e}")
            return verification_results
    
    def cross_verify_prices(self, stock_data):
        """Cross-verify prices from multiple sources using parallel fetching"""
        verification_result = {
            'sources_compared': 0,
            'discrepancies': [],
            'primary_price': stock_data['current_price']
        }
        
        sources = [('Yahoo Finance', stock_data['current_price'])]
        
        # Prepare parallel fetch tasks
        fetch_tasks = []
        
        if self.av_key and self.av_key != 'your_alpha_vantage_key_here':
            fetch_tasks.append(('Alpha Vantage', self.get_alpha_vantage_price))
        
        if VERIFICATION_SETTINGS['WEB_SCRAPING_BACKUP']:
            fetch_tasks.append(('Web Scraping', self.scrape_web_price))
        
        fetch_tasks.append(('Alternative API', self.get_alternative_price))
        
        # Fetch prices in parallel
        with ParallelProcessor(max_workers=3) as processor:
            results = processor.map(
                lambda task: (task[0], task[1]()),
                fetch_tasks
            )
        
        # Add successful fetches to sources
        for source_name, price in results:
            if price is not None:
                sources.append((source_name, price))
        
        verification_result['sources_compared'] = len(sources)
        
        # Compare prices
        if len(sources) > 1:
            base_price = sources[0][1]
            
            for source_name, price in sources[1:]:
                price_diff = abs(price - base_price)
                price_diff_percent = (price_diff / base_price) * 100
                
                if price_diff_percent > VERIFICATION_SETTINGS['PRICE_DISCREPANCY_THRESHOLD']:
                    verification_result['discrepancies'].append({
                        'source': source_name,
                        'price': price,
                        'difference': price_diff,
                        'difference_percent': price_diff_percent
                    })
        
        return verification_result
    
    @retry_with_backoff(RetryConfig(max_retries=2, initial_delay=1.0))
    @rate_limited(calls_per_second=5.0)  # Alpha Vantage free tier limit
    def get_alpha_vantage_price(self):
        """Get price from Alpha Vantage API with retry logic"""
        url = f"https://www.alphavantage.co/query"
        params = {
            'function': 'GLOBAL_QUOTE',
            'symbol': self.symbol,
            'apikey': self.av_key
        }
        
        response = safe_request(
            method='GET',
            url=url,
            params=params,
            timeout=10
        )
        
        if response and response.status_code == 200:
            data = response.json()
            
            # Check for rate limit message
            if 'Note' in data and 'API call frequency' in data['Note']:
                raise requests.exceptions.HTTPError("Alpha Vantage rate limit reached")
            
            if 'Global Quote' in data and '05. price' in data['Global Quote']:
                return float(data['Global Quote']['05. price'])
        
        return None
    
    @retry_with_backoff(RetryConfig(max_retries=2, initial_delay=2.0))
    @handle_api_errors(default_return=None)
    def scrape_web_price(self):
        """Scrape Apple stock price from a financial website with retry logic"""
        from bs4 import BeautifulSoup
        import re
        
        # Using Yahoo Finance web scraping as backup
        url = f"https://finance.yahoo.com/quote/{self.symbol}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = safe_request(
            method='GET',
            url=url,
            headers=headers,
            timeout=15,
            retry_config=RetryConfig(max_retries=2)
        )
        
        if response and response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Look for price in various possible locations
            price_selectors = [
                'fin-streamer[data-symbol="AAPL"][data-field="regularMarketPrice"]',
                'span[data-reactid*="regularMarketPrice"]',
                '.Trsdu\\(0\\.3s\\) .Fw\\(b\\) .Fz\\(36px\\)'
            ]
            
            for selector in price_selectors:
                try:
                    element = soup.select_one(selector)
                    if element:
                        price_text = element.get_text().strip()
                        # Extract number from text
                        price_match = re.search(r'[\d,]+\.?\d*', price_text)
                        if price_match:
                            price = float(price_match.group().replace(',', ''))
                            return price
                except Exception as e:
                    logger.debug(f"Failed to extract price with selector {selector}: {e}")
                    continue
        
        return None
    
    @retry_with_backoff(RetryConfig(max_retries=2, initial_delay=1.0))
    @rate_limited(calls_per_second=1.0)
    def get_alternative_price(self):
        """Get price from alternative free API with retry logic"""
        # Using Financial Modeling Prep (free tier)
        url = f"https://financialmodelingprep.com/api/v3/quote/{self.symbol}?apikey=demo"
        
        response = safe_request(
            method='GET',
            url=url,
            timeout=10,
            retry_config=RetryConfig(max_retries=2)
        )
        
        if response and response.status_code == 200:
            data = response.json()
            if data and isinstance(data, list) and len(data) > 0:
                return float(data[0].get('price', 0))
        
        return None
    
    def check_data_anomalies(self, stock_data):
        """Check for data anomalies in stock data"""
        anomalies = []
        
        try:
            current_price = stock_data['current_price']
            hist_data = stock_data['historical_data']
            
            # Check 1: Price spike detection
            recent_prices = hist_data['Close'].tail(5)
            avg_recent = recent_prices.mean()
            
            if abs(current_price - avg_recent) / avg_recent > 0.10:  # 10% difference
                anomalies.append({
                    'type': 'PRICE_SPIKE',
                    'description': f'Current price ${current_price:.2f} differs by {abs(current_price - avg_recent)/avg_recent*100:.1f}% from recent average ${avg_recent:.2f}',
                    'severity': 'HIGH' if abs(current_price - avg_recent) / avg_recent > 0.20 else 'MEDIUM'
                })
            
            # Check 2: Volume anomaly
            if 'volume_ratio' in stock_data and stock_data['volume_ratio'] > 5:
                anomalies.append({
                    'type': 'VOLUME_SPIKE',
                    'description': f'Volume is {stock_data["volume_ratio"]:.2f}x higher than average',
                    'severity': 'HIGH' if stock_data['volume_ratio'] > 10 else 'MEDIUM'
                })
            
            # Check 3: Missing data
            if len(hist_data) < 50:
                anomalies.append({
                    'type': 'INSUFFICIENT_DATA',
                    'description': f'Only {len(hist_data)} days of historical data available',
                    'severity': 'MEDIUM'
                })
            
            # Check 4: Zero or negative prices
            if current_price <= 0:
                anomalies.append({
                    'type': 'INVALID_PRICE',
                    'description': f'Invalid current price: ${current_price}',
                    'severity': 'HIGH'
                })
            
            # Check 5: Extreme price movements
            if len(hist_data) > 1:
                yesterday_close = hist_data['Close'].iloc[-2]
                daily_change = (current_price - yesterday_close) / yesterday_close
                
                if abs(daily_change) > 0.15:  # 15% daily change
                    anomalies.append({
                        'type': 'EXTREME_MOVEMENT',
                        'description': f'Daily price change: {daily_change*100:.1f}%',
                        'severity': 'HIGH' if abs(daily_change) > 0.25 else 'MEDIUM'
                    })
            
        except Exception as e:
            anomalies.append({
                'type': 'CHECK_ERROR',
                'description': f'Error checking anomalies: {e}',
                'severity': 'LOW'
            })
        
        return {'anomalies': anomalies}
    
    def validate_data_ranges(self, stock_data):
        """Validate that data values are within expected ranges"""
        checks = []
        
        try:
            current_price = stock_data['current_price']
            
            # Check 1: Price range validation
            if current_price < 1 or current_price > 1000:
                checks.append(f"⚠️ Price ${current_price:.2f} outside expected range ($1-$1000)")
            else:
                checks.append(f"✅ Price ${current_price:.2f} within expected range")
            
            # Check 2: Volume validation
            if 'volume_ratio' in stock_data:
                volume_ratio = stock_data['volume_ratio']
                if volume_ratio < 0.1 or volume_ratio > 20:
                    checks.append(f"⚠️ Volume ratio {volume_ratio:.2f}x outside normal range (0.1-20x)")
                else:
                    checks.append(f"✅ Volume ratio {volume_ratio:.2f}x within normal range")
            
            # Check 3: Historical data completeness
            hist_data = stock_data['historical_data']
            if len(hist_data) >= 100:
                checks.append(f"✅ Sufficient historical data ({len(hist_data)} days)")
            else:
                checks.append(f"⚠️ Limited historical data ({len(hist_data)} days)")
            
            # Check 4: Data columns validation
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in hist_data.columns]
            
            if missing_columns:
                checks.append(f"⚠️ Missing data columns: {', '.join(missing_columns)}")
            else:
                checks.append("✅ All required data columns present")
            
            # Check 5: Options data validation
            if stock_data.get('options_data'):
                options = stock_data['options_data']
                put_call_ratio = options.get('put_call_ratio', 0)
                
                if 0.3 <= put_call_ratio <= 3.0:
                    checks.append(f"✅ Put/Call ratio {put_call_ratio:.2f} within normal range")
                else:
                    checks.append(f"⚠️ Put/Call ratio {put_call_ratio:.2f} outside normal range (0.3-3.0)")
            
        except Exception as e:
            checks.append(f"❌ Data validation error: {e}")
        
        return {'checks': checks}
    
    def check_data_freshness(self, stock_data):
        """Check if data is fresh and up-to-date"""
        checks = []
        
        try:
            from datetime import datetime, timedelta
            import pytz
            
            # Get current time in US Eastern (market time)
            eastern = pytz.timezone('US/Eastern')
            current_time = datetime.now(eastern)
            
            # Check 1: Historical data freshness
            hist_data = stock_data['historical_data']
            if len(hist_data) > 0:
                last_data_date = pd.to_datetime(hist_data.index[-1]).date()
                today = current_time.date()
                
                # Check if it's a weekend or after market hours
                is_weekend = current_time.weekday() >= 5  # Saturday=5, Sunday=6
                is_after_hours = current_time.hour >= 16  # Market closes at 4 PM ET
                is_before_hours = current_time.hour < 9   # Market opens at 9:30 AM ET
                
                if last_data_date == today:
                    checks.append("✅ Historical data is current (today)")
                elif last_data_date == today - timedelta(days=1):
                    if is_weekend or is_before_hours:
                        checks.append("✅ Historical data is fresh (previous trading day)")
                    else:
                        checks.append("⚠️ Historical data is from yesterday (may be delayed)")
                else:
                    days_old = (today - last_data_date).days
                    checks.append(f"⚠️ Historical data is {days_old} days old")
            
            # Check 2: Market hours context
            if is_weekend:
                checks.append("ℹ️ Market is closed (weekend)")
            elif is_before_hours:
                checks.append("ℹ️ Pre-market hours (market opens at 9:30 AM ET)")
            elif is_after_hours:
                checks.append("ℹ️ After-market hours (market closed at 4:00 PM ET)")
            else:
                checks.append("✅ Market is currently open")
            
            # Check 3: Real-time price freshness
            # This is more complex to implement without real-time APIs
            checks.append("ℹ️ Price freshness depends on data source update frequency")
            
        except Exception as e:
            checks.append(f"❌ Freshness check error: {e}")
        
        return {'checks': checks}
    
    def calculate_data_confidence(self, price_verification, anomaly_check, range_check, freshness_check):
        """Calculate overall confidence score for data quality"""
        confidence = 100  # Start with perfect score
        
        # Deduct points for price discrepancies
        if price_verification['sources_compared'] > 1:
            confidence += 10  # Bonus for multiple sources
            
            for discrepancy in price_verification['discrepancies']:
                if discrepancy.get('difference_percent', 0) > 5:
                    confidence -= 20  # Major discrepancy
                elif discrepancy.get('difference_percent', 0) > 1:
                    confidence -= 10  # Minor discrepancy
        else:
            confidence -= 5   # Only one source
        
        # Deduct points for anomalies
        for anomaly in anomaly_check['anomalies']:
            if anomaly['severity'] == 'HIGH':
                confidence -= 25
            elif anomaly['severity'] == 'MEDIUM':
                confidence -= 15
            else:
                confidence -= 5
        
        # Deduct points for range check failures
        warning_count = sum(1 for check in range_check['checks'] if check.startswith('⚠️'))
        error_count = sum(1 for check in range_check['checks'] if check.startswith('❌'))
        
        confidence -= (warning_count * 5) + (error_count * 15)
        
        # Deduct points for freshness issues
        freshness_warnings = sum(1 for check in freshness_check['checks'] if check.startswith('⚠️'))
        confidence -= freshness_warnings * 10
        
        return max(0, min(100, confidence))
    
    def display_verification_results(self, verification_results):
        """Display data verification results"""
        print("\n" + "="*60)
        print("🔍 DATA VERIFICATION RESULTS")
        print("="*60)
        
        status = verification_results['status']
        confidence = verification_results['confidence_score']
        
        # Status indicator
        if status == 'EXCELLENT':
            status_icon = "🟢"
        elif status == 'GOOD':
            status_icon = "🟡"
        elif status == 'FAIR':
            status_icon = "🟠"
        else:
            status_icon = "🔴"
        
        print(f"{status_icon} DATA QUALITY: {status}")
        print(f"📊 CONFIDENCE SCORE: {confidence:.1f}/100")
        print(f"📈 SOURCES COMPARED: {verification_results['sources_compared']}")
        
        # Price verification
        if verification_results['price_discrepancies']:
            print(f"\n⚠️ PRICE DISCREPANCIES FOUND:")
            for discrepancy in verification_results['price_discrepancies']:
                if 'source' in discrepancy:
                    print(f"  • {discrepancy['source']}: ${discrepancy['price']:.2f} "
                          f"({discrepancy['difference_percent']:.1f}% difference)")
                else:
                    print(f"  • {discrepancy.get('error', 'Unknown error')}")
        else:
            print(f"\n✅ No significant price discrepancies found")
        
        # Data quality checks
        if verification_results['data_quality_checks']:
            print(f"\n📋 DATA QUALITY CHECKS:")
            for check in verification_results['data_quality_checks']:
                print(f"  {check}")
        
        # Anomalies
        if verification_results['anomalies_detected']:
            print(f"\n🚨 ANOMALIES DETECTED:")
            for anomaly in verification_results['anomalies_detected']:
                severity_icon = "🔴" if anomaly['severity'] == 'HIGH' else "🟡" if anomaly['severity'] == 'MEDIUM' else "🟢"
                print(f"  {severity_icon} {anomaly['type']}: {anomaly['description']}")
        else:
            print(f"\n✅ No significant anomalies detected")
        
        # Recommendation
        print(f"\n💡 RECOMMENDATION:")
        if confidence >= 90:
            print("  ✅ Data quality is excellent - proceed with confidence")
        elif confidence >= 75:
            print("  ✅ Data quality is good - safe to proceed")
        elif confidence >= 50:
            print("  ⚠️ Data quality is fair - proceed with caution")
        else:
            print("  ❌ Data quality is poor - manual verification recommended")
        
        print(f"  📊 Signal reliability: {confidence:.1f}%")
    
    def display_fetched_data(self, stock_data, news_data, social_data, fundamental_data):
        """Display all fetched data in organized format"""
        print("\n" + "="*60)
        print("🍎 APPLE STOCK DATA SUMMARY")
        print("="*60)
        
        # Stock Data
        if stock_data:
            print(f"\n📈 CURRENT STOCK DATA:")
            print(f"Current Price: ${stock_data['current_price']:.2f}")
            print(f"Volume Ratio: {stock_data['volume_ratio']:.2f}x (vs 20-day avg)")
            
            if stock_data['options_data']:
                print(f"Put/Call Ratio: {stock_data['options_data']['put_call_ratio']:.2f}")
                print(f"Total Call Volume: {stock_data['options_data']['total_call_volume']:,}")
                print(f"Total Put Volume: {stock_data['options_data']['total_put_volume']:,}")
        
        # News Data
        print(f"\n📰 NEWS SENTIMENT DATA:")
        print(f"Average Sentiment: {news_data['avg_sentiment']:.1f}/100")
        print(f"Total Articles: {news_data['total_articles']}")
        print(f"Positive Articles: {news_data['positive_count']}")
        print(f"Negative Articles: {news_data['negative_count']}")
        print(f"Neutral Articles: {news_data['neutral_count']}")
        
        # Show recent headlines
        if news_data['articles']:
            print(f"\n📰 RECENT HEADLINES:")
            for i, article in enumerate(news_data['articles'][:5]):
                sentiment_icon = "🟢" if article['sentiment_score'] > 60 else "🔴" if article['sentiment_score'] < 40 else "🟡"
                print(f"{sentiment_icon} {article['title'][:80]}...")
                print(f"   Sentiment: {article['sentiment_score']:.1f}/100 ({article['sentiment_label']})")
        
        # Social Data
        print(f"\n💬 SOCIAL MEDIA SENTIMENT:")
        if social_data['reddit_data']['post_count'] > 0:
            print(f"Reddit Sentiment: {social_data['reddit_data']['avg_sentiment']:.1f}/100")
            print(f"Reddit Posts Analyzed: {social_data['reddit_data']['post_count']}")
            print(f"Bullish Posts: {social_data['reddit_data']['bullish_posts']}")
            print(f"Bearish Posts: {social_data['reddit_data']['bearish_posts']}")
        else:
            print("Reddit Sentiment: No data available")
        
        # Fundamental Data
        print(f"\n📊 FUNDAMENTAL DATA:")
        if fundamental_data:
            print(f"P/E Ratio: {fundamental_data.get('pe_ratio', 'N/A')}")
            print(f"Forward P/E: {fundamental_data.get('forward_pe', 'N/A')}")
            print(f"PEG Ratio: {fundamental_data.get('peg_ratio', 'N/A')}")
            print(f"Revenue Growth: {fundamental_data.get('revenue_growth', 0)*100:.1f}%")
            print(f"Earnings Growth: {fundamental_data.get('earnings_growth', 0)*100:.1f}%")
            print(f"Profit Margin: {fundamental_data.get('profit_margin', 0)*100:.1f}%")
            print(f"Beta: {fundamental_data.get('beta', 'N/A')}")
            if fundamental_data.get('market_cap', 0) > 0:
                print(f"Market Cap: ${fundamental_data.get('market_cap', 0)/1e12:.2f}T")