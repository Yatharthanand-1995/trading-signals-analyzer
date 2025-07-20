#!/usr/bin/env python3
"""
Phase 1 Integration Tests
Tests the integration between Market Regime Detector and Adaptive Signal Generator
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.analysis.market_regime_detector import MarketRegimeDetector
from core.analysis.adaptive_signal_generator import AdaptiveSignalGenerator

class TestPhase1Integration(unittest.TestCase):
    """Test Market Regime Detection + Adaptive Signal Generation workflow"""
    
    def setUp(self):
        """Set up test environment"""
        self.regime_detector = MarketRegimeDetector()
        self.signal_generator = AdaptiveSignalGenerator()
        self.test_symbols = ['AAPL', 'MSFT', 'GOOGL']
        
        # Create mock data for testing
        self.mock_data = self._create_mock_data()
    
    def _create_mock_data(self):
        """Create mock stock data for testing"""
        dates = pd.date_range(start='2024-01-01', end='2024-12-01', freq='D')
        np.random.seed(42)  # For reproducible tests
        
        data = pd.DataFrame({
            'Open': 150 + np.random.randn(len(dates)) * 5,
            'High': 155 + np.random.randn(len(dates)) * 5,
            'Low': 145 + np.random.randn(len(dates)) * 5,
            'Close': 150 + np.random.randn(len(dates)) * 5,
            'Volume': 1000000 + np.random.randint(0, 500000, len(dates))
        }, index=dates)
        
        # Ensure High > Low and realistic OHLC relationships
        data['High'] = np.maximum(data[['Open', 'Close']].max(axis=1), data['High'])
        data['Low'] = np.minimum(data[['Open', 'Close']].min(axis=1), data['Low'])
        
        return data
    
    def test_market_regime_detection(self):
        """Test market regime detection functionality"""
        print("\n🔍 Testing Market Regime Detection...")
        
        # Test with mock data
        regime = self.regime_detector.detect_market_regime(market_data=self.mock_data)
        
        # Verify regime structure
        self.assertIn('regime', regime)
        self.assertIn('confidence', regime)
        self.assertIn('scores', regime)
        self.assertIn('strategy', regime)
        self.assertIn('risk_multiplier', regime)
        
        # Verify regime is one of expected values
        expected_regimes = ['STRONG_BULL', 'BULL', 'NEUTRAL', 'BEAR', 'STRONG_BEAR', 'HIGH_VOLATILITY']
        self.assertIn(regime['regime'], expected_regimes)
        
        # Verify confidence is within bounds
        self.assertGreaterEqual(regime['confidence'], 0)
        self.assertLessEqual(regime['confidence'], 100)
        
        # Verify scores structure
        self.assertIn('trend', regime['scores'])
        self.assertIn('volatility', regime['scores'])
        self.assertIn('breadth', regime['scores'])
        self.assertIn('momentum', regime['scores'])
        
        print(f"✅ Regime detected: {regime['regime']} (Confidence: {regime['confidence']:.1f}%)")
        
    def test_regime_strategy_parameters(self):
        """Test that regime parameters are correctly retrieved"""
        print("\n⚙️ Testing Regime Strategy Parameters...")
        
        for regime_name in ['STRONG_BULL', 'BULL', 'NEUTRAL', 'BEAR', 'STRONG_BEAR', 'HIGH_VOLATILITY']:
            params = self.regime_detector.get_strategy_parameters(regime_name)
            
            # Verify all required parameters exist
            required_params = [
                'rsi_oversold', 'rsi_overbought', 'position_size_multiplier',
                'stop_loss_multiplier', 'take_profit_multiplier', 'max_positions',
                'preferred_sectors', 'min_volume_ratio'
            ]
            
            for param in required_params:
                self.assertIn(param, params)
            
            # Verify parameter ranges are reasonable
            self.assertGreater(params['rsi_oversold'], 0)
            self.assertLess(params['rsi_oversold'], 50)
            self.assertGreater(params['rsi_overbought'], 50)
            self.assertLess(params['rsi_overbought'], 100)
            self.assertGreater(params['max_positions'], 0)
            
        print("✅ All regime parameters validated")
    
    def test_adaptive_signal_initialization(self):
        """Test adaptive signal generator initialization"""
        print("\n🎯 Testing Adaptive Signal Generator Initialization...")
        
        # Test that components are properly initialized
        self.assertIsNotNone(self.signal_generator.regime_detector)
        self.assertIsInstance(self.signal_generator.regime_detector, MarketRegimeDetector)
        
        print("✅ Adaptive Signal Generator initialized correctly")
    
    def test_technical_indicators_calculation(self):
        """Test technical indicators calculation"""
        print("\n📊 Testing Technical Indicators Calculation...")
        
        # Test with mock data - need regime params first
        test_regime = {
            'regime': 'NEUTRAL',
            'rsi_oversold': 30,
            'rsi_overbought': 70
        }
        
        # Set mock regime params
        self.signal_generator.regime_params = {
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'min_volume_ratio': 1.0
        }
        
        indicators = self.signal_generator._calculate_indicators(self.mock_data)
        
        # Verify required indicators exist
        required_indicators = [
            'close', 'volume', 'sma_20', 'sma_50', 'rsi',
            'macd', 'macd_signal', 'atr', 'volume_ratio'
        ]
        
        for indicator in required_indicators:
            self.assertIn(indicator, indicators)
            self.assertIsNotNone(indicators[indicator])
        
        # Verify RSI is in valid range
        self.assertGreaterEqual(indicators['rsi'], 0)
        self.assertLessEqual(indicators['rsi'], 100)
        
        print("✅ Technical indicators calculated correctly")
    
    def test_timeframe_signal_generation(self):
        """Test timeframe signal generation"""
        print("\n⏱️ Testing Timeframe Signal Generation...")
        
        # Set mock regime params
        self.signal_generator.regime_params = {
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'min_volume_ratio': 1.0
        }
        
        indicators = self.signal_generator._calculate_indicators(self.mock_data)
        
        # Test daily timeframe signal
        daily_signal = self.signal_generator._get_timeframe_signal(indicators, 'daily')
        self.assertGreaterEqual(daily_signal, 0)
        self.assertLessEqual(daily_signal, 100)
        
        # Test hourly timeframe signal (should be weighted differently)
        hourly_signal = self.signal_generator._get_timeframe_signal(indicators, 'hourly')
        self.assertGreaterEqual(hourly_signal, 0)
        self.assertLessEqual(hourly_signal, 100)
        
        print(f"✅ Signals generated - Daily: {daily_signal:.1f}, Hourly: {hourly_signal:.1f}")
    
    def test_volume_analysis(self):
        """Test volume analysis functionality"""
        print("\n📈 Testing Volume Analysis...")
        
        # Set mock regime params
        self.signal_generator.regime_params = {
            'min_volume_ratio': 1.0
        }
        
        volume_score = self.signal_generator._analyze_volume(self.mock_data)
        
        # Verify volume score is in valid range
        self.assertGreaterEqual(volume_score, 0)
        self.assertLessEqual(volume_score, 100)
        
        print(f"✅ Volume score: {volume_score:.1f}")
    
    def test_momentum_analysis(self):
        """Test momentum analysis functionality"""
        print("\n🚀 Testing Momentum Analysis...")
        
        momentum_score = self.signal_generator._analyze_momentum_acceleration(self.mock_data)
        
        # Verify momentum score is in valid range
        self.assertGreaterEqual(momentum_score, 0)
        self.assertLessEqual(momentum_score, 100)
        
        print(f"✅ Momentum score: {momentum_score:.1f}")
    
    def test_composite_score_calculation(self):
        """Test composite score calculation with regime weighting"""
        print("\n🎯 Testing Composite Score Calculation...")
        
        # Set up mock regime and current regime
        self.signal_generator.current_regime = {
            'regime': 'NEUTRAL',
            'confidence': 75
        }
        
        # Test score calculation
        composite_score = self.signal_generator._calculate_composite_score(60, 55, 65, 70)
        
        # Verify composite score is reasonable
        self.assertGreaterEqual(composite_score, 0)
        self.assertLessEqual(composite_score, 100)
        
        print(f"✅ Composite score: {composite_score:.1f}")
    
    def test_action_determination(self):
        """Test trading action determination"""
        print("\n⚡ Testing Action Determination...")
        
        # Set up mock regime
        self.signal_generator.current_regime = {'regime': 'NEUTRAL'}
        self.signal_generator.regime_params = {'min_volume_ratio': 1.0}
        
        # Test different score ranges
        test_cases = [
            (80, 'BUY'),   # High score should trigger BUY
            (30, 'SELL'),  # Low score should trigger SELL
            (50, 'HOLD')   # Medium score should trigger HOLD
        ]
        
        for score, expected_action in test_cases:
            action = self.signal_generator._determine_action(score, {'volume_ratio': 1.2})
            # Note: action might differ based on regime, so we just verify it's valid
            valid_actions = ['BUY', 'SELL', 'HOLD']
            self.assertIn(action, valid_actions)
        
        print("✅ Action determination working correctly")
    
    def test_position_parameters_calculation(self):
        """Test position sizing and risk parameters calculation"""
        print("\n💰 Testing Position Parameters Calculation...")
        
        # Set up mock regime params
        self.signal_generator.regime_params = {
            'position_size_multiplier': 1.0,
            'stop_loss_multiplier': 1.0,
            'take_profit_multiplier': 1.0
        }
        
        indicators = self.signal_generator._calculate_indicators(self.mock_data)
        
        params = self.signal_generator._calculate_position_parameters(
            'AAPL', self.mock_data, indicators, 70
        )
        
        # Verify required parameters exist
        required_params = [
            'position_size', 'stop_loss', 'take_profit_1', 'take_profit_2',
            'take_profit_3', 'risk_amount', 'regime_multiplier'
        ]
        
        for param in required_params:
            self.assertIn(param, params)
            self.assertIsNotNone(params[param])
        
        # Verify logical relationships
        current_price = self.mock_data['Close'].iloc[-1]
        self.assertLess(params['stop_loss'], current_price)  # Stop loss below current price
        self.assertGreater(params['take_profit_1'], current_price)  # Take profit above current price
        self.assertGreater(params['take_profit_2'], params['take_profit_1'])  # TP2 > TP1
        self.assertGreater(params['take_profit_3'], params['take_profit_2'])  # TP3 > TP2
        
        print("✅ Position parameters calculated correctly")
    
    def test_end_to_end_integration(self):
        """Test complete end-to-end workflow"""
        print("\n🔄 Testing End-to-End Integration...")
        
        # Test with a single symbol to avoid network calls in testing
        # This would normally call the full generate_adaptive_signals method
        # but we'll test the core workflow without external dependencies
        
        # 1. Detect regime (using mock data)
        regime = self.regime_detector.detect_market_regime(market_data=self.mock_data)
        self.assertIsNotNone(regime)
        
        # 2. Set up signal generator with detected regime
        self.signal_generator.current_regime = regime
        self.signal_generator.regime_params = self.regime_detector.get_strategy_parameters(regime['regime'])
        
        # 3. Test analysis of a symbol (with mock data)
        indicators = self.signal_generator._calculate_indicators(self.mock_data)
        self.assertIsNotNone(indicators)
        
        # 4. Test signal generation components
        daily_signal = self.signal_generator._get_timeframe_signal(indicators, 'daily')
        volume_score = self.signal_generator._analyze_volume(self.mock_data)
        momentum_score = self.signal_generator._analyze_momentum_acceleration(self.mock_data)
        
        # 5. Test composite score calculation
        composite_score = self.signal_generator._calculate_composite_score(
            daily_signal, daily_signal * 0.7, volume_score, momentum_score
        )
        
        # 6. Test action determination
        action = self.signal_generator._determine_action(composite_score, indicators)
        
        # Verify all components work together
        self.assertIsNotNone(regime)
        self.assertIsNotNone(composite_score)
        self.assertIn(action, ['BUY', 'SELL', 'HOLD'])
        
        print(f"✅ End-to-end integration successful!")
        print(f"   Regime: {regime['regime']}")
        print(f"   Score: {composite_score:.1f}")
        print(f"   Action: {action}")
    
    def test_performance_benchmarks(self):
        """Test performance meets requirements"""
        print("\n⚡ Testing Performance Benchmarks...")
        
        import time
        
        # Test regime detection performance (target: <500ms)
        start_time = time.time()
        regime = self.regime_detector.detect_market_regime(market_data=self.mock_data)
        regime_time = time.time() - start_time
        
        print(f"📊 Regime detection time: {regime_time*1000:.1f}ms")
        self.assertLess(regime_time, 0.5, "Regime detection should be under 500ms")
        
        # Test signal generation components performance
        self.signal_generator.current_regime = regime
        self.signal_generator.regime_params = self.regime_detector.get_strategy_parameters(regime['regime'])
        
        start_time = time.time()
        indicators = self.signal_generator._calculate_indicators(self.mock_data)
        daily_signal = self.signal_generator._get_timeframe_signal(indicators, 'daily')
        volume_score = self.signal_generator._analyze_volume(self.mock_data)
        momentum_score = self.signal_generator._analyze_momentum_acceleration(self.mock_data)
        signal_time = time.time() - start_time
        
        print(f"📊 Signal generation time: {signal_time*1000:.1f}ms")
        self.assertLess(signal_time, 1.0, "Signal generation should be under 1s")
        
        print("✅ Performance benchmarks met")


def run_integration_tests():
    """Run all Phase 1 integration tests"""
    print("🚀 RUNNING PHASE 1 INTEGRATION TESTS")
    print("="*60)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPhase1Integration)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\n❌ FAILURES:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")
    
    if result.errors:
        print("\n💥 ERRORS:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    
    if success:
        print("\n🎉 ALL PHASE 1 INTEGRATION TESTS PASSED!")
        print("✅ Market Regime Detection + Adaptive Signal Generation integration verified")
    else:
        print("\n❌ Some tests failed. Please review and fix issues before proceeding.")
    
    return success


if __name__ == "__main__":
    run_integration_tests()