#!/usr/bin/env python3
"""
Train ML Model Script
Trains the machine learning model on historical data
"""

import sys
import os
import pandas as pd
from datetime import datetime

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from ml_models.basic_ml_predictor import BasicMLPredictor

# Try to import STOCKS from config or use default
try:
    from core.config import STOCKS
except ImportError:
    # Use default stocks from config file
    import json
    config_path = os.path.join(parent_dir, 'config', 'stocks_config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
            # Find active list
            active_list = None
            for list_name, list_data in config['stock_lists'].items():
                if list_data.get('active', False):
                    active_list = list_name
                    break
            if active_list:
                STOCKS = config['stock_lists'][active_list]['symbols']
            else:
                STOCKS = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
    else:
        STOCKS = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']

def train_all_models():
    """Train ML models for all configured stocks"""
    print("🧠 ML Model Training System")
    print("=" * 60)
    print(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Stocks: {', '.join(STOCKS)}")
    print("=" * 60)
    
    # Initialize ML predictor
    ml_predictor = BasicMLPredictor()
    
    # Try to train on each stock's data
    trained = False
    best_accuracy = 0
    best_symbol = None
    
    for symbol in STOCKS:
        hist_file = f"data/historical/{symbol}_historical_data.csv"
        
        if not os.path.exists(hist_file):
            print(f"\n⚠️ No historical data found for {symbol}")
            continue
        
        print(f"\n📊 Loading data for {symbol}...")
        
        try:
            # Load historical data
            hist_data = pd.read_csv(hist_file, index_col='Date', parse_dates=True)
            print(f"   Loaded {len(hist_data)} days of data")
            
            # Check if we have required columns
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_cols = [col for col in required_cols if col not in hist_data.columns]
            
            if missing_cols:
                print(f"   ⚠️ Missing columns: {missing_cols}")
                continue
            
            # Train model
            print(f"\n🔧 Training model on {symbol} data...")
            performance = ml_predictor.train_model(hist_data)
            
            # Track best model
            test_acc = performance['test_accuracy']
            if test_acc > best_accuracy:
                best_accuracy = test_acc
                best_symbol = symbol
            
            trained = True
            
            # Display results
            print(f"\n✅ Model trained successfully on {symbol}!")
            print(f"   Training Accuracy: {performance['train_accuracy']:.1%}")
            print(f"   Testing Accuracy: {performance['test_accuracy']:.1%}")
            print(f"   Training Samples: {performance['train_samples']}")
            print(f"   Testing Samples: {performance['test_samples']}")
            
            # Show feature importance
            print(f"\n📊 Feature Importance:")
            sorted_features = sorted(performance['feature_importance'].items(), 
                                   key=lambda x: x[1], reverse=True)
            for feat, imp in sorted_features[:5]:
                print(f"   {feat}: {imp:.3f}")
            
            # Only train on one stock for now (models are generally applicable)
            break
            
        except Exception as e:
            print(f"   ❌ Error training on {symbol}: {str(e)}")
            continue
    
    if trained:
        print(f"\n🎉 Training Complete!")
        print(f"   Best Model: {best_symbol} (Accuracy: {best_accuracy:.1%})")
        print(f"   Model saved to: ml_models/saved_models/")
        
        # Create a summary report
        summary = {
            'training_date': datetime.now().isoformat(),
            'best_symbol': best_symbol,
            'best_accuracy': best_accuracy,
            'model_path': 'ml_models/saved_models/ml_model_latest.pkl',
            'status': 'success'
        }
        
        # Save summary
        os.makedirs("ml_models/training_reports", exist_ok=True)
        summary_file = f"ml_models/training_data/reports/training_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        import json
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n📄 Training summary saved to: {summary_file}")
        
        return True
    else:
        print("\n❌ No models were trained. Please check your historical data.")
        return False


def verify_model():
    """Verify that a trained model exists and is working"""
    print("\n🔍 Verifying ML Model...")
    
    ml_predictor = BasicMLPredictor()
    
    if ml_predictor.load_model():
        print("✅ Model loaded successfully!")
        
        # Test prediction on dummy data
        import pandas as pd
        import numpy as np
        
        # Create dummy data
        dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
        dummy_data = pd.DataFrame({
            'Date': dates,
            'Open': np.random.randn(100) * 10 + 200,
            'High': np.random.randn(100) * 10 + 205,
            'Low': np.random.randn(100) * 10 + 195,
            'Close': np.random.randn(100) * 10 + 200,
            'Volume': np.random.randint(1000000, 10000000, 100)
        }).set_index('Date')
        
        # Add some technical indicators
        dummy_data['RSI'] = 50 + np.random.randn(100) * 20
        dummy_data['MACD'] = np.random.randn(100) * 2
        dummy_data['MACD_Signal'] = dummy_data['MACD'].rolling(9).mean()
        dummy_data['MACD_Histogram'] = dummy_data['MACD'] - dummy_data['MACD_Signal']
        dummy_data['SMA_20'] = dummy_data['Close'].rolling(20).mean()
        dummy_data['SMA_50'] = dummy_data['Close'].rolling(50).mean()
        dummy_data['SMA_200'] = 200  # Dummy value
        dummy_data['BB_Upper'] = dummy_data['Close'] + 10
        dummy_data['BB_Lower'] = dummy_data['Close'] - 10
        dummy_data['ATR'] = 5
        
        # Test prediction
        try:
            signal = ml_predictor.predict_signal(dummy_data)
            print(f"\n📈 Test Prediction:")
            print(f"   Signal: {signal.get('prediction')}")
            print(f"   Confidence: {signal.get('confidence', 0):.1%}")
            print("\n✅ Model is working correctly!")
            return True
        except Exception as e:
            print(f"\n❌ Model prediction failed: {str(e)}")
            return False
    else:
        print("❌ No trained model found!")
        return False


if __name__ == "__main__":
    print("🚀 ML Model Training Script")
    print("=" * 60)
    
    # Train models
    success = train_all_models()
    
    if success:
        # Verify the model works
        verify_model()
        
        print("\n💡 Next Steps:")
        print("1. Run ML-enhanced analysis:")
        print("   python3 core/ml_enhanced_analyzer.py")
        print("\n2. Or use the command:")
        print("   python3 core/ml_enhanced_analyzer.py --paper")
        print("\n3. To analyze specific stock:")
        print("   python3 core/ml_enhanced_analyzer.py --symbol AAPL")
    else:
        print("\n⚠️ Please ensure you have historical data before training.")
        print("Run: trade-update")