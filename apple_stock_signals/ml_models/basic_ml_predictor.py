#!/usr/bin/env python3
"""
Basic ML Integration for Signal Validation
Provides quick accuracy boost through machine learning
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class BasicMLPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importance = {}
        self.model_path = "ml_models/saved_models"
        self.performance_path = "ml_models/performance"
        os.makedirs(self.model_path, exist_ok=True)
        os.makedirs(self.performance_path, exist_ok=True)
        
        # Features to use for prediction
        self.features = [
            'rsi', 'macd', 'macd_signal', 'macd_histogram',
            'sma_20_ratio', 'sma_50_ratio', 'sma_200_ratio',
            'volume_ratio', 'price_change_1d', 'price_change_5d',
            'bb_position', 'atr_ratio', 'obv_trend',
            'stoch_k', 'stoch_d'
        ]
        
    def prepare_features(self, data):
        """Prepare features from raw market data"""
        features_df = pd.DataFrame()
        
        # Price-based features
        features_df['price_change_1d'] = data['Close'].pct_change(1)
        features_df['price_change_5d'] = data['Close'].pct_change(5)
        
        # Technical indicators (assuming they exist in data)
        if 'RSI' in data.columns:
            features_df['rsi'] = data['RSI'] / 100  # Normalize to 0-1
        
        if 'MACD' in data.columns:
            features_df['macd'] = data['MACD']
            features_df['macd_signal'] = data['MACD_Signal']
            features_df['macd_histogram'] = data['MACD_Histogram']
        
        # Moving average ratios
        if 'SMA_20' in data.columns:
            features_df['sma_20_ratio'] = data['Close'] / data['SMA_20']
            features_df['sma_50_ratio'] = data['Close'] / data['SMA_50']
            features_df['sma_200_ratio'] = data['Close'] / data['SMA_200']
        
        # Volume features
        features_df['volume_ratio'] = data['Volume'] / data['Volume'].rolling(20).mean()
        
        # Bollinger Band position (0 = lower band, 1 = upper band)
        if 'BB_Upper' in data.columns:
            bb_width = data['BB_Upper'] - data['BB_Lower']
            features_df['bb_position'] = (data['Close'] - data['BB_Lower']) / bb_width
        
        # ATR ratio for volatility
        if 'ATR' in data.columns:
            features_df['atr_ratio'] = data['ATR'] / data['Close']
        
        # OBV trend
        obv = (np.sign(data['Close'].diff()) * data['Volume']).cumsum()
        features_df['obv_trend'] = obv.pct_change(20)
        
        # Stochastic
        low_14 = data['Low'].rolling(14).min()
        high_14 = data['High'].rolling(14).max()
        features_df['stoch_k'] = ((data['Close'] - low_14) / (high_14 - low_14)) * 100
        features_df['stoch_d'] = features_df['stoch_k'].rolling(3).mean()
        
        # Fill missing values
        features_df = features_df.fillna(method='ffill').fillna(0)
        
        return features_df
    
    def create_labels(self, data, look_forward=5, profit_threshold=0.02):
        """Create labels for training (1 = profitable trade, 0 = not profitable)"""
        # Calculate future returns
        future_returns = data['Close'].shift(-look_forward) / data['Close'] - 1
        
        # Label as 1 if future return > threshold
        labels = (future_returns > profit_threshold).astype(int)
        
        return labels
    
    def train_model(self, historical_data):
        """Train the ML model on historical data"""
        print("\n🧠 Training ML Model...")
        
        # Prepare features and labels
        features_df = self.prepare_features(historical_data)
        labels = self.create_labels(historical_data)
        
        # Remove last rows where we don't have labels
        features_df = features_df[:-5]
        labels = labels[:-5]
        
        # Remove rows with NaN
        valid_idx = ~(features_df.isna().any(axis=1) | labels.isna())
        features_df = features_df[valid_idx]
        labels = labels[valid_idx]
        
        # Select only the features we want
        X = features_df[self.features]
        y = labels
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=20,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_pred = self.model.predict(X_train_scaled)
        test_pred = self.model.predict(X_test_scaled)
        
        train_accuracy = accuracy_score(y_train, train_pred)
        test_accuracy = accuracy_score(y_test, test_pred)
        
        # Feature importance
        self.feature_importance = dict(zip(self.features, self.model.feature_importances_))
        
        # Save performance report
        performance = {
            'training_date': datetime.now().isoformat(),
            'train_accuracy': float(train_accuracy),
            'test_accuracy': float(test_accuracy),
            'train_samples': len(y_train),
            'test_samples': len(y_test),
            'feature_importance': self.feature_importance,
            'classification_report': classification_report(y_test, test_pred, output_dict=True)
        }
        
        # Save model and scaler
        self.save_model()
        
        # Save performance report
        perf_file = f"{self.performance_path}/training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(perf_file, 'w') as f:
            json.dump(performance, f, indent=2)
        
        print(f"✅ Model trained successfully!")
        print(f"   Train Accuracy: {train_accuracy:.2%}")
        print(f"   Test Accuracy: {test_accuracy:.2%}")
        print(f"\n📊 Top 5 Important Features:")
        for feat, imp in sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"   - {feat}: {imp:.3f}")
        
        return performance
    
    def predict_signal(self, current_data, traditional_signal=None):
        """Predict trading signal using ML model"""
        if self.model is None:
            self.load_model()
            if self.model is None:
                print("⚠️ No trained model found. Using traditional signal only.")
                return traditional_signal
        
        # Prepare features
        features_df = self.prepare_features(current_data)
        
        # Get latest features
        if len(features_df) == 0:
            return traditional_signal
        
        latest_features = features_df[self.features].iloc[-1:].fillna(0)
        
        # Scale features
        features_scaled = self.scaler.transform(latest_features)
        
        # Get prediction and probability
        prediction = self.model.predict(features_scaled)[0]
        probability = self.model.predict_proba(features_scaled)[0]
        
        # ML signal
        ml_signal = {
            'prediction': 'BUY' if prediction == 1 else 'HOLD',
            'confidence': float(max(probability)),
            'buy_probability': float(probability[1]),
            'sell_probability': float(probability[0])
        }
        
        # Combine with traditional signal if provided
        if traditional_signal:
            combined_signal = self._combine_signals(traditional_signal, ml_signal)
        else:
            combined_signal = ml_signal
        
        # Save prediction
        self._save_prediction(current_data.index[-1], latest_features.iloc[0].to_dict(), 
                            ml_signal, traditional_signal, combined_signal)
        
        return combined_signal
    
    def _combine_signals(self, traditional, ml):
        """Intelligently combine traditional and ML signals"""
        # If ML is highly confident (>80%) and signals match, boost confidence
        if ml['confidence'] > 0.8 and traditional.get('signal') == ml['prediction']:
            return {
                'signal': ml['prediction'],
                'confidence': (traditional.get('confidence', 50) + ml['confidence'] * 100) / 2,
                'ml_confidence': ml['confidence'],
                'traditional_score': traditional.get('score', 50),
                'source': 'combined_strong_agreement'
            }
        
        # If ML is confident but disagrees, use weighted average
        elif ml['confidence'] > 0.7:
            ml_weight = 0.6
            trad_weight = 0.4
            
            # Convert traditional signal to score
            trad_score = traditional.get('score', 50) / 100
            
            combined_score = (ml['buy_probability'] * ml_weight + trad_score * trad_weight)
            
            return {
                'signal': 'BUY' if combined_score > 0.6 else 'HOLD',
                'confidence': combined_score * 100,
                'ml_confidence': ml['confidence'],
                'traditional_score': traditional.get('score', 50),
                'source': 'weighted_combination'
            }
        
        # If ML has low confidence, trust traditional more
        else:
            return {
                'signal': traditional.get('signal', 'HOLD'),
                'confidence': traditional.get('confidence', 50) * 0.9,  # Slight reduction
                'ml_confidence': ml['confidence'],
                'traditional_score': traditional.get('score', 50),
                'source': 'traditional_primary'
            }
    
    def save_model(self):
        """Save trained model and scaler"""
        model_file = f"{self.model_path}/ml_model_{datetime.now().strftime('%Y%m%d')}.pkl"
        scaler_file = f"{self.model_path}/scaler_{datetime.now().strftime('%Y%m%d')}.pkl"
        
        joblib.dump(self.model, model_file)
        joblib.dump(self.scaler, scaler_file)
        
        # Also save as 'latest' for easy loading
        joblib.dump(self.model, f"{self.model_path}/ml_model_latest.pkl")
        joblib.dump(self.scaler, f"{self.model_path}/scaler_latest.pkl")
        
        print(f"💾 Model saved to {model_file}")
    
    def load_model(self):
        """Load the latest trained model"""
        try:
            self.model = joblib.load(f"{self.model_path}/ml_model_latest.pkl")
            self.scaler = joblib.load(f"{self.model_path}/scaler_latest.pkl")
            print("✅ ML model loaded successfully")
            return True
        except:
            print("⚠️ No saved model found")
            return False
    
    def _save_prediction(self, date, features, ml_signal, traditional_signal, combined_signal):
        """Save prediction details for analysis"""
        prediction_dir = "ml_models/predictions"
        os.makedirs(prediction_dir, exist_ok=True)
        
        prediction_file = f"{prediction_dir}/predictions_{datetime.now().strftime('%Y%m%d')}.json"
        
        # Load existing predictions
        if os.path.exists(prediction_file):
            with open(prediction_file, 'r') as f:
                predictions = json.load(f)
        else:
            predictions = []
        
        # Add new prediction
        predictions.append({
            'timestamp': datetime.now().isoformat(),
            'date': str(date),
            'features': features,
            'ml_signal': ml_signal,
            'traditional_signal': traditional_signal,
            'combined_signal': combined_signal
        })
        
        # Save
        with open(prediction_file, 'w') as f:
            json.dump(predictions, f, indent=2)


def train_ml_model(symbol='AAPL'):
    """Standalone function to train ML model"""
    print(f"\n🚀 Training ML model for {symbol}...")
    
    # Load historical data
    data_file = f"historical_data/{symbol}_historical_data.csv"
    if not os.path.exists(data_file):
        print(f"❌ Historical data not found for {symbol}")
        return
    
    # Load data
    historical_data = pd.read_csv(data_file, index_col='Date', parse_dates=True)
    
    # Initialize and train model
    ml_predictor = BasicMLPredictor()
    performance = ml_predictor.train_model(historical_data)
    
    return ml_predictor, performance


if __name__ == "__main__":
    # Train model when run directly
    train_ml_model('AAPL')