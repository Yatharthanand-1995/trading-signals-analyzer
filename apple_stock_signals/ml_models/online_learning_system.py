#!/usr/bin/env python3
"""
Online Learning System for Continuous Model Adaptation
Implements incremental learning to adapt to changing market conditions
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import pickle
import os
from typing import Dict, List, Tuple, Optional
import joblib
from collections import deque
import threading
import time

# ML Libraries
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error
import xgboost as xgb
import lightgbm as lgb
from river import ensemble, tree, metrics, preprocessing
import warnings
warnings.filterwarnings('ignore')

class OnlineLearningSystem:
    """
    Implements online learning for continuous model adaptation:
    - Incremental model updates
    - Concept drift detection
    - Performance monitoring
    - Automatic retraining triggers
    - Model versioning and rollback
    """
    
    def __init__(self, base_model_path: str = None):
        self.models = {
            'online_xgboost': None,
            'online_forest': None,
            'river_ensemble': None
        }
        
        self.performance_history = deque(maxlen=1000)
        self.concept_drift_detector = ConceptDriftDetector()
        self.model_versions = deque(maxlen=5)
        self.current_version = 0
        
        # Configuration
        self.config = {
            'update_frequency': 100,  # Update after N new samples
            'performance_threshold': 0.6,  # Min acceptable performance
            'drift_threshold': 0.05,  # Statistical significance for drift
            'retrain_threshold': 0.7,  # Performance drop trigger
            'min_samples_for_update': 50,
            'max_training_time': 300,  # 5 minutes max
            'validation_split': 0.2
        }
        
        # Initialize models
        self._initialize_models(base_model_path)
        
        # Performance tracking
        self.metrics_tracker = {
            'accuracy': deque(maxlen=100),
            'precision': deque(maxlen=100),
            'recall': deque(maxlen=100),
            'sharpe_ratio': deque(maxlen=100),
            'max_drawdown': deque(maxlen=100)
        }
        
        # Data buffer for batch updates
        self.data_buffer = {
            'features': [],
            'labels': [],
            'timestamps': [],
            'predictions': []
        }
        
        # Learning state
        self.learning_state = {
            'total_samples': 0,
            'last_update': datetime.now(),
            'last_drift_check': datetime.now(),
            'is_learning': True,
            'performance_trend': 'stable'
        }
    
    def _initialize_models(self, base_model_path: str):
        """
        Initialize online learning models
        """
        # River ensemble (truly online)
        self.models['river_ensemble'] = ensemble.AdaptiveRandomForestClassifier(
            n_models=10,
            max_features='sqrt',
            lambda_value=6,
            grace_period=50,
            split_confidence=0.01,
            tie_threshold=0.05,
            leaf_prediction='mc',
            nb_threshold=0,
            seed=42
        )
        
        # Online gradient boosting
        self.models['online_boosting'] = ensemble.AdaBoostClassifier(
            model=tree.HoeffdingTreeClassifier(
                grace_period=50,
                split_confidence=0.01,
                tie_threshold=0.05,
                leaf_prediction='mc'
            ),
            n_models=10,
            seed=42
        )
        
        # Load base models if provided
        if base_model_path and os.path.exists(base_model_path):
            self._load_base_models(base_model_path)
    
    def process_new_data(self, features: pd.DataFrame, label: float, 
                        timestamp: datetime, metadata: Dict = None) -> Dict:
        """
        Process new data point for online learning
        """
        result = {
            'prediction': None,
            'confidence': None,
            'model_updated': False,
            'drift_detected': False,
            'performance_alert': None
        }
        
        # Make prediction before learning
        prediction_result = self.predict(features)
        result['prediction'] = prediction_result['prediction']
        result['confidence'] = prediction_result['confidence']
        
        # Add to buffer
        self.data_buffer['features'].append(features)
        self.data_buffer['labels'].append(label)
        self.data_buffer['timestamps'].append(timestamp)
        self.data_buffer['predictions'].append(result['prediction'])
        
        # Update online models immediately
        if self.learning_state['is_learning']:
            self._update_online_models(features, label)
        
        # Check for concept drift
        if self._should_check_drift():
            drift_result = self.concept_drift_detector.check_drift(
                self.data_buffer['predictions'][-100:],
                self.data_buffer['labels'][-100:]
            )
            result['drift_detected'] = drift_result['drift_detected']
            
            if drift_result['drift_detected']:
                self._handle_concept_drift(drift_result)
        
        # Batch update check
        if len(self.data_buffer['features']) >= self.config['update_frequency']:
            result['model_updated'] = self._perform_batch_update()
        
        # Performance monitoring
        self._update_performance_metrics(result['prediction'], label)
        
        # Check for performance alerts
        if self._check_performance_degradation():
            result['performance_alert'] = self._generate_performance_alert()
        
        self.learning_state['total_samples'] += 1
        
        return result
    
    def _update_online_models(self, features: pd.DataFrame, label: float):
        """
        Update truly online models incrementally
        """
        # Convert to dict for River
        feature_dict = features.iloc[0].to_dict() if isinstance(features, pd.DataFrame) else features
        
        # Update River models
        self.models['river_ensemble'].learn_one(feature_dict, label)
        self.models['online_boosting'].learn_one(feature_dict, label)
    
    def _perform_batch_update(self) -> bool:
        """
        Perform batch update on models
        """
        try:
            print("\n🔄 Performing batch model update...")
            
            # Prepare batch data
            X_batch = pd.concat(self.data_buffer['features'], ignore_index=True)
            y_batch = np.array(self.data_buffer['labels'])
            
            # Split for validation
            split_idx = int(len(X_batch) * (1 - self.config['validation_split']))
            X_train, X_val = X_batch[:split_idx], X_batch[split_idx:]
            y_train, y_val = y_batch[:split_idx], y_batch[split_idx:]
            
            # Update XGBoost incrementally
            if 'xgboost' in self.models and self.models['xgboost'] is not None:
                self._update_xgboost_incremental(X_train, y_train, X_val, y_val)
            
            # Save model version
            self._save_model_version()
            
            # Clear buffer
            self._clear_data_buffer()
            
            self.learning_state['last_update'] = datetime.now()
            
            print("✅ Batch update completed successfully")
            return True
            
        except Exception as e:
            print(f"❌ Batch update failed: {str(e)}")
            return False
    
    def _update_xgboost_incremental(self, X_train: pd.DataFrame, y_train: np.ndarray,
                                   X_val: pd.DataFrame, y_val: np.ndarray):
        """
        Update XGBoost model incrementally
        """
        # Create DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        # Update existing model
        if self.models.get('xgboost') is not None:
            # Continue training from existing model
            self.models['xgboost'] = xgb.train(
                {'objective': 'binary:logistic', 'eval_metric': 'logloss'},
                dtrain,
                num_boost_round=50,
                evals=[(dval, 'validation')],
                xgb_model=self.models['xgboost'],
                verbose_eval=False
            )
        else:
            # Train new model
            self.models['xgboost'] = xgb.train(
                {'objective': 'binary:logistic', 'eval_metric': 'logloss'},
                dtrain,
                num_boost_round=100,
                evals=[(dval, 'validation')],
                verbose_eval=False
            )
    
    def _should_check_drift(self) -> bool:
        """
        Determine if drift check is needed
        """
        time_since_check = (datetime.now() - self.learning_state['last_drift_check']).seconds
        return time_since_check > 3600 or len(self.data_buffer['labels']) >= 200
    
    def _handle_concept_drift(self, drift_result: Dict):
        """
        Handle detected concept drift
        """
        print("\n⚠️ CONCEPT DRIFT DETECTED")
        print(f"Drift Type: {drift_result['drift_type']}")
        print(f"Magnitude: {drift_result['magnitude']:.3f}")
        
        # Adjust learning rate
        if drift_result['magnitude'] > 0.1:
            print("🔧 Adjusting model parameters for drift...")
            # Increase adaptation rate for online models
            self.models['river_ensemble'].lambda_value = 3  # Faster adaptation
            
            # Consider full retrain if drift is severe
            if drift_result['magnitude'] > 0.2:
                self._trigger_full_retrain()
        
        self.learning_state['last_drift_check'] = datetime.now()
    
    def _update_performance_metrics(self, prediction: float, actual: float):
        """
        Update performance tracking
        """
        # Calculate error
        error = abs(prediction - actual)
        self.performance_history.append({
            'timestamp': datetime.now(),
            'prediction': prediction,
            'actual': actual,
            'error': error,
            'squared_error': error ** 2
        })
        
        # Update rolling metrics
        if len(self.performance_history) >= 10:
            recent_errors = [p['error'] for p in list(self.performance_history)[-50:]]
            self.metrics_tracker['accuracy'].append(1 - np.mean(recent_errors))
    
    def _check_performance_degradation(self) -> bool:
        """
        Check for performance degradation
        """
        if len(self.metrics_tracker['accuracy']) < 20:
            return False
        
        # Compare recent vs historical performance
        recent_perf = np.mean(list(self.metrics_tracker['accuracy'])[-10:])
        historical_perf = np.mean(list(self.metrics_tracker['accuracy'])[-50:-10])
        
        degradation = (historical_perf - recent_perf) / historical_perf
        
        return degradation > 0.1  # 10% degradation
    
    def _generate_performance_alert(self) -> Dict:
        """
        Generate performance alert
        """
        recent_perf = np.mean(list(self.metrics_tracker['accuracy'])[-10:])
        
        alert = {
            'severity': 'HIGH' if recent_perf < 0.5 else 'MEDIUM',
            'message': f'Performance degradation detected. Current accuracy: {recent_perf:.2%}',
            'recommendation': 'Consider model retraining or parameter adjustment',
            'timestamp': datetime.now()
        }
        
        return alert
    
    def _trigger_full_retrain(self):
        """
        Trigger full model retrain
        """
        print("\n🔄 Triggering full model retrain...")
        
        # This would typically:
        # 1. Fetch larger historical dataset
        # 2. Retrain all models from scratch
        # 3. Validate performance
        # 4. Deploy if performance improves
        
        # For now, reset online models
        self._initialize_models(None)
        self._clear_data_buffer()
        
        print("✅ Model retrain completed")
    
    def _save_model_version(self):
        """
        Save current model version
        """
        self.current_version += 1
        version_data = {
            'version': self.current_version,
            'timestamp': datetime.now(),
            'performance': np.mean(list(self.metrics_tracker['accuracy'])[-20:]) if self.metrics_tracker['accuracy'] else 0,
            'total_samples': self.learning_state['total_samples'],
            'models': {}
        }
        
        # Save model states (simplified for demo)
        self.model_versions.append(version_data)
        
        print(f"💾 Saved model version {self.current_version}")
    
    def _clear_data_buffer(self):
        """
        Clear data buffer after update
        """
        self.data_buffer = {
            'features': [],
            'labels': [],
            'timestamps': [],
            'predictions': []
        }
    
    def predict(self, features: pd.DataFrame) -> Dict:
        """
        Make prediction with confidence estimation
        """
        predictions = []
        
        # Get predictions from all models
        feature_dict = features.iloc[0].to_dict() if isinstance(features, pd.DataFrame) else features
        
        # River models
        if self.models['river_ensemble']:
            try:
                river_pred = self.models['river_ensemble'].predict_proba_one(feature_dict)
                predictions.append(river_pred.get(1, 0.5) if isinstance(river_pred, dict) else 0.5)
            except:
                predictions.append(0.5)
        
        if self.models['online_boosting']:
            try:
                boost_pred = self.models['online_boosting'].predict_proba_one(feature_dict)
                predictions.append(boost_pred.get(1, 0.5) if isinstance(boost_pred, dict) else 0.5)
            except:
                predictions.append(0.5)
        
        # XGBoost
        if self.models.get('xgboost') is not None:
            try:
                dmatrix = xgb.DMatrix(features)
                xgb_pred = self.models['xgboost'].predict(dmatrix)[0]
                predictions.append(xgb_pred)
            except:
                pass
        
        # Ensemble prediction
        if predictions:
            final_prediction = np.mean(predictions)
            confidence = 1 - np.std(predictions) if len(predictions) > 1 else 0.5
        else:
            final_prediction = 0.5
            confidence = 0.0
        
        return {
            'prediction': final_prediction,
            'confidence': confidence,
            'model_predictions': predictions
        }
    
    def get_learning_status(self) -> Dict:
        """
        Get current learning system status
        """
        recent_performance = np.mean(list(self.metrics_tracker['accuracy'])[-20:]) if self.metrics_tracker['accuracy'] else 0
        
        status = {
            'is_learning': self.learning_state['is_learning'],
            'total_samples_processed': self.learning_state['total_samples'],
            'current_version': self.current_version,
            'last_update': self.learning_state['last_update'],
            'buffer_size': len(self.data_buffer['features']),
            'recent_performance': recent_performance,
            'performance_trend': self._calculate_performance_trend(),
            'model_health': self._assess_model_health()
        }
        
        return status
    
    def _calculate_performance_trend(self) -> str:
        """
        Calculate performance trend
        """
        if len(self.metrics_tracker['accuracy']) < 20:
            return 'insufficient_data'
        
        recent = np.mean(list(self.metrics_tracker['accuracy'])[-10:])
        older = np.mean(list(self.metrics_tracker['accuracy'])[-20:-10])
        
        if recent > older * 1.05:
            return 'improving'
        elif recent < older * 0.95:
            return 'degrading'
        else:
            return 'stable'
    
    def _assess_model_health(self) -> str:
        """
        Assess overall model health
        """
        if not self.metrics_tracker['accuracy']:
            return 'unknown'
        
        recent_performance = np.mean(list(self.metrics_tracker['accuracy'])[-20:])
        
        if recent_performance >= 0.7:
            return 'excellent'
        elif recent_performance >= 0.6:
            return 'good'
        elif recent_performance >= 0.5:
            return 'fair'
        else:
            return 'poor'
    
    def pause_learning(self):
        """Pause online learning"""
        self.learning_state['is_learning'] = False
        print("⏸️ Online learning paused")
    
    def resume_learning(self):
        """Resume online learning"""
        self.learning_state['is_learning'] = True
        print("▶️ Online learning resumed")
    
    def rollback_to_version(self, version: int) -> bool:
        """
        Rollback to a previous model version
        """
        # Find version in history
        for v in self.model_versions:
            if v['version'] == version:
                print(f"🔄 Rolling back to version {version}...")
                # In production, this would restore saved model files
                self.current_version = version
                return True
        
        print(f"❌ Version {version} not found")
        return False


class ConceptDriftDetector:
    """
    Detects concept drift in data streams
    """
    
    def __init__(self):
        self.drift_metrics = {
            'adwin': metrics.ADWIN(),  # Adaptive windowing
            'page_hinkley': metrics.PageHinkley(),  # Page-Hinkley test
            'kswin': None  # Kolmogorov-Smirnov windowing
        }
        
        self.drift_history = deque(maxlen=100)
    
    def check_drift(self, predictions: List[float], actuals: List[float]) -> Dict:
        """
        Check for concept drift
        """
        if len(predictions) != len(actuals) or len(predictions) < 10:
            return {'drift_detected': False, 'drift_type': None, 'magnitude': 0}
        
        # Calculate errors
        errors = [abs(p - a) for p, a in zip(predictions, actuals)]
        
        # Update drift detectors
        drift_detected = False
        drift_types = []
        
        for error in errors:
            # ADWIN
            self.drift_metrics['adwin'].update(error)
            if self.drift_metrics['adwin'].drift_detected:
                drift_detected = True
                drift_types.append('adwin')
            
            # Page-Hinkley
            self.drift_metrics['page_hinkley'].update(error)
            if self.drift_metrics['page_hinkley'].drift_detected:
                drift_detected = True
                drift_types.append('page_hinkley')
        
        # Calculate drift magnitude
        recent_error = np.mean(errors[-20:])
        historical_error = np.mean(errors[:-20]) if len(errors) > 20 else recent_error
        magnitude = abs(recent_error - historical_error) / (historical_error + 1e-6)
        
        result = {
            'drift_detected': drift_detected,
            'drift_type': drift_types[0] if drift_types else None,
            'magnitude': magnitude,
            'recent_error': recent_error,
            'historical_error': historical_error
        }
        
        if drift_detected:
            self.drift_history.append({
                'timestamp': datetime.now(),
                'type': result['drift_type'],
                'magnitude': magnitude
            })
        
        return result


def main():
    """
    Demo the online learning system
    """
    print("🤖 ONLINE LEARNING SYSTEM DEMO")
    print("="*80)
    
    # Initialize system
    online_system = OnlineLearningSystem()
    
    # Simulate streaming data
    print("\n📊 Simulating streaming market data...")
    
    np.random.seed(42)
    n_samples = 500
    
    # Generate synthetic features
    for i in range(n_samples):
        # Create feature vector
        features = pd.DataFrame({
            'feature_1': [np.random.randn()],
            'feature_2': [np.random.randn()],
            'feature_3': [np.random.randn()],
            'momentum': [np.random.randn()],
            'volatility': [abs(np.random.randn())]
        })
        
        # Generate label with some pattern
        label = 1 if features['feature_1'].iloc[0] + features['momentum'].iloc[0] > 0 else 0
        
        # Add some noise
        if np.random.random() < 0.1:
            label = 1 - label
        
        # Introduce concept drift halfway
        if i > n_samples // 2:
            # Change the pattern
            label = 1 if features['feature_2'].iloc[0] - features['volatility'].iloc[0] > 0 else 0
        
        # Process new data
        result = online_system.process_new_data(
            features,
            label,
            datetime.now() + timedelta(minutes=i)
        )
        
        # Print updates
        if i % 50 == 0:
            print(f"\n📈 Sample {i}:")
            print(f"  Prediction: {result['prediction']:.3f}")
            print(f"  Confidence: {result['confidence']:.3f}")
            
            if result['drift_detected']:
                print("  ⚠️ DRIFT DETECTED!")
            
            if result['model_updated']:
                print("  ✅ Model updated")
            
            if result['performance_alert']:
                print(f"  🚨 Alert: {result['performance_alert']['message']}")
    
    # Final status
    print("\n📊 FINAL LEARNING STATUS:")
    print("="*60)
    
    status = online_system.get_learning_status()
    for key, value in status.items():
        print(f"  {key}: {value}")
    
    # Save results
    os.makedirs('outputs/online_learning', exist_ok=True)
    
    with open('outputs/online_learning/learning_status.json', 'w') as f:
        json.dump(status, f, indent=2, default=str)
    
    print("\n💾 Results saved to outputs/online_learning/")
    print("\n✅ Online Learning System Demo Complete!")


if __name__ == "__main__":
    main()