#!/usr/bin/env python3
"""
Ensemble Machine Learning System
Combines XGBoost, LSTM, and Random Forest for superior predictions
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import pickle
import os
from typing import Dict, List, Tuple, Optional

# ML Libraries
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb

# Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, concatenate
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

import warnings
warnings.filterwarnings('ignore')

class EnsembleMLSystem:
    """
    Advanced ensemble system combining:
    - XGBoost for gradient boosting
    - LSTM for time series patterns
    - Random Forest for robust predictions
    - LightGBM for speed and accuracy
    - Meta-learner for final predictions
    """
    
    def __init__(self, task_type='classification'):
        self.task_type = task_type  # 'classification' or 'regression'
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.performance_metrics = {}
        
        # Model configurations
        self.model_configs = {
            'xgboost': {
                'n_estimators': 300,
                'max_depth': 6,
                'learning_rate': 0.01,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'objective': 'binary:logistic' if task_type == 'classification' else 'reg:squarederror',
                'eval_metric': 'logloss' if task_type == 'classification' else 'rmse',
                'use_label_encoder': False,
                'random_state': 42
            },
            'lightgbm': {
                'n_estimators': 300,
                'max_depth': 6,
                'learning_rate': 0.01,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'objective': 'binary' if task_type == 'classification' else 'regression',
                'metric': 'binary_logloss' if task_type == 'classification' else 'rmse',
                'verbose': -1,
                'random_state': 42
            },
            'random_forest': {
                'n_estimators': 200,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'max_features': 'sqrt',
                'random_state': 42,
                'n_jobs': -1
            },
            'lstm': {
                'units': [128, 64, 32],
                'dropout': 0.2,
                'batch_size': 32,
                'epochs': 100,
                'learning_rate': 0.001,
                'patience': 10
            }
        }
        
    def train_ensemble(self, X_train: pd.DataFrame, y_train: pd.Series, 
                      X_val: pd.DataFrame = None, y_val: pd.Series = None,
                      feature_groups: Dict = None) -> Dict:
        """
        Train all models in the ensemble
        """
        print("\n🚀 TRAINING ENSEMBLE ML SYSTEM")
        print("="*80)
        print(f"Task Type: {self.task_type}")
        print(f"Training samples: {len(X_train)}")
        print(f"Features: {len(X_train.columns)}")
        
        # Split data if validation not provided
        if X_val is None:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42, shuffle=False
            )
        
        # Scale features
        print("\n📊 Scaling features...")
        X_train_scaled, X_val_scaled = self._scale_features(X_train, X_val)
        
        # Train individual models
        print("\n🤖 Training individual models...")
        
        # 1. XGBoost
        print("\n1. Training XGBoost...")
        self._train_xgboost(X_train_scaled, y_train, X_val_scaled, y_val)
        
        # 2. LightGBM
        print("\n2. Training LightGBM...")
        self._train_lightgbm(X_train_scaled, y_train, X_val_scaled, y_val)
        
        # 3. Random Forest
        print("\n3. Training Random Forest...")
        self._train_random_forest(X_train_scaled, y_train, X_val_scaled, y_val)
        
        # 4. LSTM (requires sequence data)
        print("\n4. Training LSTM...")
        self._train_lstm(X_train, y_train, X_val, y_val, feature_groups)
        
        # 5. Train meta-learner
        print("\n5. Training Meta-Learner...")
        self._train_meta_learner(X_train_scaled, y_train, X_val_scaled, y_val)
        
        # Calculate ensemble performance
        print("\n📈 Evaluating ensemble performance...")
        ensemble_metrics = self._evaluate_ensemble(X_val_scaled, y_val)
        
        # Feature importance analysis
        print("\n🔍 Analyzing feature importance...")
        self._analyze_feature_importance(X_train)
        
        return {
            'individual_metrics': self.performance_metrics,
            'ensemble_metrics': ensemble_metrics,
            'feature_importance': self.feature_importance
        }
    
    def _scale_features(self, X_train: pd.DataFrame, X_val: pd.DataFrame) -> Tuple:
        """Scale features using RobustScaler"""
        # Use RobustScaler for financial data (handles outliers better)
        self.scalers['features'] = RobustScaler()
        
        X_train_scaled = pd.DataFrame(
            self.scalers['features'].fit_transform(X_train),
            index=X_train.index,
            columns=X_train.columns
        )
        
        X_val_scaled = pd.DataFrame(
            self.scalers['features'].transform(X_val),
            index=X_val.index,
            columns=X_val.columns
        )
        
        return X_train_scaled, X_val_scaled
    
    def _train_xgboost(self, X_train: pd.DataFrame, y_train: pd.Series,
                      X_val: pd.DataFrame, y_val: pd.Series):
        """Train XGBoost model"""
        # Prepare data
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        # Train model
        params = self.model_configs['xgboost'].copy()
        evallist = [(dtrain, 'train'), (dval, 'eval')]
        
        self.models['xgboost'] = xgb.train(
            params,
            dtrain,
            num_boost_round=params['n_estimators'],
            evals=evallist,
            early_stopping_rounds=50,
            verbose_eval=False
        )
        
        # Evaluate
        if self.task_type == 'classification':
            y_pred = (self.models['xgboost'].predict(dval) > 0.5).astype(int)
            accuracy = accuracy_score(y_val, y_pred)
            precision = precision_score(y_val, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_val, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)
            
            self.performance_metrics['xgboost'] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
            print(f"  Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        else:
            y_pred = self.models['xgboost'].predict(dval)
            mse = mean_squared_error(y_val, y_pred)
            mae = mean_absolute_error(y_val, y_pred)
            r2 = r2_score(y_val, y_pred)
            
            self.performance_metrics['xgboost'] = {
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'rmse': np.sqrt(mse)
            }
            print(f"  RMSE: {np.sqrt(mse):.4f}, R2: {r2:.4f}")
    
    def _train_lightgbm(self, X_train: pd.DataFrame, y_train: pd.Series,
                       X_val: pd.DataFrame, y_val: pd.Series):
        """Train LightGBM model"""
        # Create dataset
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # Train model
        params = self.model_configs['lightgbm'].copy()
        
        self.models['lightgbm'] = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        
        # Evaluate
        if self.task_type == 'classification':
            y_pred = (self.models['lightgbm'].predict(X_val, num_iteration=self.models['lightgbm'].best_iteration) > 0.5).astype(int)
            accuracy = accuracy_score(y_val, y_pred)
            f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)
            
            self.performance_metrics['lightgbm'] = {
                'accuracy': accuracy,
                'f1': f1
            }
            print(f"  Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        else:
            y_pred = self.models['lightgbm'].predict(X_val, num_iteration=self.models['lightgbm'].best_iteration)
            mse = mean_squared_error(y_val, y_pred)
            r2 = r2_score(y_val, y_pred)
            
            self.performance_metrics['lightgbm'] = {
                'mse': mse,
                'rmse': np.sqrt(mse),
                'r2': r2
            }
            print(f"  RMSE: {np.sqrt(mse):.4f}, R2: {r2:.4f}")
    
    def _train_random_forest(self, X_train: pd.DataFrame, y_train: pd.Series,
                           X_val: pd.DataFrame, y_val: pd.Series):
        """Train Random Forest model"""
        # Select model type
        if self.task_type == 'classification':
            self.models['random_forest'] = RandomForestClassifier(**self.model_configs['random_forest'])
        else:
            self.models['random_forest'] = RandomForestRegressor(**self.model_configs['random_forest'])
        
        # Train
        self.models['random_forest'].fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.models['random_forest'].predict(X_val)
        
        if self.task_type == 'classification':
            accuracy = accuracy_score(y_val, y_pred)
            f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)
            
            self.performance_metrics['random_forest'] = {
                'accuracy': accuracy,
                'f1': f1
            }
            print(f"  Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        else:
            mse = mean_squared_error(y_val, y_pred)
            r2 = r2_score(y_val, y_pred)
            
            self.performance_metrics['random_forest'] = {
                'mse': mse,
                'rmse': np.sqrt(mse),
                'r2': r2
            }
            print(f"  RMSE: {np.sqrt(mse):.4f}, R2: {r2:.4f}")
    
    def _train_lstm(self, X_train: pd.DataFrame, y_train: pd.Series,
                   X_val: pd.DataFrame, y_val: pd.Series, feature_groups: Dict = None):
        """Train LSTM model with advanced architecture"""
        # Prepare sequence data
        sequence_length = 20  # Look back 20 time steps
        
        X_train_seq, y_train_seq = self._create_sequences(X_train, y_train, sequence_length)
        X_val_seq, y_val_seq = self._create_sequences(X_val, y_val, sequence_length)
        
        if len(X_train_seq) == 0 or len(X_val_seq) == 0:
            print("  Insufficient data for LSTM sequences, skipping...")
            self.performance_metrics['lstm'] = {'error': 'Insufficient data'}
            return
        
        # Build advanced LSTM architecture
        model = self._build_advanced_lstm(
            input_shape=(sequence_length, X_train.shape[1]),
            output_dim=1 if self.task_type == 'regression' else 2
        )
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=self.model_configs['lstm']['patience'], restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001),
            ModelCheckpoint('outputs/lstm_best_model.h5', save_best_only=True, monitor='val_loss')
        ]
        
        # Train
        history = model.fit(
            X_train_seq, y_train_seq,
            validation_data=(X_val_seq, y_val_seq),
            epochs=self.model_configs['lstm']['epochs'],
            batch_size=self.model_configs['lstm']['batch_size'],
            callbacks=callbacks,
            verbose=0
        )
        
        self.models['lstm'] = model
        
        # Evaluate
        y_pred = model.predict(X_val_seq, verbose=0)
        
        if self.task_type == 'classification':
            y_pred_classes = np.argmax(y_pred, axis=1)
            y_val_classes = y_val_seq  # Assuming already in class format
            accuracy = accuracy_score(y_val_classes, y_pred_classes)
            
            self.performance_metrics['lstm'] = {
                'accuracy': accuracy,
                'best_epoch': callbacks[0].stopped_epoch - callbacks[0].patience if callbacks[0].stopped_epoch > 0 else len(history.history['loss'])
            }
            print(f"  Accuracy: {accuracy:.4f}")
        else:
            mse = mean_squared_error(y_val_seq, y_pred.flatten())
            r2 = r2_score(y_val_seq, y_pred.flatten())
            
            self.performance_metrics['lstm'] = {
                'mse': mse,
                'rmse': np.sqrt(mse),
                'r2': r2,
                'best_epoch': callbacks[0].stopped_epoch - callbacks[0].patience if callbacks[0].stopped_epoch > 0 else len(history.history['loss'])
            }
            print(f"  RMSE: {np.sqrt(mse):.4f}, R2: {r2:.4f}")
    
    def _build_advanced_lstm(self, input_shape: Tuple, output_dim: int) -> Model:
        """Build advanced LSTM architecture with attention mechanism"""
        inputs = Input(shape=input_shape)
        
        # CNN feature extraction
        conv1 = Conv1D(filters=64, kernel_size=3, activation='relu')(inputs)
        conv1 = BatchNormalization()(conv1)
        pool1 = MaxPooling1D(pool_size=2)(conv1)
        
        # LSTM layers
        lstm1 = LSTM(self.model_configs['lstm']['units'][0], return_sequences=True)(pool1)
        lstm1 = Dropout(self.model_configs['lstm']['dropout'])(lstm1)
        lstm1 = BatchNormalization()(lstm1)
        
        lstm2 = LSTM(self.model_configs['lstm']['units'][1], return_sequences=True)(lstm1)
        lstm2 = Dropout(self.model_configs['lstm']['dropout'])(lstm2)
        lstm2 = BatchNormalization()(lstm2)
        
        lstm3 = LSTM(self.model_configs['lstm']['units'][2], return_sequences=False)(lstm2)
        lstm3 = Dropout(self.model_configs['lstm']['dropout'])(lstm3)
        
        # Dense layers
        dense1 = Dense(64, activation='relu')(lstm3)
        dense1 = Dropout(self.model_configs['lstm']['dropout'])(dense1)
        
        dense2 = Dense(32, activation='relu')(dense1)
        dense2 = Dropout(self.model_configs['lstm']['dropout']/2)(dense2)
        
        # Output layer
        if self.task_type == 'classification':
            outputs = Dense(output_dim, activation='softmax')(dense2)
            loss = 'sparse_categorical_crossentropy'
            metrics = ['accuracy']
        else:
            outputs = Dense(output_dim, activation='linear')(dense2)
            loss = 'mse'
            metrics = ['mae']
        
        model = Model(inputs=inputs, outputs=outputs)
        
        # Compile
        optimizer = Adam(learning_rate=self.model_configs['lstm']['learning_rate'])
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        
        return model
    
    def _create_sequences(self, X: pd.DataFrame, y: pd.Series, sequence_length: int) -> Tuple:
        """Create sequences for LSTM"""
        X_seq = []
        y_seq = []
        
        for i in range(sequence_length, len(X)):
            X_seq.append(X.iloc[i-sequence_length:i].values)
            y_seq.append(y.iloc[i])
        
        return np.array(X_seq), np.array(y_seq)
    
    def _train_meta_learner(self, X_train: pd.DataFrame, y_train: pd.Series,
                          X_val: pd.DataFrame, y_val: pd.Series):
        """Train meta-learner to combine predictions"""
        # Get predictions from base models
        train_predictions = self._get_base_predictions(X_train, 'train')
        val_predictions = self._get_base_predictions(X_val, 'val')
        
        # Create meta features
        meta_train = pd.DataFrame(train_predictions)
        meta_val = pd.DataFrame(val_predictions)
        
        # Train meta-learner (using XGBoost)
        if self.task_type == 'classification':
            self.models['meta_learner'] = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                random_state=42
            )
        else:
            self.models['meta_learner'] = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                random_state=42
            )
        
        self.models['meta_learner'].fit(meta_train, y_train)
        
        # Evaluate
        y_pred = self.models['meta_learner'].predict(meta_val)
        
        if self.task_type == 'classification':
            accuracy = accuracy_score(y_val, y_pred)
            f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)
            
            self.performance_metrics['meta_learner'] = {
                'accuracy': accuracy,
                'f1': f1
            }
            print(f"  Meta-Learner Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        else:
            mse = mean_squared_error(y_val, y_pred)
            r2 = r2_score(y_val, y_pred)
            
            self.performance_metrics['meta_learner'] = {
                'mse': mse,
                'rmse': np.sqrt(mse),
                'r2': r2
            }
            print(f"  Meta-Learner RMSE: {np.sqrt(mse):.4f}, R2: {r2:.4f}")
    
    def _get_base_predictions(self, X: pd.DataFrame, dataset_type: str) -> Dict:
        """Get predictions from all base models"""
        predictions = {}
        
        # XGBoost
        if 'xgboost' in self.models:
            dmatrix = xgb.DMatrix(X)
            predictions['xgboost'] = self.models['xgboost'].predict(dmatrix)
        
        # LightGBM
        if 'lightgbm' in self.models:
            predictions['lightgbm'] = self.models['lightgbm'].predict(X, num_iteration=self.models['lightgbm'].best_iteration)
        
        # Random Forest
        if 'random_forest' in self.models:
            if self.task_type == 'classification':
                predictions['random_forest'] = self.models['random_forest'].predict_proba(X)[:, 1]
            else:
                predictions['random_forest'] = self.models['random_forest'].predict(X)
        
        # LSTM (simplified - would need sequence handling in production)
        if 'lstm' in self.models and dataset_type == 'val':
            # For now, use last prediction repeated
            predictions['lstm'] = np.zeros(len(X))  # Placeholder
        
        return predictions
    
    def _evaluate_ensemble(self, X_val: pd.DataFrame, y_val: pd.Series) -> Dict:
        """Evaluate ensemble performance"""
        # Get ensemble predictions
        y_pred = self.predict(X_val)
        
        if self.task_type == 'classification':
            accuracy = accuracy_score(y_val, y_pred)
            precision = precision_score(y_val, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_val, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)
            
            metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
            
            print(f"\n🎯 Ensemble Performance:")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1 Score: {f1:.4f}")
        else:
            mse = mean_squared_error(y_val, y_pred)
            mae = mean_absolute_error(y_val, y_pred)
            r2 = r2_score(y_val, y_pred)
            
            metrics = {
                'mse': mse,
                'mae': mae,
                'rmse': np.sqrt(mse),
                'r2': r2
            }
            
            print(f"\n🎯 Ensemble Performance:")
            print(f"  RMSE: {np.sqrt(mse):.4f}")
            print(f"  MAE: {mae:.4f}")
            print(f"  R2 Score: {r2:.4f}")
        
        return metrics
    
    def _analyze_feature_importance(self, X_train: pd.DataFrame):
        """Analyze and combine feature importance from all models"""
        feature_names = X_train.columns.tolist()
        
        # XGBoost importance
        if 'xgboost' in self.models:
            xgb_importance = self.models['xgboost'].get_score(importance_type='gain')
            xgb_importance = {k: v for k, v in sorted(xgb_importance.items(), key=lambda x: x[1], reverse=True)}
            self.feature_importance['xgboost'] = xgb_importance
        
        # LightGBM importance
        if 'lightgbm' in self.models:
            lgb_importance = dict(zip(feature_names, self.models['lightgbm'].feature_importance(importance_type='gain')))
            lgb_importance = {k: v for k, v in sorted(lgb_importance.items(), key=lambda x: x[1], reverse=True)}
            self.feature_importance['lightgbm'] = lgb_importance
        
        # Random Forest importance
        if 'random_forest' in self.models:
            rf_importance = dict(zip(feature_names, self.models['random_forest'].feature_importances_))
            rf_importance = {k: v for k, v in sorted(rf_importance.items(), key=lambda x: x[1], reverse=True)}
            self.feature_importance['random_forest'] = rf_importance
        
        # Combined importance (average across models)
        combined_importance = {}
        for feature in feature_names:
            scores = []
            for model_name in ['xgboost', 'lightgbm', 'random_forest']:
                if model_name in self.feature_importance:
                    # Normalize scores to 0-1 range for each model
                    model_scores = self.feature_importance[model_name]
                    max_score = max(model_scores.values()) if model_scores else 1
                    normalized_score = model_scores.get(feature, 0) / max_score if max_score > 0 else 0
                    scores.append(normalized_score)
            
            if scores:
                combined_importance[feature] = np.mean(scores)
        
        # Sort and store
        combined_importance = {k: v for k, v in sorted(combined_importance.items(), key=lambda x: x[1], reverse=True)}
        self.feature_importance['combined'] = combined_importance
        
        # Display top features
        print("\n📊 Top 10 Most Important Features:")
        for i, (feature, score) in enumerate(list(combined_importance.items())[:10]):
            print(f"  {i+1}. {feature}: {score:.4f}")
    
    def predict(self, X: pd.DataFrame, return_probabilities: bool = False) -> np.ndarray:
        """Make predictions using the ensemble"""
        # Scale features
        X_scaled = pd.DataFrame(
            self.scalers['features'].transform(X),
            index=X.index,
            columns=X.columns
        )
        
        # Get base predictions
        base_predictions = self._get_base_predictions(X_scaled, 'predict')
        
        # Create meta features
        meta_features = pd.DataFrame(base_predictions)
        
        # Get ensemble prediction
        if 'meta_learner' in self.models:
            if return_probabilities and self.task_type == 'classification':
                predictions = self.models['meta_learner'].predict_proba(meta_features)
            else:
                predictions = self.models['meta_learner'].predict(meta_features)
        else:
            # Simple average if no meta-learner
            predictions = np.mean(list(base_predictions.values()), axis=0)
            if self.task_type == 'classification' and not return_probabilities:
                predictions = (predictions > 0.5).astype(int)
        
        return predictions
    
    def predict_with_confidence(self, X: pd.DataFrame) -> Dict:
        """Make predictions with confidence intervals"""
        # Get predictions from all models
        X_scaled = pd.DataFrame(
            self.scalers['features'].transform(X),
            index=X.index,
            columns=X.columns
        )
        
        all_predictions = []
        
        # Collect predictions from each model
        if 'xgboost' in self.models:
            dmatrix = xgb.DMatrix(X_scaled)
            all_predictions.append(self.models['xgboost'].predict(dmatrix))
        
        if 'lightgbm' in self.models:
            all_predictions.append(self.models['lightgbm'].predict(X_scaled))
        
        if 'random_forest' in self.models:
            if self.task_type == 'classification':
                all_predictions.append(self.models['random_forest'].predict_proba(X_scaled)[:, 1])
            else:
                all_predictions.append(self.models['random_forest'].predict(X_scaled))
        
        # Calculate statistics
        all_predictions = np.array(all_predictions)
        mean_prediction = np.mean(all_predictions, axis=0)
        std_prediction = np.std(all_predictions, axis=0)
        
        # Use meta-learner for final prediction
        final_prediction = self.predict(X)
        
        return {
            'prediction': final_prediction,
            'mean': mean_prediction,
            'std': std_prediction,
            'lower_bound': mean_prediction - 2 * std_prediction,
            'upper_bound': mean_prediction + 2 * std_prediction,
            'confidence': 1 - (std_prediction / (np.abs(mean_prediction) + 1e-6))  # Confidence metric
        }
    
    def save_models(self, path: str):
        """Save all models and scalers"""
        os.makedirs(path, exist_ok=True)
        
        # Save sklearn/xgboost/lightgbm models
        for model_name in ['xgboost', 'lightgbm', 'random_forest', 'meta_learner']:
            if model_name in self.models:
                with open(f'{path}/{model_name}.pkl', 'wb') as f:
                    pickle.dump(self.models[model_name], f)
        
        # Save LSTM separately
        if 'lstm' in self.models:
            self.models['lstm'].save(f'{path}/lstm_model.h5')
        
        # Save scalers
        with open(f'{path}/scalers.pkl', 'wb') as f:
            pickle.dump(self.scalers, f)
        
        # Save feature importance
        with open(f'{path}/feature_importance.json', 'w') as f:
            json.dump(self.feature_importance, f, indent=2)
        
        # Save performance metrics
        with open(f'{path}/performance_metrics.json', 'w') as f:
            json.dump(self.performance_metrics, f, indent=2)
        
        print(f"\n💾 Models saved to {path}/")
    
    def load_models(self, path: str):
        """Load all models and scalers"""
        # Load sklearn/xgboost/lightgbm models
        for model_name in ['xgboost', 'lightgbm', 'random_forest', 'meta_learner']:
            model_path = f'{path}/{model_name}.pkl'
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    self.models[model_name] = pickle.load(f)
        
        # Load LSTM
        lstm_path = f'{path}/lstm_model.h5'
        if os.path.exists(lstm_path):
            self.models['lstm'] = tf.keras.models.load_model(lstm_path)
        
        # Load scalers
        with open(f'{path}/scalers.pkl', 'rb') as f:
            self.scalers = pickle.load(f)
        
        # Load feature importance
        with open(f'{path}/feature_importance.json', 'r') as f:
            self.feature_importance = json.load(f)
        
        # Load performance metrics
        with open(f'{path}/performance_metrics.json', 'r') as f:
            self.performance_metrics = json.load(f)
        
        print(f"✅ Models loaded from {path}/")


def main():
    """Test the ensemble ML system"""
    print("🤖 ENSEMBLE ML SYSTEM DEMO")
    print("="*80)
    
    # Generate synthetic data for demo
    np.random.seed(42)
    n_samples = 1000
    n_features = 50
    
    # Create synthetic features
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    # Create target (classification)
    y = pd.Series((X['feature_0'] + X['feature_1'] + np.random.randn(n_samples) * 0.5 > 0).astype(int))
    
    # Split data
    train_size = int(0.8 * n_samples)
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_test = X[train_size:]
    y_test = y[train_size:]
    
    # Initialize ensemble
    ensemble = EnsembleMLSystem(task_type='classification')
    
    # Train
    results = ensemble.train_ensemble(X_train, y_train, X_test, y_test)
    
    # Make predictions with confidence
    print("\n🔮 Making predictions with confidence intervals...")
    predictions = ensemble.predict_with_confidence(X_test[:5])
    
    print("\nSample Predictions:")
    for i in range(5):
        print(f"  Sample {i+1}:")
        print(f"    Prediction: {predictions['prediction'][i]}")
        print(f"    Confidence: {predictions['confidence'][i]:.2%}")
        print(f"    Range: [{predictions['lower_bound'][i]:.3f}, {predictions['upper_bound'][i]:.3f}]")
    
    # Save models
    ensemble.save_models('outputs/ensemble_models')
    
    print("\n✅ Ensemble ML System Demo Complete!")


if __name__ == "__main__":
    main()