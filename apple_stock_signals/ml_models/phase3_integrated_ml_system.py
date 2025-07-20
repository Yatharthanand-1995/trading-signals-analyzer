#!/usr/bin/env python3
"""
Phase 3 Integrated ML System
Combines enhanced features, ensemble models, and online learning
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import sys
from typing import Dict, List, Tuple, Optional
import yfinance as yf

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import Phase 3 ML components
from enhanced_feature_engineering import EnhancedFeatureEngineer as FeatureEngineer
from ensemble_ml_system import EnsembleMLSystem
from online_learning_system import OnlineLearningSystem
from market_microstructure_features import MarketMicrostructureAnalyzer

# Import Phase 1 & 2 systems
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'core_scripts'))
from phase2_integrated_system import Phase2IntegratedSystem

class Phase3IntegratedMLSystem:
    """
    Complete ML-enhanced trading system with:
    - 300+ engineered features
    - Ensemble ML models (XGBoost, LSTM, RF, LightGBM)
    - Online learning for continuous adaptation
    - Market microstructure analysis
    """
    
    def __init__(self, portfolio_size: float = 100000):
        # Phase 2 system (includes Phase 1)
        self.phase2_system = Phase2IntegratedSystem(portfolio_size)
        
        # Phase 3 ML components
        self.feature_engineer = FeatureEngineer()
        self.ensemble_ml = EnsembleMLSystem(task_type='classification')
        self.online_learner = OnlineLearningSystem()
        self.microstructure_analyzer = MarketMicrostructureAnalyzer()
        
        self.portfolio_size = portfolio_size
        
        # ML configuration
        self.ml_config = {
            'use_ml_filtering': True,
            'ml_confidence_threshold': 0.65,
            'feature_importance_threshold': 0.01,
            'online_learning_enabled': True,
            'microstructure_weight': 0.2
        }
        
        # Performance tracking
        self.ml_performance = {
            'predictions': [],
            'actuals': [],
            'feature_importance': {},
            'model_versions': []
        }
    
    def run_ml_enhanced_analysis(self, symbols: List[str]) -> Dict:
        """
        Run complete ML-enhanced analysis
        """
        print("\n🤖 PHASE 3 ML-ENHANCED TRADING SYSTEM")
        print("="*80)
        print(f"Portfolio Size: ${self.portfolio_size:,}")
        print(f"Analyzing {len(symbols)} symbols with advanced ML...")
        print("="*80)
        
        # Step 1: Prepare ML training data
        print("\n📊 STEP 1: PREPARING ML TRAINING DATA")
        training_data = self._prepare_ml_training_data(symbols)
        
        if training_data is None or len(training_data['X_train']) < 100:
            print("❌ Insufficient data for ML training")
            # Fall back to Phase 2 system
            return self.phase2_system.run_complete_analysis(symbols)
        
        # Step 2: Train ensemble ML models
        print("\n🤖 STEP 2: TRAINING ENSEMBLE ML MODELS")
        ml_training_results = self._train_ml_models(training_data)
        
        # Step 3: Run Phase 2 analysis to get base signals
        print("\n🔍 STEP 3: GENERATING BASE SIGNALS")
        phase2_results = self.phase2_system.run_complete_analysis(symbols)
        
        # Step 4: Enhance signals with ML predictions
        print("\n🎯 STEP 4: ENHANCING SIGNALS WITH ML")
        ml_enhanced_signals = self._enhance_signals_with_ml(
            phase2_results['enhanced_positions'],
            symbols,
            training_data['feature_columns']
        )
        
        # Step 5: Apply online learning updates
        if self.ml_config['online_learning_enabled']:
            print("\n🔄 STEP 5: APPLYING ONLINE LEARNING")
            ml_enhanced_signals = self._apply_online_learning(ml_enhanced_signals)
        
        # Step 6: Generate comprehensive report
        report = self._generate_ml_report(
            phase2_results,
            ml_enhanced_signals,
            ml_training_results
        )
        
        # Save results
        self._save_ml_results(report)
        
        return report
    
    def _prepare_ml_training_data(self, symbols: List[str]) -> Optional[Dict]:
        """
        Prepare comprehensive training data for ML models
        """
        all_features = []
        all_labels = []
        
        print("\n📊 Collecting historical data for ML training...")
        
        for symbol in symbols[:20]:  # Limit to top 20 for training
            print(f"  Processing {symbol}...", end='', flush=True)
            
            try:
                # Fetch extended historical data
                ticker = yf.Ticker(symbol)
                data = ticker.history(period='2y', interval='1d')
                
                if len(data) < 100:
                    print(" ⚠️ Insufficient data")
                    continue
                
                # Engineer features
                features = self.feature_engineer.engineer_features(data)
                
                # Add microstructure features
                micro_features = self.microstructure_analyzer.calculate_all_features(data)
                
                # Combine features
                combined_features = pd.concat([features, micro_features], axis=1)
                
                # Create labels (1 if price goes up in next 5 days, 0 otherwise)
                future_return = data['Close'].shift(-5) / data['Close'] - 1
                labels = (future_return > 0.02).astype(int)  # 2% threshold
                
                # Remove NaN rows
                valid_idx = combined_features.notna().all(axis=1) & labels.notna()
                combined_features = combined_features[valid_idx]
                labels = labels[valid_idx]
                
                if len(combined_features) > 50:
                    all_features.append(combined_features)
                    all_labels.append(labels)
                    print(" ✔️")
                else:
                    print(" ⚠️ Too few samples after cleaning")
                    
            except Exception as e:
                print(f" ❌ Error: {str(e)}")
        
        if not all_features:
            return None
        
        # Combine all data
        X = pd.concat(all_features, ignore_index=True)
        y = pd.concat(all_labels, ignore_index=True)
        
        # Remove features with too many NaN values
        nan_threshold = 0.3
        nan_ratio = X.isna().sum() / len(X)
        valid_features = nan_ratio[nan_ratio < nan_threshold].index
        X = X[valid_features]
        
        # Fill remaining NaN values
        X = X.fillna(method='ffill').fillna(0)
        
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train = X[:split_idx]
        y_train = y[:split_idx]
        X_val = X[split_idx:]
        y_val = y[split_idx:]
        
        print(f"\n✅ Training data prepared:")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Validation samples: {len(X_val)}")
        print(f"  Features: {len(X.columns)}")
        print(f"  Positive class ratio: {y_train.mean():.2%}")
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'feature_columns': list(X.columns),
            'feature_stats': X.describe()
        }
    
    def _train_ml_models(self, training_data: Dict) -> Dict:
        """
        Train ensemble ML models
        """
        # Train ensemble
        training_results = self.ensemble_ml.train_ensemble(
            training_data['X_train'],
            training_data['y_train'],
            training_data['X_val'],
            training_data['y_val']
        )
        
        # Store feature importance
        self.ml_performance['feature_importance'] = training_results['feature_importance']
        
        # Display top features
        print("\n🎯 Top 20 Most Important Features:")
        top_features = list(training_results['feature_importance']['combined'].items())[:20]
        for i, (feature, importance) in enumerate(top_features):
            print(f"  {i+1:2d}. {feature[:50]:50s} {importance:.4f}")
        
        return training_results
    
    def _enhance_signals_with_ml(self, base_signals: List[Dict], 
                                symbols: List[str],
                                feature_columns: List[str]) -> List[Dict]:
        """
        Enhance trading signals with ML predictions
        """
        enhanced_signals = []
        
        for signal in base_signals:
            symbol = signal['symbol']
            print(f"\n🎯 Enhancing {symbol} with ML predictions...")
            
            try:
                # Fetch current data
                ticker = yf.Ticker(symbol)
                data = ticker.history(period='6mo', interval='1d')
                
                # Engineer features
                features = self.feature_engineer.engineer_features(data)
                micro_features = self.microstructure_analyzer.calculate_all_features(data)
                combined_features = pd.concat([features, micro_features], axis=1)
                
                # Get latest features
                latest_features = combined_features.iloc[-1:][feature_columns]
                latest_features = latest_features.fillna(0)
                
                # Get ML predictions
                ml_prediction = self.ensemble_ml.predict_with_confidence(latest_features)
                
                # Enhance signal with ML insights
                enhanced_signal = signal.copy()
                enhanced_signal['ml_prediction'] = {
                    'probability': float(ml_prediction['prediction'][0]),
                    'confidence': float(ml_prediction['confidence'][0]),
                    'prediction_range': [
                        float(ml_prediction['lower_bound'][0]),
                        float(ml_prediction['upper_bound'][0])
                    ]
                }
                
                # Adjust confidence based on ML
                ml_boost = (ml_prediction['prediction'][0] - 0.5) * 2  # Scale to [-1, 1]
                original_confidence = signal.get('confidence', 70)
                enhanced_signal['ml_enhanced_confidence'] = original_confidence + (ml_boost * 20)
                enhanced_signal['ml_enhanced_confidence'] = max(0, min(100, enhanced_signal['ml_enhanced_confidence']))
                
                # Add microstructure insights
                latest_micro = micro_features.iloc[-1] if not micro_features.empty else pd.Series()
                enhanced_signal['market_microstructure'] = {
                    'order_flow_imbalance': float(latest_micro.get('order_flow_order_flow_imbalance', 0)),
                    'liquidity_score': float(1 / (latest_micro.get('liquidity_amihud_illiquidity', 1) + 1)),
                    'market_stress': float(latest_micro.get('market_quality_market_stress', 1)),
                    'information_share': float(latest_micro.get('information_flow_info_share', 0.5))
                }
                
                # ML-based position sizing adjustment
                if enhanced_signal['ml_enhanced_confidence'] > 80:
                    enhanced_signal['ml_position_multiplier'] = 1.2
                elif enhanced_signal['ml_enhanced_confidence'] < 50:
                    enhanced_signal['ml_position_multiplier'] = 0.8
                else:
                    enhanced_signal['ml_position_multiplier'] = 1.0
                
                # Apply multiplier to position
                enhanced_signal['ml_adjusted_shares'] = int(
                    enhanced_signal['shares'] * enhanced_signal['ml_position_multiplier']
                )
                enhanced_signal['ml_adjusted_value'] = (
                    enhanced_signal['ml_adjusted_shares'] * enhanced_signal['entry_price']
                )
                
                enhanced_signals.append(enhanced_signal)
                
                print(f"  ML Probability: {ml_prediction['prediction'][0]:.3f}")
                print(f"  ML Confidence: {ml_prediction['confidence'][0]:.3f}")
                print(f"  Enhanced Confidence: {enhanced_signal['ml_enhanced_confidence']:.1f}%")
                print(f"  Position Multiplier: {enhanced_signal['ml_position_multiplier']:.2f}x")
                
            except Exception as e:
                print(f"  ❌ ML enhancement failed: {str(e)}")
                enhanced_signals.append(signal)  # Use original signal
        
        return enhanced_signals
    
    def _apply_online_learning(self, signals: List[Dict]) -> List[Dict]:
        """
        Apply online learning adjustments
        """
        for signal in signals:
            # Prepare features for online learning
            features = pd.DataFrame({
                'ml_probability': [signal['ml_prediction']['probability']],
                'confidence': [signal['ml_enhanced_confidence']],
                'order_flow': [signal['market_microstructure']['order_flow_imbalance']],
                'liquidity': [signal['market_microstructure']['liquidity_score']],
                'market_stress': [signal['market_microstructure']['market_stress']]
            })
            
            # Get online learning prediction
            online_result = self.online_learner.predict(features)
            
            # Add online learning insights
            signal['online_learning'] = {
                'adjusted_probability': online_result['prediction'],
                'learning_confidence': online_result['confidence'],
                'model_agreement': abs(online_result['prediction'] - signal['ml_prediction']['probability']) < 0.1
            }
            
            # Final confidence adjustment
            if signal['online_learning']['model_agreement']:
                signal['final_confidence'] = signal['ml_enhanced_confidence'] * 1.1
            else:
                signal['final_confidence'] = signal['ml_enhanced_confidence'] * 0.9
            
            signal['final_confidence'] = max(0, min(100, signal['final_confidence']))
        
        # Get learning status
        learning_status = self.online_learner.get_learning_status()
        print(f"\n🔄 Online Learning Status:")
        print(f"  Model Version: {learning_status['current_version']}")
        print(f"  Performance Trend: {learning_status['performance_trend']}")
        print(f"  Model Health: {learning_status['model_health']}")
        
        return signals
    
    def _generate_ml_report(self, phase2_results: Dict, 
                          ml_signals: List[Dict],
                          ml_training: Dict) -> Dict:
        """
        Generate comprehensive ML-enhanced report
        """
        report = {
            'timestamp': datetime.now(),
            'portfolio_size': self.portfolio_size,
            'phase2_results': phase2_results,
            'ml_enhanced_signals': ml_signals,
            'ml_training_metrics': ml_training,
            'ml_statistics': self._calculate_ml_statistics(ml_signals),
            'recommendations': self._generate_ml_recommendations(ml_signals)
        }
        
        return report
    
    def _calculate_ml_statistics(self, signals: List[Dict]) -> Dict:
        """
        Calculate ML enhancement statistics
        """
        if not signals:
            return {}
        
        ml_probs = [s['ml_prediction']['probability'] for s in signals]
        ml_confidences = [s['ml_prediction']['confidence'] for s in signals]
        final_confidences = [s.get('final_confidence', s['ml_enhanced_confidence']) for s in signals]
        
        return {
            'avg_ml_probability': np.mean(ml_probs),
            'avg_ml_confidence': np.mean(ml_confidences),
            'avg_final_confidence': np.mean(final_confidences),
            'high_confidence_signals': sum(1 for c in final_confidences if c > 75),
            'ml_agreement_rate': sum(1 for s in signals if s.get('online_learning', {}).get('model_agreement', False)) / len(signals),
            'position_adjustments': {
                'increased': sum(1 for s in signals if s.get('ml_position_multiplier', 1) > 1),
                'decreased': sum(1 for s in signals if s.get('ml_position_multiplier', 1) < 1),
                'unchanged': sum(1 for s in signals if s.get('ml_position_multiplier', 1) == 1)
            }
        }
    
    def _generate_ml_recommendations(self, signals: List[Dict]) -> List[str]:
        """
        Generate ML-based recommendations
        """
        recommendations = []
        
        # High confidence signals
        high_conf = [s for s in signals if s.get('final_confidence', 0) > 80]
        if high_conf:
            recommendations.append(
                f"🎯 HIGH CONFIDENCE: {len(high_conf)} signals with >80% confidence - "
                f"consider increased allocation"
            )
        
        # Low confidence warnings
        low_conf = [s for s in signals if s.get('final_confidence', 0) < 50]
        if low_conf:
            recommendations.append(
                f"⚠️ CAUTION: {len(low_conf)} signals with <50% confidence - "
                f"consider reducing or skipping"
            )
        
        # Market stress
        avg_stress = np.mean([s['market_microstructure']['market_stress'] for s in signals])
        if avg_stress > 1.5:
            recommendations.append(
                "🚨 HIGH MARKET STRESS detected - consider reducing overall exposure"
            )
        
        # Model agreement
        agreement_rate = sum(1 for s in signals if s.get('online_learning', {}).get('model_agreement', False)) / len(signals) if signals else 0
        if agreement_rate < 0.5:
            recommendations.append(
                "⚠️ Low model agreement - markets may be in transition, trade cautiously"
            )
        
        return recommendations
    
    def _save_ml_results(self, report: Dict):
        """
        Save ML-enhanced results
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = 'outputs/phase3_ml_enhanced'
        os.makedirs(output_dir, exist_ok=True)
        
        # Save detailed report
        with open(f'{output_dir}/ml_report_{timestamp}.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Save model
        self.ensemble_ml.save_models(f'{output_dir}/ml_models_{timestamp}')
        
        # Generate readable summary
        self._generate_readable_ml_summary(report, output_dir, timestamp)
        
        print(f"\n💾 ML-enhanced results saved to {output_dir}/")
    
    def _generate_readable_ml_summary(self, report: Dict, output_dir: str, timestamp: str):
        """
        Generate human-readable ML summary
        """
        summary = "="*80 + "\n"
        summary += "🤖 PHASE 3 ML-ENHANCED TRADING SYSTEM REPORT\n"
        summary += "="*80 + "\n"
        summary += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        summary += f"Portfolio Size: ${self.portfolio_size:,}\n\n"
        
        # ML Training Performance
        summary += "🎯 ML MODEL PERFORMANCE:\n"
        summary += "-"*60 + "\n"
        
        if 'ensemble_metrics' in report['ml_training_metrics']:
            metrics = report['ml_training_metrics']['ensemble_metrics']
            if 'accuracy' in metrics:
                summary += f"Accuracy: {metrics['accuracy']:.2%}\n"
                summary += f"Precision: {metrics.get('precision', 0):.2%}\n"
                summary += f"Recall: {metrics.get('recall', 0):.2%}\n"
                summary += f"F1 Score: {metrics.get('f1', 0):.2%}\n"
            else:
                summary += f"RMSE: {metrics.get('rmse', 0):.4f}\n"
                summary += f"R² Score: {metrics.get('r2', 0):.4f}\n"
        
        # ML-Enhanced Signals
        summary += "\n📊 ML-ENHANCED TRADING SIGNALS:\n"
        summary += "-"*60 + "\n"
        
        for i, signal in enumerate(report['ml_enhanced_signals'], 1):
            summary += f"\n{i}. {signal['symbol']}\n"
            summary += f"   Original Confidence: {signal.get('confidence', 0):.1f}%\n"
            summary += f"   ML Probability: {signal['ml_prediction']['probability']:.3f}\n"
            summary += f"   ML Confidence: {signal['ml_prediction']['confidence']:.3f}\n"
            summary += f"   Final Confidence: {signal.get('final_confidence', signal['ml_enhanced_confidence']):.1f}%\n"
            summary += f"   Position Adjustment: {signal.get('ml_position_multiplier', 1):.2f}x\n"
            
            # Microstructure insights
            micro = signal['market_microstructure']
            summary += f"   Market Quality:\n"
            summary += f"     - Order Flow: {'BUY' if micro['order_flow_imbalance'] > 0 else 'SELL'} "
            summary += f"({abs(micro['order_flow_imbalance']):.3f})\n"
            summary += f"     - Liquidity: {micro['liquidity_score']:.2f}\n"
            summary += f"     - Stress: {micro['market_stress']:.2f}\n"
        
        # ML Statistics
        if 'ml_statistics' in report:
            stats = report['ml_statistics']
            summary += "\n📈 ML ENHANCEMENT STATISTICS:\n"
            summary += "-"*60 + "\n"
            summary += f"Average ML Probability: {stats.get('avg_ml_probability', 0):.3f}\n"
            summary += f"Average Final Confidence: {stats.get('avg_final_confidence', 0):.1f}%\n"
            summary += f"High Confidence Signals: {stats.get('high_confidence_signals', 0)}\n"
            summary += f"Model Agreement Rate: {stats.get('ml_agreement_rate', 0):.1%}\n"
        
        # Recommendations
        if report.get('recommendations'):
            summary += "\n💡 ML-BASED RECOMMENDATIONS:\n"
            summary += "-"*60 + "\n"
            for rec in report['recommendations']:
                summary += f"\n{rec}\n"
        
        summary += "\n" + "="*80 + "\n"
        summary += "⚠️ DISCLAIMER: This is for educational purposes only. Not financial advice.\n"
        summary += "="*80 + "\n"
        
        # Save summary
        with open(f'{output_dir}/ml_summary_{timestamp}.txt', 'w') as f:
            f.write(summary)
        
        # Print summary
        print(summary)


def main():
    """
    Run the Phase 3 ML-enhanced system
    """
    # Initialize system
    ml_system = Phase3IntegratedMLSystem(portfolio_size=100000)
    
    # Define symbols to analyze
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 
               'META', 'TSLA', 'BRK-B', 'JPM', 'JNJ',
               'V', 'PG', 'UNH', 'HD', 'MA']
    
    # Run ML-enhanced analysis
    results = ml_system.run_ml_enhanced_analysis(symbols)
    
    print("\n✅ Phase 3 ML-Enhanced Analysis Complete!")
    
    if results.get('ml_enhanced_signals'):
        print(f"\nGenerated {len(results['ml_enhanced_signals'])} ML-enhanced signals with:")
        print("  • 300+ engineered features")
        print("  • Ensemble ML predictions (XGBoost, LSTM, RF, LightGBM)")
        print("  • Market microstructure analysis")
        print("  • Online learning adaptations")
        print("  • Confidence-based position sizing")
    else:
        print("\nNo signals generated - review market conditions or ML training data")


if __name__ == "__main__":
    main()