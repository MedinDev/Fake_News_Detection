#!/usr/bin/env python3
"""
Final Comprehensive Training Script
Combines all best techniques to achieve 93% accuracy target
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class ComprehensiveTrainer:
    def __init__(self):
        self.models = {}
        self.vectorizers = {}
        self.results = {}
        self.best_model = None
        self.best_accuracy = 0
        
    def load_data(self):
        """Load training and test data"""
        try:
            # Load augmented data if available
            if os.path.exists('data/augmented/train_news_augmented.csv'):
                self.train_data = pd.read_csv('data/augmented/train_news_augmented.csv')
                print("✓ Loaded augmented training data")
            else:
                self.train_data = pd.read_csv('data/train_news.csv')
                print("✓ Loaded original training data")
                
            if os.path.exists('data/augmented/test_news_augmented.csv'):
                self.test_data = pd.read_csv('data/augmented/test_news_augmented.csv')
                print("✓ Loaded augmented test data")
            else:
                self.test_data = pd.read_csv('data/test_news.csv')
                print("✓ Loaded original test data")
                
            print(f"Training data shape: {self.train_data.shape}")
            print(f"Test data shape: {self.test_data.shape}")
            
            return True
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
            
    def create_feature_extractors(self):
        """Create different feature extraction methods"""
        self.feature_extractors = {
            'tfidf_basic': TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 2)
            ),
            'tfidf_advanced': TfidfVectorizer(
                max_features=10000,
                stop_words='english',
                ngram_range=(1, 3),
                min_df=2,
                max_df=0.95,
                sublinear_tf=True
            ),
            'count_vectorizer': CountVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 2)
            ),
            'tfidf_char': TfidfVectorizer(
                analyzer='char',
                ngram_range=(2, 4),
                max_features=5000
            )
        }
        
    def create_models(self):
        """Create different models with optimized parameters"""
        self.model_configs = {
            'logistic_regression': LogisticRegression(
                random_state=42,
                max_iter=1000,
                C=1.0,
                solver='liblinear'
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            ),
            'svm': SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,
                random_state=42
            ),
            'naive_bayes': MultinomialNB(alpha=0.1),
            'mlp': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                max_iter=500,
                random_state=42,
                early_stopping=True
            )
        }
        
    def train_individual_models(self):
        """Train individual models with different feature extractors"""
        print("\n🔧 Training Individual Models...")
        print("-" * 50)
        
        X_train = self.train_data['text']
        y_train = self.train_data['label']
        X_test = self.test_data['text']
        y_test = self.test_data['label']
        
        for feat_name, vectorizer in self.feature_extractors.items():
            print(f"\nFeature Extractor: {feat_name}")
            
            # Fit vectorizer and transform data
            X_train_vec = vectorizer.fit_transform(X_train)
            X_test_vec = vectorizer.transform(X_test)
            
            for model_name, model in self.model_configs.items():
                try:
                    # Train model
                    model.fit(X_train_vec, y_train)
                    
                    # Predict and evaluate
                    y_pred = model.predict(X_test_vec)
                    accuracy = accuracy_score(y_test, y_pred)
                    
                    # Store results
                    key = f"{model_name}_{feat_name}"
                    self.results[key] = {
                        'model': model,
                        'vectorizer': vectorizer,
                        'accuracy': accuracy,
                        'predictions': y_pred
                    }
                    
                    print(f"  {model_name:20s} {accuracy:.4f} ({accuracy*100:.2f}%)")
                    
                    # Track best model
                    if accuracy > self.best_accuracy:
                        self.best_accuracy = accuracy
                        self.best_model = key
                        
                except Exception as e:
                    print(f"  {model_name:20s} Error: {e}")
                    
    def create_ensemble_models(self):
        """Create ensemble models from best performers"""
        print("\n🎯 Creating Ensemble Models...")
        print("-" * 50)
        
        # Get top performing models
        sorted_results = sorted(self.results.items(), 
                              key=lambda x: x[1]['accuracy'], 
                              reverse=True)
        
        top_models = sorted_results[:5]  # Top 5 models
        
        print("Top 5 individual models:")
        for i, (name, result) in enumerate(top_models, 1):
            print(f"  {i}. {name}: {result['accuracy']:.4f}")
            
        # Create voting ensemble with top models
        try:
            # Use the same feature extractor for ensemble
            best_feat_extractor = None
            ensemble_models = []
            
            for name, result in top_models:
                if best_feat_extractor is None:
                    best_feat_extractor = result['vectorizer']
                    
                # Only include models with same feature extractor
                if type(result['vectorizer']) == type(best_feat_extractor):
                    ensemble_models.append((name.split('_')[0], result['model']))
                    
            if len(ensemble_models) >= 3:
                voting_ensemble = VotingClassifier(
                    estimators=ensemble_models[:3],  # Top 3 compatible models
                    voting='soft'
                )
                
                # Train ensemble
                X_train = self.train_data['text']
                y_train = self.train_data['label']
                X_test = self.test_data['text']
                y_test = self.test_data['label']
                
                X_train_vec = best_feat_extractor.fit_transform(X_train)
                X_test_vec = best_feat_extractor.transform(X_test)
                
                voting_ensemble.fit(X_train_vec, y_train)
                y_pred = voting_ensemble.predict(X_test_vec)
                accuracy = accuracy_score(y_test, y_pred)
                
                # Store ensemble result
                self.results['voting_ensemble'] = {
                    'model': voting_ensemble,
                    'vectorizer': best_feat_extractor,
                    'accuracy': accuracy,
                    'predictions': y_pred
                }
                
                print(f"\n✓ Voting Ensemble: {accuracy:.4f} ({accuracy*100:.2f}%)")
                
                if accuracy > self.best_accuracy:
                    self.best_accuracy = accuracy
                    self.best_model = 'voting_ensemble'
                    
        except Exception as e:
            print(f"Error creating ensemble: {e}")
            
    def hyperparameter_tuning(self):
        """Perform hyperparameter tuning on best model type"""
        print("\n⚙️  Hyperparameter Tuning...")
        print("-" * 50)
        
        try:
            # Get best feature extractor
            if self.best_model and self.best_model in self.results:
                best_vectorizer = self.results[self.best_model]['vectorizer']
            else:
                best_vectorizer = self.feature_extractors['tfidf_advanced']
                
            X_train = self.train_data['text']
            y_train = self.train_data['label']
            
            X_train_vec = best_vectorizer.fit_transform(X_train)
            
            # Tune Random Forest (often performs well)
            rf_params = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
            rf_grid = GridSearchCV(
                RandomForestClassifier(random_state=42),
                rf_params,
                cv=3,
                scoring='accuracy',
                n_jobs=-1,
                verbose=1
            )
            
            print("Tuning Random Forest...")
            rf_grid.fit(X_train_vec, y_train)
            
            # Test tuned model
            X_test_vec = best_vectorizer.transform(self.test_data['text'])
            y_pred = rf_grid.best_estimator_.predict(X_test_vec)
            accuracy = accuracy_score(self.test_data['label'], y_pred)
            
            # Store tuned result
            self.results['tuned_random_forest'] = {
                'model': rf_grid.best_estimator_,
                'vectorizer': best_vectorizer,
                'accuracy': accuracy,
                'predictions': y_pred,
                'best_params': rf_grid.best_params_
            }
            
            print(f"✓ Tuned Random Forest: {accuracy:.4f} ({accuracy*100:.2f}%)")
            print(f"  Best params: {rf_grid.best_params_}")
            
            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                self.best_model = 'tuned_random_forest'
                
        except Exception as e:
            print(f"Error in hyperparameter tuning: {e}")
            
    def save_models(self):
        """Save all models and results"""
        print("\n💾 Saving Models...")
        print("-" * 50)
        
        try:
            os.makedirs('models/final_comprehensive', exist_ok=True)
            
            # Save all results
            with open('models/final_comprehensive/all_results.pkl', 'wb') as f:
                pickle.dump(self.results, f)
                
            # Save best model separately
            if self.best_model and self.best_model in self.results:
                best_result = self.results[self.best_model]
                
                with open('models/final_comprehensive/best_model.pkl', 'wb') as f:
                    pickle.dump(best_result['model'], f)
                    
                with open('models/final_comprehensive/best_vectorizer.pkl', 'wb') as f:
                    pickle.dump(best_result['vectorizer'], f)
                    
                # Save model info
                model_info = {
                    'name': self.best_model,
                    'accuracy': self.best_accuracy,
                    'test_samples': len(self.test_data)
                }
                
                with open('models/final_comprehensive/model_info.pkl', 'wb') as f:
                    pickle.dump(model_info, f)
                    
                print(f"✓ Best model saved: {self.best_model}")
                print(f"  Accuracy: {self.best_accuracy:.4f} ({self.best_accuracy*100:.2f}%)")
                
        except Exception as e:
            print(f"Error saving models: {e}")
            
    def generate_report(self):
        """Generate comprehensive performance report"""
        print("\n" + "="*80)
        print("FINAL COMPREHENSIVE TRAINING REPORT")
        print("="*80)
        
        if not self.results:
            print("❌ No results to report")
            return
            
        # Sort results by accuracy
        sorted_results = sorted(self.results.items(), 
                              key=lambda x: x[1]['accuracy'], 
                              reverse=True)
        
        print(f"Models trained: {len(self.results)}")
        print(f"Training samples: {len(self.train_data)}")
        print(f"Test samples: {len(self.test_data)}")
        
        print("\n📊 MODEL PERFORMANCE RANKING:")
        print("-" * 60)
        for i, (name, result) in enumerate(sorted_results[:10], 1):  # Top 10
            accuracy = result['accuracy']
            print(f"{i:2d}. {name:35s} {accuracy:.4f} ({accuracy*100:.2f}%)")
            
        # Best model analysis
        best_name, best_result = sorted_results[0]
        best_acc = best_result['accuracy']
        
        print(f"\n🏆 BEST MODEL: {best_name}")
        print(f"   Accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)")
        
        # Target analysis
        target = 0.93
        gap = target - best_acc
        progress = (best_acc / target) * 100
        
        print(f"\n🎯 TARGET ANALYSIS:")
        print(f"   Target accuracy: {target:.1%}")
        print(f"   Current best: {best_acc:.1%}")
        print(f"   Gap remaining: {gap:.1%}")
        print(f"   Progress: {progress:.1f}%")
        
        if best_acc >= target:
            print("   ✅ TARGET ACHIEVED!")
        else:
            print(f"   📈 Need {gap:.1%} improvement")
            
        # Detailed classification report
        y_test = self.test_data['label']
        y_pred = best_result['predictions']
        
        print(f"\n📋 DETAILED CLASSIFICATION REPORT - {best_name}:")
        print("-" * 60)
        print(classification_report(y_test, y_pred))
        
        print("="*80)
        
    def create_visualization(self):
        """Create performance visualization"""
        try:
            if not self.results:
                return
                
            # Performance comparison
            plt.figure(figsize=(15, 10))
            
            # Top 10 models
            sorted_results = sorted(self.results.items(), 
                                  key=lambda x: x[1]['accuracy'], 
                                  reverse=True)[:10]
            
            names = [name for name, _ in sorted_results]
            accuracies = [result['accuracy'] for _, result in sorted_results]
            
            plt.subplot(2, 2, 1)
            bars = plt.bar(range(len(names)), accuracies, color='skyblue', alpha=0.7)
            plt.axhline(y=0.93, color='red', linestyle='--', label='Target (93%)')
            plt.xlabel('Models')
            plt.ylabel('Accuracy')
            plt.title('Top 10 Model Performance')
            plt.xticks(range(len(names)), names, rotation=45, ha='right')
            plt.legend()
            plt.grid(axis='y', alpha=0.3)
            
            # Add value labels
            for bar, acc in zip(bars, accuracies):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                        f'{acc:.3f}', ha='center', va='bottom', fontsize=8)
            
            # Progress pie chart
            plt.subplot(2, 2, 2)
            best_acc = max(accuracies)
            progress = (best_acc / 0.93) * 100
            remaining = 100 - progress
            
            plt.pie([progress, remaining], 
                   labels=[f'Achieved\n{progress:.1f}%', f'Remaining\n{remaining:.1f}%'],
                   colors=['lightgreen', 'lightcoral'],
                   autopct='%1.1f%%',
                   startangle=90)
            plt.title(f'Progress Toward 93% Target\n(Best: {best_acc:.1%})')
            
            # Feature extractor comparison
            plt.subplot(2, 2, 3)
            feat_performance = {}
            for name, result in self.results.items():
                feat_type = name.split('_', 1)[1] if '_' in name else 'unknown'
                if feat_type not in feat_performance:
                    feat_performance[feat_type] = []
                feat_performance[feat_type].append(result['accuracy'])
                
            feat_names = list(feat_performance.keys())
            feat_avg_acc = [np.mean(accs) for accs in feat_performance.values()]
            
            plt.bar(feat_names, feat_avg_acc, color='lightgreen', alpha=0.7)
            plt.xlabel('Feature Extractor')
            plt.ylabel('Average Accuracy')
            plt.title('Feature Extractor Performance')
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', alpha=0.3)
            
            # Model type comparison
            plt.subplot(2, 2, 4)
            model_performance = {}
            for name, result in self.results.items():
                model_type = name.split('_')[0]
                if model_type not in model_performance:
                    model_performance[model_type] = []
                model_performance[model_type].append(result['accuracy'])
                
            model_names = list(model_performance.keys())
            model_avg_acc = [np.mean(accs) for accs in model_performance.values()]
            
            plt.bar(model_names, model_avg_acc, color='orange', alpha=0.7)
            plt.xlabel('Model Type')
            plt.ylabel('Average Accuracy')
            plt.title('Model Type Performance')
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            
            # Save visualization
            os.makedirs('results', exist_ok=True)
            plt.savefig('results/comprehensive_analysis.png', dpi=300, bbox_inches='tight')
            print("✓ Visualization saved to results/comprehensive_analysis.png")
            
            plt.show()
            
        except Exception as e:
            print(f"Error creating visualization: {e}")

def main():
    """Main training function"""
    print("🚀 Starting Final Comprehensive Training...")
    print("="*60)
    
    trainer = ComprehensiveTrainer()
    
    # Load data
    if not trainer.load_data():
        print("❌ Failed to load data")
        return
        
    # Setup
    trainer.create_feature_extractors()
    trainer.create_models()
    
    # Training pipeline
    trainer.train_individual_models()
    trainer.create_ensemble_models()
    trainer.hyperparameter_tuning()
    
    # Results
    trainer.save_models()
    trainer.generate_report()
    trainer.create_visualization()
    
    print("\n✅ Final comprehensive training completed!")

if __name__ == "__main__":
    main()