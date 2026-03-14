#!/usr/bin/env python3
"""
Simple Final Test
Recreates the training pipeline to ensure proper evaluation
"""

import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load training and test data"""
    # Determine project root relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))

    # Load augmented data if available
    train_path = os.path.join(project_root, 'data/augmented/train_news_augmented.csv')
    if os.path.exists(train_path):
        train_data = pd.read_csv(train_path)
        print("✓ Loaded augmented training data")
    else:
        # Fallback to original
        train_path = os.path.join(project_root, 'data/train_news.csv')
        train_data = pd.read_csv(train_path)
        print("✓ Loaded original training data")
        
    test_path = os.path.join(project_root, 'data/augmented/test_news_augmented.csv')
    if os.path.exists(test_path):
        test_data = pd.read_csv(test_path)
        print("✓ Loaded augmented test data")
    else:
        # Fallback to original
        test_path = os.path.join(project_root, 'data/test_news.csv')
        test_data = pd.read_csv(test_path)
        print("✓ Loaded original test data")
        
    return train_data, test_data

def create_and_test_models(train_data, test_data):
    """Create and test multiple models"""
    print("\n🔧 Training and Testing Models...")
    print("="*50)
    
    # Prepare data
    X_train = train_data['text']
    y_train = train_data['label']
    X_test = test_data['text']
    y_test = test_data['label']
    
    # Feature extraction - Advanced TF-IDF (best performing)
    vectorizer = TfidfVectorizer(
        max_features=10000,
        stop_words='english',
        ngram_range=(1, 3),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )
    
    print("Extracting features...")
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    print(f"Feature dimensions: {X_train_vec.shape[1]}")
    
    # Define models
    models = {
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        ),
        'Logistic Regression': LogisticRegression(
            random_state=42,
            max_iter=1000,
            C=1.0,
            solver='liblinear'
        ),
        'SVM': SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            probability=True,
            random_state=42
        ),
        'Naive Bayes': MultinomialNB(alpha=0.1),
        'Neural Network': MLPClassifier(
            hidden_layer_sizes=(100, 50),
            max_iter=500,
            random_state=42,
            early_stopping=True
        )
    }
    
    results = {}
    
    # Train and test each model
    for name, model in models.items():
        try:
            print(f"\nTraining {name}...")
            
            # Train model
            model.fit(X_train_vec, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_vec)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Cross-validation on training data
            cv_scores = cross_val_score(model, X_train_vec, y_train, cv=5, scoring='accuracy')
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'predictions': y_pred,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            print(f"  Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            print(f"  CV Accuracy: {cv_scores.mean():.4f} (±{cv_scores.std()*2:.4f})")
            
        except Exception as e:
            print(f"  Error training {name}: {e}")
    
    # Create ensemble from top 3 models
    if len(results) >= 3:
        try:
            print(f"\nCreating Ensemble...")
            
            # Get top 3 models by accuracy
            sorted_models = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
            top_3 = sorted_models[:3]
            
            ensemble_estimators = [(name.lower().replace(' ', '_'), result['model']) 
                                 for name, result in top_3]
            
            ensemble = VotingClassifier(
                estimators=ensemble_estimators,
                voting='soft'
            )
            
            ensemble.fit(X_train_vec, y_train)
            y_pred_ensemble = ensemble.predict(X_test_vec)
            ensemble_accuracy = accuracy_score(y_test, y_pred_ensemble)
            
            cv_scores_ensemble = cross_val_score(ensemble, X_train_vec, y_train, cv=5, scoring='accuracy')
            
            results['Ensemble (Top 3)'] = {
                'model': ensemble,
                'accuracy': ensemble_accuracy,
                'predictions': y_pred_ensemble,
                'cv_mean': cv_scores_ensemble.mean(),
                'cv_std': cv_scores_ensemble.std()
            }
            
            print(f"  Test Accuracy: {ensemble_accuracy:.4f} ({ensemble_accuracy*100:.2f}%)")
            print(f"  CV Accuracy: {cv_scores_ensemble.mean():.4f} (±{cv_scores_ensemble.std()*2:.4f})")
            
        except Exception as e:
            print(f"  Error creating ensemble: {e}")
    
    return results, vectorizer, y_test

def analyze_results(results, y_test):
    """Analyze and display results"""
    print("\n" + "="*80)
    print("🏆 FINAL RESULTS ANALYSIS")
    print("="*80)
    
    # Sort by accuracy
    sorted_results = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    
    print(f"\n📊 MODEL PERFORMANCE RANKING:")
    print("-" * 60)
    for i, (name, result) in enumerate(sorted_results, 1):
        accuracy = result['accuracy']
        cv_mean = result['cv_mean']
        cv_std = result['cv_std']
        print(f"{i:2d}. {name:20s} Test: {accuracy:.4f} ({accuracy*100:6.2f}%) | "
              f"CV: {cv_mean:.4f} (±{cv_std*2:.4f})")
    
    # Best model analysis
    best_name, best_result = sorted_results[0]
    best_accuracy = best_result['accuracy']
    
    print(f"\n🏆 BEST MODEL: {best_name}")
    print(f"   Test Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
    print(f"   CV Accuracy: {best_result['cv_mean']:.4f} (±{best_result['cv_std']*2:.4f})")
    
    # Target analysis
    target = 0.93
    gap = target - best_accuracy
    progress = (best_accuracy / target) * 100
    
    print(f"\n🎯 TARGET ANALYSIS:")
    print("-" * 30)
    print(f"Target Accuracy: {target:.1%}")
    print(f"Best Achieved: {best_accuracy:.1%}")
    print(f"Gap Remaining: {gap:.1%}")
    print(f"Progress: {progress:.1f}%")
    
    if best_accuracy >= target:
        print("✅ TARGET ACHIEVED!")
        status = "SUCCESS"
    else:
        print(f"📈 Need {gap:.1%} improvement")
        if best_accuracy >= 0.80:
            status = "STRONG PERFORMANCE"
        elif best_accuracy >= 0.70:
            status = "GOOD PERFORMANCE"
        else:
            status = "NEEDS IMPROVEMENT"
    
    # Detailed analysis of best model
    print(f"\n📋 DETAILED ANALYSIS - {best_name}:")
    print("-" * 50)
    y_pred = best_result['predictions']
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(f"                 Predicted")
    print(f"                Real  Fake")
    print(f"Actual Real    {cm[0,0]:4d}  {cm[0,1]:4d}")
    print(f"       Fake    {cm[1,0]:4d}  {cm[1,1]:4d}")
    
    return best_name, best_accuracy, status, cm

def create_visualization(results, best_name, best_accuracy, cm):
    """Create final visualization"""
    try:
        # Determine project root relative to this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
        media_dir = os.path.join(project_root, 'article-writing/media/general')
        os.makedirs(media_dir, exist_ok=True)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Model comparison
        names = list(results.keys())
        accuracies = [results[name]['accuracy'] for name in names]
        
        bars = ax1.bar(range(len(names)), accuracies, color='skyblue', alpha=0.7)
        ax1.axhline(y=0.93, color='red', linestyle='--', label='Target (93%)')
        ax1.set_xlabel('Models')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Model Performance Comparison')
        ax1.set_xticks(range(len(names)))
        ax1.set_xticklabels(names, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, acc in zip(bars, accuracies):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{acc:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 2. Progress toward target
        target = 0.93
        progress = (best_accuracy / target) * 100
        remaining = 100 - progress
        
        ax2.pie([progress, remaining], 
               labels=[f'Achieved\n{progress:.1f}%', f'Remaining\n{remaining:.1f}%'],
               colors=['lightgreen', 'lightcoral'],
               autopct='%1.1f%%',
               startangle=90)
        ax2.set_title(f'Progress Toward 93% Target\n(Best: {best_accuracy:.1%})')
        
        # 3. Confusion Matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3,
                   xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
        ax3.set_title(f'Confusion Matrix - {best_name}')
        ax3.set_xlabel('Predicted')
        ax3.set_ylabel('Actual')
        
        # 4. Cross-validation comparison
        cv_means = [results[name]['cv_mean'] for name in names]
        cv_stds = [results[name]['cv_std'] for name in names]
        
        ax4.bar(range(len(names)), cv_means, yerr=[std*2 for std in cv_stds], 
               color='orange', alpha=0.7, capsize=5)
        ax4.set_xlabel('Models')
        ax4.set_ylabel('CV Accuracy')
        ax4.set_title('Cross-Validation Performance')
        ax4.set_xticks(range(len(names)))
        ax4.set_xticklabels(names, rotation=45, ha='right')
        ax4.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        # Save visualization
        output_path = os.path.join(media_dir, 'simple_final_test.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Visualization saved to {output_path}")
        
        plt.show()
        
    except Exception as e:
        print(f"Error creating visualization: {e}")

def save_final_report(best_name, best_accuracy, status, results):
    """Save final report"""
    try:
        import json
        import os
        
        # Determine project root relative to this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
        reports_dir = os.path.join(project_root, 'article-writing/research/reports')
        os.makedirs(reports_dir, exist_ok=True)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'best_model': best_name,
            'best_accuracy': float(best_accuracy),
            'target_accuracy': 0.93,
            'gap': float(0.93 - best_accuracy),
            'progress_percentage': float((best_accuracy / 0.93) * 100),
            'status': status,
            'all_results': {
                name: {
                    'accuracy': float(result['accuracy']),
                    'cv_mean': float(result['cv_mean']),
                    'cv_std': float(result['cv_std'])
                }
                for name, result in results.items()
            }
        }
        
        output_path = os.path.join(reports_dir, 'simple_final_report.json')
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        print(f"✓ Final report saved to {output_path}")
        
    except Exception as e:
        print(f"Error saving report: {e}")

def main():
    """Main function"""
    print("🚀 Simple Final Test - Fake News Detection")
    print("="*60)
    
    # Load data
    train_data, test_data = load_data()
    print(f"Training samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")
    
    # Train and test models
    results, vectorizer, y_test = create_and_test_models(train_data, test_data)
    
    if not results:
        print("❌ No models were successfully trained")
        return
    
    # Analyze results
    best_name, best_accuracy, status, cm = analyze_results(results, y_test)
    
    # Create visualization
    create_visualization(results, best_name, best_accuracy, cm)
    
    # Save report
    save_final_report(best_name, best_accuracy, status, results)
    
    print(f"\n🏁 FINAL STATUS: {status}")
    print("="*60)
    print("✅ Simple final test completed!")

if __name__ == "__main__":
    main()