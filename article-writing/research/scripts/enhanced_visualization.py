#!/usr/bin/env python3
"""
Enhanced Fake News Detection Visualization
Comprehensive graphs and curves showing model performance and comparisons
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, learning_curve, validation_curve
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score, accuracy_score
)
from sklearn.preprocessing import label_binarize
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class EnhancedVisualization:
    def __init__(self):
        # Determine project root relative to this script
        # Script is in article-writing/research/scripts/
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_root = os.path.dirname(os.path.dirname(os.path.dirname(self.script_dir)))
        
        self.models = {}
        self.results = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        self.media_dir = os.path.join(self.project_root, 'article-writing/media/general')
        self.reports_dir = os.path.join(self.project_root, 'article-writing/research/reports')
        os.makedirs(self.media_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)
        
    def load_and_prepare_data(self):
        """Load and prepare the dataset"""
        print("📊 Loading and preparing data...")
        
        # Load the cleaned dataset
        data_path = os.path.join(self.project_root, 'data/processed/cleaned_dataset.csv')
        if not os.path.exists(data_path):
            # Try augmented data
            data_path = os.path.join(self.project_root, 'data/augmented/train_news_augmented.csv')
        
        df = pd.read_csv(data_path)
        
        # Prepare features and labels
        X = df['text'].fillna('')
        y = df['label']
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Vectorize the text
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        
        self.X_train_vec = self.vectorizer.fit_transform(self.X_train)
        self.X_test_vec = self.vectorizer.transform(self.X_test)
        
        print(f"✅ Data loaded: {len(self.X_train)} training, {len(self.X_test)} testing samples")
        
    def train_models(self):
        """Train multiple models for comparison"""
        print("🤖 Training multiple models...")
        
        self.models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'Naive Bayes': MultinomialNB(),
            'SVM': SVC(probability=True, random_state=42)
        }
        
        # Train models and collect results
        for name, model in self.models.items():
            print(f"  Training {name}...")
            model.fit(self.X_train_vec, self.y_train)
            
            # Predictions
            y_pred = model.predict(self.X_test_vec)
            y_pred_proba = model.predict_proba(self.X_test_vec)[:, 1]
            
            # Store results
            self.results[name] = {
                'model': model,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'accuracy': accuracy_score(self.y_test, y_pred)
            }
            
        print("✅ All models trained successfully!")
        
    def plot_model_comparison(self):
        """Create comprehensive model comparison visualization"""
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('🎯 Comprehensive Model Performance Comparison', fontsize=20, fontweight='bold')
        
        # 1. Accuracy Comparison Bar Chart
        ax1 = axes[0, 0]
        accuracies = [self.results[name]['accuracy'] for name in self.models.keys()]
        colors = sns.color_palette("viridis", len(self.models))
        bars = ax1.bar(range(len(self.models)), accuracies, color=colors)
        ax1.set_title('📊 Model Accuracy Comparison', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Models')
        ax1.set_ylabel('Accuracy')
        ax1.set_xticks(range(len(self.models)))
        ax1.set_xticklabels(self.models.keys(), rotation=45, ha='right')
        ax1.set_ylim(0.7, 1.0)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. ROC Curves Comparison
        ax2 = axes[0, 1]
        for i, (name, result) in enumerate(self.results.items()):
            fpr, tpr, _ = roc_curve(self.y_test, result['y_pred_proba'])
            roc_auc = auc(fpr, tpr)
            ax2.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC = {roc_auc:.3f})')
        
        ax2.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
        ax2.set_title('📈 ROC Curves Comparison', fontsize=14, fontweight='bold')
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.legend(loc='lower right')
        ax2.grid(True, alpha=0.3)
        
        # 3. Precision-Recall Curves
        ax3 = axes[0, 2]
        for name, result in self.results.items():
            precision, recall, _ = precision_recall_curve(self.y_test, result['y_pred_proba'])
            avg_precision = average_precision_score(self.y_test, result['y_pred_proba'])
            ax3.plot(recall, precision, linewidth=2, label=f'{name} (AP = {avg_precision:.3f})')
        
        ax3.set_title('📊 Precision-Recall Curves', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Recall')
        ax3.set_ylabel('Precision')
        ax3.legend(loc='lower left')
        ax3.grid(True, alpha=0.3)
        
        # 4. Confusion Matrix Heatmap (Best Model)
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['accuracy'])
        best_result = self.results[best_model_name]
        
        ax4 = axes[1, 0]
        cm = confusion_matrix(self.y_test, best_result['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4,
                   xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
        ax4.set_title(f'🎯 Confusion Matrix - {best_model_name}', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Predicted')
        ax4.set_ylabel('Actual')
        
        # 5. Performance Metrics Radar Chart
        ax5 = axes[1, 1]
        metrics_data = []
        model_names = []
        
        for name, result in self.results.items():
            report = classification_report(self.y_test, result['y_pred'], output_dict=True)
            metrics = [
                report['accuracy'],
                report['macro avg']['precision'],
                report['macro avg']['recall'],
                report['macro avg']['f1-score']
            ]
            metrics_data.append(metrics)
            model_names.append(name)
        
        metrics_df = pd.DataFrame(metrics_data, 
                                columns=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                                index=model_names)
        
        # Create grouped bar chart instead of radar
        x = np.arange(len(metrics_df.columns))
        width = 0.15
        
        for i, (model, values) in enumerate(metrics_df.iterrows()):
            ax5.bar(x + i * width, values, width, label=model, alpha=0.8)
        
        ax5.set_title('📊 Performance Metrics Comparison', fontsize=14, fontweight='bold')
        ax5.set_xlabel('Metrics')
        ax5.set_ylabel('Score')
        ax5.set_xticks(x + width * 2)
        ax5.set_xticklabels(metrics_df.columns)
        ax5.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax5.grid(True, alpha=0.3)
        
        # 6. Model Ranking
        ax6 = axes[1, 2]
        ranking_data = sorted([(name, result['accuracy']) for name, result in self.results.items()], 
                            key=lambda x: x[1], reverse=True)
        
        models_ranked = [item[0] for item in ranking_data]
        scores_ranked = [item[1] for item in ranking_data]
        
        colors_ranked = sns.color_palette("RdYlGn", len(models_ranked))
        bars = ax6.barh(range(len(models_ranked)), scores_ranked, color=colors_ranked)
        ax6.set_title('🏆 Model Ranking by Accuracy', fontsize=14, fontweight='bold')
        ax6.set_xlabel('Accuracy Score')
        ax6.set_yticks(range(len(models_ranked)))
        ax6.set_yticklabels(models_ranked)
        
        # Add value labels
        for i, (bar, score) in enumerate(zip(bars, scores_ranked)):
            ax6.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                    f'{score:.3f}', ha='left', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.media_dir, 'comprehensive_model_comparison.png'), dpi=300, bbox_inches='tight')
        print("✅ Model comparison visualization saved!")
        
    def plot_learning_curves(self):
        """Create learning curves for the best models"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('📈 Learning Curves Analysis', fontsize=20, fontweight='bold')
        
        # Select top 4 models for learning curves
        top_models = sorted(self.results.items(), key=lambda x: x[1]['accuracy'], reverse=True)[:4]
        
        for idx, (name, result) in enumerate(top_models):
            ax = axes[idx // 2, idx % 2]
            model = result['model']
            
            # Calculate learning curve
            train_sizes, train_scores, val_scores = learning_curve(
                model, self.X_train_vec, self.y_train,
                cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10),
                scoring='accuracy', random_state=42
            )
            
            # Calculate mean and std
            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            val_mean = np.mean(val_scores, axis=1)
            val_std = np.std(val_scores, axis=1)
            
            # Plot learning curves
            ax.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
            ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                          alpha=0.1, color='blue')
            
            ax.plot(train_sizes, val_mean, 'o-', color='red', label='Validation Score')
            ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, 
                          alpha=0.1, color='red')
            
            ax.set_title(f'📊 {name}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Training Set Size')
            ax.set_ylabel('Accuracy Score')
            ax.legend(loc='lower right')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0.7, 1.0)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.media_dir, 'learning_curves_analysis.png'), dpi=300, bbox_inches='tight')
        print("✅ Learning curves visualization saved!")
        
    def plot_validation_curves(self):
        """Create validation curves for hyperparameter analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('🔧 Validation Curves - Hyperparameter Analysis', fontsize=20, fontweight='bold')
        
        # Random Forest - n_estimators
        ax1 = axes[0, 0]
        param_range = [10, 50, 100, 200, 300]
        train_scores, val_scores = validation_curve(
            RandomForestClassifier(random_state=42), self.X_train_vec, self.y_train,
            param_name='n_estimators', param_range=param_range,
            cv=5, scoring='accuracy', n_jobs=-1
        )
        
        self._plot_validation_curve(ax1, param_range, train_scores, val_scores,
                                  'Random Forest - n_estimators', 'Number of Estimators')
        
        # Gradient Boosting - learning_rate
        ax2 = axes[0, 1]
        param_range = [0.01, 0.05, 0.1, 0.2, 0.3]
        train_scores, val_scores = validation_curve(
            GradientBoostingClassifier(random_state=42), self.X_train_vec, self.y_train,
            param_name='learning_rate', param_range=param_range,
            cv=5, scoring='accuracy', n_jobs=-1
        )
        
        self._plot_validation_curve(ax2, param_range, train_scores, val_scores,
                                  'Gradient Boosting - Learning Rate', 'Learning Rate')
        
        # Logistic Regression - C
        ax3 = axes[1, 0]
        param_range = [0.01, 0.1, 1, 10, 100]
        train_scores, val_scores = validation_curve(
            LogisticRegression(random_state=42, max_iter=1000), self.X_train_vec, self.y_train,
            param_name='C', param_range=param_range,
            cv=5, scoring='accuracy', n_jobs=-1
        )
        
        self._plot_validation_curve(ax3, param_range, train_scores, val_scores,
                                  'Logistic Regression - C Parameter', 'C Parameter')
        ax3.set_xscale('log')
        
        # SVM - C parameter
        ax4 = axes[1, 1]
        param_range = [0.1, 1, 10, 100]
        train_scores, val_scores = validation_curve(
            SVC(random_state=42), self.X_train_vec, self.y_train,
            param_name='C', param_range=param_range,
            cv=3, scoring='accuracy', n_jobs=-1  # Reduced CV for SVM speed
        )
        
        self._plot_validation_curve(ax4, param_range, train_scores, val_scores,
                                  'SVM - C Parameter', 'C Parameter')
        ax4.set_xscale('log')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.media_dir, 'validation_curves_analysis.png'), dpi=300, bbox_inches='tight')
        print("✅ Validation curves visualization saved!")
        
    def _plot_validation_curve(self, ax, param_range, train_scores, val_scores, title, xlabel):
        """Helper function to plot validation curves"""
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        ax.plot(param_range, train_mean, 'o-', color='blue', label='Training Score')
        ax.fill_between(param_range, train_mean - train_std, train_mean + train_std, 
                       alpha=0.1, color='blue')
        
        ax.plot(param_range, val_mean, 'o-', color='red', label='Validation Score')
        ax.fill_between(param_range, val_mean - val_std, val_mean + val_std, 
                       alpha=0.1, color='red')
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Accuracy Score')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
    def create_performance_dashboard(self):
        """Create a comprehensive performance dashboard"""
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        fig.suptitle('🎯 Fake News Detection - Performance Dashboard', fontsize=24, fontweight='bold')
        
        # Best model info
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['accuracy'])
        best_accuracy = self.results[best_model_name]['accuracy']
        
        # Main accuracy display
        ax_main = fig.add_subplot(gs[0, :2])
        ax_main.text(0.5, 0.5, f'{best_accuracy:.1%}', 
                    ha='center', va='center', fontsize=72, fontweight='bold', color='green')
        ax_main.text(0.5, 0.2, f'Best Model: {best_model_name}', 
                    ha='center', va='center', fontsize=16, fontweight='bold')
        ax_main.set_xlim(0, 1)
        ax_main.set_ylim(0, 1)
        ax_main.axis('off')
        ax_main.set_title('🏆 Best Accuracy Achieved', fontsize=18, fontweight='bold')
        
        # Model comparison pie chart
        ax_pie = fig.add_subplot(gs[0, 2:])
        accuracies = [self.results[name]['accuracy'] for name in self.models.keys()]
        colors = sns.color_palette("Set3", len(self.models))
        wedges, texts, autotexts = ax_pie.pie(accuracies, labels=self.models.keys(), 
                                            autopct='%1.1f%%', colors=colors, startangle=90)
        ax_pie.set_title('📊 Model Performance Distribution', fontsize=14, fontweight='bold')
        
        # ROC curves (compact)
        ax_roc = fig.add_subplot(gs[1, :2])
        for name, result in self.results.items():
            fpr, tpr, _ = roc_curve(self.y_test, result['y_pred_proba'])
            roc_auc = auc(fpr, tpr)
            ax_roc.plot(fpr, tpr, linewidth=2, label=f'{name} ({roc_auc:.3f})')
        
        ax_roc.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax_roc.set_title('📈 ROC Curves', fontsize=14, fontweight='bold')
        ax_roc.set_xlabel('False Positive Rate')
        ax_roc.set_ylabel('True Positive Rate')
        ax_roc.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax_roc.grid(True, alpha=0.3)
        
        # Confusion matrix
        ax_cm = fig.add_subplot(gs[1, 2:])
        cm = confusion_matrix(self.y_test, self.results[best_model_name]['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm,
                   xticklabels=['Real News', 'Fake News'], 
                   yticklabels=['Real News', 'Fake News'])
        ax_cm.set_title(f'🎯 Confusion Matrix - {best_model_name}', fontsize=14, fontweight='bold')
        
        # Performance metrics table
        ax_table = fig.add_subplot(gs[2:, :])
        
        # Create detailed metrics table
        table_data = []
        for name, result in self.results.items():
            report = classification_report(self.y_test, result['y_pred'], output_dict=True)
            table_data.append([
                name,
                f"{result['accuracy']:.3f}",
                f"{report['macro avg']['precision']:.3f}",
                f"{report['macro avg']['recall']:.3f}",
                f"{report['macro avg']['f1-score']:.3f}",
                f"{auc(*roc_curve(self.y_test, result['y_pred_proba'])[:2]):.3f}"
            ])
        
        # Sort by accuracy
        table_data.sort(key=lambda x: float(x[1]), reverse=True)
        
        # Create table
        table = ax_table.table(cellText=table_data,
                             colLabels=['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC'],
                             cellLoc='center',
                             loc='center',
                             colWidths=[0.2, 0.15, 0.15, 0.15, 0.15, 0.15])
        
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 2)
        
        # Style the table
        for i in range(len(table_data) + 1):
            for j in range(6):
                cell = table[(i, j)]
                if i == 0:  # Header
                    cell.set_facecolor('#4CAF50')
                    cell.set_text_props(weight='bold', color='white')
                elif i == 1:  # Best model
                    cell.set_facecolor('#E8F5E8')
                    cell.set_text_props(weight='bold')
                else:
                    cell.set_facecolor('#F5F5F5')
        
        ax_table.axis('off')
        ax_table.set_title('📊 Detailed Performance Metrics', fontsize=16, fontweight='bold', pad=20)
        
        plt.savefig(os.path.join(self.media_dir, 'performance_dashboard.png'), dpi=300, bbox_inches='tight')
        print("✅ Performance dashboard saved!")
        
    def generate_summary_report(self):
        """Generate a comprehensive summary report"""
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['accuracy'])
        best_result = self.results[best_model_name]
        
        report = {
            'best_model': best_model_name,
            'best_accuracy': best_result['accuracy'],
            'all_results': {name: result['accuracy'] for name, result in self.results.items()},
            'dataset_info': {
                'training_samples': len(self.X_train),
                'testing_samples': len(self.X_test),
                'features': self.X_train_vec.shape[1]
            }
        }
        
        # Save detailed report
        with open(os.path.join(self.reports_dir, 'enhanced_analysis_report.json'), 'w') as f:
            import json
            json.dump(report, f, indent=2)
        
        print("✅ Enhanced analysis report saved!")
        return report

def main():
    print("🚀 Starting Enhanced Fake News Detection Visualization")
    print("=" * 60)
    
    # Initialize visualization
    viz = EnhancedVisualization()
    
    # Load and prepare data
    viz.load_and_prepare_data()
    
    # Train models
    viz.train_models()
    
    # Create all visualizations
    print("\n📊 Creating comprehensive visualizations...")
    viz.plot_model_comparison()
    viz.plot_learning_curves()
    viz.plot_validation_curves()
    viz.create_performance_dashboard()
    
    # Generate summary report
    report = viz.generate_summary_report()
    
    print("\n" + "=" * 60)
    print("🎉 ENHANCED VISUALIZATION COMPLETE!")
    print("=" * 60)
    print(f"🏆 Best Model: {report['best_model']}")
    print(f"🎯 Best Accuracy: {report['best_accuracy']:.1%}")
    print(f"📊 Total Models Compared: {len(report['all_results'])}")
    print(f"📈 Visualizations Created:")
    print("   • comprehensive_model_comparison.png")
    print("   • learning_curves_analysis.png") 
    print("   • validation_curves_analysis.png")
    print("   • performance_dashboard.png")
    print("   • enhanced_analysis_report.json")
    print("\n🎨 All graphs and curves are ready for analysis!")

if __name__ == "__main__":
    main()