#!/usr/bin/env python3
"""
Advanced Deep Analysis for Fake News Detection
More sophisticated visualizations and statistical analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score, accuracy_score,
    f1_score, precision_score, recall_score
)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import scipy.stats as stats
from wordcloud import WordCloud
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# Enhanced styling
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class AdvancedDeepAnalysis:
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
        self.feature_names = None
        
        self.media_dir = os.path.join(self.project_root, 'article-writing/media/general')
        self.reports_dir = os.path.join(self.project_root, 'article-writing/research/reports')
        os.makedirs(self.media_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)
        
    def load_and_prepare_data(self):
        """Load and prepare the dataset with enhanced preprocessing"""
        print("🔍 Loading and preparing data for deep analysis...")
        
        # Load the cleaned dataset
        data_path = os.path.join(self.project_root, 'data/processed/cleaned_dataset.csv')
        if not os.path.exists(data_path):
            data_path = os.path.join(self.project_root, 'data/augmented/train_news_augmented.csv')
        
        df = pd.read_csv(data_path)
        
        # Prepare features and labels
        X = df['text'].fillna('')
        y = df['label']
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Enhanced vectorization with more features
        self.vectorizer = TfidfVectorizer(
            max_features=15000,
            stop_words='english',
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True
        )
        
        self.X_train_vec = self.vectorizer.fit_transform(self.X_train)
        self.X_test_vec = self.vectorizer.transform(self.X_test)
        self.feature_names = self.vectorizer.get_feature_names_out()
        
        print(f"✅ Enhanced data loaded: {len(self.X_train)} training, {len(self.X_test)} testing samples")
        print(f"📊 Features extracted: {len(self.feature_names)}")
        
    def train_enhanced_models(self):
        """Train enhanced models with optimized parameters"""
        print("🚀 Training enhanced models with optimized parameters...")
        
        self.models = {
            'Optimized Logistic Regression': LogisticRegression(
                C=10, max_iter=2000, random_state=42, solver='liblinear'
            ),
            'Enhanced Random Forest': RandomForestClassifier(
                n_estimators=200, max_depth=15, min_samples_split=5,
                min_samples_leaf=2, random_state=42, n_jobs=-1
            ),
            'Tuned Gradient Boosting': GradientBoostingClassifier(
                n_estimators=150, learning_rate=0.1, max_depth=6, random_state=42
            ),
            'Advanced Naive Bayes': MultinomialNB(alpha=0.1),
            'Optimized SVM': SVC(
                C=10, kernel='rbf', probability=True, random_state=42
            )
        }
        
        # Train models and collect enhanced results
        for name, model in self.models.items():
            print(f"  🔧 Training {name}...")
            model.fit(self.X_train_vec, self.y_train)
            
            # Predictions
            y_pred = model.predict(self.X_test_vec)
            y_pred_proba = model.predict_proba(self.X_test_vec)[:, 1]
            
            # Cross-validation scores
            cv_scores = cross_val_score(model, self.X_train_vec, self.y_train, 
                                      cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                                      scoring='accuracy', n_jobs=-1)
            
            # Store enhanced results
            self.results[name] = {
                'model': model,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'accuracy': accuracy_score(self.y_test, y_pred),
                'precision': precision_score(self.y_test, y_pred),
                'recall': recall_score(self.y_test, y_pred),
                'f1': f1_score(self.y_test, y_pred),
                'cv_scores': cv_scores,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
        print("✅ All enhanced models trained successfully!")
        
    def create_feature_importance_analysis(self):
        """Analyze and visualize feature importance"""
        fig, axes = plt.subplots(2, 3, figsize=(24, 16))
        fig.suptitle('🔍 Feature Importance Deep Analysis', fontsize=24, fontweight='bold')
        
        # Get models that have feature importance
        importance_models = {
            name: result for name, result in self.results.items() 
            if hasattr(result['model'], 'feature_importances_') or hasattr(result['model'], 'coef_')
        }
        
        plot_idx = 0
        for name, result in list(importance_models.items())[:6]:
            ax = axes[plot_idx // 3, plot_idx % 3]
            model = result['model']
            
            if hasattr(model, 'feature_importances_'):
                # Tree-based models
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                # Linear models
                importances = np.abs(model.coef_[0])
            else:
                continue
                
            # Get top 20 features
            top_indices = np.argsort(importances)[-20:]
            top_features = [self.feature_names[i] for i in top_indices]
            top_importances = importances[top_indices]
            
            # Create horizontal bar plot
            colors = plt.cm.viridis(np.linspace(0, 1, len(top_features)))
            bars = ax.barh(range(len(top_features)), top_importances, color=colors)
            ax.set_yticks(range(len(top_features)))
            ax.set_yticklabels(top_features, fontsize=10)
            ax.set_title(f'🎯 {name}', fontsize=14, fontweight='bold')
            ax.set_xlabel('Feature Importance')
            
            # Add value labels
            for i, (bar, imp) in enumerate(zip(bars, top_importances)):
                ax.text(bar.get_width() + max(top_importances) * 0.01, 
                       bar.get_y() + bar.get_height()/2,
                       f'{imp:.4f}', ha='left', va='center', fontsize=8)
            
            plot_idx += 1
        
        # Remove empty subplots
        for i in range(plot_idx, 6):
            if i // 3 < 2 and i % 3 < 3:
                fig.delaxes(axes[i // 3, i % 3])
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.media_dir, 'feature_importance_analysis.png'), dpi=300, bbox_inches='tight')
        print("✅ Feature importance analysis saved!")
        
    def create_statistical_comparison(self):
        """Create statistical comparison and significance tests"""
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('📊 Statistical Analysis & Model Comparison', fontsize=24, fontweight='bold')
        
        # 1. Cross-validation scores distribution
        ax1 = axes[0, 0]
        cv_data = []
        model_names = []
        
        for name, result in self.results.items():
            cv_data.extend(result['cv_scores'])
            model_names.extend([name] * len(result['cv_scores']))
        
        cv_df = pd.DataFrame({'Model': model_names, 'CV_Score': cv_data})
        
        sns.boxplot(data=cv_df, x='Model', y='CV_Score', ax=ax1)
        ax1.set_title('📈 Cross-Validation Score Distribution', fontsize=16, fontweight='bold')
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
        ax1.set_ylabel('Accuracy Score')
        ax1.grid(True, alpha=0.3)
        
        # 2. Performance metrics heatmap
        ax2 = axes[0, 1]
        metrics_data = []
        
        for name, result in self.results.items():
            metrics_data.append([
                result['accuracy'],
                result['precision'],
                result['recall'],
                result['f1'],
                result['cv_mean']
            ])
        
        metrics_df = pd.DataFrame(
            metrics_data,
            columns=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'CV Mean'],
            index=list(self.results.keys())
        )
        
        sns.heatmap(metrics_df, annot=True, fmt='.3f', cmap='RdYlGn', 
                   ax=ax2, cbar_kws={'label': 'Score'})
        ax2.set_title('🎯 Performance Metrics Heatmap', fontsize=16, fontweight='bold')
        ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0)
        
        # 3. Statistical significance tests
        ax3 = axes[1, 0]
        
        # Perform pairwise t-tests
        model_list = list(self.results.keys())
        p_values = np.ones((len(model_list), len(model_list)))
        
        for i, model1 in enumerate(model_list):
            for j, model2 in enumerate(model_list):
                if i != j:
                    scores1 = self.results[model1]['cv_scores']
                    scores2 = self.results[model2]['cv_scores']
                    _, p_val = stats.ttest_rel(scores1, scores2)
                    p_values[i, j] = p_val
        
        # Create significance heatmap
        mask = np.triu(np.ones_like(p_values, dtype=bool))
        sns.heatmap(p_values, mask=mask, annot=True, fmt='.3f', cmap='RdYlBu_r',
                   xticklabels=[m.split()[-2:] for m in model_list],
                   yticklabels=[m.split()[-2:] for m in model_list],
                   ax=ax3, cbar_kws={'label': 'p-value'})
        ax3.set_title('🔬 Statistical Significance (p-values)', fontsize=16, fontweight='bold')
        
        # 4. Model stability analysis
        ax4 = axes[1, 1]
        
        stability_data = []
        for name, result in self.results.items():
            stability_data.append({
                'Model': name.split()[-2:],  # Shortened name
                'Mean': result['cv_mean'],
                'Std': result['cv_std'],
                'Stability': 1 / (result['cv_std'] + 0.001)  # Higher is more stable
            })
        
        stability_df = pd.DataFrame(stability_data)
        
        # Create scatter plot
        scatter = ax4.scatter(stability_df['Mean'], stability_df['Stability'], 
                            s=200, alpha=0.7, c=range(len(stability_df)), cmap='viridis')
        
        for i, row in stability_df.iterrows():
            ax4.annotate(' '.join(row['Model']), 
                        (row['Mean'], row['Stability']),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=10, fontweight='bold')
        
        ax4.set_xlabel('Mean CV Accuracy')
        ax4.set_ylabel('Stability Score (1/std)')
        ax4.set_title('🎯 Model Performance vs Stability', fontsize=16, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.media_dir, 'statistical_comparison.png'), dpi=300, bbox_inches='tight')
        print("✅ Statistical comparison saved!")
        
    def create_dimensionality_analysis(self):
        """Create dimensionality reduction and clustering analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('🔬 Dimensionality Reduction & Clustering Analysis', fontsize=24, fontweight='bold')
        
        # Prepare data for visualization (sample for speed)
        n_samples = min(1000, self.X_train_vec.shape[0])
        indices = np.random.choice(self.X_train_vec.shape[0], n_samples, replace=False)
        X_sample = self.X_train_vec[indices].toarray()
        y_sample = self.y_train.iloc[indices]
        
        # 1. PCA Analysis
        ax1 = axes[0, 0]
        pca = PCA(n_components=50)
        X_pca = pca.fit_transform(X_sample)
        
        # Plot explained variance
        cumsum_var = np.cumsum(pca.explained_variance_ratio_)
        ax1.plot(range(1, 51), cumsum_var, 'bo-', linewidth=2, markersize=4)
        ax1.axhline(y=0.95, color='r', linestyle='--', label='95% Variance')
        ax1.set_xlabel('Number of Components')
        ax1.set_ylabel('Cumulative Explained Variance')
        ax1.set_title('📊 PCA - Explained Variance', fontsize=16, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. PCA 2D Visualization
        ax2 = axes[0, 1]
        pca_2d = PCA(n_components=2)
        X_pca_2d = pca_2d.fit_transform(X_sample)
        
        colors = ['red' if label == 1 else 'blue' for label in y_sample]
        scatter = ax2.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=colors, alpha=0.6, s=20)
        ax2.set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.2%} variance)')
        ax2.set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.2%} variance)')
        ax2.set_title('🎯 PCA - 2D Projection', fontsize=16, fontweight='bold')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='red', label='Fake News'),
                          Patch(facecolor='blue', label='Real News')]
        ax2.legend(handles=legend_elements)
        
        # 3. t-SNE Visualization
        ax3 = axes[1, 0]
        
        # Use smaller sample for t-SNE (it's computationally expensive)
        n_tsne = min(500, len(X_sample))
        tsne_indices = np.random.choice(len(X_sample), n_tsne, replace=False)
        X_tsne_sample = X_sample[tsne_indices]
        y_tsne_sample = y_sample.iloc[tsne_indices]
        
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        X_tsne = tsne.fit_transform(X_tsne_sample)
        
        colors_tsne = ['red' if label == 1 else 'blue' for label in y_tsne_sample]
        ax3.scatter(X_tsne[:, 0], X_tsne[:, 1], c=colors_tsne, alpha=0.6, s=20)
        ax3.set_xlabel('t-SNE 1')
        ax3.set_ylabel('t-SNE 2')
        ax3.set_title('🔍 t-SNE - Non-linear Projection', fontsize=16, fontweight='bold')
        ax3.legend(handles=legend_elements)
        
        # 4. Clustering Analysis
        ax4 = axes[1, 1]
        
        # K-means clustering on PCA-reduced data
        kmeans = KMeans(n_clusters=2, random_state=42)
        cluster_labels = kmeans.fit_predict(X_pca_2d)
        
        # Calculate clustering accuracy
        from sklearn.metrics import adjusted_rand_score
        ari_score = adjusted_rand_score(y_sample, cluster_labels)
        
        scatter = ax4.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], 
                            c=cluster_labels, cmap='viridis', alpha=0.6, s=20)
        
        # Plot cluster centers
        centers = kmeans.cluster_centers_
        ax4.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=200, linewidths=3)
        
        ax4.set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.2%} variance)')
        ax4.set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.2%} variance)')
        ax4.set_title(f'🎯 K-Means Clustering (ARI: {ari_score:.3f})', fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.media_dir, 'dimensionality_analysis.png'), dpi=300, bbox_inches='tight')
        print("✅ Dimensionality analysis saved!")
        
    def create_word_importance_visualization(self):
        """Create word clouds and importance visualizations"""
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('📝 Word Importance & Text Analysis', fontsize=24, fontweight='bold')
        
        # Get the best model for feature analysis
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['accuracy'])
        best_model = self.results[best_model_name]['model']
        
        # 1. Word Cloud for Fake News
        ax1 = axes[0, 0]
        fake_texts = ' '.join(self.X_train[self.y_train == 1].values)
        
        wordcloud_fake = WordCloud(
            width=800, height=400, 
            background_color='white',
            colormap='Reds',
            max_words=100
        ).generate(fake_texts)
        
        ax1.imshow(wordcloud_fake, interpolation='bilinear')
        ax1.axis('off')
        ax1.set_title('☠️ Fake News - Word Cloud', fontsize=16, fontweight='bold')
        
        # 2. Word Cloud for Real News
        ax2 = axes[0, 1]
        real_texts = ' '.join(self.X_train[self.y_train == 0].values)
        
        wordcloud_real = WordCloud(
            width=800, height=400,
            background_color='white',
            colormap='Blues',
            max_words=100
        ).generate(real_texts)
        
        ax2.imshow(wordcloud_real, interpolation='bilinear')
        ax2.axis('off')
        ax2.set_title('✅ Real News - Word Cloud', fontsize=16, fontweight='bold')
        
        # 3. Most Discriminative Features
        ax3 = axes[1, 0]
        
        if hasattr(best_model, 'coef_'):
            # For linear models, get coefficients
            coef = best_model.coef_[0]
            
            # Get most positive (fake news indicators)
            fake_indices = np.argsort(coef)[-15:]
            fake_features = [self.feature_names[i] for i in fake_indices]
            fake_scores = coef[fake_indices]
            
            bars = ax3.barh(range(len(fake_features)), fake_scores, color='red', alpha=0.7)
            ax3.set_yticks(range(len(fake_features)))
            ax3.set_yticklabels(fake_features, fontsize=10)
            ax3.set_title('🚨 Top Fake News Indicators', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Feature Coefficient')
            
            # Add value labels
            for bar, score in zip(bars, fake_scores):
                ax3.text(bar.get_width() + max(fake_scores) * 0.01,
                        bar.get_y() + bar.get_height()/2,
                        f'{score:.3f}', ha='left', va='center', fontsize=8)
        
        # 4. Real News Indicators
        ax4 = axes[1, 1]
        
        if hasattr(best_model, 'coef_'):
            # Get most negative (real news indicators)
            real_indices = np.argsort(coef)[:15]
            real_features = [self.feature_names[i] for i in real_indices]
            real_scores = np.abs(coef[real_indices])  # Take absolute values for visualization
            
            bars = ax4.barh(range(len(real_features)), real_scores, color='blue', alpha=0.7)
            ax4.set_yticks(range(len(real_features)))
            ax4.set_yticklabels(real_features, fontsize=10)
            ax4.set_title('✅ Top Real News Indicators', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Feature Coefficient (Absolute)')
            
            # Add value labels
            for bar, score in zip(bars, real_scores):
                ax4.text(bar.get_width() + max(real_scores) * 0.01,
                        bar.get_y() + bar.get_height()/2,
                        f'{score:.3f}', ha='left', va='center', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.media_dir, 'word_importance_analysis.png'), dpi=300, bbox_inches='tight')
        print("✅ Word importance analysis saved!")
        
    def create_ensemble_analysis(self):
        """Create ensemble model analysis"""
        print("🤖 Creating ensemble model analysis...")
        
        # Create ensemble model
        ensemble = VotingClassifier(
            estimators=[
                ('lr', self.results['Optimized Logistic Regression']['model']),
                ('rf', self.results['Enhanced Random Forest']['model']),
                ('gb', self.results['Tuned Gradient Boosting']['model'])
            ],
            voting='soft'
        )
        
        ensemble.fit(self.X_train_vec, self.y_train)
        
        # Ensemble predictions
        y_pred_ensemble = ensemble.predict(self.X_test_vec)
        y_pred_proba_ensemble = ensemble.predict_proba(self.X_test_vec)[:, 1]
        
        # Add ensemble to results
        self.results['Ensemble Model'] = {
            'model': ensemble,
            'y_pred': y_pred_ensemble,
            'y_pred_proba': y_pred_proba_ensemble,
            'accuracy': accuracy_score(self.y_test, y_pred_ensemble),
            'precision': precision_score(self.y_test, y_pred_ensemble),
            'recall': recall_score(self.y_test, y_pred_ensemble),
            'f1': f1_score(self.y_test, y_pred_ensemble)
        }
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('🎯 Ensemble Model Analysis', fontsize=24, fontweight='bold')
        
        # 1. Individual vs Ensemble Performance
        ax1 = axes[0, 0]
        
        model_names = list(self.results.keys())
        accuracies = [self.results[name]['accuracy'] for name in model_names]
        
        colors = ['gold' if 'Ensemble' in name else 'skyblue' for name in model_names]
        bars = ax1.bar(range(len(model_names)), accuracies, color=colors, alpha=0.8)
        
        ax1.set_title('🏆 Individual vs Ensemble Performance', fontsize=16, fontweight='bold')
        ax1.set_xlabel('Models')
        ax1.set_ylabel('Accuracy')
        ax1.set_xticks(range(len(model_names)))
        ax1.set_xticklabels([name.replace(' ', '\n') for name in model_names], 
                           rotation=0, ha='center', fontsize=10)
        
        # Add value labels
        for bar, acc in zip(bars, accuracies):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. ROC Comparison with Ensemble
        ax2 = axes[0, 1]
        
        for name, result in self.results.items():
            fpr, tpr, _ = roc_curve(self.y_test, result['y_pred_proba'])
            roc_auc = auc(fpr, tpr)
            
            linewidth = 3 if 'Ensemble' in name else 2
            linestyle = '-' if 'Ensemble' in name else '--'
            
            ax2.plot(fpr, tpr, linewidth=linewidth, linestyle=linestyle,
                    label=f'{name} (AUC = {roc_auc:.3f})')
        
        ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax2.set_title('📈 ROC Curves - Including Ensemble', fontsize=16, fontweight='bold')
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # 3. Confusion Matrix - Ensemble
        ax3 = axes[1, 0]
        
        cm_ensemble = confusion_matrix(self.y_test, y_pred_ensemble)
        sns.heatmap(cm_ensemble, annot=True, fmt='d', cmap='Blues', ax=ax3,
                   xticklabels=['Real News', 'Fake News'],
                   yticklabels=['Real News', 'Fake News'])
        ax3.set_title('🎯 Ensemble Model - Confusion Matrix', fontsize=16, fontweight='bold')
        
        # 4. Performance Metrics Comparison
        ax4 = axes[1, 1]
        
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        ensemble_metrics = [self.results['Ensemble Model'][metric] for metric in metrics]
        
        # Get average of other models
        other_models = [name for name in self.results.keys() if 'Ensemble' not in name]
        avg_metrics = []
        for metric in metrics:
            avg_val = np.mean([self.results[name][metric] for name in other_models])
            avg_metrics.append(avg_val)
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax4.bar(x - width/2, avg_metrics, width, label='Average Individual Models', 
                       color='lightblue', alpha=0.8)
        bars2 = ax4.bar(x + width/2, ensemble_metrics, width, label='Ensemble Model', 
                       color='gold', alpha=0.8)
        
        ax4.set_title('📊 Ensemble vs Individual Models', fontsize=16, fontweight='bold')
        ax4.set_xlabel('Metrics')
        ax4.set_ylabel('Score')
        ax4.set_xticks(x)
        ax4.set_xticklabels([m.capitalize() for m in metrics])
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.media_dir, 'ensemble_analysis.png'), dpi=300, bbox_inches='tight')
        print("✅ Ensemble analysis saved!")
        
    def generate_comprehensive_report(self):
        """Generate comprehensive analysis report"""
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['accuracy'])
        best_result = self.results[best_model_name]
        
        # Calculate additional statistics
        all_accuracies = [result['accuracy'] for result in self.results.values()]
        
        report = {
            'analysis_type': 'Advanced Deep Analysis',
            'best_model': best_model_name,
            'best_accuracy': best_result['accuracy'],
            'best_precision': best_result['precision'],
            'best_recall': best_result['recall'],
            'best_f1': best_result['f1'],
            'model_count': len(self.results),
            'average_accuracy': np.mean(all_accuracies),
            'accuracy_std': np.std(all_accuracies),
            'perfect_models': sum(1 for acc in all_accuracies if acc >= 0.99),
            'dataset_info': {
                'training_samples': len(self.X_train),
                'testing_samples': len(self.X_test),
                'total_features': len(self.feature_names),
                'feature_extraction': 'Enhanced TF-IDF with n-grams (1,3)'
            },
            'all_results': {
                name: {
                    'accuracy': result['accuracy'],
                    'precision': result['precision'],
                    'recall': result['recall'],
                    'f1': result['f1']
                } for name, result in self.results.items()
            }
        }
        
        # Save comprehensive report
        with open(os.path.join(self.reports_dir, 'advanced_deep_analysis_report.json'), 'w') as f:
            import json
            json.dump(report, f, indent=2)
        
        print("✅ Advanced deep analysis report saved!")
        return report

def main():
    print("🚀 Starting Advanced Deep Analysis for Fake News Detection")
    print("=" * 70)
    
    # Initialize advanced analysis
    analyzer = AdvancedDeepAnalysis()
    
    # Load and prepare data
    analyzer.load_and_prepare_data()
    
    # Train enhanced models
    analyzer.train_enhanced_models()
    
    # Create all advanced visualizations
    print("\n🔬 Creating advanced visualizations...")
    analyzer.create_feature_importance_analysis()
    analyzer.create_statistical_comparison()
    analyzer.create_dimensionality_analysis()
    analyzer.create_word_importance_visualization()
    analyzer.create_ensemble_analysis()
    
    # Generate comprehensive report
    report = analyzer.generate_comprehensive_report()
    
    print("\n" + "=" * 70)
    print("🎉 ADVANCED DEEP ANALYSIS COMPLETE!")
    print("=" * 70)
    print(f"🏆 Best Model: {report['best_model']}")
    print(f"🎯 Best Accuracy: {report['best_accuracy']:.1%}")
    print(f"📊 Models Analyzed: {report['model_count']}")
    print(f"🔥 Perfect Models (≥99%): {report['perfect_models']}")
    print(f"📈 Advanced Visualizations Created:")
    print("   • feature_importance_analysis.png")
    print("   • statistical_comparison.png")
    print("   • dimensionality_analysis.png")
    print("   • word_importance_analysis.png")
    print("   • ensemble_analysis.png")
    print("   • advanced_deep_analysis_report.json")
    print("\n🎨 All advanced graphs and statistical analysis ready!")

if __name__ == "__main__":
    main()