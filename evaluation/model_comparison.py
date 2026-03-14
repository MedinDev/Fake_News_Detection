import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, learning_curve
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve
)
from sklearn.calibration import calibration_curve
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding, Conv1D, GlobalMaxPooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from scipy import sparse
import matplotlib.pyplot as plt
import seaborn as sns
import time
import psutil
import os
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class ComprehensiveModelComparison:
    """
    Comprehensive 6-Model Comparison System for Fake News Detection
    
    Traditional ML Models:
    1. Random Forest
    2. Support Vector Machine (SVM)
    3. Naive Bayes
    
    Deep Learning Models:
    4. Multi-Layer Perceptron (MLP)
    5. Long Short-Term Memory (LSTM)
    6. Convolutional Neural Network (CNN)
    """
    
    def __init__(self, max_vocab_size=10000, max_sequence_length=500, random_state=42):
        self.random_state = random_state
        self.max_vocab_size = max_vocab_size
        self.max_sequence_length = max_sequence_length
        
        # Initialize models
        self.traditional_models = {}
        self.deep_learning_models = {}
        self.tokenizer = None
        
        # Results storage
        self.results = defaultdict(dict)
        self.training_times = {}
        self.prediction_times = {}
        self.memory_usage = {}
        
        self._initialize_models()
    
    def _initialize_models(self):
        """
        Initialize all 6 models with optimized hyperparameters
        """
        # Traditional ML Models (1-3)
        self.traditional_models = {
            '1_Random_Forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                bootstrap=True,
                random_state=self.random_state,
                n_jobs=-1
            ),
            '2_SVM': SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,
                random_state=self.random_state
            ),
            '3_Naive_Bayes': MultinomialNB(
                alpha=0.1,
                fit_prior=True
            )
        }
        
        # Deep Learning Models will be created dynamically
        self.deep_learning_configs = {
            '4_MLP': {
                'hidden_layer_sizes': (128, 64, 32),
                'activation': 'relu',
                'solver': 'adam',
                'alpha': 0.001,
                'learning_rate': 'adaptive',
                'max_iter': 500
            },
            '5_LSTM': {
                'embedding_dim': 128,
                'lstm_units': 64,
                'dropout': 0.3,
                'recurrent_dropout': 0.3
            },
            '6_CNN': {
                'embedding_dim': 128,
                'filters': 128,
                'kernel_size': 3,
                'dropout': 0.5
            }
        }
    
    def _create_mlp_model(self, input_dim):
        """
        Create MLP model for traditional features
        """
        config = self.deep_learning_configs['4_MLP']
        
        model = MLPClassifier(
            hidden_layer_sizes=config['hidden_layer_sizes'],
            activation=config['activation'],
            solver=config['solver'],
            alpha=config['alpha'],
            learning_rate=config['learning_rate'],
            max_iter=config['max_iter'],
            random_state=self.random_state
        )
        
        return model
    
    def _create_lstm_model(self, vocab_size):
        """
        Create LSTM model for text sequences
        """
        config = self.deep_learning_configs['5_LSTM']
        
        model = Sequential([
            Embedding(vocab_size, config['embedding_dim'], input_length=self.max_sequence_length),
            LSTM(config['lstm_units'], 
                 dropout=config['dropout'], 
                 recurrent_dropout=config['recurrent_dropout'],
                 return_sequences=True),
            LSTM(config['lstm_units']//2, 
                 dropout=config['dropout'], 
                 recurrent_dropout=config['recurrent_dropout']),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(32, activation='relu'),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _create_cnn_model(self, vocab_size):
        """
        Create CNN model for text classification
        """
        config = self.deep_learning_configs['6_CNN']
        
        model = Sequential([
            Embedding(vocab_size, config['embedding_dim'], input_length=self.max_sequence_length),
            Conv1D(config['filters'], config['kernel_size'], activation='relu'),
            Conv1D(config['filters'], config['kernel_size'], activation='relu'),
            GlobalMaxPooling1D(),
            Dense(128, activation='relu'),
            Dropout(config['dropout']),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _prepare_text_sequences(self, texts, fit=True):
        """
        Prepare text sequences for deep learning models
        """
        if fit:
            self.tokenizer = Tokenizer(
                num_words=self.max_vocab_size,
                oov_token='<OOV>',
                lower=True
            )
            self.tokenizer.fit_on_texts(texts)
        
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded_sequences = pad_sequences(
            sequences, 
            maxlen=self.max_sequence_length,
            padding='post',
            truncating='post'
        )
        
        return padded_sequences
    
    def _measure_performance(self, func, *args, **kwargs):
        """
        Measure execution time and memory usage
        """
        process = psutil.Process(os.getpid())
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
        start_time = time.time()
        
        result = func(*args, **kwargs)
        
        end_time = time.time()
        end_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        execution_time = end_time - start_time
        memory_used = end_memory - start_memory
        
        return result, execution_time, memory_used
    
    def _evaluate_traditional_model(self, model_name, model, X_train, X_test, y_train, y_test, cv_folds=5):
        """
        Comprehensive evaluation for traditional ML models
        """
        print(f"  Evaluating {model_name}...")
        
        # Convert sparse matrices if needed
        if sparse.issparse(X_train):
            X_train_dense = X_train.toarray() if model_name != '3_Naive_Bayes' else X_train
            X_test_dense = X_test.toarray() if model_name != '3_Naive_Bayes' else X_test
        else:
            X_train_dense = X_train
            X_test_dense = X_test
        
        # Training with performance measurement
        trained_model, train_time, train_memory = self._measure_performance(
            model.fit, X_train_dense, y_train
        )
        
        # Prediction with performance measurement
        y_pred_proba, pred_time, pred_memory = self._measure_performance(
            model.predict_proba, X_test_dense
        )
        y_pred = (y_pred_proba[:, 1] > 0.5).astype(int)
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        cv_scores = cross_validate(
            model, X_train_dense, y_train, cv=cv,
            scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
            return_train_score=True
        )
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba[:, 1]),
            'cv_accuracy_mean': cv_scores['test_accuracy'].mean(),
            'cv_accuracy_std': cv_scores['test_accuracy'].std(),
            'cv_precision_mean': cv_scores['test_precision'].mean(),
            'cv_precision_std': cv_scores['test_precision'].std(),
            'cv_recall_mean': cv_scores['test_recall'].mean(),
            'cv_recall_std': cv_scores['test_recall'].std(),
            'cv_f1_mean': cv_scores['test_f1'].mean(),
            'cv_f1_std': cv_scores['test_f1'].std(),
            'cv_roc_auc_mean': cv_scores['test_roc_auc'].mean(),
            'cv_roc_auc_std': cv_scores['test_roc_auc'].std(),
            'training_time': train_time,
            'prediction_time': pred_time,
            'memory_usage': max(train_memory, pred_memory)
        }
        
        # Store additional data for visualization
        self.results[model_name]['y_true'] = y_test
        self.results[model_name]['y_pred'] = y_pred
        self.results[model_name]['y_pred_proba'] = y_pred_proba[:, 1]
        self.results[model_name]['confusion_matrix'] = confusion_matrix(y_test, y_pred)
        
        return metrics
    
    def _evaluate_deep_learning_model(self, model_name, model, X_train, X_test, y_train, y_test, epochs=20):
        """
        Comprehensive evaluation for deep learning models
        """
        print(f"  Evaluating {model_name}...")
        
        # Early stopping callback
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        # Training with performance measurement
        def train_model():
            return model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=32,
                validation_split=0.2,
                callbacks=[early_stopping],
                verbose=0
            )
        
        history, train_time, train_memory = self._measure_performance(train_model)
        
        # Prediction with performance measurement
        y_pred_proba, pred_time, pred_memory = self._measure_performance(
            model.predict, X_test, verbose=0
        )
        y_pred_proba = y_pred_proba.flatten()
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'training_time': train_time,
            'prediction_time': pred_time,
            'memory_usage': max(train_memory, pred_memory),
            'final_train_loss': history.history['loss'][-1],
            'final_val_loss': history.history['val_loss'][-1],
            'final_train_acc': history.history['accuracy'][-1],
            'final_val_acc': history.history['val_accuracy'][-1]
        }
        
        # Store additional data for visualization
        self.results[model_name]['y_true'] = y_test
        self.results[model_name]['y_pred'] = y_pred
        self.results[model_name]['y_pred_proba'] = y_pred_proba
        self.results[model_name]['confusion_matrix'] = confusion_matrix(y_test, y_pred)
        self.results[model_name]['training_history'] = history.history
        
        return metrics
    
    def run_comprehensive_comparison(self, texts, X_features, y, test_size=0.2):
        """
        Run comprehensive comparison of all 6 models
        """
        print("Comprehensive 6-Model Comparison System")
        print("=======================================")
        print(f"Total samples: {len(texts)}")
        print(f"Feature dimensions: {X_features.shape[1]}")
        
        # Split data
        from sklearn.model_selection import train_test_split
        
        indices = np.arange(len(texts))
        train_idx, test_idx = train_test_split(
            indices, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        texts_train = [texts[i] for i in train_idx]
        texts_test = [texts[i] for i in test_idx]
        X_train = X_features[train_idx]
        X_test = X_features[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]
        
        print(f"Training samples: {len(train_idx)}")
        print(f"Testing samples: {len(test_idx)}")
        
        # Prepare text sequences for deep learning
        X_seq_train = self._prepare_text_sequences(texts_train, fit=True)
        X_seq_test = self._prepare_text_sequences(texts_test, fit=False)
        vocab_size = min(self.max_vocab_size, len(self.tokenizer.word_index) + 1)
        
        # Convert features to dense for MLP
        if sparse.issparse(X_train):
            X_train_dense = X_train.toarray()
            X_test_dense = X_test.toarray()
        else:
            X_train_dense = X_train
            X_test_dense = X_test
        
        all_results = {}
        
        print("\n=== Traditional ML Models ===")
        
        # Evaluate Traditional ML Models (1-3)
        for model_name, model in self.traditional_models.items():
            metrics = self._evaluate_traditional_model(
                model_name, model, X_train, X_test, y_train, y_test
            )
            all_results[model_name] = metrics
            print(f"    ✓ {model_name}: Accuracy = {metrics['accuracy']:.4f}, F1 = {metrics['f1_score']:.4f}")
        
        print("\n=== Deep Learning Models ===")
        
        # Evaluate MLP (4)
        mlp_model = self._create_mlp_model(X_train_dense.shape[1])
        metrics = self._evaluate_traditional_model(
            '4_MLP', mlp_model, X_train_dense, X_test_dense, y_train, y_test
        )
        all_results['4_MLP'] = metrics
        print(f"    ✓ 4_MLP: Accuracy = {metrics['accuracy']:.4f}, F1 = {metrics['f1_score']:.4f}")
        
        # Evaluate LSTM (5)
        lstm_model = self._create_lstm_model(vocab_size)
        metrics = self._evaluate_deep_learning_model(
            '5_LSTM', lstm_model, X_seq_train, X_seq_test, y_train, y_test
        )
        all_results['5_LSTM'] = metrics
        print(f"    ✓ 5_LSTM: Accuracy = {metrics['accuracy']:.4f}, F1 = {metrics['f1_score']:.4f}")
        
        # Evaluate CNN (6)
        cnn_model = self._create_cnn_model(vocab_size)
        metrics = self._evaluate_deep_learning_model(
            '6_CNN', cnn_model, X_seq_train, X_seq_test, y_train, y_test
        )
        all_results['6_CNN'] = metrics
        print(f"    ✓ 6_CNN: Accuracy = {metrics['accuracy']:.4f}, F1 = {metrics['f1_score']:.4f}")
        
        # Store models for later use
        self.trained_models = {
            **self.traditional_models,
            '4_MLP': mlp_model,
            '5_LSTM': lstm_model,
            '6_CNN': cnn_model
        }
        
        print("\n✓ Comprehensive model comparison completed!")
        
        return all_results
    
    def generate_comparison_report(self, results):
        """
        Generate comprehensive comparison report
        """
        # Create results DataFrame
        df_results = pd.DataFrame(results).T
        
        # Sort by F1 score
        df_results = df_results.sort_values('f1_score', ascending=False)
        
        print("\n" + "="*80)
        print("COMPREHENSIVE MODEL COMPARISON REPORT")
        print("="*80)
        
        # Performance Summary
        print("\n1. PERFORMANCE RANKING (by F1-Score):")
        print("-" * 50)
        for i, (model, row) in enumerate(df_results.iterrows(), 1):
            print(f"{i}. {model:<15} | F1: {row['f1_score']:.4f} | Accuracy: {row['accuracy']:.4f} | AUC: {row['roc_auc']:.4f}")
        
        # Detailed Metrics Table
        print("\n2. DETAILED PERFORMANCE METRICS:")
        print("-" * 50)
        metrics_cols = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        print(df_results[metrics_cols].round(4).to_string())
        
        # Computational Efficiency
        print("\n3. COMPUTATIONAL EFFICIENCY:")
        print("-" * 50)
        efficiency_cols = ['training_time', 'prediction_time', 'memory_usage']
        efficiency_df = df_results[efficiency_cols].round(4)
        print(efficiency_df.to_string())
        
        # Cross-validation Results (for traditional models)
        print("\n4. CROSS-VALIDATION RESULTS (Traditional ML):")
        print("-" * 50)
        cv_models = [m for m in df_results.index if 'cv_accuracy_mean' in df_results.columns]
        if cv_models:
            cv_cols = ['cv_accuracy_mean', 'cv_precision_mean', 'cv_recall_mean', 'cv_f1_mean', 'cv_roc_auc_mean']
            available_cv_cols = [col for col in cv_cols if col in df_results.columns]
            if available_cv_cols:
                cv_df = df_results.loc[cv_models, available_cv_cols].round(4)
                print(cv_df.to_string())
        
        # Model Recommendations
        print("\n5. MODEL RECOMMENDATIONS:")
        print("-" * 50)
        
        best_overall = df_results.index[0]
        best_traditional = df_results.loc[df_results.index.str.contains('1_|2_|3_')].index[0]
        best_deep_learning = df_results.loc[df_results.index.str.contains('4_|5_|6_')].index[0]
        fastest_model = df_results.loc[df_results['training_time'].idxmin()].name
        most_efficient = df_results.loc[df_results['memory_usage'].idxmin()].name
        
        print(f"• Best Overall Performance: {best_overall} (F1: {df_results.loc[best_overall, 'f1_score']:.4f})")
        print(f"• Best Traditional ML: {best_traditional} (F1: {df_results.loc[best_traditional, 'f1_score']:.4f})")
        print(f"• Best Deep Learning: {best_deep_learning} (F1: {df_results.loc[best_deep_learning, 'f1_score']:.4f})")
        print(f"• Fastest Training: {fastest_model} ({df_results.loc[fastest_model, 'training_time']:.2f}s)")
        print(f"• Most Memory Efficient: {most_efficient} ({df_results.loc[most_efficient, 'memory_usage']:.2f}MB)")
        
        # Statistical Significance Analysis
        print("\n6. STATISTICAL ANALYSIS:")
        print("-" * 50)
        
        # Performance gaps
        f1_scores = df_results['f1_score'].values
        accuracy_scores = df_results['accuracy'].values
        
        print(f"• F1-Score Range: {f1_scores.min():.4f} - {f1_scores.max():.4f} (Gap: {f1_scores.max() - f1_scores.min():.4f})")
        print(f"• Accuracy Range: {accuracy_scores.min():.4f} - {accuracy_scores.max():.4f} (Gap: {accuracy_scores.max() - accuracy_scores.min():.4f})")
        print(f"• Performance Std Dev: F1={f1_scores.std():.4f}, Accuracy={accuracy_scores.std():.4f}")
        
        return df_results
    
    def plot_comparison_visualizations(self, results, save_path=None):
        """
        Generate comprehensive comparison visualizations
        """
        df_results = pd.DataFrame(results).T
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Performance Metrics Comparison
        ax1 = plt.subplot(2, 3, 1)
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        x = np.arange(len(df_results))
        width = 0.15
        
        for i, metric in enumerate(metrics):
            plt.bar(x + i*width, df_results[metric], width, label=metric.replace('_', ' ').title())
        
        plt.xlabel('Models')
        plt.ylabel('Score')
        plt.title('Performance Metrics Comparison')
        plt.xticks(x + width*2, df_results.index, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. Training Time vs Accuracy
        ax2 = plt.subplot(2, 3, 2)
        scatter = plt.scatter(df_results['training_time'], df_results['accuracy'], 
                            s=100, c=df_results['f1_score'], cmap='viridis', alpha=0.7)
        plt.xlabel('Training Time (seconds)')
        plt.ylabel('Accuracy')
        plt.title('Training Time vs Accuracy')
        plt.colorbar(scatter, label='F1-Score')
        
        for i, model in enumerate(df_results.index):
            plt.annotate(model, (df_results.iloc[i]['training_time'], df_results.iloc[i]['accuracy']),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # 3. Memory Usage Comparison
        ax3 = plt.subplot(2, 3, 3)
        colors = plt.cm.Set3(np.linspace(0, 1, len(df_results)))
        bars = plt.bar(df_results.index, df_results['memory_usage'], color=colors)
        plt.xlabel('Models')
        plt.ylabel('Memory Usage (MB)')
        plt.title('Memory Usage Comparison')
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}MB', ha='center', va='bottom', fontsize=8)
        
        # 4. ROC Curves Comparison
        ax4 = plt.subplot(2, 3, 4)
        colors = plt.cm.tab10(np.linspace(0, 1, len(df_results)))
        
        for i, (model_name, _) in enumerate(df_results.iterrows()):
            if model_name in self.results:
                y_true = self.results[model_name]['y_true']
                y_pred_proba = self.results[model_name]['y_pred_proba']
                fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
                auc_score = roc_auc_score(y_true, y_pred_proba)
                plt.plot(fpr, tpr, color=colors[i], label=f'{model_name} (AUC={auc_score:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # 5. Confusion Matrices
        ax5 = plt.subplot(2, 3, 5)
        n_models = len(df_results)
        fig_cm, axes_cm = plt.subplots(2, 3, figsize=(15, 10))
        axes_cm = axes_cm.flatten()
        
        for i, (model_name, _) in enumerate(df_results.iterrows()):
            if i < 6 and model_name in self.results:
                cm = self.results[model_name]['confusion_matrix']
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes_cm[i])
                axes_cm[i].set_title(f'{model_name}')
                axes_cm[i].set_xlabel('Predicted')
                axes_cm[i].set_ylabel('Actual')
        
        plt.tight_layout()
        if save_path:
            fig_cm.savefig(f"{save_path}_confusion_matrices.png", dpi=300, bbox_inches='tight')
        
        # 6. Performance Radar Chart
        ax6 = plt.subplot(2, 3, 6, projection='polar')
        
        # Normalize metrics for radar chart
        metrics_radar = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        angles = np.linspace(0, 2*np.pi, len(metrics_radar), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        for i, (model_name, row) in enumerate(df_results.iterrows()):
            values = [row[metric] for metric in metrics_radar]
            values += values[:1]  # Complete the circle
            
            plt.plot(angles, values, 'o-', linewidth=2, label=model_name, color=colors[i])
            plt.fill(angles, values, alpha=0.1, color=colors[i])
        
        plt.xticks(angles[:-1], [m.replace('_', ' ').title() for m in metrics_radar])
        plt.ylim(0, 1)
        plt.title('Performance Radar Chart')
        plt.legend(bbox_to_anchor=(1.3, 1.0))
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(f"{save_path}_comparison.png", dpi=300, bbox_inches='tight')
            plt.show()
        
        return fig
    
    def export_results(self, results, filepath):
        """
        Export comprehensive results to CSV and JSON
        """
        df_results = pd.DataFrame(results).T
        
        # Export to CSV
        df_results.to_csv(f"{filepath}.csv")
        
        # Export detailed results to JSON
        detailed_results = {
            'summary': df_results.to_dict(),
            'detailed_results': dict(self.results),
            'model_configurations': {
                'traditional_models': {name: str(model) for name, model in self.traditional_models.items()},
                'deep_learning_configs': self.deep_learning_configs
            }
        }
        
        import json
        with open(f"{filepath}.json", 'w') as f:
            json.dump(detailed_results, f, indent=2, default=str)
        
        print(f"✓ Results exported to {filepath}.csv and {filepath}.json")

if __name__ == "__main__":
    # Example usage
    print("Comprehensive 6-Model Comparison System")
    print("======================================")
    
    # Sample data for demonstration
    np.random.seed(42)
    n_samples = 1000
    
    sample_texts = [
        "This is a legitimate news article with verified information.",
        "BREAKING: Shocking revelation that will change everything!!!",
        "Scientists have made significant progress in renewable energy research."
    ] * (n_samples // 3)
    
    sample_labels = np.array([1, 0, 1] * (n_samples // 3))
    sample_features = np.random.rand(n_samples, 100)
    
    # Initialize comparison system
    comparison = ComprehensiveModelComparison()
    
    # Run comprehensive comparison
    results = comparison.run_comprehensive_comparison(sample_texts, sample_features, sample_labels)
    
    # Generate report
    df_results = comparison.generate_comparison_report(results)
    
    # Generate visualizations
    comparison.plot_comparison_visualizations(results, save_path="model_comparison")
    
    # Export results
    comparison.export_results(results, "comprehensive_model_comparison")
    
    print("\n✓ Comprehensive 6-model comparison completed!")