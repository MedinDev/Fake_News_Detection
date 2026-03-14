import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve

class VisualizationUtils:
    @staticmethod
    def plot_model_comparison(results, save_dir='plots'):
        # Accuracy comparison
        plt.figure(figsize=(12, 6))
        accuracies = [results[model]['accuracy'] for model in results]
        plt.bar(results.keys(), accuracies, color=['#2ecc71', '#3498db', '#e74c3c', '#f1c40f'])
        plt.title('Model Accuracy Comparison', fontsize=14, pad=20)
        plt.ylabel('Accuracy', fontsize=12)
        plt.xlabel('Models', fontsize=12)
        plt.ylim(0.5, 1.0)
        for i, v in enumerate(accuracies):
            plt.text(i, v + 0.01, f'{v:.4f}', ha='center', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{save_dir}/accuracy_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Confusion matrices
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        for i, (name, metrics) in enumerate(results.items()):
            ax = axes[i//2, i%2]
            sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', ax=ax, 
                       cmap='Blues', cbar=True)
            ax.set_title(f'{name.upper()} Confusion Matrix', fontsize=12, pad=20)
            ax.set_xlabel('Predicted Label', fontsize=10)
            ax.set_ylabel('True Label', fontsize=10)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Metrics comparison
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        plt.figure(figsize=(14, 7))
        x = np.arange(len(metrics))
        width = 0.2
        colors = ['#2ecc71', '#3498db', '#e74c3c', '#f1c40f']
        
        for i, (name, result) in enumerate(results.items()):
            values = [result[metric] for metric in metrics]
            plt.bar(x + i*width, values, width, label=name.upper(), color=colors[i])
        
        plt.xlabel('Metrics', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.title('Model Performance Comparison', fontsize=14, pad=20)
        plt.xticks(x + width*1.5, [m.capitalize() for m in metrics], fontsize=10)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{save_dir}/metrics_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

    @staticmethod
    def plot_learning_curves(history, save_dir='plots'):
        plt.figure(figsize=(15, 5))
        
        # Accuracy subplot
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training', color='#2ecc71')
        plt.plot(history.history['val_accuracy'], label='Validation', color='#3498db')
        plt.title('Model Accuracy Over Time', fontsize=14, pad=20)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # Loss subplot
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training', color='#e74c3c')
        plt.plot(history.history['val_loss'], label='Validation', color='#f1c40f')
        plt.title('Model Loss Over Time', fontsize=14, pad=20)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/learning_curves.png', dpi=300, bbox_inches='tight')
        plt.close()

    @staticmethod
    def plot_feature_importance(model, feature_names, top_n=20, save_dir='plots'):
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:top_n]
            
            plt.figure(figsize=(12, 6))
            plt.title(f'Top {top_n} Most Important Features', fontsize=14, pad=20)
            plt.bar(range(top_n), importances[indices], color='#3498db')
            plt.xticks(range(top_n), [feature_names[i] for i in indices], rotation=45, ha='right')
            plt.xlabel('Features', fontsize=12)
            plt.ylabel('Importance', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'{save_dir}/feature_importance.png', dpi=300, bbox_inches='tight')
            plt.close()