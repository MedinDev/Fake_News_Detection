import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from collections import Counter, defaultdict
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

class ErrorAnalysisFramework:
    """
    Comprehensive Error Analysis and Interpretability Framework
    
    Features:
    - Confusion matrix analysis with detailed breakdowns
    - Error pattern identification and clustering
    - Feature importance analysis for misclassified samples
    - Linguistic pattern analysis of errors
    - Confidence-based error analysis
    - Adversarial example detection
    - Model interpretability through LIME/SHAP-like analysis
    - Cross-model error comparison
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.sia = SentimentIntensityAnalyzer()
        
        # Results storage
        self.error_patterns = {}
        self.linguistic_analysis = {}
        self.confidence_analysis = {}
        self.feature_importance = {}
        self.interpretability_results = {}
        
    def comprehensive_error_analysis(self, models_predictions, X_test, y_test, 
                                   text_data=None, model_names=None):
        """
        Perform comprehensive error analysis across multiple models
        """
        print("Comprehensive Error Analysis Framework")
        print("=====================================")
        
        if model_names is None:
            model_names = [f"Model_{i}" for i in range(len(models_predictions))]
        
        results = {}
        
        for i, (model_name, predictions) in enumerate(zip(model_names, models_predictions)):
            print(f"\nAnalyzing {model_name}...")
            
            # Basic error analysis
            results[model_name] = self._analyze_single_model(
                predictions, X_test, y_test, text_data, model_name
            )
        
        # Cross-model comparison
        results['cross_model_analysis'] = self._cross_model_error_analysis(
            models_predictions, y_test, model_names
        )
        
        return results
    
    def _analyze_single_model(self, predictions, X_test, y_test, text_data, model_name):
        """
        Analyze errors for a single model
        """
        # Get predictions and probabilities
        if hasattr(predictions, 'predict_proba'):
            y_pred = predictions.predict(X_test)
            y_proba = predictions.predict_proba(X_test)
        else:
            y_pred = predictions
            y_proba = None
        
        # Basic metrics
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Error indices
        error_indices = np.where(y_test != y_pred)[0]
        correct_indices = np.where(y_test == y_pred)[0]
        
        # False positives and false negatives
        fp_indices = np.where((y_test == 0) & (y_pred == 1))[0]
        fn_indices = np.where((y_test == 1) & (y_pred == 0))[0]
        
        analysis_results = {
            'confusion_matrix': cm,
            'classification_report': report,
            'error_indices': error_indices,
            'correct_indices': correct_indices,
            'false_positives': fp_indices,
            'false_negatives': fn_indices,
            'error_rate': len(error_indices) / len(y_test),
            'fp_rate': len(fp_indices) / len(y_test),
            'fn_rate': len(fn_indices) / len(y_test)
        }
        
        # Confidence analysis
        if y_proba is not None:
            analysis_results['confidence_analysis'] = self._confidence_analysis(
                y_test, y_pred, y_proba, error_indices
            )
        
        # Text-based analysis
        if text_data is not None:
            analysis_results['linguistic_analysis'] = self._linguistic_error_analysis(
                text_data, y_test, y_pred, error_indices, fp_indices, fn_indices
            )
            
            analysis_results['error_clustering'] = self._cluster_error_patterns(
                text_data, error_indices, y_test, y_pred
            )
        
        # Feature importance for errors
        if hasattr(X_test, 'toarray'):  # Sparse matrix
            X_dense = X_test.toarray()
        else:
            X_dense = X_test
            
        analysis_results['feature_analysis'] = self._feature_importance_analysis(
            X_dense, y_test, y_pred, error_indices
        )
        
        return analysis_results
    
    def _confidence_analysis(self, y_true, y_pred, y_proba, error_indices):
        """
        Analyze prediction confidence patterns
        """
        # Get confidence scores (max probability)
        confidence_scores = np.max(y_proba, axis=1)
        
        # Separate confidence for correct and incorrect predictions
        correct_confidence = confidence_scores[y_true == y_pred]
        error_confidence = confidence_scores[error_indices]
        
        # Confidence-based error analysis
        confidence_analysis = {
            'mean_correct_confidence': np.mean(correct_confidence),
            'mean_error_confidence': np.mean(error_confidence),
            'std_correct_confidence': np.std(correct_confidence),
            'std_error_confidence': np.std(error_confidence),
            'confidence_scores': confidence_scores,
            'correct_confidence': correct_confidence,
            'error_confidence': error_confidence
        }
        
        # High-confidence errors (potentially adversarial)
        high_conf_threshold = np.percentile(confidence_scores, 75)
        high_conf_errors = error_indices[confidence_scores[error_indices] > high_conf_threshold]
        
        confidence_analysis['high_confidence_errors'] = {
            'indices': high_conf_errors,
            'count': len(high_conf_errors),
            'percentage': len(high_conf_errors) / len(error_indices) * 100 if len(error_indices) > 0 else 0
        }
        
        # Low-confidence correct predictions (uncertain but right)
        low_conf_threshold = np.percentile(confidence_scores, 25)
        correct_indices = np.where(y_true == y_pred)[0]
        low_conf_correct = correct_indices[confidence_scores[correct_indices] < low_conf_threshold]
        
        confidence_analysis['low_confidence_correct'] = {
            'indices': low_conf_correct,
            'count': len(low_conf_correct),
            'percentage': len(low_conf_correct) / len(correct_indices) * 100 if len(correct_indices) > 0 else 0
        }
        
        return confidence_analysis
    
    def _linguistic_error_analysis(self, text_data, y_true, y_pred, error_indices, fp_indices, fn_indices):
        """
        Analyze linguistic patterns in misclassified texts
        """
        print("  Performing linguistic error analysis...")
        
        # Extract error texts
        error_texts = [text_data[i] for i in error_indices]
        fp_texts = [text_data[i] for i in fp_indices]
        fn_texts = [text_data[i] for i in fn_indices]
        
        # Correct texts for comparison
        correct_indices = np.where(y_true == y_pred)[0]
        correct_texts = [text_data[i] for i in correct_indices[:min(1000, len(correct_indices))]]
        
        linguistic_analysis = {
            'error_text_stats': self._compute_text_statistics(error_texts),
            'fp_text_stats': self._compute_text_statistics(fp_texts),
            'fn_text_stats': self._compute_text_statistics(fn_texts),
            'correct_text_stats': self._compute_text_statistics(correct_texts)
        }
        
        # Sentiment analysis
        linguistic_analysis['sentiment_analysis'] = {
            'error_sentiment': self._analyze_sentiment(error_texts),
            'fp_sentiment': self._analyze_sentiment(fp_texts),
            'fn_sentiment': self._analyze_sentiment(fn_texts),
            'correct_sentiment': self._analyze_sentiment(correct_texts)
        }
        
        # Common words/phrases in errors
        linguistic_analysis['error_vocabulary'] = self._analyze_error_vocabulary(
            error_texts, correct_texts
        )
        
        return linguistic_analysis
    
    def _compute_text_statistics(self, texts):
        """
        Compute basic text statistics
        """
        if not texts:
            return {}
        
        stats = {
            'count': len(texts),
            'avg_length': np.mean([len(text) for text in texts]),
            'avg_word_count': np.mean([len(text.split()) for text in texts]),
            'avg_sentence_count': np.mean([len(re.split(r'[.!?]+', text)) for text in texts]),
            'avg_exclamation_count': np.mean([text.count('!') for text in texts]),
            'avg_question_count': np.mean([text.count('?') for text in texts]),
            'avg_caps_ratio': np.mean([sum(1 for c in text if c.isupper()) / len(text) if len(text) > 0 else 0 for text in texts]),
            'avg_digit_ratio': np.mean([sum(1 for c in text if c.isdigit()) / len(text) if len(text) > 0 else 0 for text in texts])
        }
        
        return stats
    
    def _analyze_sentiment(self, texts):
        """
        Analyze sentiment patterns
        """
        if not texts:
            return {}
        
        sentiments = [self.sia.polarity_scores(text) for text in texts]
        
        sentiment_analysis = {
            'avg_compound': np.mean([s['compound'] for s in sentiments]),
            'avg_positive': np.mean([s['pos'] for s in sentiments]),
            'avg_negative': np.mean([s['neg'] for s in sentiments]),
            'avg_neutral': np.mean([s['neu'] for s in sentiments]),
            'sentiment_distribution': {
                'positive': sum(1 for s in sentiments if s['compound'] > 0.05),
                'negative': sum(1 for s in sentiments if s['compound'] < -0.05),
                'neutral': sum(1 for s in sentiments if -0.05 <= s['compound'] <= 0.05)
            }
        }
        
        return sentiment_analysis
    
    def _analyze_error_vocabulary(self, error_texts, correct_texts):
        """
        Analyze vocabulary differences between errors and correct predictions
        """
        if not error_texts or not correct_texts:
            return {}
        
        # Tokenize and clean
        stop_words = set(stopwords.words('english'))
        
        def clean_tokenize(texts):
            all_words = []
            for text in texts:
                words = word_tokenize(text.lower())
                words = [w for w in words if w.isalpha() and w not in stop_words and len(w) > 2]
                all_words.extend(words)
            return all_words
        
        error_words = clean_tokenize(error_texts)
        correct_words = clean_tokenize(correct_texts)
        
        # Word frequency analysis
        error_freq = Counter(error_words)
        correct_freq = Counter(correct_words)
        
        # Find words more common in errors
        error_vocab = {}
        total_error_words = len(error_words)
        total_correct_words = len(correct_words)
        
        for word, error_count in error_freq.most_common(50):
            correct_count = correct_freq.get(word, 0)
            
            error_rate = error_count / total_error_words
            correct_rate = correct_count / total_correct_words if total_correct_words > 0 else 0
            
            if error_rate > correct_rate * 1.5:  # At least 50% more common in errors
                error_vocab[word] = {
                    'error_count': error_count,
                    'correct_count': correct_count,
                    'error_rate': error_rate,
                    'correct_rate': correct_rate,
                    'ratio': error_rate / correct_rate if correct_rate > 0 else float('inf')
                }
        
        return {
            'distinctive_error_words': error_vocab,
            'total_error_vocabulary': len(set(error_words)),
            'total_correct_vocabulary': len(set(correct_words)),
            'vocabulary_overlap': len(set(error_words) & set(correct_words))
        }
    
    def _cluster_error_patterns(self, text_data, error_indices, y_true, y_pred):
        """
        Cluster error patterns to identify common failure modes
        """
        print("  Clustering error patterns...")
        
        if len(error_indices) < 10:  # Need minimum samples for clustering
            return {'message': 'Insufficient error samples for clustering'}
        
        # Extract error texts
        error_texts = [text_data[i] for i in error_indices]
        
        # Vectorize error texts
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', 
                                   ngram_range=(1, 2), min_df=2)
        
        try:
            error_vectors = vectorizer.fit_transform(error_texts)
            
            # Determine optimal number of clusters
            n_clusters = min(5, max(2, len(error_indices) // 10))
            
            # K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state)
            cluster_labels = kmeans.fit_predict(error_vectors)
            
            # Analyze clusters
            cluster_analysis = {}
            feature_names = vectorizer.get_feature_names_out()
            
            for cluster_id in range(n_clusters):
                cluster_indices = np.where(cluster_labels == cluster_id)[0]
                cluster_texts = [error_texts[i] for i in cluster_indices]
                
                # Get cluster centroid and top features
                centroid = kmeans.cluster_centers_[cluster_id]
                top_features_idx = centroid.argsort()[-10:][::-1]
                top_features = [feature_names[i] for i in top_features_idx]
                
                # Analyze error types in this cluster
                cluster_error_indices = [error_indices[i] for i in cluster_indices]
                cluster_true_labels = [y_true[i] for i in cluster_error_indices]
                cluster_pred_labels = [y_pred[i] for i in cluster_error_indices]
                
                fp_count = sum(1 for true, pred in zip(cluster_true_labels, cluster_pred_labels) 
                              if true == 0 and pred == 1)
                fn_count = sum(1 for true, pred in zip(cluster_true_labels, cluster_pred_labels) 
                              if true == 1 and pred == 0)
                
                cluster_analysis[f'cluster_{cluster_id}'] = {
                    'size': len(cluster_indices),
                    'percentage': len(cluster_indices) / len(error_indices) * 100,
                    'top_features': top_features,
                    'false_positives': fp_count,
                    'false_negatives': fn_count,
                    'sample_texts': cluster_texts[:3],  # First 3 examples
                    'text_stats': self._compute_text_statistics(cluster_texts)
                }
            
            return {
                'n_clusters': n_clusters,
                'cluster_analysis': cluster_analysis,
                'silhouette_score': self._compute_silhouette_score(error_vectors, cluster_labels)
            }
            
        except Exception as e:
            return {'error': f'Clustering failed: {str(e)}'}
    
    def _compute_silhouette_score(self, X, labels):
        """
        Compute silhouette score for clustering quality
        """
        try:
            from sklearn.metrics import silhouette_score
            return silhouette_score(X, labels)
        except:
            return None
    
    def _feature_importance_analysis(self, X, y_true, y_pred, error_indices):
        """
        Analyze feature importance for misclassified samples
        """
        print("  Analyzing feature importance for errors...")
        
        if len(error_indices) == 0:
            return {'message': 'No errors to analyze'}
        
        # Get error samples and correct samples
        X_errors = X[error_indices]
        X_correct = X[y_true == y_pred]
        
        # Compute feature statistics
        error_means = np.mean(X_errors, axis=0)
        correct_means = np.mean(X_correct, axis=0)
        
        # Feature differences
        feature_diffs = error_means - correct_means
        feature_ratios = np.divide(error_means, correct_means, 
                                 out=np.zeros_like(error_means), 
                                 where=correct_means!=0)
        
        # Statistical significance of differences
        from scipy.stats import ttest_ind
        
        p_values = []
        for i in range(X.shape[1]):
            if np.var(X_errors[:, i]) > 0 and np.var(X_correct[:, i]) > 0:
                _, p_val = ttest_ind(X_errors[:, i], X_correct[:, i])
                p_values.append(p_val)
            else:
                p_values.append(1.0)
        
        # Identify most discriminative features
        feature_importance = {
            'feature_differences': feature_diffs,
            'feature_ratios': feature_ratios,
            'p_values': np.array(p_values),
            'significant_features': np.where(np.array(p_values) < 0.05)[0],
            'top_error_features': np.argsort(np.abs(feature_diffs))[-20:][::-1],
            'error_feature_stats': {
                'mean': error_means,
                'std': np.std(X_errors, axis=0),
                'min': np.min(X_errors, axis=0),
                'max': np.max(X_errors, axis=0)
            },
            'correct_feature_stats': {
                'mean': correct_means,
                'std': np.std(X_correct, axis=0),
                'min': np.min(X_correct, axis=0),
                'max': np.max(X_correct, axis=0)
            }
        }
        
        return feature_importance
    
    def _cross_model_error_analysis(self, models_predictions, y_true, model_names):
        """
        Analyze error patterns across multiple models
        """
        print("\nPerforming cross-model error analysis...")
        
        n_models = len(models_predictions)
        n_samples = len(y_true)
        
        # Create prediction matrix
        pred_matrix = np.zeros((n_samples, n_models))
        for i, predictions in enumerate(models_predictions):
            if hasattr(predictions, 'predict'):
                pred_matrix[:, i] = predictions.predict(y_true.reshape(-1, 1) if len(y_true.shape) == 1 else y_true)
            else:
                pred_matrix[:, i] = predictions
        
        # Analyze agreement patterns
        agreement_analysis = {
            'unanimous_correct': 0,
            'unanimous_incorrect': 0,
            'majority_correct': 0,
            'majority_incorrect': 0,
            'split_decisions': 0
        }
        
        # Sample-wise analysis
        sample_analysis = []
        
        for i in range(n_samples):
            sample_preds = pred_matrix[i, :]
            true_label = y_true[i]
            
            correct_models = np.sum(sample_preds == true_label)
            total_models = len(sample_preds)
            
            if correct_models == total_models:
                agreement_analysis['unanimous_correct'] += 1
                agreement_type = 'unanimous_correct'
            elif correct_models == 0:
                agreement_analysis['unanimous_incorrect'] += 1
                agreement_type = 'unanimous_incorrect'
            elif correct_models > total_models / 2:
                agreement_analysis['majority_correct'] += 1
                agreement_type = 'majority_correct'
            elif correct_models < total_models / 2:
                agreement_analysis['majority_incorrect'] += 1
                agreement_type = 'majority_incorrect'
            else:
                agreement_analysis['split_decisions'] += 1
                agreement_type = 'split_decisions'
            
            sample_analysis.append({
                'sample_index': i,
                'true_label': true_label,
                'predictions': sample_preds.tolist(),
                'correct_count': correct_models,
                'agreement_type': agreement_type
            })
        
        # Model pair agreement analysis
        pairwise_agreement = np.zeros((n_models, n_models))
        for i in range(n_models):
            for j in range(n_models):
                if i != j:
                    agreement = np.mean(pred_matrix[:, i] == pred_matrix[:, j])
                    pairwise_agreement[i, j] = agreement
                else:
                    pairwise_agreement[i, j] = 1.0
        
        # Find hardest samples (most models get wrong)
        error_counts = np.sum(pred_matrix != y_true.reshape(-1, 1), axis=1)
        hardest_samples = np.argsort(error_counts)[-20:][::-1]  # Top 20 hardest
        
        cross_model_analysis = {
            'agreement_summary': agreement_analysis,
            'agreement_percentages': {k: v/n_samples*100 for k, v in agreement_analysis.items()},
            'pairwise_agreement': pairwise_agreement,
            'pairwise_agreement_df': pd.DataFrame(pairwise_agreement, 
                                                index=model_names, 
                                                columns=model_names),
            'hardest_samples': {
                'indices': hardest_samples.tolist(),
                'error_counts': error_counts[hardest_samples].tolist()
            },
            'sample_analysis': sample_analysis[:100]  # First 100 for brevity
        }
        
        return cross_model_analysis
    
    def generate_error_analysis_report(self, analysis_results):
        """
        Generate comprehensive error analysis report
        """
        print("\n" + "="*80)
        print("COMPREHENSIVE ERROR ANALYSIS REPORT")
        print("="*80)
        
        # 1. Overall Error Summary
        print("\n1. OVERALL ERROR SUMMARY:")
        print("-" * 50)
        
        for model_name, results in analysis_results.items():
            if model_name == 'cross_model_analysis':
                continue
                
            print(f"\n{model_name}:")
            print(f"  Error Rate: {results['error_rate']:.4f} ({results['error_rate']*100:.2f}%)")
            print(f"  False Positive Rate: {results['fp_rate']:.4f} ({results['fp_rate']*100:.2f}%)")
            print(f"  False Negative Rate: {results['fn_rate']:.4f} ({results['fn_rate']*100:.2f}%)")
            
            # Confusion matrix
            cm = results['confusion_matrix']
            print(f"  Confusion Matrix: TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}")
        
        # 2. Confidence Analysis
        print("\n\n2. CONFIDENCE ANALYSIS:")
        print("-" * 50)
        
        for model_name, results in analysis_results.items():
            if model_name == 'cross_model_analysis' or 'confidence_analysis' not in results:
                continue
                
            conf_analysis = results['confidence_analysis']
            print(f"\n{model_name}:")
            print(f"  Mean Confidence (Correct): {conf_analysis['mean_correct_confidence']:.4f}")
            print(f"  Mean Confidence (Errors): {conf_analysis['mean_error_confidence']:.4f}")
            print(f"  High-Confidence Errors: {conf_analysis['high_confidence_errors']['count']} "
                  f"({conf_analysis['high_confidence_errors']['percentage']:.1f}% of errors)")
        
        # 3. Linguistic Analysis
        print("\n\n3. LINGUISTIC ERROR PATTERNS:")
        print("-" * 50)
        
        for model_name, results in analysis_results.items():
            if model_name == 'cross_model_analysis' or 'linguistic_analysis' not in results:
                continue
                
            ling_analysis = results['linguistic_analysis']
            print(f"\n{model_name}:")
            
            # Text statistics comparison
            error_stats = ling_analysis['error_text_stats']
            correct_stats = ling_analysis['correct_text_stats']
            
            print(f"  Average Text Length - Errors: {error_stats.get('avg_length', 0):.1f}, "
                  f"Correct: {correct_stats.get('avg_length', 0):.1f}")
            print(f"  Average Word Count - Errors: {error_stats.get('avg_word_count', 0):.1f}, "
                  f"Correct: {correct_stats.get('avg_word_count', 0):.1f}")
            print(f"  Caps Ratio - Errors: {error_stats.get('avg_caps_ratio', 0):.4f}, "
                  f"Correct: {correct_stats.get('avg_caps_ratio', 0):.4f}")
            
            # Sentiment analysis
            error_sent = ling_analysis['sentiment_analysis']['error_sentiment']
            correct_sent = ling_analysis['sentiment_analysis']['correct_sentiment']
            
            print(f"  Sentiment (Compound) - Errors: {error_sent.get('avg_compound', 0):.4f}, "
                  f"Correct: {correct_sent.get('avg_compound', 0):.4f}")
        
        # 4. Error Clustering
        print("\n\n4. ERROR PATTERN CLUSTERING:")
        print("-" * 50)
        
        for model_name, results in analysis_results.items():
            if model_name == 'cross_model_analysis' or 'error_clustering' not in results:
                continue
                
            clustering = results['error_clustering']
            if 'cluster_analysis' in clustering:
                print(f"\n{model_name}:")
                print(f"  Number of Error Clusters: {clustering['n_clusters']}")
                
                for cluster_name, cluster_info in clustering['cluster_analysis'].items():
                    print(f"  {cluster_name}: {cluster_info['size']} samples "
                          f"({cluster_info['percentage']:.1f}% of errors)")
                    print(f"    Top features: {', '.join(cluster_info['top_features'][:5])}")
                    print(f"    FP: {cluster_info['false_positives']}, FN: {cluster_info['false_negatives']}")
        
        # 5. Cross-Model Analysis
        if 'cross_model_analysis' in analysis_results:
            print("\n\n5. CROSS-MODEL ERROR ANALYSIS:")
            print("-" * 50)
            
            cross_analysis = analysis_results['cross_model_analysis']
            agreement_pct = cross_analysis['agreement_percentages']
            
            print(f"  Unanimous Correct: {agreement_pct['unanimous_correct']:.1f}%")
            print(f"  Unanimous Incorrect: {agreement_pct['unanimous_incorrect']:.1f}%")
            print(f"  Majority Correct: {agreement_pct['majority_correct']:.1f}%")
            print(f"  Majority Incorrect: {agreement_pct['majority_incorrect']:.1f}%")
            print(f"  Split Decisions: {agreement_pct['split_decisions']:.1f}%")
            
            print("\n  Pairwise Model Agreement:")
            agreement_df = cross_analysis['pairwise_agreement_df']
            print(agreement_df.round(4).to_string())
        
        # 6. Key Insights and Recommendations
        print("\n\n6. KEY INSIGHTS AND RECOMMENDATIONS:")
        print("-" * 50)
        
        insights = self._generate_insights(analysis_results)
        for insight in insights:
            print(f"  • {insight}")
    
    def _generate_insights(self, analysis_results):
        """
        Generate key insights from error analysis
        """
        insights = []
        
        # Analyze error rates
        error_rates = []
        for model_name, results in analysis_results.items():
            if model_name != 'cross_model_analysis':
                error_rates.append(results['error_rate'])
        
        if error_rates:
            best_error_rate = min(error_rates)
            worst_error_rate = max(error_rates)
            
            if worst_error_rate - best_error_rate > 0.05:
                insights.append(f"Significant performance gap between models ({worst_error_rate-best_error_rate:.3f})")
        
        # Analyze confidence patterns
        for model_name, results in analysis_results.items():
            if model_name == 'cross_model_analysis' or 'confidence_analysis' not in results:
                continue
                
            conf_analysis = results['confidence_analysis']
            high_conf_errors = conf_analysis['high_confidence_errors']['percentage']
            
            if high_conf_errors > 20:
                insights.append(f"{model_name}: High rate of confident errors ({high_conf_errors:.1f}%) suggests overconfidence")
        
        # Cross-model insights
        if 'cross_model_analysis' in analysis_results:
            cross_analysis = analysis_results['cross_model_analysis']
            unanimous_incorrect = cross_analysis['agreement_percentages']['unanimous_incorrect']
            
            if unanimous_incorrect > 5:
                insights.append(f"High rate of unanimous errors ({unanimous_incorrect:.1f}%) indicates systematic blind spots")
        
        if not insights:
            insights.append("Error patterns appear normal with no major systematic issues detected")
        
        return insights
    
    def plot_error_visualizations(self, analysis_results, save_path=None):
        """
        Generate comprehensive error analysis visualizations
        """
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        
        model_names = [name for name in analysis_results.keys() if name != 'cross_model_analysis']
        
        # 1. Error rate comparison
        ax1 = axes[0, 0]
        error_rates = [analysis_results[name]['error_rate'] for name in model_names]
        bars = ax1.bar(model_names, error_rates, color=plt.cm.Set3(np.linspace(0, 1, len(model_names))))
        ax1.set_title('Error Rate Comparison')
        ax1.set_ylabel('Error Rate')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, rate in zip(bars, error_rates):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{rate:.3f}', ha='center', va='bottom')
        
        # 2. Confusion matrices
        ax2 = axes[0, 1]
        if model_names:
            cm = analysis_results[model_names[0]]['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2)
            ax2.set_title(f'Confusion Matrix - {model_names[0]}')
            ax2.set_xlabel('Predicted')
            ax2.set_ylabel('Actual')
        
        # 3. Confidence distribution
        ax3 = axes[0, 2]
        for i, model_name in enumerate(model_names[:3]):  # Max 3 models
            if 'confidence_analysis' in analysis_results[model_name]:
                conf_analysis = analysis_results[model_name]['confidence_analysis']
                correct_conf = conf_analysis['correct_confidence']
                error_conf = conf_analysis['error_confidence']
                
                ax3.hist(correct_conf, alpha=0.5, label=f'{model_name} Correct', bins=20)
                ax3.hist(error_conf, alpha=0.5, label=f'{model_name} Errors', bins=20)
        
        ax3.set_title('Confidence Distribution')
        ax3.set_xlabel('Confidence Score')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        
        # 4. FP vs FN rates
        ax4 = axes[1, 0]
        fp_rates = [analysis_results[name]['fp_rate'] for name in model_names]
        fn_rates = [analysis_results[name]['fn_rate'] for name in model_names]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        ax4.bar(x - width/2, fp_rates, width, label='False Positive Rate', alpha=0.8)
        ax4.bar(x + width/2, fn_rates, width, label='False Negative Rate', alpha=0.8)
        ax4.set_title('False Positive vs False Negative Rates')
        ax4.set_ylabel('Rate')
        ax4.set_xticks(x)
        ax4.set_xticklabels(model_names, rotation=45)
        ax4.legend()
        
        # 5. Cross-model agreement heatmap
        ax5 = axes[1, 1]
        if 'cross_model_analysis' in analysis_results:
            agreement_matrix = analysis_results['cross_model_analysis']['pairwise_agreement']
            sns.heatmap(agreement_matrix, annot=True, fmt='.3f', cmap='YlOrRd', 
                       xticklabels=model_names, yticklabels=model_names, ax=ax5)
            ax5.set_title('Model Agreement Matrix')
        
        # 6. Error clustering visualization
        ax6 = axes[1, 2]
        if model_names and 'error_clustering' in analysis_results[model_names[0]]:
            clustering = analysis_results[model_names[0]]['error_clustering']
            if 'cluster_analysis' in clustering:
                cluster_sizes = [info['size'] for info in clustering['cluster_analysis'].values()]
                cluster_labels = list(clustering['cluster_analysis'].keys())
                
                ax6.pie(cluster_sizes, labels=cluster_labels, autopct='%1.1f%%')
                ax6.set_title(f'Error Clusters - {model_names[0]}')
        
        # 7. Linguistic features comparison
        ax7 = axes[2, 0]
        if model_names and 'linguistic_analysis' in analysis_results[model_names[0]]:
            ling_analysis = analysis_results[model_names[0]]['linguistic_analysis']
            
            features = ['avg_length', 'avg_word_count', 'avg_caps_ratio']
            error_values = [ling_analysis['error_text_stats'].get(f, 0) for f in features]
            correct_values = [ling_analysis['correct_text_stats'].get(f, 0) for f in features]
            
            x = np.arange(len(features))
            width = 0.35
            
            ax7.bar(x - width/2, error_values, width, label='Errors', alpha=0.8)
            ax7.bar(x + width/2, correct_values, width, label='Correct', alpha=0.8)
            ax7.set_title('Linguistic Features Comparison')
            ax7.set_xticks(x)
            ax7.set_xticklabels(['Length', 'Words', 'Caps Ratio'], rotation=45)
            ax7.legend()
        
        # 8. Sentiment analysis
        ax8 = axes[2, 1]
        if model_names and 'linguistic_analysis' in analysis_results[model_names[0]]:
            ling_analysis = analysis_results[model_names[0]]['linguistic_analysis']
            
            error_sent = ling_analysis['sentiment_analysis']['error_sentiment']
            correct_sent = ling_analysis['sentiment_analysis']['correct_sentiment']
            
            sentiments = ['avg_positive', 'avg_negative', 'avg_neutral']
            error_sent_values = [error_sent.get(s, 0) for s in sentiments]
            correct_sent_values = [correct_sent.get(s, 0) for s in sentiments]
            
            x = np.arange(len(sentiments))
            width = 0.35
            
            ax8.bar(x - width/2, error_sent_values, width, label='Errors', alpha=0.8)
            ax8.bar(x + width/2, correct_sent_values, width, label='Correct', alpha=0.8)
            ax8.set_title('Sentiment Analysis Comparison')
            ax8.set_xticks(x)
            ax8.set_xticklabels(['Positive', 'Negative', 'Neutral'])
            ax8.legend()
        
        # 9. Model agreement breakdown
        ax9 = axes[2, 2]
        if 'cross_model_analysis' in analysis_results:
            agreement_pct = analysis_results['cross_model_analysis']['agreement_percentages']
            
            categories = list(agreement_pct.keys())
            values = list(agreement_pct.values())
            
            colors = ['green', 'red', 'lightgreen', 'lightcoral', 'yellow']
            ax9.pie(values, labels=categories, autopct='%1.1f%%', colors=colors[:len(values)])
            ax9.set_title('Cross-Model Agreement Breakdown')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f"{save_path}_error_analysis.png", dpi=300, bbox_inches='tight')
        
        plt.show()
        return fig

if __name__ == "__main__":
    # Example usage
    print("Error Analysis and Interpretability Framework")
    print("============================================")
    
    # This would typically be called with real model predictions and data
    print("\n✓ Error Analysis Framework ready for use!")
    print("\nUsage:")
    print("1. error_analyzer = ErrorAnalysisFramework()")
    print("2. results = error_analyzer.comprehensive_error_analysis(predictions, X_test, y_test, text_data)")
    print("3. error_analyzer.generate_error_analysis_report(results)")
    print("4. error_analyzer.plot_error_visualizations(results)")