import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import ttest_rel, wilcoxon, friedmanchisquare, kruskal
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.contingency_tables import mcnemar
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class StatisticalSignificanceTesting:
    """
    Comprehensive Statistical Significance Testing Framework
    
    Features:
    - Cross-validation with statistical rigor
    - Paired statistical tests (t-test, Wilcoxon)
    - Multiple comparison corrections (Bonferroni, Holm)
    - Effect size calculations (Cohen's d, Cliff's delta)
    - ANOVA and post-hoc analysis
    - McNemar's test for classifier comparison
    - Bootstrap confidence intervals
    """
    
    def __init__(self, cv_folds=10, cv_repeats=5, alpha=0.05, random_state=42):
        self.cv_folds = cv_folds
        self.cv_repeats = cv_repeats
        self.alpha = alpha
        self.random_state = random_state
        
        # Results storage
        self.cv_results = defaultdict(dict)
        self.statistical_tests = {}
        self.effect_sizes = {}
        self.confidence_intervals = {}
        
    def repeated_cross_validation(self, models, X, y, scoring_metrics=None):
        """
        Perform repeated stratified cross-validation for robust evaluation
        """
        if scoring_metrics is None:
            scoring_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        
        print(f"Performing {self.cv_repeats}x{self.cv_folds}-fold cross-validation...")
        
        # Initialize repeated stratified k-fold
        rskf = RepeatedStratifiedKFold(
            n_splits=self.cv_folds,
            n_repeats=self.cv_repeats,
            random_state=self.random_state
        )
        
        results = defaultdict(lambda: defaultdict(list))
        
        for model_name, model in models.items():
            print(f"  Evaluating {model_name}...")
            
            for metric in scoring_metrics:
                scores = cross_val_score(
                    model, X, y, cv=rskf, scoring=metric, n_jobs=-1
                )
                results[model_name][metric] = scores
                
                # Store summary statistics
                self.cv_results[model_name][metric] = {
                    'scores': scores,
                    'mean': np.mean(scores),
                    'std': np.std(scores),
                    'min': np.min(scores),
                    'max': np.max(scores),
                    'median': np.median(scores),
                    'q25': np.percentile(scores, 25),
                    'q75': np.percentile(scores, 75)
                }
        
        print("✓ Cross-validation completed")
        return dict(results)
    
    def paired_statistical_tests(self, cv_results, metric='accuracy'):
        """
        Perform paired statistical tests between all model pairs
        """
        print(f"\nPerforming paired statistical tests for {metric}...")
        
        model_names = list(cv_results.keys())
        n_models = len(model_names)
        
        # Initialize result matrices
        ttest_pvalues = np.zeros((n_models, n_models))
        wilcoxon_pvalues = np.zeros((n_models, n_models))
        effect_sizes = np.zeros((n_models, n_models))
        
        test_results = {
            'ttest_pvalues': pd.DataFrame(index=model_names, columns=model_names),
            'wilcoxon_pvalues': pd.DataFrame(index=model_names, columns=model_names),
            'effect_sizes': pd.DataFrame(index=model_names, columns=model_names),
            'significant_pairs': []
        }
        
        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names):
                if i != j:
                    scores1 = cv_results[model1][metric]
                    scores2 = cv_results[model2][metric]
                    
                    # Paired t-test
                    t_stat, t_pval = ttest_rel(scores1, scores2)
                    ttest_pvalues[i, j] = t_pval
                    
                    # Wilcoxon signed-rank test (non-parametric)
                    try:
                        w_stat, w_pval = wilcoxon(scores1, scores2)
                        wilcoxon_pvalues[i, j] = w_pval
                    except ValueError:
                        wilcoxon_pvalues[i, j] = 1.0
                    
                    # Cohen's d (effect size)
                    pooled_std = np.sqrt(((len(scores1) - 1) * np.var(scores1, ddof=1) + 
                                         (len(scores2) - 1) * np.var(scores2, ddof=1)) / 
                                        (len(scores1) + len(scores2) - 2))
                    cohens_d = (np.mean(scores1) - np.mean(scores2)) / pooled_std
                    effect_sizes[i, j] = cohens_d
                    
                    # Store in DataFrames
                    test_results['ttest_pvalues'].loc[model1, model2] = t_pval
                    test_results['wilcoxon_pvalues'].loc[model1, model2] = w_pval
                    test_results['effect_sizes'].loc[model1, model2] = cohens_d
                    
                    # Check significance
                    if t_pval < self.alpha:
                        test_results['significant_pairs'].append({
                            'model1': model1,
                            'model2': model2,
                            'ttest_pvalue': t_pval,
                            'wilcoxon_pvalue': w_pval,
                            'cohens_d': cohens_d,
                            'effect_magnitude': self._interpret_effect_size(abs(cohens_d))
                        })
        
        self.statistical_tests[metric] = test_results
        return test_results
    
    def multiple_comparison_correction(self, pvalues, method='bonferroni'):
        """
        Apply multiple comparison correction
        """
        from statsmodels.stats.multitest import multipletests
        
        # Flatten p-values (excluding diagonal)
        flat_pvalues = []
        for i in range(len(pvalues)):
            for j in range(len(pvalues)):
                if i != j:
                    flat_pvalues.append(pvalues.iloc[i, j])
        
        # Apply correction
        rejected, corrected_pvalues, _, _ = multipletests(
            flat_pvalues, alpha=self.alpha, method=method
        )
        
        # Reshape back to matrix
        corrected_matrix = pd.DataFrame(
            index=pvalues.index, columns=pvalues.columns, dtype=float
        )
        
        idx = 0
        for i in range(len(pvalues)):
            for j in range(len(pvalues)):
                if i != j:
                    corrected_matrix.iloc[i, j] = corrected_pvalues[idx]
                    idx += 1
                else:
                    corrected_matrix.iloc[i, j] = 1.0
        
        return corrected_matrix, rejected
    
    def anova_analysis(self, cv_results, metric='accuracy'):
        """
        Perform ANOVA and post-hoc analysis
        """
        print(f"\nPerforming ANOVA analysis for {metric}...")
        
        # Prepare data for ANOVA
        all_scores = []
        model_labels = []
        
        for model_name, results in cv_results.items():
            scores = results[metric]
            all_scores.extend(scores)
            model_labels.extend([model_name] * len(scores))
        
        # One-way ANOVA
        model_groups = [cv_results[model][metric] for model in cv_results.keys()]
        f_stat, anova_pval = stats.f_oneway(*model_groups)
        
        # Kruskal-Wallis test (non-parametric alternative)
        kw_stat, kw_pval = kruskal(*model_groups)
        
        anova_results = {
            'f_statistic': f_stat,
            'anova_pvalue': anova_pval,
            'kruskal_wallis_stat': kw_stat,
            'kruskal_wallis_pvalue': kw_pval,
            'significant': anova_pval < self.alpha
        }
        
        # Post-hoc analysis (Tukey HSD) if ANOVA is significant
        if anova_pval < self.alpha:
            print("  Performing Tukey HSD post-hoc analysis...")
            
            # Prepare data for Tukey HSD
            df_tukey = pd.DataFrame({
                'score': all_scores,
                'model': model_labels
            })
            
            tukey_results = pairwise_tukeyhsd(
                endog=df_tukey['score'],
                groups=df_tukey['model'],
                alpha=self.alpha
            )
            
            anova_results['tukey_hsd'] = {
                'summary': str(tukey_results),
                'pairwise_comparisons': []
            }
            
            # Parse Tukey results
            for i in range(len(tukey_results.groupsunique)):
                for j in range(i+1, len(tukey_results.groupsunique)):
                    group1 = tukey_results.groupsunique[i]
                    group2 = tukey_results.groupsunique[j]
                    
                    # Find the comparison in results
                    comparison_idx = None
                    for idx, (g1, g2) in enumerate(zip(tukey_results.data[0], tukey_results.data[1])):
                        if (g1 == group1 and g2 == group2) or (g1 == group2 and g2 == group1):
                            comparison_idx = idx
                            break
                    
                    if comparison_idx is not None:
                        anova_results['tukey_hsd']['pairwise_comparisons'].append({
                            'group1': group1,
                            'group2': group2,
                            'meandiff': tukey_results.meandiffs[comparison_idx],
                            'pvalue': tukey_results.pvalues[comparison_idx],
                            'significant': tukey_results.reject[comparison_idx]
                        })
        
        return anova_results
    
    def bootstrap_confidence_intervals(self, cv_results, metric='accuracy', n_bootstrap=1000):
        """
        Calculate bootstrap confidence intervals for model performance
        """
        print(f"\nCalculating bootstrap confidence intervals for {metric}...")
        
        confidence_intervals = {}
        
        for model_name, results in cv_results.items():
            scores = results[metric]
            
            # Bootstrap resampling
            bootstrap_means = []
            for _ in range(n_bootstrap):
                bootstrap_sample = np.random.choice(scores, size=len(scores), replace=True)
                bootstrap_means.append(np.mean(bootstrap_sample))
            
            # Calculate confidence intervals
            ci_lower = np.percentile(bootstrap_means, (1 - 0.95) / 2 * 100)
            ci_upper = np.percentile(bootstrap_means, (1 + 0.95) / 2 * 100)
            
            confidence_intervals[model_name] = {
                'mean': np.mean(scores),
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'ci_width': ci_upper - ci_lower,
                'bootstrap_means': bootstrap_means
            }
        
        self.confidence_intervals[metric] = confidence_intervals
        return confidence_intervals
    
    def mcnemar_test(self, predictions_dict, y_true):
        """
        Perform McNemar's test for comparing classifier predictions
        """
        print("\nPerforming McNemar's test for classifier comparison...")
        
        model_names = list(predictions_dict.keys())
        n_models = len(model_names)
        
        mcnemar_results = pd.DataFrame(
            index=model_names, columns=model_names, dtype=float
        )
        
        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names):
                if i != j:
                    pred1 = predictions_dict[model1]
                    pred2 = predictions_dict[model2]
                    
                    # Create contingency table
                    correct1 = (pred1 == y_true)
                    correct2 = (pred2 == y_true)
                    
                    # McNemar's table: [[both_correct, model1_correct_model2_wrong],
                    #                   [model1_wrong_model2_correct, both_wrong]]
                    both_correct = np.sum(correct1 & correct2)
                    model1_only = np.sum(correct1 & ~correct2)
                    model2_only = np.sum(~correct1 & correct2)
                    both_wrong = np.sum(~correct1 & ~correct2)
                    
                    contingency_table = np.array([[both_correct, model1_only],
                                                 [model2_only, both_wrong]])
                    
                    # McNemar's test
                    try:
                        result = mcnemar(contingency_table, exact=True)
                        mcnemar_results.loc[model1, model2] = result.pvalue
                    except:
                        mcnemar_results.loc[model1, model2] = 1.0
                else:
                    mcnemar_results.loc[model1, model2] = 1.0
        
        return mcnemar_results
    
    def _interpret_effect_size(self, cohens_d):
        """
        Interpret Cohen's d effect size
        """
        if cohens_d < 0.2:
            return 'negligible'
        elif cohens_d < 0.5:
            return 'small'
        elif cohens_d < 0.8:
            return 'medium'
        else:
            return 'large'
    
    def comprehensive_statistical_analysis(self, models, X, y, predictions_dict=None):
        """
        Run comprehensive statistical analysis
        """
        print("Comprehensive Statistical Significance Testing")
        print("=============================================")
        
        # 1. Repeated Cross-Validation
        cv_results = self.repeated_cross_validation(models, X, y)
        
        # 2. Statistical Tests for each metric
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        
        for metric in metrics:
            print(f"\n--- Analysis for {metric.upper()} ---")
            
            # Paired tests
            self.paired_statistical_tests(cv_results, metric)
            
            # ANOVA
            anova_results = self.anova_analysis(cv_results, metric)
            
            # Bootstrap CI
            self.bootstrap_confidence_intervals(cv_results, metric)
        
        # 3. McNemar's test (if predictions provided)
        mcnemar_results = None
        if predictions_dict is not None:
            mcnemar_results = self.mcnemar_test(predictions_dict, y)
        
        return {
            'cv_results': cv_results,
            'statistical_tests': self.statistical_tests,
            'confidence_intervals': self.confidence_intervals,
            'mcnemar_results': mcnemar_results
        }
    
    def generate_statistical_report(self, analysis_results):
        """
        Generate comprehensive statistical report
        """
        print("\n" + "="*80)
        print("COMPREHENSIVE STATISTICAL ANALYSIS REPORT")
        print("="*80)
        
        cv_results = analysis_results['cv_results']
        
        # 1. Cross-Validation Summary
        print("\n1. CROSS-VALIDATION SUMMARY:")
        print("-" * 50)
        
        for metric in ['accuracy', 'f1', 'roc_auc']:
            print(f"\n{metric.upper()} Results:")
            summary_data = []
            
            for model_name in cv_results.keys():
                stats = self.cv_results[model_name][metric]
                summary_data.append({
                    'Model': model_name,
                    'Mean': f"{stats['mean']:.4f}",
                    'Std': f"{stats['std']:.4f}",
                    'Min': f"{stats['min']:.4f}",
                    'Max': f"{stats['max']:.4f}",
                    'Median': f"{stats['median']:.4f}"
                })
            
            df_summary = pd.DataFrame(summary_data)
            print(df_summary.to_string(index=False))
        
        # 2. Statistical Significance Tests
        print("\n\n2. STATISTICAL SIGNIFICANCE TESTS:")
        print("-" * 50)
        
        for metric in ['accuracy', 'f1']:
            if metric in self.statistical_tests:
                print(f"\n{metric.upper()} - Paired T-Test P-values:")
                pvalues = self.statistical_tests[metric]['ttest_pvalues']
                print(pvalues.round(4).to_string())
                
                print(f"\n{metric.upper()} - Effect Sizes (Cohen's d):")
                effect_sizes = self.statistical_tests[metric]['effect_sizes']
                print(effect_sizes.round(4).to_string())
                
                # Significant pairs
                sig_pairs = self.statistical_tests[metric]['significant_pairs']
                if sig_pairs:
                    print(f"\nSignificant differences (p < {self.alpha}):")
                    for pair in sig_pairs:
                        print(f"  {pair['model1']} vs {pair['model2']}: "
                              f"p={pair['ttest_pvalue']:.4f}, d={pair['cohens_d']:.4f} ({pair['effect_magnitude']})")
        
        # 3. Confidence Intervals
        print("\n\n3. BOOTSTRAP CONFIDENCE INTERVALS (95%):")
        print("-" * 50)
        
        for metric in ['accuracy', 'f1']:
            if metric in self.confidence_intervals:
                print(f"\n{metric.upper()}:")
                ci_data = []
                
                for model_name, ci_info in self.confidence_intervals[metric].items():
                    ci_data.append({
                        'Model': model_name,
                        'Mean': f"{ci_info['mean']:.4f}",
                        'CI Lower': f"{ci_info['ci_lower']:.4f}",
                        'CI Upper': f"{ci_info['ci_upper']:.4f}",
                        'CI Width': f"{ci_info['ci_width']:.4f}"
                    })
                
                df_ci = pd.DataFrame(ci_data)
                print(df_ci.to_string(index=False))
        
        # 4. McNemar's Test Results
        if analysis_results['mcnemar_results'] is not None:
            print("\n\n4. MCNEMAR'S TEST RESULTS:")
            print("-" * 50)
            print("P-values for classifier comparison:")
            print(analysis_results['mcnemar_results'].round(4).to_string())
        
        # 5. Statistical Conclusions
        print("\n\n5. STATISTICAL CONCLUSIONS:")
        print("-" * 50)
        
        # Find best performing model
        accuracy_means = {model: self.cv_results[model]['accuracy']['mean'] 
                         for model in cv_results.keys()}
        best_model = max(accuracy_means, key=accuracy_means.get)
        
        print(f"• Best performing model: {best_model} (Accuracy: {accuracy_means[best_model]:.4f})")
        
        # Count significant differences
        if 'accuracy' in self.statistical_tests:
            n_significant = len(self.statistical_tests['accuracy']['significant_pairs'])
            total_comparisons = len(cv_results) * (len(cv_results) - 1)
            print(f"• Significant pairwise differences: {n_significant}/{total_comparisons} ({n_significant/total_comparisons*100:.1f}%)")
        
        # Confidence interval analysis
        if 'accuracy' in self.confidence_intervals:
            ci_widths = [ci_info['ci_width'] for ci_info in self.confidence_intervals['accuracy'].values()]
            avg_ci_width = np.mean(ci_widths)
            print(f"• Average confidence interval width: {avg_ci_width:.4f}")
            print(f"• Most precise model: {min(self.confidence_intervals['accuracy'], key=lambda x: self.confidence_intervals['accuracy'][x]['ci_width'])}")
    
    def plot_statistical_visualizations(self, analysis_results, save_path=None):
        """
        Generate statistical visualization plots
        """
        cv_results = analysis_results['cv_results']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Box plots for cross-validation scores
        ax1 = axes[0, 0]
        accuracy_data = [cv_results[model]['accuracy'] for model in cv_results.keys()]
        model_names = list(cv_results.keys())
        
        bp = ax1.boxplot(accuracy_data, labels=model_names, patch_artist=True)
        ax1.set_title('Cross-Validation Accuracy Distribution')
        ax1.set_ylabel('Accuracy')
        ax1.tick_params(axis='x', rotation=45)
        
        # Color the boxes
        colors = plt.cm.Set3(np.linspace(0, 1, len(model_names)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        # 2. Confidence intervals plot
        ax2 = axes[0, 1]
        if 'accuracy' in self.confidence_intervals:
            ci_data = self.confidence_intervals['accuracy']
            models = list(ci_data.keys())
            means = [ci_data[model]['mean'] for model in models]
            ci_lowers = [ci_data[model]['ci_lower'] for model in models]
            ci_uppers = [ci_data[model]['ci_upper'] for model in models]
            
            y_pos = np.arange(len(models))
            ax2.errorbar(means, y_pos, xerr=[np.array(means) - np.array(ci_lowers), 
                                           np.array(ci_uppers) - np.array(means)], 
                        fmt='o', capsize=5)
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(models)
            ax2.set_xlabel('Accuracy')
            ax2.set_title('95% Confidence Intervals')
            ax2.grid(True, alpha=0.3)
        
        # 3. P-value heatmap
        ax3 = axes[0, 2]
        if 'accuracy' in self.statistical_tests:
            pvalues = self.statistical_tests['accuracy']['ttest_pvalues'].astype(float)
            sns.heatmap(pvalues, annot=True, fmt='.3f', cmap='RdYlBu_r', 
                       ax=ax3, cbar_kws={'label': 'P-value'})
            ax3.set_title('Pairwise T-Test P-values')
        
        # 4. Effect size heatmap
        ax4 = axes[1, 0]
        if 'accuracy' in self.statistical_tests:
            effect_sizes = self.statistical_tests['accuracy']['effect_sizes'].astype(float)
            sns.heatmap(effect_sizes, annot=True, fmt='.3f', cmap='RdBu_r', 
                       center=0, ax=ax4, cbar_kws={'label': "Cohen's d"})
            ax4.set_title('Effect Sizes (Cohen\'s d)')
        
        # 5. Performance comparison across metrics
        ax5 = axes[1, 1]
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        x = np.arange(len(model_names))
        width = 0.15
        
        for i, metric in enumerate(metrics):
            if all(metric in cv_results[model] for model in model_names):
                means = [self.cv_results[model][metric]['mean'] for model in model_names]
                ax5.bar(x + i*width, means, width, label=metric.replace('_', ' ').title())
        
        ax5.set_xlabel('Models')
        ax5.set_ylabel('Score')
        ax5.set_title('Performance Across Metrics')
        ax5.set_xticks(x + width*2)
        ax5.set_xticklabels(model_names, rotation=45)
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Statistical power analysis
        ax6 = axes[1, 2]
        if 'accuracy' in self.confidence_intervals:
            ci_data = self.confidence_intervals['accuracy']
            models = list(ci_data.keys())
            ci_widths = [ci_data[model]['ci_width'] for model in models]
            
            bars = ax6.bar(models, ci_widths, color=colors[:len(models)])
            ax6.set_ylabel('Confidence Interval Width')
            ax6.set_title('Statistical Precision (CI Width)')
            ax6.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, width in zip(bars, ci_widths):
                ax6.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                        f'{width:.4f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f"{save_path}_statistical_analysis.png", dpi=300, bbox_inches='tight')
        
        plt.show()
        return fig

if __name__ == "__main__":
    # Example usage
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.naive_bayes import MultinomialNB
    
    print("Statistical Significance Testing Framework")
    print("=========================================")
    
    # Sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 50
    
    X = np.random.rand(n_samples, n_features)
    y = np.random.randint(0, 2, n_samples)
    
    # Sample models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'Naive Bayes': MultinomialNB()
    }
    
    # Initialize testing framework
    stat_tester = StatisticalSignificanceTesting(cv_folds=5, cv_repeats=3)
    
    # Run comprehensive analysis
    results = stat_tester.comprehensive_statistical_analysis(models, X, y)
    
    # Generate report
    stat_tester.generate_statistical_report(results)
    
    # Generate visualizations
    stat_tester.plot_statistical_visualizations(results, save_path="statistical_analysis")
    
    print("\n✓ Statistical significance testing completed!")