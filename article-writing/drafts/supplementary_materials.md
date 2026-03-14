# Supplementary Materials

## Multi-Stage Ensemble and Deep Learning for Fake News Detection: A Hierarchical Feature Fusion Approach

This document contains additional figures, tables, and extended analyses that support the findings presented in the main manuscript. These materials provide deeper insights into the ensemble architecture, hyperparameter sensitivity, and high-dimensional feature spaces utilized in our study.

---

## Section S1: Extended Model Analysis

### Figure S1: Meta-Ensemble Architecture Analysis
> **Figure S1:** `../results/supplementary_figures/S1_Ensemble_Analysis.png`  
> **Description:** This figure provides a detailed breakdown of the meta-ensemble's decision-making process. It illustrates the correlation between the base classifiers' predictions (Stage 1 and Stage 2 models) and the final meta-learner (Logistic Regression) output. High correlation among diverse models (e.g., CNN and Gradient Boosting) confirms the effectiveness of capturing complementary linguistic signals.

### Figure S2: Dimensionality Reduction and Feature Space
> **Figure S2:** `../results/supplementary_figures/S2_Dimensionality_Analysis.png`  
> **Description:** A visualization of the high-dimensional feature space (10,034 features reduced to 5,000 via Mutual Information) projected into a 2D/3D space using t-SNE and PCA. The clear separation between "Real" and "Fake" clusters demonstrates the discriminative power of the combined TF-IDF and handcrafted (Lexical, Syntactic, Semantic, Stylometric) feature set prior to any model training.

### Figure S3: Extended Validation Curves
> **Figure S3:** `../results/supplementary_figures/S3_Validation_Curves.png`  
> **Description:** Extended validation curves for key hyperparameters across the primary traditional models (e.g., tree depth in Random Forest, C parameter in SVM, and learning rate in Gradient Boosting). The shaded regions represent the 95% confidence intervals across 10-fold cross-validation, illustrating the bias-variance tradeoff and confirming the optimal hyperparameter selections described in the methodology.

### Figure S4: Comprehensive Model Comparison
> **Figure S4:** `../results/supplementary_figures/S4_Comprehensive_Comparison.png`  
> **Description:** A broad comparative visualization incorporating all 9 individual models alongside the meta-ensemble. This expands upon Table III and IV from the main text by visually representing the precision-recall curves and ROC curves for every evaluated configuration, highlighting the strict dominance of the Hybrid DL and Meta-Ensemble approaches in the high-recall regime.

---

## Section S2: Extended Data Tables

### Table S1: Comprehensive Handcrafted Feature Dictionary
The following table details the 33 custom-engineered features extracted across four linguistic levels (Lexical, Syntactic, Semantic, and Stylometric) before combination with the TF-IDF matrix. These features are specifically designed to capture the nuanced differences in writing style, emotional tone, and syntactic structure between credible journalism and fabricated content.

| Linguistic Level | Feature Name | Description | Mathematical Formulation / Implementation Note |
|---|---|---|---|
| Lexical | `word_count` | Total number of tokens in the document | $\|W\|$ |
| Lexical | `char_count` | Total number of characters | $\sum_{w \in W} \text{len}(w)$ |
| Lexical | `avg_word_length` | Average character length of words | `char_count` / `word_count` |
| Lexical | `sentence_count` | Total number of sentences | $\|S\|$ |
| Lexical | `avg_sentence_length`| Average number of words per sentence | `word_count` / `sentence_count` |
| Lexical | `TTR` | Type-Token Ratio (Vocabulary richness) | $\|\{w : w \in W\}\| / \|W\|$ |
| Lexical | `stopword_ratio` | Proportion of words that are stopwords | $\|\{w \in W : w \in \mathcal{SW}\}\| / \|W\|$ |
| Lexical | `capital_ratio` | Proportion of uppercase characters | $\|\text{uppercase chars}\| / \text{char\_count}$ |
| Lexical | `punctuation_density`| Proportion of punctuation marks | $\|\text{punctuation chars}\| / \text{char\_count}$ |
| Lexical | `number_ratio` | Proportion of numeric tokens | $\|\text{numeric tokens}\| / \|W\|$ |
| Syntactic | `noun_ratio` | Proportion of noun POS tags (NN, NNS, NNP, NNPS) | $\rho_N$ |
| Syntactic | `verb_ratio` | Proportion of verb POS tags (VB, VBD, VBG, VBN, VBP, VBZ) | $\rho_V$ |
| Syntactic | `adj_ratio` | Proportion of adjective POS tags (JJ, JJR, JJS) | $\rho_{ADJ}$ |
| Syntactic | `adv_ratio` | Proportion of adverb POS tags (RB, RBR, RBS) | $\rho_{ADV}$ |
| Syntactic | `pronoun_ratio` | Proportion of pronoun POS tags (PRP, PRP$) | $\rho_{PRO}$ |
| Syntactic | `det_ratio` | Proportion of determiner POS tags (DT, PDT, WDT) | $\rho_{DET}$ |
| Syntactic | `prep_ratio` | Proportion of preposition POS tags (IN) | $\rho_{PREP}$ |
| Syntactic | `pos_diversity` | Unique POS tags normalized by document length | $\delta_{POS}$ |
| Semantic | `vader_compound` | Normalized, weighted composite sentiment score | $v_c \in [-1,1]$ |
| Semantic | `vader_pos` | Proportion of text falling in positive sentiment | $v_+ \in [0,1]$ |
| Semantic | `vader_neg` | Proportion of text falling in negative sentiment | $v_- \in [0,1]$ |
| Semantic | `vader_neu` | Proportion of text falling in neutral sentiment | $v_0 \in [0,1]$ |
| Semantic | `entity_count` | Total number of Named Entities extracted via NLTK NER | $\|\text{Entities}\|$ |
| Semantic | `entity_density` | Named Entities normalized by word count | `entity_count` / `word_count` |
| Semantic | `entity_diversity` | Number of unique entity types (PERSON, ORG, etc.) | $\|\text{Unique Entity Types}\|$ |
| Semantic | `emotional_intensity`| Aggregate score of sensationalist markers | $(\|!\| + \|?\| + \|\text{ALL\_CAPS}\|) / \|W\|$ |
| Stylometric| `flesch_kincaid` | Flesch-Kincaid Readability Grade Level | $0.39(\|W\|/\|S\|) + 11.8(\text{Syllables}/\|W\|) - 15.59$ |
| Stylometric| `ari` | Automated Readability Index | $4.71(\text{char\_count}/\|W\|) + 0.5(\|W\|/\|S\|) - 21.43$ |
| Stylometric| `flesch_ease` | Reading ease score (higher is easier) | $206.835 - 1.015(\|W\|/\|S\|) - 84.6(\text{Syllables}/\|W\|)$ |
| Stylometric| `yules_k` | Lexical diversity measure independent of length | $10^4 \times (\sum(f_r r^2) - \|W\|) / \|W\|^2$ |
| Stylometric| `function_ratio` | Ratio of structural grammatical words to total words | $\|\text{function words}\| / \|W\|$ |
| Stylometric| `sentence_len_std` | Standard deviation of sentence lengths (pacing) | $\sigma_{sl}$ |
| Stylometric| `bigram_repetition` | Count of sequentially repeated identical bigrams | $\sum \mathbb{I}(w_i w_{i+1} = w_{i+2} w_{i+3})$ |

<br>

### Table S2: Extended Hyperparameter Search Space and Optimal Configurations
The table below documents the hyperparameter optimization process across all base learners and deep learning architectures. A randomized grid search with 5-fold cross-validation was used for Stage 1 and Stage 2 models, while Stage 3 neural networks utilized learning rate schedulers and early stopping based on a 15% validation split.

| Model Stage | Algorithm | Hyperparameter | Search Space | Optimal Value | Rationale / Effect |
|---|---|---|---|---|---|
| Stage 1 | Random Forest | `n_estimators` | [50, 100, 200, 500] | 200 | Balances ensemble diversity with computational cost |
| Stage 1 | Random Forest | `max_depth` | [10, 20, 30, None] | 30 | Prevents overfitting on high-dimensional text data |
| Stage 1 | Random Forest | `min_samples_split` | [2, 5, 10] | 5 | Regularization to smooth leaf node predictions |
| Stage 1 | SVM (RBF) | `C` (Regularization) | [0.1, 1, 10, 100] | 10 | Higher C penalizes misclassifications more strictly |
| Stage 1 | SVM (RBF) | `gamma` | ['scale', 'auto', 0.1, 0.01] | 'scale' | Adapts kernel coefficient to feature variance |
| Stage 1 | Naive Bayes | `alpha` (Smoothing) | [0.1, 0.5, 1.0, 2.0] | 1.0 | Standard Laplace smoothing to handle zero probabilities |
| Stage 2 | Gradient Boosting | `n_estimators` | [100, 200, 300] | 200 | Optimal point before validation curve plateaus |
| Stage 2 | Gradient Boosting | `learning_rate` | [0.01, 0.05, 0.1, 0.2] | 0.1 | Shrinkage parameter balancing step size and estimators |
| Stage 2 | Gradient Boosting | `max_depth` | [3, 5, 7] | 3 | Shallow trees prevent overfitting on residuals |
| Stage 2 | Logistic Regression | `C` (Inverse Reg.)| [0.01, 0.1, 1.0, 10.0] | 1.0 | Standard L2 penalty strength |
| Stage 2 | Logistic Regression | `solver` | ['lbfgs', 'saga', 'liblinear']| 'lbfgs' | Efficient optimization for dense feature matrices |
| Stage 2 | MLP Classifier | `hidden_layer_sizes`| [(50,), (100,), (100, 50)] | (100,) | Single wide layer prevents rapid overfitting on features |
| Stage 2 | MLP Classifier | `alpha` (L2 penalty) | [0.0001, 0.001, 0.01] | 0.0001 | Standard weight decay |
| Stage 3 | CNN (1D) | `filters` | [64, 128, 256] | 128 | Sufficient capacity for n-gram feature extraction |
| Stage 3 | CNN (1D) | `kernel_size` | [3, 5, 7] | 5 | Captures local syntactic dependencies effectively |
| Stage 3 | LSTM | `units` | [32, 64, 128] | 64 | Maintains sequence state without vanishing gradients |
| Stage 3 | LSTM | `dropout` | [0.1, 0.2, 0.3, 0.5] | 0.3 | High dropout needed to regularize recurrent connections |
| Stage 3 | DL Common | `batch_size` | [16, 32, 64, 128] | 32 | Smaller batches provide better generalization |
| Stage 3 | DL Common | `learning_rate` | [1e-4, 5e-4, 1e-3, 5e-3] | 1e-3 | Adam optimizer default, adjusted by ReduceLROnPlateau |
| Meta | Logistic Regression | `C` | [0.1, 0.5, 1.0, 5.0] | 0.5 | Stronger regularization for meta-features to avoid leakage|
| Meta | Logistic Regression | `class_weight` | [None, 'balanced'] | 'balanced' | Ensures equal sensitivity across True/Fake classes |

---

## Section S3: Reproduction and Code Availability

To ensure full transparency and reproducibility of our results, all source code, preprocessing scripts, and training configurations are publicly available in the accompanying repository. 

**Repository Link:** [Insert GitHub/Zenodo Link Here]

The repository includes:
1. **Data Preprocessing Pipeline:** Implementation of the 3-stage cleaning and 4-level feature engineering pipeline (`data/preprocessor.py`, `data/advanced_feature_engineering.py`).
2. **Model Training Scripts:** Scripts for training traditional models, deep learning architectures, and the final meta-ensemble.
3. **Visualization Generation:** Code to perfectly recreate all figures found in the main manuscript and this supplementary document (`generate_ieee_figures.py`).
4. **Environment Configuration:** A `requirements.txt` file ensuring exact version matching for critical libraries (e.g., `scikit-learn`, `tensorflow`, `nltk`).

For detailed step-by-step instructions on setting up the environment and executing the analysis pipeline, please refer to the `README.md` file located in the root of the reproduction repository.
