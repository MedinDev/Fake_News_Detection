import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import SelectKBest, mutual_info_classif, chi2
from sklearn.decomposition import LatentDirichletAllocation
from textstat import flesch_reading_ease, flesch_kincaid_grade, automated_readability_index
from collections import Counter
import spacy
from scipy import sparse
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class AdvancedFeatureEngineering:
    """
    Multi-level Feature Engineering Pipeline for Fake News Detection
    
    Implements 4-level feature extraction:
    - Level 1: Lexical features (word frequency, vocabulary richness)
    - Level 2: Syntactic features (POS patterns, dependency structures)
    - Level 3: Semantic features (topic modeling, sentiment analysis)
    - Level 4: Stylometric features (writing style, readability metrics)
    """
    
    def __init__(self, max_features=10000, ngram_range=(1, 3)):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.tfidf_vectorizer = None
        self.count_vectorizer = None
        self.lda_model = None
        self.feature_selector = None
        self.scaler = StandardScaler()
        self.sia = SentimentIntensityAnalyzer()
        
        # Initialize NLP models
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Warning: spaCy model not found. Some features will be unavailable.")
            self.nlp = None
        
        # Download required NLTK data
        self._download_nltk_data()
    
    def _download_nltk_data(self):
        """Download required NLTK datasets"""
        required_data = ['punkt', 'stopwords', 'averaged_perceptron_tagger', 
                        'maxent_ne_chunker', 'words', 'vader_lexicon']
        
        for data in required_data:
            try:
                nltk.data.find(f'tokenizers/{data}')
            except LookupError:
                try:
                    nltk.download(data, quiet=True)
                except:
                    pass
    
    def multi_stage_text_cleaning(self, text):
        """
        Multi-stage text cleaning pipeline
        """
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # Stage 1: Basic cleaning
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\@\w+|\#\w+', '', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Stage 2: Normalize whitespace and case
        text = re.sub(r'\s+', ' ', text).strip().lower()
        
        # Stage 3: Remove excessive repetition
        text = re.sub(r'(.)\1{3,}', r'\1\1', text)
        
        return text
    
    def extract_lexical_features(self, texts):
        """
        Level 1: Lexical Feature Extraction
        """
        features = []
        
        for text in texts:
            if pd.isna(text):
                text = ""
            
            words = word_tokenize(str(text).lower())
            sentences = sent_tokenize(str(text))
            
            # Basic lexical statistics
            word_count = len(words)
            sentence_count = len(sentences)
            char_count = len(str(text))
            
            # Vocabulary richness
            unique_words = len(set(words))
            type_token_ratio = unique_words / max(word_count, 1)
            
            # Average word/sentence length
            avg_word_length = np.mean([len(word) for word in words]) if words else 0
            avg_sentence_length = word_count / max(sentence_count, 1)
            
            # Stopword ratio
            stop_words = set(stopwords.words('english'))
            stopword_count = sum(1 for word in words if word in stop_words)
            stopword_ratio = stopword_count / max(word_count, 1)
            
            # Punctuation density
            punct_count = sum(1 for char in str(text) if char in '.,!?;:')
            punct_density = punct_count / max(char_count, 1)
            
            # Capital letter ratio
            capital_count = sum(1 for char in str(text) if char.isupper())
            capital_ratio = capital_count / max(char_count, 1)
            
            features.append([
                word_count, sentence_count, char_count, unique_words,
                type_token_ratio, avg_word_length, avg_sentence_length,
                stopword_ratio, punct_density, capital_ratio
            ])
        
        feature_names = [
            'word_count', 'sentence_count', 'char_count', 'unique_words',
            'type_token_ratio', 'avg_word_length', 'avg_sentence_length',
            'stopword_ratio', 'punct_density', 'capital_ratio'
        ]
        
        return np.array(features), feature_names
    
    def extract_syntactic_features(self, texts):
        """
        Level 2: Syntactic Feature Extraction
        """
        features = []
        
        for text in texts:
            if pd.isna(text):
                text = ""
            
            words = word_tokenize(str(text).lower())
            
            # POS tagging
            pos_tags = pos_tag(words)
            pos_counts = Counter([tag for word, tag in pos_tags])
            
            # Key POS ratios
            total_words = len(words)
            noun_ratio = (pos_counts.get('NN', 0) + pos_counts.get('NNS', 0) + 
                         pos_counts.get('NNP', 0) + pos_counts.get('NNPS', 0)) / max(total_words, 1)
            
            verb_ratio = (pos_counts.get('VB', 0) + pos_counts.get('VBD', 0) + 
                         pos_counts.get('VBG', 0) + pos_counts.get('VBN', 0) + 
                         pos_counts.get('VBP', 0) + pos_counts.get('VBZ', 0)) / max(total_words, 1)
            
            adj_ratio = (pos_counts.get('JJ', 0) + pos_counts.get('JJR', 0) + 
                        pos_counts.get('JJS', 0)) / max(total_words, 1)
            
            adv_ratio = (pos_counts.get('RB', 0) + pos_counts.get('RBR', 0) + 
                        pos_counts.get('RBS', 0)) / max(total_words, 1)
            
            # Function word ratios
            pronoun_ratio = (pos_counts.get('PRP', 0) + pos_counts.get('PRP$', 0)) / max(total_words, 1)
            determiner_ratio = pos_counts.get('DT', 0) / max(total_words, 1)
            preposition_ratio = pos_counts.get('IN', 0) / max(total_words, 1)
            
            # Syntactic complexity
            pos_diversity = len(set([tag for word, tag in pos_tags])) / max(total_words, 1)
            
            features.append([
                noun_ratio, verb_ratio, adj_ratio, adv_ratio,
                pronoun_ratio, determiner_ratio, preposition_ratio, pos_diversity
            ])
        
        feature_names = [
            'noun_ratio', 'verb_ratio', 'adj_ratio', 'adv_ratio',
            'pronoun_ratio', 'determiner_ratio', 'preposition_ratio', 'pos_diversity'
        ]
        
        return np.array(features), feature_names
    
    def extract_semantic_features(self, texts):
        """
        Level 3: Semantic Feature Extraction
        """
        features = []
        
        for text in texts:
            if pd.isna(text):
                text = ""
            
            # Sentiment analysis
            sentiment_scores = self.sia.polarity_scores(str(text))
            compound_sentiment = sentiment_scores['compound']
            positive_sentiment = sentiment_scores['pos']
            negative_sentiment = sentiment_scores['neg']
            neutral_sentiment = sentiment_scores['neu']
            
            # Named entity analysis
            words = word_tokenize(str(text))
            pos_tags = pos_tag(words)
            
            try:
                chunks = ne_chunk(pos_tags)
                named_entities = []
                for chunk in chunks:
                    if hasattr(chunk, 'label'):
                        named_entities.append(chunk.label())
                
                entity_count = len(named_entities)
                entity_density = entity_count / max(len(words), 1)
                unique_entities = len(set(named_entities))
                entity_diversity = unique_entities / max(entity_count, 1)
            except:
                entity_count = entity_density = unique_entities = entity_diversity = 0
            
            # Emotional intensity (based on exclamation marks and caps)
            exclamation_count = str(text).count('!')
            question_count = str(text).count('?')
            caps_words = sum(1 for word in words if word.isupper())
            emotional_intensity = (exclamation_count + question_count + caps_words) / max(len(words), 1)
            
            features.append([
                compound_sentiment, positive_sentiment, negative_sentiment, neutral_sentiment,
                entity_count, entity_density, unique_entities, entity_diversity,
                emotional_intensity
            ])
        
        feature_names = [
            'compound_sentiment', 'positive_sentiment', 'negative_sentiment', 'neutral_sentiment',
            'entity_count', 'entity_density', 'unique_entities', 'entity_diversity',
            'emotional_intensity'
        ]
        
        return np.array(features), feature_names
    
    def extract_stylometric_features(self, texts):
        """
        Level 4: Stylometric Feature Extraction
        """
        features = []
        
        for text in texts:
            if pd.isna(text):
                text = ""
            
            text_str = str(text)
            
            # Readability metrics
            try:
                flesch_ease = flesch_reading_ease(text_str)
                flesch_grade = flesch_kincaid_grade(text_str)
                ari_score = automated_readability_index(text_str)
            except:
                flesch_ease = flesch_grade = ari_score = 0
            
            # Writing style indicators
            words = word_tokenize(text_str.lower())
            sentences = sent_tokenize(text_str)
            
            # Sentence structure variety
            sentence_lengths = [len(word_tokenize(sent)) for sent in sentences]
            sentence_length_std = np.std(sentence_lengths) if sentence_lengths else 0
            
            # Lexical diversity (Yule's K)
            word_freq = Counter(words)
            if len(words) > 0:
                yules_k = 10000 * (sum([freq**2 for freq in word_freq.values()]) - len(words)) / (len(words)**2)
            else:
                yules_k = 0
            
            # Function word usage patterns
            function_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
            function_word_count = sum(1 for word in words if word in function_words)
            function_word_ratio = function_word_count / max(len(words), 1)
            
            # Repetition patterns
            bigrams = [(words[i], words[i+1]) for i in range(len(words)-1)]
            bigram_repetition = len(bigrams) - len(set(bigrams)) if bigrams else 0
            
            features.append([
                flesch_ease, flesch_grade, ari_score, sentence_length_std,
                yules_k, function_word_ratio, bigram_repetition
            ])
        
        feature_names = [
            'flesch_ease', 'flesch_grade', 'ari_score', 'sentence_length_std',
            'yules_k', 'function_word_ratio', 'bigram_repetition'
        ]
        
        return np.array(features), feature_names
    
    def extract_tfidf_features(self, texts, fit=True):
        """
        Enhanced TF-IDF with optimized parameters
        """
        if fit:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                stop_words='english',
                lowercase=True,
                strip_accents='ascii',
                token_pattern=r'\b[a-zA-Z]{2,}\b',
                min_df=2,
                max_df=0.95,
                sublinear_tf=True,
                norm='l2'
            )
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        else:
            tfidf_matrix = self.tfidf_vectorizer.transform(texts)
        
        return tfidf_matrix
    
    def dynamic_feature_selection(self, X, y, k=5000):
        """
        Dynamic feature selection with mutual information ranking
        """
        if self.feature_selector is None:
            self.feature_selector = SelectKBest(score_func=mutual_info_classif, k=min(k, X.shape[1]))
            X_selected = self.feature_selector.fit_transform(X, y)
        else:
            X_selected = self.feature_selector.transform(X)
        
        return X_selected
    
    def fit_transform(self, texts, labels):
        """
        Complete feature engineering pipeline - fit and transform
        """
        print("Starting advanced feature engineering pipeline...")
        
        # Clean texts
        cleaned_texts = [self.multi_stage_text_cleaning(text) for text in texts]
        
        # Extract multi-level features
        print("Extracting Level 1: Lexical features...")
        lexical_features, lexical_names = self.extract_lexical_features(cleaned_texts)
        
        print("Extracting Level 2: Syntactic features...")
        syntactic_features, syntactic_names = self.extract_syntactic_features(cleaned_texts)
        
        print("Extracting Level 3: Semantic features...")
        semantic_features, semantic_names = self.extract_semantic_features(cleaned_texts)
        
        print("Extracting Level 4: Stylometric features...")
        stylometric_features, stylometric_names = self.extract_stylometric_features(cleaned_texts)
        
        print("Extracting TF-IDF features...")
        tfidf_features = self.extract_tfidf_features(cleaned_texts, fit=True)
        
        # Combine all features
        print("Combining multi-level features...")
        traditional_features = np.hstack([
            lexical_features, syntactic_features, 
            semantic_features, stylometric_features
        ])
        
        # Scale traditional features
        traditional_features_scaled = self.scaler.fit_transform(traditional_features)
        
        # Combine with TF-IDF
        if sparse.issparse(tfidf_features):
            traditional_sparse = sparse.csr_matrix(traditional_features_scaled)
            combined_features = sparse.hstack([traditional_sparse, tfidf_features])
        else:
            combined_features = np.hstack([traditional_features_scaled, tfidf_features.toarray()])
        
        # Dynamic feature selection
        print("Applying dynamic feature selection...")
        final_features = self.dynamic_feature_selection(combined_features, labels)
        
        # Store feature names
        self.feature_names = (lexical_names + syntactic_names + 
                            semantic_names + stylometric_names)
        
        if hasattr(self.tfidf_vectorizer, 'get_feature_names_out'):
            tfidf_names = self.tfidf_vectorizer.get_feature_names_out().tolist()
        else:
            tfidf_names = [f'tfidf_{i}' for i in range(tfidf_features.shape[1])]
        
        self.all_feature_names = self.feature_names + tfidf_names
        
        print(f"✓ Feature engineering completed!")
        print(f"  - Traditional features: {len(self.feature_names)}")
        print(f"  - TF-IDF features: {len(tfidf_names)}")
        print(f"  - Selected features: {final_features.shape[1]}")
        
        return final_features
    
    def transform(self, texts):
        """
        Transform new texts using fitted pipeline
        """
        # Clean texts
        cleaned_texts = [self.multi_stage_text_cleaning(text) for text in texts]
        
        # Extract multi-level features
        lexical_features, _ = self.extract_lexical_features(cleaned_texts)
        syntactic_features, _ = self.extract_syntactic_features(cleaned_texts)
        semantic_features, _ = self.extract_semantic_features(cleaned_texts)
        stylometric_features, _ = self.extract_stylometric_features(cleaned_texts)
        tfidf_features = self.extract_tfidf_features(cleaned_texts, fit=False)
        
        # Combine and scale
        traditional_features = np.hstack([
            lexical_features, syntactic_features, 
            semantic_features, stylometric_features
        ])
        traditional_features_scaled = self.scaler.transform(traditional_features)
        
        # Combine with TF-IDF
        if sparse.issparse(tfidf_features):
            traditional_sparse = sparse.csr_matrix(traditional_features_scaled)
            combined_features = sparse.hstack([traditional_sparse, tfidf_features])
        else:
            combined_features = np.hstack([traditional_features_scaled, tfidf_features.toarray()])
        
        # Apply feature selection
        final_features = self.feature_selector.transform(combined_features)
        
        return final_features
    
    def get_feature_importance_analysis(self, model, feature_names=None):
        """
        Analyze feature importance for interpretability
        """
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0]) if len(model.coef_.shape) > 1 else np.abs(model.coef_)
        else:
            return None
        
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(importances))]
        
        # Get selected feature names
        if hasattr(self.feature_selector, 'get_support'):
            selected_mask = self.feature_selector.get_support()
            selected_names = [name for i, name in enumerate(self.all_feature_names) if selected_mask[i]]
        else:
            selected_names = feature_names
        
        importance_df = pd.DataFrame({
            'feature': selected_names[:len(importances)],
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return importance_df

if __name__ == "__main__":
    # Example usage
    print("Advanced Feature Engineering Pipeline")
    print("=====================================")
    
    # Sample data
    sample_texts = [
        "This is a real news article with factual information.",
        "BREAKING: Shocking revelation that will change everything!!!",
        "Scientists have discovered a new method for renewable energy."
    ]
    sample_labels = [1, 0, 1]  # 1 = real, 0 = fake
    
    # Initialize feature engineering
    feature_engineer = AdvancedFeatureEngineering(max_features=1000)
    
    # Extract features
    features = feature_engineer.fit_transform(sample_texts, sample_labels)
    
    print(f"\nExtracted features shape: {features.shape}")
    print("Feature engineering pipeline ready for model training!")