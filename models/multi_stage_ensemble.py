import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding, Conv1D, GlobalMaxPooling1D, Input, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from scipy import sparse
import joblib
import warnings
warnings.filterwarnings('ignore')

class MultiStageEnsembleFramework:
    """
    Multi-Stage Ensemble Framework for Fake News Detection
    
    Architecture:
    - Stage 1: Traditional ML Models (Random Forest, SVM, Naive Bayes)
    - Stage 2: Advanced ML Models (Gradient Boosting, Neural Networks)
    - Stage 3: Deep Learning Models (LSTM, CNN, Hybrid)
    - Stage 4: Meta-Learning Ensemble Combination
    """
    
    def __init__(self, max_vocab_size=10000, max_sequence_length=500, random_state=42):
        self.random_state = random_state
        self.max_vocab_size = max_vocab_size
        self.max_sequence_length = max_sequence_length
        
        # Initialize model stages
        self.stage1_models = {}  # Traditional ML
        self.stage2_models = {}  # Advanced ML
        self.stage3_models = {}  # Deep Learning
        self.meta_learner = None
        
        # Text preprocessing for deep learning
        self.tokenizer = None
        
        # Performance tracking
        self.model_performances = {}
        self.ensemble_weights = {}
        
        self._initialize_models()
    
    def _initialize_models(self):
        """
        Initialize all models in the ensemble framework
        """
        # Stage 1: Traditional ML Models
        self.stage1_models = {
            'random_forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                n_jobs=-1
            ),
            'svm': SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,
                random_state=self.random_state
            ),
            'naive_bayes': MultinomialNB(
                alpha=0.1
            )
        }
        
        # Stage 2: Advanced ML Models
        self.stage2_models = {
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                min_samples_split=5,
                random_state=self.random_state
            ),
            'logistic_regression': LogisticRegression(
                C=1.0,
                solver='liblinear',
                random_state=self.random_state,
                max_iter=1000
            ),
            'mlp': MLPClassifier(
                hidden_layer_sizes=(128, 64),
                activation='relu',
                solver='adam',
                alpha=0.001,
                learning_rate='adaptive',
                max_iter=500,
                random_state=self.random_state
            )
        }
    
    def _create_lstm_model(self, vocab_size, embedding_dim=128, lstm_units=64):
        """
        Create LSTM model for sequence processing
        """
        model = Sequential([
            Embedding(vocab_size, embedding_dim, input_length=self.max_sequence_length),
            LSTM(lstm_units, dropout=0.3, recurrent_dropout=0.3, return_sequences=True),
            LSTM(lstm_units//2, dropout=0.3, recurrent_dropout=0.3),
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
    
    def _create_cnn_model(self, vocab_size, embedding_dim=128, filters=128):
        """
        Create CNN model for text classification
        """
        model = Sequential([
            Embedding(vocab_size, embedding_dim, input_length=self.max_sequence_length),
            Conv1D(filters, 3, activation='relu'),
            Conv1D(filters, 3, activation='relu'),
            GlobalMaxPooling1D(),
            Dense(128, activation='relu'),
            Dropout(0.5),
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
    
    def _create_hybrid_model(self, vocab_size, feature_dim, embedding_dim=128):
        """
        Create hybrid model combining text and engineered features
        """
        # Text input branch
        text_input = Input(shape=(self.max_sequence_length,), name='text_input')
        embedding = Embedding(vocab_size, embedding_dim)(text_input)
        lstm_out = LSTM(64, dropout=0.3, recurrent_dropout=0.3)(embedding)
        
        # Feature input branch
        feature_input = Input(shape=(feature_dim,), name='feature_input')
        feature_dense = Dense(64, activation='relu')(feature_input)
        feature_dropout = Dropout(0.3)(feature_dense)
        
        # Combine branches
        combined = concatenate([lstm_out, feature_dropout])
        combined_dense = Dense(128, activation='relu')(combined)
        combined_dropout = Dropout(0.5)(combined_dense)
        final_dense = Dense(64, activation='relu')(combined_dropout)
        final_dropout = Dropout(0.3)(final_dense)
        output = Dense(1, activation='sigmoid')(final_dropout)
        
        model = Model(inputs=[text_input, feature_input], outputs=output)
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
    
    def _evaluate_model(self, model, X, y, cv_folds=5):
        """
        Comprehensive model evaluation with cross-validation
        """
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        # Cross-validation scores
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        cv_f1 = cross_val_score(model, X, y, cv=cv, scoring='f1')
        cv_precision = cross_val_score(model, X, y, cv=cv, scoring='precision')
        cv_recall = cross_val_score(model, X, y, cv=cv, scoring='recall')
        
        return {
            'accuracy_mean': cv_scores.mean(),
            'accuracy_std': cv_scores.std(),
            'f1_mean': cv_f1.mean(),
            'f1_std': cv_f1.std(),
            'precision_mean': cv_precision.mean(),
            'precision_std': cv_precision.std(),
            'recall_mean': cv_recall.mean(),
            'recall_std': cv_recall.std()
        }
    
    def fit_stage1_models(self, X_features, y, texts=None):
        """
        Train Stage 1: Traditional ML Models
        """
        print("Training Stage 1: Traditional ML Models...")
        
        # Convert sparse matrix to dense for some models
        if sparse.issparse(X_features):
            X_dense = X_features.toarray()
        else:
            X_dense = X_features
        
        stage1_predictions = np.zeros((len(y), len(self.stage1_models)))
        
        for i, (name, model) in enumerate(self.stage1_models.items()):
            print(f"  Training {name}...")
            
            # Special handling for Naive Bayes (requires non-negative features)
            if name == 'naive_bayes' and sparse.issparse(X_features):
                X_train = X_features
            else:
                X_train = X_dense
            
            # Train model
            model.fit(X_train, y)
            
            # Evaluate model
            performance = self._evaluate_model(model, X_train, y)
            self.model_performances[f'stage1_{name}'] = performance
            
            # Get predictions for meta-learning
            stage1_predictions[:, i] = model.predict_proba(X_train)[:, 1]
            
            print(f"    ✓ {name}: Accuracy = {performance['accuracy_mean']:.4f} (±{performance['accuracy_std']:.4f})")
        
        return stage1_predictions
    
    def fit_stage2_models(self, X_features, y, stage1_predictions):
        """
        Train Stage 2: Advanced ML Models
        """
        print("\nTraining Stage 2: Advanced ML Models...")
        
        # Convert sparse matrix to dense
        if sparse.issparse(X_features):
            X_dense = X_features.toarray()
        else:
            X_dense = X_features
        
        # Combine original features with Stage 1 predictions
        X_combined = np.hstack([X_dense, stage1_predictions])
        
        stage2_predictions = np.zeros((len(y), len(self.stage2_models)))
        
        for i, (name, model) in enumerate(self.stage2_models.items()):
            print(f"  Training {name}...")
            
            # Train model
            model.fit(X_combined, y)
            
            # Evaluate model
            performance = self._evaluate_model(model, X_combined, y)
            self.model_performances[f'stage2_{name}'] = performance
            
            # Get predictions for meta-learning
            stage2_predictions[:, i] = model.predict_proba(X_combined)[:, 1]
            
            print(f"    ✓ {name}: Accuracy = {performance['accuracy_mean']:.4f} (±{performance['accuracy_std']:.4f})")
        
        return stage2_predictions
    
    def fit_stage3_models(self, texts, X_features, y, stage1_predictions, stage2_predictions):
        """
        Train Stage 3: Deep Learning Models
        """
        print("\nTraining Stage 3: Deep Learning Models...")
        
        # Prepare text sequences
        X_sequences = self._prepare_text_sequences(texts, fit=True)
        vocab_size = min(self.max_vocab_size, len(self.tokenizer.word_index) + 1)
        
        # Convert features to dense
        if sparse.issparse(X_features):
            X_dense = X_features.toarray()
        else:
            X_dense = X_features
        
        # Callbacks for training
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
        ]
        
        stage3_predictions = np.zeros((len(y), 3))  # LSTM, CNN, Hybrid
        
        # LSTM Model
        print("  Training LSTM model...")
        lstm_model = self._create_lstm_model(vocab_size)
        lstm_history = lstm_model.fit(
            X_sequences, y,
            epochs=20,
            batch_size=32,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=0
        )
        stage3_predictions[:, 0] = lstm_model.predict(X_sequences, verbose=0).flatten()
        self.stage3_models['lstm'] = lstm_model
        
        # CNN Model
        print("  Training CNN model...")
        cnn_model = self._create_cnn_model(vocab_size)
        cnn_history = cnn_model.fit(
            X_sequences, y,
            epochs=20,
            batch_size=32,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=0
        )
        stage3_predictions[:, 1] = cnn_model.predict(X_sequences, verbose=0).flatten()
        self.stage3_models['cnn'] = cnn_model
        
        # Hybrid Model
        print("  Training Hybrid model...")
        hybrid_model = self._create_hybrid_model(vocab_size, X_dense.shape[1])
        hybrid_history = hybrid_model.fit(
            [X_sequences, X_dense], y,
            epochs=20,
            batch_size=32,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=0
        )
        stage3_predictions[:, 2] = hybrid_model.predict([X_sequences, X_dense], verbose=0).flatten()
        self.stage3_models['hybrid'] = hybrid_model
        
        print("    ✓ Deep learning models trained successfully")
        
        return stage3_predictions
    
    def fit_meta_learner(self, stage1_preds, stage2_preds, stage3_preds, y):
        """
        Train Stage 4: Meta-Learning Ensemble
        """
        print("\nTraining Stage 4: Meta-Learning Ensemble...")
        
        # Combine all stage predictions
        meta_features = np.hstack([stage1_preds, stage2_preds, stage3_preds])
        
        # Use a calibrated ensemble for meta-learning
        base_meta_learner = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        self.meta_learner = CalibratedClassifierCV(
            base_meta_learner,
            method='isotonic',
            cv=5
        )
        
        self.meta_learner.fit(meta_features, y)
        
        # Evaluate meta-learner
        performance = self._evaluate_model(self.meta_learner, meta_features, y)
        self.model_performances['meta_learner'] = performance
        
        print(f"    ✓ Meta-learner: Accuracy = {performance['accuracy_mean']:.4f} (±{performance['accuracy_std']:.4f})")
        
        return meta_features
    
    def fit(self, texts, X_features, y):
        """
        Complete multi-stage ensemble training pipeline
        """
        print("Multi-Stage Ensemble Framework Training")
        print("=======================================")
        print(f"Training samples: {len(texts)}")
        print(f"Feature dimensions: {X_features.shape[1]}")
        
        # Stage 1: Traditional ML
        stage1_preds = self.fit_stage1_models(X_features, y, texts)
        
        # Stage 2: Advanced ML
        stage2_preds = self.fit_stage2_models(X_features, y, stage1_preds)
        
        # Stage 3: Deep Learning
        stage3_preds = self.fit_stage3_models(texts, X_features, y, stage1_preds, stage2_preds)
        
        # Stage 4: Meta-Learning
        meta_features = self.fit_meta_learner(stage1_preds, stage2_preds, stage3_preds, y)
        
        print("\n✓ Multi-stage ensemble training completed!")
        
        return self
    
    def predict_proba(self, texts, X_features):
        """
        Generate predictions using the complete ensemble
        """
        # Stage 1 predictions
        if sparse.issparse(X_features):
            X_dense = X_features.toarray()
        else:
            X_dense = X_features
        
        stage1_preds = np.zeros((len(texts), len(self.stage1_models)))
        for i, (name, model) in enumerate(self.stage1_models.items()):
            if name == 'naive_bayes' and sparse.issparse(X_features):
                stage1_preds[:, i] = model.predict_proba(X_features)[:, 1]
            else:
                stage1_preds[:, i] = model.predict_proba(X_dense)[:, 1]
        
        # Stage 2 predictions
        X_combined = np.hstack([X_dense, stage1_preds])
        stage2_preds = np.zeros((len(texts), len(self.stage2_models)))
        for i, (name, model) in enumerate(self.stage2_models.items()):
            stage2_preds[:, i] = model.predict_proba(X_combined)[:, 1]
        
        # Stage 3 predictions
        X_sequences = self._prepare_text_sequences(texts, fit=False)
        stage3_preds = np.zeros((len(texts), 3))
        stage3_preds[:, 0] = self.stage3_models['lstm'].predict(X_sequences, verbose=0).flatten()
        stage3_preds[:, 1] = self.stage3_models['cnn'].predict(X_sequences, verbose=0).flatten()
        stage3_preds[:, 2] = self.stage3_models['hybrid'].predict([X_sequences, X_dense], verbose=0).flatten()
        
        # Meta-learner prediction
        meta_features = np.hstack([stage1_preds, stage2_preds, stage3_preds])
        final_probas = self.meta_learner.predict_proba(meta_features)
        
        return final_probas
    
    def predict(self, texts, X_features, threshold=0.5):
        """
        Generate binary predictions
        """
        probas = self.predict_proba(texts, X_features)
        return (probas[:, 1] > threshold).astype(int)
    
    def get_model_performances(self):
        """
        Get comprehensive performance report
        """
        return pd.DataFrame(self.model_performances).T
    
    def save_models(self, filepath_prefix):
        """
        Save all trained models
        """
        # Save traditional ML models
        for stage_name, models in [('stage1', self.stage1_models), ('stage2', self.stage2_models)]:
            for model_name, model in models.items():
                joblib.dump(model, f"{filepath_prefix}_{stage_name}_{model_name}.pkl")
        
        # Save deep learning models
        for model_name, model in self.stage3_models.items():
            model.save(f"{filepath_prefix}_stage3_{model_name}.h5")
        
        # Save meta-learner
        joblib.dump(self.meta_learner, f"{filepath_prefix}_meta_learner.pkl")
        
        # Save tokenizer
        joblib.dump(self.tokenizer, f"{filepath_prefix}_tokenizer.pkl")
        
        print(f"✓ All models saved with prefix: {filepath_prefix}")

if __name__ == "__main__":
    # Example usage
    print("Multi-Stage Ensemble Framework")
    print("==============================")
    
    # Sample data
    sample_texts = [
        "This is a legitimate news article with verified facts.",
        "SHOCKING: You won't believe this incredible discovery!!!",
        "Research shows promising results in renewable energy development."
    ] * 100  # Replicate for demonstration
    
    sample_labels = [1, 0, 1] * 100  # 1 = real, 0 = fake
    sample_features = np.random.rand(300, 50)  # Mock features
    
    # Initialize and train ensemble
    ensemble = MultiStageEnsembleFramework()
    ensemble.fit(sample_texts, sample_features, sample_labels)
    
    # Get performance report
    performance_df = ensemble.get_model_performances()
    print("\nModel Performance Summary:")
    print(performance_df)
    
    print("\nMulti-stage ensemble framework ready!")