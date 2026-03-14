import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

class DataPreprocessor:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            max_df=0.95,
            min_df=2
        )
        self.feature_names = None

    def clean_text(self, text):
        if not isinstance(text, str):
            text = str(text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = ' '.join(word for word in text.split() if not word.startswith('http'))
        
        # Remove special characters and digits
        text = ''.join(c for c in text if c.isalpha() or c.isspace())
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text

    def prepare_data(self, dataset_path, save_processed=True):
        print("Loading dataset...")
        df = pd.read_csv(dataset_path)
        
        print("Cleaning text...")
        df['text'] = df['text'].apply(self.clean_text)
        
        if save_processed:
            df.to_csv('data/processed/cleaned_dataset.csv', index=False)
        
        print("Vectorizing text...")
        X = self.vectorizer.fit_transform(df['text'])
        self.feature_names = self.vectorizer.get_feature_names_out()
        y = df['label']
        
        print("Splitting dataset...")
        X_train, X_test, y_train, y_test = train_test_split(
            X.toarray(),
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )
        
        print("Data preparation completed.")
        print(f"Training set shape: {X_train.shape}")
        print(f"Testing set shape: {X_test.shape}")
        print(f"Class distribution in training set:\n{y_train.value_counts(normalize=True)}")
        
        return X_train, X_test, y_train, y_test

    def get_feature_names(self):
        return self.feature_names