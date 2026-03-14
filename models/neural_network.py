from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np

class NeuralNetworkModel:
    def __init__(self, input_dim):
        self.model = self._create_model(input_dim)
        self.history = None

    def _create_model(self, input_dim):
        model = Sequential([
            Dense(128, activation='relu', input_shape=(input_dim,)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dropout(0.1),
            Dense(1, activation='sigmoid')
        ])

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        return model

    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        print("\nTraining Neural Network...")
        
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=5,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=3
            )
        ]

        self.history = self.model.fit(
            X_train, y_train,
            epochs=30,
            batch_size=8,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=1
        )

        # Make predictions
        y_pred = (self.model.predict(X_test) > 0.5).astype(int)

        # Calculate metrics
        results = {
            'nn': {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'confusion_matrix': confusion_matrix(y_test, y_pred)
            }
        }

        print("Neural Network Results:")
        print(f"Accuracy: {results['nn']['accuracy']:.4f}")
        print(f"Precision: {results['nn']['precision']:.4f}")
        print(f"Recall: {results['nn']['recall']:.4f}")
        print(f"F1-score: {results['nn']['f1']:.4f}")

        return results

    def get_training_history(self):
        return self.history