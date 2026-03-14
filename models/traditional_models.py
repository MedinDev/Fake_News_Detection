from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

class TraditionalModels:
    def __init__(self, random_state=42):
        self.models = {
            'rf': RandomForestClassifier(
                n_estimators=200,
                max_depth=30,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=random_state
            ),
            'svm': SVC(
                C=10,
                kernel='rbf',
                probability=True,
                random_state=random_state
            ),
            'gb': GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=3,
                random_state=random_state
            )
        }
        self.best_model = None
        self.best_accuracy = 0

    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        results = {}
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            # Use cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=2)
            print(f"Cross-validation scores: {cv_scores}")
            print(f"Mean CV accuracy: {cv_scores.mean():.4f}")
            
            # Train on full training set
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'confusion_matrix': confusion_matrix(y_test, y_pred)
            }
            
            # Update best model
            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                self.best_model = model

            print(f"{name} Results:")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1-score: {f1:.4f}")
        
        return results

    def get_feature_importance(self, feature_names=None):
        if self.best_model and hasattr(self.best_model, 'feature_importances_'):
            importances = self.best_model.feature_importances_
            if feature_names is None:
                feature_names = [f'feature_{i}' for i in range(len(importances))]
            
            return dict(zip(feature_names, importances))
        return None