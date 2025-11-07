from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
import numpy as np
import joblib
from pathlib import Path
import time


class BirdClassifier:

    def __init__(self, model_type='random_forest', random_state=42, **kwargs):
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_fitted = False

        # Initialize model based on type
        if model_type == 'knn':
            n_neighbors = kwargs.get('n_neighbors', 5)
            metric = kwargs.get('metric', 'euclidean')
            self.model = KNeighborsClassifier(
                n_neighbors=n_neighbors,
                metric=metric,
                n_jobs=-1
            )
            print(f"KNN Classifier initialized (n_neighbors={n_neighbors}, metric={metric})")

        elif model_type == 'random_forest':
            n_estimators = kwargs.get('n_estimators', 100)
            max_depth = kwargs.get('max_depth', None)
            min_samples_split = kwargs.get('min_samples_split', 2)
            min_samples_leaf = kwargs.get('min_samples_leaf', 1)
            self.model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=random_state,
                n_jobs=-1,
                verbose=0
            )
            print(f"Random Forest Classifier initialized (n_estimators={n_estimators}, max_depth={max_depth})")

        elif model_type == 'svm':
            use_linear = kwargs.get('use_linear', True)

            if use_linear:
                C = kwargs.get('C', 1.0)
                max_iter = kwargs.get('max_iter', 10000)
                self.model = LinearSVC(
                    C=C,
                    max_iter=max_iter,
                    random_state=random_state,
                    dual='auto',
                    verbose=0
                )
                print(f"Linear SVM Classifier initialized (C={C}, max_iter={max_iter})")
                
            else:
                #traditional kernel SVM
                kernel = kwargs.get('kernel', 'rbf')
                C = kwargs.get('C', 1.0)
                gamma = kwargs.get('gamma', 'scale')
                self.model = SVC(
                    kernel=kernel,
                    C=C,
                    gamma=gamma,
                    random_state=random_state,
                    verbose=False
                )
                print(f"Kernel SVM Classifier initialized (kernel={kernel}, C={C})")

        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def fit(self, X, y):
        print(f"\nTraining {self.model_type.upper()} model")
        print(f"  Training samples: {len(X)}")
        print(f"  Features: {X.shape[1]}")
        print(f"  Classes: {len(np.unique(y))}")

        start_time = time.time()

        y_encoded = self.label_encoder.fit_transform(y)

        X_scaled = self.scaler.fit_transform(X)

        self.model.fit(X_scaled, y_encoded)

        training_time = time.time() - start_time
        print(f"  Training completed in {training_time:.2f} seconds")

        self.is_fitted = True

        train_pred = self.model.predict(X_scaled)
        train_acc = np.mean(train_pred == y_encoded)
        print(f"  Training accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")

        return self

    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        X_scaled = self.scaler.transform(X)

        y_pred_encoded = self.model.predict(X_scaled)

        y_pred = self.label_encoder.inverse_transform(y_pred_encoded)

        return y_pred

    def predict_proba(self, X):
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        if not hasattr(self.model, 'predict_proba'):
            raise ValueError(f"{self.model_type} does not support probability predictions")

        X_scaled = self.scaler.transform(X)

        proba = self.model.predict_proba(X_scaled)

        return proba

    def save(self, filepath):
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")

        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'model_type': self.model_type,
            'is_fitted': self.is_fitted
        }

        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")

    def load(self, filepath):
        model_data = joblib.load(filepath)

        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.model_type = model_data['model_type']
        self.is_fitted = model_data['is_fitted']

        print(f"Model loaded from {filepath}")

    def get_feature_importance(self):
        if self.model_type != 'random_forest':
            raise ValueError("Feature importance only available for Random Forest")

        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        return self.model.feature_importances_


def train_all_models(X_train, X_test, y_train, y_test, save_dir='../models'):
    from utils import evaluate_model

    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True, parents=True)

    results = {}

    models_config = {
        'KNN': {
            'model_type': 'knn',
            'params': {'n_neighbors': 7}
        },
        'Random Forest': {
            'model_type': 'random_forest',
            'params': {
                'n_estimators': 200,
                'max_depth': 30,
                'min_samples_split': 5,
                'min_samples_leaf': 2
            }
        },
        'SVM': {
            'model_type': 'svm',
            'params': {
                'use_linear': True, #LinearSVC for speed
                'C': 1.0,
                'max_iter': 10000
            }
        }
    }

    for model_name, config in models_config.items():
        print("\n" + "-"*50)
        print(f"Training {model_name}")
        print("-"*50)

        classifier = BirdClassifier(
            model_type=config['model_type'],
            **config['params']
        )
        classifier.fit(X_train, y_train)

        print(f"\nEvaluating {model_name} on test set")
        y_pred = classifier.predict(X_test)

        metrics = evaluate_model(y_test, y_pred, model_name=model_name)

        model_filename = save_path / f"{config['model_type']}_model.pkl"
        classifier.save(model_filename)

        results[model_name] = {
            'classifier': classifier,
            'metrics': metrics,
            'y_pred': y_pred
        }

    return results


if __name__ == "__main__":
    print("Testing model implementations with dummy data")

    np.random.seed(42)
    n_samples = 1000
    n_features = 40
    n_classes = 10

    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, n_classes, n_samples)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nDummy dataset:")
    print(f"  Train: {X_train.shape}")
    print(f"  Test: {X_test.shape}")
    print(f"  Classes: {n_classes}")

    for model_type in ['knn', 'random_forest', 'svm']:
        print(f"\n")
        print(f"Testing {model_type.upper()}")
        print(f"{'-'*50}")

        clf = BirdClassifier(model_type=model_type)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        accuracy = np.mean(y_pred == y_test)
        print(f"Test accuracy: {accuracy:.4f}")

    print("Model testing completed")
