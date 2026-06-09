"""
Iris Random Forest Classifier pipeline

This script performs the following:
1. Loads the Iris dataset from a remote CSV using pandas
2. Performs basic EDA and preprocessing
3. Splits data into train/test (80:20)
4. Trains a RandomForestClassifier
5. Evaluates accuracy, precision, recall, F1-score and confusion matrix
6. Plots feature importance and confusion matrix
7. Performs hyperparameter tuning with GridSearchCV
8. Saves the best model as `iris_random_forest.pkl`
9. Exposes `predict_flower()` to predict new measurements

Run as: python iris_random_forest.py
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

DATA_URL = (
    "https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv"
)
MODEL_FILENAME = "iris_random_forest.pkl"


def load_data(url=DATA_URL):
    """Load the Iris dataset from a CSV URL into a pandas DataFrame."""
    df = pd.read_csv(url)
    return df


def perform_eda(df):
    """Basic exploratory data analysis: show head, info, describe, and class counts."""
    print("\n--- Data head ---")
    print(df.head())

    print("\n--- Data info ---")
    print(df.info())

    print("\n--- Descriptive statistics ---")
    print(df.describe())

    print("\n--- Class distribution ---")
    print(df['species'].value_counts())

    # Pairplot (comment/uncomment for more visuals). This can be slow.
    try:
        sns.pairplot(df, hue='species')
        plt.suptitle('Pairplot of Iris features', y=1.02)
        plt.show()
    except Exception:
        pass

    # Correlation heatmap
    plt.figure(figsize=(6, 5))
    sns.heatmap(df.drop(columns=['species']).corr(), annot=True, cmap='coolwarm')
    plt.title('Feature Correlation')
    plt.show()


def preprocess(df):
    """Preprocess the dataset: separate X and y, encode labels."""
    X = df.drop(columns=['species'])
    y = df['species']

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    return X, y_enc, le


def train_random_forest(X_train, y_train, random_state=42):
    """Train a RandomForestClassifier with default settings as baseline."""
    model = RandomForestClassifier(random_state=random_state)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, label_encoder=None):
    """Compute evaluation metrics and show confusion matrix and classification report."""
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

    print('\n--- Evaluation Metrics ---')
    print(f'Accuracy: {acc:.4f}')
    print(f'Precision (macro): {prec:.4f}')
    print(f'Recall (macro): {rec:.4f}')
    print(f'F1-score (macro): {f1:.4f}')

    print('\n--- Classification Report ---')
    target_names = None
    if label_encoder is not None:
        target_names = label_encoder.classes_
    print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'confusion_matrix': cm}


def plot_feature_importance(model, feature_names):
    """Plot feature importance from a RandomForest model."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(8, 5))
    sns.barplot(x=importances[indices], y=np.array(feature_names)[indices], palette='viridis')
    plt.title('Feature Importances')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()


def hyperparameter_tuning(X_train, y_train, random_state=42):
    """Run GridSearchCV to find the best hyperparameters for RandomForest."""
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [None, 3, 5, 7],
        'min_samples_split': [2, 4, 6],
        'min_samples_leaf': [1, 2, 3],
    }

    rf = RandomForestClassifier(random_state=random_state)
    grid = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)

    print('\n--- GridSearchCV Results ---')
    print(f'Best Score: {grid.best_score_:.4f}')
    print('Best Params:', grid.best_params_)

    return grid.best_estimator_, grid


def save_model(model, filename=MODEL_FILENAME):
    """Save the trained model to disk using joblib."""
    joblib.dump(model, filename)
    print(f'Model saved to {filename}')


def load_model(filename=MODEL_FILENAME):
    """Load a saved model from disk."""
    if not os.path.exists(filename):
        raise FileNotFoundError(f'Model file not found: {filename}')
    return joblib.load(filename)


def predict_flower(measurements, model, label_encoder=None):
    """Predict species for new flower measurements.

    measurements: array-like shape (4,) or (n_samples, 4)
    model: trained classifier
    label_encoder: optional LabelEncoder used during training (for inverse mapping)
    """
    arr = np.array(measurements)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    preds = model.predict(arr)
    probs = None
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(arr)

    if label_encoder is not None:
        preds_readable = label_encoder.inverse_transform(preds)
    else:
        preds_readable = preds

    return preds_readable, probs


def main():
    df = load_data()
    perform_eda(df)

    X, y_enc, le = preprocess(df)

    # Split data: 80% train, 20% test, stratify to preserve class proportions
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    # Baseline model training
    baseline_model = train_random_forest(X_train, y_train)
    print('\nBaseline model evaluation:')
    baseline_metrics = evaluate_model(baseline_model, X_test, y_test, label_encoder=le)
    plot_feature_importance(baseline_model, X.columns)

    # Hyperparameter tuning
    best_model, grid = hyperparameter_tuning(X_train, y_train)
    print('\nBest model evaluation (after GridSearchCV):')
    best_metrics = evaluate_model(best_model, X_test, y_test, label_encoder=le)
    plot_feature_importance(best_model, X.columns)

    # Save the best model
    save_model({'model': best_model, 'label_encoder': le}, MODEL_FILENAME)

    # Example prediction for a new sample
    sample = [5.1, 3.5, 1.4, 0.2]  # example: Iris-setosa
    pred, probs = predict_flower(sample, best_model, label_encoder=le)
    print(f'\nExample sample: {sample}')
    print(f'Predicted species: {pred[0]}')
    if probs is not None:
        print(f'Class probabilities: {probs[0]}')

    # Brief explanation of results
    print('\n--- Model Explanation ---')
    print('The Random Forest model achieves high accuracy on the Iris dataset,')
    print('which is expected because Iris is a small, well-separated, and balanced dataset.')
    print('Feature importances show which measurements most influence predictions.')
    print('GridSearchCV refines hyperparameters to possibly improve generalization.')


if __name__ == '__main__':
    main()