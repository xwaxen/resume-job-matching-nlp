import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

print("Loading dataset...")
pairs_df = pd.read_csv("dataset_pairs.csv")
print(f"Loaded {len(pairs_df):,} pairs")

pairs_df['combined_text'] = (
    pairs_df['resume_text_cleaned'] + ' [SEP] ' + pairs_df['job_description_cleaned']
)

vectorizer = pickle.load(open('models/tfidf_vectorizer.pkl', 'rb'))
X_tfidf = vectorizer.transform(pairs_df['combined_text'])
y_labels = pairs_df['label'].values

print(f"Features: {X_tfidf.shape[0]:,} samples × {X_tfidf.shape[1]:,} features")
print(f"Sparsity: {100 * (1 - X_tfidf.nnz / (X_tfidf.shape[0] * X_tfidf.shape[1])):.2f}%")

print("Splitting data (80% train, 20% test)...")

X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y_labels,
    test_size=0.2,
    random_state=42,
    stratify=y_labels
)

print(f"Train: {X_train.shape[0]:,} samples, Test: {X_test.shape[0]:,} samples")

print("Hyperparameter tuning with GridSearchCV...")

param_grid = {
    'C': [0.1, 1.0, 10.0],
    'solver': ['lbfgs', 'saga'],
    'max_iter': [500],
}

base_model = LogisticRegression(
    random_state=42,
    n_jobs=-1,
    verbose=0
)

grid_search = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=2,
    return_train_score=True
)

grid_search.fit(X_train, y_train)

print("\nBest hyperparameters:")
for param, value in grid_search.best_params_.items():
    print(f"  {param}: {value}")
print(f"Best CV F1: {grid_search.best_score_:.4f}\n")

best_model = grid_search.best_estimator_

print("Evaluating on test set...")

y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print("\nLogistic Regression Performance:")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"ROC AUC:   {roc_auc:.4f}\n")

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Not Fit (0)', 'Good Fit (1)']))

print("Saving model and results...")

os.makedirs('models', exist_ok=True)

with open('models/logreg_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
with open('models/logreg_grid_search.pkl', 'wb') as f:
    pickle.dump(grid_search, f)
np.savez('models/logreg_predictions.npz', 
         y_test=y_test, 
         y_pred=y_pred, 
         y_pred_proba=y_pred_proba)

print("Saved: model, predictions")
print("\nGenerating visualizations...")

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Not Fit (0)', 'Good Fit (1)'],
            yticklabels=['Not Fit (0)', 'Good Fit (1)'])
plt.title('Logistic Regression - Confusion Matrix (TF-IDF)', fontsize=14, fontweight='bold')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('models/logreg_confusion_matrix.png', dpi=300, bbox_inches='tight')

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, linewidth=2, label=f'Logistic Regression (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve - Logistic Regression (TF-IDF)', fontsize=14, fontweight='bold')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('models/logreg_roc_curve.png', dpi=300, bbox_inches='tight')

print("\nSaved: confusion_matrix.png, roc_curve.png")
print("Logistic Regression training complete!")