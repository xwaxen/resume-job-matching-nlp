import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sentence_transformers import SentenceTransformer
import torch
from sklearn.metrics.pairwise import cosine_similarity

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

print("Loading dataset...")
pairs_df = pd.read_csv("dataset_pairs.csv")
print(f"Loaded {len(pairs_df):,} pairs")

print("Loading SBERT model...")
sbert_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device=device)

print("Generating embeddings...")
resume_embeddings = sbert_model.encode(
    pairs_df['resume_text_cleaned'].tolist(),
    convert_to_numpy=True,
    show_progress_bar=True,
    batch_size=32
)
jd_embeddings = sbert_model.encode(
    pairs_df['job_description_cleaned'].tolist(),
    convert_to_numpy=True,
    show_progress_bar=True,
    batch_size=32
)

cos_sim = np.array([
    cosine_similarity([resume_embeddings[i]], [jd_embeddings[i]])[0][0]
    for i in range(len(resume_embeddings))
]).reshape(-1, 1)

diff_vector = resume_embeddings - jd_embeddings
X_sbert = np.hstack([resume_embeddings, jd_embeddings, diff_vector, cos_sim])
y_labels = pairs_df['label'].values

if X_sbert.dtype == np.float16:
    X_sbert = X_sbert.astype(np.float32)

print(f"Features: {X_sbert.shape[0]:,} samples × {X_sbert.shape[1]:,} features")

print("Splitting data (80% train, 20% test)...")

X_train, X_test, y_train, y_test = train_test_split(
    X_sbert, y_labels,
    test_size=0.2,
    random_state=42,
    stratify=y_labels
)

print(f"Train: {X_train.shape[0]:,} samples, Test: {X_test.shape[0]:,} samples")

print("Hyperparameter tuning with RandomizedSearchCV...")
param_distributions = {
    'max_depth': [5, 7],
    'learning_rate': [0.1, 0.2],
    'n_estimators': [100, 200],
    'min_child_weight': [1],
    'subsample': [0.9],
    'colsample_bytree': [0.9],
}

base_model = xgb.XGBClassifier(
    device='cuda:0',
    tree_method='hist',
    random_state=42,
    eval_metric='logloss',
    use_label_encoder=False
)

random_search = RandomizedSearchCV(
    estimator=base_model,
    param_distributions=param_distributions,
    n_iter=5,
    cv=2,
    scoring='f1',
    n_jobs=1,
    verbose=2,
    random_state=42,
    return_train_score=True
)

random_search.fit(X_train, y_train)

print("\nBest hyperparameters:")
for param, value in random_search.best_params_.items():
    print(f"  {param}: {value}")
print(f"Best CV F1: {random_search.best_score_:.4f}\n")

best_model = random_search.best_estimator_

print("Evaluating on test set...")

y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print("\nXGBoost Performance:")
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

best_model.save_model('models/xgboost_model.json')
with open('models/xgboost_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
with open('models/xgboost_random_search.pkl', 'wb') as f:
    pickle.dump(random_search, f)
np.savez('models/xgboost_predictions.npz', 
         y_test=y_test, 
         y_pred=y_pred, 
         y_pred_proba=y_pred_proba)

print("Saved: model, predictions")
print("\nGenerating visualizations...")

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
            xticklabels=['Not Fit (0)', 'Good Fit (1)'],
            yticklabels=['Not Fit (0)', 'Good Fit (1)'])
plt.title('XGBoost - Confusion Matrix (SBERT)', fontsize=14, fontweight='bold')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('models/xgboost_confusion_matrix.png', dpi=300, bbox_inches='tight')

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, linewidth=2, label=f'XGBoost (AUC = {roc_auc:.4f})', color='green')
plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve - XGBoost (SBERT)', fontsize=14, fontweight='bold')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('models/xgboost_roc_curve.png', dpi=300, bbox_inches='tight')

feature_importance = best_model.feature_importances_
top_n = 20
top_indices = np.argsort(feature_importance)[-top_n:][::-1]
top_scores = feature_importance[top_indices]

feature_names = []
for i in range(2305):
    if i < 768:
        feature_names.append(f'Resume_{i}')
    elif i < 1536:
        feature_names.append(f'JD_{i-768}')
    elif i < 2304:
        feature_names.append(f'Diff_{i-1536}')
    else:
        feature_names.append('Cosine_Sim')

top_features = [feature_names[i] for i in top_indices]

plt.figure(figsize=(10, 8))
plt.barh(range(top_n), top_scores, color='green', alpha=0.7)
plt.yticks(range(top_n), top_features)
plt.xlabel('Feature Importance Score', fontsize=12)
plt.title(f'Top {top_n} Features (XGBoost on SBERT)', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('models/xgboost_feature_importance.png', dpi=300, bbox_inches='tight')

print("\nSaved: confusion_matrix.png, roc_curve.png, feature_importance.png")
print("XGBoost training complete!")