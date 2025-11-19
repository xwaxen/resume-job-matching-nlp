import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

logreg_data = np.load('models/logreg_predictions.npz')
xgb_data = np.load('models/xgboost_predictions.npz')

lr_acc = accuracy_score(logreg_data['y_test'], logreg_data['y_pred'])
lr_prec = precision_score(logreg_data['y_test'], logreg_data['y_pred'])
lr_rec = recall_score(logreg_data['y_test'], logreg_data['y_pred'])
lr_f1 = f1_score(logreg_data['y_test'], logreg_data['y_pred'])
lr_auc = roc_auc_score(logreg_data['y_test'], logreg_data['y_pred_proba'])

xgb_acc = accuracy_score(xgb_data['y_test'], xgb_data['y_pred'])
xgb_prec = precision_score(xgb_data['y_test'], xgb_data['y_pred'])
xgb_rec = recall_score(xgb_data['y_test'], xgb_data['y_pred'])
xgb_f1 = f1_score(xgb_data['y_test'], xgb_data['y_pred'])
xgb_auc = roc_auc_score(xgb_data['y_test'], xgb_data['y_pred_proba'])

print("Logistic Regression (TF-IDF):")
print(f"  Accuracy:  {lr_acc:.4f}")
print(f"  Precision: {lr_prec:.4f}")
print(f"  Recall:    {lr_rec:.4f}")
print(f"  F1 Score:  {lr_f1:.4f}")
print(f"  ROC AUC:   {lr_auc:.4f}\n")

print("XGBoost (SBERT):")
print(f"  Accuracy:  {xgb_acc:.4f}")
print(f"  Precision: {xgb_prec:.4f}")
print(f"  Recall:    {xgb_rec:.4f}")
print(f"  F1 Score:  {xgb_f1:.4f}")
print(f"  ROC AUC:   {xgb_auc:.4f}\n")
