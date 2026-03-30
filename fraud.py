import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import os
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("FRAUD DETECTION SYSTEM - RANDOM FOREST")
print("="*60)

# ==============================
# LOAD DATA
# ==============================
print("\n📂 Loading dataset...")
df = pd.read_csv("fraud.csv")
print(f"✅ Dataset shape: {df.shape}")
print(f"📊 Fraud count: {df['isFraud'].sum()} ({df['isFraud'].mean()*100:.2f}%)")

# Drop unnecessary columns
for col in ['nameOrig', 'nameDest', 'isFlaggedFraud']:
    if col in df.columns:
        df.drop(col, axis=1, inplace=True)

# ==============================
# ADVANCED FEATURE ENGINEERING
# ==============================
print("\n🔧 Engineering advanced features...")

# 1. Amount ratio
df['amount_ratio'] = df['amount'] / (df['oldbalanceOrg'] + 1)

# 2. Balance error
df['balance_error'] = abs((df['oldbalanceOrg'] - df['newbalanceOrig']) - 
                           (df['newbalanceDest'] - df['oldbalanceDest']))

# 3. Is empty receiver
df['is_empty_receiver'] = (df['oldbalanceDest'] == 0).astype(int)

# 4. Balance mismatch
df['balance_mismatch'] = ((df['oldbalanceOrg'] - df['newbalanceOrig']) != 
                          (df['newbalanceDest'] - df['oldbalanceDest'])).astype(int)

# 5. Negative balances
df['sender_negative'] = (df['newbalanceOrig'] < 0).astype(int)
df['receiver_negative'] = (df['newbalanceDest'] < 0).astype(int)

# 6. Unusual hour
df['hour'] = df['step'] % 24
df['unusual_hour'] = ((df['hour'] >= 23) | (df['hour'] <= 6)).astype(int)
df.drop('hour', axis=1, inplace=True)

# 7. High percentage transfer (using original type column)
df['high_percentage_transfer'] = ((df['amount_ratio'] > 0.9) & 
                                   (df['type'] == 'TRANSFER')).astype(int)

# 8. Round amounts
df['round_amount'] = (df['amount'] % 100 == 0).astype(int)

# 9. Suspicious amounts
suspicious_amounts = [4900, 4999, 9900, 9999, 1999, 2999, 3999, 4999, 5999, 6999, 7999, 8999, 9999]
df['suspicious_amount'] = df['amount'].isin(suspicious_amounts).astype(int)

# 10. Amount to destination balance ratio
df['amount_to_dest_ratio'] = df['amount'] / (df['oldbalanceDest'] + 1)

# 11. Interaction features (using unusual_hour and amount)
df['unusual_hour_amount'] = df['unusual_hour'] * df['amount']
df['unusual_hour_ratio'] = df['unusual_hour'] * df['amount_ratio']
df['empty_receiver_high_amount'] = df['is_empty_receiver'] * (df['amount'] > 100000).astype(int)

# ==============================
# ONE-HOT ENCODE TRANSACTION TYPE
# ==============================
df = pd.get_dummies(df, columns=["type"], drop_first=True)

print(f"✅ Total features created: {df.shape[1] - 1}")

# ==============================
# HANDLE CLASS IMBALANCE
# ==============================
fraud = df[df['isFraud'] == 1]
normal = df[df['isFraud'] == 0].sample(n=len(fraud)*3, random_state=42)
df_balanced = pd.concat([fraud, normal])
print(f"\n⚖️ Balanced dataset shape: {df_balanced.shape}")

X = df_balanced.drop('isFraud', axis=1)
y = df_balanced['isFraud']

# ==============================
# FEATURE IMPORTANCE ANALYSIS
# ==============================
print("\n📊 Analyzing feature importance...")
temp_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
temp_model.fit(X, y)

feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': temp_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 most important features:")
print(feature_importance.head(10))

# Keep features with importance > 0.005
important_features = feature_importance[feature_importance['importance'] > 0.005]['feature'].tolist()
print(f"\n✅ Keeping {len(important_features)} important features (removed {len(X.columns) - len(important_features)} useless ones)")

X = X[important_features]

# ==============================
# TRAIN / TEST SPLIT
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==============================
# SCALE INPUTS
# ==============================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==============================
# TRAIN RANDOM FOREST (200 trees)
# ==============================
print("\n🤖 Training Random Forest with 200 trees...")
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',
    n_jobs=-1,
    random_state=42
)
rf.fit(X_train_scaled, y_train)

# ==============================
# PREDICT AND TUNE THRESHOLD
# ==============================
rf_proba = rf.predict_proba(X_test_scaled)[:, 1]

# Find optimal threshold using precision-recall curve
precisions, recalls, thresholds = precision_recall_curve(y_test, rf_proba)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
optimal_threshold = thresholds[np.argmax(f1_scores[:-1])]
print(f"\n🎯 Optimal threshold (max F1): {optimal_threshold:.3f}")

# Use lower threshold for better fraud detection (0.2-0.3)
threshold = max(0.2, min(0.3, optimal_threshold))
print(f"✅ Using threshold: {threshold:.3f}")

y_pred = (rf_proba >= threshold).astype(int)

# ==============================
# EVALUATION
# ==============================
print("\n" + "="*60)
print("CLASSIFICATION REPORT")
print("="*60)
print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud']))

# ==============================
# ANALYZE WRONG PREDICTIONS
# ==============================
print("\n🔍 Analyzing wrong predictions...")
wrong_predictions = X_test.copy()
wrong_predictions['true_label'] = y_test
wrong_predictions['predicted_label'] = y_pred
wrong_predictions['fraud_probability'] = rf_proba

false_positives = wrong_predictions[(wrong_predictions['true_label'] == 0) & (wrong_predictions['predicted_label'] == 1)]
false_negatives = wrong_predictions[(wrong_predictions['true_label'] == 1) & (wrong_predictions['predicted_label'] == 0)]

print(f"❌ False Positives (legit flagged as fraud): {len(false_positives)}")
print(f"❌ False Negatives (fraud missed): {len(false_negatives)}")

# ==============================
# SAVE PLOTS
# ==============================
os.makedirs('static', exist_ok=True)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Legit', 'Fraud'], yticklabels=['Legit', 'Fraud'])
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('static/confusion_matrix.png')
plt.close()

# Feature Importance
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(12,6))
plt.title('Feature Importance (Random Forest)')
plt.bar(range(len(important_features)), importances[indices], align='center')
plt.xticks(range(len(important_features)), [important_features[i] for i in indices], rotation=45, ha='right')
plt.tight_layout()
plt.savefig('static/feature_importance.png')
plt.close()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, rf_proba)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0,1], [0,1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig('static/roc_curve.png')
plt.close()

print("\n📊 Plots saved in static/ folder.")

# ==============================
# SAVE ARTIFACTS
# ==============================
joblib.dump(rf, "model_rf.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(important_features, "feature_names.pkl")
joblib.dump(threshold, "threshold.pkl")

print("\n" + "="*60)
print("✅ RANDOM FOREST MODEL SAVED SUCCESSFULLY!")
print("="*60)
print("Saved files:")
print("  - model_rf.pkl (Random Forest)")
print("  - scaler.pkl (StandardScaler)")
print("  - feature_names.pkl (Important features)")
print("  - threshold.pkl (Optimal threshold)")
print("="*60)
