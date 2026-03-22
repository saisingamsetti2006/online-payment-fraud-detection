import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import plot_tree


# ==============================
# LOAD DATA
# ==============================
df = pd.read_csv("fraud.csv")

target = 'isFraud'


# ==============================
# PREPROCESSING
# ==============================
le = LabelEncoder()
df['type'] = le.fit_transform(df['type'])

# Drop unnecessary columns
for col in ['nameOrig', 'nameDest', 'isFlaggedFraud']:
    if col in df.columns:
        df.drop(col, axis=1, inplace=True)


# ==============================
# HANDLE IMBALANCE (SAMPLING)
# ==============================
fraud = df[df[target] == 1]
normal = df[df[target] == 0].sample(len(fraud)*3, random_state=42)

df_balanced = pd.concat([fraud, normal])


# ==============================
# SPLIT DATA
# ==============================
X = df_balanced.drop(target, axis=1)
y = df_balanced[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ==============================
# MODEL TRAINING
# ==============================
model = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    random_state=42
)

model.fit(X_train, y_train)


# ==============================
# PREDICTION
# ==============================
y_pred = model.predict(X_test)


# ==============================
# ACCURACY & REPORT
# ==============================
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)

print("\nClassification Report:\n", classification_report(y_test, y_pred))


# ==============================
# CONFUSION MATRIX
# ==============================
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()


# ==============================
# FEATURE IMPORTANCE
# ==============================
plt.figure(figsize=(8,5))

pd.Series(model.feature_importances_, index=X.columns)\
    .sort_values(ascending=False)\
    .plot(kind='bar')

plt.title("Feature Importance")
plt.ylabel("Score")
plt.tight_layout()
plt.show()


# ==============================
# RANDOM FOREST TREE (FIXED POSITION)
# ==============================
plt.figure(figsize=(20, 12))

plot_tree(
    model.estimators_[0],   # one tree
    feature_names=X.columns,
    class_names=["Legit", "Fraud"],
    filled=True,
    max_depth=3,
    fontsize=10
)

plt.title("Random Forest - Sample Decision Tree", fontsize=16)

plt.tight_layout()   # 🔥 FIXES SIDE SHIFT ISSUE
plt.show()


# ==============================
# SAVE MODEL
# ==============================
joblib.dump(model, "model.pkl")
joblib.dump(le, "encoder.pkl")

print("\nModel saved successfully!")