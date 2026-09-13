import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# =====================================================
# 1. Load dataset
# =====================================================
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
df = pd.read_csv(url, sep=';')
print("Dataset shape:", df.shape)

# =====================================================
# 2. Feature set (same for both models)
# =====================================================
X = df.drop('quality', axis=1)
feature_names = X.columns.tolist()

# Train/test split (shared)
X_train, X_test, y_train_full, y_test_full = train_test_split(
    X, df['quality'], test_size=0.2, random_state=42, stratify=df['quality']
)

# Scaler (shared)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =====================================================
# 3. BINARY MODEL (used in the web app)
# =====================================================
print("\n" + "=" * 60)
print("BINARY MODEL (Good = quality >= 7  vs  Bad = quality < 7)")
print("=" * 60)

y_train_bin = (y_train_full >= 7).astype(int)
y_test_bin = (y_test_full >= 7).astype(int)

binary_model = LogisticRegression(max_iter=1000, random_state=42)
binary_model.fit(X_train_scaled, y_train_bin)

binary_pred = binary_model.predict(X_test_scaled)
binary_acc = accuracy_score(y_test_bin, binary_pred)
print(f"Binary Accuracy: {binary_acc:.4f}")
print(classification_report(y_test_bin, binary_pred, target_names=["Bad", "Good"]))

# =====================================================
# 4. MULTICLASS MODEL (predicts exact score 3-8)
# =====================================================
print("\n" + "=" * 60)
print("MULTICLASS MODEL (predicts raw quality score 3-8)")
print("=" * 60)

multiclass_model = LogisticRegression(
    max_iter=2000,
    random_state=42
)
multiclass_model.fit(X_train_scaled, y_train_full)

multi_pred = multiclass_model.predict(X_test_scaled)
multi_acc = accuracy_score(y_test_full, multi_pred)
print(f"Multiclass Accuracy: {multi_acc:.4f}")
print(classification_report(y_test_full, multi_pred))

# =====================================================
# 5. Save everything the web app needs
# =====================================================
os.makedirs('model', exist_ok=True)
joblib.dump(binary_model, 'model/wine_model.pkl')            # web app uses this
joblib.dump(scaler, 'model/scaler.pkl')
joblib.dump(feature_names, 'model/features.pkl')
joblib.dump(multiclass_model, 'model/wine_model_multiclass.pkl')  # for comparison

print("\n✅ Models saved to model/ folder:")
print("   - wine_model.pkl            (binary: Good/Bad)")
print("   - wine_model_multiclass.pkl (multiclass: 3-8)")
print("   - scaler.pkl")
print("   - features.pkl")

# =====================================================
# 6. Summary for your report
# =====================================================
print("\n" + "=" * 60)
print("SUMMARY FOR REPORT")
print("=" * 60)
print(f"Binary Model Accuracy:     {binary_acc*100:.2f}%")
print(f"Multiclass Model Accuracy: {multi_acc*100:.2f}%")
print(f"\nWhy the difference?")
print("  Most wines score 5 or 6 — the multiclass model struggles")
print("  to distinguish between adjacent scores. Binning into")
print("  Good/Bad produces a more reliable classifier.")