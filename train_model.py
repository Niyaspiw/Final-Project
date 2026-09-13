import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
df = pd.read_csv(url, sep=';')
print("Dataset shape:", df.shape)

# 2. Create binary target (good=1 if quality >= 7, else 0)
df['quality_label'] = df['quality'].apply(lambda x: 1 if x >= 7 else 0)

# 3. Split features and target
X = df.drop(['quality', 'quality_label'], axis=1)
y = df['quality_label']
feature_names = X.columns.tolist()

# 4. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Train model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

# 7. Evaluate
y_pred = model.predict(X_test_scaled)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 8. Save model, scaler, and feature names
os.makedirs('model', exist_ok=True)
joblib.dump(model, 'model/wine_model.pkl')
joblib.dump(scaler, 'model/scaler.pkl')
joblib.dump(feature_names, 'model/features.pkl')
print("\nModel saved to model/ folder.")