import os
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("final_dataset_with_real_gene_names.csv")

# Select top 15 features (based on previous feature importance)
top_features = [
    "PIK3CA_expr", "Pat_STK11_mutation", "HER2_expr", "BRCA1_expr", "KRAS_expr",
    "EGFR_expr", "TP53_expr", "Pat_KRAS_mutation", "Pat_TP53_mutation", "age",
    "Pat_Packs_Per_Year", "Pat_Smoking_Status", "CDH1_expr", "Pat_EGFR_mutation",
    "Pat_ALK_translocation"
]

# Prepare data
X = df[top_features].apply(pd.to_numeric, errors='coerce').fillna(df[top_features].mean())
y = LabelEncoder().fit_transform(df["disease_type"])  # Encode target: breast → 0, lung → 1

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Create model directory and save model
os.makedirs("model", exist_ok=True)
model_path = os.path.join("model", "cancer_predictor_rf_top15.pkl")
joblib.dump(model, model_path)

# Evaluate model
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)
confidence = model.predict_proba(X_test).max(axis=1).mean()

# Output results
print("\n✅ Model training complete!")
print(f"Model saved at: {os.path.abspath(model_path)}")
print(f"Training Accuracy: {train_acc:.4f}")
print(f"Testing Accuracy:  {test_acc:.4f}")
print(f"Average Confidence: {confidence:.4f}")
