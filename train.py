import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib

# --- Load data ---
data = pd.read_csv("Telco-Customer-Churn.csv")

# --- Preprocessing ---
# Convert TotalCharges to numeric
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
data['TotalCharges'].fillna(0, inplace=True)

# Drop irrelevant column
if 'customerID' in data.columns:
    data = data.drop('customerID', axis=1)

# Convert binary columns to 0/1
binary_cols = ['Partner','Dependents','SeniorCitizen']
for col in binary_cols:
    data[col] = data[col].replace({'Yes':1,'No':0})

# Define target and features
target = 'Churn'
X = data.drop(target, axis=1)
y = data[target].map({'Yes':1, 'No':0})

# Separate numeric and categorical features
numeric_features = ['SeniorCitizen','tenure','MonthlyCharges','TotalCharges','Partner','Dependents']
categorical_features = ['gender','Contract','InternetService','DeviceProtection','PaperlessBilling','PhoneService']

# --- Train-test split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- Pipeline ---
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', GradientBoostingClassifier())
])

# --- Train ---
pipeline.fit(X_train, y_train)

# --- Predictions & metrics ---
y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:,1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

print("--- Gradient Boosting ---")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")

# --- Save pipeline ---
joblib.dump(pipeline, "churn_pipeline.pkl")
print("Pipeline saved as churn_pipeline.pkl")
