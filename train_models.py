# train_models.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, classification_report
import pickle
import os

# Create model directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Load dataset
train_df = pd.read_csv('adult_train.csv')
test_df = pd.read_csv('adult_test.csv')

# Remove invalid rows from test set (cross-validator row and rows with NaN target)
test_df = test_df[test_df['Target'].notna()]

# Data Preprocessing
def preprocess_data(df):
    df = df.copy()
    
    # Remove leading/trailing spaces
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    
    # Convert numeric columns
    numeric_cols = ['Age', 'fnlwgt', 'Education_Num', 'Capital_Gain', 'Capital_Loss', 'Hours_per_week']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Handle missing values
    df.fillna(df.median(numeric_only=True), inplace=True)
    
    # Separate features and target
    X = df.drop('Target', axis=1)
    # Handle all variations of target format (with/without period, with/without space)
    y = df['Target'].str.strip().str.rstrip('.').map({'<=50K': 0, '>50K': 1})
    
    # Remove rows with NaN target values
    valid_idx = ~y.isna()
    X = X[valid_idx]
    y = y[valid_idx]
    
    # Encode categorical features
    le_dict = {}
    categorical_cols = X.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        le_dict[col] = le
    
    return X, y, le_dict

# Preprocess training data
X_train, y_train, label_encoders = preprocess_data(train_df)

# Preprocess test data
X_test, y_test, _ = preprocess_data(test_df)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Save label encoders
with open('models/label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)

# Define models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'K-Nearest Neighbor': KNeighborsClassifier(n_neighbors=5),
    'Naive Bayes': GaussianNB(),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False)
}

# Train and evaluate models
results = []

for model_name, model in models.items():
    print(f"\nTraining {model_name}...")
    
    # Train model
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)
    
    results.append({
        'Model': model_name,
        'Accuracy': accuracy,
        'AUC': auc,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'MCC': mcc
    })
    
    print(f"{model_name} - Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")
    
    # Save model
    with open(f'models/{model_name.replace(" ", "_")}.pkl', 'wb') as f:
        pickle.dump(model, f)

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv('models/results.csv', index=False)
print("\n" + "="*80)
print(results_df.to_string())
print("="*80)