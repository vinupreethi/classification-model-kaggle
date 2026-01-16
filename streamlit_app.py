# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


# Page configuration
st.set_page_config(page_title="ML Classification Models", layout="wide")

# Title and description
st.title("🎯 Adult Income Classification Models")
st.markdown("""
This application demonstrates multiple machine learning classification models trained on the Adult dataset.
The goal is to predict whether a person's income is above or below $50,000 per year.
""")

# Load models and data
@st.cache_resource
def load_models():
    models = {}
    model_files = ['Logistic_Regression', 'Decision_Tree', 'K-Nearest_Neighbor', 
                   'Naive_Bayes', 'Random_Forest', 'XGBoost']
    
    for model_name in model_files:
        try:
            with open(f'models/{model_name}.pkl', 'rb') as f:
                models[model_name.replace('_', ' ')] = pickle.load(f)
        except:
            pass
    
    return models

@st.cache_resource
def load_scaler():
    with open('models/scaler.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_resource
def load_label_encoders():
    with open('models/label_encoders.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_data
def load_results():
    return pd.read_csv('models/results.csv')

# Load all resources
models = load_models()
scaler = load_scaler()
label_encoders = load_label_encoders()
results_df = load_results()

# Sidebar
st.sidebar.header("📊 Options")
selected_model = st.sidebar.selectbox("Select a Model:", list(models.keys()))

# Main content
col1, col2 = st.columns(2)

with col1:
    st.header("📈 Model Performance Metrics")
    
    # Filter results for selected model
    model_results = results_df[results_df['Model'] == selected_model]
    
    if not model_results.empty:
        metrics = model_results.iloc[0]
        
        col_metric1, col_metric2, col_metric3 = st.columns(3)
        with col_metric1:
            st.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
            st.metric("Precision", f"{metrics['Precision']:.4f}")
        with col_metric2:
            st.metric("AUC Score", f"{metrics['AUC']:.4f}")
            st.metric("Recall", f"{metrics['Recall']:.4f}")
        with col_metric3:
            st.metric("F1 Score", f"{metrics['F1']:.4f}")
            st.metric("MCC Score", f"{metrics['MCC']:.4f}")

with col2:
    st.header("📋 All Models Comparison")
    st.dataframe(results_df.round(4), use_container_width=True)

# Dataset upload section
st.header("📁 Test Data Upload")
uploaded_file = st.file_uploader("Upload a CSV file (test data)", type="csv")

if uploaded_file is not None:
    test_data = pd.read_csv(uploaded_file)
    st.write(f"Dataset shape: {test_data.shape}")
    st.dataframe(test_data.head())
    
    if st.button("Make Predictions"):
        try:
            # Preprocess the uploaded data
            test_data_processed = test_data.copy()
            test_data_processed = test_data_processed.applymap(lambda x: x.strip() if isinstance(x, str) else x)
            
            # Convert numeric columns
            numeric_cols = ['Age', 'fnlwgt', 'Education_Num', 'Capital_Gain', 'Capital_Loss', 'Hours_per_week']
            for col in numeric_cols:
                test_data_processed[col] = pd.to_numeric(test_data_processed[col], errors='coerce')
            
            test_data_processed.fillna(test_data_processed.median(numeric_only=True), inplace=True)
            
            # Remove target column if exists
            if 'Target' in test_data_processed.columns:
                y_actual = test_data_processed['Target'].map({' <=50K': 0, ' <=50K.': 0, ' >50K': 1, ' >50K.': 1})
                test_data_processed = test_data_processed.drop('Target', axis=1)
            else:
                y_actual = None
            
            # Encode categorical features
            for col in test_data_processed.select_dtypes(include=['object']).columns:
                if col in label_encoders:
                    test_data_processed[col] = label_encoders[col].transform(test_data_processed[col])
            
            # Scale features
            test_data_scaled = scaler.transform(test_data_processed)
            
            # Make predictions
            model = models[selected_model]
            predictions = model.predict(test_data_scaled)
            probabilities = model.predict_proba(test_data_scaled)
            
            # Display results
            st.subheader(f"Predictions using {selected_model}")
            
            results_table = pd.DataFrame({
                'Prediction': ['<=50K' if p == 0 else '>50K' for p in predictions],
                'Probability (<=50K)': probabilities[:, 0],
                'Probability (>50K)': probabilities[:, 1]
            })
            
            st.dataframe(results_table, use_container_width=True)
            
            # Show confusion matrix if actual labels are available
            if y_actual is not None:
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_actual, predictions)
                
                fig, ax = plt.subplots(figsize=(6, 5))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                           xticklabels=['<=50K', '>50K'], yticklabels=['<=50K', '>50K'])
                ax.set_ylabel('Actual')
                ax.set_xlabel('Predicted')
                st.pyplot(fig)
                
                # Classification Report
                st.subheader("Classification Report")
                report = classification_report(y_actual, predictions, 
                                             target_names=['<=50K', '>50K'])
                st.text(report)
        
        except Exception as e:
            st.error(f"Error making predictions: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
**About this App:** This Streamlit application demonstrates 6 machine learning classification models 
trained on the Adult Income dataset for binary income classification.
""")