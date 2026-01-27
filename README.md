# Adult Income Classification - Machine Learning Project

## Problem Statement

The goal of this project is to build and evaluate multiple machine learning classification models 
to predict whether a person's income is above or below $50,000 per year based on demographic and 
employment features. This is a binary classification problem.

## Dataset Description

**Dataset Name:** Adult Income Dataset (also known as Census Income Dataset)

**Source:** UCI Machine Learning Repository

**Dataset Statistics:**
- **Training Samples:** 32,563 instances
- **Test Samples:** 16,281 instances
- **Total Features:** 14 attributes
- **Target Variable:** Income (binary: <=50K or >50K)

**Features:**
1. **Age** (numeric): Age of the individual
2. **Workclass** (categorical): Employment sector
3. **fnlwgt** (numeric): Final weight (census sampling weight)
4. **Education** (categorical): Highest level of education attained
5. **Education_Num** (numeric): Numerical representation of education level
6. **Martial_Status** (categorical): Marital status of the individual
7. **Occupation** (categorical): Type of occupation
8. **Relationship** (categorical): Relationship status
9. **Race** (categorical): Race/ethnicity
10. **Sex** (categorical): Gender (Male/Female)
11. **Capital_Gain** (numeric): Capital gains
12. **Capital_Loss** (numeric): Capital losses
13. **Hours_per_week** (numeric): Hours worked per week
14. **Country** (categorical): Native country

**Data Preprocessing:**
- Handled missing values using median imputation for numeric features
- Standardized numeric features using StandardScaler
- Encoded categorical features using LabelEncoder
- Target variable: Binary encoding (<=50K: 0, >50K: 1)

## Models Used

### 1. Logistic Regression
- **Description:** Linear model for binary classification
- **Hyperparameters:** max_iter=1000
- **Advantages:** Interpretable, fast training, works well for linearly separable data
- **Disadvantages:** Assumes linear relationship between features and target

### 2. Decision Tree Classifier
- **Description:** Tree-based model that recursively splits data
- **Hyperparameters:** Default parameters with random_state=42
- **Advantages:** Easy to interpret, handles non-linear relationships
- **Disadvantages:** Prone to overfitting, sensitive to data variations

### 3. K-Nearest Neighbor (KNN) Classifier
- **Description:** Instance-based learning algorithm
- **Hyperparameters:** n_neighbors=5
- **Advantages:** Simple to understand, no training phase required
- **Disadvantages:** Computationally expensive, sensitive to feature scaling

### 4. Naive Bayes Classifier (Gaussian)
- **Description:** Probabilistic classifier based on Bayes' theorem
- **Hyperparameters:** Default Gaussian parameters
- **Advantages:** Fast, effective for text and categorical data
- **Disadvantages:** Assumes feature independence

### 5. Random Forest Classifier (Ensemble)
- **Description:** Ensemble of decision trees
- **Hyperparameters:** n_estimators=100, random_state=42
- **Advantages:** Reduces overfitting, handles non-linear data well, feature importance
- **Disadvantages:** Less interpretable, slower prediction

### 6. XGBoost Classifier (Ensemble)
- **Description:** Gradient boosting machine with optimized performance
- **Hyperparameters:** random_state=42, eval_metric='logloss'
- **Advantages:** State-of-the-art performance, handles missing values
- **Disadvantages:** Complex tuning, requires careful hyperparameter optimization

## Evaluation Metrics

The following metrics were calculated for each model:

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8254 | 0.8527 | 0.7039 | 0.4501 | 0.5491 | 0.4644 |
| Decision Tree | 0.8097 | 0.7413 | 0.5945 | 0.6115 | 0.6029 | 0.4779 |
| K-Nearest Neighbor | 0.8289 | 0.8505 | 0.6566 | 0.5777 | 0.6147 | 0.5070 |
| Naive Bayes | 0.8051 | 0.8554 | 0.6781 | 0.3331 | 0.4467 | 0.3769 |
| Random Forest (Ensemble) | 0.8552 | 0.9046 | 0.7358 | 0.6040 | 0.6634 | 0.5769 |
| XGBoost (Ensemble) | 0.8701 | 0.9239 | 0.7867 | 0.6175 | 0.6919 | 0.6183 |

**Metric Definitions:**
- **Accuracy:** Proportion of correct predictions among total predictions
- **AUC (Area Under Curve):** Probability that the model ranks a random positive example higher than a random negative example
- **Precision:** Proportion of true positives among predicted positives
- **Recall:** Proportion of true positives among actual positives
- **F1 Score:** Harmonic mean of precision and recall
- **MCC (Matthews Correlation Coefficient):** Correlation between predicted and actual values (-1 to 1)

## Model Performance Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Linear model with reasonable performance; suitable as baseline |
| Decision Tree | May overfit on training data; useful for feature importance analysis |
| K-Nearest Neighbor | Sensitive to feature scaling; moderate performance |
| Naive Bayes | Fast training and prediction; assumes feature independence |
| Random Forest (Ensemble) | Strong ensemble performance with reduced overfitting |
| XGBoost (Ensemble) | Best overall performance; captures complex patterns in data |
