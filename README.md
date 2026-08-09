# 💳 Credit Card Risk Analyzer
link : https://credit-card-risk-analyzer-ef5tcdrjp9sbhl5njhveyw.streamlit.app/
-
A machine learning application for predicting credit card default risk and identifying customer segments based on customer financial and payment behavior.

## 🎯 Project Objective

The project aims to:

- Predict the probability of credit card default.
- Classify customers into different risk levels.
- Segment customers based on credit and payment behavior.
- Provide an interactive web interface for prediction.

## 🤖 Machine Learning

### Default Prediction
- XGBoost
- Probability-based default prediction
- ROC-AUC used for model evaluation

### Customer Segmentation
- K-Means Clustering
- Feature Scaling
- PCA for visualization

## 🔄 Project Pipeline

```text
Data
 ↓
Data Cleaning & EDA
 ↓
Feature Engineering
 ↓
XGBoost Model
 ↓
Default Probability
 ↓
Risk Classification

Customer Financial Data
 ↓
Scaling
 ↓
K-Means Clustering
 ↓
Customer Segment
