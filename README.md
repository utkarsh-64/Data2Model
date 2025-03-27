# Data2Model

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.29.0-FF4B4B)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3.0-orange)

A comprehensive web application for data preprocessing, exploratory data analysis, and machine learning model training/evaluation.

## Features

### Data Preprocessing
- **Data Cleaning**:
  - Missing value handling (Mean/Median/Mode imputation, KNN imputation)
  - Outlier detection & treatment (Z-score, IQR, Percentile methods)
  - Value replacement and column operations
- **Feature Engineering**:
  - One-Hot, Label, and Frequency encoding
  - DateTime feature extraction
  - Feature creation through mathematical operations
- **Advanced Processing**:
  - Dimensionality reduction (PCA)
  - Feature selection (SelectKBest)

### Machine Learning
- **Model Training**:
  - Classification: Logistic Regression, Random Forest, SVM, XGBoost
  - Regression: Linear Regression, Decision Tree Regressor, Gradient Boosting
- **Hyperparameter Tuning**:
  - GridSearchCV integration
  - Custom parameter grids
- **Model Evaluation**:
  - Classification Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC
  - Regression Metrics: MAE, MSE, RMSE, R² Score
  - Visualization: Confusion matrices, ROC curves, Residual plots

### EDA & Visualization
- Interactive data profiling with Sweetviz
- Statistical summaries
- Correlation analysis
- Distribution plots (Histograms, Box plots, Scatter plots)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ml-data-prep-studio.git
cd ml-data-prep-studio
