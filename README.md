# Data-Science-ML-model
ML model on California Housing Dataset

This project is a machine learning pipeline for predicting California housing prices using the **California Housing Dataset**. It includes data preprocessing, model training, evaluation using cross-validation, and inference — all in a clean and production-friendly script.

## Features

- Stratified train/test split using income categories
- Data preprocessing with 'Pipeline' and 'ColumnTransformer'
- One-hot encoding for categorical data
- SimpleImputer to handle missing data
- 'RandomForestRegressor' model
- Cross-validated RMSE evaluation
- Model and pipeline persistence with 'joblib'
- Inference support on unseen data ('input.csv' → 'output.csv')

**Download the dataset**
- Get "housing.csv" from the official GitHub repository of the Hands-On ML book
  Place it in the same directory as main_new.py.

**Train or Inference**
- The script auto-detects if the model is already trained:

If no model exists, it will:
- Load data
- Preprocess
- Train the model
- Save the model & pipeline
- Export test set to input.csv

If model exists, it will:
- Load input.csv
- Predict housing prices
- Save results to output.csv

**Dependencies**
- Scikit-Learn
- Pandas
- Numpy
- Joblib
