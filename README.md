# Steel Surface Fault Classification using Machine Learning

## Overview
This project implements an end-to-end machine learning solution to classify surface faults in steel manufacturing using sensor and process data. The system automates fault identification to support faster and more consistent quality control decisions.

## Problem Statement
Manual steel surface inspection is time-consuming and inconsistent. This project predicts the type of surface fault using historical sensor data, enabling automated and reliable defect classification.

## Approach
- Consolidated multiple fault indicator columns into a single multiclass target
- Applied robust preprocessing for numerical features while preserving binary indicators
- Trained a Random Forest classifier suitable for non-linear and imbalanced data
- Optimized performance using cross-validation and hyperparameter tuning
- Exported trained model and preprocessing components for reuse and deployment

## Tools & Technologies
Python, Pandas, NumPy, Scikit-learn, Random Forest, ColumnTransformer, RobustScaler, GridSearchCV, Joblib

## Model Artifacts
- rf_fault_model.pkl – Trained Random Forest classifier
- preprocessor.pkl – Preprocessing pipeline

## Usage (Inference Example)
```python
import joblib
import pandas as pd

model = joblib.load("rf_fault_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")

sample = pd.DataFrame([{
    "X_Minimum": 42,
    "X_Maximum": 184,
    "Y_Minimum": 23,
    "Y_Maximum": 256,
    "TypeOfSteel_A300": 1,
    "TypeOfSteel_A400": 0
}])

sample_processed = preprocessor.transform(sample)
prediction = model.predict(sample_processed)
print("Predicted Fault Class:", prediction)
```

## Outcome
- Automated classification of multiple steel surface fault types
- Improved consistency and reliability in defect detection
- Delivered deployment-ready ML artifacts for industrial workflows

## Author
Vishal Gaud
