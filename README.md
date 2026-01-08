
# Steel Surface Fault Classification using Machine Learning

## 💡 Idea
This project aims to automate the identification of surface faults in steel manufacturing using machine learning. By leveraging sensor and process data, the system predicts fault types accurately, reducing dependency on manual inspection and improving quality control efficiency.

## ⚠️ Challenges
- Fault labels were distributed across multiple one-hot encoded columns  
- Class imbalance among different fault categories  
- Presence of outliers in sensor measurements  
- Maintaining preprocessing consistency during inference  

## 🛠️ How These Challenges Were Overcome
- Merged fault indicator columns into a single multiclass target variable  
- Applied balanced learning strategies with Random Forest  
- Used RobustScaler to minimize the impact of outliers  
- Saved preprocessing and model pipelines separately for reproducibility  

## 🚀 Approach
- Data preprocessing using Pandas and Scikit-learn  
- Feature scaling with ColumnTransformer and RobustScaler  
- Model training using Random Forest classifier  
- Hyperparameter tuning with GridSearchCV  
- Exported deployment-ready artifacts using Joblib  

## 🧰 Tools & Technologies
Python, Pandas, NumPy, Scikit-learn, Random Forest, ColumnTransformer, RobustScaler, GridSearchCV, Joblib

## 📦 Project Structure
```
.
├── data/
│   └── faults.csv
├── faulty_steel.ipynb
├── preprocessor.pkl
├── rf_fault_model.pkl
├── README.md
```

## ✅ Result
- Automated multiclass classification of steel surface faults  
- Improved consistency and scalability in defect detection  
- Delivered reusable, deployment-ready ML components  

## 🔍 How to Use the Model
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

## 👤 Author
**Vishal Gaud**
