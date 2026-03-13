# MLE Assignment 3: End-to-End ML Pipeline (California Housing)

**Machine Learning Essentials Lab - Assignment 3**  
**MIT School of Computing, MIT ADT University**  
**Submitted by**: Pranav Wagh (ADT23SOCB0752) | **AY 2025-26 Sem II**  
**Professor**: Prof. Vijaya Patil | **Roll No**: 23

**Problem**: Complete ML pipeline - impute missing values, scale features, train/test split, Linear Regression, evaluate, save predictions [Lab Manual pg 10-12].

## ✅ **All 10 Questions Completed**

| Question | Task | Result |
|----------|------|--------|
| **Q1** | Load dataset | 20,640 samples, 8 features, target: `median_house_value` |
| **Q2** | DataFrame + head(10) | First 10 rows displayed |
| **Q3** | Missing values | `total_bedrooms`: 207 missing (1%) |
| **Q4** | Impute | `SimpleImputer(strategy='median')` → 0 missing |
| **Q5** | Train-test split | 80/20 → 16,512 train, 4,128 test |
| **Q6** | Feature scaling | `StandardScaler()` (mean=0, std=1) |
| **Q7** | Train LinearRegression | Model trained successfully |
| **Q8** | Evaluate | **MSE: 5,059,928,371 \| RMSE: 71,133 \| R²: 0.6139** |
| **Q9** | Save predictions | `predictions.csv` (4,128 test predictions) ✅ |
| **Q10** | Reflection | Scaling most critical; errors in leakage/imputation |

## 📊 **Model Results**
MSE: 5,059,928,371.17 # Squared error scale ($200k+ prices)
RMSE: 71,133.17 # Avg $71k error (median target: $206k)
R² Score: 0.6139 # 61% variance explained (solid baseline)


**Pipeline**: `SimpleImputer` → `StandardScaler` → `LinearRegression`  
**Features**: longitude, latitude, housing_median_age, total_rooms, total_bedrooms, population, households, median_income

## 🚀 **Key Code** (`assign_3_california.py`)

```python
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('housing.csv')
X = df.drop('median_house_value', axis=1)
y = df['median_house_value']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('regressor', LinearRegression())
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
print(f"R²: {r2_score(y_test, y_pred):.4f}")

pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}).to_csv('predictions.csv')
```

## 💡 **Key Learnings** (Q10 Reflection)

**Most useful**: Feature scaling (prevents total_rooms dominating housing_median_age)

**Error risks**: Data leakage (scale before split), wrong imputation strategy

**Improve**: Add rooms/household ratios, encode ocean_proximity, RandomForestRegressor

## 📁 **Files**
**assign_3_california.py** - Production pipeline (R²~0.65 w/ categoricals)

**Assignment_3_Housing_Pipeline.ipynb** - Lab demo (step-by-step Q1-Q10)

**housing.csv** - 20,640 California housing records

**predictions.csv** - 4,128 test predictions (actual vs predicted)

**MLE_Assign_3_Housing.pdf** - Complete writeup


## 📈 **Performance Interpretation**

**RMSE $71k**: Acceptable for $206k median house prices

**R² 0.61**: Strong linear baseline (beats mean predictor)

**Ready**: Full pipeline automates preprocessing+prediction

**Conclusion**: Successfully built complete ML pipeline yielding RMSE=71,133, R²=0.6139. Feature scaling critical. Ready for RandomForest/XGBoost improvements.