# MLE Assignment 4: End-to-End ML Pipeline LINEAR & POLYNOMIAL REGRESSION - AUTO MPG ANALYSIS 

**Machine Learning Essentials Lab - Assignment 4**  
**MIT School of Computing, MIT ADT University**  
**Submitted by**: Pranav Wagh (ADT23SOCB0752) | **AY 2025-26 Sem II**  
**Professor**: Prof. Vijaya Patil | **Roll No**: 23


## 📋 Overview
Predicts fuel efficiency (MPG) from engine specs. Compares **linear underfitting**, **degree 2 optimality**, **degree 3+ overfitting** via diagnostics.

**Results Summary:**
| Model | Train R² | Test R² | RMSE  | MAE   | Status |
|-------|----------|---------|-------|-------|--------|
| Linear (1) | 0.6996 | **0.7271** | 3.8302 | 3.1192 | Underfits |
| Polynomial 2 | 0.7544 | **0.7782** | ↓ | ↓ | 🏆 BEST |
| Polynomial 3 | 0.7828 | 0.7808 | ↑ | ↑ | Overfits |


## 📁 Repository Contents
| File | Size | Purpose |
|------|------|---------|
| `assign4_autompg.ipynb` | Notebook | **Main executable analysis** |
| `MLE_Assign4_Autompg.pdf` | 1.7MB | PDF submission |
| `auto-mpg.csv` | 18KB | Dataset (398×9) |
| `Learning-Curve-Linear-Regression.jpg` | 53KB | Underfitting proof |
| `Validation-Curve.jpg` | 59KB | Overfitting proof |
| `README.md` | - | This file |

## 🚀 Quick Start
```
# 1. Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn jupyter

# 2. Run analysis
jupyter notebook assign4_autompg.ipynb

# 3. Key cells:
# - Q1: Load → (398, 9)
# - Q2: Median impute horsepower (?→93.5)
# - Q5: Intercept=45.6376
# - Q6-Q9: Curves confirm theory
```
# 🔬 **Detailed Results**
Linear Regression:
Intercept: 45.6376
Coeffs: cylinders(-0.1862), hp(-0.0403), weight(-0.0053), disp(-0.0056)
MAE: 3.1192 | MSE: 14.6704 | RMSE: 3.8302 | R²: 0.7271

Polynomial Comparison:
Deg 2: Train=0.7544/Test=0.7782 (+6.9% vs linear)
Deg 3: Train=0.7828/Test=0.7808 (gap=0.002 → early overfit)

# 📈 **Analysis Highlights**
**Underfitting**: Learning curve high MSE plateau → linear too simple

**Optimal**: Validation peak degree 2 → best generalization

**Overfitting**: Post-degree 2 test R² drop → complexity penalty

**Features**: Weight/horsepower dominate (negative coeffs expected)

# 🎯 **Lab Objectives Met**
✅ Q1-Q4: Data prep & split
✅ Q5-Q6: Linear model + metrics
✅ Q7: Polynomial comparison
✅ Q8: Learning curve (bias detection)
✅ Q9: Validation curve (hyperparameter tuning)
✅ Q10: Theoretical reflection

# 📚 **References**
**Dataset**: UCI Auto MPG 


