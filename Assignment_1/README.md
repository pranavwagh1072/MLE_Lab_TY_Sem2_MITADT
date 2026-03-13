# MLE Assignment 1: Titanic Data Acquisition & EDA

**Machine Learning Essentials Lab - Assignment 1**  
**MIT School of Computing, MIT ADT University**  
**Submitted by**: Pranav Wagh (ADT23SOCB0752) | **AY 2025-26 Sem II**  
**Professor**: Prof. Vijaya Patil | **Roll No**: 23

**Problem**: Load Titanic dataset, clean missing values (Age/Embarked), encode categoricals (Sex/Embarked), summarize statistics, visualize survival by gender/Pclass.

## ✅ **Completed Tasks (All 10 Questions)**

| Question | Task | Status | Output |
|----------|------|--------|--------|
| **Q1** | Load `train.csv` + inspect (shape, dtypes, head) | ✅ | 891 rows × 12 cols |
| **Q2** | Missing values | ✅ | Age: 177 (20%), Embarked: 2 |
| **Q3** | Clean: Median Age (28), Mode Embarked ('S') | ✅ | 0 missing post-clean |
| **Q4** | Encode: Sex (M=0,F=1), Embarked (S=0,C=1,Q=2) | ✅ | Numeric `df_clean` |
| **Q5** | Summary stats (`describe()`) | ✅ | Survival rate: **38.4%** |
| **Q6** | Groupby: Survival by Gender/Pclass | ✅ | Female: **74%**, Male: **19%**<br>1st Class: **63%**, 3rd: **24%** |
| **Q7** | Survival count plot | ✅ | [Output image] |
| **Q8** | Survival by Gender bar | ✅ | ![Gender] |
| **Q9** | Survival by Pclass+Gender | ✅ | ![Pclass+Gender] |
| **Q10** | Histogram Age (bonus) | ✅ | Age dist plotted |

## 🚀 **Key Code** (`assign_1_titanic.py`)

```python
import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from sklearn.impute import SimpleImputer

df = pd.read_csv('train.csv')  # 891 passengers
print("Shape:", df.shape)

# Q2-3: Clean missing
age_imputer = SimpleImputer(strategy='median')
df['Age'] = age_imputer.fit_transform(df[['Age']]).ravel()  # Median=28

emb_imputer = SimpleImputer(strategy='most_frequent')
df['Embarked'] = emb_imputer.fit_transform(df[['Embarked']]).ravel()  # 'S'

# Q4: Encode
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = pd.Categorical(df['Embarked'], categories=['S', 'C', 'Q']).codes

df_clean = df.drop(['Cabin', 'Name', 'Ticket', 'PassengerId'], axis=1)

# Q5-6: Summary
print("Survival Rate:", round(df_clean['Survived'].mean(), 3))  # 0.384
print("By Gender:\n", df_clean.groupby('Sex')['Survived'].mean())
print("By Pclass:\n", df_clean.groupby('Pclass')['Survived'].mean())

# Q7-9: Visualize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
sns.barplot(data=df_clean, x='Sex', y='Survived', ax=ax1)
ax1.set_title('Survival Rate by Gender (0=Male, 1=Female)')
sns.barplot(data=df_clean, x='Pclass', y='Survived', hue='Sex', ax=ax2)
ax2.set_title('Survival Rate by Pclass & Gender')
plt.tight_layout()
plt.savefig('titanic_survival_analysis.png', dpi=300, bbox_inches='tight')
plt.show()
📊 Key Insights
Overall Survival: 38.4% (342/891 survived)
CRITICAL HIERARCHY:
1. 1st Class Females: ~97%
2. 2nd Class Females: ~94% 
3. 1st Class Males:   ~37%
4. 3rd Class Females: ~50%
5. 3rd Class Males:   ~14%
Dropped: Cabin (77% missing), Name/Ticket/PassengerId (non-predictive)

Ready for ML: df_clean (891×8, numeric, no nulls)

📁 Files
assign_1_titanic.py - Complete Python script

train.csv - Titanic Kaggle dataset (891 records)

MLE_Assign_1_Titanic.pdf - Full submission report

visual_output_titanic_survival_analysis.jpg - Gender/Pclass plots

Output.jpg - Pandas summary/console
