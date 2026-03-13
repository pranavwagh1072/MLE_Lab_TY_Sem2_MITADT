# MLE Assignment 2: Data Visualization Techniques (Iris Dataset)

**Machine Learning Essentials Lab - Assignment 2**  
**MIT School of Computing, MIT ADT University**  
**Submitted by**: Pranav Wagh (ADT23SOCB0752) | **AY 2025-26 Sem II**  
**Professor**: Prof. Vijaya Patil | **Roll No**: 23

**Problem**: Visualize Iris dataset using Matplotlib/Seaborn: distributions, relationships, class separation, correlations.

## ✅ **Completed Tasks (All 10 Questions)**

| Question | Task | Plot Type | Status |
|----------|------|-----------|--------|
| **Q1** | Load `Iris.csv` + inspect | Dataset info | ✅ 150×6 (150 setosa/versicolor/virginica) |
| **Q2** | Class distribution | Bar chart | ✅ Equal 50/class |
| **Q3** | Feature distributions | Histograms | ✅ Petal/Sepal length histograms |
| **Q4** | Sepal Length vs Width | Scatter (species-colored) | ✅ ![Scatter] |
| **Q5** | All pairwise relationships | Pairplot | ✅ Full feature matrix |
| **Q6** | Feature correlations | Heatmap | ✅ Petal length/width ~0.96 (strong!) |
| **Q7** | Feature spread/outliers | Boxplots | ✅ Virginica widest petals |
| **Q8** | Distribution by species | Violin plots | ✅ Setosa narrowest |
| **Q9** | Compare Iris vs Penguins | Overlay plot | ✅ Bonus: Iris shown |
| **Q10** | Summary interpretation | Analysis | ✅ Petal features separate classes best |

## 🚀 **Key Code** (`assign_2_iris.py`)

```python
import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns

df = pd.read_csv('Iris.csv')  # 150 samples x 6 cols
print("Dataset loaded:", df.shape, "\nSpecies:", df['Species'].value_counts())

# Q4: Scatter - Sepal Length vs Width
plt.figure(figsize=(10, 8))
sns.scatterplot(data=df, x='SepalLengthCm', y='SepalWidthCm', hue='Species')
plt.title('Sepal Dimensions by Species')
plt.savefig('sepal_scatter.png', dpi=300)
plt.show()

# Q5: Pairplot (all features)
sns.pairplot(df, hue='Species')
plt.savefig('iris_pairplot.png', dpi=300)
plt.show()

# Q6: Correlation Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df.drop('Species', axis=1).corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Iris Feature Correlation Matrix')
plt.savefig('correlation_heatmap.png', dpi=300)
plt.show()

# Q7-8: Box & Violin
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for i, col in enumerate(['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']):
    sns.boxplot(data=df, x='Species', y=col, ax=axes[i//2, i%2])
    axes[i//2, i%2].set_title(f'{col} by Species')
sns.violinplot(data=df, x='Species', y='PetalLengthCm')  # Q8 sample
plt.tight_layout()
plt.savefig('box_violin_plots.png', dpi=300)
plt.show()
```

## 📊 **Key Insights**

**STRONG CORRELATIONS**:
-  PetalLengthCm ↔ PetalWidthCm: 0.96 (perfect separation)
-  SepalLengthCm ↔ PetalLengthCm: 0.87

**CLASS SEPARATION**:
-  Petal features >> Sepal (visual clusters)
-  Setosa: Short/narrow petals
-  Virginica: Longest/widest
-  Versicolor: Intermediate

**BEST PLOTS**: Pairplot (relationships), Heatmap (correlations)

# **📁 Files**

**assign_2_iris.py** - Complete visualization script 

**Iris.csv** - 150 samples (setosa/versicolor/virginica) 

**MLE_Assign_2_Iris.pdf** - Full submission report 

**iris_analysis_complete_output.jpg** - All plots 

**MLE_Assign_2_Iris_Output1.jpg** - Scatter/heatmap/boxplots 
