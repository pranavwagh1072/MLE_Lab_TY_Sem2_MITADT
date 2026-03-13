import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
import os

df = pd.read_csv('D:/MLE/Assignment_1/train.csv')  

print("## Titanic Dataset Loaded ##")
print("Shape:", df.shape)
print("\nMissing values before cleaning:")
print(df.isnull().sum())

age_imputer = SimpleImputer(strategy='median')
df['Age'] = age_imputer.fit_transform(df[['Age']]).ravel()

emb_imputer = SimpleImputer(strategy='most_frequent')
df['Embarked'] = emb_imputer.fit_transform(df[['Embarked']]).ravel()

print("\nMissing values after cleaning:")
print(df[['Age', 'Embarked']].isnull().sum())

df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})  
df['Embarked'] = pd.Categorical(df['Embarked'], categories=['S', 'C', 'Q']).codes  

df_clean = df.drop(['Cabin', 'Name', 'Ticket', 'PassengerId'], axis=1)

# Step 4: Summarize with Pandas
print("\n## PANDAS SUMMARY ##")
print(df_clean.describe(include='all'))
print("\nOverall Survival Rate:", round(df_clean['Survived'].mean(), 3))

print("\nSurvival by Gender:", df_clean.groupby('Sex')['Survived'].mean())
print("Survival by Pclass:", df_clean.groupby('Pclass')['Survived'].mean())

plt.style.use('default')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

sns.barplot(data=df_clean, x='Sex', y='Survived', ax=ax1)
ax1.set_title('Survival Rate by Gender\n(0=Male, 1=Female)', fontsize=14)
ax1.set_xlabel('Sex')
for p in ax1.patches:
    ax1.annotate(f'{p.get_height():.1%}', (p.get_x() + p.get_width()/2., p.get_height()), 
                ha='center', va='bottom')

sns.barplot(data=df_clean, x='Pclass', y='Survived', hue='Sex', ax=ax2)
ax2.set_title('Survival Rate by Pclass & Gender', fontsize=14)
ax2.set_xlabel('Passenger Class')
ax2.legend(title='Sex')

plt.tight_layout()
plt.savefig('titanic_survival_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n## VISUALIZATION SAVED ##")
print("✓ titanic_survival_analysis.png (submit this!)")
print("✓ Console output above for summary")
print("\nAssignment Complete! Dataset: cleaned, encoded, summarized, visualized.")
