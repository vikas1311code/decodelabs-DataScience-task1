import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("  DECODELABS INTERNSHIP - DATA SCIENCE PROJECT")
print("  Titanic Survival Prediction")
print("=" * 60)

# TASK 2: DATA CLEANING
print("\n--- TASK 2: DATA CLEANING & PREPROCESSING ---")
df = pd.read_csv('titanic.csv')
print(f"Dataset Shape: {df.shape}")
print(f"\n[Before Cleaning] Missing Values:\n{df.isnull().sum()}")

df['age'].fillna(df['age'].median(), inplace=True)
df['age'] = df['age'].fillna(df['age'].median())
df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)
before = len(df)
df.drop_duplicates(inplace=True)
print(f"Duplicates removed: {before - len(df)}")
df['sex'] = df['sex'].astype('category')
df['embarked'] = df['embarked'].astype('category')
df['pclass'] = df['pclass'].astype('category')

print(f"\n[After Cleaning] Missing Values:\n{df.isnull().sum()}")
print("✅ Task 2 Complete!")

# TASK 3: EDA
print("\n--- TASK 3: EXPLORATORY DATA ANALYSIS ---")
print(df.describe())
print(f"\nOverall Survival Rate: {df['survived'].mean()*100:.1f}%")
print(f"Survival by Gender:\n{df.groupby('sex')['survived'].mean().apply(lambda x: f'{x*100:.1f}%')}")
print(f"Survival by Class:\n{df.groupby('pclass')['survived'].mean().apply(lambda x: f'{x*100:.1f}%')}")

fig = plt.figure(figsize=(16, 12))
fig.suptitle('TASK 3: EDA - Titanic Dataset', fontsize=16, fontweight='bold')
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

ax1 = fig.add_subplot(gs[0, 0])
sc = df['survived'].value_counts().sort_index()
bars = ax1.bar(['Not Survived', 'Survived'], sc.values, color=['#e74c3c', '#2ecc71'], edgecolor='white')
for bar, val in zip(bars, sc.values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3, str(val), ha='center', fontweight='bold')
ax1.set_title('Survival Count', fontweight='bold')

ax2 = fig.add_subplot(gs[0, 1])
gs2 = df.groupby('sex')['survived'].mean() * 100
bars2 = ax2.bar(gs2.index.astype(str), gs2.values, color=['#3498db', '#e91e8c'], edgecolor='white')
for bar, val in zip(bars2, gs2.values):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val:.1f}%', ha='center', fontweight='bold')
ax2.set_title('Survival by Gender', fontweight='bold')
ax2.set_ylim(0, 100)

ax3 = fig.add_subplot(gs[0, 2])
cs = df.groupby('pclass')['survived'].mean() * 100
ax3.bar([f'Class {c}' for c in cs.index.astype(str)], cs.values, color=['#f39c12','#95a5a6','#cd7f32'], edgecolor='white')
for i, val in enumerate(cs.values):
    ax3.text(i, val + 1, f'{val:.1f}%', ha='center', fontweight='bold')
ax3.set_title('Survival by Class', fontweight='bold')
ax3.set_ylim(0, 100)

ax4 = fig.add_subplot(gs[1, 0])
ax4.hist(df[df['survived']==0]['age'], bins=25, alpha=0.6, color='#e74c3c', label='Not Survived')
ax4.hist(df[df['survived']==1]['age'], bins=25, alpha=0.6, color='#2ecc71', label='Survived')
ax4.set_title('Age Distribution', fontweight='bold')
ax4.legend()

ax5 = fig.add_subplot(gs[1, 1])
bp = ax5.boxplot([df[df['pclass']==c]['fare'].values for c in [1,2,3]],
                 labels=['1st','2nd','3rd'], patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('#3498db')
    patch.set_alpha(0.6)
ax5.set_title('Fare by Class', fontweight='bold')

ax6 = fig.add_subplot(gs[1, 2])
sns.heatmap(df[['survived','age','fare','sibsp','parch']].corr(),
            annot=True, fmt='.2f', cmap='RdYlGn', center=0, ax=ax6)
ax6.set_title('Correlation Heatmap', fontweight='bold')

plt.savefig('task3_eda.png', dpi=150, bbox_inches='tight')
print("✅ Task 3 Complete! Chart saved: task3_eda.png")

# TASK 5: PREDICTIVE MODEL
print("\n--- TASK 5: PREDICTIVE MODEL ---")
df_model = df.copy()
le = LabelEncoder()
df_model['sex_enc'] = le.fit_transform(df_model['sex'].astype(str))
df_model['embarked_enc'] = le.fit_transform(df_model['embarked'].astype(str))
df_model['pclass_enc'] = df_model['pclass'].astype(str).map({'1':1,'2':2,'3':3})

features = ['pclass_enc','sex_enc','age','sibsp','parch','fare','embarked_enc']
X = df_model[features].dropna()
y = df_model.loc[X.index, 'survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)
lr_acc = accuracy_score(y_test, lr.predict(X_test))

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)

print(f"Logistic Regression Accuracy: {lr_acc*100:.2f}%")
print(f"Random Forest Accuracy:       {rf_acc*100:.2f}%")
print(classification_report(y_test, rf_pred, target_names=['Not Survived','Survived']))

importances = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)

fig2, axes = plt.subplots(1, 3, figsize=(16, 5))
fig2.suptitle('TASK 5: ML Model Results', fontsize=14, fontweight='bold')

cm = confusion_matrix(y_test, rf_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Survived','Survived'],
            yticklabels=['Not Survived','Survived'], ax=axes[0])
axes[0].set_title('Confusion Matrix', fontweight='bold')

axes[1].barh(importances.index, importances.values, color='#3498db')
axes[1].set_title('Feature Importances', fontweight='bold')

axes[2].bar(['Logistic\nRegression','Random\nForest'], [lr_acc*100, rf_acc*100],
            color=['#e67e22','#27ae60'], edgecolor='white')
for i, val in enumerate([lr_acc*100, rf_acc*100]):
    axes[2].text(i, val + 0.5, f'{val:.1f}%', ha='center', fontweight='bold')
axes[2].set_title('Model Comparison', fontweight='bold')
axes[2].set_ylim(0, 100)

plt.tight_layout()
plt.savefig('task5_model.png', dpi=150, bbox_inches='tight')
print("✅ Task 5 Complete! Chart saved: task5_model.png")

print("\n" + "=" * 60)
print("  🎉 PROJECT COMPLETE! Tasks 2 + 3 + 5 Done!")
print("=" * 60)