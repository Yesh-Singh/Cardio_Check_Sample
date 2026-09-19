# ==============================================================================
# PROJECT 1: CARDIOVASCULAR DISEASE PREDICTION
# ==============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Machine Learning Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

# Set Visualization Style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# ==============================================================================
# STEP 1: DATA PRE-PROCESSING OPERATIONS
# ==============================================================================
print("--- STEP 1: DATA PRE-PROCESSING ---")

# 1.1 Load the Dataset handling the semicolon (;) delimiter shown in your Excel file
file_name = 'cardio_train.csv'

# Auto-check if filename is 'cardio_train (1).csv' as seen in your screenshot title bar
if not os.path.exists(file_name) and os.path.exists('cardio_train (1).csv'):
    file_name = 'cardio_train (1).csv'

print(f"Loading file: '{file_name}'...")
df = pd.read_csv(file_name, sep=';')

print(f"Successfully Loaded Dataset! Shape: {df.shape} (Rows x Columns)")
print("\nFirst 5 rows of formatted data:")
print(df.head())

# 1.2 Drop 'id' column as it doesn't contribute to predictions
if 'id' in df.columns:
    df.drop('id', axis=1, inplace=True)

# 1.3 Handle Duplicate Rows
duplicates = df.duplicated().sum()
print(f"\nDuplicate rows found and removed: {duplicates}")
df.drop_duplicates(inplace=True)

# 1.4 Check Missing Values
print(f"Total Missing Values: {df.isnull().sum().sum()}")

# 1.5 Convert 'age' from Days to Years
df['age'] = (df['age'] / 365.25).round().astype(int)

# 1.6 Feature Engineering: Add Body Mass Index (BMI)
# BMI = weight (kg) / (height (m))^2
df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)

# 1.7 Clean Anomalies/Outliers (Blood Pressure & Physical metrics)
# Systolic (ap_hi) & Diastolic (ap_lo) cleaning
df = df[(df['ap_lo'] >= 40) & (df['ap_lo'] <= 180)]
df = df[(df['ap_hi'] >= 70) & (df['ap_hi'] <= 220)]
df = df[df['ap_hi'] >= df['ap_lo']]  # Systolic must be higher than Diastolic

# Height and Weight cleaning
df = df[(df['height'] >= 100) & (df['height'] <= 220)]
df = df[(df['weight'] >= 30) & (df['weight'] <= 200)]

print(f"Cleaned Dataset Shape: {df.shape}\n")


# ==============================================================================
# STEP 2: DATA ANALYSIS AND VISUALIZATIONS (EDA)
# ==============================================================================
print("--- STEP 2: EXPLORATORY DATA ANALYSIS (EDA) ---")

# 2.1 Distribution of Cardiovascular Disease (Target Variable)
plt.figure(figsize=(6, 4))
ax = sns.countplot(x='cardio', data=df, palette='Set2')
plt.title('Target Variable Distribution (0 = Healthy, 1 = Disease)')
plt.xlabel('Cardio Disease Status')
plt.ylabel('Patient Count')
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', (p.get_x() + 0.3, p.get_height() + 100))
plt.tight_layout()
plt.show()

# 2.2 Age Distribution vs Cardiovascular Disease
plt.figure(figsize=(10, 5))
sns.countplot(x='age', hue='cardio', data=df, palette='Set1')
plt.title('Cardiovascular Disease Occurrence by Age (Years)')
plt.xlabel('Age')
plt.ylabel('Count')
plt.legend(['No Disease', 'Disease'])
plt.tight_layout()
plt.show()

# 2.3 Categorical Features Analysis vs Target
cat_features = ['cholesterol', 'gluc', 'smoke', 'alco', 'active']
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()

for idx, col in enumerate(cat_features):
    sns.countplot(x=col, hue='cardio', data=df, ax=axes[idx], palette='tab10')
    axes[idx].set_title(f'{col.capitalize()} vs Disease Target')
    axes[idx].legend(['No Disease', 'Disease'])

fig.delaxes(axes[5])  # Remove unused 6th subplot space
plt.tight_layout()
plt.show()


# ==============================================================================
# STEP 3: CORRELATION MATRIX OF FEATURES
# ==============================================================================
print("--- STEP 3: CORRELATION MATRIX ---")

plt.figure(figsize=(12, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix Heatmap of All Dataset Features')
plt.tight_layout()
plt.show()


# ==============================================================================
# STEP 4: MODEL TRAINING & ACCURACY COMPARISON
# ==============================================================================
print("--- STEP 4: MACHINE LEARNING MODELS ACCURACY EVALUATION ---")

# Separate Features (X) and Target Label (y)
X = df.drop('cardio', axis=1)
y = df['cardio']

# Split Data into 80% Training and 20% Testing Sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standardize features (scale mean=0, std=1)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the classifiers available in the current Python environment.
models = {
    'Logistic Regression (LR)': LogisticRegression(max_iter=1000, random_state=42),
    'Support Vector Machine (SVM)': LinearSVC(C=1.0, max_iter=2000, random_state=42)
}

# Evaluate all models
results = {}

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    results[name] = round(acc * 100, 2)

# Display Comparison Summary Table
results_df = pd.DataFrame(list(results.items()), columns=['ML Algorithm', 'Accuracy (%)'])
results_df = results_df.sort_values(by='Accuracy (%)', ascending=False).reset_index(drop=True)

print("\n=============================================")
print("          MODEL ACCURACY RESULTS TABLE       ")
print("=============================================")
print(results_df.to_string(index=False))
print("=============================================\n")

# Barplot comparison of Model Accuracy
plt.figure(figsize=(9, 5))
barplot = sns.barplot(x='Accuracy (%)', y='ML Algorithm', data=results_df, palette='viridis')
plt.title('Accuracy Comparison of Machine Learning Models')
plt.xlim(50, 100)
for index, value in enumerate(results_df['Accuracy (%)']):
    plt.text(value + 0.3, index, f'{value}%', va='center', fontweight='bold')
plt.tight_layout()
plt.show()


# ==============================================================================
# STEP 5: BUILD FINAL MODEL & PATIENT DIAGNOSIS DETECTOR
# ==============================================================================
print("--- STEP 5: FINAL MODEL SELECTION & DETECTION SYSTEM ---")

# Automatically select the best performing model
best_model_name = results_df.iloc[0]['ML Algorithm']
best_accuracy = results_df.iloc[0]['Accuracy (%)']

print(f"🏆 Selected Best Model: {best_model_name} with Accuracy: {best_accuracy}%\n")

final_model = models[best_model_name]
y_pred_best = final_model.predict(X_test_scaled)

# Detailed Performance Report
print("Classification Report:")
print(classification_report(y_test, y_pred_best))

# Confusion Matrix Heatmap
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred_best)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Healthy (0)', 'Disease (1)'],
            yticklabels=['Healthy (0)', 'Disease (1)'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title(f'Confusion Matrix - {best_model_name}')
plt.tight_layout()
plt.show()

# 5.1 Interactive Function for Predicting New Patient Data
def test_new_patient(age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active):
    """
    Inputs:
      - age: in years (e.g., 55)
      - gender: 1 (female), 2 (male)
      - height: in cm (e.g., 168)
      - weight: in kg (e.g., 75.0)
      - ap_hi: Systolic BP (e.g., 130)
      - ap_lo: Diastolic BP (e.g., 85)
      - cholesterol: 1 (normal), 2 (above normal), 3 (well above normal)
      - gluc: 1 (normal), 2 (above normal), 3 (well above normal)
      - smoke: 0 (no), 1 (yes)
      - alco: 0 (no), 1 (yes)
      - active: 0 (no), 1 (yes)
    """
    bmi = weight / ((height / 100) ** 2)
    
    patient_dict = {
        'age': age, 'gender': gender, 'height': height, 'weight': weight,
        'ap_hi': ap_hi, 'ap_lo': ap_lo, 'cholesterol': cholesterol,
        'gluc': gluc, 'smoke': smoke, 'alco': alco, 'active': active, 'bmi': bmi
    }
    
    patient_df = pd.DataFrame([patient_dict])[X.columns]
    patient_scaled = scaler.transform(patient_df)
    
    prediction = final_model.predict(patient_scaled)[0]
    
    print("\n-------------------------------------------")
    print("      NEW PATIENT DIAGNOSIS PREDICTION     ")
    print("-------------------------------------------")
    if prediction == 1:
        print("⚠️ RESULT: POSITIVE - HIGH RISK OF CARDIOVASCULAR DISEASE!")
    else:
        print("✅ RESULT: NEGATIVE - LOW RISK (HEALTHY)")
    print("-------------------------------------------\n")

# Example Prediction Test
# Patient: Age 52, Male, 170cm, 85kg, BP 140/90, High Cholesterol
test_new_patient(age=52, gender=2, height=170, weight=85.0, ap_hi=140, ap_lo=90, 
                 cholesterol=2, gluc=1, smoke=0, alco=0, active=1)