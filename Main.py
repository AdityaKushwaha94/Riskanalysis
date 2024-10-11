# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

# Step 1: Load the dataset
data = pd.read_csv('creditcard.csv')

# Step 2: Data exploration (optional for understanding the data)
print(data.head())
print(data.info())
print(data['Class'].value_counts())  # Check for imbalance

# Step 3: Separate features (X) and target (y)
X = data.drop(columns=['Class'])
y = data['Class']

# Step 4: Normalize the features using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 5: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Step 6: Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Step 7: Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_res, y_train_res)

# Step 8: Make predictions on the test data
y_pred = model.predict(X_test)

# Step 9: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Detailed classification report
print(classification_report(y_test, y_pred))

# Optional: Grid search for hyperparameter tuning
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
grid_search.fit(X_train_res, y_train_res)

# Best parameters after grid search
print("Best parameters:", grid_search.best_params_)

# Final evaluation with best parameters
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)
print(f"Final accuracy after tuning: {accuracy_score(y_test, y_pred_best) * 100:.2f}%")
print(classification_report(y_test, y_pred_best))
