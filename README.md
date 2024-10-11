# # Credit Card Fraud Detection with Random Forest and SMOTE

This project is a demonstration of how to detect fraudulent credit card transactions using a Random Forest Classifier. It addresses class imbalance by applying SMOTE (Synthetic Minority Over-sampling Technique) and optimizes model performance using Grid Search for hyperparameter tuning.

## Dataset

The dataset used in this project is `creditcard.csv`. It contains transactions made by credit cards in September 2013 by European cardholders, where the goal is to detect fraudulent transactions.

### Dataset Columns
- **V1 to V28**: Result of PCA transformation applied to the original features
- **Time**: Number of seconds elapsed between this transaction and the first transaction in the dataset
- **Amount**: Transaction amount
- **Class**: Fraud indicator (0 = not fraud, 1 = fraud)

## Project Steps

1. **Data Exploration**: Load and explore the dataset to understand its structure and identify class imbalance.
2. **Data Preprocessing**: Standardize the features using `StandardScaler` to improve model performance.
3. **SMOTE**: Apply SMOTE to handle the class imbalance by generating synthetic samples for the minority class.
4. **Model Training**: Train a Random Forest Classifier on the resampled data.
5. **Model Evaluation**: Evaluate the model's performance on the test set using accuracy score and classification report.
6. **Hyperparameter Tuning**: Perform Grid Search to find the best hyperparameters for the Random Forest model.

## Requirements

To run this project, you'll need the following Python libraries:

- pandas
- numpy
- scikit-learn
- imbalanced-learn (for SMOTE)

You can install these packages using `pip`:

```bash
pip install pandas numpy scikit-learn imbalanced-learn
```

## Running the Project

1. Clone this repository:

```bash
(https://github.com/AdityaKushwaha94/Riskanalysis/)
```

2. Navigate to the project directory:

```bash
cd credit-card-fraud-detection
```

3. Ensure you have the `creditcard.csv` dataset in the project directory.

4. Run the Python script:

```bash
python random_forest_creditcard.py
```

## Results

- **Initial Model Accuracy**: Achieved a decent accuracy of ~90% on the test set.
- **After Tuning**: With grid search hyperparameter tuning, accuracy improved further.
- **Classification Report**: Shows precision, recall, and F1-scores for both classes (fraud and non-fraud).

#
