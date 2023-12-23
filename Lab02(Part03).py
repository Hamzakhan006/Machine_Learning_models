import pandas as pd
import numpy as np
from scipy.stats import kstest
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, recall_score, f1_score, roc_curve, confusion_matrix
import seaborn as sns

# Reading all data and printing all features
file_path = "C:/paython_learning/EML/data files/aa-delays-2023.csv"
flight_data = pd.read_csv(file_path, sep=',')
flight_data  # Printing all features

# Setting the target of a delay > 15 minutes to 1, otherwise to 0 / Assuming a new binary column 'DELAY_TARGET' to represent delays > 15 minutes
target_feature = 'DEP_DELAY'  # Replacing with the actual column name representing delay
threshold = 15  # Setting the threshold for delay in minutes

if target_feature in flight_data.columns:
    flight_data['DELAY_TARGET'] = (flight_data[target_feature] > threshold).astype(int)
    print(f"'DELAY_TARGET' column created.")
else:
    print(f"Column '{target_feature}' not found in the dataset. Please verify your data.")

flight_data

# Decision Tree Regression


# Selecting relevant features
selected_features = [
    'CRS_DEP_TIME', 'DEP_DELAY', 'TAXI_OUT', 'WHEELS_OFF',
    'WHEELS_ON', 'TAXI_IN', 'CRS_ARR_TIME', 'CRS_ELAPSED_TIME',
    'ACTUAL_ELAPSED_TIME', 'AIR_TIME', 'DISTANCE', 'CARRIER_DELAY',
    'WEATHER_DELAY', 'NAS_DELAY', 'SECURITY_DELAY', 'LATE_AIRCRAFT_DELAY'
]

# Creating a new DataFrame with only the selected features and target variable
selected_data = flight_data[selected_features + ['DELAY_TARGET']]

# Handling missing values
selected_data.fillna(0, inplace=True)

# Splitting the data into features (X) and target variable (y)
X = selected_data.drop('DELAY_TARGET', axis=1)
y = selected_data['DELAY_TARGET']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating a decision tree regressor
dt_regressor = DecisionTreeRegressor()

# Fitting the model
dt_regressor.fit(X_train, y_train)

# Making predictions
y_pred = dt_regressor.predict(X_test)

# Printing mean squared error and R2 score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R2 Score: {r2}")

# Creating a dictionary of classification models
classification_models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree Classifier': DecisionTreeClassifier(),
    'Gradient Boosting Classifier': GradientBoostingClassifier()
}

# Create a list to store classification results
classification_results = []

# Iterate through the classification models
for model_name, classification_model in classification_models.items():
    # Fit the classification model
    classification_model.fit(X_train, y_train)

    # Predictions on test set
    y_pred = classification_model.predict(X_test)

    # Calculate AUC, Recall, and F1 Score
    auc = roc_auc_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Append results to the list
    classification_results.append({
        'Model': model_name,
        'AUC': auc,
        'Recall': recall,
        'F1 Score': f1
    })

# Create a DataFrame from the list of classification results
classification_results_df = pd.DataFrame(classification_results)


# Plot ROC curves for classification models
plt.figure(figsize=(8, 6))

for model_name, classification_model in classification_models.items():
    # Fit the model
    classification_model.fit(X_train, y_train)

    # Predictions on test set
    y_pred_proba = classification_model.predict_proba(X_test)[:, 1]

    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)

    # Plot ROC curve
    plt.plot(fpr, tpr, label=f'{model_name}')

# Plot the random classifier
plt.plot([0, 1], [0, 1], linestyle='--', label='Random Classifier', color='black')

# Set plot labels and title
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Classification Models')
plt.legend()
plt.show()

# Plot confusion matrices for classification models
plt.figure(figsize=(15, 10))

for i, (model_name, classification_model) in enumerate(classification_models.items(), 1):
    plt.subplot(2, 2, i)

    # Fit the model
    classification_model.fit(X_train, y_train)

    # Predictions on test set
    y_pred = classification_model.predict(X_test)

    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Plot confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Not Delayed', 'Delayed'],
                yticklabels=['Not Delayed', 'Delayed'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')

plt.tight_layout()
plt.show()
