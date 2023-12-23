print("hello")
import pandas as pd
import numpy as np
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
# Reading CSV file
file_path="C:\paython_learning\EML\data files\cardio_train.csv"
df = pd.read_csv(file_path, sep=';')
print(df.columns)
# Assuming 'df' is your dataframe
age_values = df['age'].values
# Computing the year of the age values
year_of_birth = age_values/365
print (year_of_birth)
mean_age = year_of_birth.mean()
print (mean_age)
# Computing maximum and minimum of the age values
max_age = year_of_birth.max()
min_age = year_of_birth.min()
print(f"The maximum age is: {max_age} years")
print(f"The minimum age is: {min_age} years")
# Defining features (X) and target (y)
X = df.drop(['id', 'cardio'], axis=1)  # Assuming 'id' is not a feature, and 'cardio' is the target
y = df['cardio']
# Splitting data into training and test sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Scaling data using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Creating a Logistic Regression model
logistic_regression_model = LogisticRegression(max_iter=1000, random_state=42)
# Training the model
logistic_regression_model.fit(X_train_scaled, y_train)
# Predicting the output for the test set
y_pred = logistic_regression_model.predict(X_test_scaled)
# Optionally, calculating the predicted probabilities
y_pred_proba = logistic_regression_model.predict_proba(X_test_scaled)
# Comparing predicted output with real values
comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
# Printing the comparison
print(comparison)
# Checking if the model is Logistic Regression
if hasattr(logistic_regression_model, 'coef_'):
    # Getting feature names
    feature_names = X.columns
    # Getting coefficients
    coefficients = logistic_regression_model.coef_[0]
    # Combining feature names and their coefficients
    feature_importance = pd.DataFrame({'Feature': feature_names, 'Importance': coefficients})
    # Sorting by importance
    feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
    print("Feature Importance:")
    print(feature_importance)
else:
    print("This model does not have feature importances.")
# Assuming 'X' is your feature DataFrame
# Selecting one example (proband)
proband_example = X.iloc[0]  # Assuming you want the first example
# Printing features and their values for the proband
for feature, value in proband_example.items():
    print(f"{feature}: {value}")
# Setting up the figure and axis
plt.figure(figsize=(10, 6))
# Creating a bar chart
plt.bar(feature_importance['Feature'], feature_importance['Importance'], color='skyblue')
# Adding labels and title
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Feature Importance')
# Rotating x-axis labels for better visibility
plt.xticks(rotation=45)
# Showing the plot
plt.tight_layout()
plt.show()
