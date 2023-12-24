import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Reading all data and printing all features
file_path = "C:/paython_learning/EML/data files/aa-delays-2023.csv"
flight_data = pd.read_csv(file_path, sep=',')
flight_data  # Printing all features

# Printing all features (column names)
print("All Features (Column Names):")
print(flight_data.columns)

# Determining how strong the influence of WEATHER_DELAY on ARR_DELAY is
correlation = flight_data['WEATHER_DELAY'].corr(flight_data['ARR_DELAY'])
print(f"Correlation between WEATHER_DELAY and ARR_DELAY: {correlation}")

# Creating a scatter plot with a regression line for WEATHER_DELAY vs ARR_DELAY
if 'WEATHER_DELAY' in flight_data.columns and 'ARR_DELAY' in flight_data.columns:
    sns.regplot(x='WEATHER_DELAY', y='ARR_DELAY', data=flight_data)
    plt.show()
else:
    print("Columns 'WEATHER_DELAY' or 'ARR_DELAY' not found in the dataset. Please verify your data.")

# If the scatter plot with the regression line appears as a straight line, it suggests a linear relationship between the two variables. A linear relationship means that as one variable (e.g., WEATHER_DELAY) increases or decreases, the other variable (ARR_DELAY) changes proportionally.

# Deleting ARR_DELAY and String data / Assuming 'ARR_DELAY' represents the total delay duration
if 'ARR_DELAY' in flight_data.columns:
    flight_data = flight_data.drop(['ARR_DELAY'], axis=1)
    print("'ARR_DELAY' column dropped.")
else:
    print("Column 'ARR_DELAY' not found in the dataset. No action taken.")

# Dropping other unwanted string columns
columns_to_drop = ['ORIGIN', 'DEST', 'Unnamed: 27', 'OP_CARRIER']
flight_data = flight_data.drop(columns_to_drop, axis=1)

# Setting the target of a delay > 15 minutes to 1, otherwise to 0 / Assuming a new binary column 'DELAY_TARGET' to represent delays > 15 minutes
target_feature = 'DEP_DELAY'  # Replacing with the actual column name representing delay
threshold = 15  # Setting the threshold for delay in minutes

if target_feature in flight_data.columns:
    flight_data['DELAY_TARGET'] = (flight_data[target_feature] > threshold).astype(int)
    print(f"'DELAY_TARGET' column created.")
else:
    print(f"Column '{target_feature}' not found in the dataset. Please verify your data.")

# The goal is to reduce the cost of flight delay. Which target feature do we choose and why? / Assuming 'ARR_DELAY' represents the total delay duration
target_feature = 'ARR_DELAY'
print(f"Chosen Target Feature: {target_feature}")

if 'FL_DATE' in flight_data.columns:
    flight_data = flight_data.drop(['FL_DATE'], axis=1)
    print("'FL_DATE' column dropped.")
flight_data

# Assuming 'DELAY_TARGET' is the target feature
target_feature = 'DELAY_TARGET'
if target_feature not in flight_data.columns:
    print(f"Error: Target feature '{target_feature}' not found in the DataFrame.")
else:
    # Choosing two specific columns for correlation
    columns_of_interest = ['WEATHER_DELAY', 'DEP_DELAY']  # Replacing with your actual column names

    # Calculating Pearson correlation coefficients with 'DELAY_TARGET' for the chosen columns
    correlations_with_target = flight_data[columns_of_interest + [target_feature]].corr()[target_feature]

    # Printing correlations
    print(f"Linear Correlations of '{target_feature}' with {columns_of_interest}:")
    print(correlations_with_target)

flight_data

