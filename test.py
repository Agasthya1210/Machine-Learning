import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression  
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load the dataset from a CSV file
data = pd.read_csv('rainfalldata.csv')

# Extract 'Month' and 'Year' from the 'Time' column
month_map = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6, 
             'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}

# Convert month names to numerical values and extract the year
data["Month"] = data["Time"].str.split("-").apply(lambda x: month_map[x[0]])  # Map month names to integers
data["Year"] = data["Time"].str.split("-").apply(lambda x: int(x[1]))  # Convert year part to integer

# Filter out rows where 'Total_rainfall' is zero or less
data = data[data['Total_rainfall'] > 0]

# Drop irrelevant columns from the dataset
data.drop("Time", axis=1, inplace=True)  # Drop 'Time' column
data.drop('cityName', axis=1, inplace=True)  # Drop 'cityName' column

# Define feature matrix X and target vector y
X = data[["Month", "Year"]]
y = data["Total_rainfall"]

# Preprocessing: One-hot encode the 'Month' column
preprocessor = ColumnTransformer(
    transformers=[
        ('month', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['Month'])  # One-hot encode months
    ],
    remainder='passthrough'  # Pass 'Year' through without transformation
)

# Create a pipeline that combines preprocessing and regression
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),  # Apply preprocessing
    ('regressor', LinearRegression())  # Fit linear regression model
])

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

# Train the model on the training data
pipeline.fit(X_train, y_train)

# Predict on the test data
y_pred = pipeline.predict(X_test)

# Evaluate the model performance using Root Mean Squared Error (RMSE)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'Root Mean Squared Error: {rmse}')

# Transform the test features using the fitted preprocessor
X_test_transformed = pipeline.named_steps['preprocessor'].transform(X_test)

# Get the feature names for the one-hot encoded 'Month' column
month_encoded_columns = pipeline.named_steps['preprocessor'].transformers_[0][1].get_feature_names_out(['Month'])

# Combine month and year columns
columns = list(month_encoded_columns) + ['Year']

# Create a DataFrame for the test set with transformed features and predictions
X_test_df = pd.DataFrame(X_test_transformed, columns=columns)
X_test_df['Actual'] = y_test.values  # Add actual rainfall values
X_test_df['Predicted'] = y_pred  # Add predicted rainfall values

# Plot actual vs predicted rainfall for test data
plt.figure(figsize=(10, 6))
plt.scatter(X_test_df['Year'], X_test_df['Actual'], color='black', label='Actual Rainfall')
plt.scatter(X_test_df['Year'], X_test_df['Predicted'], color='blue', label='Predicted Rainfall')
plt.xlabel('Year')
plt.ylabel('Rainfall')
plt.title('Rainfall Prediction')
plt.legend()
plt.show()

# Plot actual vs predicted rainfall for test data without transformed columns
plt.figure(figsize=(10, 6))
plt.scatter(X_test['Year'], y_test, color='black', label='Actual Rainfall')
plt.scatter(X_test['Year'], y_pred, color='blue', label='Predicted Rainfall')
plt.xlabel('Year')
plt.ylabel('Rainfall')
plt.title('Rainfall Prediction')
plt.legend()
plt.show()

# Predict rainfall for a specific month and year
year_to_predict = 2022
X_predict = pd.DataFrame({
    'Month': [8],  # July
    'Year': [year_to_predict]
})

# Get prediction for the given month and year
predicted_rainfall = pipeline.predict(X_predict)
print(f"Predicted rainfall for September {year_to_predict}: {predicted_rainfall[0]/1000} mm")

