import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression  
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('rainfalldata.csv')

# Extract month and year from 'Time' column
month_map = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6, 
             'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}

data["Month"] = data["Time"].str.split("-").apply(lambda x: month_map[x[0]])  # Map months
data["Year"] = data["Time"].str.split("-").apply(lambda x: int(x[1]))  # Convert year to integer

data = data[data['Total_rainfall'] > 0]

data.drop("Time", axis=1, inplace=True)
data.drop('cityName', axis=1, inplace=True)

X = data[["Month", "Year"]]
y = data["Total_rainfall"]

preprocessor = ColumnTransformer(
    transformers=[
        ('month', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['Month'])
    ],
    remainder='passthrough'  # Passes 'Year' through without transformation
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'Root Mean Squared Error: {rmse}')

X_test_transformed = pipeline.named_steps['preprocessor'].transform(X_test)

month_encoded_columns = pipeline.named_steps['preprocessor'].transformers_[0][1].get_feature_names_out(['Month'])

columns = list(month_encoded_columns) + ['Year']

X_test_df = pd.DataFrame(X_test_transformed, columns=columns)

X_test_df['Actual'] = y_test.values
X_test_df['Predicted'] = y_pred

plt.figure(figsize=(10, 6))
plt.scatter(X_test_df['Year'], X_test_df['Actual'], color='black', label='Actual Rainfall')
plt.scatter(X_test_df['Year'], X_test_df['Predicted'], color='blue', label='Predicted Rainfall')
plt.xlabel('Year')
plt.ylabel('Rainfall')
plt.title('Rainfall Prediction')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(X_test['Year'], y_test, color='black', label='Actual Rainfall')
plt.scatter(X_test['Year'], y_pred, color='blue', label='Predicted Rainfall')
plt.xlabel('Year')
plt.ylabel('Rainfall')
plt.title('Rainfall Prediction')
plt.legend()
plt.show()

year_to_predict = 2022
X_predict = pd.DataFrame({
    'Month': [7], 
    'Year': [year_to_predict]
})
predicted_rainfall = pipeline.predict(X_predict)
print(f"Predicted rainfall for September {year_to_predict}: {predicted_rainfall[0]/1000}")
