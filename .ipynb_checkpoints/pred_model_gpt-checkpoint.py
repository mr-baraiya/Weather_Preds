import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
df = pd.read_csv('weather_classification_data.csv')

# Drop rows with missing values (if any)
df = df.dropna()

# Split features and target
X = df.drop('Weather Type', axis=1)
y = df['Weather Type']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling continuous features
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train[['Temperature', 'Humidity', 'Wind Speed', 'Precipitation (%)', 'Atmospheric Pressure', 'UV Index', 'Visibility (km)']])
X_test_scaled = scaler.transform(X_test[['Temperature', 'Humidity', 'Wind Speed', 'Precipitation (%)', 'Atmospheric Pressure', 'UV Index', 'Visibility (km)']])

# Convert scaled data back to DataFrame
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=['Temperature', 'Humidity', 'Wind Speed', 'Precipitation (%)', 'Atmospheric Pressure', 'UV Index', 'Visibility (km)'])
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=['Temperature', 'Humidity', 'Wind Speed', 'Precipitation (%)', 'Atmospheric Pressure', 'UV Index', 'Visibility (km)'])

# One-hot encode categorical features
X_train_dummies = pd.get_dummies(X_train[['Cloud Cover', 'Season', 'Location']], drop_first=True, dtype=int)
X_test_dummies = pd.get_dummies(X_test[['Cloud Cover', 'Season', 'Location']], drop_first=True, dtype=int)

# Align the columns of one-hot encoded dataframes to ensure consistency
X_train_dummies, X_test_dummies = X_train_dummies.align(X_test_dummies, join='outer', axis=1, fill_value=0)

# Reset index of scaled data and dummy variables to ensure proper concatenation
X_train_scaled_df.reset_index(drop=True, inplace=True)
X_train_dummies.reset_index(drop=True, inplace=True)
X_test_scaled_df.reset_index(drop=True)
X_test_dummies.reset_index(drop=True)

# Concatenate scaled continuous features and one-hot encoded categorical features
new_X_train = pd.concat([X_train_scaled_df, X_train_dummies], axis=1)
new_X_test = pd.concat([X_test_scaled_df, X_test_dummies], axis=1)

# Ensure consistency in shapes
print("new_X_train shape:", new_X_train.shape)
print("new_X_test shape:", new_X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# Train the model
model = RandomForestClassifier()
model.fit(new_X_train, y_train)

# Make predictions
y_pred = model.predict(new_X_test)

# Ensure consistency in prediction length
print("y_pred shape:", y_pred.shape)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.2f}")

# Save the model and scaler
joblib.dump(model, 'weather_model.pkl')
joblib.dump(X.columns.tolist(), 'model_features.pkl')
joblib.dump(scaler, 'scaler.pkl')
