import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.metrics import accuracy_score
import joblib

df = pd.read_csv('weather_classification_data.csv')


X= df.drop(['Weather Type','Cloud Cover','Season','Location'], axis=1)

# X = pd.get_dummies(X, drop_first=True, dtype=int)

y = df['Weather Type']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier()

# model.fit(X_train_scaled, y_train)
# # param_grid = {
# #     'n_estimators': [100, 200, 300],
# #     'max_depth': [10, 20, 30],
# #     'min_samples_split': [2, 5, 10],
# #     'min_samples_leaf': [1, 2, 4]
# # }

# # grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')

# # grid_search.fit(X_train_scaled, y_train)

# # print("Best parameters found: ", grid_search.best_params_)
# # print("Best accuracy score: ", grid_search.best_score_)

# # best_model = grid_search.best_estimator_


model.fit(X_train_scaled, y_train)


y_pred = model.predict(X_test_scaled)


accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.2f}")

joblib.dump(model, 'weather_model.pkl')
joblib.dump(X.columns.tolist(), 'model_features.pkl')
joblib.dump(scaler, 'scaler.pkl')
