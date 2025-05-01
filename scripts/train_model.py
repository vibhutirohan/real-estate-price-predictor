import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

# Create models directory if not exists
os.makedirs("models", exist_ok=True)

# Load preprocessed data
df = pd.read_csv("data/cleaned_house_prices.csv")

# Split into features and target
X = df.drop(columns=["Price"])
y = df["Price"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------------
# Train XGBoost
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
print("XGBoost MAE:", mean_absolute_error(y_test, xgb_pred))
joblib.dump(xgb_model, "models/xgboost_model.pkl")

# -------------------------------
# Train Random Forest
rf_model = RandomForestRegressor(n_estimators=100)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
print("Random Forest MAE:", mean_absolute_error(y_test, rf_pred))
joblib.dump(rf_model, "models/randomforest_model.pkl")

# -------------------------------
# Train Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
print("Linear Regression MAE:", mean_absolute_error(y_test, lr_pred))
joblib.dump(lr_model, "models/linear_model.pkl")

print("All models trained and saved to /models/")
