import sys
import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

# Force UTF-8 encoding
sys.stdout.reconfigure(encoding='utf-8')

# Step 1: Load cleaned data
df = pd.read_csv("../data/cleaned_house_prices.csv")
print("✅ Dataset loaded!")

# Step 2: Split features and target
X = df.drop(columns=["Price"])
y = df["Price"]

# Step 3: Train-Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train XGBoost Regressor
model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Step 5: Evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"🚀 XGBoost MAE: {mae:.2f}")

# Step 6: Save the model
os.makedirs("../models", exist_ok=True)
joblib.dump(model, r"C:\Users\rohan\Documents\House Prediction Analysis\models\house_price_model.pkl")
print("✅ XGBoost model saved at /models/house_price_model.pkl")
