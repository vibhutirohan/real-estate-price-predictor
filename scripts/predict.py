import sys
import joblib
import pandas as pd

# Fix encoding for emojis on Windows
sys.stdout.reconfigure(encoding='utf-8')

# Step 1: Load the trained model
model = joblib.load("../models/house_price_model.pkl")
print("✅ Model loaded successfully!")

# Step 2: Create input data (use the same features used during training)
new_data = pd.DataFrame([{
    "Area": 2200,
    "Bedrooms": 3,
    "Bathrooms": 2,
    "GrLivArea": 2100,
    "GarageCars": 2,
    "FullBath": 2,
    "TotalBsmtSF": 1050,
    "OverallQual": 6,
    "YearBuilt": 2003
}])

# Step 3: Predict the price
predicted_price = model.predict(new_data)[0]
print(f"🏡 Predicted House Price: ${predicted_price:,.2f}")
