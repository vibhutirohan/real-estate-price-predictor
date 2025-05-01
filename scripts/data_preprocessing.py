import pandas as pd
import os

# Load original dataset (India or USA)
df = pd.read_csv("data/house_prices.csv")  # or your actual raw CSV file

# Select only the required columns
selected_columns = ["Area", "Bedrooms", "Bathrooms", "GrLivArea", "GarageCars", "FullBath", "TotalBsmtSF", "OverallQual", "YearBuilt", "Price"]
df = df[selected_columns]

# Fill missing values
df.fillna(df.mean(numeric_only=True), inplace=True)

# Save cleaned file
os.makedirs("data", exist_ok=True)
df.to_csv("data/cleaned_house_prices.csv", index=False)

print("Preprocessed dataset saved to data/cleaned_house_prices.csv")
