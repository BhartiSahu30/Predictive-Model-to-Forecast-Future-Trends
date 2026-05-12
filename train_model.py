import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Load dataset
df = pd.read_csv(
    "data/indian_store.csv",
    encoding='latin1'
)

# Detect date column
possible_date_cols = [
    'Order Date',
    'Date',
    'Order_Date'
]

date_col = None

for col in possible_date_cols:
    if col in df.columns:
        date_col = col
        break

# Convert date
df[date_col] = pd.to_datetime(df[date_col])

# Feature Engineering
df['Year'] = df[date_col].dt.year
df['Month'] = df[date_col].dt.month

# Detect columns
sales_col = 'Sales'

quantity_col = 'Quantity' if 'Quantity' in df.columns else None

discount_col = 'Discount' if 'Discount' in df.columns else None

# Create missing columns
if quantity_col is None:
    df['Quantity'] = 1
    quantity_col = 'Quantity'

if discount_col is None:
    df['Discount'] = 0
    discount_col = 'Discount'

# Remove missing values
df.dropna(inplace=True)

# Features
X = df[['Year', 'Month', quantity_col, discount_col]]

# Rename features
X.columns = ['Year', 'Month', 'Quantity', 'Discount']

# Target
y = df[sales_col]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Model
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Predict
pred = model.predict(X_test)

# Accuracy
score = r2_score(y_test, pred)

print(f"Model Accuracy: {score}")

# Create models folder
os.makedirs("models", exist_ok=True)

# Save model
joblib.dump(model, "models/sales_model.pkl")

print("✅ Model Saved Successfully!")