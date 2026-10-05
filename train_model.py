import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Sample house-price dataset
data = {
    "area": [1000, 1500, 1800, 2200, 2500, 3000, 1200, 1600, 2000, 2800],
    "bedrooms": [2, 3, 3, 4, 4, 5, 2, 3, 4, 5],
    "bathrooms": [1, 2, 2, 3, 3, 4, 1, 2, 2, 3],
    "age": [20, 15, 10, 5, 8, 2, 25, 12, 7, 3],
    "price": [
        200000,
        300000,
        360000,
        450000,
        500000,
        650000,
        230000,
        320000,
        410000,
        580000
    ]
}

df = pd.DataFrame(data)

# Input features
X = df[["area", "bedrooms", "bathrooms", "age"]]

# Target
y = df["price"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
with open("house_price_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model trained and saved successfully!")
print("R² score:", model.score(X_test, y_test))