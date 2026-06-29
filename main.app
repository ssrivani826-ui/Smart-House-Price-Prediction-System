from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# ---------------- LOAD DATA ----------------
data = pd.read_csv("Housing.csv")

print("✅ Dataset Loaded Successfully")
print(data.head())

# ---------------- FEATURES ----------------
X = data[['area', 'bedrooms', 'bathrooms']]
y = data['price']

# ---------------- SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- MODEL ----------------
model = LinearRegression()
model.fit(X_train, y_train)

print("✅ Model Trained Successfully")

# ---------------- PREDICTION ----------------
predictions = model.predict(X_test)

print("Sample Predictions:", predictions[:5])

# ---------------- ACCURACY ----------------
accuracy = r2_score(y_test, predictions)
print("📊 Model Accuracy:", accuracy)

# ---------------- SAVE MODEL ----------------
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model saved as model.pkl")

# ---------------- OPTIONAL USER INPUT ----------------
def predict_price(area, bedrooms, bathrooms):
    input_data = pd.DataFrame(
        [[area, bedrooms, bathrooms]],
        columns=['area', 'bedrooms', 'bathrooms']
    )
    return model.predict(input_data)[0]


# Example usage
print("\n🔍 Example Prediction:")
print(predict_price(2000, 3, 2))


# ---------------- GRAPH ----------------
plt.figure(figsize=(6, 4))
plt.scatter(y_test, predictions)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.grid(True)
plt.show()