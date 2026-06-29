import os
import sqlite3
from flask import Flask, render_template, request, redirect, jsonify
import pandas as pd

# ML Models
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------- DATABASE ----------------
conn = sqlite3.connect("predictions.db", check_same_thread=False)
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

cursor.execute('''
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    area REAL,
    bedrooms INTEGER,
    bathrooms INTEGER,
    location TEXT,
    price REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
''')
conn.commit()

try:
    cursor.execute("""
    ALTER TABLE predictions
    ADD COLUMN location TEXT
    """)
    conn.commit()
except:
    pass

# ---------------- LOAD DATA ----------------
data = pd.read_csv("Housing.csv")

X = data[['area', 'bedrooms', 'bathrooms']]
y = data['price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- TRAIN MODELS ----------------
lr_model = LinearRegression()
dt_model = DecisionTreeRegressor()
rf_model = RandomForestRegressor()

lr_model.fit(X_train, y_train)
dt_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

# ---------------- MODEL ACCURACY ----------------
lr_acc = r2_score(y_test, lr_model.predict(X_test))
dt_acc = r2_score(y_test, dt_model.predict(X_test))
rf_acc = r2_score(y_test, rf_model.predict(X_test))

# Store models
models = {
    "Linear Regression": (lr_model, lr_acc),
    "Decision Tree": (dt_model, dt_acc),
    "Random Forest": (rf_model, rf_acc)
}

# Best model
best_model_name = max(models, key=lambda x: models[x][1])
best_model = models[best_model_name][0]

print("✅ Model Accuracies:")
for name, (_, acc) in models.items():
    print(f"{name}: {acc}")

print(f"🔥 Best Model: {best_model_name}")

# ---------------- HOME ----------------
@app.route('/')
def home():

    cursor.execute("SELECT * FROM predictions ORDER BY id DESC")
    rows = cursor.fetchall()

    cursor.execute("SELECT COUNT(*) FROM predictions")
    total_predictions = cursor.fetchone()[0]

    cursor.execute("SELECT AVG(price) FROM predictions")
    avg_price = cursor.fetchone()[0] or 0

    cursor.execute("SELECT MAX(price) FROM predictions")
    max_price = cursor.fetchone()[0] or 0

    areas = [row['area'] for row in rows]
    prices = [row['price'] for row in rows]

    model_scores = {
        "Linear Regression": round(lr_acc, 3),
        "Decision Tree": round(dt_acc, 3),
        "Random Forest": round(rf_acc, 3)
    }

    return render_template(
        "index.html",
        data=rows,
        areas=areas,
        prices=prices,
        scores=model_scores,
        best_model=best_model_name,
        total_predictions=total_predictions,
        avg_price=round(avg_price, 2),
        max_price=round(max_price, 2)
    )   
# ---------------- PREDICT (WEB) ----------------
@app.route('/predict', methods=['POST'])
def predict():
    try:

        area = float(request.form['area'])
        bedrooms = int(request.form['bedrooms'])
        bathrooms = int(request.form['bathrooms'])
        location = request.form['location']

        # Create input dataframe
        input_data = pd.DataFrame(
            [[area, bedrooms, bathrooms]],
            columns=['area', 'bedrooms', 'bathrooms']
        )

        # Prediction
        prediction = best_model.predict(input_data)[0]
        # Location multiplier
        if location == "Mumbai":
            prediction *= 1.50

        elif location == "Bangalore":
            prediction *= 1.30

        elif location == "Hyderabad":
            prediction *= 1.20

        elif location == "Delhi":
            prediction *= 1.40

        elif location == "Chennai":
            prediction *= 1.10

        # -----------------------------
        # Property Category
        # -----------------------------
        if prediction > 10000000:
            category = "🏰 Luxury Villa"

        elif prediction > 5000000:
            category = "🏡 Premium House"

        else:
            category = "🏠 Budget House"

        # -----------------------------
        # Investment Score
        # -----------------------------
        score = 5

        if location == "Bangalore":
            score += 3

        elif location == "Hyderabad":
            score += 2

        elif location == "Mumbai":
            score += 2

        if area > 2000:
            score += 2

        if score > 10:
            score = 10

        # -----------------------------
        # AI Recommendation
        # -----------------------------
        if location == "Bangalore":
            recommendation = "Excellent for Rental Income 💰"

        elif location == "Hyderabad":
            recommendation = "Fast Growing Investment Area 🚀"

        elif location == "Mumbai":
            recommendation = "Luxury Property Zone 🏙️"

        elif location == "Delhi":
            recommendation = "High Appreciation Potential 📈"

        else:
            recommendation = "Stable Long-Term Investment 🏠"

        # -----------------------------
        # EMI Calculation
        # -----------------------------
        principal = prediction

        annual_rate = 8.5
        monthly_rate = annual_rate / 12 / 100

        years = 20
        months = years * 12

        emi = (
            principal
            * monthly_rate
            * ((1 + monthly_rate) ** months)
        ) / (
            ((1 + monthly_rate) ** months) - 1
        )

        # Save prediction
        cursor.execute(
            """
            INSERT INTO predictions
            (area, bedrooms, bathrooms, location, price)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                area,
                bedrooms,
                bathrooms,
                location,
                float(prediction)
            )
        )

        conn.commit()

        output = f"₹ {round(prediction, 2)}"

    except Exception as e:

        output = f"Error: {str(e)}"

        category = ""
        score = 0
        recommendation = ""
        emi = 0

    # Prediction History
    cursor.execute(
        "SELECT * FROM predictions ORDER BY id DESC"
    )

    rows = cursor.fetchall()

    areas = [row['area'] for row in rows]
    prices = [row['price'] for row in rows]

    model_scores = {
        "Linear Regression": round(lr_acc, 3),
        "Decision Tree": round(dt_acc, 3),
        "Random Forest": round(rf_acc, 3)
    }

    return render_template(
        "index.html",
        prediction=output,
        category=category,
        score=score,
        recommendation=recommendation,
        emi=round(emi, 2),
        data=rows,
        areas=areas,
        prices=prices,
        scores=model_scores,
        best_model=best_model_name
    )




# ---------------- API ----------------
@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json()

        area = float(data['area'])
        bedrooms = int(data['bedrooms'])
        bathrooms = int(data['bathrooms'])

        input_data = pd.DataFrame(
            [[area, bedrooms, bathrooms]],
            columns=['area', 'bedrooms', 'bathrooms']
        )

        prediction = best_model.predict(input_data)[0]

        return jsonify({
            "status": "success",
            "model_used": best_model_name,
            "predicted_price": round(prediction, 2)
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        })

# ---------------- DELETE ----------------
@app.route('/delete/<int:id>')
def delete(id):
    cursor.execute("DELETE FROM predictions WHERE id = ?", (id,))
    conn.commit()
    return redirect('/')

@app.route('/clear')
def clear():

    cursor.execute("DELETE FROM predictions")

    conn.commit()

    return redirect('/')

# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True)