from flask import Flask, render_template, request, redirect, url_for, flash
import mysql.connector
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

app = Flask(__name__)
app.secret_key = 'your_secret_key'  

# Load and prepare the dataset for the model 
housing_data = pd.read_csv(r'D:\Documents\FALL 2024 DA\House-price-prediction\Data Set.csv')
housing_data = housing_data.dropna(axis=0)
y = housing_data.Price
housing_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = housing_data[housing_features]

# Splitting dataset and training the Random Forest model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,random_state=1)

# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

print("Tuning Random Forest...")
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf_model = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_

y_pred_rf = best_rf.predict(X_test)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
print("\nRandom Forest Results:")
print(f"Mean Absolute Error: {mae_rf}")
print(f"Mean Squared Error: {mse_rf}")
print(f"R² Score: {r2_rf}")

# Connect to MySQL database
def connect_to_db():
    conn = mysql.connector.connect(host="localhost", user="root", password="", database="sarathydb")
    return conn

# Store prediction in the database
def store_prediction(conn, rooms, bathroom, landsize, latitude, longitude, predicted_price):
    cursor = conn.cursor()
    
    # Ensure all values are cast to Python float or int types compatible with MySQL
    rooms = float(rooms)
    bathroom = float(bathroom)
    landsize = float(landsize)
    latitude = float(latitude)
    longitude = float(longitude)
    predicted_price = float(predicted_price)
    
    insert_query = """INSERT INTO Predictions (rooms, bathroom, landsize, latitude, longitude, predicted_price)
                      VALUES (%s, %s, %s, %s, %s, %s)"""
    cursor.execute(insert_query, (rooms, bathroom, landsize, latitude, longitude, predicted_price))
    conn.commit()


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            rooms = float(request.form['rooms'])
            bathroom = float(request.form['bathroom'])
            landsize = float(request.form['landsize'])
            latitude = float(request.form['latitude'])
            longitude = float(request.form['longitude'])

            input_data = pd.DataFrame([[rooms, bathroom, landsize, latitude, longitude]], columns=housing_features)
            predicted_price = best_rf.predict(input_data)[0]

            # Store prediction in the database
            conn = connect_to_db()
            store_prediction(conn, rooms, bathroom, landsize, latitude, longitude, predicted_price)
            conn.close()

            flash(f"The estimated price of the house is: ${predicted_price:,.2f}", "success")
            return redirect(url_for('index'))
        except Exception as e:
            flash(f"Error: {str(e)}", "danger")

    return render_template('index.html')

@app.route('/predictions')
def predictions():
    conn = connect_to_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM Predictions")
    results = cursor.fetchall()
    conn.close()
    
    return render_template('predictions.html', predictions=results)


plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_rf, color='blue', alpha=0.6, label='Predicted vs Actual')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', linewidth=2, label='Perfect Prediction')

# Plot enhancements
plt.title('Actual vs. Predicted Prices (Random Forest)', fontsize=16)
plt.xlabel('Actual Prices', fontsize=14)
plt.ylabel('Predicted Prices', fontsize=14)
plt.legend(fontsize=12)
plt.grid(alpha=0.4)
plt.tight_layout()

# Display the plot
plt.show()

if __name__ == '__main__':
    app.run(debug=True)