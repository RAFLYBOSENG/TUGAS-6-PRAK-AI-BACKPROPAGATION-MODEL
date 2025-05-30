from flask import Flask, render_template, request
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import plotly.graph_objs as go
import plotly.io
import os
import json
import pandas as pd

app = Flask(__name__)

# Load model dan scaler
model = load_model('model/house_price_model.h5')
scaler_X = joblib.load('model/scaler_X.pkl')
scaler_y = joblib.load('model/scaler_y.pkl')

# Load data untuk evaluasi dan visualisasi
df = pd.read_csv('data/Housing.csv')
features = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']
target = ['price']
X = scaler_X.transform(df[features].values)
y_true_scaled = scaler_y.transform(df[target].values)
y_pred_scaled = model.predict(X)

# Invers transform untuk evaluasi aktual
y_true = scaler_y.inverse_transform(y_true_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        area = float(request.form['area'])
        bedrooms = int(request.form['bedrooms'])
        bathrooms = int(request.form['bathrooms'])
        stories = int(request.form['stories'])
        parking = int(request.form['parking'])

        features_input = np.array([[area, bedrooms, bathrooms, stories, parking]])
        features_scaled = scaler_X.transform(features_input)

        prediction_scaled = model.predict(features_scaled)
        prediction = scaler_y.inverse_transform(prediction_scaled)

        return render_template('index.html', prediction_text=f"Harga Rumah Diprediksi: $ {int(prediction[0][0]):,}")

    except Exception as e:
        return render_template('index.html', prediction_text=f"Terjadi kesalahan: {str(e)}")

@app.route('/evaluasi')
def evaluasi():
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return render_template('evaluasi.html', mse=f"{mse:,.2f}", mae=f"{mae:,.2f}", r2=f"{r2:.4f}")

@app.route('/visualisasi')
def visualisasi():
    history_path = 'model/training_history.json'
    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            history = json.load(f)
        trace1 = go.Scatter(y=history.get('loss', []), mode='lines+markers', name='Training Loss')
        trace2 = go.Scatter(y=history.get('val_loss', []), mode='lines+markers', name='Validation Loss')
        layout_loss = go.Layout(title='Grafik Loss Model', xaxis=dict(title='Epoch'), yaxis=dict(title='Loss'))
        loss_data = {'data': [trace1, trace2], 'layout': layout_loss}
    else:
        loss_data = {'data': [], 'layout': {'title': 'Tidak Ada Data'}}

    plot_data = {
        'loss_plot': plotly.io.to_json(loss_data)
    }

    return render_template('visualisasi.html', loss_plot_data=plot_data['loss_plot'])

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
