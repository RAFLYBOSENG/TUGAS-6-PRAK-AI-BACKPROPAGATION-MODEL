import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import os
import json

# 1. Load dataset
file_path = 'data/Housing.csv'
df = pd.read_csv(file_path)

# 2. Pilih fitur dan target
features = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']
target = ['price']
X = df[features].values
y = df[target].values

# 3. Normalisasi
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# 4. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=42
)

# 5. Bangun model JST
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='linear')
])

model.compile(optimizer='adam', loss='mean_squared_error')

# 6. Early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# 7. Latih model dan simpan history
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=8,
    callbacks=[early_stop]
).history

# 8. Simpan model dan scaler
if not os.path.exists('model'):
    os.makedirs('model')

model.save('model/house_price_model.h5')
joblib.dump(scaler_X, 'model/scaler_X.pkl')
joblib.dump(scaler_y, 'model/scaler_y.pkl')

# 9. Simpan history ke file JSON
with open('model/training_history.json', 'w') as f:
    json.dump({'loss': history['loss'], 'val_loss': history['val_loss']}, f)

print("Model, scaler, dan training history berhasil disimpan.")
