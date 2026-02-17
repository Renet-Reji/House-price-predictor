import joblib
import numpy as np
from feature_engineering import add_features

# load model and scaler
model = joblib.load("house_price_model.pkl")
scaler = joblib.load("scaler.pkl")

# new sample
sample = np.array([[3.2, 40, 4.5, 1.0, 1500, 4.2, 34.05, -118.25]])
                   #MedIncome, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude

# STEP 1: scale sample
sample_scaled = scaler.transform(sample)

# STEP 2: add engineered features
sample_final = add_features(sample_scaled)

# STEP 3: predict
prediction = model.predict(sample_final)

# convert to dollars
price = prediction[0] * 100000

print(f"Predicted price: ${price:,.2f}")
