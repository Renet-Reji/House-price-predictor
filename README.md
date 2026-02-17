# 🏠 House Price Predictor (Machine Learning Project)

A Machine Learning project that predicts house prices using regression models trained on housing data. This project demonstrates a complete ML pipeline including preprocessing, feature engineering, training, evaluation, and prediction using saved model and scaler.

---

# 📌 Features

* Data preprocessing and scaling
* Feature engineering
* Model training (Linear Regression, Random Forest)
* Hyperparameter tuning
* Model evaluation (RMSE, R² score)
* Save and load trained model
* Predict price for new custom input
* Clean project structure

---

# 📂 Project Structure

```
House-price-predictor/
│
├── main.py                  # Train and evaluate model
├── data_loader.py          # Load dataset
├── preprocess.py           # Preprocessing and scaling
├── feature_engineering.py  # Feature engineering
├── model.py                # Model training
├── evaluate.py             # Model evaluation
├── tune.py                 # Hyperparameter tuning
│
├── house_price_model.pkl   # Saved trained model
├── scaler.pkl              # Saved scaler
│
├── requirements.txt
└── README.md
```

---

# ⚙️ Installation Guide

## Step 1: Clone repository

```
git clone https://github.com/Renet-reji/House-price-predictor.git
cd House-price-predictor
```

---

## Step 2: Create virtual environment

### Windows

```
python -m venv venv
venv\Scripts\activate
```

### Linux / Mac

```
python3 -m venv venv
source venv/bin/activate
```

---

## Step 3: Install requirements

```
pip install -r requirements.txt
```

---

# ▶️ Train the Model

Run:

```
python main.py
```

This will:

* Load dataset
* Preprocess data
* Train model
* Evaluate performance
* Save model as:

```
house_price_model.pkl
scaler.pkl
```

---

# 📊 Model Accuracy

Typical performance:

| Metric   | Value       |
| -------- | ----------- |
| RMSE     | 0.45 – 0.60 |
| R² Score | 0.80 – 0.90 |
| Accuracy | ~85%        |

(Random Forest gives best performance)

---

# 🔮 Make Your Own Prediction

Create new file:

```
predict.py
```

Example code:

```python
import joblib
import numpy as np

# Load saved model and scaler
model = joblib.load("house_price_model.pkl")
scaler = joblib.load("scaler.pkl")

# Example input
# Format:
# [MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]

sample = np.array([[8.3252, 41, 6.984, 1.023, 322, 2.555, 37.88, -122.23]])

# Scale input
sample_scaled = scaler.transform(sample)

# Predict
prediction = model.predict(sample_scaled)

print("Predicted House Price:", prediction[0])
```

Run:

```
python predict.py
```

---

# 🧠 How Model Works

Pipeline:

1. Load dataset
2. Split into train and test
3. Scale features using StandardScaler
4. Train Random Forest model
5. Evaluate accuracy
6. Save model and scaler
7. Load model for prediction

---

# 📦 Requirements

Main libraries used:

```
numpy
pandas
scikit-learn
joblib
matplotlib
seaborn
```

Install using:

```
pip install -r requirements.txt
```

---

# 💾 Saved Files

| File                  | Purpose          |
| --------------------- | ---------------- |
| house_price_model.pkl | trained ML model |
| scaler.pkl            | feature scaler   |

Both required for prediction.

---

# 📈 Example Prediction

Input:

```
Median Income: 8.3
House Age: 41
Rooms: 6.9
Population: 322
```

Output:

```
Predicted Price: $412000
```

---

# 🚀 Future Improvements

* Add web interface (Flask / Streamlit)
* Use real dataset (Kaggle)
* Deploy model online
* Add visualization dashboard

---

# 👨‍💻 Author

Renet Reji

---

# 📜 License

Educational use only.