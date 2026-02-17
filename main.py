from data_loader import load_data
from preprocess import preprocess_data
from feature_engineering import add_features
from model import train_linear_regression, train_random_forest
from evaluate import evaluate
from tune import tune_random_forest

import joblib


def main():

    print("Loading data...")
    df = load_data()

    print("Preprocessing data...")
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)

    # SAVE SCALER
    joblib.dump(scaler, "scaler.pkl")

    print("Feature engineering...")
    X_train = add_features(X_train)
    X_test = add_features(X_test)

    print("Training Linear Regression...")
    lr_model = train_linear_regression(X_train, y_train)

    print("\nLinear Regression Results:")
    evaluate(lr_model, X_test, y_test)

    print("\nTraining Random Forest...")
    rf_model = train_random_forest(X_train, y_train)

    print("\nRandom Forest Results:")
    evaluate(rf_model, X_test, y_test)

    print("\nHyperparameter tuning...")
    best_model = tune_random_forest(X_train, y_train)

    print("\nBest Model Results:")
    evaluate(best_model, X_test, y_test)

    print("\nSaving model...")
    joblib.dump(best_model, "house_price_model.pkl")

    print("Done.")


if __name__ == "__main__":
    main()
