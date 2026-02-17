from sklearn.metrics import mean_squared_error, r2_score
import numpy as np


def evaluate(model, X_test, y_test):

    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    print("Model Performance")
    print("------------------")
    print("MSE :", mse)
    print("RMSE:", rmse)
    print("R2 Score:", r2)

    return mse, rmse, r2
