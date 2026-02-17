from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor


def tune_random_forest(X_train, y_train):

    param_grid = {
        "n_estimators": [50, 100],
        "max_depth": [None, 10, 20]
    }

    model = RandomForestRegressor(random_state=42)

    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=3,
        scoring="neg_mean_squared_error"
    )

    grid_search.fit(X_train, y_train)

    print("Best Parameters:", grid_search.best_params_)

    return grid_search.best_estimator_
