import numpy as np


def add_features(X):

    # Example feature engineering
    # ratio features improve performance

    rooms_per_household = X[:, 3] / X[:, 6]
    population_per_household = X[:, 5] / X[:, 6]

    new_features = np.c_[
        X,
        rooms_per_household,
        population_per_household
    ]

    return new_features
